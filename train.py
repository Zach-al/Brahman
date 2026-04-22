import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast, RobertaModel, get_cosine_schedule_with_warmup
from pathlib import Path
import json
import random
import argparse
from tqdm import tqdm
import collections

class Config:
    data_path = "data/processed/training_data.jsonl"
    srl_checkpoint = "models/pretrained_vibhakti/best_srl.pt"
    srl_tokenizer_path = "models/pretrained_vibhakti"
    save_path = "brahman_best.pth"
    ablation_save_path = "brahman_ablation.pth"
    phase1_epochs = 5
    phase2_epochs = 8
    phase3_epochs = 12
    early_stopping_patience = 8
    lr_encoder = 1e-5
    lr_task_head = 5e-5
    batch_size = 32
    grad_accumulation = 4
    max_length = 256
    num_labels = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2IDX = {"VALID": 0, "INVALID": 1, "AMBIGUOUS": 2, "UNKNOWN": 3}

class LogicDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256, phase=3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        valid_types = set()
        if phase >= 1:
            valid_types.update(["modus_ponens", "modus_tollens", "hypothetical_syllogism", "simple_chain"])
        if phase >= 2:
            valid_types.update(["fallacy_affirming_consequent", "fallacy_denying_antecedent", "causal_prevention"])
        if phase >= 3:
            # allow all examples
            self.examples = examples
        else:
            self.examples = [ex for ex in examples if ex["form_type"] in valid_types]
            
        random.shuffle(self.examples)
        print(f"Phase {phase} dataset size: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = " [SEP] ".join(ex["premises"]) + " [SEP] " + ex.get("question", "What follows?")
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        label = torch.tensor(0 if ex["is_valid"] else 1, dtype=torch.long)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
            "form_type": ex.get("form_type", "unknown"),
        }

def load_examples(path, max_n=None):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            
    if max_n is not None:
        random.shuffle(examples)
        examples = examples[:max_n]
        
    return examples

class BrahmanModel(nn.Module):
    def __init__(self, use_panini=True, srl_checkpoint=None):
        super().__init__()
        self.use_panini = use_panini
        self.encoder = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
        
        if srl_checkpoint and Path(srl_checkpoint).exists():
            state_dict = torch.load(srl_checkpoint, map_location="cpu")
            encoder_dict = {k.replace("roberta.", ""): v for k, v in state_dict.items() if k.startswith("roberta.")}
            self.encoder.load_state_dict(encoder_dict, strict=False)
            print("Loaded pre-trained SRL weights")
        else:
            print("Warning: no SRL checkpoint found, using base weights")

        hidden = 768

        if use_panini:
            self.panini_gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            )
            self.panini_lambda = nn.Parameter(torch.tensor(0.3))

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Softmax(dim=1)
        )

        self.task_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.15),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, Config.num_labels)
        )

        self.confirmation_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, label=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state  # [B, L, 768]

        if self.use_panini:
            # pairwise_features
            mean_seq = seq.mean(dim=1, keepdim=True).expand(-1, seq.size(1), -1)
            pairwise_features = torch.cat([seq, mean_seq], dim=-1) # [B, L, 1536]
            gate = self.panini_gate(pairwise_features) # [B, L, 1]
            lambda_clamped = self.panini_lambda.clamp(0.0, 1.0)
            seq = seq + lambda_clamped * gate * seq

        pool_w = self.attention_pool(seq) # [B, L, 1]
        pooled = (seq * pool_w).sum(dim=1) # [B, 768]

        task_logits = self.task_head(pooled)
        conf_logits = self.confirmation_head(pooled)

        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(task_logits, label)

        return {"loss": loss, "task_logits": task_logits, "conf_logits": conf_logits, "pooled": pooled}

def train_phase(model, train_dl, val_dl, phase, epochs, save_path):
    encoder_params = [p for n, p in model.named_parameters() if "encoder" in n]
    other_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": Config.lr_encoder},
        {"params": other_params, "lr": Config.lr_task_head}
    ])
    
    total_steps = (len(train_dl) // Config.grad_accumulation) * epochs
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dl, desc=f"Phase {phase} Epoch {epoch+1} Train")):
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            label = batch["label"].to(Config.device)
            
            outputs = model(input_ids, attention_mask, label=label)
            loss = outputs["loss"] / Config.grad_accumulation
            
            loss.backward()
            
            if (step + 1) % Config.grad_accumulation == 0 or (step + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item() * Config.grad_accumulation
            
        avg_train_loss = train_loss / len(train_dl)
        
        model.eval()
        correct = 0
        total = 0
        per_type = collections.defaultdict(lambda: {"correct": 0, "total": 0})

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Phase {phase} Epoch {epoch+1} Val"):
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)
                labels = batch["label"].to(Config.device)
                form_types = batch["form_type"]

                outputs = model(input_ids, attention_mask)
                preds = outputs["task_logits"].argmax(dim=-1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for i, ft in enumerate(form_types):
                    per_type[ft]["total"] += 1
                    if preds[i].item() == labels[i].item():
                        per_type[ft]["correct"] += 1

        val_acc = correct / total if total > 0 else 0.0
        
        print(f"Phase {phase} Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_acc={val_acc:.4f}")
        for ft, stats in per_type.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {ft}: {acc:.4f}")
            
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), save_path)
            print(f"New best: val_acc={val_acc:.4f} saved")
            best_val_acc = val_acc
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= Config.early_stopping_patience:
                print("Early stopping triggered")
                break
                
    return best_val_acc

def train(ablation_mode=False):
    print(f"=== {'Ablated' if ablation_mode else 'Full'} Model Training ===")

    print("1. Loading data...")
    examples = load_examples(Config.data_path)
    train_examples = examples[:45000]
    val_examples = examples[-5000:]

    print("2. Loading tokenizer...")
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(Config.srl_tokenizer_path)
    except:
        print("Falling back to roberta-base tokenizer")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    print("3. Building model...")
    use_panini = not ablation_mode
    srl_ckpt = None if ablation_mode else Config.srl_checkpoint
    model = BrahmanModel(use_panini=use_panini, srl_checkpoint=srl_ckpt).to(Config.device)
    
    save_path = Config.ablation_save_path if ablation_mode else Config.save_path

    print("4. Phase 1 (valid inference only)...")
    p1_train_ds = LogicDataset(train_examples, tokenizer, Config.max_length, phase=1)
    p1_val_ds = LogicDataset(val_examples, tokenizer, Config.max_length, phase=1)
    p1_train_dl = DataLoader(p1_train_ds, batch_size=Config.batch_size, shuffle=True)
    p1_val_dl = DataLoader(p1_val_ds, batch_size=Config.batch_size, shuffle=False)
    train_phase(model, p1_train_dl, p1_val_dl, 1, Config.phase1_epochs, save_path)

    print("5. Phase 2 (add fallacies)...")
    if Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=Config.device))
    p2_train_ds = LogicDataset(train_examples, tokenizer, Config.max_length, phase=2)
    p2_val_ds = LogicDataset(val_examples, tokenizer, Config.max_length, phase=2)
    p2_train_dl = DataLoader(p2_train_ds, batch_size=Config.batch_size, shuffle=True)
    p2_val_dl = DataLoader(p2_val_ds, batch_size=Config.batch_size, shuffle=False)
    train_phase(model, p2_train_dl, p2_val_dl, 2, Config.phase2_epochs, save_path)

    print("6. Phase 3 (full curriculum)...")
    if Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=Config.device))
    p3_train_ds = LogicDataset(train_examples, tokenizer, Config.max_length, phase=3)
    p3_val_ds = LogicDataset(val_examples, tokenizer, Config.max_length, phase=3)
    p3_train_dl = DataLoader(p3_train_ds, batch_size=Config.batch_size, shuffle=True)
    p3_val_dl = DataLoader(p3_val_ds, batch_size=Config.batch_size, shuffle=False)
    train_phase(model, p3_train_dl, p3_val_dl, 3, Config.phase3_epochs, save_path)

    print("7. Final Summary: Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    train(ablation_mode=args.ablation)
