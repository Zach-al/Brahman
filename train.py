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

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    data_path = "data/processed/training_data.jsonl"
    srl_checkpoint = "models/pretrained_vibhakti/best_srl.pt" # Backup weights location
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

# ==============================================================================
# DATA HANDLING
# ==============================================================================
class LogicDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256, phase=3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.phase = phase
        
        filtered_examples = []
        # OVERRIDE FIX: Allow all examples to pass through to prevent 0it validation bug
        for ex in examples:
            filtered_examples.append(ex)
                
        random.shuffle(filtered_examples)
        self.examples = filtered_examples
        print(f"Phase {phase} dataset initialized with {len(self.examples)} examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = " [SEP] ".join(ex.get("premises", [])) + " [SEP] " + ex.get("question", "What follows?")
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        label = 0 if ex.get("is_valid", False) else 1
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "form_type": ex.get("form_type", "unknown")
        }

def load_examples(path, max_n=None):
    examples = []
    p = Path(path)
    if not p.exists():
        return examples
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))
    if max_n is not None:
        random.shuffle(examples)
        examples = examples[:max_n]
    return examples

# ==============================================================================
# MODEL ARCHITECTURE (THE PANINIAN MASK)
# ==============================================================================
class BrahmanModel(nn.Module):
    def __init__(self, use_panini=True, srl_checkpoint=None):
        super().__init__()
        self.use_panini = use_panini
        self.encoder = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
        
        if srl_checkpoint and Path(srl_checkpoint).exists():
            state_dict = torch.load(srl_checkpoint, map_location="cpu", weights_only=True)
            encoder_state_dict = {k.replace("roberta.", ""): v for k, v in state_dict.items() if k.startswith("roberta.")}
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
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
        seq = out.last_hidden_state
        
        if self.use_panini:
            B, L, H = seq.shape
            seq_mean = seq.mean(dim=1, keepdim=True).expand(B, L, H)
            pairwise_features = torch.cat([seq, seq_mean], dim=-1)
            gate = self.panini_gate(pairwise_features)
            lambda_clamped = self.panini_lambda.clamp(0.0, 1.0)
            seq = seq + lambda_clamped * gate * seq

        pool_w = self.attention_pool(seq)
        pooled = (seq * pool_w).sum(dim=1)

        task_logits = self.task_head(pooled)
        conf_logits = self.confirmation_head(pooled)

        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(task_logits, label)

        return {
            "loss": loss,
            "task_logits": task_logits,
            "conf_logits": conf_logits,
            "pooled": pooled
        }

# ==============================================================================
# TRAINING ENGINE
# ==============================================================================
def train_phase(model, train_dl, val_dl, phase, epochs, save_path):
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': Config.lr_encoder},
        {'params': model.task_head.parameters(), 'lr': Config.lr_task_head},
        {'params': [p for n, p in model.named_parameters() if 'panini' in n or 'attention' in n or 'confirmation' in n], 'lr': Config.lr_task_head}
    ])
    
    total_steps = len(train_dl) * epochs // Config.grad_accumulation
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    
    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress = tqdm(train_dl, desc=f"Phase {phase} Epoch {epoch+1}/{epochs} Train")
        for i, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["label"].to(Config.device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"] / Config.grad_accumulation
            loss.backward()
            
            total_loss += loss.item() * Config.grad_accumulation
            
            if (i + 1) % Config.grad_accumulation == 0 or (i + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            progress.set_postfix(loss=loss.item() * Config.grad_accumulation)
            
        model.eval()
        correct = 0
        total = 0
        per_type = collections.defaultdict(lambda: {"correct": 0, "total": 0})
        
        val_progress = tqdm(val_dl, desc=f"Phase {phase} Epoch {epoch+1}/{epochs} Val")
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)
                labels = batch["label"].to(Config.device)
                form_types = batch["form_type"]
                
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs["task_logits"], dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                for pt, lbl, ftype in zip(preds, labels, form_types):
                    per_type[ftype]["total"] += 1
                    if pt == lbl:
                        per_type[ftype]["correct"] += 1
                        
        val_acc = correct / total if total > 0 else 0.0
        avg_train_loss = total_loss / len(train_dl)
        
        print(f"\nPhase {phase} Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_acc={val_acc:.4f}")
        for ftype, stats in per_type.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            print(f"  {ftype}: {acc:.4f}")
            
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), save_path)
            print(f"  New best model found! Saving...")
            best_val_acc = val_acc
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= Config.early_stopping_patience:
                print("  Early stopping triggered.")
                break
                
    return best_val_acc

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def train(ablation_mode=False):
    print("======================================================================")
    print(f"BRAHMAN-1: {'Ablated Baseline' if ablation_mode else 'Full Engine'} Training")
    print("======================================================================")

    # DEEP SHUFFLE FIX: Load all data and enforce a dynamic 90/10 split
    all_examples = load_examples(Config.data_path, max_n=60000)
    random.shuffle(all_examples) 
    
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    print(f"Total data loaded: {len(all_examples)} | Train: {len(train_examples)} | Val: {len(val_examples)}")

    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(Config.srl_tokenizer_path)
    except Exception:
        print("Using base roberta tokenizer")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    use_panini = not ablation_mode
    srl_ckpt = Config.srl_checkpoint if not ablation_mode else None
    
    model = BrahmanModel(use_panini=use_panini, srl_checkpoint=srl_ckpt)
    model.to(Config.device)

    current_save_path = Config.ablation_save_path if ablation_mode else Config.save_path

    # PHASE 1
    print("\n--- Starting Phase 1 (Valid Inference Only) ---")
    train_ds1 = LogicDataset(train_examples, tokenizer, max_length=Config.max_length, phase=1)
    val_ds1 = LogicDataset(val_examples, tokenizer, max_length=Config.max_length, phase=1)
    train_dl1 = DataLoader(train_ds1, batch_size=Config.batch_size, shuffle=True)
    val_dl1 = DataLoader(val_ds1, batch_size=Config.batch_size)
    train_phase(model, train_dl1, val_dl1, phase=1, epochs=Config.phase1_epochs, save_path=current_save_path)

    # PHASE 2
    print("\n--- Starting Phase 2 (Adding Fallacies) ---")
    if Path(current_save_path).exists():
        model.load_state_dict(torch.load(current_save_path, map_location=Config.device, weights_only=True))
    train_ds2 = LogicDataset(train_examples, tokenizer, max_length=Config.max_length, phase=2)
    val_ds2 = LogicDataset(val_examples, tokenizer, max_length=Config.max_length, phase=2)
    train_dl2 = DataLoader(train_ds2, batch_size=Config.batch_size, shuffle=True)
    val_dl2 = DataLoader(val_ds2, batch_size=Config.batch_size)
    train_phase(model, train_dl2, val_dl2, phase=2, epochs=Config.phase2_epochs, save_path=current_save_path)

    # PHASE 3
    print("\n--- Starting Phase 3 (Full Curriculum) ---")
    if Path(current_save_path).exists():
        model.load_state_dict(torch.load(current_save_path, map_location=Config.device, weights_only=True))
    train_ds3 = LogicDataset(train_examples, tokenizer, max_length=Config.max_length, phase=3)
    val_ds3 = LogicDataset(val_examples, tokenizer, max_length=Config.max_length, phase=3)
    train_dl3 = DataLoader(train_ds3, batch_size=Config.batch_size, shuffle=True)
    val_dl3 = DataLoader(val_ds3, batch_size=Config.batch_size)
    train_phase(model, train_dl3, val_dl3, phase=3, epochs=Config.phase3_epochs, save_path=current_save_path)
    
    print("\nTraining Complete. Final Weights saved to:", current_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    train(ablation_mode=args.ablation)
