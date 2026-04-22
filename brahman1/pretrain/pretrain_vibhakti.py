import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast, RobertaModel, get_cosine_schedule_with_warmup
from pathlib import Path
import json
import random
from tqdm import tqdm

ROLE_LABELS = ["O", "V", "AGENT", "PATIENT", "INSTRUMENT", "GOAL", "SOURCE", "POSSESSOR", "LOCATION", "TIME", "CAUSE", "MANNER"]
LABEL2IDX = {l: i for i, l in enumerate(ROLE_LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(ROLE_LABELS)}
NUM_LABELS = 12

def generate_srl_examples(n=50000):
    agents = ["Ram", "Sita", "the king", "the teacher", "the scholar", "the warrior", "the farmer", "the doctor", "Arjuna", "Krishna"]
    patients = ["the book", "the sword", "the medicine", "the grain", "the water", "the treasure", "the arrow", "the law"]
    instruments = ["a pen", "great effort", "wisdom", "a sword", "his hands", "a bow", "the law", "knowledge"]
    goals = ["the student", "the poor", "the village", "the army", "the sick", "the temple", "the people", "his disciples"]
    sources = ["the forest", "the city", "the mountain", "the river", "the palace", "the market", "the battlefield", "the east"]
    locations = ["the forest", "the palace", "the river", "the mountain", "the field", "the temple", "the city", "the garden"]
    times = ["at dawn", "at night", "in the morning", "long ago", "immediately", "always", "in winter", "yesterday"]
    causes = ["hunger", "fear", "wisdom", "duty", "love", "anger", "necessity", "courage"]
    verbs = ["taught", "struck", "helped", "protected", "found", "read", "carried", "sent", "defeated", "created"]

    examples = []
    
    def get_phrase_labels(phrase, label_type):
        return [label_type] * len(phrase.split())

    for _ in range(n):
        template = random.randint(1, 8)
        
        agent = random.choice(agents)
        verb = random.choice(verbs)
        
        if template == 1:
            patient = random.choice(patients)
            tokens = agent.split() + verb.split() + patient.split() + ["."]
            labels = get_phrase_labels(agent, "AGENT") + get_phrase_labels(verb, "V") + get_phrase_labels(patient, "PATIENT") + ["O"]
            
        elif template == 2:
            patient = random.choice(patients)
            instrument = random.choice(instruments)
            tokens = agent.split() + verb.split() + patient.split() + ["with"] + instrument.split() + ["."]
            labels = get_phrase_labels(agent, "AGENT") + get_phrase_labels(verb, "V") + get_phrase_labels(patient, "PATIENT") + ["O"] + get_phrase_labels(instrument, "INSTRUMENT") + ["O"]
            
        elif template == 3:
            patient = random.choice(patients)
            goal = random.choice(goals)
            v = "gave"
            tokens = agent.split() + [v] + patient.split() + ["to"] + goal.split() + ["."]
            labels = get_phrase_labels(agent, "AGENT") + ["V"] + get_phrase_labels(patient, "PATIENT") + ["O"] + get_phrase_labels(goal, "GOAL") + ["O"]
            
        elif template == 4:
            source = random.choice(sources)
            v = "came"
            tokens = agent.split() + [v, "from"] + source.split() + ["."]
            labels = get_phrase_labels(agent, "AGENT") + ["V", "O"] + get_phrase_labels(source, "SOURCE") + ["O"]
            
        elif template == 5:
            location = random.choice(locations)
            v = "lived"
            tokens = agent.split() + [v, "in"] + location.split() + ["."]
            labels = get_phrase_labels(agent, "AGENT") + ["V", "O"] + get_phrase_labels(location, "LOCATION") + ["O"]
            
        elif template == 6:
            cause = random.choice(causes)
            tokens = agent.split() + verb.split() + ["because", "of"] + cause.split() + ["."]
            labels = get_phrase_labels(agent, "AGENT") + get_phrase_labels(verb, "V") + ["O", "O"] + get_phrase_labels(cause, "CAUSE") + ["O"]
            
        elif template == 7:
            time = random.choice(times)
            tokens = agent.split() + verb.split() + time.split() + ["."]
            labels = get_phrase_labels(agent, "AGENT") + get_phrase_labels(verb, "V") + get_phrase_labels(time, "TIME") + ["O"]
            
        elif template == 8:
            patient = random.choice(patients)
            location = random.choice(locations)
            tokens = agent.split() + verb.split() + patient.split() + ["in"] + location.split() + ["."]
            labels = get_phrase_labels(agent, "AGENT") + get_phrase_labels(verb, "V") + get_phrase_labels(patient, "PATIENT") + ["O"] + get_phrase_labels(location, "LOCATION") + ["O"]

        sentence = " ".join(tokens)
        examples.append({
            "sentence": sentence,
            "tokens": tokens,
            "labels": labels
        })
        
    return examples

class SRLDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex["tokens"]
        word_labels = ex["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids()
        
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_str = word_labels[word_idx]
                label_ids.append(LABEL2IDX[label_str])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return item

class VibhaktiSRLHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.15),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, NUM_LABELS)
        )
        weights = torch.ones(NUM_LABELS)
        weights[0] = 0.1 # Downweight 'O' label
        self.register_buffer("class_weights", weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=self.class_weights)
            loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
            
        return {"loss": loss, "logits": logits}

def pretrain_srl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_train = 45000
    n_val = 5000
    epochs = 8
    batch_size = 32
    lr = 3e-5
    save_dir = Path("models/pretrained_vibhakti")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating examples...")
    examples = generate_srl_examples(n=n_train + n_val)
    train_examples = examples[:n_train]
    val_examples = examples[n_train:]

    print("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

    train_dataset = SRLDataset(train_examples, tokenizer)
    val_dataset = SRLDataset(val_examples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Building model...")
    model = VibhaktiSRLHead().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask, labels=labels)
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                active_labels = labels.view(-1) != -100
                correct_preds += (preds.view(-1)[active_labels] == labels.view(-1)[active_labels]).sum().item()
                total_preds += active_labels.sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_preds / total_preds if total_preds > 0 else 0

        print(f"Epoch {epoch+1} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f} | val_acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("  New best model found! Saving...")
            torch.save(model.state_dict(), save_dir / "best_srl.pt")

    tokenizer.save_pretrained(str(save_dir))
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = pretrain_srl()
    print("✓ VibhaktiEncoder pre-training complete")
    print(f"✓ Weights saved to models/pretrained_vibhakti/best_srl.pt")
