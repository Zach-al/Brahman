import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
from typing import Dict, Any

from tokenizer import PaninianTokenizer
from model import SanskritCoreModel, generate_vibhakti_mask
from dataset import SanskritLogicDataset, collate_fn

def check_model_health(model: nn.Module) -> bool:
    """Check for NaN or Inf in model parameters."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"CRITICAL: {name} has NaN/Inf values")
            return False
    return True

def train():
    parser = argparse.ArgumentParser(description="Brahman Stable Training Loop")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Brahman Stable Training (v2) ---")
    print(f"Targeting Hardware: {device}")

    # Data & Model
    tokenizer = PaninianTokenizer()
    dataset = SanskritLogicDataset(num_samples=10000)
    vocab_size = len(dataset.vocab)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer.operator_map)
    )

    model = SanskritCoreModel(vocab_size=vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-9)
    
    # Label Smoothing for stability
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    # Setup Checkpointing
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            case_ids = batch["case_ids"].to(device)
            mask = batch["vibhakti_mask"].to(device)
            
            if args.ablation:
                seq_len = input_ids.size(1)
                mask = torch.triu(torch.full((seq_len, seq_len), -1e9, device=device), diagonal=1)
                mask = mask.unsqueeze(0).unsqueeze(0)

            optimizer.zero_grad()
            
            # Forward with grammatical case information
            outputs = model(input_ids, mask, case_ids=case_ids)
            logits = outputs.view(-1, vocab_size)
            
            # Target mapping
            target = torch.zeros_like(input_ids)
            target[:, :label_ids.size(1)] = label_ids
            target = target.view(-1)
            
            loss = criterion(logits, target)
            
            if torch.isnan(loss):
                print(f"NaN loss at step {global_step}, skipping...")
                continue
                
            loss.backward()
            
            # CRITICAL: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            if global_step % 50 == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
                if not check_model_health(model):
                    return

            if global_step % 500 == 0:
                torch.save(model.state_dict(), f"{args.checkpoint_dir}/stable_ckpt_{global_step}.pth")

    torch.save(model.state_dict(), "brahman_stable_final.pth")
    print("\n✓ Stable Training Complete.")

if __name__ == "__main__":
    train()
