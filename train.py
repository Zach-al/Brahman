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

def train():
    parser = argparse.ArgumentParser(description="SanskritCore Cloud Training Loop")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ablation", action="store_true", help="Override Vibhakti Mask with standard causal mask")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # 1. Hardware Initialization (Cloud GPU Target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- SanskritCore Execution Initialized ---")
    print(f"Targeting Hardware: {device}")
    if args.ablation:
        print("MODE: Ablation Study (Causal Mask Only)")
    else:
        print("MODE: Neuro-Symbolic (Pāṇinian Vibhakti Mask)")

    # 2. Setup Checkpointing
    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    # 3. Data Preparation
    tokenizer = PaninianTokenizer()
    dataset = SanskritLogicDataset(num_samples=10000)
    vocab_size = len(dataset.vocab)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer.operator_map)
    )

    # 4. Model Initialization
    model = SanskritCoreModel(vocab_size=vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore [PAD]
    
    # Enable Automatic Mixed Precision (AMP) for T4 Tensor Cores
    scaler = torch.cuda.amp.GradScaler()

    # 5. Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            mask = batch["vibhakti_mask"].to(device)
            
            # Ablation Override: Replace Grammatical Mask with Causal Mask
            if args.ablation:
                seq_len = input_ids.size(1)
                mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
                mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]

            optimizer.zero_grad()
            
            # AMP Autocast
            with torch.cuda.amp.autocast():
                # We use the premise to predict the conclusion
                # For simplicity in this prototype, we pass input_ids through the transformer
                # and take the last hidden states to predict the label_ids sequence.
                # Since label_ids might have different length, we clip/pad to match.
                outputs = model(input_ids, mask)
                
                # Reshape for loss calculation
                # [Batch, Seq, Vocab] -> [Batch * Seq, Vocab]
                logits = outputs.view(-1, vocab_size)
                
                # Align labels (simple sequence-to-sequence target mapping)
                target = torch.zeros_like(input_ids)
                target[:, :label_ids.size(1)] = label_ids
                target = target.view(-1)
                
                loss = criterion(logits, target)

            # Scaled Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            global_step += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f}")
                
            # Periodic Cloud Checkpointing
            if global_step % 500 == 0:
                ckpt_file = checkpoint_path / f"ckpt_step_{global_step}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, ckpt_file)
                print(f"✓ Checkpoint saved: {ckpt_file}")

        print(f"Epoch {epoch} Complete. Average Loss: {total_loss / len(loader):.4f}")

    # Final Save
    torch.save(model.state_dict(), "sanskrit_core_final.pth")
    print("\n✓ Training Complete. Model saved to sanskrit_core_final.pth")

if __name__ == "__main__":
    train()
