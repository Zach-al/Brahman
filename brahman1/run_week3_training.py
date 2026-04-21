"""
Week 3 Training Launch Script.
Builds all datasets, assembles the DataLoader, and starts training.

Modes:
  python run_week3_training.py --smoke     # 100-step smoke test (fast)
  python run_week3_training.py --full      # Full 10k-step training run
  python run_week3_training.py --ablation  # Train without Pāṇini (for comparison)
"""

import sys, argparse, json
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizerFast

sys.path.insert(0, ".")

from training.dataset.corpus_downloader  import build_all_datasets
from training.dataset.parallel_corpus    import build_parallel_corpus
from training.dataset.synthetic_gen      import build_synthetic_dataset
from mac_optimized.mps_trainer           import BrahmanTrainer, TrainConfig, BrahmanTrainingDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke",    action="store_true", help="100-step smoke test")
    p.add_argument("--full",     action="store_true", help="Full training run")
    p.add_argument("--ablation", action="store_true", help="Train without Pāṇini constraints")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Build Datasets ─────────────────────────────────────
    print("Building datasets...")
    corpora  = build_all_datasets()
    parallel = build_parallel_corpus()
    synth    = build_synthetic_dataset()

    # Combine SNLI + synthetic logic for entailment objective
    snli_records   = [dict(x) for x in corpora["snli"]["train"]]
    synth_records  = [
        {"premise": r["premise_text"], "hypothesis": r["conclusion"], "label": r["label"]}
        for r in synth
    ]
    all_entailment = snli_records + synth_records

    srl_records     = [dict(x) for x in corpora["srl"]]
    lm_texts        = [x["text"] for x in corpora["lm"]]
    parallel_records = [dict(x) for x in parallel]

    print(f"\nDataset sizes:")
    print(f"  LM texts:        {len(lm_texts):,}")
    print(f"  SRL records:     {len(srl_records):,}")
    print(f"  Entailment:      {len(all_entailment):,}")
    print(f"  Parallel (en↔SIR):{len(parallel_records):,}")

    # ── Tokenizer ──────────────────────────────────────────
    print("\nLoading tokenizer...")
    tok = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

    # ── Assemble Dataset ──────────────────────────────────
    max_len = 64 if args.smoke else 128
    full_ds = BrahmanTrainingDataset(
        lm_texts      = lm_texts,
        srl_records   = srl_records,
        snli_records  = all_entailment,
        parallel_records = parallel_records,
        tokenizer     = tok,
        max_length    = max_len,
    )

    val_size   = min(500, len(full_ds) // 10)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=4 if args.smoke else 8,
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4 if args.smoke else 8,
        shuffle=False, num_workers=0, drop_last=True,
    )

    print(f"\nTrain batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    # ── Training Configuration ─────────────────────────────
    if args.smoke:
        cfg = TrainConfig(
            num_layers  = 2,
            max_steps   = 100,
            grad_accum  = 2,
            eval_every  = 50,
            save_every  = 100,
            run_name    = "brahman1_smoke",
        )
    elif args.ablation:
        cfg = TrainConfig(
            num_layers    = 6,
            max_steps     = 10000,
            ablate_panini = True,
            run_name      = "brahman1_ablation_nopanini",
        )
    else:
        cfg = TrainConfig(
            num_layers = 6,
            max_steps  = 10000,
            run_name   = "brahman1_week3_full",
        )

    # ── Train ─────────────────────────────────────────────
    trainer = BrahmanTrainer(cfg)
    trainer.train(train_loader, val_loader)

    print("\n" + "="*55)
    print(f"Training complete: models/{cfg.run_name}/")
    print("Next: python test_week3.py")
    print("="*55)


if __name__ == "__main__":
    main()
