"""
Week 3 Integration Test.
Verifies: corpus pipeline → dataset assembly → training objectives → trainer.
"""

import sys, torch, json
sys.path.insert(0, ".")

print("=" * 60)
print("BRAHMAN-1 Week 3 Integration Test")
print("=" * 60)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

# [1/5] Corpus Pipeline
print("[1/5] Testing corpus pipeline...")
from training.dataset.corpus_downloader import (
    download_dcs_corpus, _generate_synthetic_snli, build_propbank_vibhakti_dataset
)
sanskrit = download_dcs_corpus()
assert len(sanskrit) > 0
print(f"  ✓ Sanskrit sentences: {len(sanskrit)}")

snli = _generate_synthetic_snli()
assert len(snli["train"]) > 0
print(f"  ✓ SNLI/synthetic entailment: {len(snli['train'])} train pairs")

srl_ds = build_propbank_vibhakti_dataset(max_samples=500)
assert len(srl_ds) > 0
print(f"  ✓ SRL dataset: {len(srl_ds)} examples")

# [2/5] Synthetic Logic Generator
print("\n[2/5] Testing synthetic logic generator...")
from training.dataset.synthetic_gen import generate_syllogisms, generate_causal_chains
syllogisms = generate_syllogisms(100)
causals    = generate_causal_chains(50)
assert len(syllogisms) == 400
assert all("fol" in s for s in syllogisms)
valid   = sum(s["label"] == 1 for s in syllogisms)
invalid = sum(s["label"] == 0 for s in syllogisms)
print(f"  ✓ Syllogisms: {len(syllogisms)} ({valid} valid, {invalid} invalid)")
print(f"  ✓ Causal chains: {len(causals) * 3}")
print(f"  ✓ Sample FOL: {syllogisms[0]['fol'][:80]}...")

# [3/5] Multi-Task Objectives
print("\n[3/5] Testing all 5 training objectives...")
from training.objectives.all_objectives import BrahmanMultiTaskLoss

B, L, V, D = 2, 32, 50265, 512
criterion = BrahmanMultiTaskLoss(model_dim=D).to(DEVICE)

dummy = lambda *shape: torch.randn(*shape).to(DEVICE)
dummy_long = lambda *shape, val=0: torch.full(shape, val, dtype=torch.long).to(DEVICE)

losses = criterion(
    lm_logits   = dummy(B, L, V),
    lm_labels   = dummy_long(B, L),
    srl_logits  = dummy(B, L, 10),
    srl_labels  = dummy_long(B, L, val=9),
    premise_cls = dummy(B, D),
    hypothesis_cls  = dummy(B, D),
    entailment_labels = dummy_long(B, val=1),
    morph_enc       = dummy(B, L, 128),
    compound_labels = dummy_long(B, L, val=0),
    english_cls = dummy(B, D),
    sir_cls     = dummy(B, D),
)

print(f"  - LM:         {losses.lm.item()}")
print(f"  - SRL:        {losses.srl.item()}")
print(f"  - Entailment: {losses.entailment.item()}")
print(f"  - Compound:   {losses.compound.item()}")
print(f"  - Alignment:  {losses.alignment.item()}")
print(f"  - Total:      {losses.total.item()}")

assert not torch.isnan(losses.total), "NaN in total loss!"
print(f"  ✓ LM loss:         {losses.lm.item():.4f}")
print(f"  ✓ SRL loss:        {losses.srl.item():.4f}")
print(f"  ✓ Entailment loss: {losses.entailment.item():.4f}")
print(f"  ✓ Compound loss:   {losses.compound.item():.4f}")
print(f"  ✓ Alignment loss:  {losses.alignment.item():.4f}")
print(f"  ✓ Total loss:      {losses.total.item():.4f}")

# [4/5] Dataset Assembly
print("\n[4/5] Testing BrahmanTrainingDataset assembly...")
from transformers import RobertaTokenizerFast
from mac_optimized.mps_trainer import BrahmanTrainingDataset
from torch.utils.data import DataLoader

tok = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
snli_list = [dict(x) for x in snli["train"]]
srl_list  = [dict(x) for x in srl_ds]
par_list  = [{"english": "Ram goes to the forest.", "fol": "GO(ram, forest)"}]
lm_list   = ["All things that exist are subject to change."] * 20

ds = BrahmanTrainingDataset(
    lm_texts=lm_list, srl_records=srl_list,
    snli_records=snli_list, parallel_records=par_list,
    tokenizer=tok, max_length=64,
)
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
batch  = next(iter(loader))

assert "lm_input_ids"       in batch
assert "srl_labels"         in batch
assert "entailment_label"   in batch
assert "align_en_ids"       in batch
print(f"  ✓ Dataset length: {len(ds)}")
print(f"  ✓ Batch keys: {sorted(batch.keys())}")
print(f"  ✓ LM batch shape: {batch['lm_input_ids'].shape}")

# [5/5] Trainer Smoke Test (5 steps, no actual training data needed)
print("\n[5/5] Trainer smoke test (5 steps)...")
from mac_optimized.mps_trainer import BrahmanTrainer, TrainConfig

cfg     = TrainConfig(num_layers=2, max_steps=5, grad_accum=1,
                      eval_every=999, save_every=999, run_name="week3_test")
trainer = BrahmanTrainer(cfg)

mini_loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
try:
    trainer.train(mini_loader, val_loader=None)
    print("  ✓ 5-step training run completed without error")
except Exception as e:
    print(f"  ✗ Trainer error: {e}")
    raise

print("\n" + "=" * 60)
print("✓ ALL WEEK 3 TESTS PASSED")
print("=" * 60)
print("\nTo launch full training:")
print("  python run_week3_training.py --smoke    # verify first (5-10 min)")
print("  python run_week3_training.py --full     # full run (~8-12h on M-series)")
print("  python run_week3_training.py --ablation # no-Pāṇini baseline (~same time)")
print("\nBoth --full and --ablation must complete before Week 4 evaluation.")
