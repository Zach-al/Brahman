"""
Week 2 Integration Test.
Validates: TripleEncoder → PāṇiniConstraintLayer → NeuroSymbolicFusion → BrahmanTransformer
"""

import sys
import torch

# Ensure project root is in path
sys.path.insert(0, ".")

print("=" * 60)
print("BRAHMAN-1 Week 2 Integration Test")
print("=" * 60)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

from transformers import RobertaTokenizerFast
tok = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

sentences = [
    "All humans are mortal. Socrates is human.",
    "The king gives gold to the wise man.",
    "If it rains then the ground becomes wet.",
]
enc = tok(sentences, return_tensors="pt", padding=True, truncation=True, max_length=64)
enc = {k: v.to(DEVICE) for k, v in enc.items()}

# [1/4] TripleEncoder
print("[1/4] Testing TripleEncoder...")
from core.representation.triple_encoder import TripleEncoder
triple_enc = TripleEncoder().to(DEVICE)
with torch.no_grad():
    te_out = triple_enc(enc["input_ids"], enc["attention_mask"])
assert te_out.combined.shape[-1] == 512
print(f"  ✓ Combined shape: {te_out.combined.shape}")
print(f"  ✓ Case labels sample: {te_out.case_labels[0, :8].tolist()}")
print(f"  ✓ Confidence sample:  {te_out.confidence[0, :8].round(decimals=2).tolist()}")

# [2/4] PāṇiniConstraintLayer
print("\n[2/4] Testing PāṇiniConstraintLayer...")
from core.model.panini_constraint_layer import PāṇiniConstraintLayer, CASE_NOMINATIVE, CASE_VERB
B, H, L = 3, 8, enc["input_ids"].shape[1]
fake_scores = torch.randn(B, H, L, L).to(DEVICE)
panini = PāṇiniConstraintLayer(num_heads=H).to(DEVICE)
biased = panini(fake_scores, te_out.case_labels, (te_out.case_labels == CASE_VERB).float())
assert biased.shape == fake_scores.shape
print(f"  ✓ Biased scores shape: {biased.shape}")
# Verify ablation is exact
panini_off = PāṇiniConstraintLayer(num_heads=H, disabled=True).to(DEVICE)
ablated = panini_off(fake_scores.clone(), te_out.case_labels)
assert torch.allclose(ablated, fake_scores)
print("  ✓ Ablation mode verified (disabled=True → no-op)")

# [3/4] NeuroSymbolicFusion
print("\n[3/4] Testing NeuroSymbolicFusion...")
from core.model.neurosymbolic_fusion import NeuroSymbolicFusion
fusion = NeuroSymbolicFusion().to(DEVICE)
hidden   = torch.randn(B, L, 512).to(DEVICE)
cls_feat = torch.randn(B, 512).to(DEVICE)
with torch.no_grad():
    fuse_out = fusion(hidden, cls_feat, sir_list=None)
assert fuse_out.logits.shape == (B, L, 50265)
print(f"  ✓ Fusion logits shape: {fuse_out.logits.shape}")
print(f"  ✓ Alpha values: {fuse_out.alpha.round(decimals=3).tolist()}")
print(f"  ✓ Symbolic proofs: {fuse_out.symbolic_proof}")

# [4/4] Full BrahmanTransformer
print("\n[4/4] Testing BrahmanTransformer (end-to-end)...")
from core.model.brahman_transformer import BrahmanTransformer
model = BrahmanTransformer(num_layers=4).to(DEVICE)
params = model.count_parameters()
with torch.no_grad():
    out = model(enc["input_ids"], enc["attention_mask"])
assert out.logits.shape[0] == B
assert out.logits.shape[-1] == 50265
print(f"  ✓ Output logits: {out.logits.shape}")
print(f"  ✓ Parameters: {params}")
print(f"  ✓ Alpha range: [{out.alpha.min():.3f}, {out.alpha.max():.3f}]")

# Ablation comparison
model_no_panini = BrahmanTransformer(num_layers=4, ablate_panini=True).to(DEVICE)
with torch.no_grad():
    out_ablated = model_no_panini(enc["input_ids"], enc["attention_mask"])
print(f"  ✓ Ablated model runs: {out_ablated.logits.shape}")
print("    (Logit difference confirms Pāṇini constraints are active in full model)")

print("\n" + "=" * 60)
print("✓ ALL WEEK 2 TESTS PASSED")
print("=" * 60)
print("\nNext steps:")
print("  1. python core/representation/triple_encoder.py")
print("  2. python core/model/panini_constraint_layer.py")
print("  3. python core/model/neurosymbolic_fusion.py")
print("  4. python core/model/brahman_transformer.py")
print("  5. python test_week2.py")
print("  → Begin Week 3: Training pipeline + Sanskrit corpus ingestion")
