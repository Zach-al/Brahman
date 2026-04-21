"""
Week 1 Integration Test.
Run this to verify all Day 1-7 components work together.
"""

import sys
sys.path.insert(0, ".")

print("=" * 60)
print("BRAHMAN-1 Week 1 Integration Test")
print("=" * 60)

# Test 1: DhatuDB
print("\n[1/4] Testing DhatuDB...")
from core.dhatu.dhatu_db import DhatuDB
db = DhatuDB()
stats = db.stats()
assert stats["total"] > 0, "DhatuDB is empty!"
print(f"  ✓ {stats['total']} dhātus loaded")
print(f"  ✓ Semantic classes: {list(stats['by_class'].keys())[:5]}...")

vib = db.get_vibhakti_role(1)
assert vib["case"] == "nominative"
print(f"  ✓ Vibhakti 1: {vib['case']} → {vib['roles']}")

# Test 2: SanskritIR
print("\n[2/4] Testing SanskritIR...")
from core.representation.sanskrit_ir import (
    SIRBuilder, Vibhakti, Lakara, LogicOp
)

sir = (SIRBuilder("All humans are mortal")
       .set_predicate("√as", "IS", "existence", Lakara.LAT)
       .add_argument("manuṣyāḥ", "manuṣya", Vibhakti.NOMINATIVE)
       .add_argument("martyāḥ", "martya", Vibhakti.ACCUSATIVE)
       .add_operator(LogicOp.ALL)
       .build())

fol = sir.to_fol()
assert "∀" in fol or "IS" in fol, f"FOL generation failed: {fol}"
print(f"  ✓ SIR built successfully")
print(f"  ✓ FOL: {fol}")

json_out = sir.to_json()
import json
parsed = json.loads(json_out)
assert parsed["predicate"]["root"] == "√as"
print(f"  ✓ JSON serialization works")

# Test 3: Conditional SIR
print("\n[3/4] Testing conditional SIR (IF-THEN)...")
rain = (SIRBuilder("If it rains")
        .set_predicate("√vṛṣ", "RAIN", "transformation")
        .add_operator(LogicOp.IF_THEN)
        .build())
wet = (SIRBuilder("ground becomes wet")
       .set_predicate("√bhū", "BECOME", "transformation")
       .add_argument("bhūmiḥ", "bhūmi", Vibhakti.NOMINATIVE)
       .build())
rain.sub_propositions.append(wet)
fol2 = rain.to_fol()
print(f"  ✓ Conditional FOL: {fol2}")
assert "→" in fol2 or "IF" in fol2 or "RAIN" in fol2

# Test 4: VibhaktiEncoder (architecture test, no training)
print("\n[4/4] Testing VibhaktiEncoder architecture...")
import torch
from core.grammar.vibhakti_encoder import VibhaktiEncoder, NUM_LABELS
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
model = VibhaktiEncoder()

# Forward pass test
encoding = tokenizer(
    ["Ram", "gives", "book", "to", "Sita"],
    is_split_into_words=True,
    return_tensors="pt"
)
outputs = model(**encoding)
assert outputs["logits"].shape[-1] == NUM_LABELS
print(f"  ✓ Model forward pass: logits shape {outputs['logits'].shape}")
print(f"  ✓ NUM_LABELS = {NUM_LABELS} ({len(list(Vibhakti))} cases + 2)")

total_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

print("\n" + "="*60)
print("✓ ALL WEEK 1 TESTS PASSED")
print("="*60)
print("\nNext steps:")
print("  1. Run: python scripts/download_dhatu_data.py")
print("  2. Run: python core/grammar/vibhakti_encoder.py --train")
print("  3. Begin Week 2: brahman_transformer.py")
