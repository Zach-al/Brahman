"""
Builds English → SanskritIR parallel pairs for training.
These are the training examples for Objective 5: Cross-lingual Alignment.

Format: {english_text, sir_json, fol_string, case_assignments}

Generated via:
  1. Run English sentences through VibhaktiEncoder → SIR
  2. Convert SIR → FOL
  3. Store triplet (English, SIR, FOL) as training target

This teaches the model that English structure and Sanskrit
case structure are two views of the same semantic content.
"""

import json, sys
from pathlib import Path
from typing import List, Dict
from datasets import Dataset

sys.path.insert(0, ".")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)


SEED_SENTENCES = [
    # Logical
    "All humans are mortal.",
    "Socrates is a human.",
    "Therefore Socrates is mortal.",
    "If it rains then the ground becomes wet.",
    "It is raining.",
    "Therefore the ground is wet.",
    "No bird is a mammal.",
    "An eagle is a bird.",
    "Therefore an eagle is not a mammal.",
    "Every effect has a cause.",
    "This event has no cause.",
    "Therefore this event is not an effect.",
    # Causal
    "The fire caused the smoke.",
    "The smoke caused the alarm.",
    "Therefore the fire caused the alarm.",
    "The teacher gave knowledge to the student.",
    "The student applied the knowledge to solve the problem.",
    # Agent-Patient
    "Ram goes to the forest.",
    "The king gives gold to the wise man.",
    "The warrior fights the enemy with a sword.",
    "The sage teaches the students near the river.",
    "The mother protects her child from danger.",
    "The farmer cuts the grass with a sickle at dawn.",
    # Instrumental
    "He writes with a pen.",
    "She travels by horse from the city.",
    "The carpenter builds a house with wood and nails.",
    # Locative / Temporal
    "The birds sing in the morning.",
    "The lotus blooms in water.",
    "The stars appear in the sky at night.",
    # Conditional
    "If the seed is not planted the fruit will not grow.",
    "If one studies diligently one gains wisdom.",
    "If there is no desire there is no suffering.",
    # Compound / Genitive
    "The knowledge of the self is the highest knowledge.",
    "The son of the king ruled the kingdom.",
    "The color of the sky changes at sunset.",
]


def build_parallel_corpus(max_samples: int = 5000) -> Dataset:
    dest = PROC_DIR / "parallel_corpus.arrow"
    if dest.exists():
        print("  ✓ Parallel corpus already built")
        return Dataset.load_from_disk(str(dest))

    print("  Building English → SIR parallel corpus...")

    try:
        import torch
        from transformers import RobertaTokenizerFast
        from core.grammar.vibhakti_encoder import VibhaktiEncoder
        from core.representation.sanskrit_ir import SIRBuilder, Vibhakti, Lakara, LogicOp

        import os
        tok   = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
        model = VibhaktiEncoder()
        ckpt  = "models/vibhakti_encoder/best_model.pt"
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()

    except Exception as e:
        print(f"  ✗ Encoder load failed: {e} — using rule-based fallback")
        return _build_rule_based_parallel(SEED_SENTENCES * (max_samples // len(SEED_SENTENCES) + 1))

    records = []
    sentences = (SEED_SENTENCES * (max_samples // len(SEED_SENTENCES) + 1))[:max_samples]

    with torch.no_grad():
        for sent in sentences:
            try:
                sir = model.encode_to_sir(sent, tok)
                records.append({
                    "english":  sent,
                    "sir_json": sir.to_json(),
                    "fol":      sir.to_fol(),
                    "n_args":   len(sir.arguments),
                })
            except Exception:
                continue

    if len(records) == 0:
         return _build_rule_based_parallel(sentences)

    ds = Dataset.from_list(records)
    ds.save_to_disk(str(dest))
    print(f"  ✓ Parallel corpus: {len(ds):,} pairs")
    return ds


def _build_rule_based_parallel(sentences: List[str]) -> Dataset:
    """
    Rule-based fallback that creates minimal SIR from sentence structure
    without requiring a trained VibhaktiEncoder.
    """
    from core.representation.sanskrit_ir import SIRBuilder, Vibhakti, Lakara, LogicOp
    records = []
    for sent in sentences[:2000]:
        words = sent.replace(".", "").replace(",", "").split()
        if len(words) < 2:
            continue
        # Heuristic: first noun = subject, last noun = object, middle = verb
        try:
            builder = (SIRBuilder(sent)
                       .set_predicate("√kṛ", "DO", "action", Lakara.LAT)
                       .add_argument(words[0], words[0].lower(), Vibhakti.NOMINATIVE))
            if len(words) > 2:
                builder = builder.add_argument(words[-1], words[-1].lower(), Vibhakti.ACCUSATIVE)
            if "if" in sent.lower():
                builder = builder.add_operator(LogicOp.IF_THEN)
            sir = builder.build()
            records.append({
                "english":  sent,
                "sir_json": sir.to_json(),
                "fol":      sir.to_fol(),
                "n_args":   len(sir.arguments),
            })
        except Exception:
            continue
    ds = Dataset.from_list(records)
    print(f"  ✓ Rule-based parallel corpus: {len(ds)} pairs")
    return ds


if __name__ == "__main__":
    ds = build_parallel_corpus()
    if len(ds) > 0:
        print(f"\nSample record:")
        sample = ds[0]
        print(f"  English: {sample['english']}")
        print(f"  FOL:     {sample['fol']}")
        print(f"  Args:    {sample['n_args']}")
