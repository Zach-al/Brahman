"""
Generates synthetic training data using Pāṇini rule templates.
This is critical for the logical entailment objective (weight=3.0).

Types of synthetic data generated:
  1. Syllogisms (modus ponens, modus tollens, hypothetical syllogism)
  2. Causal chains (A→B→C inference)
  3. Case-role manipulation (same event, different case framings)
  4. Compound sentences with explicit logical operators
"""

import json, random, sys
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import Dataset

sys.path.insert(0, ".")
PROC_DIR = Path("data/processed")


# ─── Syllogism Templates ──────────────────────────────────────

ENTITY_CLASSES = [
    ("human",    "mortal",      "Socrates"),
    ("bird",     "animal",      "eagle"),
    ("plant",    "living",      "oak"),
    ("planet",   "spherical",   "Mars"),
    ("river",    "flowing",     "Ganges"),
    ("warrior",  "brave",       "Arjuna"),
    ("scholar",  "learned",     "Pāṇini"),
    ("king",     "powerful",    "Ashoka"),
    ("sage",     "wise",        "Vyāsa"),
    ("student",  "curious",     "Nālanda"),
]

CAUSAL_CHAINS = [
    ("rain",        "wet ground",   "mud",     "flooding"),
    ("fire",        "smoke",        "alarm",   "evacuation"),
    ("study",       "knowledge",    "wisdom",  "liberation"),
    ("seed",        "sprout",       "tree",    "fruit"),
    ("desire",      "action",       "result",  "consequence"),
    ("ignorance",   "delusion",     "bondage", "suffering"),
    ("practice",    "skill",        "mastery", "perfection"),
]


def generate_syllogisms(n: int = 3000) -> List[Dict]:
    records = []
    for _ in range(n):
        cls_a, cls_b, inst = random.choice(ENTITY_CLASSES)

        # Modus Ponens: All A are B. x is A. ∴ x is B.
        records.append({
            "type":      "modus_ponens",
            "premises":  [f"All {cls_a}s are {cls_b}.", f"{inst} is a {cls_a}."],
            "conclusion": f"{inst} is {cls_b}.",
            "label":     1,  # valid
            "fol":       f"∀x({cls_a.upper()}(x)→{cls_b.upper()}(x)) ∧ {cls_a.upper()}({inst}) → {cls_b.upper()}({inst})",
        })

        # Modus Tollens: All A are B. x is not B. ∴ x is not A.
        other_inst = random.choice([e[2] for e in ENTITY_CLASSES if e[2] != inst])
        records.append({
            "type":      "modus_tollens",
            "premises":  [f"All {cls_a}s are {cls_b}.", f"{other_inst} is not {cls_b}."],
            "conclusion": f"{other_inst} is not a {cls_a}.",
            "label":     1,
            "fol":       f"∀x({cls_a.upper()}(x)→{cls_b.upper()}(x)) ∧ ¬{cls_b.upper()}({other_inst}) → ¬{cls_a.upper()}({other_inst})",
        })

        # Invalid: All A are B. x is B. ∴ x is A.  (affirming the consequent — INVALID)
        records.append({
            "type":      "fallacy_affirming_consequent",
            "premises":  [f"All {cls_a}s are {cls_b}.", f"{other_inst} is {cls_b}."],
            "conclusion": f"{other_inst} is a {cls_a}.",
            "label":     0,  # invalid
            "fol":       f"∀x({cls_a.upper()}(x)→{cls_b.upper()}(x)) ∧ {cls_b.upper()}({other_inst}) → {cls_a.upper()}({other_inst})",
        })

        # Hypothetical Syllogism: A→B, B→C ∴ A→C
        cls_c = random.choice(ENTITY_CLASSES)[1]
        records.append({
            "type":      "hypothetical_syllogism",
            "premises":  [f"All {cls_a}s are {cls_b}.", f"All {cls_b}s are {cls_c}."],
            "conclusion": f"All {cls_a}s are {cls_c}.",
            "label":     1,
            "fol":       f"∀x({cls_a.upper()}(x)→{cls_b.upper()}(x)) ∧ ∀x({cls_b.upper()}(x)→{cls_c.upper()}(x)) → ∀x({cls_a.upper()}(x)→{cls_c.upper()}(x))",
        })

    return records


def generate_causal_chains(n: int = 1000) -> List[Dict]:
    records = []
    for _ in range(n):
        a, b, c, d = random.choice(CAUSAL_CHAINS)

        # Valid 3-step chain
        records.append({
            "type":      "causal_chain_3",
            "premises":  [f"{a} causes {b}.", f"{b} causes {c}."],
            "conclusion": f"{a} causes {c}.",
            "label":     1,
            "fol":       f"CAUSE({a},{b}) ∧ CAUSE({b},{c}) → CAUSE({a},{c})",
        })

        # Prevention breaks chain
        records.append({
            "type":      "prevention",
            "premises":  [f"{a} causes {b}.", f"X prevents {b}."],
            "conclusion": f"{a} no longer causes {c}.",
            "label":     1,
            "fol":       f"CAUSE({a},{b}) ∧ PREVENT(X,{b}) → ¬CAUSE({a},{c})",
        })

        # Invalid: reverse causation
        records.append({
            "type":      "reverse_causation",
            "premises":  [f"{a} causes {b}.", f"{b} causes {c}."],
            "conclusion": f"{c} causes {a}.",
            "label":     0,
            "fol":       f"CAUSE({a},{b}) ∧ CAUSE({b},{c}) → CAUSE({c},{a})",
        })

    return records


def build_synthetic_dataset(n_syllogisms: int = 3000, n_causal: int = 1000) -> Dataset:
    dest = PROC_DIR / "synthetic_logic.arrow"
    if dest.exists():
        print("  ✓ Synthetic logic dataset already built")
        return Dataset.load_from_disk(str(dest))

    print("  Generating synthetic logical reasoning dataset...")
    records  = generate_syllogisms(n_syllogisms)
    records += generate_causal_chains(n_causal)
    random.shuffle(records)

    # Convert premises list to string for dataset storage
    for r in records:
        r["premise_text"] = " ".join(r.pop("premises"))

    ds = Dataset.from_list(records)
    ds.save_to_disk(str(dest))
    print(f"  ✓ Synthetic logic: {len(ds):,} examples ({n_syllogisms} syllogisms + {n_causal} causal)")
    return ds


if __name__ == "__main__":
    ds = build_synthetic_dataset()
    print(f"\nLabel distribution:")
    labels = ds["label"]
    print(f"  Valid (1):   {sum(labels):,}")
    print(f"  Invalid (0): {len(labels)-sum(labels):,}")
    print(f"\nType distribution:")
    from collections import Counter
    for t, c in Counter(ds["type"]).most_common():
        print(f"  {t}: {c}")
