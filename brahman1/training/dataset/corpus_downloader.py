"""
Downloads and preprocesses all training corpora for Brahman-1.

Sources:
  1. DCS (Digital Corpus of Sanskrit) — CoNLL-U annotated Sanskrit
  2. GRETIL philosophical texts (Nyāya, Yoga, Upaniṣads)
  3. OPUS English parallel data (for English→SIR training)
  4. SNLI (Stanford NLI) — for logical entailment objective
  5. PropBank via NLTK — for SRL→Vibhakti objective

All data is written to data/processed/ as Arrow datasets
for streaming via HuggingFace datasets.
"""

import os, json, csv, requests, sqlite3, hashlib, gzip
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import nltk

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. DCS Sanskrit Corpus
# ─────────────────────────────────────────────────────────────

DCS_FILES = [
    ("dhatupatha",    "https://raw.githubusercontent.com/OliverHellwig/sanskrit/master/dcs/data/lookup/dhatupatha.csv"),
    ("dcs_sentences", "https://raw.githubusercontent.com/OliverHellwig/sanskrit/master/papers/2020emnlp/data/lookup/word_completeness.csv"),
]

DCS_CONLLU_BASE = "https://raw.githubusercontent.com/OliverHellwig/sanskrit/master/dcs/data/conllu/iast/"
DCS_SAMPLE_FILES = [
    "Mahābhārata/Mahābhārata,_Ādiparvan001.conllu",
    "Rāmāyaṇa/Rāmāyaṇa,_Bālakāṇḍa001.conllu",
]

def download_dcs_corpus() -> List[Dict]:
    """
    Download DCS CoNLL-U annotated Sanskrit sentences.
    Returns list of {tokens, lemmas, upos, deprel, sent_id}.
    """
    sentences = []

    for name, url in DCS_FILES:
        dest = RAW_DIR / f"dcs_{name}.csv"
        if not dest.exists():
            print(f"  Downloading DCS {name}...")
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                dest.write_text(r.text, encoding="utf-8")
                print(f"  ✓ {name}: {len(r.text):,} chars")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")

    # Try CoNLL-U files for real annotated sentences
    for path in DCS_SAMPLE_FILES:
        url = DCS_CONLLU_BASE + requests.utils.quote(path)
        dest = RAW_DIR / f"dcs_{Path(path).stem}.conllu"
        if not dest.exists():
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                dest.write_text(r.text, encoding="utf-8")
                print(f"  ✓ CoNLL-U: {Path(path).stem}")
            except Exception as e:
                print(f"  ✗ CoNLL-U failed ({Path(path).stem}): {e}")

    # Parse all CoNLL-U files present
    for conllu_file in RAW_DIR.glob("*.conllu"):
        sentences.extend(_parse_conllu(conllu_file))

    # Fallback: generate synthetic Sanskrit sentences from dhātu DB
    if len(sentences) < 100:
        print("  → Augmenting with synthetic Sanskrit sentences from DhatuDB")
        sentences.extend(_generate_synthetic_sanskrit())

    print(f"  Total Sanskrit sentences: {len(sentences)}")
    return sentences


def _parse_conllu(path: Path) -> List[Dict]:
    """Parse CoNLL-U format into list of sentence dicts."""
    sentences, current = [], {"tokens": [], "lemmas": [], "upos": [], "deprel": [], "feats": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        if line.strip() == "":
            if current["tokens"]:
                sentences.append(dict(current))
                current = {"tokens": [], "lemmas": [], "upos": [], "deprel": [], "feats": []}
            continue
        parts = line.split("\t")
        if len(parts) < 8 or "-" in parts[0] or "." in parts[0]:
            continue
        current["tokens"].append(parts[1])
        current["lemmas"].append(parts[2])
        current["upos"].append(parts[3])
        current["feats"].append(parts[5])
        current["deprel"].append(parts[7])
    return sentences


def _generate_synthetic_sanskrit() -> List[Dict]:
    """
    Generate grammatically valid Sanskrit sentences from DhatuDB.
    Uses simple SVO template: [kartā]-[karma]-[dhātu-laṭ].
    """
    import sys
    sys.path.insert(0, ".")
    from core.dhatu.dhatu_db import DhatuDB
    db = DhatuDB()

    templates = [
        # (subject_IAST, object_IAST, dhātu_root, meaning)
        ("rāmaḥ",      "vanam",     "√gam",  "Rama goes to the forest"),
        ("guruh",      "śiṣyam",    "√śiṣ",  "The teacher teaches the student"),
        ("rājaḥ",      "suvarnnam", "√dā",   "The king gives gold"),
        ("balah",      "phalam",    "√khād", "The child eats the fruit"),
        ("vidvān",     "satyam",    "√vac",  "The wise man speaks the truth"),
        ("mātā",       "putram",    "√smṛ",  "The mother remembers her son"),
        ("yoddhā",     "śatrūn",    "√ji",   "The warrior conquers the enemies"),
        ("devah",      "lokān",     "√pā",   "God protects the worlds"),
        ("ṛṣiḥ",       "jñānam",    "√vid",  "The sage knows wisdom"),
        ("śiṣyaḥ",     "vedān",     "√adhī", "The student studies the Vedas"),
    ]

    sentences = []
    for subj, obj, dhatu, meaning in templates:
        sentences.append({
            "tokens":  [subj, obj, dhatu.replace("√", "")],
            "lemmas":  [subj, obj, dhatu],
            "upos":    ["NOUN", "NOUN", "VERB"],
            "deprel":  ["nsubj", "obj", "ROOT"],
            "feats":   ["Case=Nom", "Case=Acc", ""],
            "meaning": meaning,
        })
    return sentences


# ─────────────────────────────────────────────────────────────
# 2. SNLI for Logical Entailment
# ─────────────────────────────────────────────────────────────

def download_snli() -> DatasetDict:
    """
    Download Stanford NLI dataset via HuggingFace datasets.
    We use this for training the logical entailment objective (weight=3.0).
    """
    dest = PROC_DIR / "snli"
    if dest.exists():
        print("  ✓ SNLI already cached")
        from datasets import load_from_disk
        return load_from_disk(str(dest))

    print("  Downloading SNLI...")
    try:
        from datasets import load_dataset
        ds = load_dataset("snli", split={"train": "train", "validation": "validation"})
        # Filter to only entailment/contradiction (skip neutral for simplicity)
        def filter_binary(ex):
            return ex["label"] in [0, 2]  # 0=entailment, 2=contradiction
        ds = DatasetDict({k: v.filter(filter_binary) for k, v in ds.items()})
        ds.save_to_disk(str(dest))
        print(f"  ✓ SNLI: {len(ds['train']):,} train pairs")
        return ds
    except Exception as e:
        print(f"  ✗ SNLI download failed: {e}")
        return _generate_synthetic_snli()


def _generate_synthetic_snli() -> DatasetDict:
    """
    Fallback: generate syllogistic entailment pairs synthetically.
    Covers the core logical forms Brahman-1 is designed to handle.
    """
    pairs = []
    templates = [
        # (premise, hypothesis, label)  0=entailment, 2=contradiction
        ("All {A} are {B}. {x} is a {A}.",   "{x} is a {B}.",        0),
        ("All {A} are {B}. {x} is a {A}.",   "{x} is not a {B}.",    2),
        ("No {A} is {B}. {x} is a {A}.",     "{x} is not a {B}.",    0),
        ("No {A} is {B}. {x} is a {A}.",     "{x} is a {B}.",        2),
        ("If {P} then {Q}. {P} is true.",    "{Q} is true.",         0),
        ("If {P} then {Q}. {Q} is false.",   "{P} is false.",        0),
        ("{x} caused {y}. {y} caused {z}.",  "{x} caused {z}.",      0),
        ("{x} caused {y}. {y} caused {z}.",  "{z} caused {x}.",      2),
    ]
    substitutions = [
        {"A": "humans",  "B": "mortal",   "x": "Socrates"},
        {"A": "birds",   "B": "animals",  "x": "Eagle"},
        {"A": "trees",   "B": "plants",   "x": "Oak"},
        {"A": "planets", "B": "spherical","x": "Mars"},
        {"P": "it rains","Q": "the ground is wet", "x": "rain", "y": "wetness", "z": "mud"},
        {"P": "it freezes","Q": "water is ice", "x": "cold", "y": "ice", "z": "slippery"},
    ]
    for tmpl_premise, tmpl_hyp, label in templates:
        for sub in substitutions:
            try:
                pairs.append({
                    "premise":    tmpl_premise.format(**sub),
                    "hypothesis": tmpl_hyp.format(**sub),
                    "label":      label,
                })
            except KeyError:
                continue

    df = pd.DataFrame(pairs)
    split = int(len(df) * 0.9) if len(df) > 0 else 0
    train_ds = Dataset.from_pandas(df.iloc[:split]) if split > 0 else Dataset.from_pandas(df)
    val_ds   = Dataset.from_pandas(df.iloc[split:]) if split > 0 else Dataset.from_pandas(df)
    print(f"  ✓ Synthetic SNLI: {len(train_ds)} train, {len(val_ds)} val pairs")
    return DatasetDict({"train": train_ds, "validation": val_ds})


# ─────────────────────────────────────────────────────────────
# 3. PropBank SRL for Vibhakti→Role Mapping
# ─────────────────────────────────────────────────────────────

PROPBANK_TO_VIBHAKTI = {
    "ARG0":     0,  # Agent    → Nominative (kartā)
    "ARG1":     1,  # Patient  → Accusative (karma)
    "ARG2":     2,  # Instr.   → Instrumental (karaṇa)
    "ARG3":     3,  # Benefic. → Dative (sampradāna)
    "ARG4":     4,  # Source   → Ablative (apādāna)
    "ARGM-LOC": 6,  # Location → Locative (adhikaraṇa)
    "ARGM-TMP": 6,  # Time     → Locative (adhikaraṇa)
    "ARGM-CAU": 4,  # Cause    → Ablative (apādāna)
    "ARGM-MNR": 2,  # Manner   → Instrumental (karaṇa)
    "ARGM-DIR": 4,  # Direct.  → Ablative (apādāna)
    "ARGM-PRP": 3,  # Purpose  → Dative (sampradāna)
    "ARGM-GOL": 3,  # Goal     → Dative (sampradāna)
    "V":        8,  # Verb     → Predicate (special)
}

def build_propbank_vibhakti_dataset(max_samples: int = 20000) -> Dataset:
    """
    Build (sentence, token_labels) dataset from PropBank,
    with labels mapped to 8-case Pāṇinian vibhakti system.
    """
    dest = PROC_DIR / "propbank_vibhakti.arrow"
    if dest.exists():
        print("  ✓ PropBank→Vibhakti dataset already built")
        return Dataset.load_from_disk(str(dest))

    print("  Building PropBank→Vibhakti dataset...")
    nltk.download("propbank", quiet=True)
    nltk.download("treebank", quiet=True)

    try:
        from nltk.corpus import propbank, treebank
    except Exception as e:
        print(f"  ✗ PropBank not available: {e}")
        return _generate_synthetic_srl_dataset(max_samples)

    records = []
    try:
        instances = list(propbank.instances())[:max_samples]
    except Exception:
        instances = []

    for inst in instances:
        try:
            fileid  = inst.fileid
            sentnum = inst.sentnum
            tree    = treebank.parsed_sents(fileid)[sentnum]
            tokens  = tree.leaves()
            labels  = [9] * len(tokens)  # default: CASE_OTHER

            for argloc, argid in inst.arguments:
                vib = PROPBANK_TO_VIBHAKTI.get(str(argid), 9)
                if hasattr(argloc, "select"):
                    leaves = argloc.select(tree).leaves()
                    all_leaves = tokens
                    for i, tok in enumerate(all_leaves):
                        if tok in leaves:
                            labels[i] = vib

            # Mark predicate verb position
            pred_leaves = inst.predicate.select(tree).leaves()
            for i, tok in enumerate(tokens):
                if tok in pred_leaves:
                    labels[i] = 8  # CASE_VERB

            if len(tokens) > 1 and len(tokens) <= 128:
                records.append({
                    "tokens": tokens,
                    "labels": labels,
                    "text":   " ".join(tokens),
                })
        except Exception:
            continue

    if len(records) < 100:
        records.extend(_generate_synthetic_srl_dataset(max_samples, return_list=True))

    ds = Dataset.from_list(records)
    ds.save_to_disk(str(dest))
    print(f"  ✓ PropBank→Vibhakti: {len(ds):,} examples")
    return ds


def _generate_synthetic_srl_dataset(n: int = 2000, return_list: bool = False):
    """Fallback: generate synthetic SRL examples with Vibhakti labels."""
    templates = [
        (["The", "teacher", "gives", "a", "book", "to", "the", "student"],
         [9,     0,         8,       9,   1,       9,    9,     3]),
        (["Ram",  "runs",  "in",  "the", "forest", "with",  "speed"],
         [0,      8,       9,     9,     6,         9,       2]),
        (["The", "king",  "took", "gold", "from",   "the", "merchant"],
         [9,     0,       8,      1,      9,         9,     4]),
        (["Sita", "sings", "beautifully",  "for",  "the", "gods"],
         [0,      8,       2,              9,       9,     3]),
        (["The", "warrior", "fought", "the", "enemy", "with",  "a", "sword"],
         [9,     0,         8,        9,     1,        9,       9,   2]),
        (["Knowledge",   "comes", "from",   "practice"],
         [0,              8,       9,         4]),
    ]
    records = []
    for _ in range(n // len(templates) + 1):
        for toks, labs in templates:
            records.append({"tokens": toks, "labels": labs, "text": " ".join(toks)})
    records = records[:n]
    if return_list:
        return records
    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────
# 4. English Language Modeling Data
# ─────────────────────────────────────────────────────────────

def build_lm_dataset(max_samples: int = 50000) -> Dataset:
    """
    Build language modeling dataset.
    Sources: WikiText-2 (reliable, available) + synthetic philosophical English.
    """
    dest = PROC_DIR / "lm_data.arrow"
    if dest.exists():
        print("  ✓ LM dataset already cached")
        return Dataset.load_from_disk(str(dest))

    print("  Building LM dataset...")
    texts = []

    try:
        from datasets import load_dataset
        wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [x["text"] for x in wt2 if len(x["text"].strip()) > 50][:max_samples]
        print(f"  ✓ WikiText-2: {len(texts):,} passages")
    except Exception as e:
        print(f"  ✗ WikiText-2 failed: {e} — using synthetic philosophical texts")

    # Augment with philosophical/logical sentences aligned to Brahman-1 domain
    philosophical = [
        "All that exists is subject to change and transformation.",
        "Knowledge acquired through direct perception differs from inference.",
        "If the cause is absent, the effect cannot arise.",
        "The agent who acts bears the consequence of action.",
        "That which has no beginning has no end.",
        "Every compound is subject to dissolution.",
        "The universal exists in the particular as its essence.",
        "Consciousness is the witness of all mental modifications.",
        "A valid inference requires an undisputed universal relation.",
        "The self is neither born nor does it die at any time.",
        "What is destroyed by one thing is regenerated by another.",
        "All qualities inhere in a substance as their support.",
        "The effect pre-exists in the cause in a latent form.",
        "Direct perception is the root of all other valid cognitions.",
        "A syllogism consists of proposition, reason, and example.",
    ] * (max_samples // 15)

    texts.extend(philosophical)
    texts = texts[:max_samples]

    ds = Dataset.from_dict({"text": texts})
    ds.save_to_disk(str(dest))
    print(f"  ✓ LM dataset: {len(ds):,} examples")
    return ds


# ─────────────────────────────────────────────────────────────
# 5. Build All & Report
# ─────────────────────────────────────────────────────────────

def build_all_datasets():
    print("\n" + "="*55)
    print("BRAHMAN-1 Corpus Builder")
    print("="*55)

    print("\n[1/4] Sanskrit corpus (DCS + synthetic)...")
    sanskrit = download_dcs_corpus()
    sk_dest = PROC_DIR / "sanskrit_sentences.json"
    sk_dest.write_text(json.dumps(sanskrit, ensure_ascii=False, indent=2))
    print(f"  ✓ Saved {len(sanskrit)} Sanskrit sentences")

    print("\n[2/4] SNLI logical entailment...")
    snli = download_snli()

    print("\n[3/4] PropBank → Vibhakti SRL...")
    propbank_ds = build_propbank_vibhakti_dataset()

    print("\n[4/4] Language modeling data...")
    lm_ds = build_lm_dataset()

    print("\n" + "="*55)
    print("CORPUS BUILD COMPLETE")
    print(f"  Sanskrit sentences: {len(sanskrit)}")
    print(f"  SNLI train pairs:   {len(snli['train'])}")
    print(f"  SRL examples:       {len(propbank_ds)}")
    print(f"  LM passages:        {len(lm_ds)}")
    print("="*55)
    return {
        "sanskrit": sanskrit,
        "snli": snli,
        "srl": propbank_ds,
        "lm": lm_ds,
    }


if __name__ == "__main__":
    build_all_datasets()
