"""
BRAHMAN-1 REAL BENCHMARK — Mac Local Version
Run from the root of your Brahman repo:
  python3 evaluation/real_benchmark.py
"""

import torch
import sqlite3
import hashlib
import json
import sys
import os
from pathlib import Path
from transformers import RobertaTokenizerFast
from tabulate import tabulate

# ── PATHS — edit these to match your local file locations ────────
REPO_ROOT      = Path(__file__).parent.parent
FULL_MODEL     = REPO_ROOT / "brahman_full.pth"
ABLATED_MODEL  = REPO_ROOT / "brahman_ablation.pth"
SRL_WEIGHTS    = REPO_ROOT / "models/pretrained_vibhakti"
DHATU_DB       = REPO_ROOT / "data/processed/dhatu.db"
TRAIN_PY       = REPO_ROOT / "train.py"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "brahman1"))

# ── 1. VERIFY FILES EXIST ────────────────────────────────────────
print("="*60)
print("BRAHMAN-1 REAL BENCHMARK — Mac Local")
print("="*60)
print("\nVerifying files...")

missing = []
for label, path in [
    ("Full model",    FULL_MODEL),
    ("Ablated model", ABLATED_MODEL),
    ("SRL weights",   SRL_WEIGHTS / "best_srl.pt"),
    ("train.py",      TRAIN_PY),
]:
    if path.exists():
        mb = path.stat().st_size / 1e6
        print(f"  ✓ {label}: {path} ({mb:.1f}MB)")
    else:
        print(f"  ✗ MISSING: {label}: {path}")
        missing.append(path)

if missing:
    print("\nMissing files. Check paths at top of script.")
    sys.exit(1)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")

# ── 2. LOAD MODEL CLASS FROM train.py ────────────────────────────
import re

print("\nLoading model architecture from train.py...")
with open(TRAIN_PY) as f:
    src = f.read()

# Fix 1: Patch SRL checkpoint path so BrahmanModel loads pretrained weights
# instead of falling back to base RoBERTa (which causes VALID bias)
src = re.sub(
    r'srl_checkpoint\s*=\s*["\'][^"\']*["\']',
    'srl_checkpoint = "models/pretrained_vibhakti/best_srl.pt"',
    src
)
src = re.sub(
    r'srl_tokenizer_path\s*=\s*["\'][^"\']*["\']',
    'srl_tokenizer_path = "models/pretrained_vibhakti"',
    src
)

train_src = src.split("if __name__")[0]
ns = {}
exec(train_src, ns)
BrahmanModel = ns["BrahmanModel"]
print("  ✓ BrahmanModel class loaded (SRL path patched)")

def load_model(path, use_panini):
    m = BrahmanModel(use_panini=use_panini)
    state = torch.load(path, map_location="cpu")
    result = m.load_state_dict(state, strict=False)
    label = "Full (Pāṇini ON)" if use_panini else "Ablated (Pāṇini OFF)"
    print(f"  ✓ {label} loaded — "
          f"missing={len(result.missing_keys)} "
          f"unexpected={len(result.unexpected_keys)}")
    return m.to(DEVICE).eval()

full_model    = load_model(FULL_MODEL,    use_panini=True)
ablated_model = load_model(ABLATED_MODEL, use_panini=False)

# ── 3. TOKENIZER ─────────────────────────────────────────────────
tok_config = SRL_WEIGHTS / "tokenizer_config.json"
tokenizer = RobertaTokenizerFast.from_pretrained(
    str(SRL_WEIGHTS) if tok_config.exists() else "roberta-base"
)
print(f"  ✓ Tokenizer loaded")

# ── 4. DHĀTU DATABASE ────────────────────────────────────────────
# Fix 2: Rebuild dhātu DB if missing or empty (0 rows)
print("\nConnecting to dhātu database...")

def _ensure_dhatu_db():
    """Ensure dhatu.db exists and is populated. Rebuild from seed if empty."""
    DHATU_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DHATU_DB)
    try:
        count = conn.execute("SELECT COUNT(*) FROM dhatus").fetchone()[0]
    except sqlite3.OperationalError:
        count = 0
    
    if count == 0:
        print("  ⚠ dhatu.db is empty — rebuilding from brahman1/core/dhatu/dhatu_db.py...")
        conn.close()
        # Run the DhatuDB initializer which populates from seed data
        dhatu_db_script = REPO_ROOT / "brahman1" / "core" / "dhatu" / "dhatu_db.py"
        if dhatu_db_script.exists():
            old_cwd = os.getcwd()
            os.chdir(str(REPO_ROOT))
            dhatu_ns = {}
            exec(open(dhatu_db_script).read(), dhatu_ns)
            DhatuDB = dhatu_ns["DhatuDB"]
            db = DhatuDB()
            stats = db.stats()
            print(f"  ✓ DhatuDB rebuilt: {stats['total']} dhātus")
            os.chdir(old_cwd)
        conn = sqlite3.connect(DHATU_DB)
        count = conn.execute("SELECT COUNT(*) FROM dhatus").fetchone()[0]
    
    print(f"  ✓ dhatu.db connected — {count} dhātus")
    return conn

conn = _ensure_dhatu_db()

def lookup_dhatu(root: str) -> dict:
    row = conn.execute(
        "SELECT root_iast, meaning, semantic_class, logic_predicate "
        "FROM dhatus WHERE root_iast = ?", (root,)
    ).fetchone()
    if row:
        return {"root": row[0], "meaning": row[1],
                "semantic_class": row[2], "logic_predicate": row[3],
                "found": True}
    # Fuzzy: search meaning
    clean = root.replace("√","").lower()
    row = conn.execute(
        "SELECT root_iast, meaning, semantic_class, logic_predicate "
        "FROM dhatus WHERE meaning LIKE ?", (f"%{clean}%",)
    ).fetchone()
    if row:
        return {"root": row[0], "meaning": row[1],
                "semantic_class": row[2], "logic_predicate": row[3],
                "found": True, "fuzzy": True}
    return {"found": False, "root": root,
            "logic_predicate": "NOT_FOUND"}

# ── 5. TEST CASES — real Sanskrit (IAST) ─────────────────────────
test_cases = [
    {
        "id": 1,
        "sanskrit": "rāmaḥ vanam gacchati",
        "english": "Rama goes to the forest",
        "verb_root": "√gam",
        "is_valid": True,
        "form_type": "simple_action",
        "expected_fol": "∃x∃w(AGENT(x,rāma) ∧ GOAL(w,vana) ∧ GO(x,w))"
    },
    {
        "id": 2,
        "sanskrit": "sarve manuṣyāḥ martyāḥ santi. sōkrāṭaḥ manuṣyaḥ asti.",
        "english": "All humans are mortal. Socrates is human.",
        "verb_root": "√as",
        "is_valid": True,
        "form_type": "modus_ponens",
        "expected_fol": "(∀x(manuṣya(x)→martya(x)) ∧ manuṣya(sōkrāṭa)) → martya(sōkrāṭa)"
    },
    {
        "id": 3,
        "sanskrit": "yadi varṣati bhūmiḥ ārdra bhavati. na varṣati.",
        "english": "If it rains the ground gets wet. It does not rain.",
        "verb_root": "√bhū",
        "is_valid": False,
        "form_type": "fallacy_denying_antecedent",
        "expected_fol": "((rain→wet) ∧ ¬rain) ↛ ¬wet"
    },
    {
        "id": 4,
        "sanskrit": "sarve śūrāḥ dhīrāḥ santi. rāmaḥ dhīraḥ asti.",
        "english": "All warriors are brave. Rama is brave.",
        "verb_root": "√as",
        "is_valid": False,
        "form_type": "fallacy_affirming_consequent",
        "expected_fol": "(∀x(śūra(x)→dhīra(x)) ∧ dhīra(rāma)) ↛ śūra(rāma)"
    },
    {
        "id": 5,
        "sanskrit": "yadi setubandhanam bhagnam nadī pravahati. "
                    "yadi nadī pravahati kṣetrāṇi jīryanti. "
                    "setubandhanam bhagnam.",
        "english": "If dam breaks river flows. If river flows fields flood. Dam broke.",
        "verb_root": "√bhañj",
        "is_valid": True,
        "form_type": "hypothetical_syllogism",
        "expected_fol": "((p→q)∧(q→r)∧p) → r"
    },
    {
        "id": 6,
        "sanskrit": "sarve martyāḥ mṛtyum bibhyati. ṛṣiḥ mṛtyum na bibhyati.",
        "english": "All mortals fear death. The sage does not fear death.",
        "verb_root": "√bhī",
        "is_valid": False,
        "form_type": "modus_tollens",
        "expected_fol": "(∀x(martya(x)→bhī(x)) ∧ ¬bhī(ṛṣi)) → ¬martya(ṛṣi)"
    },
    {
        "id": 7,
        "sanskrit": "agniḥ dhūmam karoti. dhūmaḥ nirgamanam karoti. "
                    "varṣaḥ agnim nivārayati. adya varṣaḥ.",
        "english": "Fire causes smoke. Smoke causes evacuation. Rain prevents fire. It rained.",
        "verb_root": "√kṛ",
        "is_valid": False,
        "form_type": "causal_prevention",
        "expected_fol": "(CAUSE(agni,dhūma)∧CAUSE(dhūma,nirg)∧PREVENT(varṣa,agni)∧varṣa)→¬nirg"
    },
    {
        "id": 8,
        "sanskrit": "rāmaḥ na na śūraḥ asti.",
        "english": "Rama is not not a warrior.",
        "verb_root": "√as",
        "is_valid": True,
        "form_type": "double_negation",
        "expected_fol": "¬¬śūra(rāma) ↔ śūra(rāma)"
    },
    {
        "id": 9,
        "sanskrit": "vā durbhikṣam vā yuddham kāraṇam āsīt. "
                    "durbhikṣam kāraṇam na āsīt.",
        "english": "Either famine or war was the cause. Famine was not the cause.",
        "verb_root": "√as",
        "is_valid": True,
        "form_type": "disjunctive_syllogism",
        "expected_fol": "((p∨q) ∧ ¬p) → q"
    },
    {
        "id": 10,
        "sanskrit": "nadī-tīram vittakoṣaḥ asti.",
        "english": "The riverbank is a treasury (equivocation on bank).",
        "verb_root": "√as",
        "is_valid": False,
        "form_type": "equivocation_fallacy",
        "expected_fol": "bank(river) ≠ bank(finance)"
    },
]

# ── 6. INFERENCE ─────────────────────────────────────────────────
LABELS = {0: "VALID", 1: "INVALID", 2: "AMBIGUOUS", 3: "UNKNOWN"}

def predict(model, text: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    ).to(DEVICE)
    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
    logits = out["task_logits"][0]
    probs  = logits.softmax(-1)
    idx    = probs.argmax().item()
    return LABELS[idx], probs.max().item(), logits.tolist()

def real_logic_hash(case, pred, logits, dhatu):
    state = json.dumps({
        "sanskrit":   case["sanskrit"],
        "verb_root":  case["verb_root"],
        "dhatu_pred": dhatu.get("logic_predicate", "NONE"),
        "prediction": pred,
        "logits":     [round(x, 6) for x in logits],
        "fol":        case["expected_fol"],
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(state.encode()).hexdigest()[:16]

# ── 7. RUN ───────────────────────────────────────────────────────
print("\n" + "="*90)
print("RESULTS")
print("="*90)

rows = []
full_correct = ablated_correct = dhatu_hits = 0

for case in test_cases:
    expected = "VALID" if case["is_valid"] else "INVALID"

    fp, fc, fl = predict(full_model,    case["sanskrit"])
    ap, ac, al = predict(ablated_model, case["sanskrit"])

    dhatu = lookup_dhatu(case["verb_root"])
    if dhatu["found"]:
        dhatu_hits += 1

    lhash = real_logic_hash(case, fp, fl, dhatu)

    f_right = fp == expected
    a_right = ap == expected
    if f_right: full_correct    += 1
    if a_right: ablated_correct += 1

    winner = ("BOTH"      if f_right and a_right else
              "FULL ★"    if f_right              else
              "ABLATED ★" if a_right              else "NEITHER")

    rows.append([
        case["id"],
        case["form_type"],
        expected,
        f"{fp}({fc:.2f})",
        f"{ap}({ac:.2f})",
        dhatu.get("logic_predicate", "?"),
        lhash,
        winner,
    ])

headers = ["ID", "Form", "Expected", "Full+Pāṇ",
           "Ablated", "Dhātu", "Hash", "Winner"]
print(tabulate(rows, headers=headers, tablefmt="pipe"))

n = len(test_cases)
print(f"\n{'='*90}")
print(f"  Full Brahman  (Pāṇini ON) : {full_correct}/{n} = {full_correct/n:.1%}")
print(f"  Ablated Model (Pāṇini OFF): {ablated_correct}/{n} = {ablated_correct/n:.1%}")
print(f"  Delta                     : {(full_correct-ablated_correct)/n:+.1%}")
print(f"  Dhātu DB hits             : {dhatu_hits}/{n}")

if dhatu_hits == 0:
    print("\n  ⚠ Dhātu DB hits = 0. Run dhatu_db.py first to populate the database.")

print(f"\n{'='*90}")
if full_correct > ablated_correct:
    print("  VERDICT: ✓ HYPOTHESIS SUPPORTED")
elif full_correct == ablated_correct:
    print("  VERDICT: ~ INCONCLUSIVE")
else:
    print("  VERDICT: ✗ ABLATED WINS")
print(f"{'='*90}")

# ── 8. SAVE REPORT ───────────────────────────────────────────────
out_dir = REPO_ROOT / "evaluation/results"
out_dir.mkdir(parents=True, exist_ok=True)
report = {
    "device": str(DEVICE),
    "scores": {
        "full":       f"{full_correct}/{n}",
        "ablated":    f"{ablated_correct}/{n}",
        "delta":      f"{(full_correct-ablated_correct)/n:+.1%}",
        "dhatu_hits": f"{dhatu_hits}/{n}",
    },
    "cases": [
        {
            "id":       c["id"],
            "sanskrit": c["sanskrit"],
            "english":  c["english"],
            "form":     c["form_type"],
            "expected": "VALID" if c["is_valid"] else "INVALID",
            "fol":      c["expected_fol"],
        }
        for c in test_cases
    ]
}
out_path = out_dir / "real_benchmark.json"
# Fix 3: ensure_ascii and indent are json.dump() args, not open() args
with open(out_path, "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n✓ Report saved to {out_path}")
