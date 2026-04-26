import json
import random
from pathlib import Path
import collections

AGENTS = ["Ram", "Sita", "the king", "the teacher", "the scholar", "the warrior", "the farmer", "the doctor", "the monk", "Arjuna", "Krishna"]
PROPERTIES = ["mortal", "wise", "strong", "rational", "finite", "peaceful", "conscious", "ancient", "eternal", "skilled"]
CAUSAL_EVENTS = ["fire spreads", "the river rises", "crops fail", "the bridge collapses", "disease spreads", "the army retreats"]

def get_a_b():
    props = random.sample(PROPERTIES, 2)
    return props[0], props[1]

def generate_modus_ponens():
    A, B = get_a_b()
    X = random.choice(AGENTS)
    return {
        "form_type": "modus_ponens",
        "premises": [f"All who are {A} are {B}.", f"{X} is {A}."],
        "question": f"Is {X} {B}?",
        "correct_answer": "Yes",
        "is_valid": True,
        "explanation": f"By Modus Ponens: All {A} are {B}, and {X} is {A}, so {X} must be {B}.",
        "difficulty": 1
    }

def generate_modus_tollens():
    A, B = get_a_b()
    X = random.choice(AGENTS)
    return {
        "form_type": "modus_tollens",
        "premises": [f"All who are {A} are {B}.", f"{X} is not {B}."],
        "question": f"Is {X} {A}?",
        "correct_answer": "No",
        "is_valid": True,
        "explanation": f"By Modus Tollens: All {A} are {B}. Since {X} is not {B}, {X} cannot be {A}.",
        "difficulty": 2
    }

def generate_hypothetical_syllogism():
    props = random.sample(PROPERTIES, 3)
    A, B, C = props[0], props[1], props[2]
    X = random.choice(AGENTS)
    return {
        "form_type": "hypothetical_syllogism",
        "premises": [f"If {X} is {A}, then {X} is {B}.", f"If {X} is {B}, then {X} is {C}."],
        "question": f"If {X} is {A}, is {X} {C}?",
        "correct_answer": "Yes",
        "is_valid": True,
        "explanation": f"By Hypothetical Syllogism: The condition {A} leads to {B}, which leads to {C}.",
        "difficulty": 2
    }

def generate_fallacy_affirming_consequent():
    A, B = get_a_b()
    X = random.choice(AGENTS)
    return {
        "form_type": "fallacy_affirming_consequent",
        "premises": [f"All who are {A} are {B}.", f"{X} is {B}."],
        "question": f"Is {X} definitely {A}?",
        "correct_answer": "No",
        "is_valid": False,
        "explanation": f"Affirming the consequent fallacy: Being {B} does not guarantee being {A}, as other things might also be {B}.",
        "difficulty": 3
    }

def generate_fallacy_denying_antecedent():
    A, B = get_a_b()
    X = random.choice(AGENTS)
    return {
        "form_type": "fallacy_denying_antecedent",
        "premises": [f"All who are {A} are {B}.", f"{X} is not {A}."],
        "question": f"Is {X} definitely not {B}?",
        "correct_answer": "No",
        "is_valid": False,
        "explanation": f"Denying the antecedent fallacy: Not being {A} does not mean {X} cannot be {B} through some other category.",
        "difficulty": 3
    }

def generate_causal_example(subtype):
    if subtype == "simple_chain":
        events = random.sample(CAUSAL_EVENTS, 3)
        e1, e2, e3 = events[0], events[1], events[2]
        return {
            "form_type": "simple_chain",
            "premises": [f"If {e1}, then {e2}.", f"If {e2}, then {e3}.", f"{e1} happens."],
            "question": f"Does {e3} happen?",
            "correct_answer": "Yes",
            "is_valid": True,
            "explanation": f"Valid causal chain: {e1} triggers {e2}, which triggers {e3}.",
            "difficulty": 2
        }
    elif subtype == "causal_prevention":
        events = random.sample(CAUSAL_EVENTS, 2)
        e1, e2 = events[0], events[1]
        agent = random.choice(AGENTS)
        return {
            "form_type": "causal_prevention",
            "premises": [f"If {e1}, then {e2}.", f"{agent} prevents {e1}."],
            "question": f"Does {e2} happen as a result of {e1}?",
            "correct_answer": "No",
            "is_valid": True,
            "explanation": f"Causal prevention: Since {e1} was prevented, it cannot cause {e2}.",
            "difficulty": 2
        }
    else: # upstream_prevention
        events = random.sample(CAUSAL_EVENTS, 3)
        e1, e2, e3 = events[0], events[1], events[2]
        agent = random.choice(AGENTS)
        return {
            "form_type": "upstream_prevention",
            "premises": [f"If {e1}, then {e2}.", f"If {e2}, then {e3}.", f"{agent} prevents {e1}."],
            "question": f"Does {e3} happen as a result of this chain?",
            "correct_answer": "No",
            "is_valid": True,
            "explanation": f"Upstream prevention: Breaking the chain at {e1} prevents the subsequent effect {e3}.",
            "difficulty": 3
        }

# ── DECEPTIVE FALLACY GENERATORS ─────────────────────────────────
# These are the hardest patterns — they LOOK valid but are logically broken.

CATEGORIES = ["mortal", "wise", "strong", "rational", "peaceful", "conscious", "ancient", "skilled", "brave", "learned"]

def generate_undistributed_middle():
    """All A are B. All C are B. Therefore all A are C. (INVALID)"""
    cats = random.sample(CATEGORIES, 3)
    A, B, C = cats[0], cats[1], cats[2]
    return {
        "form_type": "undistributed_middle",
        "premises": [f"All who are {A} are {B}.", f"All who are {C} are {B}."],
        "question": f"Are all who are {A} definitely {C}?",
        "correct_answer": "No",
        "is_valid": False,
        "explanation": f"Undistributed Middle fallacy: Both {A} and {C} share property {B}, but this does not make them identical categories.",
        "difficulty": 4
    }

def generate_illicit_major():
    """All A are B. No C are A. Therefore no C are B. (INVALID)"""
    cats = random.sample(CATEGORIES, 3)
    A, B, C = cats[0], cats[1], cats[2]
    return {
        "form_type": "illicit_major",
        "premises": [f"All who are {A} are {B}.", f"No one who is {C} is {A}."],
        "question": f"Is it true that no one who is {C} is {B}?",
        "correct_answer": "No",
        "is_valid": False,
        "explanation": f"Illicit Major fallacy: {C} beings might still be {B} through a category other than {A}.",
        "difficulty": 4
    }

def generate_circular_reasoning():
    """X is true because Y. Y is true because X. (INVALID)"""
    cats = random.sample(CATEGORIES, 2)
    A, B = cats[0], cats[1]
    X = random.choice(AGENTS)
    return {
        "form_type": "circular_reasoning",
        "premises": [f"{X} is {A} because {X} is {B}.", f"{X} is {B} because {X} is {A}."],
        "question": f"Is {X} proven to be {A}?",
        "correct_answer": "No",
        "is_valid": False,
        "explanation": f"Circular Reasoning: The argument for {A} depends on {B}, which itself depends on {A}. No independent evidence.",
        "difficulty": 5
    }

def generate_full_dataset():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_data.jsonl"
    
    with open(out_path, "w", encoding="utf-8") as f:
        # ── VALID patterns (35,334 total) ────────────────────────
        for _ in range(15000):
            f.write(json.dumps(generate_modus_ponens()) + "\n")
        for _ in range(8000):
            f.write(json.dumps(generate_modus_tollens()) + "\n")
        for _ in range(8000):
            f.write(json.dumps(generate_hypothetical_syllogism()) + "\n")
        causal_types = ["simple_chain", "causal_prevention", "upstream_prevention"]
        counts = [3334, 3333, 3333] # Total 10000 (but prevention is VALID in our framing)
        for i, count in enumerate(counts):
            for _ in range(count):
                f.write(json.dumps(generate_causal_example(causal_types[i])) + "\n")

        # ── INVALID patterns (24,000 total — balanced against VALID) ──
        for _ in range(7000):
            f.write(json.dumps(generate_fallacy_affirming_consequent()) + "\n")
        for _ in range(7000):
            f.write(json.dumps(generate_fallacy_denying_antecedent()) + "\n")

        # ── NEW: Deceptive fallacies (10,000 total) ──────────────
        for _ in range(3334):
            f.write(json.dumps(generate_undistributed_middle()) + "\n")
        for _ in range(3333):
            f.write(json.dumps(generate_illicit_major()) + "\n")
        for _ in range(3333):
            f.write(json.dumps(generate_circular_reasoning()) + "\n")
                
    return str(out_path)

if __name__ == "__main__":
    path = generate_full_dataset()
    with open(path) as f:
        count = sum(1 for line in f)
    print(f"VERIFICATION: {count} total examples written to {path}")
    import json, collections
    seen = collections.Counter()
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            seen[ex['form_type']] += 1
    for k, v in sorted(seen.items()):
        print(f"  {k}: {v}")
