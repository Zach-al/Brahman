import json
import os
import random
from pathlib import Path

# Vocabularies
AGENTS = [
    "Ram", "Sita", "the king", "the teacher", "the scholar", "the warrior", 
    "the farmer", "the doctor", "the monk", "Arjuna", "Krishna", "the sage", "the child"
]
PATIENTS = [
    "the book", "the sword", "the medicine", "the grain", "the water", 
    "the treasure", "the arrow", "the law", "the gift", "the knowledge"
]
PROPERTIES = [
    "mortal", "wise", "strong", "rational", "finite", "peaceful", 
    "conscious", "ancient", "eternal", "skilled"
]
CAUSAL_EVENTS = [
    "fire spreads", "the river rises", "crops fail", 
    "the bridge collapses", "disease spreads", "the army retreats", 
    "the harvest fails", "the dam breaks"
]

def generate_modus_ponens():
    agent = random.choice(AGENTS)
    props = random.sample(PROPERTIES, 2)
    p, q = props[0], props[1]
    
    return {
        "form_type": "modus_ponens",
        "premises": [
            f"If {agent} is {p}, then {agent} is {q}.",
            f"{agent} is {p}."
        ],
        "question": f"Is {agent} {q}?",
        "correct_answer": "Yes",
        "is_valid": True,
        "explanation": f"Valid Modus Ponens deduction: Given P implies Q, and P is true, Q must be true.",
        "difficulty": 1
    }

def generate_modus_tollens():
    agent = random.choice(AGENTS)
    props = random.sample(PROPERTIES, 2)
    p, q = props[0], props[1]
    
    return {
        "form_type": "modus_tollens",
        "premises": [
            f"If {agent} is {p}, then {agent} is {q}.",
            f"{agent} is not {q}."
        ],
        "question": f"Is {agent} {p}?",
        "correct_answer": "No",
        "is_valid": True,
        "explanation": f"Valid Modus Tollens deduction: Given P implies Q, and Q is false, P must be false.",
        "difficulty": 2
    }

def generate_hypothetical_syllogism():
    agent = random.choice(AGENTS)
    props = random.sample(PROPERTIES, 3)
    p, q, r = props[0], props[1], props[2]
    
    return {
        "form_type": "hypothetical_syllogism",
        "premises": [
            f"If {agent} is {p}, then {agent} is {q}.",
            f"If {agent} is {q}, then {agent} is {r}."
        ],
        "question": f"If {agent} is {p}, is {agent} {r}?",
        "correct_answer": "Yes",
        "is_valid": True,
        "explanation": f"Valid Hypothetical Syllogism: The chain connects P to R via Q.",
        "difficulty": 2
    }

def generate_fallacy_affirming_consequent():
    agent = random.choice(AGENTS)
    props = random.sample(PROPERTIES, 2)
    p, q = props[0], props[1]
    
    return {
        "form_type": "fallacy_affirming_consequent",
        "premises": [
            f"If {agent} is {p}, then {agent} is {q}.",
            f"{agent} is {q}."
        ],
        "question": f"Is {agent} {p}?",
        "correct_answer": "Unknown",
        "is_valid": False,
        "explanation": f"Invalid logic (Affirming the Consequent): {agent} being {q} does not prove they are {p}.",
        "difficulty": 3
    }

def generate_fallacy_denying_antecedent():
    agent = random.choice(AGENTS)
    props = random.sample(PROPERTIES, 2)
    p, q = props[0], props[1]
    
    return {
        "form_type": "fallacy_denying_antecedent",
        "premises": [
            f"If {agent} is {p}, then {agent} is {q}.",
            f"{agent} is not {p}."
        ],
        "question": f"Is {agent} {q}?",
        "correct_answer": "Unknown",
        "is_valid": False,
        "explanation": f"Invalid logic (Denying the Antecedent): {agent} not being {p} does not prove they are not {q}.",
        "difficulty": 3
    }

def generate_causal_example():
    subtypes = ["simple_chain", "causal_prevention", "upstream_prevention"]
    subtype = random.choice(subtypes)
    
    if subtype == "simple_chain":
        events = random.sample(CAUSAL_EVENTS, 3)
        e1, e2, e3 = events[0], events[1], events[2]
        return {
            "form_type": "causal_simple_chain",
            "premises": [f"If {e1}, then {e2}.", f"If {e2}, then {e3}.", f"{e1} happens."],
            "question": f"Does {e3} happen?",
            "correct_answer": "Yes",
            "is_valid": True,
            "explanation": "Valid causal chain sequence.",
            "difficulty": 2
        }
        
    elif subtype == "causal_prevention":
        events = random.sample(CAUSAL_EVENTS, 2)
        agent = random.choice(AGENTS)
        e1, e2 = events[0], events[1]
        return {
            "form_type": "causal_prevention",
            "premises": [f"If {e1}, then {e2}.", f"{agent} prevents {e1}."],
            "question": f"Do we know for sure that {e2} happens?",
            "correct_answer": "No",
            "is_valid": True,
            "explanation": "Since the cause was prevented, we cannot be certain the effect occurs.",
            "difficulty": 3
        }
        
    else: # upstream_prevention
        events = random.sample(CAUSAL_EVENTS, 3)
        agent = random.choice(AGENTS)
        e1, e2, e3 = events[0], events[1], events[2]
        return {
            "form_type": "causal_upstream_prevention",
            "premises": [f"If {e1}, then {e2}.", f"If {e2}, then {e3}.", f"{agent} prevents {e1}."],
            "question": f"Does {e1} cause {e3} in this scenario?",
            "correct_answer": "No",
            "is_valid": True,
            "explanation": "The causal chain is broken at the start.",
            "difficulty": 3
        }

def generate_full_dataset():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training_data.jsonl"
    
    with open(out_path, "w", encoding="utf-8") as f:
        # 15000 modus_ponens
        for _ in range(15000):
            f.write(json.dumps(generate_modus_ponens()) + "\n")
            
        # 8000 modus_tollens
        for _ in range(8000):
            f.write(json.dumps(generate_modus_tollens()) + "\n")
            
        # 8000 hypothetical_syllogism
        for _ in range(8000):
            f.write(json.dumps(generate_hypothetical_syllogism()) + "\n")
            
        # 7000 fallacy_affirming_consequent
        for _ in range(7000):
            f.write(json.dumps(generate_fallacy_affirming_consequent()) + "\n")
            
        # 7000 fallacy_denying_antecedent
        for _ in range(7000):
            f.write(json.dumps(generate_fallacy_denying_antecedent()) + "\n")
            
        # 10000 causal examples
        for _ in range(10000):
            f.write(json.dumps(generate_causal_example()) + "\n")
            
    return str(out_path)

if __name__ == "__main__":
    path = generate_full_dataset()
    # Count lines in output file and print total
    with open(path) as f:
        count = sum(1 for line in f)
    print(f"VERIFICATION: {count} total examples written to {path}")
    # Print 1 sample from each form_type to verify structure
    import json
    seen = set()
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            if ex['form_type'] not in seen:
                seen.add(ex['form_type'])
                print(f"SAMPLE [{ex['form_type']}]: {ex['premises'][0][:80]}")
