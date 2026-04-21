"""
BENCHMARK 1: Syllogistic Reasoning
500 novel syllogisms — modus ponens, modus tollens, hypothetical, fallacies.

Task: Given premises P1, P2 → predict if conclusion C follows (binary).
This tests whether Pāṇini-biased attention improves formal inference.

Metric: Accuracy on 500 held-out examples (not in training data).
Statistical test: McNemar's test (Brahman-1 full vs ablated).
"""

import sys, random
sys.path.insert(0, ".")

import torch
from typing import List, Tuple, Dict
from evaluation.benchmarks.benchmark_base import BaseBenchmark, BenchmarkExample

# Use different entity sets from training to avoid memorization
EVAL_ENTITIES = [
    ("doctor",    "knowledgeable", "Hippocrates"),
    ("mountain",  "tall",          "Everest"),
    ("metal",     "conductive",    "copper"),
    ("virus",     "microscopic",   "Influenza"),
    ("ocean",     "deep",          "Pacific"),
    ("forest",    "green",         "Amazon"),
    ("diamond",   "hard",          "Kohinoor"),
    ("river",     "flowing",       "Nile"),
    ("volcano",   "eruptible",     "Vesuvius"),
    ("desert",    "arid",          "Sahara"),
    ("philosopher","wise",          "Aristotle"),
    ("emperor",   "powerful",      "Chandragupta"),
]


class SyllogisticReasoningBenchmark(BaseBenchmark):
    def __init__(self, n_examples: int = 500):
        super().__init__("syllogistic_reasoning")
        self.n_examples = n_examples

    def build_examples(self) -> List[BenchmarkExample]:
        random.seed(42)  # Reproducible
        examples = []
        idx = 0

        templates = [
            # (type, premise_template, conclusion_template, label)
            ("modus_ponens",
             "All {A}s are {B}. {x} is a {A}.",
             "{x} is {B}.", 1),

            ("modus_tollens",
             "All {A}s are {B}. {y} is not {B}.",
             "{y} is not a {A}.", 1),

            ("hypothetical_syllogism",
             "All {A}s are {B}. All {B}s are {C}.",
             "All {A}s are {C}.", 1),

            ("disjunctive_syllogism",
             "Either {x} is {B} or {x} is {C}. {x} is not {B}.",
             "{x} is {C}.", 1),

            ("fallacy_affirming_consequent",
             "All {A}s are {B}. {y} is {B}.",
             "{y} is a {A}.", 0),   # INVALID

            ("fallacy_denying_antecedent",
             "All {A}s are {B}. {y} is not a {A}.",
             "{y} is not {B}.", 0), # INVALID

            ("universal_instantiation",
             "All {A}s are {B}. All {B}s are {C}. {x} is a {A}.",
             "{x} is {C}.", 1),

            ("contradiction_detection",
             "All {A}s are {B}. No {B} is {C}.",
             "No {A} is {C}.", 1),
        ]

        while len(examples) < self.n_examples:
            tmpl_type, p_tmpl, c_tmpl, label = random.choice(templates)
            cls_a, cls_b, inst_x = random.choice(EVAL_ENTITIES)
            cls_c = random.choice([e[1] for e in EVAL_ENTITIES if e[1] != cls_b])
            inst_y = random.choice([e[2] for e in EVAL_ENTITIES if e[2] != inst_x])

            sub = {"A": cls_a, "B": cls_b, "C": cls_c, "x": inst_x, "y": inst_y}
            try:
                premises   = p_tmpl.format(**sub)
                conclusion = c_tmpl.format(**sub)
            except KeyError:
                continue

            examples.append(BenchmarkExample(
                id       = f"syl_{idx:04d}",
                input    = {"premises": premises, "conclusion": conclusion},
                expected = label,
                metadata = {"type": tmpl_type},
            ))
            idx += 1

        return examples

    def evaluate_single(
        self,
        model, tokenizer, example: BenchmarkExample, device: torch.device
    ) -> Tuple[bool, Dict]:
        inp = example.input
        text = f"{inp['premises']} Does it follow that: {inp['conclusion']}"

        enc = tokenizer(
            text, return_tensors="pt",
            max_length=128, padding="max_length", truncation=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out     = model(enc["input_ids"], enc["attention_mask"])
        alpha   = out.alpha[0].item()
        logits  = out.logits[0, 0, :]          # CLS-position logits

        # Use "yes"/"no" token logits as proxy for entailment classification
        yes_ids = tokenizer.convert_tokens_to_ids(["Yes", "yes", "Ġyes", "ĠYes"])
        no_ids  = tokenizer.convert_tokens_to_ids(["No",  "no",  "Ġno",  "ĠNo"])
        yes_ids = [i for i in yes_ids if i != tokenizer.unk_token_id]
        no_ids  = [i for i in no_ids  if i != tokenizer.unk_token_id]

        yes_score = logits[yes_ids].max().item() if yes_ids else 0.0
        no_score  = logits[no_ids].max().item()  if no_ids  else 0.0

        predicted = 1 if yes_score > no_score else 0
        correct   = (predicted == example.expected)

        return correct, {
            "predicted": predicted,
            "yes_score": round(yes_score, 3),
            "no_score":  round(no_score, 3),
            "alpha":     round(alpha, 3),
            "type":      example.metadata["type"],
        }
