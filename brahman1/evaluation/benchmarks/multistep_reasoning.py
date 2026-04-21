"""
BENCHMARK 3: Multi-Step Causal Reasoning
200 novel causal chains with prevention, reversal, and branching.

Task: Given a causal chain, answer a query about the final state.
Tests whether the ablative case (apādāna = source/cause) rule P5
improves causal reasoning accuracy.
"""

import sys, random
sys.path.insert(0, ".")

import torch
from typing import List, Tuple, Dict
from evaluation.benchmarks.benchmark_base import BaseBenchmark, BenchmarkExample

random.seed(99)

CAUSAL_DOMAINS = [
    ("rain",          "flooding",       "crop damage",    "drainage system"),
    ("fire",          "smoke",          "evacuation",     "fire extinguisher"),
    ("study",         "understanding",  "exam success",   "distraction"),
    ("seed planting", "germination",    "harvest",        "drought"),
    ("desire",        "action",         "consequence",    "self-control"),
    ("infection",     "fever",          "weakness",       "medicine"),
    ("practice",      "skill",          "mastery",        "injury"),
    ("investment",    "profit",         "wealth",         "market crash"),
    ("teaching",      "knowledge",      "wisdom",         "forgetting"),
    ("conflict",      "destruction",    "suffering",      "peace treaty"),
]


class MultiStepReasoningBenchmark(BaseBenchmark):
    def __init__(self, n_examples: int = 200):
        super().__init__("multistep_causal_reasoning")
        self.n_examples = n_examples

    def build_examples(self) -> List[BenchmarkExample]:
        examples = []
        idx = 0

        templates = [
            (
                lambda a, b, c, p: f"{a} caused {b}. {b} caused {c}.",
                lambda a, b, c, p: f"Did {c} occur?",
                "yes", "simple_chain"
            ),
            (
                lambda a, b, c, p: f"{a} caused {b}. {b} caused {c}. {p} prevented {c}.",
                lambda a, b, c, p: f"Did {c} occur?",
                "no", "prevention"
            ),
            (
                lambda a, b, c, p: f"{a} caused {b}. {b} caused {c}. {p} prevented {b}.",
                lambda a, b, c, p: f"Did {c} occur?",
                "no", "upstream_prevention"
            ),
            (
                lambda a, b, c, p: f"{a} caused {b}. {b} caused {c}.",
                lambda a, b, c, p: f"Did {a} cause {c}?",
                "yes", "transitivity"
            ),
            (
                lambda a, b, c, p: f"{a} caused {b}. {b} caused {c}.",
                lambda a, b, c, p: f"Did {c} cause {a}?",
                "no", "asymmetry"
            ),
            (
                lambda a, b, c, p: f"{a} caused {b}. {p} prevented {b}.",
                lambda a, b, c, p: f"Did {b} occur?",
                "no", "direct_prevention"
            ),
            (
                lambda a, b, c, p: f"{a} caused {b}. {b} caused {c}. {p} prevented {c}.",
                lambda a, b, c, p: f"Did {b} occur?",
                "yes", "prevention_does_not_retroact"
            ),
            (
                lambda a, b, c, p: f"If {a} then {b}. If {b} then {c}. {a} is true.",
                lambda a, b, c, p: f"Is {c} true?",
                "yes", "conditional_chain"
            ),
        ]

        while len(examples) < self.n_examples:
            a, b, c, p = random.choice(CAUSAL_DOMAINS)
            scenario_fn, query_fn, answer, typ = random.choice(templates)
            scenario = scenario_fn(a, b, c, p)
            query    = query_fn(a, b, c, p)

            examples.append(BenchmarkExample(
                id       = f"causal_{idx:04d}",
                input    = {"scenario": scenario, "query": query},
                expected = answer,
                metadata = {"type": typ, "domain": a},
            ))
            idx += 1

        return examples

    def evaluate_single(
        self,
        model, tokenizer, example: BenchmarkExample, device: torch.device
    ) -> Tuple[bool, Dict]:
        inp  = example.input
        text = f"{inp['scenario']} {inp['query']}"

        enc = tokenizer(
            text, return_tensors="pt",
            max_length=128, padding="max_length", truncation=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out    = model(enc["input_ids"], enc["attention_mask"])
        logits = out.logits[0, 0, :]

        yes_ids = [i for i in tokenizer.convert_tokens_to_ids(
            ["Yes", "yes", "Ġyes", "ĠYes", "TRUE", "true"]
        ) if i != tokenizer.unk_token_id]
        no_ids  = [i for i in tokenizer.convert_tokens_to_ids(
            ["No",  "no",  "Ġno",  "ĠNo", "FALSE", "false"]
        ) if i != tokenizer.unk_token_id]

        yes_score = logits[yes_ids].max().item() if yes_ids else 0.0
        no_score  = logits[no_ids].max().item()  if no_ids  else 0.0
        predicted = "yes" if yes_score > no_score else "no"
        correct   = (predicted == example.expected)

        return correct, {
            "predicted": predicted,
            "yes_score": round(yes_score, 3),
            "no_score":  round(no_score, 3),
            "alpha":     round(out.alpha[0].item(), 3),
            "scenario":  inp["scenario"][:80],
        }
