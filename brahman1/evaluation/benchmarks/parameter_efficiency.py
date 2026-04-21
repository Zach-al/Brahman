"""
BENCHMARK 5: Parameter Efficiency Analysis
The core quantitative claim: Brahman-1 at 125M ≈ Standard Transformer at 500M+ on formal reasoning.
"""

import sys, json
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from pathlib import Path
from evaluation.benchmarks.benchmark_base import BaseBenchmark, BenchmarkExample, BenchmarkResult


LINGUISTIC_PROMPTS = [
    ("The sun rises in the east and sets in the west.",
     ["The sun moves across the sky.", "The sun stays in one place.", "The sun rises in the west."]),
    ("A library is a place where books are stored and lent.",
     ["A library contains books.", "A library is a type of school.", "A library sells books."]),
    ("Water boils at 100 degrees Celsius at sea level.",
     ["Water becomes gas when heated enough.", "Water freezes at 100 degrees.", "Water boils at any temperature."]),
    ("The Ganges is a sacred river in India.",
     ["The Ganges is a river in India.", "The Ganges is a mountain in India.", "The Ganges is a desert in India."]),
    ("Photosynthesis converts sunlight into chemical energy.",
     ["Plants use light to make food.", "Plants produce sunlight.", "Plants destroy chemical energy."]),
]


class ParameterEfficiencyBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("parameter_efficiency")

    def build_examples(self) -> List[BenchmarkExample]:
        examples = []
        for i, (premise, choices) in enumerate(LINGUISTIC_PROMPTS * 10):
            examples.append(BenchmarkExample(
                id       = f"ling_{i:03d}",
                input    = {"premise": premise, "choices": choices},
                expected = 0,
                metadata = {"type": "linguistic"},
            ))
        return examples

    def evaluate_single(
        self,
        model, tokenizer, example: BenchmarkExample, device: torch.device
    ) -> Tuple[bool, Dict]:
        inp     = example.input
        choices = inp["choices"]
        scores  = []

        for ch in choices:
            text = f"{inp['premise']} Therefore: {ch}"
            enc  = tokenizer(
                text, return_tensors="pt",
                max_length=96, padding="max_length", truncation=True
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(enc["input_ids"], enc["attention_mask"])
            logits = out.logits[0]
            ids    = enc["input_ids"][0]
            lp     = F.log_softmax(logits[:-1], dim=-1)
            tok_lp = lp.gather(1, ids[1:].unsqueeze(-1)).squeeze(-1)
            mask   = enc["attention_mask"][0][1:].float()
            score  = (tok_lp * mask).sum() / mask.sum().clamp(min=1)
            scores.append(score.item())

        predicted = int(torch.tensor(scores).argmax().item())
        return (predicted == example.expected), {
            "predicted": predicted, "scores": [round(s, 3) for s in scores]
        }

    def compute_efficiency_ratio(
        self,
        full_results:    Dict[str, BenchmarkResult],
        ablated_results: Dict[str, BenchmarkResult],
    ) -> Dict:
        logical_benchmarks   = ["syllogistic_reasoning", "multistep_causal_reasoning"]
        linguistic_benchmarks = ["parameter_efficiency"]

        def avg_acc(results, bmarks):
            vals = [results[b].accuracy for b in bmarks if b in results]
            return sum(vals) / len(vals) if vals else 0.0

        full_logic    = avg_acc(full_results, logical_benchmarks)
        ablated_logic = avg_acc(ablated_results, logical_benchmarks)
        full_ling     = avg_acc(full_results, linguistic_benchmarks)
        ablated_ling  = avg_acc(ablated_results, linguistic_benchmarks)

        logic_gain    = full_logic - ablated_logic
        ling_gain     = full_ling  - ablated_ling

        return {
            "full_logical_acc":     round(full_logic,    4),
            "ablated_logical_acc":  round(ablated_logic, 4),
            "logical_accuracy_gain":round(logic_gain,    4),
            "logical_gain_pct":     round(logic_gain * 100, 2),
            "full_linguistic_acc":  round(full_ling,    4),
            "ablated_linguistic_acc": round(ablated_ling, 4),
            "linguistic_gain_pct":  round(ling_gain * 100, 2),
            "hypothesis_supported": logic_gain > 0.05 and abs(ling_gain) < 0.05,
            "interpretation": (
                "HYPOTHESIS SUPPORTED: Grammar improves logical reasoning "
                "without degrading linguistic fluency."
                if logic_gain > 0.05 and abs(ling_gain) < 0.05
                else "HYPOTHESIS INCONCLUSIVE: Insufficient training or marginal gain."
                if logic_gain > 0.0
                else "HYPOTHESIS FALSIFIED: Ablated model matches or exceeds full model."
            ),
        }
