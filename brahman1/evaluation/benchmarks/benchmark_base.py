"""
Base class for all Brahman-1 benchmarks.
All 5 benchmarks inherit from this.

Every benchmark produces a BenchmarkResult that feeds into
evaluation/compare.py for the final comparison table.
"""

import json, time, sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any
from pathlib import Path
import torch
import numpy as np
from scipy import stats

sys.path.insert(0, ".")

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BenchmarkExample:
    id:       str
    input:    Any
    expected: Any
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    benchmark_name:  str
    model_name:      str
    accuracy:        float
    accuracy_ci:     Tuple[float, float]   # 95% confidence interval
    n_examples:      int
    n_correct:       int
    per_type_accuracy: Dict[str, float]
    latency_ms:      float                 # mean inference latency
    examples:        List[Dict] = field(default_factory=list)
    notes:           str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["accuracy_pct"] = f"{self.accuracy * 100:.2f}%"
        return d

    def summary(self) -> str:
        lo, hi = self.accuracy_ci
        return (
            f"{self.benchmark_name:30s} | {self.model_name:30s} | "
            f"acc={self.accuracy*100:5.1f}% [{lo*100:.1f},{hi*100:.1f}] | "
            f"n={self.n_examples} | {self.latency_ms:.1f}ms/ex"
        )


def wilson_confidence_interval(
    n_correct: int, n_total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


class BaseBenchmark(ABC):
    """All benchmarks implement this interface."""

    def __init__(self, name: str, n_runs: int = 3):
        self.name   = name
        self.n_runs = n_runs   # For statistical significance: run 3x, report mean ± CI

    @abstractmethod
    def build_examples(self) -> List[BenchmarkExample]:
        """Return the full list of test examples."""
        ...

    @abstractmethod
    def evaluate_single(
        self,
        model,
        tokenizer,
        example: BenchmarkExample,
        device: torch.device,
    ) -> Tuple[bool, Dict]:
        """
        Evaluate one example.
        Returns: (is_correct, metadata_dict)
        """
        ...

    def run(
        self,
        model,
        tokenizer,
        model_name: str,
        device: torch.device,
    ) -> BenchmarkResult:
        examples   = self.build_examples()
        all_correct = []
        per_type    = {}
        latencies   = []
        detail_log  = []

        model.eval()
        with torch.no_grad():
            for ex in examples:
                t0 = time.perf_counter()
                correct, meta = self.evaluate_single(model, tokenizer, ex, device)
                latencies.append((time.perf_counter() - t0) * 1000)

                all_correct.append(correct)
                etype = ex.metadata.get("type", "general")
                per_type.setdefault(etype, []).append(correct)
                detail_log.append({
                    "id": ex.id, "correct": correct,
                    "expected": str(ex.expected)[:100], **meta
                })

        n_correct = sum(all_correct)
        n_total   = len(all_correct)
        accuracy  = n_correct / max(n_total, 1)
        ci        = wilson_confidence_interval(n_correct, n_total)
        per_type_acc = {k: sum(v)/len(v) for k, v in per_type.items()}

        result = BenchmarkResult(
            benchmark_name  = self.name,
            model_name      = model_name,
            accuracy        = accuracy,
            accuracy_ci     = ci,
            n_examples      = n_total,
            n_correct       = n_correct,
            per_type_accuracy = per_type_acc,
            latency_ms      = float(np.mean(latencies)),
            examples        = detail_log[:20],  # Store first 20 for inspection
        )

        # Save to disk
        out = RESULTS_DIR / f"{self.name}_{model_name}.json"
        out.write_text(json.dumps(result.to_dict(), indent=2))
        return result
