"""Week 4 Integration Test — verifies all benchmarks before full evaluation."""

import sys, torch
sys.path.insert(0, ".")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("=" * 60)
print("BRAHMAN-1 Week 4 Integration Test")
print(f"Device: {DEVICE}")
print("=" * 60)

from transformers import RobertaTokenizerFast
from core.model.brahman_transformer import BrahmanTransformer

tok   = RobertaTokenizerFast.from_pretrained("roberta-base")
model = BrahmanTransformer(num_layers=2).to(DEVICE)
model.eval()

# [1/6] Benchmark base
print("\n[1/6] Testing BenchmarkBase (Wilson CI)...")
from evaluation.benchmarks.benchmark_base import wilson_confidence_interval
lo, hi = wilson_confidence_interval(80, 100)
assert 0.70 < lo < 0.80
assert 0.85 < hi < 0.95
print(f"  ✓ Wilson CI for 80/100: [{lo:.3f}, {hi:.3f}]")

# [2/6] Syllogistic Reasoning
print("\n[2/6] Testing SyllogisticReasoningBenchmark (10 examples)...")
from evaluation.benchmarks.logical_puzzles import SyllogisticReasoningBenchmark
bm1 = SyllogisticReasoningBenchmark(n_examples=10)
exs = bm1.build_examples()
assert len(exs) == 10
correct, meta = bm1.evaluate_single(model, tok, exs[0], DEVICE)
assert "predicted" in meta
print(f"  ✓ Built 10 examples | Example 0 correct={correct} alpha={meta.get('alpha')}")

# [3/6] Ambiguity Resolution
print("\n[3/6] Testing AmbiguityResolutionBenchmark...")
from evaluation.benchmarks.ambiguity_resolution import AmbiguityResolutionBenchmark
bm2 = AmbiguityResolutionBenchmark()
exs2 = bm2.build_examples()
assert len(exs2) == 10
correct2, meta2 = bm2.evaluate_single(model, tok, exs2[0], DEVICE)
assert "distance_ab" in meta2
assert "is_distinct" in meta2
print(f"  ✓ {len(exs2)} ambiguous sentences | distance_ab={meta2['distance_ab']}")

# [4/6] Causal Chain
print("\n[4/6] Testing MultiStepReasoningBenchmark (10 examples)...")
from evaluation.benchmarks.multistep_reasoning import MultiStepReasoningBenchmark
bm3 = MultiStepReasoningBenchmark(n_examples=10)
exs3 = bm3.build_examples()
correct3, meta3 = bm3.evaluate_single(model, tok, exs3[0], DEVICE)
print(f"  ✓ 10 causal examples | predicted={meta3['predicted']} alpha={meta3['alpha']}")

# [5/6] Compositionality
print("\n[5/6] Testing CompositionalityBenchmark (10 examples)...")
from evaluation.benchmarks.compositionality import CompositionalityBenchmark
bm4 = CompositionalityBenchmark(n_examples=10)
exs4 = bm4.build_examples()
correct4, meta4 = bm4.evaluate_single(model, tok, exs4[0], DEVICE)
assert "predicted_meaning" in meta4
print(f"  ✓ Compound: '{exs4[0].input['compound']}' → '{meta4['predicted_meaning'][:40]}'")

# [6/6] Full compare.py (fast mode, untrained model)
print("\n[6/6] Testing compare.py end-to-end (fast, untrained weights)...")
from evaluation.compare import run_all_benchmarks, mcnemar_test, generate_report, ParameterEfficiencyBenchmark

model2 = BrahmanTransformer(num_layers=2, ablate_panini=True).to(DEVICE)
model2.eval()

full_res    = run_all_benchmarks(model, "test_full",    tok, fast=True)
ablated_res = run_all_benchmarks(model2, "test_ablated", tok, fast=True)

mc = mcnemar_test([True, False, True, True], [False, True, True, False])
assert "p_value" in mc
print(f"  ✓ McNemar test: χ²={mc['statistic']} p={mc['p_value']}")

eff = ParameterEfficiencyBenchmark().compute_efficiency_ratio(full_res, ablated_res)
assert "interpretation" in eff
print(f"  ✓ Efficiency: {eff['interpretation'][:60]}...")

report = generate_report(full_res, ablated_res, eff, {})
assert "# Brahman-1" in report
assert "Hypothesis" in report
print(f"  ✓ Report generated: {len(report):,} chars")

print("\n" + "=" * 60)
print("✓ ALL WEEK 4 TESTS PASSED")
print("=" * 60)
