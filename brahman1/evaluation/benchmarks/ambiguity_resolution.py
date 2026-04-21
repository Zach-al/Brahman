"""
BENCHMARK 2: Structural Ambiguity Resolution
Tests whether the SIR produces DISTINCT representations for
sentences with known structural ambiguity.

Metric: Are the two SIR case assignments structurally distinct?
(Measured by cosine distance in grammatical encoding space.)
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from evaluation.benchmarks.benchmark_base import BaseBenchmark, BenchmarkExample

AMBIGUOUS_SENTENCES = [
    {
        "sentence":   "Visiting relatives can be boring.",
        "reading_a":  "The act of visiting relatives is boring.",
        "reading_b":  "Relatives who visit can be boring.",
        "preferred":  "b",
        "type":       "gerund_ambiguity",
    },
    {
        "sentence":   "The chicken is ready to eat.",
        "reading_a":  "The chicken is ready to be eaten.",
        "reading_b":  "The chicken is ready to eat something.",
        "preferred":  "a",
        "type":       "infinitive_ambiguity",
    },
    {
        "sentence":   "I saw the man with the telescope.",
        "reading_a":  "I used a telescope to see the man.",
        "reading_b":  "I saw the man who had a telescope.",
        "preferred":  "b",
        "type":       "pp_attachment",
    },
    {
        "sentence":   "The professor approved the student's thesis.",
        "reading_a":  "The professor gave approval to the thesis.",
        "reading_b":  "The professor validated the student who wrote the thesis.",
        "preferred":  "a",
        "type":       "argument_structure",
    },
    {
        "sentence":   "Flying planes can be dangerous.",
        "reading_a":  "The act of flying planes is dangerous.",
        "reading_b":  "Planes that are flying can be dangerous.",
        "preferred":  "a",
        "type":       "gerund_ambiguity",
    },
    {
        "sentence":   "She cannot bear children.",
        "reading_a":  "She is unable to give birth to children.",
        "reading_b":  "She is unable to tolerate children.",
        "preferred":  "a",
        "type":       "lexical_ambiguity",
    },
    {
        "sentence":   "The king saw the warrior with the sword.",
        "reading_a":  "The king used the sword to see the warrior.",
        "reading_b":  "The king saw the warrior who had the sword.",
        "preferred":  "b",
        "type":       "pp_attachment",
    },
    {
        "sentence":   "The teacher struck the student with the book.",
        "reading_a":  "The teacher struck the student using the book.",
        "reading_b":  "The teacher struck the student who had the book.",
        "preferred":  "a",
        "type":       "pp_attachment",
    },
    {
        "sentence":   "Every student loves a teacher.",
        "reading_a":  "There exists one teacher who is loved by all students.",
        "reading_b":  "For each student, there exists some teacher they love.",
        "preferred":  "b",
        "type":       "quantifier_scope",
    },
    {
        "sentence":   "The detective saw the thief with the binoculars.",
        "reading_a":  "The detective used binoculars to see the thief.",
        "reading_b":  "The detective saw the thief who had binoculars.",
        "preferred":  "a",
        "type":       "pp_attachment",
    },
]


class AmbiguityResolutionBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("ambiguity_resolution")
        self.threshold = 0.15  # Min cosine distance to count as "distinct"

    def build_examples(self) -> List[BenchmarkExample]:
        examples = []
        for i, s in enumerate(AMBIGUOUS_SENTENCES):
            examples.append(BenchmarkExample(
                id       = f"amb_{i:03d}",
                input    = s,
                expected = s["preferred"],
                metadata = {"type": s["type"]},
            ))
        return examples

    def _get_grammatical_encoding(
        self,
        model, tokenizer, text: str, device: torch.device
    ) -> torch.Tensor:
        """Extract mean-pooled grammatical space encoding for a text."""
        enc = tokenizer(
            text, return_tensors="pt",
            max_length=64, padding="max_length", truncation=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        triple_enc = model.triple_encoder(enc["input_ids"], enc["attention_mask"])
        mask   = enc["attention_mask"].float().unsqueeze(-1)
        gram   = (triple_enc.grammatical * mask).sum(1) / mask.sum(1).clamp(min=1)
        return gram.squeeze(0)  # (256,)

    def evaluate_single(
        self,
        model, tokenizer, example: BenchmarkExample, device: torch.device
    ) -> Tuple[bool, Dict]:
        inp = example.input

        enc_a = self._get_grammatical_encoding(model, tokenizer, inp["reading_a"], device)
        enc_b = self._get_grammatical_encoding(model, tokenizer, inp["reading_b"], device)
        enc_orig = self._get_grammatical_encoding(model, tokenizer, inp["sentence"], device)

        cos_sim_ab   = F.cosine_similarity(enc_a.unsqueeze(0), enc_b.unsqueeze(0)).item()
        distance_ab  = 1.0 - cos_sim_ab
        is_distinct  = distance_ab > self.threshold

        sim_to_a = F.cosine_similarity(enc_orig.unsqueeze(0), enc_a.unsqueeze(0)).item()
        sim_to_b = F.cosine_similarity(enc_orig.unsqueeze(0), enc_b.unsqueeze(0)).item()
        predicted_reading = "a" if sim_to_a > sim_to_b else "b"
        reading_correct   = (predicted_reading == example.expected)

        correct = is_distinct and reading_correct

        return correct, {
            "distance_ab":       round(distance_ab, 4),
            "is_distinct":       is_distinct,
            "sim_to_a":          round(sim_to_a, 4),
            "sim_to_b":          round(sim_to_b, 4),
            "predicted_reading": predicted_reading,
            "preferred_reading": example.expected,
            "threshold":         self.threshold,
        }
