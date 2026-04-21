"""
BENCHMARK 4: Novel Compound Understanding (Compositionality Test)
The most theoretically critical benchmark.

Tests TRUE compositionality: does the model correctly interpret
Sanskrit-style compound constructions it has NEVER seen before,
purely from morphological structure?
"""

import sys, random
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from evaluation.benchmarks.benchmark_base import BaseBenchmark, BenchmarkExample
from core.dhatu.dhatu_db import DhatuDB


class CompositionalityBenchmark(BaseBenchmark):
    def __init__(self, n_examples: int = 150):
        super().__init__("compositionality")
        self.n_examples = n_examples
        self.db = DhatuDB()

    def _generate_novel_compounds(self) -> List[Dict]:
        """
        Generate novel dhātu-based compound interpretations.
        """
        random.seed(7)
        compounds = []

        tatpurusa_templates = [
            ("knowledge-sword", "a sword of knowledge / knowledge that cuts through ignorance",
             ["a sword made of iron", "a wooden sword", "a useless weapon"],
             "cognition+destruction"),
            ("truth-fire", "fire of truth / that which burns away falsehood",
             ["a literal burning fire", "a cold fire", "a decorative lamp"],
             "existence+transformation"),
            ("mind-ocean", "the vast expanse of the mind / consciousness as deep as an ocean",
             ["an ocean with a mind", "a shallow pond", "a dried riverbed"],
             "cognition+relation"),
            ("dharma-warrior", "one who fights for righteousness",
             ["a warrior who broke the law", "a cowardly soldier", "a peaceful monk"],
             "relation+conflict"),
            ("time-river", "time flowing like a river / the stream of time",
             ["a river that measures time", "a frozen river", "an underground stream"],
             "motion+transformation"),
            ("karma-wheel", "the cycle of action and consequence",
             ["a wheel used for work", "a broken cart wheel", "a decorative wheel"],
             "action+relation"),
            ("speech-arrow", "words as sharp as arrows / cutting speech",
             ["an arrow used in speech", "a quiet person", "a blunt weapon"],
             "communication+destruction"),
            ("wisdom-lamp", "wisdom as a lamp that illuminates darkness",
             ["a lamp used by scholars", "a very bright light", "a candle in daylight"],
             "cognition+perception"),
        ]

        bahuvrihi_templates = [
            ("lotus-eyed", "one whose eyes are like lotuses / beautiful eyes",
             ["one who eats lotuses", "a person who is blind", "one who plants lotuses"],
             "perception+relation"),
            ("lion-hearted", "one whose heart is like a lion's / very courageous",
             ["one who has a heart disease", "a cowardly person", "a lion tamer"],
             "emotion+existence"),
            ("moon-faced", "one whose face is as beautiful as the moon",
             ["one who faces the moon", "a person with a round face", "an astronomer"],
             "perception+relation"),
            ("iron-willed", "one whose will is as hard as iron / very determined",
             ["one who makes iron objects", "an indecisive person", "a blacksmith"],
             "action+ability"),
        ]

        all_templates = tatpurusa_templates + bahuvrihi_templates
        random.shuffle(all_templates)

        for compound, correct, distractors, sem_class in all_templates:
            choices = [correct] + distractors[:3]
            random.shuffle(choices)
            correct_idx = choices.index(correct)

            compounds.append({
                "compound":    compound,
                "choices":     choices,
                "correct_idx": correct_idx,
                "sem_class":   sem_class,
                "type":        "tatpurusa" if "+" in sem_class and sem_class.index("+") > 0 else "bahuvrihi",
            })

        while len(compounds) < self.n_examples:
            base = random.choice(compounds[:len(all_templates)])
            compounds.append(dict(base))

        return compounds[:self.n_examples]

    def build_examples(self) -> List[BenchmarkExample]:
        compounds = self._generate_novel_compounds()
        examples  = []
        for i, c in enumerate(compounds):
            choices_text = "\n".join(
                f"  {chr(65+j)}) {ch}" for j, ch in enumerate(c["choices"])
            )
            examples.append(BenchmarkExample(
                id    = f"comp_{i:03d}",
                input = {
                    "prompt":  f"What does '{c['compound']}' most likely mean?\n{choices_text}",
                    "choices": c["choices"],
                    "compound": c["compound"],
                },
                expected = c["correct_idx"],
                metadata = {"type": c.get("type","compound"), "sem_class": c["sem_class"]},
            ))
        return examples

    def evaluate_single(
        self,
        model, tokenizer, example: BenchmarkExample, device: torch.device
    ) -> Tuple[bool, Dict]:
        inp     = example.input
        choices = inp["choices"]
        scores  = []

        for choice in choices:
            text = f"The compound '{inp['compound']}' means: {choice}"
            enc  = tokenizer(
                text, return_tensors="pt",
                max_length=96, padding="max_length", truncation=True
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(enc["input_ids"], enc["attention_mask"])
            logits = out.logits[0]
            ids    = enc["input_ids"][0]
            log_p  = F.log_softmax(logits[:-1], dim=-1)
            tok_lp = log_p.gather(1, ids[1:].unsqueeze(-1)).squeeze(-1)
            mask   = enc["attention_mask"][0][1:].float()
            score  = (tok_lp * mask).sum() / mask.sum().clamp(min=1)
            scores.append(score.item())

        predicted = int(torch.tensor(scores).argmax().item())
        correct   = (predicted == example.expected)

        return correct, {
            "predicted_idx": predicted,
            "correct_idx":   example.expected,
            "predicted_meaning": choices[predicted][:60],
            "scores":        [round(s, 3) for s in scores],
        }
