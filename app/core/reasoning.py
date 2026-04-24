"""
Anvaya-Bodha Reasoning Engine — Production Module.
Implements Ākāṅkṣā (Expectancy), Yogyatā (Consistency), and Sannidhi (Proximity).
"""
import hashlib
import json
import torch
from transformers import AutoTokenizer
from typing import Dict, List, Tuple

from app.core.panini_engine import PaniniEngine
from app.core.neural_bridge import (
    KarakaBridge, verify_karaka_prediction,
    IDX_TO_KARAKA, KARAKA_TO_VIBHAKTI
)
from app.schemas.pydantic_models import (
    WordAnalysis, KarakaTrace, VerifyResponse, VerificationCertificate
)


class ReasoningOracle:
    """
    The production reasoning engine.
    Evaluates sentences through the full Anvaya-Bodha pipeline.
    """

    def __init__(
        self,
        panini: PaniniEngine,
        bridge: KarakaBridge,
        tokenizer: AutoTokenizer,
        device: torch.device
    ):
        self.panini = panini
        self.bridge = bridge
        self.tokenizer = tokenizer
        self.device = device

    def verify_sentence(self, sentence: str) -> VerifyResponse:
        """
        Full Anvaya-Bodha verification pipeline.
        Returns a structured VerifyResponse.
        """
        words = sentence.strip().split()
        word_analyses: List[WordAnalysis] = []
        violations: List[str] = []
        has_verb = False
        has_karman = False

        for i, word in enumerate(words):
            # Symbolic Parse (The Law)
            analysis = self.panini.segment_word(word)

            # Neural Prediction (The Hypothesis)
            inputs = self.tokenizer(word, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _, karaka_logits = self.bridge(inputs.input_ids, inputs.attention_mask)

            pred_idx = karaka_logits[0][1].argmax().item() if karaka_logits.size(1) > 1 else 0
            predicted_karaka = IDX_TO_KARAKA.get(pred_idx % 6, 'Unknown')

            wa = WordAnalysis(
                original=word,
                pos=analysis.get('pos'),
                pratipadika=analysis.get('pratipadika'),
                dhatu=analysis.get('dhatu'),
                vibhakti=analysis.get('vibhakti') if isinstance(analysis.get('vibhakti'), list) else (
                    [analysis['vibhakti']] if analysis.get('vibhakti') else None
                ),
                gender=analysis.get('gender'),
                number=analysis.get('number'),
                tense=analysis.get('tense'),
                person=analysis.get('person'),
            )

            if analysis['pos'] == 'verb':
                has_verb = True
                wa.neural_karaka = "Verb"
                wa.karaka_valid = True
            elif analysis['pos'] == 'noun':
                is_valid, detected, allowed = verify_karaka_prediction(
                    word, predicted_karaka, self.panini
                )
                wa.neural_karaka = predicted_karaka
                wa.karaka_valid = is_valid

                if not is_valid:
                    violations.append(
                        f"LinguisticViolation: '{word}' cannot be '{predicted_karaka}'. "
                        f"Detected vibhakti {detected} incompatible with allowed {allowed}."
                    )
                else:
                    if predicted_karaka == 'Karman':
                        has_karman = True
            else:
                wa.neural_karaka = predicted_karaka
                wa.karaka_valid = None

            word_analyses.append(wa)

        # Ākāṅkṣā (Expectancy) check
        if has_verb and not has_karman and not violations:
            violations.append(
                "ĀkāṅkṣāViolation: Verb expects an object (Karman) "
                "but none was legally resolved."
            )

        is_consistent = len(violations) == 0
        status = "Valid Anvaya" if is_consistent else "Linguistic Violation"

        return VerifyResponse(
            sentence=sentence,
            status=status,
            word_analyses=word_analyses,
            violations=violations,
            is_logically_consistent=is_consistent
        )

    def generate_solnet_certificate(
        self,
        transaction_intent: str,
        node_id: str = None
    ) -> Tuple[VerificationCertificate, bool]:
        """
        Generate a SOLNET Verification Certificate.
        Returns (certificate, has_violations).
        """
        words = transaction_intent.strip().split()
        karaka_traces: List[KarakaTrace] = []
        violations: List[str] = []

        for word in words:
            analysis = self.panini.segment_word(word)

            inputs = self.tokenizer(word, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _, karaka_logits = self.bridge(inputs.input_ids, inputs.attention_mask)

            pred_idx = karaka_logits[0][1].argmax().item() if karaka_logits.size(1) > 1 else 0
            predicted_karaka = IDX_TO_KARAKA.get(pred_idx % 6, 'Unknown')

            if analysis['pos'] == 'noun':
                is_valid, detected, allowed = verify_karaka_prediction(
                    word, predicted_karaka, self.panini
                )

                trace = KarakaTrace(
                    word=word,
                    predicted_karaka=predicted_karaka,
                    allowed_vibhaktis=allowed,
                    detected_vibhaktis=detected,
                    is_valid=is_valid
                )
                karaka_traces.append(trace)

                if not is_valid:
                    violations.append(
                        f"LinguisticViolation: '{word}' assigned '{predicted_karaka}' "
                        f"but morphology shows {detected}."
                    )

        # Generate the Logic Hash (SHA-256 of the trace)
        trace_data = json.dumps(
            [t.model_dump() for t in karaka_traces],
            ensure_ascii=False,
            sort_keys=True
        )
        logic_hash = hashlib.sha256(trace_data.encode('utf-8')).hexdigest()

        is_consistent = len(violations) == 0

        certificate = VerificationCertificate(
            is_logically_consistent=is_consistent,
            karaka_trace=karaka_traces,
            logic_hash=logic_hash,
            violations=violations,
            node_id=node_id
        )

        return certificate, len(violations) > 0
