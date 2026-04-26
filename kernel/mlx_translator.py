"""
Brahman MLX Translator — Neural Text-to-Kāraka Protocol Router

Runs a quantized LLM on Apple Silicon (via MLX) to translate
raw text into the Universal Kāraka Protocol JSON format.

Architecture:
    Raw Input → MLX LLM → Kāraka Protocol JSON → Brahman Kernel → Verdict

The translator is the ONLY neural component. Everything after it
is deterministic symbolic logic.
"""

import json
import hashlib
import time
import re
from typing import Dict, Optional
from pathlib import Path


class MLXTranslator:
    """
    Translates raw text into Kāraka Protocol JSON using a local
    quantized LLM running on Apple Silicon via MLX.
    """

    # The system prompt that teaches the LLM the Kāraka Protocol schema
    SYSTEM_PROMPT = """You are a semantic parser. Your ONLY job is to convert raw text into a JSON object following the Universal Kāraka Protocol.

OUTPUT FORMAT (strict JSON, no commentary):
{
  "protocol_version": "1.0.0",
  "domain": "<domain>",
  "claim": {
    "raw_input": "<the original text>",
    "claim_type": "<assertion|implication|negation|conditional|universal|existential>"
  },
  "karaka_graph": {
    "kriya": {"id": "k0", "surface": "<the verb/action>", "resolved_root": "<canonical root form>"},
    "karta": {"id": "a0", "surface": "<the agent>", "lemma": "<normalized>"} or null,
    "karma": {"id": "o0", "surface": "<the target>", "lemma": "<normalized>"} or null,
    "karana": {"id": "i0", "surface": "<the instrument>", "lemma": "<normalized>"} or null,
    "sampradana": {"id": "r0", "surface": "<the recipient>", "lemma": "<normalized>"} or null,
    "adhikarana": {"id": "e0", "surface": "<the environment>", "lemma": "<normalized>"} or null,
    "apadana": {"id": "s0", "surface": "<the source>", "lemma": "<normalized>"} or null
  }
}

RULES:
1. Extract the MAIN ACTION as kriya. Use the verb's dictionary/root form for resolved_root.
2. Map semantic roles to the 6 kāraka slots. Set unused slots to null.
3. For domain, use the context: "sanskrit", "formal_logic", "memory_safety", "biochemistry", "rust_crypto", "thermodynamics", or "general".
4. Output ONLY the JSON. No markdown, no explanation, no backticks.

FEW-SHOT MAPPING (CRITICAL):
You must map conversational linguistic verbs to mathematical/domain roots.
- Formal Logic: If you see "is", "are", "must be", or "therefore", map it to "entails" or "implies".
  Example: "All cats are animals" -> resolved_root: "entails" (Cat -> Animal).
- Biochemistry: If you see "cuts", "breaks", map it to "cleaves". If you see "adds PO4", map it to "phosphorylates".
- Memory Safety: If you see "copies to", map to "memcpy" or "strcpy"."""

    def __init__(self, model_name: str = "mlx-community/Qwen3-1.7B-4bit", domain: str = "general"):
        """
        Initialize the translator with a quantized MLX model.

        Args:
            model_name: HuggingFace model path (must be MLX-compatible).
            domain: Default domain hint for the cartridge selector.
        """
        self.model_name = model_name
        self.domain = domain
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> str:
        """Load the quantized model into Apple Silicon memory."""
        from mlx_lm import load as mlx_load

        print(f"  Loading {self.model_name}...")
        start = time.time()
        self.model, self.tokenizer = mlx_load(self.model_name)
        elapsed = time.time() - start
        self._loaded = True
        return f"✓ Model loaded in {elapsed:.1f}s"

    def translate(self, raw_input: str, domain: Optional[str] = None) -> Dict:
        """
        Translate raw text into a Kāraka Protocol JSON instance.

        Args:
            raw_input: The raw text to parse (any language/domain).
            domain: Override the default domain hint.

        Returns:
            A Kāraka Protocol dict ready for the Brahman Kernel.
        """
        if not self._loaded:
            self.load()

        from mlx_lm import generate as mlx_generate

        effective_domain = domain or self.domain
        user_prompt = f"Domain: {effective_domain}\nInput: {raw_input}\n/nothink"

        # Build chat messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate with constrained decoding
        start = time.time()
        response = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=1024,
        )
        elapsed = time.time() - start

        # Parse the JSON from the response
        kp = self._extract_json(response, raw_input, effective_domain)
        kp["meta"] = {
            "translator_model": self.model_name,
            "extraction_time_ms": round(elapsed * 1000, 1),
            "source_hash": hashlib.sha256(raw_input.encode("utf-8")).hexdigest()
        }

        return kp

    def _extract_json(self, response: str, raw_input: str, domain: str) -> Dict:
        """
        Extract valid JSON from the LLM response.
        Falls back to a minimal AMBIGUOUS KP if parsing fails.
        """
        # Try to find JSON in the response
        # Strip Qwen3 thinking tokens (<think>...</think>)
        cleaned = response.strip()
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        # Strip markdown code fences if present
        cleaned = cleaned.strip()
        cleaned = re.sub(r'^```json\s*', '', cleaned)
        cleaned = re.sub(r'^```\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

        # Find the outermost JSON object
        brace_start = cleaned.find('{')
        if brace_start == -1:
            return self._fallback_kp(raw_input, domain, "No JSON found in response")

        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(cleaned)):
            if cleaned[i] == '{':
                depth += 1
            elif cleaned[i] == '}':
                depth -= 1
                if depth == 0:
                    json_str = cleaned[brace_start:i+1]
                    try:
                        kp = json.loads(json_str)
                        # Validate minimum required fields
                        if "karaka_graph" in kp and "kriya" in kp.get("karaka_graph", {}):
                            return kp
                        else:
                            return self._fallback_kp(raw_input, domain, "Missing karaka_graph.kriya")
                    except json.JSONDecodeError as e:
                        return self._fallback_kp(raw_input, domain, f"JSON parse error: {e}")

        return self._fallback_kp(raw_input, domain, "Unclosed JSON brace")

    @staticmethod
    def _fallback_kp(raw_input: str, domain: str, reason: str) -> Dict:
        """
        Return a minimal KP that will trigger the kernel's AMBIGUOUS
        circuit breaker. The neural translator failed — don't guess.
        """
        return {
            "protocol_version": "1.0.0",
            "domain": domain,
            "claim": {
                "raw_input": raw_input,
                "claim_type": "assertion",
                "confidence": 0.0
            },
            "karaka_graph": {
                "kriya": {
                    "id": "k0",
                    "surface": "__PARSE_FAILED__",
                    "resolved_root": None
                }
            },
            "_translator_error": reason
        }


# ══════════════════════════════════════════════════════════════════
# FULL PIPELINE: Text → Translate → Verify
# ══════════════════════════════════════════════════════════════════

def run_pipeline(text: str, domain: str, cartridge_path: str, translator: MLXTranslator):
    """
    End-to-end: raw text → MLX translation → kernel verification.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from brahman_kernel import BrahmanKernel

    kernel = BrahmanKernel()
    kernel.load_cartridge(cartridge_path)

    # Step 1: Neural translation
    kp = translator.translate(text, domain=domain)

    # Step 2: Symbolic verification
    result = kernel.verify(kp)

    return kp, result


if __name__ == "__main__":
    import sys

    CART_DIR = Path(__file__).parent / "cartridges"

    # Test cases across domains
    test_cases = [
        ("rāmaḥ pustakam paṭhati", "sanskrit", CART_DIR / "sanskrit_sutras.json"),
        ("All cats are animals. All dogs are animals. Therefore all cats are dogs.", "formal_logic", CART_DIR / "formal_logic_sutras.json"),
        ("strcpy(dest, user_input) without bounds check", "memory_safety", CART_DIR / "memory_safety_sutras.json"),
        ("CDK2 phosphorylates Rb protein without ATP", "biochemistry", CART_DIR / "biochem_sutras.json"),
    ]

    print("=" * 70)
    print("BRAHMAN MLX TRANSLATOR — Neural Text-to-JSON Pipeline")
    print("=" * 70)

    # Initialize translator (downloads model on first run)
    translator = MLXTranslator(domain="general")
    print(translator.load())
    print()

    for text, domain, cartridge in test_cases:
        print(f"── [{domain.upper()}] {text[:60]}...")
        kp, result = run_pipeline(text, domain, str(cartridge), translator)

        print(f"  Extracted kriya: {kp.get('karaka_graph', {}).get('kriya', {}).get('resolved_root', 'NONE')}")
        print(f"  Verdict: {result.verdict}")
        if result.violations:
            for v in result.violations[:2]:
                print(f"    !! {v[:80]}")
        print(f"  Logic Hash: {result.logic_hash[:24]}...")
        print(f"  Translation time: {kp.get('meta', {}).get('extraction_time_ms', '?')}ms")
        print()

    print("=" * 70)
    print("Pipeline complete.")
    print("=" * 70)
