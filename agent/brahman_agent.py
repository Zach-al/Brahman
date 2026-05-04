# Licensed under BSL 1.1 — commercial use requires written permission
# Change date: 2027-01-01 to MIT License by Bhupen Nayak
# Contact: askzachn@gmail.com

"""
Brahman AI Agent — Pre-Execution Formal Verification

The agent sits between the user and the blockchain. Before it signs
any transaction, it asks Brahman first:

    User: "Swap 1000 USDC for SOL"
            ↓
    AI Agent receives intent
            ↓
    Gemini translates to Kāraka Protocol JSON
            ↓
    BrahmanKernel.verify() — 0.1ms deterministic check
      VALID     → Agent signs and submits via SOLNET mesh
      INVALID   → Agent refuses and explains why
      AMBIGUOUS → Agent asks user to confirm
            ↓
    Transaction flows through SOLNET to validator

No other AI agent has formal verification before execution.
They all just sign and hope.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict

# ── Resolve project root so kernel imports work ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kernel"))

from brahman_kernel import BrahmanKernel, VerificationResult, Verdict

# ── Load .env if python-dotenv is available ──────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "agent" / ".env")
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — The Bulletproof JSON Lock
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are the semantic translation layer for the Brahman Formal Verification Engine.
Your sole purpose is to parse natural language blockchain requests into strict Kāraka Protocol JSON.

You DO NOT execute transactions. You DO NOT give financial advice. You ONLY output JSON.

The Kāraka Protocol maps every action to Sanskrit grammatical structures:
1. "kriya"      — The ACTION (verb): "transfer", "swap", "bridge", "stake", "complete_transfer", "borrow", "mint_to"
2. "karta"      — The AGENT (who initiates): includes "is_signer", "signature_verified_by_secp256k1", "sig_account_owner_is_bridge"
3. "karma"      — The TARGET (what is acted upon): includes asset, amount, destination, "program_id_verified", "ltv_checked"
4. "karana"      — The INSTRUMENT (mechanism used): includes "oracle_is_trusted", "uses_twap", "overflow_checked", "instruction_sysvar_account"
5. "sampradana"  — The RECIPIENT (beneficiary): includes "is_designated_recipient"
6. "adhikarana"  — The ENVIRONMENT (context): includes "collateral_fully_verified"

CRITICAL SECURITY FIELDS:
- For standard transfers: karta.is_signer = true (the sender has authority)
- For bridge operations: karta.signature_verified_by_secp256k1 and karta.sig_account_owner_is_bridge
- For oracle swaps: karana.oracle_is_trusted and karana.uses_twap
- For Wormhole-style bridges: karana.instruction_sysvar_account must equal "Sysvar1nstruction1111111111111111111111111"
- If a user mentions "FakeSysvar" or a suspicious sysvar account, set instruction_sysvar_account to that fake value
- If a user says "unverified bridge" → set signature_verified_by_secp256k1 = false and sig_account_owner_is_bridge = false
- If a user says "oracle price" (not TWAP) → set uses_twap = false

OUTPUT RULES:
- Your response must be ONLY valid JSON. No markdown, no backticks, no explanations.
- Always include the "kriya" field with "surface" and "resolved_root".
- Always include "karta" with at minimum "surface" and "is_signer".

EXAMPLES:

Input: "Send 100 USDC to alice.sol"
Output: {"kriya":{"id":"k0","surface":"transfer","resolved_root":"transfer"},"karta":{"id":"a0","surface":"user_wallet","is_signer":true,"constraints":[{"rule_id":"RC-001","check":"signer authority","field":"is_signer","expected":true,"actual":true}]},"karma":{"id":"o0","surface":"100 USDC","lemma":"usdc","constraints":[{"rule_id":"RC-002","check":"program verification","field":"program_id_verified","expected":true,"actual":true}]},"sampradana":{"id":"r0","surface":"alice.sol","lemma":"alice.sol","constraints":[{"rule_id":"RC-003","check":"designated recipient","field":"is_designated_recipient","expected":true,"actual":true}]}}

Input: "Transfer my entire wallet balance using an unverified bridge"
Output: {"kriya":{"id":"k0","surface":"complete_transfer","resolved_root":"complete_transfer"},"karta":{"id":"a0","surface":"user_wallet","is_signer":true,"signature_verified_by_secp256k1":false,"sig_account_owner_is_bridge":false,"constraints":[{"rule_id":"RC-006","check":"secp256k1 signature verification","field":"signature_verified_by_secp256k1","expected":true,"actual":false},{"rule_id":"RC-007","check":"bridge program ownership","field":"sig_account_owner_is_bridge","expected":true,"actual":false}]},"karma":{"id":"o0","surface":"entire wallet balance","lemma":"max_balance"}}

Input: "Swap SOL for MNGO using the exact current oracle price"
Output: {"kriya":{"id":"k0","surface":"swap","resolved_root":"swap"},"karta":{"id":"a0","surface":"user_wallet","is_signer":true,"constraints":[{"rule_id":"RC-001","check":"signer authority","field":"is_signer","expected":true,"actual":true}]},"karma":{"id":"o0","surface":"SOL to MNGO","lemma":"sol_mngo_swap"},"karana":{"id":"i0","surface":"oracle price feed","lemma":"oracle","constraints":[{"rule_id":"RC-008","check":"trusted oracle","field":"oracle_is_trusted","expected":true,"actual":true},{"rule_id":"RC-009","check":"TWAP pricing","field":"uses_twap","expected":true,"actual":false}]}}

Input: "Complete the bridge transfer using this signature set: FakeSysvar111"
Output: {"kriya":{"id":"k0","surface":"complete_transfer","resolved_root":"complete_transfer"},"karta":{"id":"a0","surface":"user_wallet","is_signer":true,"signature_verified_by_secp256k1":false,"sig_account_owner_is_bridge":false,"constraints":[{"rule_id":"RC-006","check":"secp256k1 verification","field":"signature_verified_by_secp256k1","expected":true,"actual":false},{"rule_id":"RC-007","check":"bridge ownership","field":"sig_account_owner_is_bridge","expected":true,"actual":false}]},"karma":{"id":"o0","surface":"bridge transfer","lemma":"bridge_transfer"},"karana":{"id":"i0","surface":"FakeSysvar111","lemma":"fake_sysvar","instruction_sysvar_account":"FakeSysvar111","constraints":[{"rule_id":"WH-001","check":"sysvar instruction account","field":"instruction_sysvar_account","expected":"Sysvar1nstruction1111111111111111111111111","actual":"FakeSysvar111"}]}}
"""


# ══════════════════════════════════════════════════════════════════
# THE AGENT
# ══════════════════════════════════════════════════════════════════

class BrahmanAgent:
    """
    AI Agent with pre-execution formal verification.

    Every action the agent takes gets formally verified by
    Sanskrit grammar before execution. Wormhole-proof by design.
    """

    def __init__(self):
        self.kernel = BrahmanKernel()
        self.gemini_model = None

        # Load the rust_crypto cartridge (covers Wormhole, Mango, Cashio)
        cartridge_path = PROJECT_ROOT / "kernel" / "cartridges" / "rust_crypto_sutras.json"
        if cartridge_path.exists():
            msg = self.kernel.load_cartridge(str(cartridge_path))
            print(f"[BRAHMAN-AGENT] {msg}")
        else:
            print(f"[BRAHMAN-AGENT] ⚠ Cartridge not found at {cartridge_path}")

    def _init_gemini(self):
        """Lazy-load Gemini on first use."""
        if self.gemini_model is not None:
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Export it or add to agent/.env"
            )

        import google.generativeai as genai
        genai.configure(api_key=api_key)

        self.gemini_model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

    # ── Step 1: Understand ───────────────────────────────────────

    def understand_intent(self, user_input: str) -> Dict:
        """
        Use Gemini to parse user intent into Kāraka Protocol JSON.
        Returns a dict that BrahmanKernel.verify() can consume.
        """
        self._init_gemini()

        start = time.time()
        response = self.gemini_model.generate_content(user_input)
        elapsed_ms = (time.time() - start) * 1000

        raw_text = response.text.strip()

        try:
            kp = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from markdown fences
            import re
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                kp = json.loads(match.group())
            else:
                raise ValueError(f"Gemini returned non-JSON: {raw_text[:200]}")

        print(f"    ⏱  Gemini translation: {elapsed_ms:.0f}ms")
        return kp

    # ── Step 2: Verify ───────────────────────────────────────────

    def verify_with_brahman(self, kp: Dict) -> VerificationResult:
        """
        Send to BrahmanKernel for formal verification.
        Pure deterministic logic — no neural networks, no guessing.
        """
        start = time.time()
        result = self.kernel.verify(kp)
        elapsed_ms = (time.time() - start) * 1000
        print(f"    ⏱  Brahman verification: {elapsed_ms:.2f}ms")
        return result

    # ── Step 3: Act or Refuse ────────────────────────────────────

    def execute(self, user_input: str) -> str:
        """
        Main entry point.

        User text → Gemini (Kāraka Protocol) → BrahmanKernel (Verdict)
          VALID     → "Transaction approved. Handing off to SOLNET."
          INVALID   → "BLOCKED — [Sūtra ID] explanation"
          AMBIGUOUS → "Manual review required."
        """
        print(f"\n    📥 Agent received: \"{user_input}\"")

        # Step 1: Understand
        try:
            kp = self.understand_intent(user_input)
        except Exception as e:
            return f"❌ Translation Error: {e}"

        kriya = kp.get("kriya", {}).get("surface", "unknown")
        print(f"    🔍 Parsed intent: {kriya}")

        # Step 2: Verify
        result = self.verify_with_brahman(kp)

        # Step 3: Act or refuse
        if result.verdict == Verdict.VALID:
            return (
                f"✅ Transaction approved by Brahman.\n"
                f"    Action: {kriya}\n"
                f"    Sūtras checked: {', '.join(result.matched_sutras)}\n"
                f"    Logic Hash: {result.logic_hash[:16]}...\n"
                f"    → Handing off to SOLNET Mesh for execution."
            )

        elif result.verdict == Verdict.INVALID:
            violation_text = "\n    ".join(result.violations)
            return (
                f"🛑 Transaction BLOCKED by Brahman.\n"
                f"    Violations:\n    {violation_text}\n"
                f"    Sūtras matched: {', '.join(result.matched_sutras)}\n"
                f"    Logic Hash: {result.logic_hash[:16]}..."
            )

        else:  # AMBIGUOUS
            violation_text = "\n    ".join(result.violations) if result.violations else "Insufficient information."
            return (
                f"⚠️  Transaction flagged as AMBIGUOUS.\n"
                f"    Reason: {violation_text}\n"
                f"    Manual review required before execution."
            )


# ══════════════════════════════════════════════════════════════════
# CLI — Interactive Mode
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🔱 BRAHMAN AI AGENT")
    print("  Pre-Execution Formal Verification")
    print("  Every action verified by Sanskrit grammar before signing.")
    print("=" * 60)

    agent = BrahmanAgent()

    print("\nType a transaction intent, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("What would you like to do? > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        response = agent.execute(user_input)
        print(f"\n{response}\n")
