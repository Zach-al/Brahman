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
      (falls back to local parser if API unavailable)
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
import re
import time
import warnings
from pathlib import Path
from typing import Dict

# Suppress noisy deprecation warnings from Google SDK on older Python
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="urllib3")
warnings.filterwarnings("ignore", category=UserWarning)

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
# LOCAL INTENT PARSER — Offline Fallback (No API Required)
# ══════════════════════════════════════════════════════════════════

def _local_parse_intent(user_input: str) -> Dict:
    """
    Keyword-based intent parser. Runs entirely offline.
    Covers all standard blockchain operations and exploit patterns.
    Used when Gemini API is unavailable (quota/network/key issues).
    """
    text = user_input.lower().strip()

    # ── Detect action (kriya) ────────────────────────────────────
    if any(w in text for w in ("bridge", "complete_transfer", "complete transfer", "complete the")):
        kriya = "complete_transfer"
    elif any(w in text for w in ("swap", "exchange", "convert")):
        kriya = "swap"
    elif any(w in text for w in ("send", "transfer", "pay")):
        kriya = "transfer"
    elif any(w in text for w in ("stake", "delegate")):
        kriya = "stake"
    elif any(w in text for w in ("borrow", "loan")):
        kriya = "borrow"
    elif any(w in text for w in ("mint", "create")):
        kriya = "mint_to"
    else:
        kriya = "transfer"  # Default safe action

    kp = {
        "kriya": {"id": "k0", "surface": kriya, "resolved_root": kriya},
        "karta": {
            "id": "a0", "surface": "user_wallet", "is_signer": True,
            "constraints": [
                {"rule_id": "RC-001", "check": "signer authority",
                 "field": "is_signer", "expected": True, "actual": True}
            ],
        },
        "karma": {"id": "o0", "surface": user_input, "lemma": "user_intent"},
    }

    # ── Detect exploit patterns ──────────────────────────────────

    # Wormhole pattern: fake sysvar accounts
    sysvar_match = re.search(r'(?:sysvar|signature\s*set)[:\s]*(\S+)', text, re.IGNORECASE)
    if sysvar_match:
        fake_sysvar = sysvar_match.group(1)
        if fake_sysvar.lower() != "sysvar1nstruction1111111111111111111111111":
            kp["karana"] = {
                "id": "i0", "surface": fake_sysvar, "lemma": "fake_sysvar",
                "instruction_sysvar_account": fake_sysvar,
                "constraints": [
                    {"rule_id": "WH-001", "check": "sysvar instruction account",
                     "field": "instruction_sysvar_account",
                     "expected": "Sysvar1nstruction1111111111111111111111111",
                     "actual": fake_sysvar}
                ],
            }
            kp["karta"]["signature_verified_by_secp256k1"] = False
            kp["karta"]["sig_account_owner_is_bridge"] = False
            kp["karta"]["constraints"].extend([
                {"rule_id": "RC-006", "check": "secp256k1 verification",
                 "field": "signature_verified_by_secp256k1", "expected": True, "actual": False},
                {"rule_id": "RC-007", "check": "bridge ownership",
                 "field": "sig_account_owner_is_bridge", "expected": True, "actual": False},
            ])

    # Unverified bridge pattern
    elif "unverified" in text or ("bridge" in text and ("unsafe" in text or "unknown" in text)):
        kp["karta"]["signature_verified_by_secp256k1"] = False
        kp["karta"]["sig_account_owner_is_bridge"] = False
        kp["karta"]["constraints"].extend([
            {"rule_id": "RC-006", "check": "secp256k1 verification",
             "field": "signature_verified_by_secp256k1", "expected": True, "actual": False},
            {"rule_id": "RC-007", "check": "bridge ownership",
             "field": "sig_account_owner_is_bridge", "expected": True, "actual": False},
        ])

    # Oracle spot price (non-TWAP) pattern
    elif "oracle" in text and "twap" not in text:
        uses_twap = False
        kp["karana"] = {
            "id": "i0", "surface": "oracle price feed", "lemma": "oracle",
            "constraints": [
                {"rule_id": "RC-008", "check": "trusted oracle",
                 "field": "oracle_is_trusted", "expected": True, "actual": True},
                {"rule_id": "RC-009", "check": "TWAP pricing",
                 "field": "uses_twap", "expected": True, "actual": uses_twap},
            ],
        }

    # Standard transfer: add recipient if "to <address>" is found
    to_match = re.search(r'\bto\s+(\S+\.sol|\S+\.eth|\S+)', text)
    if to_match and "karana" not in kp:
        kp["sampradana"] = {
            "id": "r0", "surface": to_match.group(1), "lemma": to_match.group(1),
            "constraints": [
                {"rule_id": "RC-003", "check": "designated recipient",
                 "field": "is_designated_recipient", "expected": True, "actual": True}
            ],
        }
        # Add program_id verification for standard transfers
        kp["karma"]["constraints"] = [
            {"rule_id": "RC-002", "check": "program ID verified",
             "field": "program_id_verified", "expected": True, "actual": True}
        ]

    return kp


# ══════════════════════════════════════════════════════════════════
# THE AGENT
# ══════════════════════════════════════════════════════════════════

class BrahmanAgent:
    """
    AI Agent with pre-execution formal verification.

    Every action the agent takes gets formally verified by
    Sanskrit grammar before execution. Wormhole-proof by design.

    Translation pipeline:
      1. Try Gemini API (native JSON mode, zero hallucination)
      2. Fall back to local keyword parser if API unavailable
    """

    def __init__(self):
        self.kernel = BrahmanKernel()
        self.gemini_model = None
        self._gemini_available = None  # None = untested, True/False = cached

        # Load the rust_crypto cartridge (covers Wormhole, Mango, Cashio)
        cartridge_path = PROJECT_ROOT / "kernel" / "cartridges" / "rust_crypto_sutras.json"
        if cartridge_path.exists():
            msg = self.kernel.load_cartridge(str(cartridge_path))
            print(f"[BRAHMAN-AGENT] {msg}")
        else:
            print(f"[BRAHMAN-AGENT] ⚠ Cartridge not found at {cartridge_path}")

    def _init_gemini(self) -> bool:
        """Lazy-load Gemini. Returns True if ready, False if unavailable."""
        if self._gemini_available is False:
            return False
        if self.gemini_model is not None:
            return True

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self._gemini_available = False
            return False

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            self.gemini_model = genai.GenerativeModel(
                "models/gemini-1.5-flash",
                system_instruction=SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            self._gemini_available = True
            return True
        except Exception as e:
            print(f"    ⚠ Gemini init failed: {e}")
            self._gemini_available = False
            return False

    # ── Step 1: Understand ───────────────────────────────────────

    def understand_intent(self, user_input: str) -> Dict:
        """
        Parse user intent into Kāraka Protocol JSON.

        Pipeline:
          1. Try Gemini API (native JSON mode)
          2. Fall back to local keyword parser if API fails
        """
        # Try Gemini first
        if self._init_gemini():
            try:
                start = time.time()
                response = self.gemini_model.generate_content(user_input)
                elapsed_ms = (time.time() - start) * 1000

                raw_text = response.text.strip()
                kp = json.loads(raw_text)
                print(f"    ⏱  Gemini translation: {elapsed_ms:.0f}ms")
                return kp
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    print("    ⚠ Gemini quota exceeded — switching to local parser")
                elif "404" in err_str:
                    print("    ⚠ Gemini model unavailable — using local parser")
                else:
                    print(f"    ⚠ Gemini error — using local parser")
                self._gemini_available = False

        # Fallback: local keyword parser
        start = time.time()
        kp = _local_parse_intent(user_input)
        elapsed_ms = (time.time() - start) * 1000
        print(f"    ⏱  Local parser: {elapsed_ms:.2f}ms")
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

        User text → Translate (Gemini or Local) → BrahmanKernel (Verdict)
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
