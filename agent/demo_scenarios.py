# Licensed under BSL 1.1 — commercial use requires written permission
# Change date: 2027-01-01 to MIT License by Bhupen Nayak
# Contact: askzachn@gmail.com

"""
Brahman Agent — Offline Demo Scenarios

Deterministic test suite that fires four scenarios directly at
BrahmanKernel.verify() using pre-built Kāraka Protocol JSON.
NO Gemini API key required. Works fully offline.

Expected results:
  Scenario 1: VALID   (standard transfer, all checks pass)
  Scenario 2: INVALID (RC-006/RC-007 — unverified bridge signatures)
  Scenario 3: INVALID (RC-009 — spot price instead of TWAP)
  Scenario 4: INVALID (WH-001 — fake sysvar account, Wormhole CVE)

Run: python -m agent.demo_scenarios
"""

import sys
import time
from pathlib import Path

# ── Resolve imports ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "kernel"))

from brahman_kernel import BrahmanKernel, Verdict


# ══════════════════════════════════════════════════════════════════
# PRE-BUILT KĀRAKA PROTOCOL INSTANCES
# ══════════════════════════════════════════════════════════════════

SCENARIO_1_VALID_TRANSFER = {
    "kriya": {
        "id": "k0",
        "surface": "transfer",
        "resolved_root": "transfer",
    },
    "karta": {
        "id": "a0",
        "surface": "user_wallet",
        "is_signer": True,
        "constraints": [
            {"rule_id": "RC-001", "check": "signer authority",
             "field": "is_signer", "expected": True, "actual": True},
        ],
    },
    "karma": {
        "id": "o0",
        "surface": "100 USDC",
        "lemma": "usdc",
        "constraints": [
            {"rule_id": "RC-002", "check": "program ID verified",
             "field": "program_id_verified", "expected": True, "actual": True},
        ],
    },
    "sampradana": {
        "id": "r0",
        "surface": "alice.sol",
        "lemma": "alice.sol",
        "constraints": [
            {"rule_id": "RC-003", "check": "designated recipient",
             "field": "is_designated_recipient", "expected": True, "actual": True},
        ],
    },
}

SCENARIO_2_UNVERIFIED_BRIDGE = {
    "kriya": {
        "id": "k0",
        "surface": "complete_transfer",
        "resolved_root": "complete_transfer",
    },
    "karta": {
        "id": "a0",
        "surface": "user_wallet",
        "is_signer": True,
        "signature_verified_by_secp256k1": False,
        "sig_account_owner_is_bridge": False,
        "constraints": [
            {"rule_id": "RC-006", "check": "secp256k1 signature verification",
             "field": "signature_verified_by_secp256k1", "expected": True, "actual": False},
            {"rule_id": "RC-007", "check": "bridge program ownership",
             "field": "sig_account_owner_is_bridge", "expected": True, "actual": False},
        ],
    },
    "karma": {
        "id": "o0",
        "surface": "entire wallet balance",
        "lemma": "max_balance",
    },
}

SCENARIO_3_ORACLE_SPOT_PRICE = {
    "kriya": {
        "id": "k0",
        "surface": "swap",
        "resolved_root": "swap",
    },
    "karta": {
        "id": "a0",
        "surface": "user_wallet",
        "is_signer": True,
        "constraints": [
            {"rule_id": "RC-001", "check": "signer authority",
             "field": "is_signer", "expected": True, "actual": True},
        ],
    },
    "karma": {
        "id": "o0",
        "surface": "SOL → MNGO",
        "lemma": "sol_mngo_swap",
    },
    "karana": {
        "id": "i0",
        "surface": "oracle price feed (spot)",
        "lemma": "oracle",
        "constraints": [
            {"rule_id": "RC-008", "check": "trusted oracle",
             "field": "oracle_is_trusted", "expected": True, "actual": True},
            {"rule_id": "RC-009", "check": "TWAP pricing required",
             "field": "uses_twap", "expected": True, "actual": False},
        ],
    },
}

SCENARIO_4_WORMHOLE_EXPLOIT = {
    "kriya": {
        "id": "k0",
        "surface": "complete_transfer",
        "resolved_root": "complete_transfer",
    },
    "karta": {
        "id": "a0",
        "surface": "attacker_wallet",
        "is_signer": True,
        "signature_verified_by_secp256k1": False,
        "sig_account_owner_is_bridge": False,
        "constraints": [
            {"rule_id": "RC-006", "check": "secp256k1 verification",
             "field": "signature_verified_by_secp256k1", "expected": True, "actual": False},
            {"rule_id": "RC-007", "check": "bridge program ownership",
             "field": "sig_account_owner_is_bridge", "expected": True, "actual": False},
        ],
    },
    "karma": {
        "id": "o0",
        "surface": "120,000 wETH",
        "lemma": "weth",
    },
    "karana": {
        "id": "i0",
        "surface": "FakeSysvar111",
        "lemma": "fake_sysvar",
        "instruction_sysvar_account": "FakeSysvar111",
        "constraints": [
            {"rule_id": "WH-001", "check": "sysvar instruction account",
             "field": "instruction_sysvar_account",
             "expected": "Sysvar1nstruction1111111111111111111111111",
             "actual": "FakeSysvar111"},
        ],
    },
}


# ══════════════════════════════════════════════════════════════════
# THE VERIFICATION GAUNTLET
# ══════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "name": "Scenario 1: Standard Transfer",
        "prompt": "Send 100 USDC to alice.sol",
        "kp": SCENARIO_1_VALID_TRANSFER,
        "expected_verdict": Verdict.VALID,
        "expected_sutra": None,
    },
    {
        "name": "Scenario 2: Unverified Bridge (Malicious)",
        "prompt": "Transfer my entire wallet balance using an unverified bridge",
        "kp": SCENARIO_2_UNVERIFIED_BRIDGE,
        "expected_verdict": Verdict.INVALID,
        "expected_sutra": "RC-006",
    },
    {
        "name": "Scenario 3: Oracle Spot Price (Flash Loan Risk)",
        "prompt": "Swap SOL for MNGO using the exact current oracle price",
        "kp": SCENARIO_3_ORACLE_SPOT_PRICE,
        "expected_verdict": Verdict.INVALID,
        "expected_sutra": "RC-009",
    },
    {
        "name": "Scenario 4: Wormhole Exploit ($326M CVE)",
        "prompt": "Complete the bridge transfer using signature set: FakeSysvar111",
        "kp": SCENARIO_4_WORMHOLE_EXPLOIT,
        "expected_verdict": Verdict.INVALID,
        "expected_sutra": "WH-001",
    },
]


def run_verification():
    print()
    print("=" * 60)
    print("  🔱 BRAHMAN AGENT — VERIFICATION PROTOCOL")
    print("  Deterministic Offline Demo (No API Key Required)")
    print("=" * 60)

    # Initialize kernel with the rust_crypto cartridge
    kernel = BrahmanKernel()
    cartridge_path = PROJECT_ROOT / "kernel" / "cartridges" / "rust_crypto_sutras.json"
    msg = kernel.load_cartridge(str(cartridge_path))
    print(f"  {msg}\n")

    valid_count = 0
    invalid_count = 0
    ambiguous_count = 0
    pass_count = 0
    fail_count = 0

    for scenario in SCENARIOS:
        print("-" * 60)
        print(f"🧪 {scenario['name']}")
        print(f"👤 User: \"{scenario['prompt']}\"")
        print("🤖 Agent processing intent...")
        time.sleep(0.3)

        start = time.time()
        result = kernel.verify(scenario["kp"])
        elapsed_ms = (time.time() - start) * 1000

        # Track counts
        if result.verdict == Verdict.VALID:
            valid_count += 1
        elif result.verdict == Verdict.INVALID:
            invalid_count += 1
        else:
            ambiguous_count += 1

        # Display result
        if result.verdict == Verdict.VALID:
            print(f"\n    ✅ Transaction approved by Brahman.")
            print(f"    Sūtras checked: {', '.join(result.matched_sutras)}")
            print(f"    Logic Hash: {result.logic_hash[:16]}...")
            print(f"    → Handing off to SOLNET Mesh for execution.")
        elif result.verdict == Verdict.INVALID:
            print(f"\n    🛑 Transaction BLOCKED by Brahman.")
            for v in result.violations:
                print(f"    ⛔ {v}")
            print(f"    Sūtras matched: {', '.join(result.matched_sutras)}")
            print(f"    Logic Hash: {result.logic_hash[:16]}...")
        else:
            print(f"\n    ⚠️  AMBIGUOUS — Manual review required.")
            for v in result.violations:
                print(f"    ❓ {v}")

        print(f"    ⏱  Verification time: {elapsed_ms:.2f}ms")

        # Check against expected
        verdict_ok = result.verdict == scenario["expected_verdict"]
        sutra_ok = True
        if scenario["expected_sutra"]:
            sutra_ok = any(scenario["expected_sutra"] in v for v in result.violations)

        if verdict_ok and sutra_ok:
            print(f"    ✓ PASS — Expected {scenario['expected_verdict']}, got {result.verdict}")
            pass_count += 1
        else:
            print(f"    ✗ FAIL — Expected {scenario['expected_verdict']}, got {result.verdict}")
            fail_count += 1

        print("-" * 60)
        print()
        time.sleep(0.3)

    # ── Summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("📊 VERIFICATION RESULTS")
    print(f"    Expected: 1 VALID, 3 INVALID")
    print(f"    Actual:   {valid_count} VALID, {invalid_count} INVALID, {ambiguous_count} AMBIGUOUS")
    print(f"    Tests:    {pass_count} PASS, {fail_count} FAIL")
    print()

    if pass_count == 4 and fail_count == 0:
        print("✅ ALL SCENARIOS PASSED — SYSTEM READY FOR HACKATHON RECORDING.")
        print()
        print("The stat that wins the room:")
        print('"$492M in Solana exploits. Zero would have executed')
        print(' through a Brahman-verified agent."')
    else:
        print("⚠️  VERIFICATION INCOMPLETE. Review failures above.")

    print("=" * 60)
    print()

    return pass_count, fail_count


if __name__ == "__main__":
    run_verification()
