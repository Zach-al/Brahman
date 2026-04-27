"""
Brahman Kernel Self-Test — Proves the same kernel verifies
Sanskrit, Smart Contracts, and Thermodynamics by swapping cartridges.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from brahman_kernel import BrahmanKernel, Verdict

CARTRIDGE_DIR = Path(__file__).parent / "cartridges"

def test_sanskrit():
    """Sanskrit: Transitive verb without object → INVALID."""
    kernel = BrahmanKernel()
    print(kernel.load_cartridge(str(CARTRIDGE_DIR / "sanskrit_sutras.json")))

    # Valid: "rāmaḥ pustakam paṭhati" (Ram reads the book)
    result = kernel.verify({
        "protocol_version": "1.0.0",
        "domain": "sanskrit",
        "claim": {"raw_input": "rāmaḥ pustakam paṭhati", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "paṭhati", "resolved_root": "paṭhati"},
            "karta": {"id": "a0", "surface": "rāmaḥ", "lemma": "rāma",
                      "constraints": [{"rule_id": "S2.3.1", "check": "case", "field": "vibhakti", "actual": "nominative"}]},
            "karma": {"id": "o0", "surface": "pustakam", "lemma": "pustaka",
                      "constraints": [{"rule_id": "S2.3.2", "check": "case", "field": "vibhakti", "actual": "accusative"}]},
        }
    })
    assert result.verdict == Verdict.VALID, f"Expected VALID, got {result.verdict}: {result.violations}"
    assert result.dhatu_found is True
    print(f"  ✓ Valid sentence   → {result.verdict} (hash: {result.logic_hash[:16]}...)")

    # Invalid: Transitive verb with no object
    result2 = kernel.verify({
        "protocol_version": "1.0.0",
        "domain": "sanskrit",
        "claim": {"raw_input": "rāmaḥ paṭhati", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "paṭhati", "resolved_root": "paṭhati"},
            "karta": {"id": "a0", "surface": "rāmaḥ", "lemma": "rāma",
                      "constraints": [{"rule_id": "S2.3.1", "check": "case", "field": "vibhakti", "actual": "nominative"}]},
        }
    })
    assert result2.verdict == Verdict.AMBIGUOUS, f"Expected AMBIGUOUS, got {result2.verdict}"
    print(f"  ✓ Missing karma    → {result2.verdict} ({result2.violations[0][:60]}...)")

    # Ambiguous: Unknown verb root
    result3 = kernel.verify({
        "protocol_version": "1.0.0",
        "domain": "sanskrit",
        "claim": {"raw_input": "rāmaḥ xyzati", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "xyzati", "resolved_root": "xyzati"},
            "karta": {"id": "a0", "surface": "rāmaḥ", "lemma": "rāma"},
        }
    })
    assert result3.verdict == Verdict.AMBIGUOUS, f"Expected AMBIGUOUS, got {result3.verdict}"
    assert result3.dhatu_found is False
    print(f"  ✓ Unknown root     → {result3.verdict} (circuit breaker fired)")

    print(kernel.unload_cartridge())

def test_rust_crypto():
    """Solana: Transfer without signer → INVALID."""
    kernel = BrahmanKernel()
    print(kernel.load_cartridge(str(CARTRIDGE_DIR / "rust_crypto_sutras.json")))

    # Invalid: Transfer where agent is NOT a signer
    result = kernel.verify({
        "protocol_version": "1.0.0",
        "domain": "rust_crypto",
        "claim": {"raw_input": "transfer 100 SOL from wallet_A to wallet_B", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "transfer", "resolved_root": "transfer"},
            "karta": {"id": "a0", "surface": "wallet_A", "lemma": "wallet_a",
                      "constraints": [{"rule_id": "RC-001", "check": "signer", "field": "is_signer", "actual": False}]},
            "karma": {"id": "o0", "surface": "wallet_B", "lemma": "wallet_b"},
            "karana": {"id": "i0", "surface": "100 SOL", "lemma": "100_sol",
                       "constraints": [{"rule_id": "RC-005", "check": "overflow", "field": "overflow_checked", "actual": True}]},
        }
    })
    assert result.verdict == Verdict.INVALID, f"Expected INVALID, got {result.verdict}"
    print(f"  ✓ Unsigned transfer → {result.verdict} ({result.violations[0][:60]}...)")

    print(kernel.unload_cartridge())

def test_thermodynamics():
    """Thermodynamics: Combustion without oxidizer → INVALID."""
    kernel = BrahmanKernel()
    print(kernel.load_cartridge(str(CARTRIDGE_DIR / "thermo_sutras.json")))

    # Invalid: Combustion in a vacuum (no oxidizer)
    result = kernel.verify({
        "protocol_version": "1.0.0",
        "domain": "thermodynamics",
        "claim": {"raw_input": "wood combusts in vacuum", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "combusts", "resolved_root": "combusts"},
            "karta": {"id": "a0", "surface": "wood", "lemma": "wood"},
            "adhikarana": {"id": "e0", "surface": "vacuum", "lemma": "vacuum",
                          "conditions": ["pressure=0", "oxidizer=none"],
                          "constraints": [
                              {"rule_id": "TD-004", "check": "oxidizer", "field": "oxidizer_present", "actual": False}
                          ]},
        }
    })
    assert result.verdict == Verdict.INVALID, f"Expected INVALID, got {result.verdict}"
    print(f"  ✓ Vacuum combustion → {result.verdict} ({result.violations[0][:60]}...)")

    print(kernel.unload_cartridge())

if __name__ == "__main__":
    print("=" * 60)
    print("BRAHMAN KERNEL SELF-TEST")
    print("Same kernel. Three domains. Swap the cartridge.")
    print("=" * 60)

    print("\n── DOMAIN: Sanskrit ──────────────────────────────────────")
    test_sanskrit()

    print("\n── DOMAIN: Rust/Solana ───────────────────────────────────")
    test_rust_crypto()

    print("\n── DOMAIN: Thermodynamics ────────────────────────────────")
    test_thermodynamics()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED — Kernel is domain-agnostic.")
    print("=" * 60)
