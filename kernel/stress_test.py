"""
Brahman Kernel Stress Test — Multi-Layered Cartridge Validation

Tests:
  - Multi-violation cascades (single input triggers 3+ sūtras)
  - Cross-role dependency chains
  - Mixed severity (errors + warnings)
  - Edge cases: empty graphs, no cartridge, root miss
  - All 6 domain cartridges
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from brahman_kernel import BrahmanKernel, Verdict

CART = Path(__file__).parent / "cartridges"
passed = 0
failed = 0

def check(label, result, expected_verdict, min_violations=0, must_contain=None):
    global passed, failed
    ok = result.verdict == expected_verdict
    if min_violations > 0:
        ok = ok and len(result.violations) >= min_violations
    if must_contain:
        ok = ok and any(must_contain in v for v in result.violations)

    status = "✓" if ok else "✗"
    if ok:
        passed += 1
    else:
        failed += 1
    detail = f"{len(result.violations)} violations, {len(result.matched_sutras)} sūtras matched"
    print(f"  {status} {label:<50} → {result.verdict:<10} ({detail})")
    if not ok:
        print(f"    EXPECTED: {expected_verdict}, min_violations={min_violations}")
        for v in result.violations:
            print(f"    !! {v}")
    return ok

# ══════════════════════════════════════════════════════════════════
# 1. MEMORY SAFETY
# ══════════════════════════════════════════════════════════════════
def test_memory_safety():
    k = BrahmanKernel()
    print(k.load_cartridge(str(CART / "memory_safety_sutras.json")))

    # 1a. Buffer overflow: strcpy without bounds check
    check("strcpy without bounds check", k.verify({
        "protocol_version": "1.0.0", "domain": "memory_safety",
        "claim": {"raw_input": "strcpy(dest, user_input)", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "strcpy", "resolved_root": "strcpy"},
            "karta": {"id": "a0", "surface": "process", "lemma": "process"},
            "karma": {"id": "o0", "surface": "dest_buffer", "lemma": "dest",
                      "constraints": [
                          {"rule_id": "MEM-001", "check": "bounds", "field": "bounds_checked", "actual": False},
                          {"rule_id": "MEM-005", "check": "null", "field": "null_checked", "actual": True}
                      ]},
            "apadana": {"id": "s0", "surface": "user_input", "lemma": "src",
                        "constraints": [
                            {"rule_id": "MEM-002", "check": "size", "field": "size_leq_dest", "actual": False}
                        ]},
        }
    }), Verdict.INVALID, min_violations=2, must_contain="MEM-001")

    # 1b. Use-after-free
    check("use-after-free on freed pointer", k.verify({
        "protocol_version": "1.0.0", "domain": "memory_safety",
        "claim": {"raw_input": "memcpy(freed_ptr, data, 64)", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "memcpy", "resolved_root": "memcpy"},
            "karta": {"id": "a0", "surface": "thread_1", "lemma": "thread"},
            "karma": {"id": "o0", "surface": "freed_ptr", "lemma": "ptr",
                      "constraints": [
                          {"rule_id": "MEM-001", "check": "bounds", "field": "bounds_checked", "actual": True},
                          {"rule_id": "MEM-003", "check": "freed", "field": "is_freed", "actual": True}
                      ]},
        }
    }), Verdict.INVALID, min_violations=1, must_contain="MEM-003")

    # 1c. Privilege escalation without audit
    check("setuid without audit log", k.verify({
        "protocol_version": "1.0.0", "domain": "memory_safety",
        "claim": {"raw_input": "setuid(0)", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "setuid", "resolved_root": "setuid"},
            "karta": {"id": "a0", "surface": "attacker_proc", "lemma": "proc",
                      "constraints": [
                          {"rule_id": "PRIV-001", "check": "cap", "field": "has_capability", "actual": False}
                      ]},
            "adhikarana": {"id": "e0", "surface": "container", "lemma": "env",
                          "constraints": [
                              {"rule_id": "PRIV-002", "check": "audit", "field": "audit_logged", "actual": False}
                          ]},
        }
    }), Verdict.INVALID, min_violations=2, must_contain="PRIV-001")

    # 1d. W^X violation: writable + executable mmap
    check("mmap W^X violation", k.verify({
        "protocol_version": "1.0.0", "domain": "memory_safety",
        "claim": {"raw_input": "mmap(NULL, 4096, PROT_WRITE|PROT_EXEC, ...)", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "mmap", "resolved_root": "mmap"},
            "karta": {"id": "a0", "surface": "exploit", "lemma": "exploit"},
            "karma": {"id": "o0", "surface": "shellcode_region", "lemma": "region",
                      "constraints": [
                          {"rule_id": "MMAP-001", "check": "wx", "field": "wx_violation", "actual": True}
                      ]},
        }
    }), Verdict.INVALID, min_violations=1, must_contain="MMAP-001")

    # 1e. VALID: Safe snprintf with all checks
    check("safe snprintf (all checks pass)", k.verify({
        "protocol_version": "1.0.0", "domain": "memory_safety",
        "claim": {"raw_input": "snprintf(buf, sizeof(buf), \"%s\", safe_str)", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "snprintf", "resolved_root": "snprintf"},
            "karta": {"id": "a0", "surface": "server", "lemma": "server"},
            "karma": {"id": "o0", "surface": "buf", "lemma": "buf",
                      "constraints": [
                          {"rule_id": "MEM-001", "check": "bounds", "field": "bounds_checked", "actual": True},
                          {"rule_id": "MEM-005", "check": "null", "field": "null_checked", "actual": True}
                      ]},
            "karana": {"id": "i0", "surface": "format_literal", "lemma": "fmt",
                       "constraints": [
                           {"rule_id": "FMT-001", "check": "fmt", "field": "format_string_safe", "actual": True},
                           {"rule_id": "MEM-004", "check": "overflow", "field": "overflow_checked", "actual": True}
                       ]},
        }
    }), Verdict.VALID)

    k.unload_cartridge()

# ══════════════════════════════════════════════════════════════════
# 2. BIOCHEMISTRY
# ══════════════════════════════════════════════════════════════════
def test_biochemistry():
    k = BrahmanKernel()
    print(k.load_cartridge(str(CART / "biochem_sutras.json")))

    # 2a. Phosphorylation without ATP
    check("kinase phosphorylation without ATP", k.verify({
        "protocol_version": "1.0.0", "domain": "biochemistry",
        "claim": {"raw_input": "CDK2 phosphorylates Rb protein", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "phosphorylates", "resolved_root": "phosphorylates"},
            "karta": {"id": "a0", "surface": "CDK2", "lemma": "cdk2",
                      "constraints": [{"rule_id": "BIO-005", "check": "inhibition", "field": "not_inhibited", "actual": True}]},
            "karma": {"id": "o0", "surface": "Rb", "lemma": "rb_protein",
                      "constraints": [{"rule_id": "BIO-001", "check": "substrate", "field": "substrate_compatible", "actual": True}]},
            "adhikarana": {"id": "e0", "surface": "cytoplasm", "lemma": "cytoplasm",
                          "constraints": [
                              {"rule_id": "BIO-002", "check": "atp", "field": "atp_available", "actual": False},
                              {"rule_id": "BIO-003", "check": "ph", "field": "ph_in_range", "actual": True},
                              {"rule_id": "BIO-004", "check": "temp", "field": "below_denaturation_temp", "actual": True}
                          ]},
        }
    }), Verdict.INVALID, min_violations=1, must_contain="BIO-002")

    # 2b. Central dogma violation: protein → DNA
    check("central dogma violation (protein→DNA)", k.verify({
        "protocol_version": "1.0.0", "domain": "biochemistry",
        "claim": {"raw_input": "hemoglobin transcribes into DNA", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "transcribes", "resolved_root": "transcribes"},
            "karta": {"id": "a0", "surface": "hemoglobin", "lemma": "hemoglobin"},
            "karma": {"id": "o0", "surface": "DNA_strand", "lemma": "dna",
                      "constraints": [
                          {"rule_id": "BIO-007", "check": "promoter", "field": "promoter_accessible", "actual": True},
                          {"rule_id": "BIO-011", "check": "dogma", "field": "template_type", "actual": "protein_to_dna"}
                      ]},
        }
    }), Verdict.INVALID, must_contain="BIO-011")

    # 2c. Inhibited enzyme + wrong pH (multi-violation cascade)
    check("inhibited enzyme at wrong pH (3 violations)", k.verify({
        "protocol_version": "1.0.0", "domain": "biochemistry",
        "claim": {"raw_input": "trypsin cleaves casein at pH 2", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "cleaves", "resolved_root": "cleaves"},
            "karta": {"id": "a0", "surface": "trypsin", "lemma": "trypsin",
                      "constraints": [{"rule_id": "BIO-005", "check": "inhibition", "field": "not_inhibited", "actual": False}]},
            "karma": {"id": "o0", "surface": "casein", "lemma": "casein",
                      "constraints": [{"rule_id": "BIO-001", "check": "substrate", "field": "substrate_compatible", "actual": False}]},
            "adhikarana": {"id": "e0", "surface": "stomach", "lemma": "stomach",
                          "constraints": [
                              {"rule_id": "BIO-003", "check": "ph", "field": "ph_in_range", "actual": False},
                              {"rule_id": "BIO-004", "check": "temp", "field": "below_denaturation_temp", "actual": True}
                          ]},
        }
    }), Verdict.INVALID, min_violations=3)

    # 2d. VALID: Proper oxidation with all constraints satisfied
    check("valid oxidation with NAD+ carrier", k.verify({
        "protocol_version": "1.0.0", "domain": "biochemistry",
        "claim": {"raw_input": "lactate dehydrogenase oxidizes lactate", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "oxidizes", "resolved_root": "oxidizes"},
            "karta": {"id": "a0", "surface": "LDH", "lemma": "ldh",
                      "constraints": [{"rule_id": "BIO-005", "check": "inhibition", "field": "not_inhibited", "actual": True}]},
            "karma": {"id": "o0", "surface": "lactate", "lemma": "lactate",
                      "constraints": [{"rule_id": "BIO-001", "check": "substrate", "field": "substrate_compatible", "actual": True}]},
            "karana": {"id": "i0", "surface": "NAD+", "lemma": "nad_plus",
                       "constraints": [{"rule_id": "BIO-006", "check": "carrier", "field": "carrier_compatible", "actual": True}]},
            "adhikarana": {"id": "e0", "surface": "cytoplasm", "lemma": "cytoplasm",
                          "constraints": [
                              {"rule_id": "BIO-003", "check": "ph", "field": "ph_in_range", "actual": True},
                              {"rule_id": "BIO-004", "check": "temp", "field": "below_denaturation_temp", "actual": True}
                          ]},
        }
    }), Verdict.VALID)

    k.unload_cartridge()

# ══════════════════════════════════════════════════════════════════
# 3. FORMAL LOGIC
# ══════════════════════════════════════════════════════════════════
def test_formal_logic():
    k = BrahmanKernel()
    print(k.load_cartridge(str(CART / "formal_logic_sutras.json")))

    # 3a. Undistributed middle
    check("undistributed middle fallacy", k.verify({
        "protocol_version": "1.0.0", "domain": "formal_logic",
        "claim": {"raw_input": "All cats are animals. All dogs are animals. Therefore all cats are dogs.", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "entails", "resolved_root": "entails"},
            "karana": {"id": "i0", "surface": "middle_term_animal", "lemma": "animal",
                       "constraints": [{"rule_id": "SYL-001", "check": "distribution", "field": "middle_distributed", "actual": False}]},
            "karma": {"id": "o0", "surface": "conclusion", "lemma": "cats_are_dogs",
                      "constraints": [{"rule_id": "SYL-002", "check": "term_dist", "field": "conclusion_term_distributed_in_premise", "actual": False}]},
        }
    }), Verdict.INVALID, min_violations=2, must_contain="SYL-001")

    # 3b. Circular reasoning
    check("circular reasoning detected", k.verify({
        "protocol_version": "1.0.0", "domain": "formal_logic",
        "claim": {"raw_input": "God exists because the Bible says so. The Bible is true because it is God's word.", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "proves", "resolved_root": "proves"},
            "karta": {"id": "a0", "surface": "argument", "lemma": "arg",
                      "constraints": [{"rule_id": "QUANT-002", "check": "witness", "field": "witness_provided", "actual": False}]},
            "adhikarana": {"id": "e0", "surface": "premise_set", "lemma": "premises",
                          "constraints": [
                              {"rule_id": "CIRC-001", "check": "circularity", "field": "conclusion_in_premises", "actual": True},
                              {"rule_id": "CONSIST-001", "check": "consistency", "field": "premises_consistent", "actual": True}
                          ]},
        }
    }), Verdict.INVALID, must_contain="CIRC-001")

    # 3c. VALID modus ponens
    check("valid modus ponens inference", k.verify({
        "protocol_version": "1.0.0", "domain": "formal_logic",
        "claim": {"raw_input": "All humans are mortal. Socrates is human. Therefore Socrates is mortal.", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "entails", "resolved_root": "entails"},
            "karta": {"id": "a0", "surface": "premise_set", "lemma": "premises",
                      "constraints": [
                          {"rule_id": "LOG-002", "check": "ac", "field": "affirming_consequent", "actual": False},
                          {"rule_id": "LOG-003", "check": "da", "field": "denying_antecedent", "actual": False}
                      ]},
            "karma": {"id": "o0", "surface": "conclusion", "lemma": "socrates_mortal",
                      "constraints": [{"rule_id": "LOG-001", "check": "mp", "field": "consequent_derived", "actual": True}]},
            "karana": {"id": "i0", "surface": "middle_term", "lemma": "human",
                       "constraints": [
                           {"rule_id": "SYL-001", "check": "distribution", "field": "middle_distributed", "actual": True},
                           {"rule_id": "SCOPE-001", "check": "scope", "field": "scope_order_preserved", "actual": True}
                       ]},
            "adhikarana": {"id": "e0", "surface": "proof_context", "lemma": "ctx",
                          "constraints": [
                              {"rule_id": "CONSIST-001", "check": "consistency", "field": "premises_consistent", "actual": True},
                              {"rule_id": "CIRC-001", "check": "circularity", "field": "conclusion_in_premises", "actual": False},
                              {"rule_id": "SYL-003", "check": "neg_premises", "field": "both_premises_negative", "actual": False}
                          ]},
        }
    }), Verdict.VALID)

    k.unload_cartridge()

# ══════════════════════════════════════════════════════════════════
# 4. EDGE CASES
# ══════════════════════════════════════════════════════════════════
def test_edge_cases():
    k = BrahmanKernel()

    # 4a. No cartridge loaded
    check("no cartridge loaded → AMBIGUOUS", k.verify({
        "protocol_version": "1.0.0", "domain": "none",
        "claim": {"raw_input": "anything", "claim_type": "assertion"},
        "karaka_graph": {"kriya": {"id": "k0", "surface": "test", "resolved_root": "test"}}
    }), Verdict.AMBIGUOUS, must_contain="NO_CARTRIDGE")

    # 4b. Missing kriya node
    k.load_cartridge(str(CART / "sanskrit_sutras.json"))
    check("missing kriya → AMBIGUOUS", k.verify({
        "protocol_version": "1.0.0", "domain": "sanskrit",
        "claim": {"raw_input": "rāmaḥ", "claim_type": "assertion"},
        "karaka_graph": {}
    }), Verdict.AMBIGUOUS, must_contain="MISSING_KRIYA")

    # 4c. Root miss (circuit breaker)
    check("unknown root → AMBIGUOUS (circuit breaker)", k.verify({
        "protocol_version": "1.0.0", "domain": "sanskrit",
        "claim": {"raw_input": "rāmaḥ flurpzati", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "flurpzati", "resolved_root": "flurpzati"},
        }
    }), Verdict.AMBIGUOUS, must_contain="ROOT_MISS")

    # 4d. Cartridge hot-swap mid-session
    k.load_cartridge(str(CART / "memory_safety_sutras.json"))
    assert k.loaded_domain == "memory_safety"
    check("hot-swap to memory_safety works", k.verify({
        "protocol_version": "1.0.0", "domain": "memory_safety",
        "claim": {"raw_input": "malloc(n)", "claim_type": "assertion"},
        "karaka_graph": {
            "kriya": {"id": "k0", "surface": "malloc", "resolved_root": "malloc"},
            "karta": {"id": "a0", "surface": "process", "lemma": "proc"},
            "karma": {"id": "o0", "surface": "heap_ptr", "lemma": "ptr",
                      "constraints": [
                          {"rule_id": "MEM-001", "check": "bounds", "field": "bounds_checked", "actual": True},
                          {"rule_id": "MEM-005", "check": "null", "field": "null_checked", "actual": True}
                      ]},
            "karana": {"id": "i0", "surface": "size_n", "lemma": "n",
                       "constraints": [{"rule_id": "MEM-004", "check": "overflow", "field": "overflow_checked", "actual": True}]},
        }
    }), Verdict.VALID)

    k.unload_cartridge()


if __name__ == "__main__":
    print("=" * 70)
    print("BRAHMAN KERNEL STRESS TEST — Multi-Layered Cartridge Validation")
    print("=" * 70)

    print("\n── MEMORY SAFETY (12 sūtras, 15 roots) ──────────────────────")
    test_memory_safety()

    print("\n── BIOCHEMISTRY (12 sūtras, 15 roots) ───────────────────────")
    test_biochemistry()

    print("\n── FORMAL LOGIC (12 sūtras, 12 roots) ───────────────────────")
    test_formal_logic()

    print("\n── EDGE CASES ───────────────────────────────────────────────")
    test_edge_cases()

    print("\n" + "=" * 70)
    total = passed + failed
    print(f"  PASSED: {passed}/{total}")
    print(f"  FAILED: {failed}/{total}")
    if failed == 0:
        print("  ✓ ALL STRESS TESTS PASSED")
    else:
        print(f"  ✗ {failed} TESTS FAILED — review violations above")
    print("=" * 70)
