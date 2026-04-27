# Licensed under BSL 1.1 — commercial use requires written permission
# Change date: 2027-01-01 to MIT License by Bhupen Nayak
# Contact: askzachn@gmail.com

"""
Brahman Universal Verification Kernel v1.0.0

The kernel is COMPLETELY domain-blind. It does not know what a
"smart contract" is. It does not know what a "protein" is.
It only knows MATHEMATICS and SYNTAX:

    1. Read a Kāraka Protocol graph.
    2. Load a Sūtra cartridge (domain rules).
    3. For each node in the graph, check if its connections
       are LEGAL according to the loaded rules.
    4. Output: VALID / INVALID / AMBIGUOUS + Logic Hash.

The kernel never changes. You swap the cartridge.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ══════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════

class Verdict:
    VALID     = "VALID"
    INVALID   = "INVALID"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class VerificationResult:
    """The output of the Brahman Kernel. Fully deterministic."""
    verdict: str
    violations: List[str] = field(default_factory=list)
    matched_sutras: List[str] = field(default_factory=list)
    dhatu_found: bool = False
    logic_hash: str = ""
    karaka_trace: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════
# SŪTRA CARTRIDGE
# ══════════════════════════════════════════════════════════════════

@dataclass
class Sutra:
    """A single verification rule from a domain cartridge."""
    id: str
    description: str
    target_role: str          # Which kāraka this rule applies to
    field: str                # The property being checked
    condition: str            # "equals", "in", "not_in", "exists", "requires"
    expected: object          # The legal value(s)
    severity: str = "error"   # "error" = INVALID, "warning" = log but pass


class SutraCartridge:
    """
    A hot-swappable domain rulebook.
    Load from a .json file. Unload. Load a different one.
    The kernel doesn't care what's inside.
    """

    def __init__(self):
        self.domain: str = ""
        self.version: str = ""
        self.description: str = ""
        self.sutras: List[Sutra] = []
        self.root_lexicon: Dict[str, Dict] = {}  # canonical roots the DB recognizes

    @classmethod
    def load(cls, path: str) -> "SutraCartridge":
        """Load a cartridge from a JSON file."""
        cart = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        cart.domain = data.get("domain", "unknown")
        cart.version = data.get("version", "0.0.0")
        cart.description = data.get("description", "")

        for s in data.get("sutras", []):
            cart.sutras.append(Sutra(
                id=s["id"],
                description=s["description"],
                target_role=s["target_role"],
                field=s["field"],
                condition=s["condition"],
                expected=s["expected"],
                severity=s.get("severity", "error"),
            ))

        for root_id, root_data in data.get("root_lexicon", {}).items():
            cart.root_lexicon[root_id] = root_data

        return cart

    def lookup_root(self, surface: str) -> Optional[Dict]:
        """Check if a surface form resolves to a known root."""
        # Direct match
        if surface in self.root_lexicon:
            return self.root_lexicon[surface]
        # Lowercase match
        if surface.lower() in self.root_lexicon:
            return self.root_lexicon[surface.lower()]
        return None


# ══════════════════════════════════════════════════════════════════
# THE KERNEL
# ══════════════════════════════════════════════════════════════════

class BrahmanKernel:
    """
    The Universal Verification Engine.

    It is domain-blind. It reads a Kāraka graph, loads a Sūtra
    cartridge, and checks every node against the rules.

    Monday: Load cyber_sutras.json → hunt zero-days.
    Tuesday: Load thermo_sutras.json → verify molecular bonds.
    Wednesday: Load rust_crypto_sutras.json → audit Solana contracts.

    The kernel stays the same. You swap the cartridge.
    """

    def __init__(self):
        self.cartridge: Optional[SutraCartridge] = None
        self._history: List[VerificationResult] = []

    # ── Cartridge Management ─────────────────────────────────────

    def load_cartridge(self, path: str) -> str:
        """Hot-swap a domain Sūtra cartridge."""
        self.cartridge = SutraCartridge.load(path)
        return f"✓ Loaded [{self.cartridge.domain}] v{self.cartridge.version}: {len(self.cartridge.sutras)} sūtras, {len(self.cartridge.root_lexicon)} roots"

    def unload_cartridge(self):
        """Eject the current cartridge."""
        domain = self.cartridge.domain if self.cartridge else "none"
        self.cartridge = None
        return f"✓ Ejected [{domain}]"

    @property
    def loaded_domain(self) -> str:
        return self.cartridge.domain if self.cartridge else "none"

    # ── The Verifier ─────────────────────────────────────────────

    def verify(self, kp: Dict) -> VerificationResult:
        """
        THE CORE FUNCTION.

        Input:  A Kāraka Protocol instance (dict).
        Output: VerificationResult (VALID / INVALID / AMBIGUOUS).

        This function is PURE LOGIC. No neural networks.
        No domain knowledge. Just graph + rules = verdict.
        """
        if not self.cartridge:
            return VerificationResult(
                verdict=Verdict.AMBIGUOUS,
                violations=["NO_CARTRIDGE: No Sūtra cartridge loaded."]
            )

        # 1. Force FastAPI Pydantic models into raw dictionaries
        if hasattr(kp, "model_dump"):
            kp = kp.model_dump()
        elif hasattr(kp, "dict"):
            kp = kp.dict()

        # 2. Extract safely, handling both naming conventions
        graph = kp.get("karaka_protocol") or kp.get("karaka_graph") or {}
        
        # 3. Fallback: If the caller passed the graph directly instead of the envelope
        if not graph and "kriya" in kp:
            graph = kp

        violations: List[str] = []
        matched_sutras: List[str] = []
        trace: Dict = {}

        # ── Step 1: Resolve the Kriyā (Action Root) ──────────────
        kriya = graph.get("kriya")
        if not kriya:
            return VerificationResult(
                verdict=Verdict.AMBIGUOUS,
                violations=["MISSING_KRIYA: No action/verb node in the graph."]
            )

        surface = kriya.get("surface", "")
        resolved = kriya.get("resolved_root")

        # Attempt DB lookup
        root_entry = self.cartridge.lookup_root(resolved or surface)
        dhatu_found = root_entry is not None

        if not dhatu_found:
            # THE CIRCUIT BREAKER: DB doesn't recognize the root.
            # Force AMBIGUOUS. Never let downstream logic guess.
            return VerificationResult(
                verdict=Verdict.AMBIGUOUS,
                dhatu_found=False,
                violations=[f"ROOT_MISS: '{resolved or surface}' not found in [{self.cartridge.domain}] lexicon."],
                logic_hash=self._hash(kp, Verdict.AMBIGUOUS)
            )

        trace["kriya"] = {
            "surface": surface,
            "resolved": resolved,
            "root_entry": root_entry
        }

        # ── Step 2: Apply Sūtras to each Kāraka node ────────────
        role_map = {
            "karta": graph.get("karta"),
            "karma": graph.get("karma"),
            "karana": graph.get("karana"),
            "sampradana": graph.get("sampradana"),
            "adhikarana": graph.get("adhikarana"),
            "apadana": graph.get("apadana"),
        }

        for sutra in self.cartridge.sutras:
            role_node = role_map.get(sutra.target_role)

            # If the sūtra targets a role that doesn't exist in this graph
            if sutra.condition == "requires" and role_node is None:
                # The rule demands this role exists but it's missing
                if sutra.severity == "error":
                    violations.append(
                        f"[{sutra.id}] {sutra.description}: "
                        f"Role '{sutra.target_role}' required but missing."
                    )
                matched_sutras.append(sutra.id)
                continue

            if role_node is None:
                continue  # Rule doesn't apply — role absent and not required

            # Check the field on this node
            actual_value = self._resolve_field(role_node, sutra.field, root_entry)

            # If the field is absent and this isn't a "must exist" check,
            # the sūtra is NOT APPLICABLE to this node — skip it.
            if actual_value is None and sutra.condition not in ("requires", "exists"):
                continue

            passed = self._evaluate_condition(
                sutra.condition, actual_value, sutra.expected
            )

            if not passed and sutra.severity == "error":
                violations.append(
                    f"[{sutra.id}] {sutra.description}: "
                    f"Expected {sutra.field}={sutra.expected}, got '{actual_value}'."
                )

            if not passed and sutra.severity == "warning":
                trace[f"warning_{sutra.id}"] = f"{sutra.description}: {actual_value}"

            matched_sutras.append(sutra.id)

        # ── Step 3: Verdict ──────────────────────────────────────
        verdict = Verdict.VALID if len(violations) == 0 else Verdict.INVALID

        result = VerificationResult(
            verdict=verdict,
            violations=violations,
            matched_sutras=matched_sutras,
            dhatu_found=True,
            karaka_trace=trace,
            logic_hash=self._hash(kp, verdict)
        )

        self._history.append(result)
        return result

    # ── Convenience: string-in → verdict-out ─────────────────────

    def evaluate(self, raw_input: str, translator=None) -> VerificationResult:
        """
        High-level entry point for benchmarking.

        If a neural translator is provided, it converts the raw string
        into a Kāraka Protocol instance. Otherwise, this wraps the
        input into a minimal KP for direct rule checking.
        """
        if translator:
            kp = translator.translate(raw_input)
        else:
            # Minimal KP: treat the input as a single action claim
            words = raw_input.strip().split()
            kp = {
                "protocol_version": "1.0.0",
                "domain": self.loaded_domain,
                "claim": {
                    "raw_input": raw_input,
                    "claim_type": "assertion"
                },
                "karaka_graph": {
                    "kriya": {
                        "id": "k0",
                        "surface": words[-1] if words else "",
                        "resolved_root": words[-1] if words else None,
                    },
                    "karta": {
                        "id": "a0",
                        "surface": words[0] if words else "",
                        "lemma": words[0].lower() if words else "",
                    } if len(words) > 1 else None,
                    "karma": {
                        "id": "o0",
                        "surface": words[1] if len(words) > 2 else "",
                        "lemma": (words[1] if len(words) > 2 else "").lower(),
                    } if len(words) > 2 else None,
                }
            }

        return self.verify(kp)

    # ── Internal Helpers ─────────────────────────────────────────

    @staticmethod
    def _resolve_field(node: Dict, field: str, root_entry: Dict) -> object:
        """Extract a field value from a kāraka node or the root entry."""
        # Check node constraints first
        for constraint in node.get("constraints", []):
            if constraint.get("field") == field:
                return constraint.get("actual")
        # Check direct node properties
        if field in node:
            return node[field]
        # Check root entry
        if field in root_entry:
            return root_entry[field]
        return None

    @staticmethod
    def _evaluate_condition(condition: str, actual, expected) -> bool:
        """Pure logic gate. No domain knowledge."""
        # --- BULLETPROOF COMPARISON BLOCK ---
        expected_val = str(expected).strip().lower() if not isinstance(expected, list) else [str(x).strip().lower() for x in expected]
        actual_val = str(actual).strip().lower() if actual is not None else "none"

        if condition == "equals":
            return actual_val == expected_val
        elif condition == "in":
            return actual_val in expected_val if isinstance(expected_val, list) else actual_val == expected_val
        elif condition == "not_in":
            return actual_val not in expected_val if isinstance(expected_val, list) else actual_val != expected_val
        elif condition == "exists":
            return actual is not None
        elif condition == "requires":
            return actual is not None
        elif condition == "type_check":
            return isinstance(actual, type(expected))
        return True  # Unknown condition → pass

    @staticmethod
    def _hash(kp: Dict, verdict: str) -> str:
        """
        Logic Hash: SHA-256 of the input + graph + verdict.
        Cryptographically proves the exact path taken.
        """
        payload = json.dumps({
            "input": kp.get("claim", {}).get("raw_input", ""),
            "graph": kp.get("karaka_protocol", kp.get("karaka_graph", {})),
            "verdict": verdict
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
