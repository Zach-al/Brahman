"""
Brahman On-Chain Verification Protocol — Anchor-compatible data structures

Defines the on-chain state and instruction logic for recording Brahman
verdicts on the Solana blockchain, expressed as Python dataclasses that
mirror the Rust/Anchor structs. This module serves as both:

  1. The specification for the on-chain Anchor program
  2. The client-side serialization layer for the Python validator

On-Chain Flow:
  1. Sovereign Node verifies a transaction → Verdict + LogicHash
  2. Node calls submit_verdict(tx_hash, verdict, logic_hash, domain)
  3. VerificationRecord PDA accumulates votes
  4. When quorum is reached → finalize_verification()
  5. Final verdict is written immutably on-chain

Quorum Rule:
  Because the Brahman Kernel is DETERMINISTIC, consensus reduces to
  hash comparison. 2/3 of nodes must produce the SAME logic_hash.
  If they don't, the transaction is flagged DISPUTED.
"""

import hashlib
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from enum import IntEnum


# ══════════════════════════════════════════════════════════════════
# ON-CHAIN CONSTANTS
# ══════════════════════════════════════════════════════════════════

# PDA seeds for on-chain accounts
VERIFICATION_SEED = b"brahman-verdict"
QUORUM_SEED = b"brahman-quorum"

# Minimum number of validators for a quorum
MIN_QUORUM_NODES = 3

# Quorum threshold (2/3 supermajority)
QUORUM_THRESHOLD = 2 / 3

# Verdict timeout (seconds) — auto-reject if quorum not reached
VERDICT_TIMEOUT = 30


# ══════════════════════════════════════════════════════════════════
# ON-CHAIN ENUMS
# ══════════════════════════════════════════════════════════════════

class OnChainVerdict(IntEnum):
    """Maps to the Anchor enum. Uses u8 discriminators."""
    PENDING   = 0
    VALID     = 1
    INVALID   = 2
    AMBIGUOUS = 3
    DISPUTED  = 4  # Quorum could not agree

    @classmethod
    def from_kernel(cls, verdict_str: str) -> "OnChainVerdict":
        return {
            "VALID":     cls.VALID,
            "INVALID":   cls.INVALID,
            "AMBIGUOUS": cls.AMBIGUOUS,
        }.get(verdict_str, cls.AMBIGUOUS)


class QuorumStatus(IntEnum):
    """Lifecycle of a verification quorum."""
    COLLECTING = 0  # Still accepting votes
    REACHED    = 1  # 2/3 agree
    DISPUTED   = 2  # Cannot reach agreement
    TIMED_OUT  = 3  # Deadline passed without quorum
    FINALIZED  = 4  # Written to chain, immutable


# ══════════════════════════════════════════════════════════════════
# ON-CHAIN ACCOUNT STRUCTS
# ══════════════════════════════════════════════════════════════════

@dataclass
class NodeVerdict:
    """
    A single node's vote on a transaction.
    Mirrors the Anchor NodeVerdict struct.

    On-chain space: 32 + 1 + 32 + 8 + 32 + 4+20 = 129 bytes
    """
    node_pubkey: str              # 32 bytes — the voting node's public key
    verdict: OnChainVerdict       # 1 byte
    logic_hash: str               # 32 bytes — SHA-256 of (input+graph+verdict)
    timestamp: float              # 8 bytes — Unix timestamp
    cartridge_hash: str           # 32 bytes — SHA-256 of the loaded cartridge
    cartridge_domain: str         # 4+20 bytes — domain string

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = int(self.verdict)
        return d


@dataclass
class VerificationRecord:
    """
    On-chain PDA that accumulates votes for a single transaction.
    Seeds: [VERIFICATION_SEED, tx_hash]

    On-chain space: 8 + 32 + 1 + 32 + 1 + 8 + 8 + (N * 129) + 4+20
    """
    tx_hash: str                              # Transaction being verified
    verdicts: List[NodeVerdict] = field(default_factory=list)
    quorum_status: QuorumStatus = QuorumStatus.COLLECTING
    final_verdict: OnChainVerdict = OnChainVerdict.PENDING
    final_logic_hash: str = ""
    created_at: float = 0.0
    finalized_at: float = 0.0
    required_nodes: int = MIN_QUORUM_NODES
    cartridge_domain: str = ""

    def submit_vote(self, vote: NodeVerdict) -> bool:
        """
        Submit a node's verdict. Returns True if this vote triggered quorum.
        """
        # Reject duplicate votes from the same node
        for existing in self.verdicts:
            if existing.node_pubkey == vote.node_pubkey:
                return False

        self.verdicts.append(vote)

        # Check if quorum is reached
        if len(self.verdicts) >= self.required_nodes:
            return self._check_quorum()
        return False

    def _check_quorum(self) -> bool:
        """
        Check if 2/3 of nodes agree on the same logic_hash.
        Because the kernel is deterministic, this is a hash comparison.
        """
        hash_counts: Dict[str, int] = {}
        for v in self.verdicts:
            hash_counts[v.logic_hash] = hash_counts.get(v.logic_hash, 0) + 1

        total = len(self.verdicts)
        for logic_hash, count in hash_counts.items():
            if count / total >= QUORUM_THRESHOLD:
                # QUORUM REACHED — all agreeing nodes produced the same proof
                self.quorum_status = QuorumStatus.REACHED
                self.final_logic_hash = logic_hash

                # The verdict is whatever the agreeing nodes said
                for v in self.verdicts:
                    if v.logic_hash == logic_hash:
                        self.final_verdict = v.verdict
                        break

                self.finalized_at = time.time()
                return True

        # No hash reached threshold — DISPUTED
        if len(self.verdicts) >= self.required_nodes:
            self.quorum_status = QuorumStatus.DISPUTED
            self.final_verdict = OnChainVerdict.DISPUTED

        return False

    def check_timeout(self) -> bool:
        """Auto-timeout if quorum not reached within deadline."""
        if self.quorum_status == QuorumStatus.COLLECTING:
            if time.time() - self.created_at > VERDICT_TIMEOUT:
                self.quorum_status = QuorumStatus.TIMED_OUT
                self.final_verdict = OnChainVerdict.DISPUTED
                return True
        return False

    def is_finalized(self) -> bool:
        return self.quorum_status in (
            QuorumStatus.REACHED,
            QuorumStatus.DISPUTED,
            QuorumStatus.TIMED_OUT,
            QuorumStatus.FINALIZED,
        )

    def to_dict(self) -> dict:
        return {
            "tx_hash": self.tx_hash,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "quorum_status": QuorumStatus(self.quorum_status).name,
            "final_verdict": OnChainVerdict(self.final_verdict).name,
            "final_logic_hash": self.final_logic_hash,
            "created_at": self.created_at,
            "finalized_at": self.finalized_at,
            "required_nodes": self.required_nodes,
            "votes_collected": len(self.verdicts),
            "cartridge_domain": self.cartridge_domain,
        }


# ══════════════════════════════════════════════════════════════════
# ANCHOR PROGRAM IDL (for reference / codegen)
# ══════════════════════════════════════════════════════════════════

ANCHOR_IDL_STUB = {
    "name": "brahman_verification",
    "version": "1.0.0",
    "instructions": [
        {
            "name": "submit_verdict",
            "args": [
                {"name": "tx_hash", "type": "bytes32"},
                {"name": "verdict", "type": "u8"},
                {"name": "logic_hash", "type": "bytes32"},
                {"name": "cartridge_domain", "type": "string"},
                {"name": "cartridge_hash", "type": "bytes32"},
            ],
            "accounts": [
                {"name": "node_signer", "isMut": False, "isSigner": True},
                {"name": "node_account", "isMut": False, "isSigner": False},
                {"name": "verification_record", "isMut": True, "isSigner": False},
                {"name": "system_program", "isMut": False, "isSigner": False},
            ]
        },
        {
            "name": "finalize_verification",
            "args": [{"name": "tx_hash", "type": "bytes32"}],
            "accounts": [
                {"name": "authority", "isMut": False, "isSigner": True},
                {"name": "verification_record", "isMut": True, "isSigner": False},
            ]
        },
    ],
    "accounts": [
        {
            "name": "VerificationRecord",
            "fields": [
                {"name": "tx_hash", "type": "bytes32"},
                {"name": "quorum_status", "type": "u8"},
                {"name": "final_verdict", "type": "u8"},
                {"name": "final_logic_hash", "type": "bytes32"},
                {"name": "created_at", "type": "i64"},
                {"name": "finalized_at", "type": "i64"},
                {"name": "required_nodes", "type": "u8"},
                {"name": "vote_count", "type": "u8"},
                {"name": "cartridge_domain", "type": "string"},
            ]
        }
    ]
}
