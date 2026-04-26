"""
Brahman Mesh Consensus Engine — Deterministic Hash-Based Quorum

Coordinates multiple Sovereign Nodes to reach consensus on transaction
verification. Because the Brahman Kernel is deterministic (same input +
same cartridge = same logic_hash), consensus reduces to hash comparison.

Architecture:
    1. A verification request arrives at the coordinator
    2. The coordinator fans out the KP graph to N peer nodes
    3. Each node runs the kernel independently
    4. Nodes return their verdict + logic_hash
    5. If 2/3 produce the SAME hash → consensus reached
    6. The finalized verdict is submitted on-chain

Anti-Spoofing:
    A malicious node cannot forge a logic_hash because it is a SHA-256
    of (input + graph + verdict) — the entire sūtra traversal is baked in.
    To produce the same hash, you must run the same deterministic proof.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from verification_protocol import (
    VerificationRecord, NodeVerdict, OnChainVerdict,
    QuorumStatus, QUORUM_THRESHOLD, MIN_QUORUM_NODES, VERDICT_TIMEOUT,
)


# ══════════════════════════════════════════════════════════════════
# PEER NODE
# ══════════════════════════════════════════════════════════════════

@dataclass
class PeerNode:
    """Represents a registered Sovereign Node in the mesh."""
    node_id: str
    endpoint: str           # HTTP/WS endpoint (e.g., "http://localhost:8420")
    pubkey: str             # On-chain public key
    reputation: int = 100   # From on-chain NodeAccount
    is_active: bool = True
    last_seen: float = 0.0
    verdicts_submitted: int = 0
    verdicts_agreed: int = 0  # How many times this node agreed with consensus

    @property
    def agreement_rate(self) -> float:
        if self.verdicts_submitted == 0:
            return 1.0
        return self.verdicts_agreed / self.verdicts_submitted


# ══════════════════════════════════════════════════════════════════
# CONSENSUS ENGINE
# ══════════════════════════════════════════════════════════════════

class MeshConsensus:
    """
    Coordinates verification consensus across the Sovereign Node mesh.

    The engine is TRUSTLESS: it does not trust any individual node's
    verdict. It only trusts the MATHEMATICAL AGREEMENT of the hashes.
    """

    def __init__(self, min_quorum: int = MIN_QUORUM_NODES,
                 threshold: float = QUORUM_THRESHOLD,
                 timeout: float = VERDICT_TIMEOUT):
        self.peers: Dict[str, PeerNode] = {}
        self.pending: Dict[str, VerificationRecord] = {}  # tx_hash → record
        self.finalized: Dict[str, VerificationRecord] = {}
        self.min_quorum = min_quorum
        self.threshold = threshold
        self.timeout = timeout

    # ── Peer Management ──────────────────────────────────────────

    def register_peer(self, node_id: str, endpoint: str, pubkey: str,
                      reputation: int = 100) -> PeerNode:
        """Register a Sovereign Node as a mesh peer."""
        peer = PeerNode(
            node_id=node_id,
            endpoint=endpoint,
            pubkey=pubkey,
            reputation=reputation,
            last_seen=time.time(),
        )
        self.peers[node_id] = peer
        return peer

    def deregister_peer(self, node_id: str):
        """Remove a peer from the mesh."""
        self.peers.pop(node_id, None)

    @property
    def active_peers(self) -> List[PeerNode]:
        return [p for p in self.peers.values() if p.is_active]

    @property
    def quorum_possible(self) -> bool:
        return len(self.active_peers) >= self.min_quorum

    # ── Verification Flow ────────────────────────────────────────

    def initiate_verification(self, tx_hash: str,
                              cartridge_domain: str = "") -> VerificationRecord:
        """
        Create a new verification record for a transaction.
        Returns the record (in COLLECTING state).
        """
        record = VerificationRecord(
            tx_hash=tx_hash,
            created_at=time.time(),
            required_nodes=max(self.min_quorum, len(self.active_peers)),
            cartridge_domain=cartridge_domain,
        )
        self.pending[tx_hash] = record
        return record

    def submit_verdict(self, tx_hash: str, node_id: str,
                       verdict: str, logic_hash: str,
                       cartridge_domain: str = "",
                       cartridge_hash: str = "") -> Dict:
        """
        Submit a single node's verdict for a pending transaction.

        Returns:
            dict with 'quorum_reached', 'status', and 'record' fields.
        """
        if tx_hash not in self.pending:
            return {"error": "No pending verification for this tx_hash",
                    "quorum_reached": False}

        record = self.pending[tx_hash]

        if record.is_finalized():
            return {"error": "Verification already finalized",
                    "quorum_reached": True,
                    "record": record.to_dict()}

        peer = self.peers.get(node_id)
        if not peer:
            return {"error": f"Unknown node: {node_id}",
                    "quorum_reached": False}

        # Build the vote
        vote = NodeVerdict(
            node_pubkey=peer.pubkey,
            verdict=OnChainVerdict.from_kernel(verdict),
            logic_hash=logic_hash,
            timestamp=time.time(),
            cartridge_hash=cartridge_hash,
            cartridge_domain=cartridge_domain,
        )

        # Submit and check for quorum
        quorum_reached = record.submit_vote(vote)
        peer.verdicts_submitted += 1
        peer.last_seen = time.time()

        if quorum_reached:
            self._finalize(tx_hash, record)
            # Update agreement tracking for all peers
            for v in record.verdicts:
                for p in self.peers.values():
                    if p.pubkey == v.node_pubkey:
                        if v.logic_hash == record.final_logic_hash:
                            p.verdicts_agreed += 1
                        break

        return {
            "quorum_reached": quorum_reached,
            "status": record.quorum_status.name,
            "votes": len(record.verdicts),
            "required": record.required_nodes,
            "record": record.to_dict(),
        }

    def _finalize(self, tx_hash: str, record: VerificationRecord):
        """Move a verification from pending to finalized."""
        record.quorum_status = QuorumStatus.FINALIZED
        self.finalized[tx_hash] = record
        self.pending.pop(tx_hash, None)

    # ── Consensus Queries ────────────────────────────────────────

    def get_status(self, tx_hash: str) -> Optional[Dict]:
        """Get the current status of a verification."""
        if tx_hash in self.finalized:
            return self.finalized[tx_hash].to_dict()
        if tx_hash in self.pending:
            return self.pending[tx_hash].to_dict()
        return None

    def check_timeouts(self) -> List[str]:
        """Check for timed-out pending verifications."""
        timed_out = []
        for tx_hash, record in list(self.pending.items()):
            if record.check_timeout():
                self._finalize(tx_hash, record)
                timed_out.append(tx_hash)
        return timed_out

    def get_mesh_stats(self) -> Dict:
        """Get overall mesh statistics."""
        return {
            "total_peers": len(self.peers),
            "active_peers": len(self.active_peers),
            "quorum_possible": self.quorum_possible,
            "pending_verifications": len(self.pending),
            "finalized_verifications": len(self.finalized),
            "min_quorum": self.min_quorum,
            "threshold": self.threshold,
            "peers": {
                pid: {
                    "node_id": p.node_id,
                    "endpoint": p.endpoint,
                    "reputation": p.reputation,
                    "agreement_rate": round(p.agreement_rate, 3),
                    "verdicts": p.verdicts_submitted,
                }
                for pid, p in self.peers.items()
            },
        }

    # ── Fan-Out Verification (Simulated) ─────────────────────────

    def simulate_fanout(self, tx_hash: str, kp: Dict,
                        node_results: Dict[str, Dict]) -> Dict:
        """
        Simulate fanning out a KP to all peers and collecting results.

        Args:
            tx_hash: Transaction hash
            kp: The Kāraka Protocol instance
            node_results: {node_id: {"verdict": "...", "logic_hash": "..."}}

        Returns:
            Final consensus result.
        """
        record = self.initiate_verification(
            tx_hash,
            cartridge_domain=kp.get("domain", ""),
        )

        for node_id, result in node_results.items():
            self.submit_verdict(
                tx_hash=tx_hash,
                node_id=node_id,
                verdict=result["verdict"],
                logic_hash=result["logic_hash"],
                cartridge_domain=result.get("domain", ""),
                cartridge_hash=result.get("cartridge_hash", ""),
            )

        return self.get_status(tx_hash)
