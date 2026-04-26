# Licensed under BSL 1.1 — commercial use requires written permission
# Change date: 2027-01-01 to MIT License by Bhupen Nayak
# Contact: askzachn@gmail.com

"""
Brahman Validator Client — Off-Chain Transaction Verification Daemon

The validator client intercepts Solana transactions, deserializes them
into Kāraka Protocol graphs, runs them through the Brahman Kernel, and
submits signed verdicts to the mesh consensus layer.

Modes:
  1. STANDALONE: Single node, local verification only
  2. MESH: Fan-out to peer nodes, collect verdicts, reach consensus
  3. MONITOR: Watch-only — log verdicts without submitting on-chain
"""

import sys
import json
import time
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from brahman_kernel import BrahmanKernel, Verdict, VerificationResult
from tx_deserializer import TransactionDeserializer
from mesh_consensus import MeshConsensus, PeerNode
from verification_protocol import (
    VerificationRecord, NodeVerdict, OnChainVerdict,
    QuorumStatus,
)


# ══════════════════════════════════════════════════════════════════
# VALIDATOR CONFIGURATION
# ══════════════════════════════════════════════════════════════════

@dataclass
class ValidatorConfig:
    """Configuration for the validator client."""
    node_id: str = ""
    pubkey: str = ""
    cartridge_path: str = ""
    cartridge_domain: str = "rust_crypto"
    mode: str = "standalone"        # standalone | mesh | monitor
    rpc_endpoint: str = "http://localhost:8899"
    ws_endpoint: str = "ws://localhost:8900"
    peer_endpoints: List[str] = field(default_factory=list)
    auto_slash: bool = True         # Auto-slash nodes that disagree
    log_verdicts: bool = True


# ══════════════════════════════════════════════════════════════════
# THE VALIDATOR
# ══════════════════════════════════════════════════════════════════

class BrahmanValidator:
    """
    Off-chain validator that intercepts and verifies Solana transactions
    using the Brahman Kernel + Sūtra cartridges.

    For every transaction:
      1. Deserialize TX → Kāraka Protocol graph
      2. Verify KP against loaded cartridge
      3. Produce signed Verdict + LogicHash
      4. Submit to mesh consensus (if in mesh mode)
      5. Record on-chain (if quorum reached)
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.kernel = BrahmanKernel()
        self.deserializer = TransactionDeserializer()
        self.consensus = MeshConsensus()
        self._verified: List[Dict] = []
        self._started = False

        # Generate node ID if not set
        if not config.node_id:
            config.node_id = f"validator-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"

        if not config.pubkey:
            config.pubkey = hashlib.sha256(config.node_id.encode()).hexdigest()

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> str:
        """Initialize the validator."""
        # Load cartridge
        if self.config.cartridge_path:
            msg = self.kernel.load_cartridge(self.config.cartridge_path)
        else:
            # Default to rust_crypto for Solana validation
            default = Path(__file__).parent.parent / "cartridges" / "rust_crypto_sutras.json"
            msg = self.kernel.load_cartridge(str(default))

        # Register self as a peer
        self.consensus.register_peer(
            node_id=self.config.node_id,
            endpoint=f"local://{self.config.node_id}",
            pubkey=self.config.pubkey,
        )

        # Register known peers
        for i, endpoint in enumerate(self.config.peer_endpoints):
            peer_id = f"peer-{i}"
            self.consensus.register_peer(
                node_id=peer_id,
                endpoint=endpoint,
                pubkey=hashlib.sha256(peer_id.encode()).hexdigest(),
            )

        self._started = True
        return f"[{self.config.node_id}] Validator online — {msg} — Mode: {self.config.mode}"

    # ── Core: Verify a Transaction ───────────────────────────────

    def verify_transaction(self, tx_data: Dict) -> Dict:
        """
        Verify a single transaction through the full pipeline.

        Args:
            tx_data: Raw transaction data (from RPC or simulated)

        Returns:
            Verification result with verdict, logic_hash, and consensus info.
        """
        if not self._started:
            self.start()

        start_time = time.time()
        tx_hash = tx_data.get("signature", hashlib.sha256(
            json.dumps(tx_data, sort_keys=True).encode()
        ).hexdigest())

        # Step 1: Deserialize TX → KP
        kp = self.deserializer.deserialize(tx_data)

        # Step 2: Run through Brahman Kernel
        result = self.kernel.verify(kp)

        # Step 3: Compute cartridge hash for consensus verification
        cartridge_hash = ""
        if self.kernel.cartridge:
            cart_data = json.dumps({
                "domain": self.kernel.cartridge.domain,
                "version": self.kernel.cartridge.version,
                "sutras": len(self.kernel.cartridge.sutras),
            }, sort_keys=True)
            cartridge_hash = hashlib.sha256(cart_data.encode()).hexdigest()

        elapsed_ms = (time.time() - start_time) * 1000

        # Build the verification report
        report = {
            "tx_hash": tx_hash,
            "node_id": self.config.node_id,
            "verdict": result.verdict,
            "violations": result.violations,
            "matched_sutras": result.matched_sutras,
            "logic_hash": result.logic_hash,
            "dhatu_found": result.dhatu_found,
            "cartridge_domain": self.kernel.loaded_domain,
            "cartridge_hash": cartridge_hash,
            "processing_time_ms": round(elapsed_ms, 2),
            "timestamp": time.time(),
            "mode": self.config.mode,
        }

        # Step 4: Submit to consensus (if in mesh mode)
        if self.config.mode == "mesh":
            consensus_result = self.consensus.submit_verdict(
                tx_hash=tx_hash,
                node_id=self.config.node_id,
                verdict=result.verdict,
                logic_hash=result.logic_hash,
                cartridge_domain=self.kernel.loaded_domain,
                cartridge_hash=cartridge_hash,
            )
            report["consensus"] = consensus_result

        self._verified.append(report)
        return report

    # ── Batch Verification ───────────────────────────────────────

    def verify_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Verify a batch of transactions."""
        return [self.verify_transaction(tx) for tx in transactions]

    # ── Mesh Consensus Simulation ────────────────────────────────

    def simulate_mesh_verification(self, tx_data: Dict,
                                   peer_results: Dict[str, Dict]) -> Dict:
        """
        Simulate a full mesh verification with multiple nodes.

        Args:
            tx_data: The transaction to verify
            peer_results: {node_id: {"verdict": "...", "logic_hash": "..."}}

        Returns:
            Full consensus result.
        """
        # This node's verification
        my_report = self.verify_transaction(tx_data)
        tx_hash = my_report["tx_hash"]

        # Initiate consensus
        kp = self.deserializer.deserialize(tx_data)
        if tx_hash not in self.consensus.pending:
            self.consensus.initiate_verification(tx_hash, self.kernel.loaded_domain)

        # Submit this node's verdict
        self.consensus.submit_verdict(
            tx_hash=tx_hash,
            node_id=self.config.node_id,
            verdict=my_report["verdict"],
            logic_hash=my_report["logic_hash"],
            cartridge_domain=my_report["cartridge_domain"],
            cartridge_hash=my_report["cartridge_hash"],
        )

        # Submit peer verdicts
        for node_id, result in peer_results.items():
            self.consensus.submit_verdict(
                tx_hash=tx_hash,
                node_id=node_id,
                verdict=result["verdict"],
                logic_hash=result["logic_hash"],
                cartridge_domain=result.get("domain", ""),
                cartridge_hash=result.get("cartridge_hash", ""),
            )

        return {
            "my_verdict": my_report,
            "consensus": self.consensus.get_status(tx_hash),
        }

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get validator statistics."""
        verdict_counts = {}
        for r in self._verified:
            v = r["verdict"]
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        return {
            "node_id": self.config.node_id,
            "mode": self.config.mode,
            "loaded_domain": self.kernel.loaded_domain,
            "total_verified": len(self._verified),
            "verdicts": verdict_counts,
            "mesh_stats": self.consensus.get_mesh_stats(),
        }
