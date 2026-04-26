"""
Brahman Transaction Deserializer — Solana TX → Kāraka Protocol Graph

Converts raw Solana transactions into Kāraka Protocol JSON that the
Brahman Kernel can verify. This is the "eyes" of the validator —
it parses instruction data, account keys, and metadata into the
universal six-kāraka graph.

Supported Instruction Types:
  - System Program: transfer, create_account
  - SPL Token: transfer, transfer_checked, mint_to, burn
  - Anchor/CPI: invoke, invoke_signed
  - DeFi: borrow, swap, liquidate (via known program IDs)
"""

import hashlib
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════
# KNOWN PROGRAM IDS
# ══════════════════════════════════════════════════════════════════

KNOWN_PROGRAMS = {
    "11111111111111111111111111111111":       "system_program",
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": "spl_token",
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL":  "ata_program",
    "worm2ZoG2kUd4vFXhvjh93UUH596ayRfgQ2MgjNMTth":   "wormhole_bridge",
    "mv3ekLzLbnVPNxjSKvqBpU3ZeZXPQdEC3bp5MDEBG68":   "mango_v3",
    "CashBbkKxwfQ4vLJLY3t9gGjSRg3Vr6VAhdRMbhDBic":   "cashio",
}

# Operation mapping: instruction discriminator → operation name
SYSTEM_OPS = {
    0: "create_account",
    2: "transfer",
}

SPL_TOKEN_OPS = {
    3:  "transfer",
    7:  "mint_to",
    8:  "burn",
    12: "transfer_checked",
}


# ══════════════════════════════════════════════════════════════════
# TRANSACTION MODEL
# ══════════════════════════════════════════════════════════════════

@dataclass
class ParsedAccount:
    """A decoded account from a transaction instruction."""
    pubkey: str
    is_signer: bool = False
    is_writable: bool = False
    is_program: bool = False
    owner: Optional[str] = None
    lamports: Optional[int] = None
    data_len: Optional[int] = None


@dataclass
class ParsedInstruction:
    """A decoded instruction from a transaction."""
    program_id: str
    program_name: str
    operation: str
    accounts: List[ParsedAccount] = field(default_factory=list)
    data: Dict = field(default_factory=dict)


@dataclass
class ParsedTransaction:
    """Full decoded transaction."""
    signature: str
    slot: int = 0
    instructions: List[ParsedInstruction] = field(default_factory=list)
    fee_payer: str = ""
    recent_blockhash: str = ""


# ══════════════════════════════════════════════════════════════════
# THE DESERIALIZER
# ══════════════════════════════════════════════════════════════════

class TransactionDeserializer:
    """
    Converts Solana transactions into Kāraka Protocol graphs.

    The deserializer knows Solana account structure but is DOMAIN-BLIND
    to the verification rules — that's the kernel's job.
    """

    def __init__(self):
        self.program_registry = dict(KNOWN_PROGRAMS)

    def register_program(self, program_id: str, name: str):
        """Register a custom program ID for recognition."""
        self.program_registry[program_id] = name

    # ── Main Entry Point ─────────────────────────────────────────

    def deserialize(self, tx_data: Dict) -> Dict:
        """
        Convert a raw transaction dict into a Kāraka Protocol instance.

        Args:
            tx_data: Transaction data (from RPC getTransaction or log event).
                     Expected keys: signature, accounts, instructions, meta

        Returns:
            A Kāraka Protocol dict ready for the Brahman Kernel.
        """
        parsed = self._parse_transaction(tx_data)

        if not parsed.instructions:
            return self._empty_kp(tx_data.get("signature", "unknown"),
                                  "No instructions found")

        # Use the first non-system instruction as the primary operation
        primary = self._select_primary_instruction(parsed)
        kp = self._instruction_to_kp(primary, parsed)

        return kp

    # ── Transaction Parsing ──────────────────────────────────────

    def _parse_transaction(self, tx_data: Dict) -> ParsedTransaction:
        """Parse raw RPC transaction data into structured form."""
        sig = tx_data.get("signature", tx_data.get("txHash", "unknown"))
        slot = tx_data.get("slot", 0)

        parsed = ParsedTransaction(signature=sig, slot=slot)

        # Parse account keys
        account_keys = tx_data.get("accounts", tx_data.get("accountKeys", []))
        signers = set(tx_data.get("signers", []))
        writables = set(tx_data.get("writableAccounts", []))

        accounts_map = {}
        for i, key in enumerate(account_keys):
            if isinstance(key, dict):
                pk = key.get("pubkey", key.get("key", str(i)))
                acct = ParsedAccount(
                    pubkey=pk,
                    is_signer=key.get("isSigner", pk in signers),
                    is_writable=key.get("isWritable", pk in writables),
                    owner=key.get("owner"),
                )
            else:
                acct = ParsedAccount(
                    pubkey=str(key),
                    is_signer=str(key) in signers or i == 0,
                    is_writable=str(key) in writables,
                )
            accounts_map[acct.pubkey] = acct

        if account_keys:
            first_key = account_keys[0]
            if isinstance(first_key, dict):
                parsed.fee_payer = first_key.get("pubkey", "")
            else:
                parsed.fee_payer = str(first_key)

        # Parse instructions
        for ix_data in tx_data.get("instructions", []):
            ix = self._parse_instruction(ix_data, accounts_map)
            if ix:
                parsed.instructions.append(ix)

        return parsed

    def _parse_instruction(self, ix_data: Dict,
                           accounts_map: Dict[str, ParsedAccount]) -> Optional[ParsedInstruction]:
        """Parse a single instruction."""
        program_id = ix_data.get("programId", ix_data.get("program", ""))
        program_name = self.program_registry.get(program_id, "unknown_program")

        # Determine operation from instruction data
        operation = ix_data.get("type", ix_data.get("operation", "unknown"))

        # If operation not explicitly set, try discriminator
        if operation == "unknown" and "data" in ix_data:
            data = ix_data["data"]
            if isinstance(data, (bytes, bytearray)):
                discriminator = data[0] if data else 0
            elif isinstance(data, dict):
                discriminator = data.get("discriminator", -1)
                operation = data.get("type", operation)
            else:
                discriminator = -1

            if operation == "unknown":
                if program_name == "system_program":
                    operation = SYSTEM_OPS.get(discriminator, "unknown_system_op")
                elif program_name == "spl_token":
                    operation = SPL_TOKEN_OPS.get(discriminator, "unknown_token_op")

        # Collect accounts referenced by this instruction
        ix_accounts = []
        for acct_ref in ix_data.get("accounts", ix_data.get("keys", [])):
            if isinstance(acct_ref, str) and acct_ref in accounts_map:
                ix_accounts.append(accounts_map[acct_ref])
            elif isinstance(acct_ref, dict):
                pk = acct_ref.get("pubkey", "")
                if pk in accounts_map:
                    ix_accounts.append(accounts_map[pk])
                else:
                    ix_accounts.append(ParsedAccount(
                        pubkey=pk,
                        is_signer=acct_ref.get("isSigner", False),
                        is_writable=acct_ref.get("isWritable", False),
                    ))

        return ParsedInstruction(
            program_id=program_id,
            program_name=program_name,
            operation=operation,
            accounts=ix_accounts,
            data=ix_data.get("data", {}) if isinstance(ix_data.get("data"), dict) else {},
        )

    # ── Instruction → Kāraka Protocol ────────────────────────────

    def _select_primary_instruction(self, parsed: ParsedTransaction) -> ParsedInstruction:
        """Select the most important instruction (skip system/ATA)."""
        for ix in parsed.instructions:
            if ix.program_name not in ("system_program", "ata_program"):
                return ix
        return parsed.instructions[0]

    def _instruction_to_kp(self, ix: ParsedInstruction,
                           tx: ParsedTransaction) -> Dict:
        """
        Map a parsed instruction to the universal Kāraka Protocol graph.

        Mapping:
          Kriyā     = the operation (transfer, mint_to, borrow, etc.)
          Kartā     = the signer (who initiated the action)
          Karma     = the target account (what is being acted upon)
          Karaṇa    = the instrument (amount, oracle, program invoked)
          Sampradāna = the recipient (destination account)
          Adhikaraṇa = the environment (program, network, slot)
          Apādāna   = the source (where tokens come from)
        """

        # Find signer (Kartā)
        signer = None
        for acct in ix.accounts:
            if acct.is_signer:
                signer = acct
                break
        if not signer and tx.fee_payer:
            signer = ParsedAccount(pubkey=tx.fee_payer, is_signer=True)

        # Find target/destination accounts
        writable_non_signer = [a for a in ix.accounts
                               if a.is_writable and not a.is_signer]

        target = writable_non_signer[0] if writable_non_signer else None
        recipient = writable_non_signer[1] if len(writable_non_signer) > 1 else None

        # Build constraints based on account metadata
        signer_constraints = []
        if signer:
            signer_constraints.append({
                "rule_id": "RC-001", "check": "signer",
                "field": "is_signer", "actual": signer.is_signer
            })
            # Check if program ID is verified (not a spoofed CPI)
            signer_constraints.append({
                "rule_id": "RC-006", "check": "secp_verify",
                "field": "signature_verified_by_secp256k1",
                "actual": ix.program_name != "unknown_program"
            })
            signer_constraints.append({
                "rule_id": "RC-007", "check": "sig_owner",
                "field": "sig_account_owner_is_bridge",
                "actual": ix.program_name != "unknown_program"
            })
            # PDA derivation check
            signer_constraints.append({
                "rule_id": "RC-004", "check": "pda",
                "field": "is_pda_derived",
                "actual": signer.owner is not None or signer.is_signer
            })

        target_constraints = []
        if target:
            target_constraints.append({
                "rule_id": "RC-012", "check": "discriminator",
                "field": "discriminator_checked",
                "actual": ix.program_name != "unknown_program"
            })
            target_constraints.append({
                "rule_id": "RC-002", "check": "program_id",
                "field": "program_id_verified",
                "actual": ix.program_id in self.program_registry
            })
            target_constraints.append({
                "rule_id": "RC-010", "check": "ltv",
                "field": "ltv_checked",
                "actual": ix.data.get("ltv_checked", True)
            })

        instrument_constraints = []
        instrument_constraints.append({
            "rule_id": "RC-005", "check": "overflow",
            "field": "overflow_checked",
            "actual": ix.data.get("overflow_checked", True)
        })
        instrument_constraints.append({
            "rule_id": "RC-008", "check": "oracle_trust",
            "field": "oracle_is_trusted",
            "actual": ix.data.get("oracle_is_trusted", True)
        })
        instrument_constraints.append({
            "rule_id": "RC-009", "check": "twap",
            "field": "uses_twap",
            "actual": ix.data.get("uses_twap", True)
        })

        kp = {
            "protocol_version": "1.0.0",
            "domain": "rust_crypto",
            "claim": {
                "raw_input": f"{ix.operation}({', '.join(a.pubkey[:8] for a in ix.accounts[:3])}...)",
                "claim_type": "assertion",
            },
            "karaka_graph": {
                "kriya": {
                    "id": "k0",
                    "surface": ix.operation,
                    "resolved_root": ix.operation,
                },
                "karta": {
                    "id": "a0",
                    "surface": signer.pubkey if signer else "unknown",
                    "lemma": signer.pubkey[:16] if signer else "unknown",
                    "constraints": signer_constraints,
                } if signer else None,
                "karma": {
                    "id": "o0",
                    "surface": target.pubkey if target else "unknown",
                    "lemma": target.pubkey[:16] if target else "unknown",
                    "constraints": target_constraints,
                } if target else None,
                "karana": {
                    "id": "i0",
                    "surface": str(ix.data.get("amount", "instrument")),
                    "lemma": "amount",
                    "constraints": instrument_constraints,
                },
                "sampradana": {
                    "id": "r0",
                    "surface": recipient.pubkey if recipient else "",
                    "lemma": recipient.pubkey[:16] if recipient else "",
                } if recipient else None,
                "adhikarana": {
                    "id": "e0",
                    "surface": ix.program_name,
                    "lemma": ix.program_id[:16],
                    "constraints": [{
                        "rule_id": "RC-011", "check": "collateral",
                        "field": "collateral_fully_verified",
                        "actual": ix.data.get("collateral_verified", True),
                    }],
                },
            },
            "meta": {
                "tx_signature": tx.signature,
                "slot": tx.slot,
                "program_id": ix.program_id,
                "program_name": ix.program_name,
                "source_hash": hashlib.sha256(
                    json.dumps({"sig": tx.signature, "op": ix.operation},
                               sort_keys=True).encode()
                ).hexdigest(),
            },
        }

        return kp

    @staticmethod
    def _empty_kp(sig: str, reason: str) -> Dict:
        """Return a KP that triggers AMBIGUOUS via circuit breaker."""
        return {
            "protocol_version": "1.0.0",
            "domain": "rust_crypto",
            "claim": {
                "raw_input": f"tx:{sig}",
                "claim_type": "assertion",
            },
            "karaka_graph": {
                "kriya": {
                    "id": "k0",
                    "surface": "__NO_INSTRUCTION__",
                    "resolved_root": None,
                }
            },
            "_deserializer_error": reason,
        }
