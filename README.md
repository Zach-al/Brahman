# Brahman — Universal Neuro-Symbolic Verification Engine

**Brahman** is a domain-agnostic verification kernel that combines formal grammar systems with modern neuro-symbolic AI to produce deterministic, zero-hallucination logic verdicts.

The engine translates problems (smart contract audits, biochemical reactions, formal logic, memory safety) into a universal **Kāraka Protocol** graph, then verifies it against domain-specific **Sūtra cartridges** using state-machine traversal.

Every verdict is cryptographically sealed with a **Logic Hash** (SHA-256 of the full traversal path), making results auditable and reproducible.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    BRAHMAN STACK                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────────┐    │
│  │ Raw Input │───▶│ MLX Translator │───▶│  Kāraka Protocol │    │
│  │ (text/tx) │    │ (Qwen3-1.7B)  │    │   JSON Graph     │    │
│  └──────────┘    └───────────────┘    └────────┬─────────┘    │
│                                                 │              │
│                                    ┌────────────▼────────────┐ │
│                                    │    BRAHMAN KERNEL        │ │
│                                    │  ┌────────────────────┐  │ │
│                                    │  │  Sūtra Cartridge   │  │ │
│                                    │  │  (domain rules)    │  │ │
│                                    │  └────────┬───────────┘  │ │
│                                    │           │ traverse     │ │
│                                    │  ┌────────▼───────────┐  │ │
│                                    │  │  Gate Evaluator    │  │ │
│                                    │  │  VALID / INVALID   │  │ │
│                                    │  │  / AMBIGUOUS       │  │ │
│                                    │  └────────┬───────────┘  │ │
│                                    │           │              │ │
│                                    │  Logic Hash (SHA-256)    │ │
│                                    └────────────┬────────────┘ │
│                                                 │              │
│                              ┌──────────────────▼────────────┐ │
│                              │     VALIDATOR MESH            │ │
│                              │  Consensus via hash agreement │ │
│                              │  2/3 quorum = finalized       │ │
│                              └───────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

Brahman is hardened for enterprise production environments:

- **Authentication**: All API and WebSocket endpoints enforce strict API key validation (`X-API-Key` header / first-message WS auth protocol). Startup aborts in production if keys are missing or default.
- **Model Integrity**: Supply-chain protection via SHA-256 checksum verification (`BRAHMAN_MODEL_SHA256`). Mismatched or compromised weights are automatically deleted, failing-closed on startup.
- **HTTP Hardening**: Native enforcement of HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, and X-XSS-Protection headers. 
- **Rate Limiting & Abuse Prevention**: Sliding-window rate limiters (10 req/sec per IP) and strict 1MB request body size limits.
- **Container Security**: Multi-stage Docker builds running under a non-root `brahman` user, stripped of build-time dependencies, with granular file ownership and healthchecks.
- **Path Traversal Guards**: Strict basename validation and path resolution restrictions on hot-swappable cartridge endpoints.

---

## Core Components

### 1. Brahman Kernel — `kernel/brahman_kernel.py`

The kernel is **completely domain-blind**. It only evaluates graph syntax:

1. Read a Kāraka Protocol graph
2. Load a Sūtra cartridge (domain rules)
3. For each node, check if connections are **legal** according to the loaded rules
4. Output: `VALID` / `INVALID` / `AMBIGUOUS` + Logic Hash

### 2. Kāraka Protocol — `kernel/karaka_protocol.schema.json`

The universal intermediate representation. Every problem must be translated into exactly **six semantic roles** (kārakas):

| Role | Meaning | Example |
|------|---------|---------|
| **Kriyā** | Action/Operation | `transfer`, `phosphorylate`, `entails` |
| **Kartā** | Agent | signer, enzyme, premise |
| **Karma** | Target/Patient | escrow, substrate, conclusion |
| **Karaṇa** | Instrument | amount, ATP, oracle price |
| **Sampradāna** | Recipient | destination wallet, product |
| **Adhikaraṇa** | Environment | Solana mainnet, pH 7.4, vacuum |

### 3. MLX Neural Translator — `kernel/mlx_translator.py`

A quantized **Qwen3-1.7B-4bit** model running on Apple Silicon via MLX. Translates raw human text into Kāraka Protocol JSON using few-shot prompting. Runs in `nothink` mode for deterministic, clean output.

### 4. Sūtra Cartridges — `kernel/cartridges/`

Domain-specific rule sets loaded at runtime:

| Cartridge | Domain | Sūtras | Roots | Description |
|-----------|--------|--------|-------|-------------|
| `sanskrit_sutras.json` | Sanskrit Grammar | 4 | 11 | Pāṇinian Dhātupāṭha verb-case constraints |
| `rust_crypto_sutras.json` | Solana/DeFi Security | 12 | 16 | Wormhole, Mango Markets, Cashio exploit classes |
| `formal_logic_sutras.json` | Propositional Logic | 12 | 12 | Syllogistic validity, quantifier scoping |
| `memory_safety_sutras.json` | Systems Security | 12 | 15 | Buffer overflow, use-after-free, privilege escalation |
| `biochem_sutras.json` | Biochemistry | 12 | 15 | Enzyme catalysis, thermodynamic feasibility |
| `thermo_sutras.json` | Thermodynamics | 4 | 8 | Conservation laws, state transformations |

### 5. Sovereign Node — `kernel/sovereign_node.py`

FastAPI + WebSocket RPC server that exposes the kernel as a network-accessible verification node:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/verify` | POST | Raw text → MLX → Kernel → Signed Verdict |
| `/verify/kp` | POST | Pre-built KP graph → Pure symbolic verification |
| `/cartridge/load` | POST | Hot-swap domain cartridge at runtime |
| `/cartridges` | GET | List all available cartridges |
| `/health` | GET | Node status, uptime, verdict statistics |
| `/ws/verify` | WS | WebSocket for streaming mesh verification |

### 6. Validator Client — `kernel/validator/`

Custom validator that enforces Brahman rules at the blockchain level:

| Module | Purpose |
|--------|---------|
| `validator_client.py` | Off-chain daemon: TX intercept → KP → kernel → verdict |
| `tx_deserializer.py` | Solana TX → Kāraka Protocol graph translation |
| `mesh_consensus.py` | Deterministic hash-based 2/3 quorum consensus |
| `verification_protocol.py` | On-chain VerificationRecord PDA + QuorumStatus lifecycle |
| `test_validator.py` | 30 integration tests (consensus, anti-spoofing, full pipeline) |

---

## Crucible Test — Historical Zero-Day Exploit Detection

The engine was tested against the **three largest Solana exploits in history**. It correctly detected all three with zero false positives:

| Exploit | Date | Damage | Root Cause | Sūtras Triggered |
|---------|------|--------|------------|-------------------|
| **Wormhole Bridge** | Feb 2022 | $326M | Spoofed guardian `SignatureSet` — bridge checked writable boolean instead of account owner | RC-001, RC-006, RC-007 |
| **Mango Markets** | Oct 2022 | $114M | Oracle price manipulation via self-trading — used spot price instead of TWAP | RC-008, RC-009, RC-010 |
| **Cashio Collapse** | Mar 2022 | $52M | Fake collateral account bypassed incomplete verification chain | RC-004, RC-011, RC-012 |

Run: `python3 kernel/crucible_test.py`

---

## Test Suite

```
Kernel self-test ........ 6/6   ✓   (Sanskrit, Rust/Crypto, Thermodynamics)
Stress test ............ 16/16  ✓   (Memory Safety, Biochem, Formal Logic, Edge Cases)
Crucible (exploits) ....  6/6   ✓   (Wormhole, Mango Markets, Cashio)
Validator mesh ......... 30/30  ✓   (Deserializer, Consensus, Anti-Spoofing, Full Pipeline)
─────────────────────────────────
TOTAL                    52/52  ✓
```

Run all:
```bash
python3 kernel/test_kernel.py
python3 kernel/stress_test.py
python3 kernel/crucible_test.py
python3 kernel/validator/test_validator.py
```

---

## Quick Start

### Local 

```bash
# Clone
git clone https://github.com/Zach-al/Brahman.git
cd Brahman

# Install dependencies
pip install fastapi uvicorn

# Setup environment variables
export BRAHMAN_API_KEY="your-secret-key"

# Run tests
python3 kernel/test_kernel.py
python3 kernel/crucible_test.py
python3 kernel/validator/test_validator.py

# Start the Sovereign Node
python3 kernel/sovereign_node.py
```

### Docker

```bash
docker build -t brahman .
docker run -p 8080:8080 -p 8420:8420 -e BRAHMAN_API_KEY="secret" brahman
```

### API Usage

```bash
# Health check
curl http://localhost:8420/health

# Verify a pre-built Kāraka Protocol graph
curl -X POST http://localhost:8420/verify/kp \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "karaka_protocol": {
      "protocol_version": "1.0.0",
      "domain": "rust_crypto",
      "claim": {"raw_input": "transfer(escrow, attacker, 1000)", "claim_type": "assertion"},
      "karaka_graph": {
        "kriya": {"id": "k0", "surface": "transfer", "resolved_root": "transfer"},
        "karta": {"id": "a0", "surface": "attacker", "lemma": "attacker",
          "constraints": [{"rule_id": "RC-001", "field": "is_signer", "actual": false}]}
      }
    }
  }'
```

---

## License

MIT
