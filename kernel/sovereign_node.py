# Licensed under BSL 1.1 — commercial use requires written permission
# Change date: 2027-01-01 to MIT License by Bhupen Nayak
# Contact: askzachn@gmail.com

"""
Brahman Sovereign Node — Solnet Mesh RPC Wrapper

Exposes the Brahman Kernel as a network-accessible verification node.
When a transaction or synthesis request hits the Solnet P2P mesh, it
is routed to this node. The pipeline:

    1. Receive payload via HTTP/WebSocket
    2. MLX Qwen translates raw input → Kāraka Protocol JSON
    3. Brahman Kernel verifies against loaded Sūtra cartridge
    4. Node broadcasts Verdict + Logic Hash to the mesh

NOTE: Logic hashes are deterministic SHA-256 proofs of the exact
verification path taken. Full cryptographic signing of verdicts
requires a node keypair (planned for mesh deployment v2).

Endpoints:
    POST /verify         — Raw text + domain → Verdict
    POST /verify/kp      — Pre-built Kāraka Protocol → Verdict
    GET  /cartridges     — List available Sūtra cartridges
    POST /cartridge/load — Hot-swap a cartridge
    GET  /health         — Node health + loaded domain
    WS   /ws/verify      — WebSocket for streaming verification
"""

import os
import json
import time
import hashlib
import asyncio
import sys
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent))
from brahman_kernel import BrahmanKernel, Verdict, VerificationResult


# ══════════════════════════════════════════════════════════════════
# NODE CONFIGURATION
# ══════════════════════════════════════════════════════════════════

NODE_ID = os.environ.get("BRAHMAN_NODE_ID", f"sovereign-{hashlib.sha256(os.urandom(8)).hexdigest()[:8]}")

# SECURITY: Enforce secure API key. Refuse to start in production without one.
BRAHMAN_ENV = os.environ.get("BRAHMAN_ENV", "development")
API_KEY = os.environ.get("BRAHMAN_API_KEY")

if not API_KEY:
    if BRAHMAN_ENV == "production":
        print("\n✗ FATAL: BRAHMAN_API_KEY is not set.")
        print("  Production mode requires an explicit API key.")
        print("  Set it with: export BRAHMAN_API_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')")
        sys.exit(1)
    else:
        API_KEY = secrets.token_hex(32)
        print(f"⚠ WARNING: BRAHMAN_API_KEY not set (env={BRAHMAN_ENV}). Generated temporary key:")
        print(f"  {API_KEY}")
        print(f"  This key is ephemeral and will change on restart.")

DEFAULT_KEYS = {"brahman-dev-secret-key-123", "test", "dev", "password", "secret"}
if API_KEY in DEFAULT_KEYS:
    if BRAHMAN_ENV == "production":
        print(f"\n✗ FATAL: API key is a known default ('{API_KEY}'). This is not allowed in production.")
        sys.exit(1)
    else:
        print(f"⚠ WARNING: API key matches a known default. Do NOT use this in production.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key"
    )
CARTRIDGE_DIR = Path(__file__).parent / "cartridges"
DEFAULT_CARTRIDGE = "rust_crypto_sutras.json"

# Global state
kernel = BrahmanKernel()
translator = None  # Lazy-loaded MLX translator
node_start_time = time.time()
request_count = 0
verified_count = {"VALID": 0, "INVALID": 0, "AMBIGUOUS": 0}


# ══════════════════════════════════════════════════════════════════
# RATE LIMITER
# ══════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Sliding-window rate limiter.
    Tracks request timestamps per client IP. Rejects requests that
    exceed the configured requests-per-second threshold.
    """
    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: Dict[str, list] = defaultdict(list)

    def allow(self, client_id: str) -> bool:
        now = time.time()
        bucket = self._buckets[client_id]
        # Evict expired entries
        self._buckets[client_id] = [t for t in bucket if now - t < self.window]
        bucket = self._buckets[client_id]
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True

# 10 requests per second per client (HTTP + WS)
rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)


async def rate_limit_dependency(request: Request):
    """FastAPI dependency for HTTP rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 10 requests/sec."
        )


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject HTTP request bodies larger than max_bytes (default 1MB)."""
    def __init__(self, app, max_bytes: int = 1_048_576):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large. Max {self.max_bytes} bytes."
            )
        return await call_next(request)


# ══════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════

class VerifyRequest(BaseModel):
    raw_input: str = Field(..., description="Raw text to verify")
    domain: str = Field("general", description="Domain hint for the MLX translator")
    use_mlx: bool = Field(True, description="Use MLX neural translator (False = manual KP)")

class KPVerifyRequest(BaseModel):
    karaka_protocol: dict = Field(..., description="Pre-built Kāraka Protocol JSON")

class LoadCartridgeRequest(BaseModel):
    cartridge: str = Field(..., description="Cartridge filename (e.g., 'rust_crypto_sutras.json')")

class VerifyResponse(BaseModel):
    node_id: str
    verdict: str
    violations: List[str]
    matched_sutras: List[str]
    dhatu_found: bool
    logic_hash: str
    domain: str
    processing_time_ms: float
    timestamp: float

class NodeHealth(BaseModel):
    node_id: str
    status: str
    loaded_domain: str
    uptime_seconds: float
    total_requests: int
    verdicts: Dict[str, int]
    available_cartridges: List[str]
    mlx_loaded: bool


# ══════════════════════════════════════════════════════════════════
# APP LIFECYCLE
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load default cartridge
    default_path = CARTRIDGE_DIR / DEFAULT_CARTRIDGE
    if default_path.exists():
        msg = kernel.load_cartridge(str(default_path))
        print(f"[{NODE_ID}] {msg}")
    print(f"[{NODE_ID}] Sovereign Node online.")
    yield
    # Shutdown
    print(f"[{NODE_ID}] Node shutting down. Verified {sum(verified_count.values())} transactions.")


app = FastAPI(
    title="Brahman Sovereign Node",
    description="Deterministic neuro-symbolic verification node for the Solnet P2P mesh.",
    version="1.0.0",
    lifespan=lifespan,
)

# SECURITY: Restrict CORS to known origins in production.
# Override BRAHMAN_CORS_ORIGINS env var with comma-separated origins.
cors_origins = os.environ.get("BRAHMAN_CORS_ORIGINS", "http://localhost:3000,http://localhost:8420")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# SECURITY: Request body size limit (1MB)
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=1_048_576)


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/health", response_model=NodeHealth, dependencies=[Depends(get_api_key)])
async def health():
    """Node health check — returns status, loaded domain, and statistics."""
    cartridges = [f.name for f in CARTRIDGE_DIR.glob("*.json")]
    return NodeHealth(
        node_id=NODE_ID,
        status="online",
        loaded_domain=kernel.loaded_domain,
        uptime_seconds=round(time.time() - node_start_time, 1),
        total_requests=request_count,
        verdicts=verified_count,
        available_cartridges=sorted(cartridges),
        mlx_loaded=translator is not None and translator._loaded,
    )


@app.get("/cartridges", dependencies=[Depends(get_api_key)])
async def list_cartridges():
    """List all available Sūtra cartridges."""
    cartridges = []
    for f in sorted(CARTRIDGE_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        cartridges.append({
            "filename": f.name,
            "domain": data.get("domain"),
            "version": data.get("version"),
            "description": data.get("description", ""),
            "sutras_count": len(data.get("sutras", [])),
            "roots_count": len(data.get("root_lexicon", {})),
        })
    return {"cartridges": cartridges, "loaded": kernel.loaded_domain}


@app.post("/cartridge/load", dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
async def load_cartridge(req: LoadCartridgeRequest):
    """Hot-swap a Sūtra cartridge at runtime."""
    # SECURITY: Path traversal guard — enforce basename-only, .json extension,
    # and canonical path must remain within the cartridges directory.
    basename = Path(req.cartridge).name
    if not basename.endswith(".json") or basename != req.cartridge:
        raise HTTPException(status_code=400, detail="Invalid cartridge name. Must be a .json basename (no paths).")
    path = (CARTRIDGE_DIR / basename).resolve()
    if not path.is_relative_to(CARTRIDGE_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal denied.")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Cartridge '{basename}' not found.")
    msg = kernel.load_cartridge(str(path))
    return {"status": "loaded", "message": msg, "domain": kernel.loaded_domain}


@app.post("/verify", response_model=VerifyResponse, dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
async def verify_text(req: VerifyRequest):
    """
    Verify raw text through the full pipeline:
    Raw Text → MLX Translation → Kernel Verification → Signed Verdict
    """
    global request_count, translator
    request_count += 1
    start = time.time()

    if req.use_mlx:
        # Lazy-load the MLX translator on first use
        if translator is None:
            from mlx_translator import MLXTranslator
            translator = MLXTranslator(domain=req.domain)
            translator.load()

        kp = translator.translate(req.raw_input, domain=req.domain)
    else:
        # Minimal KP fallback (no neural translation)
        kp = kernel.evaluate(req.raw_input).to_dict()
        # Re-verify via full path
        result = kernel.evaluate(req.raw_input)
        elapsed = (time.time() - start) * 1000
        verified_count[result.verdict] = verified_count.get(result.verdict, 0) + 1
        return VerifyResponse(
            node_id=NODE_ID,
            verdict=result.verdict,
            violations=result.violations,
            matched_sutras=result.matched_sutras,
            dhatu_found=result.dhatu_found,
            logic_hash=result.logic_hash,
            domain=kernel.loaded_domain,
            processing_time_ms=round(elapsed, 1),
            timestamp=time.time(),
        )

    result = kernel.verify(kp)
    elapsed = (time.time() - start) * 1000
    verified_count[result.verdict] = verified_count.get(result.verdict, 0) + 1

    return VerifyResponse(
        node_id=NODE_ID,
        verdict=result.verdict,
        violations=result.violations,
        matched_sutras=result.matched_sutras,
        dhatu_found=result.dhatu_found,
        logic_hash=result.logic_hash,
        domain=kernel.loaded_domain,
        processing_time_ms=round(elapsed, 1),
        timestamp=time.time(),
    )


@app.post("/verify/kp", response_model=VerifyResponse, dependencies=[Depends(get_api_key), Depends(rate_limit_dependency)])
async def verify_kp(req: KPVerifyRequest):
    """
    Verify a pre-built Kāraka Protocol instance directly.
    Bypasses the neural translator — pure symbolic verification.
    """
    global request_count
    request_count += 1
    start = time.time()

    result = kernel.verify(req.karaka_protocol)
    elapsed = (time.time() - start) * 1000
    verified_count[result.verdict] = verified_count.get(result.verdict, 0) + 1

    return VerifyResponse(
        node_id=NODE_ID,
        verdict=result.verdict,
        violations=result.violations,
        matched_sutras=result.matched_sutras,
        dhatu_found=result.dhatu_found,
        logic_hash=result.logic_hash,
        domain=kernel.loaded_domain,
        processing_time_ms=round(elapsed, 1),
        timestamp=time.time(),
    )


# ══════════════════════════════════════════════════════════════════
# WEBSOCKET — Streaming Verification
# ══════════════════════════════════════════════════════════════════

@app.websocket("/ws/verify")
async def ws_verify(websocket: WebSocket):
    """
    WebSocket endpoint for streaming verification.
    Auth: Send {"type": "auth", "api_key": "..."} as the FIRST message.
    Then send {"raw_input": "...", "domain": "..."} for verification.
    API key is never exposed in query strings or proxy logs.
    """
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"

    # SECURITY: First-message authentication protocol.
    # The client must send an auth payload before any verification requests.
    try:
        auth_data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
    except (asyncio.TimeoutError, Exception):
        await websocket.close(code=1008, reason="Auth timeout")
        return

    if auth_data.get("type") != "auth" or auth_data.get("api_key") != API_KEY:
        await websocket.send_json({"error": "AUTH_FAILED", "detail": "Invalid or missing API key."})
        await websocket.close(code=1008, reason="Auth failed")
        return

    await websocket.send_json({"type": "auth_ok", "node_id": NODE_ID})
    print(f"[{NODE_ID}] WebSocket client connected (authenticated, ip={client_ip}).")

    try:
        while True:
            data = await websocket.receive_json()

            # SECURITY: Per-IP rate limiting
            if not rate_limiter.allow(client_ip):
                await websocket.send_json({
                    "error": "RATE_LIMITED",
                    "detail": "Too many requests. Max 10/sec.",
                    "timestamp": time.time(),
                })
                continue

            raw_input = data.get("raw_input", "")
            domain = data.get("domain", kernel.loaded_domain)

            global request_count
            request_count += 1
            start = time.time()

            # Use kernel's evaluate for WS (no MLX overhead per message)
            result = kernel.evaluate(raw_input)
            elapsed = (time.time() - start) * 1000
            verified_count[result.verdict] = verified_count.get(result.verdict, 0) + 1

            await websocket.send_json({
                "node_id": NODE_ID,
                "verdict": result.verdict,
                "violations": result.violations,
                "logic_hash": result.logic_hash,
                "processing_time_ms": round(elapsed, 1),
                "timestamp": time.time(),
            })
    except WebSocketDisconnect:
        print(f"[{NODE_ID}] WebSocket client disconnected.")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("BRAHMAN_PORT", 8420))
    print(f"\n{'='*60}")
    print(f"  BRAHMAN SOVEREIGN NODE")
    print(f"  Node ID: {NODE_ID}")
    print(f"  Port:    {port}")
    print(f"  Cartridges: {CARTRIDGE_DIR}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
