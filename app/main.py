"""
Brahman 2.0 — Reasoning-as-a-Service (RaaS)
FastAPI Production Entrypoint.

Loads the Symbolic Core, Neural Bridge, and SQLite Bedrock once at startup.
All inference runs against the pre-loaded state — zero cold-start per request.
"""
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.sqlite_manager import SQLiteManager
from app.core.panini_engine import PaniniEngine
from app.core.neural_bridge import KarakaBridge, get_device, load_bridge
from app.core.reasoning import ReasoningOracle
from app.api import verify, solnet
from app.schemas.pydantic_models import HealthResponse

# ── Globals (populated at startup) ───────────────────────────────────────
oracle: ReasoningOracle = None
db_manager: SQLiteManager = None
device: torch.device = None
dhatu_count: int = 0
model_loaded: bool = False

logger = logging.getLogger("brahman")
logging.basicConfig(level=logging.INFO)

# ── Resolve data paths ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_PATH = DATA_DIR / "brahman_v2_core.pth"
DB_PATH = DATA_DIR / "brahman_v2.db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Loads all heavy resources ONCE into memory at startup.
    """
    global oracle, db_manager, device, dhatu_count, model_loaded

    logger.info("=" * 60)
    logger.info("BRAHMAN 2.0 — REASONING ORACLE STARTING")
    logger.info("=" * 60)

    # 1. Detect device
    device = get_device()
    logger.info(f"Inference device: {device}")

    # 2. Connect to SQLite (WAL mode)
    db_manager = SQLiteManager(db_path=str(DB_PATH))
    dhatu_count = db_manager.get_dhatu_count()
    logger.info(f"SQLite connected — {dhatu_count} dhātus loaded (WAL mode)")

    # 3. Initialize the Symbolic Core
    dhatus_raw = db_manager.get_all_dhatus()
    dhatus_for_engine = [{"root": d["root"]} for d in dhatus_raw]
    panini = PaniniEngine(dhatus=dhatus_for_engine)
    logger.info("Pāṇini Rule Engine initialized")

    # 4. Load the Neural Bridge
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    if WEIGHTS_PATH.exists():
        bridge = load_bridge(str(WEIGHTS_PATH), device)
        model_loaded = True
        logger.info(f"KarakaBridge loaded from {WEIGHTS_PATH}")
    else:
        logger.warning(f"No weights at {WEIGHTS_PATH} — running with untrained bridge")
        bridge = KarakaBridge().to(device)
        bridge.eval()
        model_loaded = False

    # 5. Assemble the Oracle
    oracle = ReasoningOracle(
        panini=panini,
        bridge=bridge,
        tokenizer=tokenizer,
        device=device
    )

    logger.info("=" * 60)
    logger.info("BRAHMAN 2.0 ORACLE — ONLINE")
    logger.info("=" * 60)

    yield

    # Shutdown
    if db_manager:
        db_manager.close()
    logger.info("Brahman 2.0 Oracle shut down gracefully.")


# ── FastAPI App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Brahman 2.0 — Reasoning Oracle",
    description=(
        "Neuro-Symbolic Sanskrit Reasoning Engine. "
        "Provides Linguistic Verification Certificates via Pāṇinian logic. "
        "Zero-Hallucination Policy enforced."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — environment-configurable origin allowlist (no wildcards with credentials)
import os as _os
_cors_origins = _os.environ.get("BRAHMAN_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# ── Routes ───────────────────────────────────────────────────────────────
app.include_router(verify.router)
app.include_router(solnet.router)


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "Brahman 2.0 — Reasoning Oracle",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    return HealthResponse(
        status="online",
        version="2.0.0",
        dhatus_loaded=dhatu_count,
        device=str(device),
        model_loaded=model_loaded
    )
