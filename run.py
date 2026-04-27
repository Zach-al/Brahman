import os
import subprocess
import hashlib as _hashlib
import uvicorn
from pathlib import Path

DATA_DIR = Path("/app/app/data")
MODEL_PATH = DATA_DIR / "brahman_v2_core.pth"
DB_PATH = DATA_DIR / "brahman_v2.db"

# The direct binary download link for the 545MB weights
MODEL_URL = "https://github.com/Zach-al/Brahman/releases/download/v2.0.0/brahman_v2_core.pth"
# SECURITY: SHA-256 checksum of the expected model artifact.
# Update this hash whenever you publish a new model release.
MODEL_SHA256 = os.environ.get("BRAHMAN_MODEL_SHA256", "")

def _verify_sha256(filepath: Path, expected_hash: str) -> bool:
    """Verify file integrity via SHA-256."""
    sha = _hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != expected_hash:
        print(f"✗ INTEGRITY FAILURE: Expected SHA-256 {expected_hash}")
        print(f"  Actual SHA-256: {actual}")
        return False
    return True

BRAHMAN_ENV = os.environ.get("BRAHMAN_ENV", "development")

def bootstrap_brain():
    print("🧠 Bootstrapper: Checking Persistent Volume...")
    
    # 1. Check and download Weights
    if not MODEL_PATH.exists():
        print(f"⚠️ Weights missing. Downloading 545MB model to {MODEL_PATH}...")
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(MODEL_PATH), MODEL_URL], check=True)
        # SECURITY: Verify artifact integrity — MANDATORY in production
        if MODEL_SHA256:
            if not _verify_sha256(MODEL_PATH, MODEL_SHA256):
                MODEL_PATH.unlink()  # Delete compromised artifact
                raise SystemExit("Aborting: Model artifact failed integrity check.")
            print("✅ Weights downloaded and integrity verified (SHA-256 match).")
        elif BRAHMAN_ENV == "production":
            MODEL_PATH.unlink()  # Don't keep unverified artifacts in production
            raise SystemExit(
                "✗ FATAL: BRAHMAN_MODEL_SHA256 is not set.\n"
                "  Production mode requires mandatory artifact integrity verification.\n"
                "  Set BRAHMAN_MODEL_SHA256 to the expected SHA-256 hash of the model file."
            )
        else:
            print("✅ Weights downloaded. ⚠ No BRAHMAN_MODEL_SHA256 set — skipping integrity check (dev mode).")
    else:
        print("✅ Weights found on disk.")

    # 2. Check and generate Database
    if not DB_PATH.exists():
        print(f"⚠️ SQLite DB missing. Generating Dhātupāṭha at {DB_PATH}...")
        # Use absolute path based on run.py's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ingestion_script = os.path.join(base_dir, "ingestion_engine.py")
        # Pass the correct DB path so it writes where the app expects
        subprocess.run([
            "python", "-c",
            f"import sys; sys.path.insert(0,'{base_dir}'); "
            f"from ingestion_engine import BrahmanIngestion; "
            f"b = BrahmanIngestion('{DB_PATH}'); b.ingest_dhatupatha(); b.close()"
        ], check=True)
        print("✅ Database generated successfully.")
    else:
        print("✅ Database found on disk.")

if __name__ == "__main__":
    # Ensure the data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run the checks
    bootstrap_brain()
    
    # Grab port and start server
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting Uvicorn Oracle on port {port}...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
