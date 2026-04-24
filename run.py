import os
import subprocess
import uvicorn
from pathlib import Path

DATA_DIR = Path("/app/app/data")
MODEL_PATH = DATA_DIR / "brahman_v2_core.pth"
DB_PATH = DATA_DIR / "brahman_v2.db"

# The direct binary download link for the 545MB weights
MODEL_URL = "https://github.com/Zach-al/Brahman/releases/download/v2.0.0/brahman_v2_core.pth"

def bootstrap_brain():
    print("🧠 Bootstrapper: Checking Persistent Volume...")
    
    # 1. Check and download Weights
    if not MODEL_PATH.exists():
        print(f"⚠️ Weights missing. Downloading 545MB model to {MODEL_PATH}...")
        # Using -q --show-progress makes the Railway logs much cleaner to read
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(MODEL_PATH), MODEL_URL], check=True)
        print("✅ Weights downloaded successfully.")
    else:
        print("✅ Weights found on disk.")

    # 2. Check and generate Database
    if not DB_PATH.exists():
        print(f"⚠️ SQLite DB missing. Generating Dhātupāṭha at {DB_PATH}...")
        subprocess.run(["python", "app/ingestion_engine.py"], check=True)
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
