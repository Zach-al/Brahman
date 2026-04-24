import os
import uvicorn

if __name__ == "__main__":
    # This securely grabs the port from Railway, defaults to 8080 if running locally
    port = int(os.environ.get("PORT", 8080))
    
    # Starts the server natively via Python
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
