"""
Main entry point for Cerebrium deployment.
This file is required by Cerebrium platform.
"""

from src.app import app

# Cerebrium will automatically serve this app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
