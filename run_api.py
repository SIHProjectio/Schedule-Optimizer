#!/usr/bin/env python3
"""
Startup script for Metro Train Scheduling API
"""
import uvicorn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    print("=" * 60)
    print("Metro Train Scheduling API")
    print("=" * 60)
    print()
    print("Starting FastAPI server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print()
    print("Example endpoints:")
    print("  - GET  /health")
    print("  - GET  /api/v1/schedule/example")
    print("  - POST /api/v1/generate")
    print("  - POST /api/v1/generate/quick?date=2025-10-25&num_trains=30")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "DataService.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
