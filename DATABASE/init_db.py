"""
Initialize the PostgreSQL database schema.
Run once before starting the app:
    python DATABASE/init_db.py
"""
import sys
import os

# Allow imports from BACKEND/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "BACKEND"))

from database.db import init_db

if __name__ == "__main__":
    print("Initializing PostgreSQL database...")
    init_db()
    print("Done.")
