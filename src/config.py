import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=False)
except ImportError:
    pass

def get(var: str, default: str | None = None) -> str | None:
    return os.getenv(var, default)
