from pathlib import Path

CACHE_DIR = Path("~/.cache/pier").expanduser()
TASK_CACHE_DIR = CACHE_DIR / "tasks"
