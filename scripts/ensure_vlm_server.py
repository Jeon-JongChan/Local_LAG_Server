"""local-rag 편의 래퍼 → tools/labeler/ensure_vlm_server.py"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_target = Path(__file__).resolve().parent.parent / "tools" / "labeler" / "ensure_vlm_server.py"
raise SystemExit(subprocess.call([sys.executable, str(_target), *sys.argv[1:]]))
