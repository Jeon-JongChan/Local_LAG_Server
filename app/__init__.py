"""local-rag 애플리케이션 패키지 (app/).

CLI: python app/main.py
패키지 import(from app....) 시 settings 등 평면 import가 동작하도록 app/ 를 path에 올린다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
