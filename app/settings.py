"""local-rag 전역 설정 및 파사드(Settings Facade) 모듈.

.env 파일 자동 로드와 하드웨어 사양 프리셋(저사양 low / 중간 mid / 고사양 high)을 지원하며,
파사드 패턴을 통해 환경변수 우선순위를 지능적으로 병합하여 최적 설정을 제공합니다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# 1. .env 파일 자동 로드
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    # python-dotenv 미설치 환경 대비 간단한 수동 .env 파서 내장
    env_file = BASE_DIR / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# ---------------------------------------------------------------------------
# 2. 디렉터리 경로 상수 (프로젝트 고정)
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"          # 원본 색인 PDF 폴더
REJECTED_DIR = DATA_DIR / "rejected"  # 파싱 실패·손상 파일 격리 폴더
INDEX_DIR = DATA_DIR / "index"        # FAISS·SQLite DB 저장 폴더
PARSED_DIR = DATA_DIR / "parsed"      # OpenDataLoader 임시 파싱 폴더

SQLITE_PATH = INDEX_DIR / "chunks.sqlite3"   # 청크 본문 및 FTS5 DB
FAISS_PATH = INDEX_DIR / "faiss.index"       # Dense 벡터 인덱스
INDEX_META_PATH = INDEX_DIR / "meta.json"    # 인덱스 메타데이터

MODELS_DIR = DATA_DIR / "models"
CHAT_MODELS_DIR = MODELS_DIR / "chat"           # GGUF 모델 디렉토리
EMBED_MODELS_DIR = MODELS_DIR / "embeddings"    # 임베딩 모델 디렉토리
RERANK_MODELS_DIR = MODELS_DIR / "reranker"     # 리랭커 모델 디렉토리


def hf_repo_local_name(repo_id: str) -> str:
    """HuggingFace repo ID에서 로컬 폴더명을 추출합니다."""
    return repo_id.strip().split("/")[-1] or "model"


def hf_model_dir(repo_id: str, category: Path) -> Path:
    """카테고리별 모델 로컬 경로를 반환합니다."""
    return category / hf_repo_local_name(repo_id)


# ---------------------------------------------------------------------------
# 3. 사양별 프리셋 정의 (low: 저사양 / mid: 중간 / high: 고사양)
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    # 저사양: U시리즈 노트북, 내장 그래픽, 8GB~16GB RAM 환경
    "low": {
        "ACCELERATOR": "cpu",
        "SERVING_ENGINE": "llama.cpp",
        "LLAMA_GGUF_REPO": "bartowski/Qwen_Qwen2.5-1.5B-Instruct-GGUF",
        "LLAMA_GGUF_FILE": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "LLAMA_CPP_VARIANT": "cpu",
        "LLAMA_CTX_SIZE": 2048,
        "LLAMA_NGL": 0,
        "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "EMBED_DEVICE": "cpu",
        "RERANK_STRATEGY": "origin",
        "FAISS_TOP_K": 4,
        "RERANK_TOP_K": 2,
        "DOC_PARSER": "pypdf",
        "LLAMA_NUM_PREDICT": 2048,
        "LLAMA_NUM_PREDICT_THINK": 4096,
    },
    # 중간: 일반 게이밍 노트북/데스크탑, 6GB~8GB VRAM GPU (기본값)
    "mid": {
        "ACCELERATOR": "vulkan",
        "SERVING_ENGINE": "llama.cpp",
        "LLAMA_GGUF_REPO": "bartowski/Qwen_Qwen3.5-9B-GGUF",
        "LLAMA_GGUF_FILE": "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        "LLAMA_CPP_VARIANT": "vulkan",
        "LLAMA_CTX_SIZE": 4096,
        "LLAMA_NGL": 999,
        "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "EMBED_DEVICE": "auto",
        "RERANK_STRATEGY": "origin",
        "FAISS_TOP_K": 8,
        "RERANK_TOP_K": 4,
        "DOC_PARSER": "opendataloader",
        "LLAMA_NUM_PREDICT": 4096,
        "LLAMA_NUM_PREDICT_THINK": 16384,
    },
    # 고사양: 16GB~24GB+ VRAM 하이엔드 GPU (RTX 3090/4090 등)
    "high": {
        "ACCELERATOR": "cuda",
        "SERVING_ENGINE": "vllm",
        "LLAMA_GGUF_REPO": "bartowski/Qwen_Qwen2.5-14B-Instruct-GGUF",
        "LLAMA_GGUF_FILE": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "LLAMA_CPP_VARIANT": "cuda",
        "LLAMA_CTX_SIZE": 8192,
        "LLAMA_NGL": 999,
        "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "EMBED_DEVICE": "cuda",
        "RERANK_STRATEGY": "laya",
        "FAISS_TOP_K": 16,
        "RERANK_TOP_K": 6,
        "DOC_PARSER": "opendataloader",
        "LLAMA_NUM_PREDICT": 8192,
        "LLAMA_NUM_PREDICT_THINK": 32768,
    },
}


# ---------------------------------------------------------------------------
# 4. 설정 파사드 (SettingsFacade)
# ---------------------------------------------------------------------------
class SettingsFacade:
    """
    설정 파사드 클래스.

    우선순위:
      1순위: OS 환경변수 또는 .env 파일에 명시된 사용자 설정
      2순위: PROFILE (low | mid | high) 프리셋 기본값
      3순위: 시스템 전역 기본값
    """

    def __init__(self) -> None:
        self.profile = os.getenv("PROFILE", os.getenv("HARDWARE_PROFILE", "mid")).strip().lower()
        if self.profile not in PRESETS:
            self.profile = "mid"
        self._preset = PRESETS.get(self.profile, PRESETS["mid"])

    def get(self, key: str, default: Any = None) -> Any:
        """환경변수 우선, 프리셋 차순, 기본값 후순으로 조회."""
        env_val = os.getenv(key)
        if env_val is not None and env_val != "":
            return env_val
        if key in self._preset:
            return self._preset[key]
        return default

    def get_int(self, key: str, default: int) -> int:
        val = self.get(key, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float) -> float:
        val = self.get(key, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool) -> bool:
        val = str(self.get(key, str(default))).strip().lower()
        return val not in ("0", "false", "no", "off")


# 싱글톤 파사드 인스턴스 생성
settings = SettingsFacade()

def detect_accelerator() -> str:
    """하드웨어 가속 환경을 CUDA > Vulkan > CPU 순으로 자동 감지합니다."""
    # 1. CUDA 감지 (PyTorch 및 nvidia-smi 점검)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is not None:
        try:
            res = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return "cuda"
        except Exception:
            pass

    # 2. Vulkan 감지 (Windows vulkan-1.dll 또는 Linux vulkaninfo)
    if sys.platform == "win32":
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        if (system32 / "vulkan-1.dll").is_file():
            return "vulkan"
    else:
        if shutil.which("vulkaninfo") is not None:
            return "vulkan"

    # 3. CPU fallback
    return "cpu"


# ---------------------------------------------------------------------------
# 5. 파사드 기반 전역 변수 매핑 (완벽한 하위 호환성 제공)
# ---------------------------------------------------------------------------
PROFILE = settings.profile

raw_accel = settings.get("ACCELERATOR", "auto").strip().lower()
if raw_accel == "auto":
    ACCELERATOR = detect_accelerator()
else:
    ACCELERATOR = raw_accel

# 서빙 엔진 및 모델
SERVING_ENGINE = settings.get("SERVING_ENGINE", "llama.cpp").strip().lower()
LLAMA_GGUF_REPO = settings.get("LLAMA_GGUF_REPO", "bartowski/Qwen_Qwen3.5-9B-GGUF")
LLAMA_GGUF_FILE = settings.get("LLAMA_GGUF_FILE", "Qwen_Qwen3.5-9B-Q4_K_M.gguf")

raw_variant = settings.get("LLAMA_CPP_VARIANT", "").strip().lower()
if not raw_variant or raw_variant == "auto":
    LLAMA_CPP_VARIANT = ACCELERATOR
else:
    LLAMA_CPP_VARIANT = raw_variant

LLAMA_CTX_SIZE = settings.get_int("LLAMA_CTX_SIZE", 4096)
LLAMA_NGL = settings.get_int("LLAMA_NGL", 999)

# 임베딩 및 디바이스
EMBED_MODEL = settings.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DEVICE = settings.get("EMBED_DEVICE", "auto")
if EMBED_DEVICE == "auto":
    EMBED_DEVICE = "cuda" if ACCELERATOR == "cuda" else "cpu"


# 리랭커 설정
RERANK_STRATEGY = settings.get("RERANK_STRATEGY", "laya").strip().lower()
RERANK_MODEL = settings.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANK_ORIGIN_MODEL = settings.get("RERANK_ORIGIN_MODEL", RERANK_MODEL)
RERANK_ORIGIN_SCORER = settings.get("RERANK_ORIGIN_SCORER", "qwen_yesno")
RERANK_ORIGIN_DEVICE = settings.get("RERANK_ORIGIN_DEVICE", EMBED_DEVICE)

RERANK_LAYA_MODEL = settings.get("RERANK_LAYA_MODEL", "convaiinnovations/laya-multilingual")
RERANK_LAYA_DECISION = settings.get("RERANK_LAYA_DECISION", "noul")
RERANK_LAYA_MAX_LENGTH = settings.get_int("RERANK_LAYA_MAX_LENGTH", 1024)

RERANK_CLM_BASE_URL = settings.get("RERANK_CLM_BASE_URL", "http://127.0.0.1:8700").rstrip("/")
RERANK_CLM_TIMEOUT = settings.get_float("RERANK_CLM_TIMEOUT", 10.0)
RERANK_CLM_ACTION_CACHE = settings.get_bool("RERANK_CLM_ACTION_CACHE", True)

# 검색 및 RRF 랭킹 파라미터
FAISS_TOP_K = settings.get_int("FAISS_TOP_K", 8)
BM25_TOP_K = settings.get_int("BM25_TOP_K", 16)
HYBRID_RRF_K = settings.get_int("HYBRID_RRF_K", 60)
RERANK_TOP_K = settings.get_int("RERANK_TOP_K", 4)
HYBRID_BM25 = settings.get_bool("HYBRID_BM25", True)

# 문서 파서 설정
DOC_PARSER = settings.get("DOC_PARSER", "opendataloader").strip().lower()
OPENDATALOADER_FORMAT = settings.get("OPENDATALOADER_FORMAT", "json")

# 텍스트 청킹
CHUNK_MIN_CHARS = settings.get_int("CHUNK_MIN_CHARS", 200)
CHUNK_MAX_CHARS = settings.get_int("CHUNK_MAX_CHARS", 800)
CHUNK_OVERLAP_CHARS = settings.get_int("CHUNK_OVERLAP_CHARS", 80)

# LLM 서빙 연결
LLAMA_CPP_BASE_URL = settings.get("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1").rstrip("/")
LLAMA_CPP_MODEL = settings.get("LLAMA_CPP_MODEL", "").strip() or None
LLAMA_CPP_API_KEY = settings.get("LLAMA_CPP_API_KEY", "no-key")
CHAT_TEMPERATURE = settings.get_float("CHAT_TEMPERATURE", 0.2)
LLAMA_NUM_PREDICT = settings.get_int("LLAMA_NUM_PREDICT", 4096)
LLAMA_NUM_PREDICT_THINK = settings.get_int("LLAMA_NUM_PREDICT_THINK", 16384)
THINK_AUTO_MIN_CHARS = settings.get_int("THINK_AUTO_MIN_CHARS", 200)
CHAT_HISTORY_MAX_MESSAGES = settings.get_int("CHAT_HISTORY_MAX_MESSAGES", 20)
REQUEST_TIMEOUT = settings.get_float("REQUEST_TIMEOUT", 600.0)

# 웹 서버
WEB_HOST = settings.get("WEB_HOST", "0.0.0.0")
WEB_PORT = settings.get_int("WEB_PORT", 8787)
