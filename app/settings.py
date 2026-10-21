"""local-rag 전역 설정 및 파사드(Settings Facade) 모듈.

두 축의 프리셋:
  - PROFILE (low|mid|high): 하드웨어·RAG 공통 (엔진을 고르지 않음)
  - ENGINE_PRESETS[엔진][PROFILE]: 해당 서빙 엔진 전용 모델/기동 값

우선순위: env/.env > 엔진 프리셋 > PROFILE 프리셋 > 코드 default
"""

from __future__ import annotations

import os
import sys
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
DOCS_DIR = DATA_DIR / "docs"
REJECTED_DIR = DATA_DIR / "rejected"
INDEX_DIR = DATA_DIR / "index"
PARSED_DIR = DATA_DIR / "parsed"

SQLITE_PATH = INDEX_DIR / "chunks.sqlite3"
FAISS_PATH = INDEX_DIR / "faiss.index"
INDEX_META_PATH = INDEX_DIR / "meta.json"

MODELS_DIR = DATA_DIR / "models"
CHAT_MODELS_DIR = MODELS_DIR / "chat"
EMBED_MODELS_DIR = MODELS_DIR / "embeddings"
RERANK_MODELS_DIR = MODELS_DIR / "reranker"


def hf_repo_local_name(repo_id: str) -> str:
    """HuggingFace repo ID에서 로컬 폴더명을 추출합니다."""
    name = repo_id.strip().split("/")[-1] or "model"
    for ch in ':<>"/\\|?*':
        name = name.replace(ch, "_")
    return name


def hf_model_dir(repo_id: str, category: Path) -> Path:
    """카테고리별 모델 로컬 경로를 반환합니다."""
    return category / hf_repo_local_name(repo_id)


# ---------------------------------------------------------------------------
# 3-A. PROFILE 프리셋 — 하드웨어·RAG 공통 (SERVING_ENGINE 없음)
# ---------------------------------------------------------------------------
PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "low": {
        "ACCELERATOR": "auto",
        "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "RERANK_STRATEGY": "origin",
        "FAISS_TOP_K": 4,
        "RERANK_TOP_K": 2,
        "DOC_PARSER": "pypdf",
        "LLAMA_NUM_PREDICT": 512,
        "LLAMA_NUM_PREDICT_THINK": 2048,
    },
    "mid": {
        # auto: GPU(cuda) > Vulkan > CPU 순 감지
        "ACCELERATOR": "auto",
        "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "RERANK_STRATEGY": "origin",
        "FAISS_TOP_K": 8,
        "RERANK_TOP_K": 4,
        "DOC_PARSER": "opendataloader",
        "LLAMA_NUM_PREDICT": 4096,
        "LLAMA_NUM_PREDICT_THINK": 16384,
    },
    "high": {
        "ACCELERATOR": "auto",
        "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "RERANK_STRATEGY": "laya",
        "FAISS_TOP_K": 16,
        "RERANK_TOP_K": 6,
        "DOC_PARSER": "opendataloader",
        "LLAMA_NUM_PREDICT": 8192,
        "LLAMA_NUM_PREDICT_THINK": 32768,
    },
}

# 하위 호환 별칭
PRESETS = PROFILE_PRESETS

# ---------------------------------------------------------------------------
# 3-B. 서빙 엔진별 프리셋 — PROFILE 티어별로 엔진 전용 값만
# ---------------------------------------------------------------------------
_ENGINE_ALIASES: Dict[str, str] = {
    "llama.cpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "ollama": "ollama",
    "vllm": "vllm",
    "sglang": "sglang",
    "exllamav3": "exllamav3",
    "exllama": "exllamav3",
    "exllama_v3": "exllamav3",
    "tabbyapi": "exllamav3",
    "mlc_llm": "mlc_llm",
    "mlc": "mlc_llm",
    "mlcllm": "mlc_llm",
    "llamafile": "llamafile",
    "mozilla_llamafile": "llamafile",
    "openai": "openai",
    "openai_compatible": "openai",
    "airllm": "airllm",
    "air_llm": "airllm",
    "air-llm": "airllm",
}

ENGINE_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "llama.cpp": {
        # VARIANT 는 ACCELERATOR(GPU>Vulkan>CPU)를 따르므로 여기에 두지 않음
        "low": {
            "LLAMA_GGUF_REPO": "bartowski/Qwen_Qwen2.5-1.5B-Instruct-GGUF",
            "LLAMA_GGUF_FILE": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            "LLAMA_CTX_SIZE": 2048,
            "LLAMA_NGL": 0,
        },
        "mid": {
            "LLAMA_GGUF_REPO": "bartowski/Qwen_Qwen3.5-9B-GGUF",
            "LLAMA_GGUF_FILE": "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
            "LLAMA_CTX_SIZE": 4096,
            "LLAMA_NGL": 999,
        },
        "high": {
            "LLAMA_GGUF_REPO": "bartowski/Qwen_Qwen2.5-14B-Instruct-GGUF",
            "LLAMA_GGUF_FILE": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            "LLAMA_CTX_SIZE": 8192,
            "LLAMA_NGL": 999,
        },
    },
    "ollama": {
        "low": {"OLLAMA_MODEL": "qwen2.5:1.5b"},
        "mid": {"OLLAMA_MODEL": "qwen2.5:7b"},
        "high": {"OLLAMA_MODEL": "qwen2.5:14b"},
    },
    "vllm": {
        "low": {"VLLM_MODEL": "Qwen/Qwen2.5-1.5B-Instruct"},
        "mid": {"VLLM_MODEL": "Qwen/Qwen2.5-7B-Instruct"},
        "high": {"VLLM_MODEL": "Qwen/Qwen2.5-14B-Instruct"},
    },
    "sglang": {
        "low": {"SGLANG_MODEL": "Qwen/Qwen2.5-1.5B-Instruct"},
        "mid": {"SGLANG_MODEL": "Qwen/Qwen2.5-7B-Instruct"},
        "high": {"SGLANG_MODEL": "Qwen/Qwen2.5-14B-Instruct"},
    },
    "exllamav3": {
        "low": {"EXLLAMA_MODEL": "Qwen/Qwen2.5-1.5B-Instruct"},
        "mid": {"EXLLAMA_MODEL": "Qwen/Qwen2.5-7B-Instruct"},
        "high": {"EXLLAMA_MODEL": "Qwen/Qwen2.5-14B-Instruct"},
    },
    "mlc_llm": {
        "low": {"MLC_LLM_MODEL": "Qwen/Qwen2.5-1.5B-Instruct"},
        "mid": {"MLC_LLM_MODEL": "Qwen/Qwen2.5-7B-Instruct"},
        "high": {"MLC_LLM_MODEL": "Qwen/Qwen2.5-7B-Instruct"},
    },
    "llamafile": {
        "low": {"LLAMAFILE_MODEL": "LLaMA_CPP"},
        "mid": {"LLAMAFILE_MODEL": "LLaMA_CPP"},
        "high": {"LLAMAFILE_MODEL": "LLaMA_CPP"},
    },
    "airllm": {
        "low": {
            "AIRLLM_MODEL": "Qwen/Qwen2.5-7B-Instruct",
            "AIRLLM_COMPRESSION": "4bit",
            "AIRLLM_MAX_LENGTH": 2048,
        },
        "mid": {
            "AIRLLM_MODEL": "Qwen/Qwen2.5-14B-Instruct",
            "AIRLLM_COMPRESSION": None,
            "AIRLLM_MAX_LENGTH": 4096,
        },
        "high": {
            "AIRLLM_MODEL": "Qwen/Qwen2.5-32B-Instruct",
            "AIRLLM_COMPRESSION": None,
            "AIRLLM_MAX_LENGTH": 4096,
        },
    },
    "openai": {},
}


def canonical_serving_engine(name: Optional[str]) -> str:
    """서빙 엔진 이름 정규화."""
    raw = (name or "llama.cpp").strip().lower()
    return _ENGINE_ALIASES.get(raw, raw)


# ---------------------------------------------------------------------------
# 4. 설정 파사드
# ---------------------------------------------------------------------------
class SettingsFacade:
    """
    설정 파사드.

    우선순위 (높음 → 낮음):
      1) 프로세스 환경변수 / .env 사용자 값
      2) ENGINE_PRESETS[SERVING_ENGINE][PROFILE]  — 엔진 전용
      3) PROFILE_PRESETS[PROFILE]                 — RAG·하드웨어 공통
      4) 코드 default

    SERVING_ENGINE 은 PROFILE 이 정하지 않는다. env 또는 default(llama.cpp).
    """

    def __init__(self) -> None:
        """PROFILE·엔진을 해석하고 병합용 프리셋 테이블을 준비한다."""
        self.profile = os.getenv("PROFILE", os.getenv("HARDWARE_PROFILE", "mid")).strip().lower()
        if self.profile not in PROFILE_PRESETS:
            self.profile = "mid"
        self._profile_preset = PROFILE_PRESETS[self.profile]

        # 엔진은 env만 — 프리셋이 엔진을 강제하지 않음
        env_engine = os.getenv("SERVING_ENGINE", "")
        if env_engine is None or str(env_engine).strip() == "":
            self.serving_engine = "llama.cpp"
        else:
            self.serving_engine = canonical_serving_engine(env_engine)

        by_engine = ENGINE_PRESETS.get(self.serving_engine, {})
        self._engine_preset = by_engine.get(self.profile) or by_engine.get("mid") or {}

    def get(self, key: str, default: Any = None) -> Any:
        """env → 엔진 프리셋 → PROFILE 프리셋 → default."""
        if key == "SERVING_ENGINE":
            env_val = os.getenv("SERVING_ENGINE")
            if env_val is not None and env_val != "":
                return canonical_serving_engine(env_val)
            return self.serving_engine

        env_val = os.getenv(key)
        if env_val is not None and env_val != "":
            return env_val
        if key in self._engine_preset:
            return self._engine_preset[key]
        if key in self._profile_preset:
            return self._profile_preset[key]
        return default

    def get_int(self, key: str, default: int) -> int:
        """정수 조회."""
        val = self.get(key, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float) -> float:
        """실수 조회."""
        val = self.get(key, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool) -> bool:
        """불리언 조회."""
        val = str(self.get(key, str(default))).strip().lower()
        return val not in ("0", "false", "no", "off")


settings = SettingsFacade()


def detect_accelerator() -> str:
    """가속 장치 자동 감지: GPU(cuda) > Vulkan > CPU.

    팩토리(get_accelerator_manager) 및 전략 패턴을 통해 가용 가속기를 판별합니다.
    """
    from services.gpu_manager import get_accelerator_manager
    return get_accelerator_manager("auto").accelerator_type


def resolve_accelerator(raw: str) -> str:
    """ACCELERATOR 값 정규화 및 전략 매핑 (auto|gpu|cuda|vulkan|cpu)."""
    from services.gpu_manager import get_accelerator_manager
    return get_accelerator_manager(raw).accelerator_type


# ---------------------------------------------------------------------------
# 5. 파사드 기반 전역 변수 매핑
# ---------------------------------------------------------------------------
PROFILE = settings.profile

ACCELERATOR = resolve_accelerator(str(settings.get("ACCELERATOR", "auto")))
SERVING_ENGINE = canonical_serving_engine(str(settings.get("SERVING_ENGINE", "llama.cpp")))

LLAMA_GGUF_REPO = settings.get("LLAMA_GGUF_REPO", "bartowski/Qwen_Qwen3.5-9B-GGUF")
LLAMA_GGUF_FILE = settings.get("LLAMA_GGUF_FILE", "Qwen_Qwen3.5-9B-Q4_K_M.gguf")

raw_variant = str(settings.get("LLAMA_CPP_VARIANT", "") or "").strip().lower()
if not raw_variant or raw_variant == "auto":
    LLAMA_CPP_VARIANT = ACCELERATOR
else:
    LLAMA_CPP_VARIANT = raw_variant

LLAMA_CTX_SIZE = settings.get_int("LLAMA_CTX_SIZE", 4096)
LLAMA_NGL = settings.get_int("LLAMA_NGL", 999)
LLAMA_MMAP = settings.get_bool("LLAMA_MMAP", True)
LLAMA_SAFETY_MARGIN_MB = settings.get_int("LLAMA_SAFETY_MARGIN_MB", 500)

EMBED_MODEL = settings.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_GGUF_REPO = settings.get("EMBED_GGUF_REPO", "ggml-org/embeddinggemma-300M-GGUF")
EMBED_GGUF_FILE = settings.get("EMBED_GGUF_FILE", "embeddinggemma-300M-Q8_0.gguf")
EMBED_GGUF_PORT = settings.get_int("EMBED_GGUF_PORT", 8085)

# ACCELERATOR를 AI/임베딩 디바이스의 단일 진실 원천(Source of Truth)으로 사용.
# EMBED_DEVICE가 명시적으로 지정된 경우에만 하위 호환으로 존중하고, 기본값은 ACCELERATOR를 따름.
_raw_embed_device = str(settings.get("EMBED_DEVICE", "") or "").strip().lower()
if _raw_embed_device and _raw_embed_device not in ("auto", ""):
    EMBED_DEVICE = _raw_embed_device
else:
    EMBED_DEVICE = "cuda" if ACCELERATOR == "cuda" else "cpu"

RERANK_STRATEGY = settings.get("RERANK_STRATEGY", "laya").strip().lower()
RERANK_MODEL = settings.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANK_ORIGIN_MODEL = settings.get("RERANK_ORIGIN_MODEL", RERANK_MODEL)
RERANK_ORIGIN_SCORER = settings.get("RERANK_ORIGIN_SCORER", "qwen_yesno")
RERANK_ORIGIN_DEVICE = EMBED_DEVICE

RERANK_LAYA_MODEL = settings.get("RERANK_LAYA_MODEL", "convaiinnovations/laya-multilingual")
RERANK_LAYA_DECISION = settings.get("RERANK_LAYA_DECISION", "noul")
RERANK_LAYA_MAX_LENGTH = settings.get_int("RERANK_LAYA_MAX_LENGTH", 1024)

RERANK_CLM_BASE_URL = settings.get("RERANK_CLM_BASE_URL", "http://127.0.0.1:8700").rstrip("/")
RERANK_CLM_TIMEOUT = settings.get_float("RERANK_CLM_TIMEOUT", 10.0)
RERANK_CLM_ACTION_CACHE = settings.get_bool("RERANK_CLM_ACTION_CACHE", True)

FAISS_TOP_K = settings.get_int("FAISS_TOP_K", 8)
BM25_TOP_K = settings.get_int("BM25_TOP_K", 16)
HYBRID_RRF_K = settings.get_int("HYBRID_RRF_K", 60)
RERANK_TOP_K = settings.get_int("RERANK_TOP_K", 4)
HYBRID_BM25 = settings.get_bool("HYBRID_BM25", True)

DOC_PARSER = settings.get("DOC_PARSER", "opendataloader").strip().lower()
OPENDATALOADER_FORMAT = settings.get("OPENDATALOADER_FORMAT", "json")

CHUNK_MIN_CHARS = settings.get_int("CHUNK_MIN_CHARS", 200)
CHUNK_MAX_CHARS = settings.get_int("CHUNK_MAX_CHARS", 800)
CHUNK_OVERLAP_CHARS = settings.get_int("CHUNK_OVERLAP_CHARS", 80)

LLAMA_CPP_BASE_URL = settings.get("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1").rstrip("/")
LLAMA_CPP_MODEL = settings.get("LLAMA_CPP_MODEL", "").strip() or None
LLAMA_CPP_API_KEY = settings.get("LLAMA_CPP_API_KEY", "no-key")
CHAT_TEMPERATURE = settings.get_float("CHAT_TEMPERATURE", 0.2)
LLAMA_NUM_PREDICT = settings.get_int("LLAMA_NUM_PREDICT", 4096)
LLAMA_NUM_PREDICT_THINK = settings.get_int("LLAMA_NUM_PREDICT_THINK", 16384)
THINK_AUTO_MIN_CHARS = settings.get_int("THINK_AUTO_MIN_CHARS", 200)
CHAT_HISTORY_MAX_MESSAGES = settings.get_int("CHAT_HISTORY_MAX_MESSAGES", 20)
REQUEST_TIMEOUT = settings.get_float("REQUEST_TIMEOUT", 600.0)

OLLAMA_BASE_URL = settings.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = settings.get("OLLAMA_MODEL", "qwen2.5:7b").strip() or "qwen2.5:7b"

VLLM_BASE_URL = settings.get("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
VLLM_MODEL = settings.get("VLLM_MODEL", "").strip() or None
VLLM_API_KEY = settings.get("VLLM_API_KEY", "EMPTY")

SGLANG_BASE_URL = settings.get("SGLANG_BASE_URL", "http://localhost:30000").rstrip("/")
SGLANG_MODEL = settings.get("SGLANG_MODEL", "").strip() or None
SGLANG_API_KEY = settings.get("SGLANG_API_KEY", "EMPTY")

EXLLAMA_BASE_URL = settings.get("EXLLAMA_BASE_URL", "http://localhost:5000").rstrip("/")
EXLLAMA_MODEL = settings.get("EXLLAMA_MODEL", "").strip() or None
EXLLAMA_API_KEY = settings.get("EXLLAMA_API_KEY", "admin")

MLC_LLM_BASE_URL = settings.get("MLC_LLM_BASE_URL", "http://localhost:8000").rstrip("/")
MLC_LLM_MODEL = settings.get("MLC_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip() or "Qwen/Qwen2.5-7B-Instruct"
MLC_LLM_API_KEY = settings.get("MLC_LLM_API_KEY", "EMPTY")

LLAMAFILE_BASE_URL = settings.get("LLAMAFILE_BASE_URL", "http://localhost:8080").rstrip("/")
LLAMAFILE_MODEL = settings.get("LLAMAFILE_MODEL", "LLaMA_CPP").strip() or "LLaMA_CPP"
LLAMAFILE_API_KEY = settings.get("LLAMAFILE_API_KEY", "no-key")

AIRLLM_MODEL = settings.get("AIRLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
AIRLLM_DEVICE = str(settings.get("AIRLLM_DEVICE", "cuda" if ACCELERATOR == "cuda" else "cpu")).strip().lower()
AIRLLM_MAX_LENGTH = settings.get_int("AIRLLM_MAX_LENGTH", 2048)
AIRLLM_COMPRESSION = settings.get("AIRLLM_COMPRESSION", None)

MANAGED_SERVING_ENGINES = frozenset({"llama.cpp", "llama_cpp", "llamacpp"})

WEB_HOST = settings.get("WEB_HOST", "0.0.0.0")
WEB_PORT = settings.get_int("WEB_PORT", 8787)


def active_rerank_model() -> str:
    """현재 전략이 실제로 쓰는 모델 id 또는 CLM 주소."""
    if RERANK_STRATEGY == "laya":
        return RERANK_LAYA_MODEL
    if RERANK_STRATEGY == "clm":
        return RERANK_CLM_BASE_URL
    return RERANK_MODEL


def rerank_label() -> str:
    """로그용 리랭크 한 줄. origin의 RERANK_MODEL을 다른 전략에 붙이지 않는다."""
    return f"{RERANK_STRATEGY} ({active_rerank_model()})"


def serving_label() -> str:
    """로그용 서빙 엔진 한 줄. llama.cpp만 이 프로젝트가 받아 기동한다."""
    if SERVING_ENGINE == "airllm":
        return f"airllm ({AIRLLM_MODEL}) · 인메모리 레이어 스트리밍"
    urls = {
        "llama.cpp": LLAMA_CPP_BASE_URL,
        "ollama": OLLAMA_BASE_URL,
        "vllm": VLLM_BASE_URL,
        "sglang": SGLANG_BASE_URL,
        "exllamav3": EXLLAMA_BASE_URL,
        "mlc_llm": MLC_LLM_BASE_URL,
        "llamafile": LLAMAFILE_BASE_URL,
    }
    url = urls.get(SERVING_ENGINE, "")
    if SERVING_ENGINE in MANAGED_SERVING_ENGINES:
        return f"llama.cpp ({url}) · data/serving_engines"
    if url:
        return f"{SERVING_ENGINE} ({url}) · 외부"
    return f"{SERVING_ENGINE} · 외부"


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "DOCS_DIR",
    "REJECTED_DIR",
    "INDEX_DIR",
    "PARSED_DIR",
    "SQLITE_PATH",
    "FAISS_PATH",
    "INDEX_META_PATH",
    "MODELS_DIR",
    "CHAT_MODELS_DIR",
    "EMBED_MODELS_DIR",
    "RERANK_MODELS_DIR",
    "hf_repo_local_name",
    "hf_model_dir",
    "PROFILE_PRESETS",
    "PRESETS",
    "ENGINE_PRESETS",
    "canonical_serving_engine",
    "SettingsFacade",
    "settings",
    "detect_accelerator",
    "resolve_accelerator",
    "PROFILE",
    "ACCELERATOR",
    "SERVING_ENGINE",
    "LLAMA_GGUF_REPO",
    "LLAMA_GGUF_FILE",
    "LLAMA_CPP_VARIANT",
    "LLAMA_CTX_SIZE",
    "LLAMA_NGL",
    "LLAMA_MMAP",
    "LLAMA_SAFETY_MARGIN_MB",
    "EMBED_MODEL",
    "EMBED_GGUF_REPO",
    "EMBED_GGUF_FILE",
    "EMBED_GGUF_PORT",
    "EMBED_DEVICE",
    "RERANK_STRATEGY",
    "RERANK_MODEL",
    "RERANK_ORIGIN_MODEL",
    "RERANK_ORIGIN_SCORER",
    "RERANK_ORIGIN_DEVICE",
    "RERANK_LAYA_MODEL",
    "RERANK_LAYA_DECISION",
    "RERANK_LAYA_MAX_LENGTH",
    "RERANK_CLM_BASE_URL",
    "RERANK_CLM_TIMEOUT",
    "RERANK_CLM_ACTION_CACHE",
    "FAISS_TOP_K",
    "BM25_TOP_K",
    "HYBRID_RRF_K",
    "RERANK_TOP_K",
    "HYBRID_BM25",
    "DOC_PARSER",
    "OPENDATALOADER_FORMAT",
    "CHUNK_MIN_CHARS",
    "CHUNK_MAX_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "LLAMA_CPP_BASE_URL",
    "LLAMA_CPP_MODEL",
    "LLAMA_CPP_API_KEY",
    "CHAT_TEMPERATURE",
    "LLAMA_NUM_PREDICT",
    "LLAMA_NUM_PREDICT_THINK",
    "THINK_AUTO_MIN_CHARS",
    "CHAT_HISTORY_MAX_MESSAGES",
    "REQUEST_TIMEOUT",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "VLLM_API_KEY",
    "SGLANG_BASE_URL",
    "SGLANG_MODEL",
    "SGLANG_API_KEY",
    "EXLLAMA_BASE_URL",
    "EXLLAMA_MODEL",
    "EXLLAMA_API_KEY",
    "MLC_LLM_BASE_URL",
    "MLC_LLM_MODEL",
    "MLC_LLM_API_KEY",
    "LLAMAFILE_BASE_URL",
    "LLAMAFILE_MODEL",
    "LLAMAFILE_API_KEY",
    "AIRLLM_MODEL",
    "AIRLLM_DEVICE",
    "AIRLLM_MAX_LENGTH",
    "AIRLLM_COMPRESSION",
    "MANAGED_SERVING_ENGINES",
    "WEB_HOST",
    "WEB_PORT",
    "active_rerank_model",
    "rerank_label",
    "serving_label",
]
