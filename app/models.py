"""
임베딩·리랭킹 — 로컬 data/models/ 로드 (없으면 Hugging Face에서 자동 다운로드).

기본: Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B
다운로드는 여기 또는 scripts/ensure_ml_models.py / local-rag.bat models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from settings import (
    EMBED_DEVICE,
    EMBED_MODEL,
    EMBED_MODELS_DIR,
    RERANK_MODEL,
    RERANK_MODELS_DIR,
    hf_model_dir,
)

_embedder = None
_reranker = None
_reranker_backend: Optional[str] = None  # "cross_encoder" | "qwen_causal"


def _resolve_device() -> str:
    if EMBED_DEVICE not in ("auto", ""):
        return EMBED_DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _is_ready(local_dir: Path) -> bool:
    if not local_dir.is_dir():
        return False
    if (local_dir / "config.json").is_file():
        return True
    for pattern in ("*.safetensors", "*.bin", "*.onnx"):
        if list(local_dir.glob(pattern)):
            return True
    return False


def ensure_model_downloaded(repo_id: str, local_dir: Path) -> Path:
    """로컬에 없으면 huggingface_hub 로 스냅샷 다운로드."""
    local_dir.mkdir(parents=True, exist_ok=True)
    if _is_ready(local_dir):
        return local_dir

    print(f"[models] 다운로드 중 → {local_dir}", flush=True)
    print(f"[models]   repo: {repo_id}", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    if not _is_ready(local_dir):
        raise RuntimeError(f"모델 다운로드 후에도 유효하지 않습니다: {local_dir}")
    print(f"[models] 완료: {local_dir}", flush=True)
    return local_dir


def _local_model(repo_id: str, category: Path) -> Path:
    local = hf_model_dir(repo_id, category)
    return ensure_model_downloaded(repo_id, local)


def current_embed_device() -> str:
    """임베딩·리랭커에 쓰이는 torch 디바이스 (cpu / cuda)."""
    return _resolve_device()


def _is_qwen_embedding(repo_id: str) -> bool:
    name = repo_id.lower()
    return "qwen" in name and "embed" in name


def _is_qwen_reranker(repo_id: str) -> bool:
    name = repo_id.lower()
    return "qwen" in name and "rerank" in name


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        local = _local_model(EMBED_MODEL, EMBED_MODELS_DIR)
        device = _resolve_device()
        kwargs = {"device": device}
        if _is_qwen_embedding(EMBED_MODEL):
            # Qwen3-Embedding: left padding 권장
            kwargs["tokenizer_kwargs"] = {"padding_side": "left"}
        _embedder = SentenceTransformer(str(local), **kwargs)
    return _embedder


class _QwenCausalReranker:
    """Qwen3-Reranker — yes/no 로짓 (공식 transformers 경로)."""

    def __init__(self, local_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(local_dir), padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            # CausalLM 배치 패딩용 (eos 를 pad 로 사용)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(local_dir), trust_remote_code=True
        ).to(device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        # CPU 저사양: 문서 본문 과다 토큰 방지
        self.max_length = 2048
        self.batch_size = 4 if device == "cuda" else 2
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            "and the Instruct provided. Note that the answer can only be "
            '"yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        self._torch = torch

    def _format(self, query: str, doc: str) -> str:
        # 과도한 청크는 앞부분만 (리랭크 입력 길이 제한)
        if len(doc) > 1500:
            doc = doc[:1500]
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def _score_batch(self, formatted: List[str]) -> List[float]:
        torch = self._torch
        max_body = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        inputs = self.tokenizer(
            formatted,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max(64, max_body),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :]
            true_v = logits[:, self.token_true_id]
            false_v = logits[:, self.token_false_id]
            stacked = torch.stack([false_v, true_v], dim=1)
            scores = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
        return [float(x) for x in scores.tolist()]

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        formatted = [self._format(q, d) for q, d in pairs]
        out: List[float] = []
        for start in range(0, len(formatted), self.batch_size):
            batch = formatted[start : start + self.batch_size]
            out.extend(self._score_batch(batch))
        return out


def get_reranker():
    global _reranker, _reranker_backend
    if _reranker is None:
        local = _local_model(RERANK_MODEL, RERANK_MODELS_DIR)
        device = _resolve_device()

        if _is_qwen_reranker(RERANK_MODEL):
            # ST 5.3 이하는 SequenceClassification 로 잘못 로드됨 → 항상 공식 CausalLM 경로
            print(f"[models] Qwen 리랭커: transformers CausalLM ({device})", flush=True)
            _reranker = _QwenCausalReranker(local, device)
            _reranker_backend = "qwen_causal"
        else:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(str(local), device=device)
            _reranker_backend = "cross_encoder"
    return _reranker


def embed_texts(
    texts: Sequence[str],
    *,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    is_query: bool = False,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model = get_embedder()
    device = _resolve_device()
    batch_size = 32 if device == "cuda" else 8 if _is_qwen_embedding(EMBED_MODEL) else 16
    total = len(texts)
    parts: List[np.ndarray] = []

    encode_kwargs = {
        "normalize_embeddings": True,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    # Qwen3: 질의만 query 프롬프트 (문서는 프롬프트 없음)
    if is_query and _is_qwen_embedding(EMBED_MODEL):
        encode_kwargs["prompt_name"] = "query"

    for start in range(0, total, batch_size):
        batch = list(texts[start : start + batch_size])
        vectors = model.encode(batch, **encode_kwargs)
        parts.append(np.asarray(vectors, dtype=np.float32))
        done = min(start + len(batch), total)
        if on_progress:
            on_progress(done, total, f"배치 {done}/{total}")

    if len(parts) == 1:
        return parts[0]
    return np.vstack(parts)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query], is_query=True)[0]


def rerank(query: str, candidates: List[dict], top_k: int) -> List[dict]:
    if not candidates:
        return []
    if top_k <= 0:
        return list(candidates)
    if len(candidates) <= top_k:
        return candidates

    pairs: List[Tuple[str, str]] = [(query, c["text"]) for c in candidates]
    model = get_reranker()
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    out: List[dict] = []
    for item, score in ranked[:top_k]:
        enriched = dict(item)
        enriched["rerank_score"] = float(score)
        out.append(enriched)
    return out
