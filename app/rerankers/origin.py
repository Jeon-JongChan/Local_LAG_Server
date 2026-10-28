"""
origin 리랭커 어댑터 구현체 (Qwen CausalLM / SentenceTransformers CrossEncoder).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .base import BaseRerankerAdapter

from settings import (
    RERANK_MODELS_DIR,
    RERANK_ORIGIN_DEVICE,
    RERANK_ORIGIN_MODEL,
    RERANK_ORIGIN_SCORER,
    hf_model_dir,
)


def _resolve_device() -> str:
    if RERANK_ORIGIN_DEVICE not in ("auto", ""):
        return RERANK_ORIGIN_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class _QwenCausalScorer:
    """Qwen3-Reranker — Yes/No 로짓 점수 계산 (공식 transformers CausalLM)."""

    def __init__(self, local_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(local_dir), padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            str(local_dir), torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
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
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
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

    def close(self) -> None:
        del self.model
        del self.tokenizer


class OriginRerankerAdapter(BaseRerankerAdapter):
    """기존 구현 방식(Qwen / CrossEncoder)을 계승한 어댑터 구현체."""

    backend_name = "origin"

    def __init__(
        self,
        model_id: str = RERANK_ORIGIN_MODEL,
        scorer: str = RERANK_ORIGIN_SCORER,
        device: str = RERANK_ORIGIN_DEVICE,
    ) -> None:
        self.model_id = model_id
        self.scorer_type = scorer
        self.device = _resolve_device() if device in ("auto", "") else device
        self._scorer = None

    def _get_scorer(self):
        if self._scorer is None:
            local = hf_model_dir(self.model_id, RERANK_MODELS_DIR)
            name = self.model_id.lower()
            if "qwen" in name and "rerank" in name:
                print(f"[origin] Qwen 리랭커 로드: CausalLM ({self.device})", flush=True)
                self._scorer = _QwenCausalScorer(local, self.device)
            else:
                from sentence_transformers import CrossEncoder
                print(f"[origin] 일반 CrossEncoder 로드 ({self.device})", flush=True)
                self._scorer = CrossEncoder(str(local), device=self.device)
        return self._scorer

    def _calculate_scores(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[float]:
        """질의-후보쌍 점수 계산."""
        pairs = [(query, str(c.get("text") or "")) for c in candidates]
        scorer = self._get_scorer()
        return scorer.predict(pairs)

    def close(self) -> None:
        if self._scorer is not None and hasattr(self._scorer, "close"):
            self._scorer.close()
        self._scorer = None
