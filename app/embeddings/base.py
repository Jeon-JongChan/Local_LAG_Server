"""
통합 텍스트 임베딩 모듈.

Qwen, Gemma, BGE 등 다양한 임베딩 모델을 단일 클래스에서 지원합니다.
- 일반 HF 모델: SentenceTransformer (PyTorch / CUDA / CPU)
- GGUF 모델: llama-server --embedding (C++ / CUDA 고속 연산)
- 템플릿 메서드: embed_query, embed_documents (배치 슬라이싱 + 체크포인트 재개)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import numpy as np

from settings import (
    EMBED_GGUF_FILE,
    EMBED_GGUF_PORT,
    EMBED_GGUF_REPO,
    EMBED_MODELS_DIR,
    hf_model_dir,
)


def is_gguf_model(model_name: str) -> bool:
    """GGUF 임베딩 모델 여부 판별."""
    lower = model_name.strip().lower()
    return (
        "embeddinggemma" in lower
        or lower.endswith(".gguf")
        or "gguf" in lower
        or ":" in lower
    )


def ensure_gguf_model(
    repo_id: str = EMBED_GGUF_REPO,
    filename: str = EMBED_GGUF_FILE,
    target_dir: Path = EMBED_MODELS_DIR,
) -> Path:
    """GGUF 임베딩 모델 단일 파일이 로컬에 없으면 Hugging Face에서 다운로드합니다."""
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / filename
    if local_path.is_file() and local_path.stat().st_size > 10_000_000:
        return local_path

    print(f"[embedding-gguf] GGUF 임베딩 모델 다운로드 시작: {repo_id}/{filename} ...", flush=True)
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(target_dir),
    )
    return local_path


class EmbeddingAdapter:
    """통합 텍스트 임베딩 어댑터 클래스."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._is_gguf = is_gguf_model(model_name)
        self._dimension: Optional[int] = 768 if self._is_gguf else None
        self._model: Optional[Any] = None
        self._server_proc: Optional[subprocess.Popen] = None
        self._port = EMBED_GGUF_PORT
        self._base_url = f"http://127.0.0.1:{self._port}"

    # --- SentenceTransformer 로딩 및 추론 ---
    def _ensure_hf_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        local_path = hf_model_dir(self.model_name, EMBED_MODELS_DIR)
        load_target = str(local_path) if local_path.is_dir() else self.model_name
        kwargs: dict = {"device": self.device}
        if "qwen" in self.model_name.lower():
            kwargs["tokenizer_kwargs"] = {"padding_side": "left"}

        print(f"[embedding] 모델 로드 중: {self.model_name} ({self.device})", flush=True)
        self._model = SentenceTransformer(load_target, **kwargs)
        return self._model

    # --- GGUF llama-server 로딩 및 추론 ---
    def _is_server_alive(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._base_url}/health")
            with urllib.request.urlopen(req, timeout=1.5) as res:
                return bool(res.status == 200)
        except Exception:
            return False

    def _ensure_gguf_server(self) -> None:
        if self._is_server_alive():
            return
        from services.llama_engine_manager import LlamaEngineManager

        engine_mgr = LlamaEngineManager()
        exe = engine_mgr.ensure_binary()
        model_path = ensure_gguf_model()
        ngl = 999 if self.device == "cuda" else 0

        cmd = [
            str(exe),
            "-m",
            str(model_path),
            "--embedding",
            "--port",
            str(self._port),
            "--host",
            "127.0.0.1",
            "-ngl",
            str(ngl),
            "-c",
            "4096",
            "-b",
            "4096",
            "-ub",
            "4096",
        ]
        print(f"[embedding-gguf] GGUF 임베딩 서버 기동 중 (포트 {self._port}): {' '.join(cmd)}", flush=True)
        popen_kw: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            detached = 0x00000008
            new_group = 0x00000200
            breakaway = 0x01000000
            popen_kw["close_fds"] = True
            try:
                self._server_proc = subprocess.Popen(
                    cmd, creationflags=detached | new_group | breakaway, **popen_kw
                )
            except OSError:
                self._server_proc = subprocess.Popen(
                    cmd, creationflags=detached | new_group, **popen_kw
                )
        else:
            self._server_proc = subprocess.Popen(cmd, start_new_session=True, **popen_kw)

        for _ in range(50):
            time.sleep(0.5)
            if self._is_server_alive():
                print(f"[embedding-gguf] GGUF 임베딩 서버 기동 완료 ({self._base_url})", flush=True)
                return
        raise RuntimeError(f"GGUF 임베딩 서버 기동 실패 (포트 {self._port})")

    def _post_gguf_embeddings(self, inputs: List[str]) -> List[List[float]]:
        self._ensure_gguf_server()
        url = f"{self._base_url}/v1/embeddings"
        sub_batch_size = 4
        all_embeddings: List[List[float]] = []
        for i in range(0, len(inputs), sub_batch_size):
            sub_inputs = inputs[i : i + sub_batch_size]
            payload = json.dumps({"input": sub_inputs, "model": self.model_name}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120.0) as res:
                data = json.loads(res.read().decode("utf-8"))
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            all_embeddings.extend([it["embedding"] for it in items])
        return all_embeddings

    # --- 핵심 임베딩 오버라이드 지점 ---
    def _embed_single_query(self, query: str) -> np.ndarray:
        if self._is_gguf:
            vecs = self._post_gguf_embeddings([query])
            if not vecs:
                return np.zeros((self._dimension or 768,), dtype=np.float32)
            vec = np.asarray(vecs[0], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 0 else vec

        model = self._ensure_hf_loaded()
        kwargs = {"normalize_embeddings": True, "show_progress_bar": False, "convert_to_numpy": True}
        if "qwen" in self.model_name.lower():
            kwargs["prompt_name"] = "query"
        vec = model.encode([query], **kwargs)[0]
        return np.asarray(vec, dtype=np.float32)

    def _embed_batch_documents(self, documents: List[str]) -> np.ndarray:
        if not documents:
            dim = self._dimension or 1024
            return np.zeros((0, dim), dtype=np.float32)

        if self._is_gguf:
            raw_vecs = self._post_gguf_embeddings(documents)
            vecs = np.asarray(raw_vecs, dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms

        model = self._ensure_hf_loaded()
        kwargs = {"normalize_embeddings": True, "show_progress_bar": False, "convert_to_numpy": True}
        vecs = model.encode(documents, **kwargs)
        return np.asarray(vecs, dtype=np.float32)

    # --- 공통 템플릿 메서드 ---
    def embed_query(self, query: str) -> np.ndarray:
        cleaned = (query or "").strip()
        if not cleaned:
            dim = self._dimension or 1024
            return np.zeros((dim,), dtype=np.float32)
        vec = self._embed_single_query(cleaned)
        return np.asarray(vec, dtype=np.float32)

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 16,
        checkpoint_path: Optional[Path] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> np.ndarray:
        if not texts:
            dim = self._dimension or 1024
            return np.zeros((0, dim), dtype=np.float32)

        total = len(texts)
        parts: List[np.ndarray] = []
        start_idx = 0

        if checkpoint_path and checkpoint_path.is_file():
            try:
                loaded = np.load(checkpoint_path)
                if loaded.ndim == 2 and 0 < loaded.shape[0] <= total:
                    start_idx = int(loaded.shape[0])
                    parts.append(loaded.astype(np.float32))
                    if on_progress:
                        on_progress(
                            start_idx,
                            total,
                            f"체크포인트 복원 ({start_idx}/{total}개 완료됨)",
                        )
            except Exception:
                parts.clear()
                start_idx = 0

        for start in range(start_idx, total, batch_size):
            batch = list(texts[start : start + batch_size])
            batch_vecs = self._embed_batch_documents(batch)
            parts.append(np.asarray(batch_vecs, dtype=np.float32))

            done = min(start + len(batch), total)
            if on_progress:
                on_progress(done, total, f"임베딩 진행 중: {done}/{total}")

            if checkpoint_path and (done % (batch_size * 10) == 0 or done == total):
                try:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    current_arr = np.vstack(parts)
                    tmp_cp = checkpoint_path.with_suffix(".tmp.npy")
                    np.save(tmp_cp, current_arr)
                    tmp_cp.replace(checkpoint_path)
                except Exception:
                    pass

        if len(parts) == 1:
            return parts[0]
        return np.vstack(parts)

    def close(self) -> None:
        if self._server_proc is not None:
            try:
                self._server_proc.terminate()
            except Exception:
                pass
            self._server_proc = None


BaseEmbeddingAdapter = EmbeddingAdapter
