"""Hugging Face 기반 모델 다운로드 관리자."""

from __future__ import annotations

from pathlib import Path

from .base import BaseModelManager


class HuggingFaceModelManager(BaseModelManager):
    """huggingface_hub API를 사용하는 모델 관리자."""

    def _is_model_ready(self, local_dir: Path) -> bool:
        """로컬 모델 디렉터리가 로드 가능한지 검사."""
        if not local_dir.is_dir():
            return False
        if (local_dir / "config.json").is_file() or (local_dir / "mlc-chat-config.json").is_file():
            return True
        for pattern in ("*.safetensors", "*.bin", "*.onnx", "*.gguf"):
            if list(local_dir.glob(pattern)):
                return True
        return False

    def _download_model(self, repo_id: str, local_dir: Path) -> None:
        """모델 스냅샷 다운로드."""
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))

    def _download_single_file(self, repo_id: str, filename: str, local_dir: Path) -> None:
        """단일 파일 다운로드."""
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
