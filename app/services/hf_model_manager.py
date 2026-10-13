"""Hugging Face 기반 모델 다운로드 관리자 어댑터 구현체.

일반 HF 모델뿐만 아니라 MLC LLM 전용 가중치 탐색 및 자동 다운로드/로컬 변환을 지원합니다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional
from .base import BaseModelManager


class HuggingFaceModelManager(BaseModelManager):
    """huggingface_hub API를 사용하는 모델 관리자."""

    def _is_model_ready(self, local_dir: Path) -> bool:
        if not local_dir.is_dir():
            return False
        if (local_dir / "config.json").is_file() or (local_dir / "mlc-chat-config.json").is_file():
            return True
        for pattern in ("*.safetensors", "*.bin", "*.onnx", "*.gguf"):
            if list(local_dir.glob(pattern)):
                return True
        return False

    def _download_model(self, repo_id: str, local_dir: Path) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))

    def _download_single_file(self, repo_id: str, filename: str, local_dir: Path) -> None:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

    def resolve_mlc_model(self, model_name: str, target_dir: Path) -> Path:
        """
        MLC LLM 가중치 자동 탐색 및 변환 파이프라인.

        1. 로컬 캐시에 mlc-chat-config.json이 이미 존재하는지 확인
        2. Hugging Face mlc-ai/ 또는 {model_name}-MLC 저장소 자동 검색 및 다운로드
        3. 온라인 저장소가 없을 경우 로컬 mlc_llm convert_weight CLI 변환 시도
        """
        if (target_dir / "mlc-chat-config.json").is_file():
            print(f"[MLC] 이미 준비된 로컬 MLC 모델 사용: {target_dir}")
            return target_dir

        try:
            from huggingface_hub import HfApi
            api = HfApi()
        except ImportError:
            api = None

        base_name = model_name.strip().split("/")[-1]
        candidates = [
            model_name,
            f"mlc-ai/{base_name}-q4f16_1-MLC",
            f"mlc-ai/{base_name}-q0f16-MLC",
            f"{model_name}-MLC",
            f"mlc-ai/{base_name}-MLC",
        ]

        # 1) Hugging Face Hub에서 사전 컴파일된 MLC 모델 탐색
        found_repo: Optional[str] = None
        if api:
            for cand in candidates:
                try:
                    info = api.repo_info(repo_id=cand)
                    if info:
                        found_repo = cand
                        print(f"[MLC] Hugging Face MLC 저장소 발견: {cand}")
                        break
                except Exception:
                    continue

        if found_repo:
            print(f"[MLC] 사전 변환된 MLC 가중치 다운로드 시작 ({found_repo}) -> {target_dir}")
            self.ensure_model(found_repo, target_dir)
            return target_dir

        # 2) 저장소에 없을 경우 mlc_llm CLI 로컬 변환 시도
        print(f"[MLC] 온라인 MLC 저장소를 찾을 수 없어 로컬 변환을 시도합니다: {model_name}")
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [
                sys.executable,
                "-m",
                "mlc_llm",
                "convert_weight",
                str(model_name),
                "--quantization",
                "q4f16_1",
                "-o",
                str(target_dir),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                print(f"[MLC] 로컬 가중치 변환 성공: {target_dir}")
                return target_dir
            else:
                print(f"[MLC 경고] 로컬 변환 실패: {res.stderr[:200]}")
        except Exception as e:
            print(f"[MLC 경고] mlc_llm 실행 실패 ({e})")

        return target_dir
