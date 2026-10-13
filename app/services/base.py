"""
서비스 및 라이프사이클 관리 추상 베이스 어댑터 모듈.

모델 다운로드/준비(ModelManager)와 서빙 엔진 바이너리/프로세스 수명주기(EngineManager)의
추상 인터페이스와 템플릿 메서드를 정의합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseModelManager(ABC):
    """모델 다운로드 및 파일 준비 관리자 추상 베이스 클래스."""

    def ensure_model(self, repo_id: str, local_dir: Path) -> Path:
        """
        [템플릿 메서드] 모델 로컬 준비 확인 및 부재 시 다운로드.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        if self._is_model_ready(local_dir):
            return local_dir

        print(f"[service_manager] 모델 준비 시작: {repo_id} -> {local_dir}", flush=True)
        self._download_model(repo_id, local_dir)

        if not self._is_model_ready(local_dir):
            raise RuntimeError(f"모델 다운로드 후에도 유효하지 않습니다: {local_dir}")
        print(f"[service_manager] 모델 준비 완료: {local_dir}", flush=True)
        return local_dir

    def ensure_file(self, repo_id: str, filename: str, local_dir: Path) -> Path:
        """
        [템플릿 메서드] 단일 모델 파일(GGUF 등) 준비 확인 및 다운로드.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        target_path = local_dir / filename
        if target_path.is_file() and target_path.stat().st_size > 0:
            return target_path

        print(f"[service_manager] 파일 다운로드 시작: {repo_id}/{filename} -> {target_path}", flush=True)
        self._download_single_file(repo_id, filename, local_dir)

        if not (target_path.is_file() and target_path.stat().st_size > 0):
            raise RuntimeError(f"파일 다운로드 실패: {target_path}")
        print(f"[service_manager] 파일 준비 완료: {target_path}", flush=True)
        return target_path

    @abstractmethod
    def _is_model_ready(self, local_dir: Path) -> bool:
        """로컬 모델 디렉터리가 로드 가능한 상태인지 검사."""
        pass

    @abstractmethod
    def _download_model(self, repo_id: str, local_dir: Path) -> None:
        """서브클래스별 모델 스냅샷 다운로드 구현."""
        pass

    @abstractmethod
    def _download_single_file(self, repo_id: str, filename: str, local_dir: Path) -> None:
        """서브클래스별 단일 파일 다운로드 구현."""
        pass


class BaseEngineManager(ABC):
    """서빙 엔진 런타임/프로세스 수명주기 관리자 추상 베이스 클래스."""

    @abstractmethod
    def is_running(self) -> bool:
        """서버 프로세스 및 포트 활성 여부 확인."""
        pass

    @abstractmethod
    def ensure_binary(self) -> Path:
        """서빙 바이너리 존재 확인 및 자동 설치."""
        pass

    @abstractmethod
    def start(self) -> bool:
        """서버 프로세스 기동."""
        pass

