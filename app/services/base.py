"""서비스 라이프사이클 관리 베이스.

ModelManager: Template Method (ensure_model / ensure_file).
EngineManager: Template Method (ensure_running) + Port(인터페이스).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseModelManager(ABC):
    """모델 다운로드 및 파일 준비 관리자."""

    def ensure_model(self, repo_id: str, local_dir: Path) -> Path:
        """[템플릿] 로컬 준비 확인 → 없으면 다운로드 → 재검증."""
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
        """[템플릿] 단일 파일 준비 확인 → 없으면 다운로드 → 재검증."""
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
        """모델 스냅샷 다운로드."""
        pass

    @abstractmethod
    def _download_single_file(self, repo_id: str, filename: str, local_dir: Path) -> None:
        """단일 파일 다운로드."""
        pass


class BaseEngineManager(ABC):
    """서빙 엔진 프로세스 수명주기 관리자."""

    def ensure_running(self, timeout: float = 120.0) -> bool:
        """[템플릿] 프로세스 기동 확인(미기동 시 바이너리 준비 후 start) → 준비 완료(200 OK) 대기."""
        if not self.is_running():
            self.ensure_binary()
            if not self.start():
                return False
        return self.wait_until_ready(timeout=timeout)

    @abstractmethod
    def is_running(self) -> bool:
        """서버 프로세스 및 포트 활성 여부."""
        pass

    @abstractmethod
    def wait_until_ready(self, timeout: float = 120.0) -> bool:
        """서버가 실제 요청을 수신할 준비가 완료(HTTP 200)될 때까지 대기."""
        pass

    @abstractmethod
    def ensure_binary(self) -> Path:
        """서빙 바이너리 존재 확인 및 자동 설치."""
        pass

    @abstractmethod
    def start(self) -> bool:
        """서버 프로세스 기동."""
        pass


class BaseAcceleratorManager(ABC):
    """하드웨어 가속기 탐색 및 런타임 준비 관리자 베이스 (Strategy + Template Method)."""

    def ensure_ready(self, force: bool = False) -> bool:
        """[템플릿 메서드] 가속기 하드웨어 가용성 검증 → 필요 시 런타임 의존성(CUDA 등) 준비."""
        if not self.is_available():
            return False
        return self._prepare_runtime(force=force)

    @abstractmethod
    def is_available(self) -> bool:
        """물리적/시스템 하드웨어 가속기 가용 여부."""
        pass

    @abstractmethod
    def _prepare_runtime(self, force: bool = False) -> bool:
        """가속기를 구동하기 위한 소프트웨어 환경(예: PyTorch CUDA 휠 등) 준비 로직."""
        pass

    @property
    @abstractmethod
    def accelerator_type(self) -> str:
        """가속기 식별자 ('cuda' | 'vulkan' | 'cpu')."""
        pass

    @property
    @abstractmethod
    def torch_device(self) -> str:
        """PyTorch 텐서 연산에 사용할 디바이스 문자열 ('cuda' | 'cpu')."""
        pass

    @property
    @abstractmethod
    def device_name(self) -> str:
        """사용자 표시용 장치 이름."""
        pass

    def get_vram_bytes(self) -> int:
        """가용 GPU VRAM(바이트)을 반환합니다. 미지원 또는 CPU 환경에서는 0 반환."""
        return 0
