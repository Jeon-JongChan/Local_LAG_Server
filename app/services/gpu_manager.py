"""가속기(CUDA / Vulkan / CPU) 수명주기 및 런타임 관리 모듈.

전략(Strategy), 템플릿 메서드(Template Method), 팩토리(Factory) 디자인 패턴을 적용하여
하드웨어 가속기 탐색, 가용성 검증, 런타임 의존성 자동 설치를 다형적으로 캡슐화합니다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from .base import BaseAcceleratorManager


class CudaAcceleratorManager(BaseAcceleratorManager):
    """NVIDIA CUDA GPU 가속 전략 구현체."""

    def is_available(self) -> bool:
        """물리적 NVIDIA GPU 및 드라이버 통신 가능 여부를 검사합니다."""
        # 1) torch.cuda가 이미 동작 중인 경우
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except ImportError:
            pass

        # 2) nvidia-smi 도구 확인
        nvsmi = shutil.which("nvidia-smi")
        if not nvsmi:
            return False
        try:
            res = subprocess.run(
                [nvsmi],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return res.returncode == 0
        except Exception:
            return False

    def _prepare_runtime(self, force: bool = False) -> bool:
        """NVIDIA GPU 가용 시 CUDA 지원 PyTorch가 미설치되어 있으면 자동 설치합니다."""
        try:
            import torch
            if torch.cuda.is_available() and not force:
                dev_name = torch.cuda.get_device_name(0)
                print(f"[gpu] GPU 가속 활성화됨: {dev_name} (CUDA {torch.version.cuda})", flush=True)
                return True
        except ImportError:
            pass

        print("[gpu] NVIDIA GPU 감지됨. PyTorch CUDA 가속 패키지(cu124) 설치/전환을 진행합니다...", flush=True)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ]
        try:
            res = subprocess.run(cmd, check=True)
            if res.returncode == 0:
                print("[gpu] CUDA 지원 PyTorch 설치가 성공적으로 완료되었습니다.", flush=True)
                return True
        except Exception as exc:
            print(f"[gpu] [경고] CUDA PyTorch 설치 중 오류 발생: {exc}", flush=True)
            print("[gpu] CPU 모드로 계속 실행합니다.", flush=True)

        return False

    @property
    def accelerator_type(self) -> str:
        """가속기 식별자."""
        return "cuda"

    @property
    def torch_device(self) -> str:
        """PyTorch 연산 디바이스."""
        return "cuda"

    @property
    def device_name(self) -> str:
        """표시용 장치 이름."""
        try:
            import torch
            if torch.cuda.is_available():
                return str(torch.cuda.get_device_name(0))
        except Exception:
            pass
        return "NVIDIA CUDA GPU"

    def get_vram_bytes(self) -> int:
        """NVIDIA GPU VRAM 크기(바이트) 반환. torch.cuda 우선, 미가용 시 nvidia-smi 쿼리."""
        # 1) torch.cuda 활용
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            pass

        # 2) nvidia-smi 폴백
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi:
            try:
                res = subprocess.run(
                    [nvsmi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if res.returncode == 0 and res.stdout.strip():
                    mb = int(res.stdout.strip().splitlines()[0].strip())
                    return mb * 1024 * 1024
            except Exception:
                pass
        return 0


class VulkanAcceleratorManager(BaseAcceleratorManager):
    """Vulkan 가속 전략 구현체 (AMD/Intel iGPU 및 범용 GPU)."""

    def is_available(self) -> bool:
        """Vulkan 런타임 라이브러리 존재 여부를 검사합니다."""
        if sys.platform == "win32":
            system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
            return (system32 / "vulkan-1.dll").is_file()
        return shutil.which("vulkaninfo") is not None

    def _prepare_runtime(self, force: bool = False) -> bool:
        """Vulkan은 서빙 엔진(llama.cpp) 차원에서 직접 활용되므로 별도 torch 휠 변경 없음."""
        return True

    @property
    def accelerator_type(self) -> str:
        """가속기 식별자."""
        return "vulkan"

    @property
    def torch_device(self) -> str:
        """PyTorch 연산 디바이스 (Vulkan 환경에서는 CPU 임베딩)."""
        return "cpu"

    @property
    def device_name(self) -> str:
        """표시용 장치 이름."""
        return "Vulkan (Universal GPU)"


class CpuAcceleratorManager(BaseAcceleratorManager):
    """기본 CPU 가속 전략 구현체."""

    def is_available(self) -> bool:
        """CPU는 항상 가용합니다."""
        return True

    def _prepare_runtime(self, force: bool = False) -> bool:
        """추가 런타임 설정 없이 통과합니다."""
        return True

    @property
    def accelerator_type(self) -> str:
        """가속기 식별자."""
        return "cpu"

    @property
    def torch_device(self) -> str:
        """PyTorch 연산 디바이스."""
        return "cpu"

    @property
    def device_name(self) -> str:
        """표시용 장치 이름."""
        return "CPU"


# ---------------------------------------------------------------------------
# 팩토리 함수 (Factory Pattern)
# ---------------------------------------------------------------------------
_accelerator_registry: Dict[str, BaseAcceleratorManager] = {}


def get_accelerator_manager(accelerator: Optional[str] = None) -> BaseAcceleratorManager:
    """
    가속기 매니저 팩토리 함수 (Factory Method Pattern).

    Args:
        accelerator: 'cuda' | 'vulkan' | 'cpu' | 'auto' | None

    Returns:
        BaseAcceleratorManager 구현체 인스턴스 (싱글톤 레지스트리 재사용)
    """
    global _accelerator_registry

    raw = (accelerator or "auto").strip().lower()

    if raw in ("cuda", "gpu", "nvidia"):
        target_key = "cuda"
    elif raw == "vulkan":
        target_key = "vulkan"
    elif raw == "cpu":
        target_key = "cpu"
    else:
        # 'auto' 모드: CUDA -> Vulkan -> CPU 우선순위 자동 탐색
        cuda_mgr = _accelerator_registry.setdefault("cuda", CudaAcceleratorManager())
        if cuda_mgr.is_available():
            return cuda_mgr

        vulkan_mgr = _accelerator_registry.setdefault("vulkan", VulkanAcceleratorManager())
        if vulkan_mgr.is_available():
            return vulkan_mgr

        target_key = "cpu"

    if target_key not in _accelerator_registry:
        if target_key == "cuda":
            _accelerator_registry["cuda"] = CudaAcceleratorManager()
        elif target_key == "vulkan":
            _accelerator_registry["vulkan"] = VulkanAcceleratorManager()
        else:
            _accelerator_registry["cpu"] = CpuAcceleratorManager()

    return _accelerator_registry[target_key]


def reset_accelerator_manager() -> None:
    """테스트 및 설정 변경용 가속기 매니저 레지스트리 초기화."""
    global _accelerator_registry
    _accelerator_registry.clear()


def ensure_gpu_torch(force: bool = False) -> bool:
    """[파사드 편의 함수] 현재 활성 가속기 매니저의 런타임 준비 템플릿 메서드를 호출합니다."""
    manager = get_accelerator_manager()
    return manager.ensure_ready(force=force)


def check_nvidia_gpu() -> bool:
    """[하위 호환 편의 함수] NVIDIA GPU 가용 여부를 반환합니다."""
    return CudaAcceleratorManager().is_available()


def is_torch_cuda_ready() -> bool:
    """[하위 호환 편의 함수] PyTorch CUDA 사용 가능 여부를 반환합니다."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def calculate_safe_ngl(
    vram_bytes: int,
    model_size_bytes: int,
    total_layers: int = 36,
    safety_margin_bytes: int = 500 * 1024 * 1024,
    base_overhead_bytes: int = 250 * 1024 * 1024,
    requested_ngl: int = 999,
) -> int:
    """가용 VRAM, 모델 크기, 레이어 수를 기반으로 OOM 없는 안전한 GPU 오프로딩 레이어 수(ngl)를 동적으로 산출합니다.

    Args:
        vram_bytes: 감지된 물리 GPU VRAM 크기(바이트).
        model_size_bytes: 로드할 GGUF 모델 가중치 크기(바이트).
        total_layers: 모델의 총 레이어 수 (기본 36, Qwen 9B 기준).
        safety_margin_bytes: OS/DWM 디스플레이용 안전 여유 마진 (기본 500MB).
        base_overhead_bytes: CUDA 컨텍스트 및 런타임 스크래치 버퍼 (기본 250MB).
        requested_ngl: 사용자가 설정한 목표 ngl (기본 999, 전체 오프로드 요청).

    Returns:
        int: 안전한 GPU 오프로드 레이어 수.
    """
    # 1. CPU 전용이거나 사용자가 명시적으로 0 또는 음수를 요청한 경우
    if vram_bytes <= 0 or requested_ngl <= 0:
        return 0

    # 2. VRAM이 안전 마진 이하인 경우 OOM 방지를 위해 CPU(0) 반환
    if vram_bytes <= safety_margin_bytes:
        return 0

    # 3. 사용자가 999(자동 풀 오프로드 요청)가 아닌 구체적인 양수 ngl을 명시한 경우 사용자 설정 존중
    if requested_ngl != 999:
        return requested_ngl

    usable_vram = vram_bytes - safety_margin_bytes

    # 4. 모델 크기를 알 수 없는 경우 (0 이하)
    if model_size_bytes <= 0:
        if vram_bytes <= 3 * 1024 * 1024 * 1024:
            return 6
        return 999

    # 5. 모델 전체가 가용 VRAM에 여유 있게 들어가는 경우 (풀 오프로드)
    if usable_vram >= (model_size_bytes + base_overhead_bytes):
        return 999

    # 6. 부분 GPU 오프로딩 (Partial Offload) 계산
    net_vram = max(0, usable_vram - base_overhead_bytes)
    if net_vram <= 0:
        return 0

    layers = max(1, total_layers)
    bytes_per_layer = model_size_bytes / layers
    calculated_layers = int(net_vram // bytes_per_layer)
    if calculated_layers <= 0:
        return 0

    # 7. 2GB~3GB 저사양 GPU 특화 클램핑 (안전 범위 4~8 레이어)
    if vram_bytes <= 3 * 1024 * 1024 * 1024:
        calculated_layers = max(4, min(8, calculated_layers))

    return min(999, calculated_layers)


__all__ = [
    "BaseAcceleratorManager",
    "CudaAcceleratorManager",
    "VulkanAcceleratorManager",
    "CpuAcceleratorManager",
    "get_accelerator_manager",
    "reset_accelerator_manager",
    "ensure_gpu_torch",
    "check_nvidia_gpu",
    "is_torch_cuda_ready",
    "calculate_safe_ngl",
]
