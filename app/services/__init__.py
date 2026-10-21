"""services 패키지 — 모델/엔진 매니저 싱글톤.

구현 각 1개. get_* 는 편의 싱글톤이며 다중 provider Factory 가 아니다.
BaseModelManager / BaseEngineManager 에 Template(ensure_*) 이 있다.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseAcceleratorManager, BaseEngineManager, BaseModelManager

_active_model_manager: Optional[Any] = None
_active_engine_manager: Optional[Any] = None


def get_model_manager() -> BaseModelManager:
    """HuggingFace 모델 매니저 싱글톤."""
    global _active_model_manager
    if _active_model_manager is None:
        from .hf_model_manager import HuggingFaceModelManager
        _active_model_manager = HuggingFaceModelManager()
    return _active_model_manager


def reset_model_manager() -> None:
    """테스트용 모델 매니저 싱글톤 초기화."""
    global _active_model_manager
    _active_model_manager = None


def get_engine_manager() -> BaseEngineManager:
    """llama-server 엔진 매니저 싱글톤."""
    global _active_engine_manager
    if _active_engine_manager is None:
        from .llama_engine_manager import LlamaEngineManager
        _active_engine_manager = LlamaEngineManager()
    return _active_engine_manager


def reset_engine_manager() -> None:
    """테스트용 엔진 매니저 싱글톤 초기화."""
    global _active_engine_manager
    _active_engine_manager = None


from .gpu_manager import (
    BaseAcceleratorManager,
    CudaAcceleratorManager,
    VulkanAcceleratorManager,
    CpuAcceleratorManager,
    calculate_safe_ngl,
    check_nvidia_gpu,
    ensure_gpu_torch,
    get_accelerator_manager,
    is_torch_cuda_ready,
    reset_accelerator_manager,
)
from .memory_manager import (
    AirLlmMemoryStrategy,
    BaseEngineMemoryStrategy,
    EngineMemoryOptions,
    ExLlamaV3MemoryStrategy,
    LlamaCppMemoryStrategy,
    MemoryBudget,
    MemoryOptimizationFacade,
    OllamaMemoryStrategy,
    VllmMemoryStrategy,
    get_memory_facade,
    get_memory_strategy,
    reset_memory_facade,
    reset_memory_strategies,
)

__all__ = [
    "BaseModelManager",
    "BaseEngineManager",
    "BaseAcceleratorManager",
    "HuggingFaceModelManager",
    "LlamaEngineManager",
    "CudaAcceleratorManager",
    "VulkanAcceleratorManager",
    "CpuAcceleratorManager",
    "calculate_safe_ngl",
    "get_model_manager",
    "reset_model_manager",
    "get_engine_manager",
    "reset_engine_manager",
    "get_accelerator_manager",
    "reset_accelerator_manager",
    "ensure_gpu_torch",
    "check_nvidia_gpu",
    "is_torch_cuda_ready",
    "MemoryBudget",
    "EngineMemoryOptions",
    "BaseEngineMemoryStrategy",
    "LlamaCppMemoryStrategy",
    "VllmMemoryStrategy",
    "OllamaMemoryStrategy",
    "ExLlamaV3MemoryStrategy",
    "AirLlmMemoryStrategy",
    "MemoryOptimizationFacade",
    "get_memory_facade",
    "reset_memory_facade",
    "get_memory_strategy",
    "reset_memory_strategies",
]

