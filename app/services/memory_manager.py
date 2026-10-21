"""다중 서빙 엔진 메모리 최적화 파사드 및 전략 모듈.

local-rag 5대 디자인 패턴 적용:
- Facade: MemoryOptimizationFacade (단일 진입점 파사드)
- Strategy: BaseEngineMemoryStrategy (LlamaCpp, Vllm, Ollama, ExLlamaV3, AirLlm)
- Factory: get_memory_strategy(), reset_memory_strategies(), get_memory_facade(), reset_memory_facade()
- Template Method: BaseEngineMemoryStrategy.build_options() -> _build_engine_parameters()
- Adapter: 서빙 엔진 및 가속기 어댑터 연결
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from settings import settings
from .base import BaseAcceleratorManager
from . import gpu_manager


@dataclass
class MemoryBudget:
    """하드웨어 사양 및 가용 리소스 분석 후 산출된 메모리 예산 컨테이너."""
    total_vram_bytes: int
    usable_vram_bytes: int
    safety_margin_bytes: int
    is_gpu_available: bool
    accelerator_type: str
    model_size_bytes: int = 0
    total_layers: int = 36


@dataclass
class EngineMemoryOptions:
    """엔진별 최적화 메모리 실행 옵션 컨테이너."""
    engine_type: str
    ngl: int = 0
    use_mmap: bool = True
    ctx_size: int = 4096
    threads: int = 4
    cli_args: List[str] = field(default_factory=list)
    payload_options: Dict[str, Any] = field(default_factory=dict)
    extra_config: Dict[str, Any] = field(default_factory=dict)


class BaseEngineMemoryStrategy(ABC):
    """엔진별 메모리 파라미터 빌더 베이스 전략 추상 클래스 (Template Method Pattern)."""

    def __init__(self, engine_type: str) -> None:
        """전략 인스턴스 초기화."""
        self.engine_type = engine_type

    def build_options(
        self,
        budget: MemoryBudget,
        user_options: Optional[Dict[str, Any]] = None,
    ) -> EngineMemoryOptions:
        """[템플릿 메서드] 공통 사용자 옵션 정규화 후 서브클래스 전용 파라미터 빌드 호출."""
        user_opts = user_options or {}
        return self._build_engine_parameters(budget, user_opts)

    @abstractmethod
    def _build_engine_parameters(
        self,
        budget: MemoryBudget,
        user_opts: Dict[str, Any],
    ) -> EngineMemoryOptions:
        """각 서빙 엔진 특화 파라미터 조립 단계 (하위 클래스 구현 필수)."""
        pass


class LlamaCppMemoryStrategy(BaseEngineMemoryStrategy):
    """llama.cpp 서빙 엔진 메모리 최적화 전략."""

    def __init__(self) -> None:
        super().__init__("llama.cpp")

    def _build_engine_parameters(
        self,
        budget: MemoryBudget,
        user_opts: Dict[str, Any],
    ) -> EngineMemoryOptions:
        """llama.cpp용 CLI 인자 및 ngl/mmap 최적화 옵션 빌드."""
        # 1. ngl 계산 (사용자 지정 옵션 > 환경변수 LLAMA_NGL > 기본 동적 목표 999)
        raw_env_ngl = os.getenv("LLAMA_NGL")
        if "ngl" in user_opts:
            target_ngl = user_opts["ngl"]
        elif raw_env_ngl is not None and raw_env_ngl.strip() != "":
            try:
                target_ngl = int(raw_env_ngl)
            except ValueError:
                target_ngl = 999
        else:
            target_ngl = 999

        if not budget.is_gpu_available or target_ngl == 0:
            ngl = 0
        else:
            ngl = gpu_manager.calculate_safe_ngl(
                vram_bytes=budget.total_vram_bytes,
                model_size_bytes=budget.model_size_bytes,
                total_layers=budget.total_layers,
                safety_margin_bytes=budget.safety_margin_bytes,
                requested_ngl=target_ngl,
            )

        # 2. mmap 여부
        use_mmap = user_opts.get("use_mmap", settings.get_bool("LLAMA_MMAP", True))

        # 3. 컨텍스트 크기 및 스레드 수
        ctx_size = user_opts.get("ctx_size", settings.get_int("LLAMA_CTX_SIZE", 4096))
        threads = user_opts.get("threads", max(1, (os.cpu_count() or 4) - 1))

        # 4. CLI 인자 리스트 구성
        cli_args = [
            "-ngl", str(ngl),
            "-c", str(ctx_size),
            "-t", str(threads),
        ]
        if use_mmap:
            cli_args.append("--mmap")
        else:
            cli_args.append("--no-mmap")

        return EngineMemoryOptions(
            engine_type=self.engine_type,
            ngl=ngl,
            use_mmap=use_mmap,
            ctx_size=ctx_size,
            threads=threads,
            cli_args=cli_args,
            payload_options={},
            extra_config={"ngl": ngl, "use_mmap": use_mmap, "ctx_size": ctx_size, "threads": threads},
        )


class VllmMemoryStrategy(BaseEngineMemoryStrategy):
    """vLLM 서빙 엔진 메모리 최적화 전략."""

    def __init__(self) -> None:
        super().__init__("vllm")

    def _build_engine_parameters(
        self,
        budget: MemoryBudget,
        user_opts: Dict[str, Any],
    ) -> EngineMemoryOptions:
        """vLLM용 GPU 메모리 활용률 및 CPU 오프로드 인자 빌드."""
        # 4GB 이하 저사양 GPU에서는 0.70 및 컨텍스트 2048, 고사양에서는 0.90 및 4096
        if budget.total_vram_bytes <= 4 * 1024 * 1024 * 1024:
            gpu_util = user_opts.get("gpu_memory_utilization", 0.70)
            max_model_len = user_opts.get("max_model_len", 2048)
            cpu_offload_gb = max(0.0, (budget.model_size_bytes - budget.usable_vram_bytes) / (1024**3))
        else:
            gpu_util = user_opts.get("gpu_memory_utilization", 0.90)
            max_model_len = user_opts.get("max_model_len", 4096)
            cpu_offload_gb = 0.0

        cli_args = [
            f"--gpu-memory-utilization={gpu_util:.2f}",
            f"--max-model-len={max_model_len}",
        ]
        if cpu_offload_gb > 0:
            cli_args.append(f"--cpu-offload-gb={cpu_offload_gb:.1f}")

        return EngineMemoryOptions(
            engine_type=self.engine_type,
            ctx_size=max_model_len,
            cli_args=cli_args,
            extra_config={
                "gpu_memory_utilization": gpu_util,
                "max_model_len": max_model_len,
                "cpu_offload_gb": cpu_offload_gb,
            },
        )


class OllamaMemoryStrategy(BaseEngineMemoryStrategy):
    """Ollama 서빙 엔진 메모리 최적화 전략."""

    def __init__(self) -> None:
        super().__init__("ollama")

    def _build_engine_parameters(
        self,
        budget: MemoryBudget,
        user_opts: Dict[str, Any],
    ) -> EngineMemoryOptions:
        """Ollama /api/chat payload options 딕셔너리 빌드."""
        use_mmap = user_opts.get("use_mmap", settings.get_bool("LLAMA_MMAP", True))
        ctx_size = user_opts.get("num_ctx", 4096 if settings.profile == "mid" else 2048)

        requested_num_gpu = user_opts.get("num_gpu")
        if not budget.is_gpu_available or requested_num_gpu == 0:
            num_gpu = 0
        elif budget.total_vram_bytes <= 3 * 1024 * 1024 * 1024:
            # 2GB 환경: 4~8 레이어로 안전 클램핑
            num_gpu = max(4, min(8, 6 if requested_num_gpu is None else requested_num_gpu))
        else:
            num_gpu = 999 if requested_num_gpu is None else requested_num_gpu

        payload_options = {
            "num_gpu": num_gpu,
            "num_ctx": ctx_size,
            "use_mmap": use_mmap,
        }

        return EngineMemoryOptions(
            engine_type=self.engine_type,
            ngl=num_gpu,
            use_mmap=use_mmap,
            ctx_size=ctx_size,
            payload_options=payload_options,
        )


class ExLlamaV3MemoryStrategy(BaseEngineMemoryStrategy):
    """ExLlamaV3 서빙 엔진 메모리 최적화 전략."""

    def __init__(self) -> None:
        super().__init__("exllamav3")

    def _build_engine_parameters(
        self,
        budget: MemoryBudget,
        user_opts: Dict[str, Any],
    ) -> EngineMemoryOptions:
        """ExLlamaV3 gpu_split 및 max_seq_len 인자 빌드."""
        max_seq_len = user_opts.get("max_seq_len", 4096 if settings.profile == "mid" else 2048)
        if budget.total_vram_bytes <= 3 * 1024 * 1024 * 1024:
            gpu_split = user_opts.get("gpu_split", "1.5,0")
        else:
            gpu_split = user_opts.get("gpu_split", None)

        cli_args = [f"--max_seq_len={max_seq_len}"]
        if gpu_split:
            cli_args.append(f"--gpu_split={gpu_split}")

        return EngineMemoryOptions(
            engine_type=self.engine_type,
            ctx_size=max_seq_len,
            cli_args=cli_args,
            extra_config={"gpu_split": gpu_split, "max_seq_len": max_seq_len},
        )


class AirLlmMemoryStrategy(BaseEngineMemoryStrategy):
    """AirLLM 인메모리 스트리밍 서빙 엔진 메모리 최적화 전략 (R3 연계)."""

    def __init__(self) -> None:
        super().__init__("airllm")

    def _build_engine_parameters(
        self,
        budget: MemoryBudget,
        user_opts: Dict[str, Any],
    ) -> EngineMemoryOptions:
        """AirLLM 인메모리 레이어 스트리밍 메모리 제한 및 양자화 설정 빌드."""
        compression = user_opts.get(
            "compression",
            "4bit" if budget.total_vram_bytes <= 4 * 1024**3 else None,
        )
        usable_gib = max(1.0, budget.usable_vram_bytes / (1024**3))
        max_memory = {0: f"{usable_gib:.1f}GiB", "cpu": "12GiB"}

        return EngineMemoryOptions(
            engine_type=self.engine_type,
            extra_config={
                "compression": compression,
                "layer_streaming": True,
                "max_memory": max_memory,
            },
        )


# ---------------------------------------------------------------------------
# 전략 레지스트리 및 팩토리 함수 (Factory Pattern)
# ---------------------------------------------------------------------------
_strategy_registry: Dict[str, BaseEngineMemoryStrategy] = {}


def get_memory_strategy(engine_type: Optional[str] = None) -> BaseEngineMemoryStrategy:
    """서빙 엔진별 메모리 최적화 전략 인스턴스를 반환하는 팩토리 함수.

    Args:
        engine_type: 'llama.cpp', 'vllm', 'ollama', 'exllamav3', 'airllm' 등

    Returns:
        BaseEngineMemoryStrategy 구현체 인스턴스
    """
    global _strategy_registry
    from serving_engines.factory import canonical_engine
    target = canonical_engine(engine_type)

    if target not in _strategy_registry:
        if target == "llama.cpp":
            _strategy_registry[target] = LlamaCppMemoryStrategy()
        elif target == "vllm":
            _strategy_registry[target] = VllmMemoryStrategy()
        elif target == "ollama":
            _strategy_registry[target] = OllamaMemoryStrategy()
        elif target == "exllamav3":
            _strategy_registry[target] = ExLlamaV3MemoryStrategy()
        elif target == "airllm":
            _strategy_registry[target] = AirLlmMemoryStrategy()
        else:
            _strategy_registry[target] = LlamaCppMemoryStrategy()

    return _strategy_registry[target]


def reset_memory_strategies() -> None:
    """테스트 격리용 전략 레지스트리 캐시 초기화."""
    global _strategy_registry
    _strategy_registry.clear()


# ---------------------------------------------------------------------------
# 메모리 최적화 파사드 (Facade Pattern)
# ---------------------------------------------------------------------------
class MemoryOptimizationFacade:
    """다중 서빙 엔진 메모리 최적화 단일 진입점 파사드."""

    def __init__(
        self,
        accelerator_manager: Optional[BaseAcceleratorManager] = None,
        safety_margin_mb: Optional[int] = None,
    ) -> None:
        """파사드 초기화."""
        self._acc_manager = accelerator_manager
        margin_mb = safety_margin_mb if safety_margin_mb is not None else settings.get_int("LLAMA_SAFETY_MARGIN_MB", 500)
        self.safety_margin_bytes = margin_mb * 1024 * 1024

    @property
    def acc_manager(self) -> BaseAcceleratorManager:
        """가속기 매니저 (주입된 인스턴스 또는 팩토리 조회)."""
        if self._acc_manager is not None:
            return self._acc_manager
        return gpu_manager.get_accelerator_manager()

    def inspect_budget(
        self,
        model_size_bytes: int = 0,
        total_layers: int = 36,
    ) -> MemoryBudget:
        """현재 하드웨어 및 모델 크기를 기반으로 표준 메모리 예산(MemoryBudget)을 산출합니다."""
        acc_type = self.acc_manager.accelerator_type
        total_vram = self.acc_manager.get_vram_bytes()
        is_gpu = (acc_type in ("cuda", "gpu")) and (total_vram > 0)
        usable_vram = max(0, total_vram - self.safety_margin_bytes) if is_gpu else 0

        return MemoryBudget(
            total_vram_bytes=total_vram,
            usable_vram_bytes=usable_vram,
            safety_margin_bytes=self.safety_margin_bytes,
            is_gpu_available=is_gpu,
            accelerator_type=acc_type,
            model_size_bytes=model_size_bytes,
            total_layers=total_layers,
        )

    def get_engine_options(
        self,
        engine_type: str,
        model_path_or_size: Optional[Union[Path, int]] = None,
        total_layers: int = 36,
        user_options: Optional[Dict[str, Any]] = None,
    ) -> EngineMemoryOptions:
        """[단일 진입점] 지정된 서빙 엔진에 최적화된 메모리 실행 옵션을 산출합니다."""
        model_size = 0
        if isinstance(model_path_or_size, Path) and model_path_or_size.is_file():
            model_size = model_path_or_size.stat().st_size
        elif isinstance(model_path_or_size, int):
            model_size = model_path_or_size

        budget = self.inspect_budget(model_size_bytes=model_size, total_layers=total_layers)
        strategy = get_memory_strategy(engine_type)
        return strategy.build_options(budget, user_options)

    def calculate_partial_ngl(
        self,
        model_path_or_size: Optional[Union[Path, int]] = None,
        total_layers: int = 36,
    ) -> int:
        """[편의 메서드] llama.cpp용 안전 GPU 부분 오프로딩 레이어 수(ngl)를 반환합니다."""
        opts = self.get_engine_options("llama.cpp", model_path_or_size, total_layers)
        return opts.ngl

    def should_use_mmap(self) -> bool:
        """[편의 메서드] 메모리 매핑(mmap) 활성화 여부를 반환합니다."""
        return settings.get_bool("LLAMA_MMAP", True)


_facade_instance: Optional[MemoryOptimizationFacade] = None


def get_memory_facade() -> MemoryOptimizationFacade:
    """메모리 최적화 파사드 싱글톤 인스턴스 반환 함수."""
    global _facade_instance
    if _facade_instance is None:
        _facade_instance = MemoryOptimizationFacade()
    return _facade_instance


def reset_memory_facade() -> None:
    """테스트 격리용 메모리 최적화 파사드 및 전략 캐시 초기화."""
    global _facade_instance
    _facade_instance = None
    reset_memory_strategies()


__all__ = [
    "MemoryBudget",
    "EngineMemoryOptions",
    "BaseEngineMemoryStrategy",
    "LlamaCppMemoryStrategy",
    "VllmMemoryStrategy",
    "OllamaMemoryStrategy",
    "ExLlamaV3MemoryStrategy",
    "AirLlmMemoryStrategy",
    "get_memory_strategy",
    "reset_memory_strategies",
    "MemoryOptimizationFacade",
    "get_memory_facade",
    "reset_memory_facade",
]
