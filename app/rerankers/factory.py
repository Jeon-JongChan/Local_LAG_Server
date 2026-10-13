"""리랭커 어댑터 팩토리 모듈.

전략('origin', 'laya', 'clm')에 따라 적합한 BaseRerankerAdapter 서브클래스 인스턴스를 생성하고 관리합니다.
"""

from __future__ import annotations

from typing import Optional
from .base import BaseRerankerAdapter, RerankerError

try:
    from settings import RERANK_STRATEGY
except ModuleNotFoundError:
    from app.settings import RERANK_STRATEGY

_active_adapter: Optional[BaseRerankerAdapter] = None
_active_strategy: Optional[str] = None


def get_reranker_adapter(strategy: Optional[str] = None) -> BaseRerankerAdapter:
    """
    리랭커 어댑터 팩토리 함수.

    Args:
        strategy: 'origin' | 'laya' | 'clm' (기본: settings.RERANK_STRATEGY)

    Returns:
        BaseRerankerAdapter 서브클래스 인스턴스
    """
    global _active_adapter, _active_strategy

    target_strategy = (strategy or RERANK_STRATEGY).strip().lower()
    if target_strategy not in ("origin", "laya", "clm"):
        raise RerankerError(
            f"지원하지 않는 리랭킹 전략입니다: '{target_strategy}'. "
            f"('origin', 'laya', 'clm' 중 하나를 선택하세요)"
        )

    if _active_adapter is not None and _active_strategy == target_strategy:
        return _active_adapter

    # 기존 리랭커 리소스 정리
    if _active_adapter is not None:
        try:
            _active_adapter.close()
        except Exception:
            pass
        _active_adapter = None
        _active_strategy = None

    print(f"[reranker_factory] 리랭커 어댑터 초기화: {target_strategy}", flush=True)

    if target_strategy == "origin":
        from .origin import OriginRerankerAdapter
        _active_adapter = OriginRerankerAdapter()
    elif target_strategy == "laya":
        from .laya import LayaRerankerAdapter
        _active_adapter = LayaRerankerAdapter()
    elif target_strategy == "clm":
        from .clm import CLMRerankerAdapter
        _active_adapter = CLMRerankerAdapter()

    _active_strategy = target_strategy
    return _active_adapter


def reset_reranker_adapter() -> None:
    """테스트 및 종료용 인스턴스 리셋."""
    global _active_adapter, _active_strategy
    if _active_adapter is not None:
        try:
            _active_adapter.close()
        except Exception:
            pass
    _active_adapter = None
    _active_strategy = None

