"""
리랭커 백엔드(origin, laya, clm) 벤치마크 및 비교 측정 스크립트.

동일한 질문과 8개의 후보 청크를 대상으로:
  1. 백엔드 가용성(PASS / BLOCKED / UNAVAILABLE) 진단
  2. 리랭킹 소요 시간 (전처리+추론+후처리)
  3. Top-4 선별 순위 및 점수 비교
  4. 메모리 사용량 추적
을 수행하고 요약 테이블을 출력합니다.
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트 및 app 경로 추가
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
APP_DIR = BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from rerankers import RerankerUnavailableError, get_reranker_adapter

# 한국어 도메인 벤치마크 샘플 질문 및 8개 후보 청크 (RRF 검색 결과 모사)
SAMPLE_QUERY = "차량 스마트키 배터리가 방전되었을 때 시동을 거는 방법과 비상 키 위치는 어디인가요?"

SAMPLE_CANDIDATES: List[Dict[str, Any]] = [
    {
        "id": 101,
        "file_name": "genesis_g80_manual.pdf",
        "page": 45,
        "bbox": "[100, 200, 400, 300]",
        "text": "스마트키 배터리가 방전된 경우, 스마트키 본체로 시동 버튼을 직접 접촉하여 누르면 시동이 걸립니다. (림폼 방식)",
        "score": 0.88,
    },
    {
        "id": 102,
        "file_name": "genesis_g80_manual.pdf",
        "page": 46,
        "bbox": "[100, 350, 400, 450]",
        "text": "비상 키는 스마트키 뒷면의 릴리스 버튼을 누른 채 금속제 기계식 키를 뽑아내어 도어 잠금을 해제할 수 있습니다.",
        "score": 0.84,
    },
    {
        "id": 103,
        "file_name": "genesis_g80_manual.pdf",
        "page": 78,
        "bbox": "[150, 210, 450, 320]",
        "text": "스마트키의 CR2032 규격 리튬 동전형 배터리는 일자 드라이버로 케이스를 열어 교체 가능합니다.",
        "score": 0.79,
    },
    {
        "id": 104,
        "file_name": "genesis_g80_manual.pdf",
        "page": 110,
        "bbox": "[200, 100, 500, 200]",
        "text": "엔진 오일 교환 주기는 통상 조건 15,000km 또는 12개월마다 점검 및 교환을 권장합니다.",
        "score": 0.72,
    },
    {
        "id": 105,
        "file_name": "genesis_g80_manual.pdf",
        "page": 125,
        "bbox": "[100, 500, 400, 600]",
        "text": "타이어 공기압 경보 시스템(TPMS)은 주행 중 타이어 압력이 기준치 이하로 떨어지면 계기판에 경고등을 점등합니다.",
        "score": 0.68,
    },
    {
        "id": 106,
        "file_name": "genesis_g80_manual.pdf",
        "page": 47,
        "bbox": "[100, 500, 400, 580]",
        "text": "시동 버튼 접촉 시 스마트키 내부의 트랜스폰더 안테나와 시동 버튼 안테나가 상호 인증을 수행합니다.",
        "score": 0.65,
    },
    {
        "id": 107,
        "file_name": "genesis_g80_manual.pdf",
        "page": 210,
        "bbox": "[120, 300, 420, 400]",
        "text": "와이퍼 블레이드 교체 시 서비스 모드로 와이퍼 암을 세운 뒤 고정 클립을 눌러 분리하십시오.",
        "score": 0.60,
    },
    {
        "id": 108,
        "file_name": "genesis_g80_manual.pdf",
        "page": 15,
        "bbox": "[100, 100, 400, 180]",
        "text": "차량 출고 시 기본 제공되는 스마트키는 총 2개이며, 스마트 카드키 1개가 추가 제공됩니다.",
        "score": 0.58,
    },
]


def benchmark_strategy(strategy: str, top_k: int = 4) -> Dict[str, Any]:
    """단일 전략에 대한 가용성 및 성능 측정."""
    tracemalloc.start()
    start_time = time.perf_counter()

    result = {
        "strategy": strategy,
        "status": "UNKNOWN",
        "latency_sec": 0.0,
        "peak_mem_mb": 0.0,
        "top_ids": [],
        "scores": [],
        "error_msg": "",
    }

    try:
        reranker = get_reranker_adapter(strategy)
        # 워밍업 겸 1회 실행
        hits = reranker.rerank(SAMPLE_QUERY, SAMPLE_CANDIDATES, top_k=top_k)
        elapsed = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        result["status"] = "PASS"
        result["latency_sec"] = elapsed
        result["peak_mem_mb"] = peak_mem / (1024 * 1024)
        result["top_ids"] = [h["id"] for h in hits]
        result["scores"] = [round(h.get("rerank_score", 0.0), 4) for h in hits]

    except RerankerUnavailableError as e:
        elapsed = time.perf_counter() - start_time
        result["status"] = "UNAVAILABLE"
        result["latency_sec"] = elapsed
        result["error_msg"] = str(e).split("\n")[0]
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        result["status"] = "BLOCKED"
        result["latency_sec"] = elapsed
        result["error_msg"] = f"{type(e).__name__}: {e}"
    finally:
        tracemalloc.stop()
        reset_reranker()

    return result


def main():
    print("=" * 70)
    print("   [Local-RAG] 리랭커 백엔드 3종(origin, laya, clm) 벤치마크 측정")
    print("=" * 70)
    print(f" 질의: {SAMPLE_QUERY}")
    print(f" 후보 청크: {len(SAMPLE_CANDIDATES)}개  /  최종 선별 목표(Top-K): 4개\n")

    strategies = ["origin", "laya", "clm"]
    results = []

    for strat in strategies:
        print(f"-> 전략 측정 진행 중: [{strat}] ...", flush=True)
        res = benchmark_strategy(strat, top_k=4)
        results.append(res)
        print(f"   결과: {res['status']} ({res['latency_sec']:.2f}s)\n")

    print("=" * 70)
    print(f"{'전략(Strategy)':<10} | {'상태':<12} | {'지연시간':<10} | {'Peak RAM':<10} | {'선별 청크 IDs (순위별)'}")
    print("-" * 70)
    for r in results:
        top_str = str(r["top_ids"]) if r["top_ids"] else (r["error_msg"][:30] + "...")
        print(f"{r['strategy']:<14} | {r['status']:<14} | {r['latency_sec']:<8.3f}s | {r['peak_mem_mb']:<7.2f}MB | {top_str}")
    print("=" * 70)


if __name__ == "__main__":
    main()

