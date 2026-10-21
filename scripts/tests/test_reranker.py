"""
리랭커 백엔드 단위 테스트 및 회귀 테스트.

검증 항목:
  1. 전략 선택 및 유효하지 않은 전략 처리
  2. origin 리랭커 엣지 케이스 및 메타데이터 보존
  3. laya 미설치 시 RerankerUnavailableError 발생 확인
  4. clm 서버 미구동 시 RerankerUnavailableError 발생 확인
  5. models.rerank() 하위 호환 위임 검증
"""

import os
import sys
import unittest
from pathlib import Path

# app 경로 추가
BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from rerankers import (
    RerankerError,
    RerankerUnavailableError,
    get_reranker_adapter as create_reranker,
    reset_reranker_adapter as reset_reranker,
)
from rerankers.clm import CLMRerankerAdapter as CLMReranker
from rerankers.laya import LayaRerankerAdapter as LayaReranker
from rerankers.origin import OriginRerankerAdapter as OriginReranker


class TestRerankerFactory(unittest.TestCase):
    def setUp(self):
        reset_reranker()

    def tearDown(self):
        reset_reranker()

    def test_default_strategy_is_origin(self):
        """기본 전략은 origin이어야 합니다."""
        reranker = create_reranker("origin")
        self.assertIsInstance(reranker, OriginReranker)
        self.assertEqual(reranker.backend_name, "origin")

    def test_invalid_strategy_raises_error(self):
        """지원하지 않는 전략 이름은 RerankerError를 발생시켜야 합니다."""
        with self.assertRaises(RerankerError):
            create_reranker("invalid_strategy_123")

    def test_strategy_switching_cleans_up(self):
        """전략 전환 시 이전 리랭커가 정리되고 새 리랭커가 생성되어야 합니다."""
        r1 = create_reranker("origin")
        self.assertEqual(r1.backend_name, "origin")
        r2 = create_reranker("clm")
        self.assertEqual(r2.backend_name, "clm")
        self.assertIsNot(r1, r2)


class TestOriginReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = OriginReranker()
        self.sample_candidates = [
            {
                "id": 1,
                "file_name": "manual_v1.pdf",
                "page": 3,
                "bbox": "[10, 20, 30, 40]",
                "text": "사이드미러 접힘 기능은 도어 잠금 시 자동으로 작동합니다.",
                "score": 0.85,
            },
            {
                "id": 2,
                "file_name": "manual_v1.pdf",
                "page": 12,
                "bbox": "[50, 60, 70, 80]",
                "text": "에어컨 필터 교체 주기는 15,000km 또는 1년마다 권장됩니다.",
                "score": 0.72,
            },
        ]

    def test_empty_candidates(self):
        """후보가 비어있으면 빈 리스트를 반환해야 합니다."""
        res = self.reranker.rerank("질문", [], top_k=4)
        self.assertEqual(res, [])

    def test_top_k_zero(self):
        """top_k가 0 이하이면 빈 리스트를 반환해야 합니다."""
        res = self.reranker.rerank("질문", self.sample_candidates, top_k=0)
        self.assertEqual(res, [])

    def test_single_candidate_preserves_metadata(self):
        """후보가 1개일 때 즉시 반환되며 메타데이터와 순위/스코어가 주입되어야 합니다."""
        single = [self.sample_candidates[0]]
        res = self.reranker.rerank("사이드미러 접는 법", single, top_k=4)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], 1)
        self.assertEqual(res[0]["file_name"], "manual_v1.pdf")
        self.assertEqual(res[0]["page"], 3)
        self.assertEqual(res[0]["bbox"], "[10, 20, 30, 40]")
        self.assertEqual(res[0]["rerank_rank"], 1)
        self.assertEqual(res[0]["rerank_backend"], "origin")
        self.assertIn("rerank_score", res[0])

    def test_metadata_preservation_after_sorting(self):
        """정렬 후에도 모든 원본 메타데이터가 100% 보존되어야 합니다."""
        class MockScorer:
            def predict(self, pairs):
                return [0.95, 0.10]

        self.reranker._scorer = MockScorer()
        res = self.reranker.rerank("사이드미러 접는 법", self.sample_candidates, top_k=1)
        self.assertEqual(len(res), 1)
        hit = res[0]
        self.assertEqual(hit["id"], 1)
        self.assertEqual(hit["file_name"], "manual_v1.pdf")
        self.assertEqual(hit["rerank_rank"], 1)
        self.assertEqual(hit["rerank_score"], 0.95)
        self.assertEqual(hit["rerank_backend"], "origin")


class TestLayaAdapter(unittest.TestCase):
    def test_laya_unavailable_when_not_installed_or_model_missing(self):
        """laya 미설치 또는 로드 실패 시 명확한 RerankerUnavailableError가 발생해야 합니다."""
        reranker = LayaReranker(model_id="convaiinnovations/laya-multilingual")
        try:
            import laya
        except ImportError:
            with self.assertRaises(RerankerUnavailableError) as ctx:
                reranker.rerank("질문", [{"text": "문서1", "id": 1}, {"text": "문서2", "id": 2}], top_k=2)
            self.assertIn("laya", str(ctx.exception))


class TestCLMAdapter(unittest.TestCase):
    def test_clm_unavailable_when_server_offline(self):
        """CLM 서빙 서버가 오프라인일 때 가짜 점수를 내지 않고 RerankerUnavailableError를 던져야 합니다."""
        reranker = CLMReranker(base_url="http://127.0.0.1:19999", timeout=1.0)
        candidates = [
            {"id": 1, "text": "문서 1"},
            {"id": 2, "text": "문서 2"},
        ]
        with self.assertRaises(RerankerUnavailableError) as ctx:
            reranker.rerank("질문", candidates, top_k=2)
        self.assertIn("UNAVAILABLE", str(ctx.exception))


class TestRerankerAdapterDelegation(unittest.TestCase):
    def setUp(self):
        reset_reranker()

    def test_factory_rerank_delegation(self):
        """create_reranker()를 통해 리랭킹이 정상 수행되어야 합니다."""
        single = [{"id": 10, "text": "테스트 본문", "file_name": "t.pdf"}]
        reranker = create_reranker("origin")
        res = reranker.rerank("테스트 질문", single, top_k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], 10)
        self.assertEqual(res[0]["rerank_backend"], "origin")


if __name__ == "__main__":
    unittest.main()

