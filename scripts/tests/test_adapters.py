"""어댑터 및 팩토리 패턴 종합 단위 테스트 모듈.

각 컴포넌트(임베딩, 리랭커, 서빙엔진, 저장소, 서비스 관리)의
베이스 인터페이스 상속, 팩토리 생성, 템플릿 메서드 동작을 검증합니다.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = BASE_DIR / "app"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from embeddings import BaseEmbeddingAdapter, get_embedding_adapter
from rerankers import BaseRerankerAdapter, get_reranker_adapter
from serving_engines import (
    BaseServingAdapter,
    ExLlamaV3ServingAdapter,
    LlamaCppServingAdapter,
    LlamafileServingAdapter,
    MlcLlmServingAdapter,
    OllamaServingAdapter,
    OpenAICompatibleServingAdapter,
    SglangServingAdapter,
    VllmServingAdapter,
    get_serving_adapter,
)
from storage import (
    BaseChatStore,
    BaseChunkStore,
    BaseVectorStore,
    get_chat_store,
    get_chunk_store,
    get_vector_store,
    get_rag_storage,
)
from services import (
    BaseModelManager,
    BaseEngineManager,
    get_model_manager,
    get_engine_manager,
)
from parsers import (
    BaseDocumentParserAdapter,
    OpenDataLoaderParserAdapter,
    PyPdfParserAdapter,
    get_document_parser,
)
from settings import SettingsFacade, settings, PRESETS


class TestEmbeddingAdapter(unittest.TestCase):
    """임베딩 어댑터 및 팩토리 테스트."""

    def test_factory_creation(self):
        """팩토리를 통한 Qwen 어댑터 인스턴스 생성 검증."""
        adapter = get_embedding_adapter("test-model", device="cpu")
        self.assertIsInstance(adapter, BaseEmbeddingAdapter)
        self.assertEqual(adapter.model_name, "test-model")
        self.assertEqual(adapter.device, "cpu")

    def test_template_methods(self):
        """템플릿 메서드 embed_query, embed_documents 검증."""
        class DummyEmbeddingAdapter(BaseEmbeddingAdapter):
            def _embed_single_query(self, query: str) -> np.ndarray:
                return np.array([1.0, 0.0], dtype=np.float32)

            def _embed_batch_documents(self, documents):
                return np.array([[1.0, 0.0] for _ in documents], dtype=np.float32)

        dummy = DummyEmbeddingAdapter("dummy", "cpu")
        query_vec = dummy.embed_query("질문")
        self.assertEqual(list(query_vec), [1.0, 0.0])

        docs_vecs = dummy.embed_documents(["문서1", "문서2"])
        self.assertEqual(len(docs_vecs), 2)
        self.assertEqual(list(docs_vecs[0]), [1.0, 0.0])


class TestRerankerAdapter(unittest.TestCase):
    """리랭커 어댑터 및 팩토리 테스트."""

    def test_factory_variants(self):
        """전략별 리랭커 어댑터 생성 검증."""
        for strat in ["origin", "laya", "clm"]:
            adapter = get_reranker_adapter(strat)
            self.assertIsInstance(adapter, BaseRerankerAdapter)

    def test_empty_candidates(self):
        """후보군이 없을 때 즉시 빈 리스트 반환(템플릿 메서드 엣지케이스)."""
        adapter = get_reranker_adapter("origin")
        result = adapter.rerank("질문", [])
        self.assertEqual(result, [])

    def test_ranking_and_metadata_preservation(self):
        """점수에 따른 내림차순 정렬 및 기존 메타데이터 보존 검증."""
        class MockReranker(BaseRerankerAdapter):
            def _calculate_scores(self, query, candidates):
                # 후보별 점수 계산
                return [0.1, 0.9, 0.5]

        mock_adapter = MockReranker()
        candidates = [
            {"id": "doc1", "text": "첫번째", "original_score": 10},
            {"id": "doc2", "text": "두번째", "original_score": 20},
            {"id": "doc3", "text": "세번째", "original_score": 30},
        ]
        reranked = mock_adapter.rerank("질문", candidates, top_k=2)

        self.assertEqual(len(reranked), 2)
        # 점수 순서: doc2 (0.9) -> doc3 (0.5)
        self.assertEqual(reranked[0]["id"], "doc2")
        self.assertEqual(reranked[0]["rerank_score"], 0.9)
        self.assertEqual(reranked[0]["original_score"], 20)

        self.assertEqual(reranked[1]["id"], "doc3")
        self.assertEqual(reranked[1]["rerank_score"], 0.5)


class TestServingAdapter(unittest.TestCase):
    """서빙 엔진 어댑터 및 팩토리 테스트."""

    def test_factory_creation(self):
        """엔진 타입별 서빙 어댑터 생성 검증 (llama.cpp, ollama, vllm, sglang, exllamav3, mlc_llm, llamafile, openai)."""
        llama_adapter = get_serving_adapter("llama_cpp")
        self.assertIsInstance(llama_adapter, LlamaCppServingAdapter)

        ollama_adapter = get_serving_adapter("ollama")
        self.assertIsInstance(ollama_adapter, OllamaServingAdapter)

        vllm_adapter = get_serving_adapter("vllm")
        self.assertIsInstance(vllm_adapter, VllmServingAdapter)

        sglang_adapter = get_serving_adapter("sglang")
        self.assertIsInstance(sglang_adapter, SglangServingAdapter)

        exllama_adapter = get_serving_adapter("exllamav3")
        self.assertIsInstance(exllama_adapter, ExLlamaV3ServingAdapter)

        mlc_adapter = get_serving_adapter("mlc_llm")
        self.assertIsInstance(mlc_adapter, MlcLlmServingAdapter)

        llamafile_adapter = get_serving_adapter("llamafile")
        self.assertIsInstance(llamafile_adapter, LlamafileServingAdapter)

        openai_adapter = get_serving_adapter("openai_compatible")
        self.assertIsInstance(openai_adapter, OpenAICompatibleServingAdapter)

    def test_template_generate_thinking_toggle(self):
        """/think 키워드에 따른 사고 모드 토글 템플릿 검증."""
        class MockServingAdapter(BaseServingAdapter):
            def _call_engine(self, messages, options):
                use_think = options.get("use_think", False)
                return f"<think>thinking</think>응답" if use_think else "응답"

            def health_check(self) -> bool:
                return True

        adapter = MockServingAdapter("http://localhost:8080")

        # 1) 일반 프롬프트 (사고 모드 비활성)
        res1 = adapter.generate("안녕하세요", think_mode="instant")
        self.assertEqual(res1, "응답")

        # 2) /think 접두사 포함 (사고 모드 활성화 및 _clean_output 검증)
        res2 = adapter.generate("/think 복잡한 질문", think_mode="instant")
        self.assertEqual(res2, "응답")


class TestStorageAdapter(unittest.TestCase):
    """저장소 어댑터 및 팩토리 테스트."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_chat.db")
        self.chunk_db_path = os.path.join(self.temp_dir.name, "test_chunk.db")
        self.chat_store = None
        self.chunk_store = None

    def tearDown(self):
        if self.chat_store is not None:
            self.chat_store.close()
        if self.chunk_store is not None:
            self.chunk_store.close()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_chat_store(self):
        """SqliteChatStore 기본 CRUD 검증."""
        self.chat_store = get_chat_store(self.db_path)
        self.assertIsInstance(self.chat_store, BaseChatStore)

        # 세션 생성 및 조회
        session = self.chat_store.create_session("테스트 세션")
        session_id = session["id"]
        self.assertEqual(session["title"], "테스트 세션")

        sessions = self.chat_store.list_sessions()
        self.assertEqual(len(sessions), 1)

        # 메시지 추가 및 조회
        msg = self.chat_store.add_message(session_id, "user", "안녕하세요")
        self.assertEqual(msg["role"], "user")

        messages = self.chat_store.list_messages(session_id)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "안녕하세요")

        # 웹 서버 다중 요청 시뮬레이션 (요청 종료 시 close 후 다음 요청에서 get_chat_store 재호출)
        self.chat_store.close()
        new_store = get_chat_store(self.db_path)
        try:
            fetched = new_store.get_session(session_id)
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched["id"], session_id)
        finally:
            new_store.close()

    def test_chunk_store(self):
        """SqliteChunkStore 및 BM25 FTS5 검색 검증."""
        self.chunk_store = get_chunk_store(self.chunk_db_path)
        self.assertIsInstance(self.chunk_store, BaseChunkStore)

        self.chunk_store.insert_chunk(
            chunk_id=1,
            file_name="doc1.txt",
            source_path="/path/doc1.txt",
            page=1,
            bbox=None,
            element_type="text",
            text="파이썬 언어 모델 아키텍처",
        )
        self.chunk_store.insert_chunk(
            chunk_id=2,
            file_name="doc2.txt",
            source_path="/path/doc2.txt",
            page=1,
            bbox=None,
            element_type="text",
            text="자바스크립트 프론트엔드",
        )
        self.assertEqual(self.chunk_store.count(), 2)

        # BM25 검색
        results = self.chunk_store.search_bm25("파이썬", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)


class TestServiceManagerAdapter(unittest.TestCase):
    """서비스 매니저 어댑터 및 팩토리 테스트."""

    def test_model_manager_factory(self):
        """HuggingFace 모델 매니저 생성 검증."""
        mgr = get_model_manager("huggingface")
        self.assertIsInstance(mgr, BaseModelManager)

    def test_engine_manager_factory(self):
        """Llama 엔진 매니저 생성 검증."""
        mgr = get_engine_manager("llama_cpp")
        self.assertIsInstance(mgr, BaseEngineManager)


class TestDocumentParserAdapter(unittest.TestCase):
    """문서 파서 어댑터 및 팩토리 테스트."""

    def test_factory_creation(self):
        """파서 타입별 인스턴스 생성 검증."""
        odl = get_document_parser("opendataloader")
        self.assertIsInstance(odl, OpenDataLoaderParserAdapter)

        pypdf = get_document_parser("pypdf")
        self.assertIsInstance(pypdf, PyPdfParserAdapter)

    def test_template_batch_parse(self):
        """배치 파싱 템플릿 메서드 검증 (정상 파싱 및 에러 격리)."""
        class DummyParser(BaseDocumentParserAdapter):
            def parse_file(self, file_path):
                if "bad" in file_path.name:
                    raise ValueError("corrupted file")
                return [{"file_name": file_path.name, "text": "테스트 본문", "page": 1}]

        dummy = DummyParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "good.pdf"
            p1.write_text("dummy")
            p2 = Path(tmpdir) / "bad.pdf"
            p2.write_text("corrupted")

            progress_logs = []

            def on_progress(done, total, detail):
                progress_logs.append((done, total, detail))

            res = dummy.parse_batch([p1, p2], on_progress=on_progress)
            self.assertEqual(len(res.elements), 1)
            self.assertEqual(res.elements[0]["file_name"], "good.pdf")
            self.assertEqual(len(res.quarantined), 1)
            self.assertEqual(res.quarantined[0][0], "bad.pdf")
            self.assertTrue(len(progress_logs) > 0)


class TestSettingsFacade(unittest.TestCase):
    """설정 파사드 및 사양 프리셋 테스트."""

    def test_preset_profiles_exist(self):
        """low, mid, high 사양 프리셋 구성 검증."""
        self.assertIn("low", PRESETS)
        self.assertIn("mid", PRESETS)
        self.assertIn("high", PRESETS)

        # low 프로필: 저사양 최적화 (CPU/1.5B/2K 컨텍스트/pypdf)
        low = PRESETS["low"]
        self.assertEqual(low["ACCELERATOR"], "cpu")
        self.assertEqual(low["LLAMA_CTX_SIZE"], 2048)
        self.assertEqual(low["DOC_PARSER"], "pypdf")

        # high 프로필: 고사양 최적화 (CUDA/8K+ 컨텍스트/vLLM)
        high = PRESETS["high"]
        self.assertEqual(high["ACCELERATOR"], "cuda")
        self.assertEqual(high["SERVING_ENGINE"], "vllm")
        self.assertEqual(high["LLAMA_CTX_SIZE"], 8192)

    def test_facade_environment_override(self):
        """파사드에서 사용자 환경변수가 프리셋보다 최우선 적용되는지 검증."""
        test_env = dict(os.environ)
        test_env.update({"PROFILE": "low", "SERVING_ENGINE": "ollama", "LLAMA_CTX_SIZE": "8192"})
        test_env.pop("DOC_PARSER", None)  # 프리셋 fallback 검증을 위해 명시적 제거
        with patch.dict(os.environ, test_env, clear=True):
            facade = SettingsFacade()
            self.assertEqual(facade.profile, "low")
            # 환경변수로 명시한 ollama가 low 프리셋(llama.cpp)을 덮어써야 함
            self.assertEqual(facade.get("SERVING_ENGINE"), "ollama")
            # 환경변수로 명시한 8192가 low 프리셋(2048)을 덮어써야 함
            self.assertEqual(facade.get_int("LLAMA_CTX_SIZE", 2048), 8192)
            # 명시하지 않은 항목은 low 프리셋(pypdf) 유지
            self.assertEqual(facade.get("DOC_PARSER"), "pypdf")


if __name__ == "__main__":
    unittest.main()
