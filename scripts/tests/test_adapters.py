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
    OllamaServingAdapter,
    OpenAICompatibleServingAdapter,
    get_serving_adapter,
    reset_serving_adapters,
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
from settings import ENGINE_PRESETS, PRESETS, PROFILE_PRESETS, SettingsFacade, settings


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

    def test_checkpoint_resume(self):
        """임베딩 체크포인트 복원 및 주기적 저장 검증."""
        class MockTrackingAdapter(BaseEmbeddingAdapter):
            def __init__(self):
                super().__init__("mock", "cpu")
                self.calls = []

            def _embed_single_query(self, query: str) -> np.ndarray:
                return np.array([1.0, 0.0], dtype=np.float32)

            def _embed_batch_documents(self, documents):
                self.calls.append(documents)
                return np.array([[float(len(d)), 0.0] for d in documents], dtype=np.float32)

        adapter = MockTrackingAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "checkpoint.npy"

            # 1) 기존 체크포인트 파일 생성 (앞의 2개 문서 이미 계산된 상태)
            initial_data = np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
            np.save(cp_path, initial_data)

            # 2) 4개 문서 인코딩 요청 -> 앞 2개는 복원하고 뒤 2개만 추가 인코딩되어야 함
            texts = ["문서1", "문서2", "문서3", "문서4"]
            progress_calls = []
            res = adapter.embed_documents(
                texts,
                batch_size=2,
                checkpoint_path=cp_path,
                on_progress=lambda d, t, msg: progress_calls.append(d),
            )

            # 결과 총 4개 확인
            self.assertEqual(len(res), 4)
            # 앞 2개는 체크포인트 값
            self.assertEqual(list(res[0]), [10.0, 0.0])
            self.assertEqual(list(res[1]), [20.0, 0.0])
            # 실제 배치 인코딩은 뒤 2개만 1번 호출되었는지 검증
            self.assertEqual(len(adapter.calls), 1)
            self.assertEqual(adapter.calls[0], ["문서3", "문서4"])


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

    def setUp(self):
        reset_serving_adapters()

    def tearDown(self):
        reset_serving_adapters()

    def test_factory_creation(self):
        """엔진 타입별 생성: Ollama만 별도 클래스, 나머지는 OpenAI 호환 + engine_type."""
        cases = [
            ("llama_cpp", "llama.cpp", OpenAICompatibleServingAdapter),
            ("ollama", "ollama", OllamaServingAdapter),
            ("vllm", "vllm", OpenAICompatibleServingAdapter),
            ("sglang", "sglang", OpenAICompatibleServingAdapter),
            ("exllamav3", "exllamav3", OpenAICompatibleServingAdapter),
            ("mlc_llm", "mlc_llm", OpenAICompatibleServingAdapter),
            ("llamafile", "llamafile", OpenAICompatibleServingAdapter),
            ("openai_compatible", "openai", OpenAICompatibleServingAdapter),
        ]
        for name, expected_type, cls in cases:
            with self.subTest(engine=name):
                adapter = get_serving_adapter(name)
                self.assertIsInstance(adapter, cls)
                self.assertEqual(adapter.engine_type, expected_type)

    def test_openai_url_no_double_v1(self):
        """base_url이 .../v1 로 끝나도 chat/models 경로가 이중 /v1 이 되지 않는다."""
        adapter = OpenAICompatibleServingAdapter("http://localhost:8080/v1")
        self.assertEqual(adapter._chat_url(), "http://localhost:8080/v1/chat/completions")
        self.assertEqual(adapter._models_url(), "http://localhost:8080/v1/models")
        self.assertEqual(adapter._host_root(), "http://localhost:8080")

    def test_factory_uses_settings_serving_engine(self):
        """인자 없을 때 settings.SERVING_ENGINE 을 사용한다."""
        from settings import SERVING_ENGINE
        from serving_engines import canonical_engine
        self.assertEqual(canonical_engine(None), canonical_engine(SERVING_ENGINE))

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

    def test_clean_output_preserves_content_on_unclosed_think(self):
        """닫히지 않은 <think> 태그 시 전체 삭제(0자 버그) 방지 및 내용 보존 검증."""
        from serving_engines.base import BaseServingAdapter

        class DummyAdapter(BaseServingAdapter):
            def _call_engine(self, messages, options):
                return "<think>사고 과정 진행 중단..."
            def health_check(self):
                return True

        adapter = DummyAdapter("http://localhost:8080")
        res = adapter.generate("질문", think_mode="instant")
        self.assertEqual(res, "사고 과정 진행 중단...")

    def test_openai_adapter_reasoning_content_handling(self):
        """DeepSeek reasoning 포맷(reasoning_content 필드) 정상 처리 및 0자 방지 검증."""
        import json
        from unittest.mock import MagicMock
        adapter = OpenAICompatibleServingAdapter("http://localhost:8080/v1")
        adapter._session = MagicMock()

        # 케이스 1: content는 비어있고 reasoning_content만 있는 경우
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "추론 결과입니다."
                }
            }]
        }
        adapter._session.post.return_value = mock_resp

        res = adapter.generate("질문", think_mode="instant")
        self.assertEqual(res, "추론 결과입니다.")

        # 케이스 2: reasoning_content와 content가 둘 다 있는 경우
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "최종 답변입니다.",
                    "reasoning_content": "깊은 생각..."
                }
            }]
        }
        res2 = adapter.generate("질문", think_mode="instant")
        self.assertEqual(res2, "최종 답변입니다.")


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
        mgr = get_model_manager()
        self.assertIsInstance(mgr, BaseModelManager)

    def test_engine_manager_factory(self):
        """Llama 엔진 매니저 생성 검증."""
        mgr = get_engine_manager()
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
    """설정 파사드 — PROFILE 공통 + 엔진별 프리셋."""

    def test_profile_does_not_lock_engine(self):
        """PROFILE 프리셋에 SERVING_ENGINE 이 없다."""
        for name, pack in PROFILE_PRESETS.items():
            self.assertNotIn("SERVING_ENGINE", pack, msg=name)
        self.assertEqual(PRESETS, PROFILE_PRESETS)
        self.assertIn("llama.cpp", ENGINE_PRESETS)
        self.assertIn("vllm", ENGINE_PRESETS)

        low = PROFILE_PRESETS["low"]
        self.assertEqual(low["ACCELERATOR"], "auto")
        self.assertEqual(low["DOC_PARSER"], "pypdf")
        self.assertNotIn("LLAMA_CTX_SIZE", low)

        high = PROFILE_PRESETS["high"]
        self.assertEqual(high["ACCELERATOR"], "auto")
        self.assertEqual(ENGINE_PRESETS["llama.cpp"]["high"]["LLAMA_CTX_SIZE"], 8192)

    def test_engine_preset_follows_serving_engine(self):
        """같은 PROFILE 이라도 SERVING_ENGINE 에 따라 엔진 키가 갈린다."""
        base = {k: v for k, v in os.environ.items()}
        base.update({"PROFILE": "high", "SERVING_ENGINE": "vllm"})
        for k in ("VLLM_MODEL", "LLAMA_GGUF_FILE", "DOC_PARSER", "OLLAMA_MODEL", "LLAMA_CTX_SIZE"):
            base.pop(k, None)
        with patch.dict(os.environ, base, clear=True):
            facade = SettingsFacade()
            self.assertEqual(facade.serving_engine, "vllm")
            self.assertEqual(facade.get("VLLM_MODEL"), "Qwen/Qwen2.5-14B-Instruct")
            # llama.cpp 전용 키는 vllm 엔진 프리셋에 없음 → default
            self.assertIsNone(facade.get("LLAMA_GGUF_FILE"))
            self.assertEqual(facade.get("DOC_PARSER"), "opendataloader")

        base["SERVING_ENGINE"] = "llama.cpp"
        with patch.dict(os.environ, base, clear=True):
            facade = SettingsFacade()
            self.assertEqual(facade.get("LLAMA_CTX_SIZE"), 8192)
            self.assertIn("14B", facade.get("LLAMA_GGUF_FILE"))

    def test_facade_environment_override(self):
        """사용자 env 가 엔진·PROFILE 프리셋보다 우선."""
        test_env = dict(os.environ)
        test_env.update({
            "PROFILE": "low",
            "SERVING_ENGINE": "ollama",
            "OLLAMA_MODEL": "custom:tag",
            "LLAMA_CTX_SIZE": "8192",
        })
        test_env.pop("DOC_PARSER", None)
        with patch.dict(os.environ, test_env, clear=True):
            facade = SettingsFacade()
            self.assertEqual(facade.profile, "low")
            self.assertEqual(facade.get("SERVING_ENGINE"), "ollama")
            self.assertEqual(facade.get("OLLAMA_MODEL"), "custom:tag")
            # CTX 는 llama 전용 — ollama 엔진 프리셋에 없고 env 로 주면 env 우선
            self.assertEqual(facade.get_int("LLAMA_CTX_SIZE", 2048), 8192)
            self.assertEqual(facade.get("DOC_PARSER"), "pypdf")



class TestAcceleratorAdapter(unittest.TestCase):
    """가속기 매니저 전략 및 팩토리 테스트."""

    def test_factory_types(self):
        """가속기 타입별 올바른 매니저 클래스 반환 검증."""
        from services import (
            BaseAcceleratorManager,
            CudaAcceleratorManager,
            VulkanAcceleratorManager,
            CpuAcceleratorManager,
            get_accelerator_manager,
        )
        cuda = get_accelerator_manager("cuda")
        self.assertIsInstance(cuda, CudaAcceleratorManager)
        self.assertEqual(cuda.accelerator_type, "cuda")
        self.assertEqual(cuda.torch_device, "cuda")

        vulkan = get_accelerator_manager("vulkan")
        self.assertIsInstance(vulkan, VulkanAcceleratorManager)
        self.assertEqual(vulkan.accelerator_type, "vulkan")
        self.assertEqual(vulkan.torch_device, "cpu")

        cpu = get_accelerator_manager("cpu")
        self.assertIsInstance(cpu, CpuAcceleratorManager)
        self.assertEqual(cpu.accelerator_type, "cpu")
        self.assertEqual(cpu.torch_device, "cpu")

    def test_auto_resolution(self):
        """auto 지정 시 가용 장치에 맞는 전략 인스턴스 반환."""
        from services import get_accelerator_manager, BaseAcceleratorManager
        mgr = get_accelerator_manager("auto")
        self.assertIsInstance(mgr, BaseAcceleratorManager)
        self.assertIn(mgr.accelerator_type, ("cuda", "vulkan", "cpu"))


if __name__ == "__main__":
    unittest.main()

