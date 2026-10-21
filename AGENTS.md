# local-rag Project Rules

## 1. labeler = 별도 저장소 (절대 금지)

`tools/labeler/` 는 **local-rag와 다른 독립 git 프로젝트**다.

### 절대 하지 말 것
- `tools/labeler/.git` 삭제·이동·이름 변경
- `git -C tools/labeler ...` 또는 labeler 안에서 `git init` / `git remote` / `git push` / `git commit`
- nested repo 경고가 떠도 **flatten 하지 말 것** (`git rm --cached` 후 파일로 재추가 금지)
- parent 저장소에 labeler 소스를 일반 파일로 흡수·커밋하지 말 것
- Recycle Bin / 디스크 전역 검색으로 labeler `.git` 복구 시도하지 말 것 (사용자가 명시할 때만)

### 해야 할 것
- labeler **코드/기능** 수정은 허용 (서버, UI, config 등)
- parent(`local-rag`) git 작업 시 labeler는 **submodule로 두거나 무시** — `.git` 은 그대로 둘 것
- nested repo hint가 보이면: **멈추고 사용자에게 확인**. 임의로 `.git` 제거 금지

---

## 2. 디자인 패턴 및 코드 구조화 규칙

local-rag 프로젝트는 외부 라이브러리, 하드웨어 런타임, AI 모델의 교체가 잦은 RAG 시스템이므로,
**파사드(Facade), 팩토리(Factory), 전략(Strategy), 어댑터(Adapter), 템플릿 메서드(Template Method)** 5대 핵심 패턴을 필수로 적용하여 코드를 최적화하고 유지보수성을 극대화합니다.

### 5대 핵심 디자인 패턴 적용 원칙

1. **Facade (파사드 패턴) — 복잡성 캡슐화**
   - 원칙: 다수의 서브시스템, 복합 데이터 스토리지, 다계층 설정 로직에 클라이언트 코드가 직접 접근하지 않고, 단일 진입점 파사드를 통해서만 상호작용합니다.
   - 적용 사례:
     - `SettingsFacade` (`app/settings.py`): 시스템 환경변수, `.env`, PROFILE(low/mid/high), 서빙 엔진 프리셋의 복잡한 우선순위 병합을 `settings.get()` 단일 인터페이스로 제공.
     - `RagStorage` (`app/storage/rag_storage.py`): SQLite 청크/FTS5 메타데이터와 FAISS 벡터 인덱스 간의 동기화 저장/검색을 단일 창구로 캡슐화.

2. **Factory (팩토리 패턴) — 생성 책임 분리 및 다형성 보장**
   - 원칙: 구체 클래스(Concrete Class)를 클라이언트가 직접 인스턴스화하지 않으며, 팩토리 함수(`get_*`)를 통해 추상 베이스 인터페이스 타입으로 반환받습니다.
   - 적용 사례:
     - `get_accelerator_manager(type)` (`app/services/gpu_manager.py`)
     - `get_serving_adapter(engine)` (`app/serving_engines/factory.py`)
     - `get_reranker_adapter(strategy)` (`app/rerankers/factory.py`)
     - `get_embedding_adapter(model, device)` (`app/embeddings/__init__.py`)
     - `get_document_parser(parser_type)` (`app/parsers/factory.py`)
   - 요구사항: 팩토리는 싱글톤 캐싱 및 테스트/설정 변경 시 인스턴스를 초기화하는 리셋(`reset_*`) 메서드를 반드시 지원해야 합니다.

3. **Strategy & Adapter (전략 및 어댑터 패턴) — 런타임 교체 및 외부 격리**
   - 원칙: 외부 라이브러리(PyTorch, HuggingFace, llama.cpp, OpenDataLoader 등)나 하드웨어 환경(CUDA, Vulkan, CPU)의 차이는 `Base*` 추상 클래스로 감싸(Adapter) 내부 도메인과 격리하고, 환경 설정에 따라 자유롭게 갈아끼울 수 있는 구조(Strategy)를 유지합니다.
   - 적용 사례:
     - 가속기 전략: `BaseAcceleratorManager` → `CudaAcceleratorManager`, `VulkanAcceleratorManager`, `CpuAcceleratorManager`
     - 서빙 엔진 어댑터: `BaseServingAdapter` → `OpenAICompatibleServingAdapter`, `OllamaServingAdapter`
     - 리랭커 어댑터: `BaseRerankerAdapter` → `OriginRerankerAdapter`, `LayaRerankerAdapter`, `ClmRerankerAdapter`
     - 문서 파서 어댑터: `BaseDocumentParser` → `OpenDataLoaderParserAdapter`, `PyPdfParserAdapter`

4. **Template Method (템플릿 메서드 패턴) — 실행 흐름 표준화**
   - 원칙: 불변의 공통 실행 흐름(유효성 검증 → 리소스 가용 확인 → 배치 분할 → 핵심 알고리즘 호출 → 후처리 및 진행률 통지)은 `Base*` 부모 클래스의 템플릿 메서드에 정의하고, 하위 클래스는 알고리즘 세부 단계(`_prepare_*`, `_embed_*`, `parse_file` 등)만 오버라이드합니다.
   - 적용 사례:
     - `BaseAcceleratorManager.ensure_ready()`: 가용성 검증 → 필요 시 CUDA 런타임 휠 설치 → 최종 확인.
     - `BaseModelManager.ensure_model()` / `ensure_file()`: 로컬 존재 검증 → 미존재 시 다운로드 → 무결성 검증.
     - `BaseEngineManager.ensure_running()`: 프로세스 활성 검증 → 미기동 시 바이너리 확보 → 프로세스 기동.
     - `BaseEmbeddingAdapter.embed_documents()`: 입력 검증 → 배치 슬라이싱 → 서브클래스 배치 인코딩 → 진행률 콜백.

### 신규 기능 작성 시 필수 체크리스트
1. **독자적인 날것(Raw) 유틸 함수 지양**: 하드웨어·엔진·모델·스토리지 관련 기능은 상응하는 `Base*` 인터페이스 정의 또는 상속.
2. **단일 진실 원천(Source of Truth) 준수**: 하드웨어 및 런타임 디바이스는 `ACCELERATOR`를 단일 기준으로 관리.
3. **오류 안전망 및 자동 치유(Auto-healing)**: 누락된 바이너리, 잘못된 버전, 손상된 바인딩의 자동 감지/복구 흐름 설계.
