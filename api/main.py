import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from pydantic import (
    BaseModel,
    Field,
)

from repository.scanner import (
    validate_repository,
)
from rag.hybrid import (
    prepare_bm25_retriever,
    prepare_dense_retriever,
)
from tools.registry import (
    AgentToolbox,
    TOOL_SCHEMAS,
)
from agent.core import (
    ToolCallingLLMClient,
)
from agent.citations import (
    CitationAwareRepoDoctor,
    CitationValidator,
    EvidenceTrackingToolbox,
)
from tools.log_analysis import (
    TracebackDiagnosisAgent,
)
from agent.extensions import (
    ConversationMemory,
    MemoryRepoDoctorAgent,
)
from rag.query_rewriter import (
    QueryRewriter,
    RewrittenHybridRetriever,
)
from rag.reranker import (
    CodeReranker,
    RerankedCodeSearch,
    environment_flag,
)


PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

RUNTIME_DIRECTORY = (
    PROJECT_DIR / "runtime_data"
)

DEFAULT_ALLOWED_ROOT = (
    r"D:\桌面"
)

DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:8501",
    "http://localhost:8501",
]

MAX_SESSION_ID_LENGTH = 128


app = FastAPI(
    title="Repo Doctor Agent API",
    description=(
        "面向本地代码仓库的多工具分析、"
        "问答和 Traceback 诊断后端"
    ),
    version="0.1.0",
)


def load_cors_origins():
    raw_origins = os.getenv(
        "REPO_DOCTOR_CORS_ORIGINS",
        "",
    ).strip()

    if not raw_origins:
        return DEFAULT_CORS_ORIGINS

    return [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=load_cors_origins(),
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "Content-Type",
        "Authorization",
    ],
)


class IndexRequest(BaseModel):
    repository_path: str = Field(
        min_length=1,
        description=(
            "需要建立索引的本地仓库路径"
        ),
    )

    rebuild_index: bool = Field(
        default=False,
        description=(
            "是否强制重新生成 Dense 和 BM25 索引"
        ),
    )

    enable_reranker: bool = Field(
        default=True,
        description=(
            "是否启用 CrossEncoder Reranker"
        ),
    )


class ChatRequest(BaseModel):
    repository_path: str = Field(
        min_length=1,
    )

    message: str = Field(
        min_length=1,
        max_length=100000,
    )

    session_id: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
        max_length=MAX_SESSION_ID_LENGTH,
    )


class AnalyzeLogRequest(BaseModel):
    repository_path: str = Field(
        min_length=1,
    )

    log_text: str = Field(
        min_length=1,
        max_length=200000,
    )

    session_id: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
        max_length=MAX_SESSION_ID_LENGTH,
    )


class SessionRequest(BaseModel):
    repository_path: str = Field(
        min_length=1,
    )

    session_id: str = Field(
        min_length=1,
        max_length=MAX_SESSION_ID_LENGTH,
    )


def load_allowed_roots():
    raw_roots = os.getenv(
        "REPO_DOCTOR_ALLOWED_ROOTS",
        DEFAULT_ALLOWED_ROOT,
    )

    roots = []

    for raw_root in raw_roots.split(
        os.pathsep
    ):
        raw_root = raw_root.strip()

        if not raw_root:
            continue

        root = Path(
            raw_root
        ).resolve()

        if root.exists() and root.is_dir():
            roots.append(root)

    if not roots:
        raise RuntimeError(
            "没有有效的允许访问目录。"
            "请设置 REPO_DOCTOR_ALLOWED_ROOTS"
        )

    return roots


def resolve_allowed_repository(
    repository_path,
):
    repository = validate_repository(
        repository_path
    )

    allowed_roots = load_allowed_roots()

    for allowed_root in allowed_roots:
        try:
            repository.relative_to(
                allowed_root
            )
            return repository
        except ValueError:
            continue

    raise ValueError(
        "仓库不在允许访问的目录中。"
        f"仓库：{repository}；"
        f"允许目录：{allowed_roots}"
    )


def validate_session_id(
    session_id,
):
    session_id = session_id.strip()

    if not session_id:
        raise ValueError(
            "session_id 不能为空"
        )

    if len(session_id) > (
        MAX_SESSION_ID_LENGTH
    ):
        raise ValueError(
            "session_id 过长"
        )

    allowed_characters = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-_"
    )

    if any(
        character not in allowed_characters
        for character in session_id
    ):
        raise ValueError(
            "session_id 只能包含字母、数字、"
            "短横线和下划线"
        )

    return session_id


def repository_identifier(
    repository_path,
):
    normalized_path = str(
        repository_path
    ).lower()

    return sha256(
        normalized_path.encode("utf-8")
    ).hexdigest()[:16]


class RerankedSearchAdapter:
    """
    将 Day 16 的返回格式适配成
    Day 9 AgentToolbox 需要的 search() 格式。
    """

    def __init__(
        self,
        search_pipeline,
    ):
        self.search_pipeline = (
            search_pipeline
        )

    def search(
        self,
        query,
        top_k=5,
        candidate_k=20,
    ):
        recall_k = max(
            top_k * 4,
            20,
        )

        retrieval_candidate_k = max(
            candidate_k,
            recall_k,
            30,
        )

        result = (
            self.search_pipeline.search(
                query=query,
                final_k=top_k,
                recall_k=recall_k,
                retrieval_candidate_k=(
                    retrieval_candidate_k
                ),
            )
        )

        if result["skipped"]:
            return []

        return result["results"]


@dataclass
class RepositoryRuntime:
    repository_path: Path
    runtime_id: str
    runtime_directory: Path
    original_toolbox: AgentToolbox
    dense_retriever: Any
    bm25_retriever: Any
    query_rewriter: QueryRewriter
    reranker: CodeReranker
    search_pipeline: RerankedCodeSearch
    reranker_enabled: bool


@dataclass
class SessionRuntime:
    repository_path: Path
    session_id: str
    tracking_toolbox: EvidenceTrackingToolbox
    memory: ConversationMemory
    memory_agent: MemoryRepoDoctorAgent
    citation_agent: CitationAwareRepoDoctor
    diagnosis_agent: TracebackDiagnosisAgent
    lock: RLock


class BackendManager:
    def __init__(self):
        self.repositories: Dict[
            str,
            RepositoryRuntime,
        ] = {}

        self.sessions: Dict[
            str,
            SessionRuntime,
        ] = {}

        self.rerankers: Dict[
            str,
            CodeReranker,
        ] = {}

        self.llm_client = None
        self.lock = RLock()

    def _repository_key(
        self,
        repository_path,
    ):
        return str(
            repository_path
        ).lower()

    def _session_key(
        self,
        repository_path,
        session_id,
    ):
        return (
            f"{self._repository_key(repository_path)}"
            f"::{session_id}"
        )

    def get_llm_client(self):
        with self.lock:
            if self.llm_client is None:
                self.llm_client = (
                    ToolCallingLLMClient
                    .from_environment()
                )

            return self.llm_client

    def get_reranker(
        self,
        enabled,
    ):
        model_name = os.getenv(
            "RERANKER_MODEL_NAME",
            "BAAI/bge-reranker-base",
        )

        cache_key = (
            f"{model_name}::{enabled}"
        )

        with self.lock:
            if cache_key not in self.rerankers:
                self.rerankers[
                    cache_key
                ] = CodeReranker(
                    model_name=model_name,
                    enabled=enabled,
                    batch_size=4,
                    max_length=512,
                    allow_fallback=True,
                )

            return self.rerankers[
                cache_key
            ]

    def index_repository(
        self,
        repository_path,
        rebuild_index=False,
        enable_reranker=True,
    ):
        repository_path = (
            resolve_allowed_repository(
                repository_path
            )
        )

        repository_key = (
            self._repository_key(
                repository_path
            )
        )

        with self.lock:
            existing_runtime = (
                self.repositories.get(
                    repository_key
                )
            )

            if (
                existing_runtime is not None
                and not rebuild_index
            ):
                if (
                    existing_runtime
                    .reranker_enabled
                    != enable_reranker
                ):
                    raise ValueError(
                        "该仓库已使用不同的 "
                        "Reranker 设置初始化。"
                        "请设置 rebuild_index=true"
                    )

                return existing_runtime

            runtime_id = (
                repository_identifier(
                    repository_path
                )
            )

            runtime_directory = (
                RUNTIME_DIRECTORY
                / runtime_id
            )

            runtime_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            dense_index_directory = (
                runtime_directory
                / "dense_index"
            )

            bm25_index_path = (
                runtime_directory
                / "bm25_index"
                / "index.json"
            )

            llm_client = (
                self.get_llm_client()
            )

            dense_retriever = (
                prepare_dense_retriever(
                    repository_path=(
                        repository_path
                    ),
                    index_directory=(
                        dense_index_directory
                    ),
                    rebuild_index=(
                        rebuild_index
                    ),
                )
            )

            bm25_retriever = (
                prepare_bm25_retriever(
                    repository_path=(
                        repository_path
                    ),
                    index_path=(
                        bm25_index_path
                    ),
                    rebuild_index=(
                        rebuild_index
                    ),
                )
            )

            query_rewriter = (
                QueryRewriter(
                    repository_path=(
                        repository_path
                    ),
                    llm_client=llm_client,
                )
            )

            reranker = self.get_reranker(
                enabled=enable_reranker
            )

            rewritten_retriever = (
                RewrittenHybridRetriever(
                    query_rewriter=(
                        query_rewriter
                    ),
                    dense_retriever=(
                        dense_retriever
                    ),
                    bm25_retriever=(
                        bm25_retriever
                    ),
                    rrf_k=60,
                    dense_weight=1.0,
                    bm25_weight=1.0,
                )
            )

            search_pipeline = (
                RerankedCodeSearch(
                    rewritten_retriever=(
                        rewritten_retriever
                    ),
                    reranker=reranker,
                )
            )

            search_adapter = (
                RerankedSearchAdapter(
                    search_pipeline
                )
            )

            original_toolbox = (
                AgentToolbox(
                    repository_path=(
                        repository_path
                    ),
                    hybrid_retriever=(
                        search_adapter
                    ),
                )
            )

            runtime = RepositoryRuntime(
                repository_path=(
                    repository_path
                ),
                runtime_id=runtime_id,
                runtime_directory=(
                    runtime_directory
                ),
                original_toolbox=(
                    original_toolbox
                ),
                dense_retriever=(
                    dense_retriever
                ),
                bm25_retriever=(
                    bm25_retriever
                ),
                query_rewriter=(
                    query_rewriter
                ),
                reranker=reranker,
                search_pipeline=(
                    search_pipeline
                ),
                reranker_enabled=(
                    enable_reranker
                ),
            )

            self.repositories[
                repository_key
            ] = runtime

            if rebuild_index:
                session_prefix = (
                    f"{repository_key}::"
                )

                session_keys = [
                    key
                    for key in self.sessions
                    if key.startswith(
                        session_prefix
                    )
                ]

                for session_key in session_keys:
                    del self.sessions[
                        session_key
                    ]

            return runtime

    def get_repository(
        self,
        repository_path,
    ):
        repository_path = (
            resolve_allowed_repository(
                repository_path
            )
        )

        repository_key = (
            self._repository_key(
                repository_path
            )
        )

        with self.lock:
            runtime = (
                self.repositories.get(
                    repository_key
                )
            )

        if runtime is None:
            raise LookupError(
                "仓库尚未初始化。"
                "请先调用 POST /api/v1/index"
            )

        return runtime

    def get_session(
        self,
        repository_path,
        session_id,
    ):
        repository_runtime = (
            self.get_repository(
                repository_path
            )
        )

        session_id = validate_session_id(
            session_id
        )

        session_key = self._session_key(
            repository_runtime.repository_path,
            session_id,
        )

        with self.lock:
            existing_session = (
                self.sessions.get(
                    session_key
                )
            )

            if existing_session is not None:
                return existing_session

            tracking_toolbox = (
                EvidenceTrackingToolbox(
                    toolbox=(
                        repository_runtime
                        .original_toolbox
                    ),
                    repository_path=(
                        repository_runtime
                        .repository_path
                    ),
                )
            )

            memory = ConversationMemory(
                max_turns=6,
                max_chars=18000,
            )

            llm_client = (
                self.get_llm_client()
            )

            memory_agent = (
                MemoryRepoDoctorAgent(
                    llm_client=llm_client,
                    toolbox=(
                        tracking_toolbox
                    ),
                    tool_schemas=(
                        TOOL_SCHEMAS
                    ),
                    memory=memory,
                    max_tool_steps=6,
                )
            )

            citation_validator = (
                CitationValidator(
                    repository_runtime
                    .repository_path
                )
            )

            citation_agent = (
                CitationAwareRepoDoctor(
                    base_agent=memory_agent,
                    llm_client=llm_client,
                    tracking_toolbox=(
                        tracking_toolbox
                    ),
                    citation_validator=(
                        citation_validator
                    ),
                )
            )

            diagnosis_agent = (
                TracebackDiagnosisAgent(
                    tracking_toolbox=(
                        tracking_toolbox
                    ),
                    llm_client=llm_client,
                    citation_validator=(
                        citation_validator
                    ),
                )
            )

            session = SessionRuntime(
                repository_path=(
                    repository_runtime
                    .repository_path
                ),
                session_id=session_id,
                tracking_toolbox=(
                    tracking_toolbox
                ),
                memory=memory,
                memory_agent=memory_agent,
                citation_agent=(
                    citation_agent
                ),
                diagnosis_agent=(
                    diagnosis_agent
                ),
                lock=RLock(),
            )

            self.sessions[
                session_key
            ] = session

            return session

    def clear_session(
        self,
        repository_path,
        session_id,
    ):
        session = self.get_session(
            repository_path,
            session_id,
        )

        with session.lock:
            session.memory.clear()
            session.tracking_toolbox.reset()

        return session

    def get_status(self):
        with self.lock:
            return {
                "repository_count": len(
                    self.repositories
                ),
                "session_count": len(
                    self.sessions
                ),
                "reranker_count": len(
                    self.rerankers
                ),
                "llm_initialized": (
                    self.llm_client
                    is not None
                ),
            }


manager = BackendManager()


def summarize_evidence(
    evidence,
):
    summaries = []

    for item in evidence:
        content = item.get(
            "content"
        )

        content_preview = None

        if content:
            content_preview = (
                content[:1000]
            )

        summaries.append(
            {
                "source_tool": item.get(
                    "source_tool"
                ),
                "file": item.get("file"),
                "start_line": item.get(
                    "start_line"
                ),
                "end_line": item.get(
                    "end_line"
                ),
                "description": item.get(
                    "description"
                ),
                "content_preview": (
                    content_preview
                ),
            }
        )

    return summaries


def sanitize_tool_trace(
    tool_trace,
):
    sanitized_trace = []

    for item in tool_trace:
        copied_item = dict(item)

        arguments = dict(
            copied_item.get(
                "arguments",
                {},
            )
        )

        if "log_text" in arguments:
            log_text = str(
                arguments["log_text"]
            )

            arguments["log_text"] = (
                f"<日志长度："
                f"{len(log_text)} 字符>"
            )

        copied_item["arguments"] = (
            arguments
        )

        sanitized_trace.append(
            copied_item
        )

    return sanitized_trace


def repository_runtime_response(
    runtime,
):
    return {
        "status": "ready",
        "repository_path": str(
            runtime.repository_path
        ),
        "runtime_id": (
            runtime.runtime_id
        ),
        "dense_chunk_count": len(
            runtime.dense_retriever.chunks
        ),
        "bm25_document_count": len(
            runtime.bm25_retriever.chunks
        ),
        "reranker_enabled": (
            runtime.reranker_enabled
        ),
        "reranker_available": (
            runtime.reranker.is_available()
        ),
        "reranker_model": (
            runtime.reranker.model_name
        ),
        "runtime_directory": str(
            runtime.runtime_directory
        ),
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "service": (
            "repo-doctor-agent"
        ),
        "manager": (
            manager.get_status()
        ),
    }


@app.post("/api/v1/index")
def index_repository(
    request: IndexRequest,
):
    try:
        runtime = manager.index_repository(
            repository_path=(
                request.repository_path
            ),
            rebuild_index=(
                request.rebuild_index
            ),
            enable_reranker=(
                request.enable_reranker
            ),
        )

        return repository_runtime_response(
            runtime
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        ) from error


@app.post("/api/v1/chat")
def chat(
    request: ChatRequest,
):
    try:
        session = manager.get_session(
            repository_path=(
                request.repository_path
            ),
            session_id=request.session_id,
        )

        with session.lock:
            result = (
                session.citation_agent.run(
                    request.message
                )
            )

            if result[
                "citation_repaired"
            ]:
                session.memory.replace_last_assistant(
                    result["answer"]
                )

            memory_status = (
                session.memory.get_status()
            )

        return {
            "session_id": (
                session.session_id
            ),
            "answer": result["answer"],
            "tool_trace": (
                sanitize_tool_trace(
                    result["tool_trace"]
                )
            ),
            "evidence": (
                summarize_evidence(
                    result["evidence"]
                )
            ),
            "citation_report": (
                result["citation_report"]
            ),
            "citation_repaired": (
                result[
                    "citation_repaired"
                ]
            ),
            "reached_step_limit": (
                result[
                    "reached_step_limit"
                ]
            ),
            "memory": memory_status,
        }

    except LookupError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        ) from error


@app.post("/api/v1/analyze-log")
def analyze_log(
    request: AnalyzeLogRequest,
):
    try:
        session = manager.get_session(
            repository_path=(
                request.repository_path
            ),
            session_id=request.session_id,
        )

        with session.lock:
            result = (
                session.diagnosis_agent
                .diagnose(
                    request.log_text
                )
            )

            session.memory.add_turn(
                user_message=(
                    "请分析下面的错误日志：\n"
                    + request.log_text[
                        :5000
                    ]
                ),
                assistant_message=(
                    result["answer"]
                ),
            )

            memory_status = (
                session.memory.get_status()
            )

        return {
            "session_id": (
                session.session_id
            ),
            "answer": result["answer"],
            "parsed_traceback": (
                result[
                    "parsed_traceback"
                ]
            ),
            "search_queries": (
                result["search_queries"]
            ),
            "repository_files": (
                result[
                    "repository_files"
                ]
            ),
            "tool_trace": (
                sanitize_tool_trace(
                    result["tool_trace"]
                )
            ),
            "evidence": (
                summarize_evidence(
                    result["evidence"]
                )
            ),
            "citation_report": (
                result[
                    "citation_report"
                ]
            ),
            "citation_repaired": (
                result[
                    "citation_repaired"
                ]
            ),
            "memory": memory_status,
        }

    except LookupError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        ) from error


@app.post("/api/v1/clear-memory")
def clear_memory(
    request: SessionRequest,
):
    try:
        session = manager.clear_session(
            repository_path=(
                request.repository_path
            ),
            session_id=request.session_id,
        )

        return {
            "status": "cleared",
            "session_id": (
                session.session_id
            ),
            "memory": (
                session.memory.get_status()
            ),
        }

    except LookupError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.get("/api/v1/memory")
def get_memory(
    repository_path: str = Query(
        min_length=1,
    ),
    session_id: str = Query(
        min_length=1,
        max_length=MAX_SESSION_ID_LENGTH,
    ),
):
    try:
        session = manager.get_session(
            repository_path=(
                repository_path
            ),
            session_id=session_id,
        )

        with session.lock:
            status = (
                session.memory.get_status()
            )

        return {
            "session_id": session_id,
            "memory": status,
        }

    except LookupError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error