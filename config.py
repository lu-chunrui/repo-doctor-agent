import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Code Analysis
    long_function_line_threshold: int = 50

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 120
    llm_max_retries: int = 3
    llm_temperature: float = 0.1

    # Agent
    max_tool_steps: int = 4
    max_tool_result_chars: int = 30000
    max_tool_content_chars: int = 12000
    memory_max_turns: int = 6
    memory_max_chars: int = 18000

    # Dense + BM25 + RRF
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_batch_size: int = 16
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 20
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0

    # Query Rewrite
    query_rewrite_enabled: bool = True
    query_rewrite_temperature: float = 0.0
    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Reranker
    reranker_enabled: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = 4
    reranker_max_length: int = 512
    reranker_recall_k: int = 20
    reranker_final_k: int = 5
    reranker_allow_fallback: bool = True
    reranker_max_content_chars: int = 8000

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = (
        "http://127.0.0.1:8501,http://localhost:8501"
    )
    allowed_roots: str = ""
    max_session_id_length: int = 128
    max_message_length: int = 100000
    max_log_length: int = 200000

    # UI
    backend_url: str = "http://127.0.0.1:8000"
    default_repository: str = ""
    request_timeout: int = 300
    index_timeout: int = 900

    # Paths
    runtime_directory: Path = PROJECT_ROOT / "runtime_data"

    # 软熔断
    soft_fuse_enabled: bool = False
    soft_fuse_similarity_threshold: Optional[float] = None

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def parsed_cors_origins(self):
        return [
            value.strip()
            for value in self.cors_origins.split(",")
            if value.strip()
        ]

    def parsed_allowed_roots(self):
        return [
            Path(value.strip()).expanduser().resolve()
            for value in self.allowed_roots.split(os.pathsep)
            if value.strip()
        ]
    def resolved_default_repository(self) -> Path:

        raw_path = self.default_repository.strip()

        if not raw_path:
            raise ValueError(
                "DEFAULT_REPOSITORY 不能为空"
            )

        repository_path = (
            Path(raw_path)
            .expanduser()
            .resolve()
        )

        if not repository_path.is_dir():
            raise FileNotFoundError(
                f"仓库目录不存在：{repository_path}"
            )

        return repository_path


settings = Settings()