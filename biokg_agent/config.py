"""Configuration helpers for BioKG-Agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ProjectConfig:
    """Lightweight runtime configuration."""

    working_dir: str = field(default_factory=lambda: str(_project_root()))
    checkpoint_dir: str = field(default_factory=lambda: str(_project_root() / "checkpoints"))
    prebuilt_data_dir: str = field(default_factory=lambda: str(_project_root() / "checkpoints"))
    model_id: str = "demo-model"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    dense_backend: str = "auto"
    reranker_backend: str = "heuristic"
    enable_live_apis: bool = False
    enable_reranker: bool = True
    enable_graph_retrieval: bool = True
    enable_metadata_filters: bool = True
    rag_top_k: int = 10
    string_score_threshold: int = 700
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    graph_weight: float = 0.2
    rerank_weight: float = 0.25
    max_retrieval_iterations: int = 2
    min_supporting_evidence: int = 2
    confidence_threshold: float = 0.55
    route_confidence_threshold: float = 0.60
    rerank_top_n: int = 8
    max_graph_hops: int = 2
    # LLM settings
    llm_model_id: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    llm_device: str = "auto"
    llm_max_new_tokens: int = 1024
    # Disabled by default — enabled automatically when GROQ_API_KEY or
    # BIOKG_LLM_BACKEND=local is set via from_env()
    enable_llm_synthesis: bool = False
    enable_llm_planner: bool = False

    # API settings
    ncbi_api_key: str | None = None
    ncbi_email: str | None = None
    uniprot_timeout: int = 15
    kegg_timeout: int = 15
    drugbank_data_path: str | None = None

    # FAISS settings
    faiss_backend: str = "auto"  # "auto", "faiss", "hashed"
    faiss_index_type: str = "IndexFlatIP"

    # Groq API (free tier — cloud LLM, no GPU needed)
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    llm_backend: str = "auto"  # "auto", "groq", "local"

    # Ngrok
    ngrok_auth_token: str | None = None

    gradio_share: bool = False
    gradio_port: int = 7860
    demo_bundle_name: str = "demo_bundle.json"
    retriever_checkpoint_name: str = "simple_retrieval_index.pkl"
    graph_checkpoint_name: str = "kg_session.pkl"
    graph_html_name: str = "kg_graph.html"
    query_plan_name: str = "query_plan.json"
    retrieval_trace_name: str = "retrieval_trace.json"
    rerank_trace_name: str = "rerank_trace.json"
    iteration_trace_name: str = "iteration_trace.json"
    confidence_report_name: str = "confidence_report.json"
    provenance_table_name: str = "provenance_table.json"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def working_dir_path(self) -> Path:
        return Path(self.working_dir)

    @property
    def checkpoint_dir_path(self) -> Path:
        return Path(self.checkpoint_dir)

    @property
    def prebuilt_data_dir_path(self) -> Path:
        return Path(self.prebuilt_data_dir)

    @property
    def bundle_checkpoint_path(self) -> Path:
        return self.checkpoint_dir_path / self.demo_bundle_name

    @property
    def retriever_checkpoint_path(self) -> Path:
        return self.checkpoint_dir_path / self.retriever_checkpoint_name

    @property
    def graph_checkpoint_path(self) -> Path:
        return self.checkpoint_dir_path / self.graph_checkpoint_name

    @property
    def graph_html_path(self) -> Path:
        return self.checkpoint_dir_path / self.graph_html_name

    @property
    def query_plan_path(self) -> Path:
        return self.checkpoint_dir_path / self.query_plan_name

    @property
    def retrieval_trace_path(self) -> Path:
        return self.checkpoint_dir_path / self.retrieval_trace_name

    @property
    def rerank_trace_path(self) -> Path:
        return self.checkpoint_dir_path / self.rerank_trace_name

    @property
    def iteration_trace_path(self) -> Path:
        return self.checkpoint_dir_path / self.iteration_trace_name

    @property
    def confidence_report_path(self) -> Path:
        return self.checkpoint_dir_path / self.confidence_report_name

    @property
    def provenance_table_path(self) -> Path:
        return self.checkpoint_dir_path / self.provenance_table_name


    @classmethod
    def from_env(cls) -> "ProjectConfig":
        """Load config with env var overrides."""
        config = cls()
        import os
        config.ncbi_api_key = os.environ.get("NCBI_API_KEY", config.ncbi_api_key)
        config.ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN", config.ngrok_auth_token)
        config.drugbank_data_path = os.environ.get("DRUGBANK_DATA_PATH", config.drugbank_data_path)
        config.llm_model_id = os.environ.get("BIOKG_MODEL_ID", config.llm_model_id)
        config.llm_device = os.environ.get("BIOKG_DEVICE", config.llm_device)
        config.groq_api_key = os.environ.get("GROQ_API_KEY", "") or None
        config.groq_model = os.environ.get("GROQ_MODEL", config.groq_model)
        config.llm_backend = os.environ.get("BIOKG_LLM_BACKEND", config.llm_backend)
        if os.environ.get("BIOKG_ENABLE_LIVE_APIS"):
            config.enable_live_apis = True
        # Auto-enable LLM synthesis when a valid backend is available:
        #   • GROQ_API_KEY is set (cloud, free tier, recommended)
        #   • BIOKG_LLM_BACKEND=local explicitly requested (local GPU)
        _has_groq = bool(config.groq_api_key)
        _has_local = config.llm_backend == "local"
        if _has_groq or _has_local:
            config.enable_llm_synthesis = True
            config.enable_llm_planner = True
        return config


def default_config() -> ProjectConfig:
    return ProjectConfig()
