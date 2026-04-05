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
    dense_backend: str = "hashed"
    reranker_backend: str = "heuristic"
    enable_live_apis: bool = False
    enable_reranker: bool = True
    enable_graph_retrieval: bool = True
    enable_metadata_filters: bool = True
    rag_top_k: int = 5
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
    ncbi_api_key: str | None = None
    ncbi_email: str | None = None
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


def default_config() -> ProjectConfig:
    return ProjectConfig()
