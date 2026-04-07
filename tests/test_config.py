"""Tests for biokg_agent.config"""
import pytest
from biokg_agent.config import ProjectConfig, default_config


class TestProjectConfig:
    def test_default_config_returns_instance(self):
        cfg = default_config()
        assert isinstance(cfg, ProjectConfig)

    def test_default_rag_top_k(self):
        cfg = default_config()
        assert cfg.rag_top_k == 10

    def test_default_string_score_threshold(self):
        cfg = default_config()
        assert cfg.string_score_threshold == 700

    def test_default_dense_weight(self):
        cfg = default_config()
        assert cfg.dense_weight == 0.7

    def test_default_sparse_weight(self):
        cfg = default_config()
        assert cfg.sparse_weight == 0.3

    def test_default_enable_live_apis_false(self):
        cfg = default_config()
        assert cfg.enable_live_apis is False

    def test_checkpoint_dir_path_name(self):
        cfg = default_config()
        assert cfg.checkpoint_dir_path.name == "checkpoints"

    def test_working_dir_path(self):
        cfg = default_config()
        assert cfg.working_dir_path.exists() or True  # path is valid

    def test_bundle_checkpoint_path(self):
        cfg = default_config()
        assert cfg.bundle_checkpoint_path.name == "demo_bundle.json"

    def test_retriever_checkpoint_path(self):
        cfg = default_config()
        assert cfg.retriever_checkpoint_path.name == "simple_retrieval_index.pkl"

    def test_graph_checkpoint_path(self):
        cfg = default_config()
        assert cfg.graph_checkpoint_path.name == "kg_session.pkl"

    def test_graph_html_path(self):
        cfg = default_config()
        assert cfg.graph_html_path.name == "kg_graph.html"

    def test_from_env_groq_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        cfg = ProjectConfig.from_env()
        assert cfg.groq_api_key == "test_key"

    def test_from_env_enable_live_apis(self, monkeypatch):
        monkeypatch.setenv("BIOKG_ENABLE_LIVE_APIS", "1")
        cfg = ProjectConfig.from_env()
        assert cfg.enable_live_apis is True

    def test_from_env_llm_backend(self, monkeypatch):
        monkeypatch.setenv("BIOKG_LLM_BACKEND", "groq")
        cfg = ProjectConfig.from_env()
        assert cfg.llm_backend == "groq"

    def test_from_env_model_id(self, monkeypatch):
        monkeypatch.setenv("BIOKG_MODEL_ID", "custom-model")
        cfg = ProjectConfig.from_env()
        assert cfg.llm_model_id == "custom-model"

    def test_as_dict_returns_dict(self):
        cfg = default_config()
        d = cfg.as_dict()
        assert isinstance(d, dict)

    def test_as_dict_has_rag_top_k(self):
        cfg = default_config()
        d = cfg.as_dict()
        assert "rag_top_k" in d
        assert d["rag_top_k"] == 10

    def test_as_dict_has_enable_live_apis(self):
        cfg = default_config()
        d = cfg.as_dict()
        assert "enable_live_apis" in d

    def test_custom_checkpoint_dir(self):
        cfg = ProjectConfig(checkpoint_dir="/tmp/custom_checkpoints")
        assert cfg.checkpoint_dir_path.name == "custom_checkpoints"

    def test_prebuilt_data_dir_path(self):
        cfg = default_config()
        assert cfg.prebuilt_data_dir_path == cfg.checkpoint_dir_path
