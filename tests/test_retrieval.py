"""Tests for biokg_agent.retrieval"""
import pytest
from biokg_agent.config import ProjectConfig
from biokg_agent.data import DEMO_BUNDLE
from biokg_agent.retrieval import (
    HybridRetrievalEngine,
    RetrievalBundle,
    RetrievalCandidate,
    _normalize_scores,
    _tokenize,
)


@pytest.fixture
def engine():
    cfg = ProjectConfig()
    cfg.dense_backend = "hashed"
    cfg.reranker_backend = "heuristic"
    return HybridRetrievalEngine.from_records(DEMO_BUNDLE["pubmed_records"], config=cfg)


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("TP53 DNA repair")
        assert "tp53" in tokens
        assert "dna" in tokens
        assert "repair" in tokens

    def test_strips_stopwords(self):
        tokens = _tokenize("what is the mechanism of TP53")
        assert "what" not in tokens
        assert "is" not in tokens
        assert "the" not in tokens
        assert "of" not in tokens
        assert "tp53" in tokens
        assert "mechanism" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_stopwords(self):
        tokens = _tokenize("the and or is")
        assert tokens == []


class TestNormalizeScores:
    def test_empty_dict(self):
        assert _normalize_scores({}) == {}

    def test_all_same_positive_values(self):
        result = _normalize_scores({"a": 5.0, "b": 5.0})
        assert result["a"] == 1.0
        assert result["b"] == 1.0

    def test_all_same_zero_values(self):
        result = _normalize_scores({"a": 0.0, "b": 0.0})
        assert result["a"] == 0.0
        assert result["b"] == 0.0

    def test_normal_range(self):
        result = _normalize_scores({"a": 1.0, "b": 3.0, "c": 5.0})
        assert result["a"] == pytest.approx(0.0)
        assert result["c"] == pytest.approx(1.0)

    def test_single_value(self):
        result = _normalize_scores({"a": 3.0})
        assert result["a"] == 1.0


class TestHybridRetrievalEngine:
    def test_from_records_creates_engine(self, engine):
        assert isinstance(engine, HybridRetrievalEngine)
        assert len(engine.records) == len(DEMO_BUNDLE["pubmed_records"])

    def test_sparse_score_returns_float(self, engine):
        score = engine.sparse_score("TP53 DNA damage", 0)
        assert isinstance(score, float)

    def test_sparse_score_positive_for_matching_doc(self, engine):
        # Record 0 is about TP53
        score = engine.sparse_score("TP53 DNA damage apoptosis", 0)
        assert score > 0.0

    def test_sparse_score_zero_for_empty_query(self, engine):
        score = engine.sparse_score("", 0)
        assert score == 0.0

    def test_sparse_score_invalid_index(self, engine):
        assert engine.sparse_score("test", -1) == 0.0
        assert engine.sparse_score("test", 9999) == 0.0

    def test_dense_score_returns_float(self, engine):
        score = engine.dense_score("TP53 DNA damage", 0)
        assert isinstance(score, float)

    def test_dense_score_invalid_index(self, engine):
        assert engine.dense_score("test", -1) == 0.0
        assert engine.dense_score("test", 9999) == 0.0

    def test_retrieve_returns_retrieval_bundle(self, engine):
        bundle = engine.retrieve("TP53 DNA damage", top_k=3)
        assert isinstance(bundle, RetrievalBundle)
        assert bundle.query == "TP53 DNA damage"

    def test_retrieve_respects_top_k(self, engine):
        bundle = engine.retrieve("TP53", top_k=2)
        assert len(bundle.candidates) <= 2

    def test_retrieve_candidates_have_scores(self, engine):
        bundle = engine.retrieve("TP53 DNA damage", top_k=5)
        for c in bundle.candidates:
            assert isinstance(c.final_score, float)

    def test_retrieve_with_metadata_filters(self, engine):
        bundle = engine.retrieve(
            "DNA repair",
            top_k=5,
            metadata_filters={"genes": ["TP53"]},
        )
        for c in bundle.candidates:
            assert c.payload.get("gene", "").upper() in ("TP53", "")

    def test_retrieve_with_graph_strategy(self, engine):
        bundle = engine.retrieve(
            "TP53 interactions",
            top_k=5,
            strategy=["dense", "bm25", "graph"],
            graph_hits={"TP53": 0.9, "BRCA1": 0.5},
        )
        assert isinstance(bundle, RetrievalBundle)

    def test_rerank_returns_bundle(self, engine):
        bundle = engine.retrieve("TP53 DNA damage", top_k=5)
        reranked = engine.rerank("TP53 DNA damage", bundle)
        assert isinstance(reranked, RetrievalBundle)

    def test_rerank_changes_scores(self, engine):
        bundle = engine.retrieve("TP53 DNA damage", top_k=5)
        original_scores = [c.final_score for c in bundle.candidates]
        reranked = engine.rerank("TP53 DNA damage", bundle)
        new_scores = [c.final_score for c in reranked.candidates]
        # Scores should change (reranking modifies final_score)
        assert new_scores != original_scores or len(bundle.candidates) == 0

    def test_search_convenience_method(self, engine):
        results = engine.search("TP53 DNA damage", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert "pmid" in r
            assert "title" in r
            assert "final_score" in r

    def test_hashed_embedding_is_deterministic(self, engine):
        emb1 = engine._hashed_embedding("TP53 DNA repair")
        emb2 = engine._hashed_embedding("TP53 DNA repair")
        assert emb1 == emb2

    def test_hashed_embedding_dimension(self, engine):
        emb = engine._hashed_embedding("test query")
        assert len(emb) == engine.dense_dim

    def test_save_and_load_round_trip(self, tmp_path, engine):
        path = tmp_path / "retriever.pkl"
        engine.save(path)
        assert path.exists()
        loaded = HybridRetrievalEngine.load(path)
        assert isinstance(loaded, HybridRetrievalEngine)
        assert len(loaded.records) == len(engine.records)

    def test_dense_search(self, engine):
        results = engine.dense_search("TP53", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)


class TestRetrievalCandidate:
    def test_to_dict(self):
        c = RetrievalCandidate(
            candidate_id="123",
            source_type="literature",
            payload={"title": "test"},
        )
        d = c.to_dict()
        assert d["candidate_id"] == "123"
        assert d["source_type"] == "literature"

    def test_default_scores(self):
        c = RetrievalCandidate(candidate_id="1", source_type="lit", payload={})
        assert c.dense_score == 0.0
        assert c.sparse_score == 0.0
        assert c.graph_score == 0.0
        assert c.rerank_score == 0.0
        assert c.final_score == 0.0


class TestRetrievalBundle:
    def test_to_dict(self):
        bundle = RetrievalBundle(
            query="test",
            strategy=["dense", "bm25"],
            candidates=[],
        )
        d = bundle.to_dict()
        assert d["query"] == "test"
        assert d["strategy"] == ["dense", "bm25"]
        assert d["candidates"] == []
