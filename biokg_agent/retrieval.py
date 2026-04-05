"""Hybrid retrieval, fusion, and reranking for BioKG-Agent."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .checkpoints import CheckpointStore, load_pickle, save_pickle
from .config import ProjectConfig, default_config

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None
    SentenceTransformer = None


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "what",
    "which",
    "who",
    "why",
    "how",
    "does",
    "do",
    "find",
    "show",
    "tell",
    "me",
}


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token and token.lower() not in STOPWORDS
    ]


def _normalize_gene(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _text_from_record(record: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(record.get("title", "")),
            str(record.get("abstract", "")),
            str(record.get("gene", "")),
            str(record.get("pathway", "")),
        ]
    ).strip()


def _normalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    if not raw_scores:
        return {}
    values = list(raw_scores.values())
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper):
        return {key: 1.0 if value > 0 else 0.0 for key, value in raw_scores.items()}
    return {
        key: (value - lower) / (upper - lower)
        for key, value in raw_scores.items()
    }


@dataclass(slots=True)
class RetrievalCandidate:
    """Unified retrieval result across dense, sparse, and graph channels."""

    candidate_id: str
    source_type: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_score: float = 0.0
    sparse_score: float = 0.0
    graph_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalBundle:
    """Container for retrieved and optionally reranked evidence."""

    query: str
    strategy: list[str]
    candidates: list[RetrievalCandidate]
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "strategy": list(self.strategy),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata_filters": dict(self.metadata_filters),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(slots=True)
class HybridRetrievalEngine:
    """Hybrid retrieval engine with dense, sparse, fusion, and reranking stages."""

    records: list[dict[str, Any]] = field(default_factory=list)
    config: ProjectConfig = field(default_factory=default_config)
    doc_tokens: list[list[str]] = field(default_factory=list)
    doc_freqs: list[Counter[str]] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)
    avg_doc_len: float = 0.0
    dense_backend: str = "hashed"
    dense_matrix: list[list[float]] = field(default_factory=list)
    dense_dim: int = 256
    reranker_backend: str = "heuristic"

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        config: ProjectConfig | None = None,
    ) -> "HybridRetrievalEngine":
        engine = cls(records=[dict(record) for record in records], config=config or default_config())
        engine.rebuild()
        return engine

    @classmethod
    def from_bundle(
        cls,
        bundle: Mapping[str, Sequence[Mapping[str, Any]]],
        config: ProjectConfig | None = None,
    ) -> "HybridRetrievalEngine":
        return cls.from_records(bundle.get("pubmed_records", []), config=config)

    def rebuild(self) -> None:
        self._build_sparse_stats()
        self._build_dense_index()

    def _build_sparse_stats(self) -> None:
        self.doc_tokens = []
        self.doc_freqs = []
        df: Counter[str] = Counter()
        for record in self.records:
            tokens = _tokenize(_text_from_record(record))
            counts = Counter(tokens)
            self.doc_tokens.append(tokens)
            self.doc_freqs.append(counts)
            for token in counts:
                df[token] += 1
        doc_count = max(len(self.records), 1)
        self.avg_doc_len = sum(len(tokens) for tokens in self.doc_tokens) / doc_count
        self.idf = {
            token: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def _build_dense_index(self) -> None:
        backend = self.config.dense_backend
        if backend in {"auto", "sentence_transformers"} and SentenceTransformer is not None:
            try:  # pragma: no cover - optional heavy path
                model = SentenceTransformer(self.config.embedding_model)
                matrix = model.encode(
                    [_text_from_record(record) for record in self.records],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self.dense_matrix = [list(map(float, row)) for row in matrix]
                self.dense_dim = len(self.dense_matrix[0]) if self.dense_matrix else 256
                self.dense_backend = "sentence_transformers"
                return
            except Exception:
                pass

        self.dense_matrix = [self._hashed_embedding(_text_from_record(record)) for record in self.records]
        self.dense_dim = len(self.dense_matrix[0]) if self.dense_matrix else 256
        self.dense_backend = "hashed"

    def _hashed_embedding(self, text: str, dim: int | None = None) -> list[float]:
        dim = dim or self.dense_dim or 256
        vector = [0.0] * dim
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def save(self, path: str | Path) -> Path:
        return save_pickle(self, path)

    def dump(self, path: str | Path) -> Path:
        return self.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRetrievalEngine":
        payload = load_pickle(path)
        if isinstance(payload, cls):
            return payload
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")

    def to_checkpoint(self, store: CheckpointStore, name: str) -> Path:
        return store.save_pickle(self, name)

    @classmethod
    def from_checkpoint(cls, store: CheckpointStore, name: str) -> "HybridRetrievalEngine":
        return cls.load(store.path(name))

    def _record_matches_filters(self, record: Mapping[str, Any], metadata_filters: Mapping[str, Any] | None) -> bool:
        if not metadata_filters:
            return True
        genes = metadata_filters.get("genes") or []
        if genes:
            record_gene = _normalize_gene(str(record.get("gene", "")))
            if record_gene not in {_normalize_gene(gene) for gene in genes}:
                return False
        source_types = metadata_filters.get("source_types") or []
        if source_types:
            record_source = str(record.get("source_type", "literature"))
            if record_source not in source_types:
                return False
        return True

    def dense_score(self, query: str, record_index: int) -> float:
        if record_index < 0 or record_index >= len(self.records):
            return 0.0
        query_embedding = self._hashed_embedding(query, dim=self.dense_dim) if self.dense_backend == "hashed" else None
        if self.dense_backend == "sentence_transformers" and SentenceTransformer is not None:  # pragma: no cover - optional heavy path
            try:
                model = SentenceTransformer(self.config.embedding_model)
                query_embedding = list(map(float, model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]))
            except Exception:
                query_embedding = self._hashed_embedding(query, dim=self.dense_dim)
        if query_embedding is None:
            query_embedding = self._hashed_embedding(query, dim=self.dense_dim)
        doc_embedding = self.dense_matrix[record_index]
        return float(sum(a * b for a, b in zip(query_embedding, doc_embedding)))

    def sparse_score(self, query: str, record_index: int, k1: float = 1.4, b: float = 0.75) -> float:
        if record_index < 0 or record_index >= len(self.records):
            return 0.0
        tokens = _tokenize(query)
        if not tokens:
            return 0.0
        freqs = self.doc_freqs[record_index]
        doc_len = len(self.doc_tokens[record_index]) or 1
        score = 0.0
        for token in tokens:
            tf = freqs.get(token)
            if not tf:
                continue
            idf = self.idf.get(token, 0.0)
            denom = tf + k1 * (1 - b + b * doc_len / (self.avg_doc_len or 1.0))
            score += idf * (tf * (k1 + 1)) / denom
        return float(score)

    def graph_prior_score(
        self,
        record: Mapping[str, Any],
        graph_hits: Mapping[str, float] | None,
    ) -> float:
        if not graph_hits:
            return 0.0
        gene = _normalize_gene(str(record.get("gene", "")))
        return float(graph_hits.get(gene, 0.0))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: Mapping[str, Any] | None = None,
        strategy: Sequence[str] | None = None,
        graph_hits: Mapping[str, float] | None = None,
    ) -> RetrievalBundle:
        strategy = list(strategy or ["dense", "bm25"])
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        graph_scores: dict[str, float] = {}
        candidates: dict[str, RetrievalCandidate] = {}

        for index, record in enumerate(self.records):
            if not self._record_matches_filters(record, metadata_filters):
                continue
            candidate_id = str(record.get("pmid", index))
            dense = self.dense_score(query, index) if "dense" in strategy else 0.0
            sparse = self.sparse_score(query, index) if "bm25" in strategy else 0.0
            graph = self.graph_prior_score(record, graph_hits) if "graph" in strategy else 0.0
            dense_scores[candidate_id] = dense
            sparse_scores[candidate_id] = sparse
            graph_scores[candidate_id] = graph
            candidates[candidate_id] = RetrievalCandidate(
                candidate_id=candidate_id,
                source_type=str(record.get("source_type", "literature")),
                payload=dict(record),
                metadata={"gene": record.get("gene", ""), "title": record.get("title", "")},
                dense_score=dense,
                sparse_score=sparse,
                graph_score=graph,
            )

        normalized_dense = _normalize_scores(dense_scores)
        normalized_sparse = _normalize_scores(sparse_scores)
        normalized_graph = _normalize_scores(graph_scores)
        dense_weight = self.config.dense_weight if "dense" in strategy else 0.0
        sparse_weight = self.config.sparse_weight if "bm25" in strategy else 0.0
        graph_weight = self.config.graph_weight if "graph" in strategy else 0.0
        weight_total = dense_weight + sparse_weight + graph_weight or 1.0

        ranked_candidates = []
        for candidate_id, candidate in candidates.items():
            candidate.dense_score = float(normalized_dense.get(candidate_id, 0.0))
            candidate.sparse_score = float(normalized_sparse.get(candidate_id, 0.0))
            candidate.graph_score = float(normalized_graph.get(candidate_id, 0.0))
            candidate.final_score = (
                dense_weight * candidate.dense_score
                + sparse_weight * candidate.sparse_score
                + graph_weight * candidate.graph_score
            ) / weight_total
            if candidate.final_score <= 0:
                continue
            ranked_candidates.append(candidate)

        ranked_candidates.sort(key=lambda item: item.final_score, reverse=True)
        return RetrievalBundle(
            query=query,
            strategy=list(strategy),
            candidates=ranked_candidates[:top_k],
            metadata_filters=dict(metadata_filters or {}),
            diagnostics={
                "dense_backend": self.dense_backend,
                "reranker_backend": self.reranker_backend,
                "candidate_pool": len(ranked_candidates),
            },
        )

    def _heuristic_rerank_score(self, query: str, candidate: RetrievalCandidate) -> float:
        query_tokens = set(_tokenize(query))
        candidate_tokens = set(_tokenize(_text_from_record(candidate.payload)))
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & candidate_tokens) / len(query_tokens)
        exact_gene_bonus = 0.15 if _normalize_gene(str(candidate.payload.get("gene", ""))) in query.upper() else 0.0
        return min(1.0, overlap + exact_gene_bonus)

    def rerank(self, query: str, bundle: RetrievalBundle, top_n: int | None = None) -> RetrievalBundle:
        top_n = top_n or self.config.rerank_top_n
        candidates = list(bundle.candidates[:top_n])
        backend = "heuristic"

        if self.config.enable_reranker and self.config.reranker_backend in {"auto", "cross_encoder"} and CrossEncoder is not None:
            try:  # pragma: no cover - optional heavy path
                backend = "cross_encoder"
                reranker = CrossEncoder(self.config.reranker_model)
                pairs = [(query, _text_from_record(candidate.payload)) for candidate in candidates]
                scores = reranker.predict(pairs)
                normalized = _normalize_scores(
                    {candidate.candidate_id: float(score) for candidate, score in zip(candidates, scores)}
                )
                for candidate in candidates:
                    candidate.rerank_score = normalized.get(candidate.candidate_id, 0.0)
            except Exception:
                backend = "heuristic"

        if backend == "heuristic":
            normalized = _normalize_scores(
                {
                    candidate.candidate_id: self._heuristic_rerank_score(query, candidate)
                    for candidate in candidates
                }
            )
            for candidate in candidates:
                candidate.rerank_score = normalized.get(candidate.candidate_id, 0.0)

        for candidate in candidates:
            candidate.final_score = (
                (1.0 - self.config.rerank_weight) * candidate.final_score
                + self.config.rerank_weight * candidate.rerank_score
            )
        candidates.sort(key=lambda item: item.final_score, reverse=True)

        updated_bundle = RetrievalBundle(
            query=bundle.query,
            strategy=list(bundle.strategy),
            candidates=candidates + list(bundle.candidates[top_n:]),
            metadata_filters=dict(bundle.metadata_filters),
            diagnostics={**bundle.diagnostics, "reranker_backend": backend},
        )
        self.reranker_backend = backend
        return updated_bundle

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        bundle = self.retrieve(query=query, top_k=top_k, strategy=["dense", "bm25"])
        bundle = self.rerank(query=query, bundle=bundle, top_n=min(top_k, self.config.rerank_top_n))
        results = []
        for candidate in bundle.candidates[:top_k]:
            payload = candidate.payload
            results.append(
                {
                    "pmid": payload.get("pmid", ""),
                    "title": payload.get("title", ""),
                    "snippet": str(payload.get("abstract", ""))[:200],
                    "gene": payload.get("gene", ""),
                    "dense_score": candidate.dense_score,
                    "sparse_score": candidate.sparse_score,
                    "rerank_score": candidate.rerank_score,
                    "similarity_score": candidate.final_score,
                    "final_score": candidate.final_score,
                }
            )
        return results


SimpleRetrievalIndex = HybridRetrievalEngine
