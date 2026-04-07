"""Explicit query planning and routing for BioKG-Agent."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from .config import ProjectConfig, default_config


@dataclass(slots=True)
class QueryPlan:
    """Decision-complete query routing output."""

    query: str
    query_type: str
    retrieval_modes: list[str]
    metadata_filters: dict[str, Any]
    detected_entities: list[str]
    use_reranker: bool
    requires_graph_expansion: bool
    requires_api_lookup: bool
    max_iterations: int
    route_confidence: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceAssessment:
    """Assessment of whether more retrieval is needed."""

    enough_evidence: bool
    average_score: float
    candidate_count: int
    reasons: list[str] = field(default_factory=list)
    reformulated_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryRouter:
    """LLM-hookable router with deterministic fallback rules."""

    def __init__(
        self,
        config: ProjectConfig | None = None,
        planner: Callable[[str, list[str]], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or default_config()
        self.planner = planner

    def plan(self, query: str, gene_catalog: Sequence[str]) -> QueryPlan:
        detected_entities = self._detect_entities(query, gene_catalog)
        if self.planner is not None:
            try:
                response = self.planner(query, list(detected_entities))
                return QueryPlan(
                    query=query,
                    query_type=str(response.get("query_type", "hybrid")),
                    retrieval_modes=list(response.get("retrieval_modes", ["dense", "bm25"])),
                    metadata_filters=dict(response.get("metadata_filters", {"genes": list(detected_entities)})),
                    detected_entities=list(response.get("detected_entities", list(detected_entities))),
                    use_reranker=bool(response.get("use_reranker", True)),
                    requires_graph_expansion=bool(response.get("requires_graph_expansion", True)),
                    requires_api_lookup=bool(response.get("requires_api_lookup", True)),
                    max_iterations=int(response.get("max_iterations", self.config.max_retrieval_iterations)),
                    route_confidence=float(response.get("route_confidence", 0.8)),
                    rationale=str(response.get("rationale", "Planner-provided route")),
                )
            except Exception:
                pass
        return self._fallback_plan(query, detected_entities)

    def _detect_entities(self, query: str, gene_catalog: Sequence[str]) -> list[str]:
        query_upper = query.upper()
        normalized_tokens = {
            re.sub(r"[^A-Z0-9]", "", token.upper())
            for token in re.findall(r"[A-Za-z0-9\-_./]+", query)
        }
        normalized_tokens.discard("")
        # Use word-boundary matching to avoid false positives like "EGF" inside "EGFR"
        # Sort catalog longest-first so longer symbols (e.g. EGFR) take priority over
        # short substrings (e.g. EGF) when the same match region is claimed.
        sorted_catalog = sorted(gene_catalog, key=len, reverse=True)
        seen: set[str] = set()
        ordered = []
        for gene in sorted_catalog:
            if not gene:
                continue
            gene_norm = re.sub(r"[^A-Z0-9]", "", gene.upper())
            exact_match = re.search(r'\b' + re.escape(gene) + r'\b', query_upper) is not None
            normalized_match = bool(gene_norm and gene_norm in normalized_tokens)
            if exact_match or normalized_match:
                if gene not in seen:
                    seen.add(gene)
                    ordered.append(gene)
        return ordered

    def _fallback_plan(self, query: str, detected_entities: Sequence[str]) -> QueryPlan:
        lowered = query.lower()
        if any(term in lowered for term in ["interact", "relationship", "bind", "complex", "link"]):
            query_type = "relationship"
            retrieval_modes = ["graph", "dense", "bm25"]
        elif any(term in lowered for term in ["mechanism", "why", "how", "pathway", "process", "therapy"]):
            query_type = "mechanistic"
            retrieval_modes = ["dense", "bm25", "graph"]
        elif any(term in lowered for term in ["paper", "study", "abstract", "literature", "pubmed"]):
            query_type = "literature"
            retrieval_modes = ["dense", "bm25"]
        elif detected_entities and len(query.split()) <= 6:
            query_type = "entity"
            retrieval_modes = ["graph", "bm25"]
        else:
            query_type = "hybrid"
            retrieval_modes = ["dense", "bm25", "graph"]

        requires_graph = query_type in {"relationship", "mechanistic", "hybrid", "entity"}
        requires_api = query_type in {"entity", "relationship", "mechanistic", "hybrid"}
        rationale = f"Rule-based fallback classified query as {query_type}."
        return QueryPlan(
            query=query,
            query_type=query_type,
            retrieval_modes=retrieval_modes,
            metadata_filters={"genes": list(detected_entities)},
            detected_entities=list(detected_entities),
            use_reranker=True,
            requires_graph_expansion=requires_graph,
            requires_api_lookup=requires_api,
            max_iterations=self.config.max_retrieval_iterations,
            route_confidence=0.7,
            rationale=rationale,
        )

    def assess(
        self,
        plan: QueryPlan,
        final_scores: Sequence[float],
        graph_summary: dict[str, Any] | None = None,
        evidence_flags: dict[str, bool] | None = None,
        iteration: int = 0,
    ) -> EvidenceAssessment:
        scores = list(final_scores)
        average_score = sum(scores) / max(len(scores), 1)
        candidate_count = len(scores)
        graph_summary = graph_summary or {}
        evidence_flags = evidence_flags or {}
        reasons: list[str] = []

        if candidate_count < self.config.min_supporting_evidence:
            reasons.append("insufficient_retrieval_hits")
        if average_score < self.config.confidence_threshold:
            reasons.append("low_average_retrieval_score")
        if plan.query_type in {"relationship", "mechanistic", "hybrid"} and len(plan.detected_entities) < 2 and iteration == 0:
            reasons.append("underspecified_target")
        if plan.requires_graph_expansion and graph_summary.get("edges", 0) == 0:
            reasons.append("missing_graph_edges")
        if plan.query_type in {"relationship", "mechanistic"} and not evidence_flags.get("has_relation_evidence", False):
            reasons.append("missing_relation_evidence")
        if plan.query_type == "mechanistic" and not evidence_flags.get("has_process_evidence", False):
            reasons.append("missing_process_evidence")

        enough_evidence = not reasons or iteration >= max(plan.max_iterations - 1, 0)
        reformulated_query = None if enough_evidence else self.reformulate_query(plan, reasons)
        return EvidenceAssessment(
            enough_evidence=enough_evidence,
            average_score=average_score,
            candidate_count=candidate_count,
            reasons=reasons,
            reformulated_query=reformulated_query,
        )

    def reformulate_query(self, plan: QueryPlan, reasons: Sequence[str]) -> str:
        additions = []
        if "missing_relation_evidence" in reasons:
            additions.extend(["interaction", "relationship", "network"])
        if "missing_process_evidence" in reasons:
            additions.extend(["pathway", "mechanism", "biological process"])
        if "insufficient_retrieval_hits" in reasons:
            additions.extend(plan.detected_entities)
        if "underspecified_target" in reasons:
            additions.extend(["therapy", "drug", "pathway"])
        additions = [item for item in additions if item]
        if not additions:
            return plan.query
        return f"{plan.query} {' '.join(dict.fromkeys(additions))}".strip()
