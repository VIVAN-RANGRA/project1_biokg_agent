"""Tests for biokg_agent.router"""
import pytest
from biokg_agent.config import ProjectConfig
from biokg_agent.router import EvidenceAssessment, QueryPlan, QueryRouter


@pytest.fixture
def router():
    cfg = ProjectConfig()
    return QueryRouter(config=cfg)


GENE_CATALOG = ["TP53", "BRCA1", "EGFR", "IL6", "PARP1", "MDM2", "AKT1", "STAT3"]


class TestQueryRouter:
    def test_plan_returns_query_plan(self, router):
        plan = router.plan("What does TP53 do?", GENE_CATALOG)
        assert isinstance(plan, QueryPlan)

    def test_plan_detects_entities(self, router):
        plan = router.plan("Tell me about TP53 and BRCA1", GENE_CATALOG)
        assert "TP53" in plan.detected_entities
        assert "BRCA1" in plan.detected_entities

    def test_detect_entities_finds_genes(self, router):
        detected = router._detect_entities("TP53 interacts with MDM2", GENE_CATALOG)
        assert "TP53" in detected
        assert "MDM2" in detected

    def test_detect_entities_case_insensitive(self, router):
        detected = router._detect_entities("tp53 interacts with mdm2", GENE_CATALOG)
        assert "TP53" in detected
        assert "MDM2" in detected

    def test_detect_entities_no_duplicates(self, router):
        detected = router._detect_entities("TP53 TP53 TP53", GENE_CATALOG)
        assert detected.count("TP53") == 1

    def test_detect_entities_empty_query(self, router):
        detected = router._detect_entities("", GENE_CATALOG)
        assert detected == []


class TestFallbackPlan:
    def test_mechanism_query(self, router):
        plan = router._fallback_plan("What is the mechanism of TP53?", ["TP53"])
        assert plan.query_type == "mechanistic"
        assert "graph" in plan.retrieval_modes

    def test_relationship_query(self, router):
        plan = router._fallback_plan("How do TP53 and BRCA1 interact?", ["TP53", "BRCA1"])
        assert plan.query_type == "relationship"

    def test_literature_query(self, router):
        plan = router._fallback_plan("Find papers about EGFR", ["EGFR"])
        assert plan.query_type == "literature"
        assert "dense" in plan.retrieval_modes
        assert "bm25" in plan.retrieval_modes

    def test_entity_query(self, router):
        plan = router._fallback_plan("TP53", ["TP53"])
        assert plan.query_type == "entity"

    def test_hybrid_query(self, router):
        plan = router._fallback_plan("general question about something unrelated", [])
        assert plan.query_type == "hybrid"

    def test_pathway_keyword_triggers_mechanistic(self, router):
        plan = router._fallback_plan("What pathway is BRCA1 in?", ["BRCA1"])
        assert plan.query_type == "mechanistic"

    def test_therapy_keyword_triggers_mechanistic(self, router):
        plan = router._fallback_plan("therapy options for EGFR mutant cancer", ["EGFR"])
        assert plan.query_type == "mechanistic"

    def test_link_keyword_triggers_relationship(self, router):
        plan = router._fallback_plan("What is the link between TP53 and MDM2?", ["TP53", "MDM2"])
        assert plan.query_type == "relationship"

    def test_fallback_plan_rationale(self, router):
        plan = router._fallback_plan("test query", [])
        assert "Rule-based fallback" in plan.rationale


class TestAssess:
    def test_sufficient_evidence(self, router):
        plan = QueryPlan(
            query="test",
            query_type="entity",
            retrieval_modes=["dense", "bm25"],
            metadata_filters={},
            detected_entities=["TP53"],
            use_reranker=True,
            requires_graph_expansion=False,
            requires_api_lookup=False,
            max_iterations=2,
            route_confidence=0.7,
            rationale="test",
        )
        assessment = router.assess(
            plan=plan,
            final_scores=[0.8, 0.7, 0.9],
            graph_summary={"edges": 5},
            evidence_flags={"has_relation_evidence": True, "has_process_evidence": True},
            iteration=0,
        )
        assert assessment.enough_evidence is True

    def test_insufficient_evidence(self, router):
        plan = QueryPlan(
            query="test",
            query_type="mechanistic",
            retrieval_modes=["dense", "bm25", "graph"],
            metadata_filters={},
            detected_entities=["TP53"],
            use_reranker=True,
            requires_graph_expansion=True,
            requires_api_lookup=True,
            max_iterations=3,
            route_confidence=0.7,
            rationale="test",
        )
        assessment = router.assess(
            plan=plan,
            final_scores=[0.2],
            graph_summary={"edges": 0},
            evidence_flags={"has_relation_evidence": False, "has_process_evidence": False},
            iteration=0,
        )
        assert assessment.enough_evidence is False
        assert len(assessment.reasons) > 0

    def test_assess_final_iteration_always_enough(self, router):
        plan = QueryPlan(
            query="test",
            query_type="mechanistic",
            retrieval_modes=["dense"],
            metadata_filters={},
            detected_entities=[],
            use_reranker=True,
            requires_graph_expansion=True,
            requires_api_lookup=True,
            max_iterations=2,
            route_confidence=0.7,
            rationale="test",
        )
        assessment = router.assess(
            plan=plan,
            final_scores=[0.1],
            graph_summary={"edges": 0},
            evidence_flags={},
            iteration=1,  # max_iterations - 1
        )
        assert assessment.enough_evidence is True


class TestReformulateQuery:
    def test_adds_relation_terms(self, router):
        plan = QueryPlan(
            query="TP53 query",
            query_type="relationship",
            retrieval_modes=["dense"],
            metadata_filters={},
            detected_entities=["TP53"],
            use_reranker=True,
            requires_graph_expansion=True,
            requires_api_lookup=True,
            max_iterations=2,
            route_confidence=0.7,
            rationale="test",
        )
        result = router.reformulate_query(plan, ["missing_relation_evidence"])
        assert "interaction" in result or "relationship" in result

    def test_adds_process_terms(self, router):
        plan = QueryPlan(
            query="TP53 query",
            query_type="mechanistic",
            retrieval_modes=["dense"],
            metadata_filters={},
            detected_entities=["TP53"],
            use_reranker=True,
            requires_graph_expansion=True,
            requires_api_lookup=True,
            max_iterations=2,
            route_confidence=0.7,
            rationale="test",
        )
        result = router.reformulate_query(plan, ["missing_process_evidence"])
        assert "pathway" in result or "mechanism" in result

    def test_no_additions_returns_original(self, router):
        plan = QueryPlan(
            query="original query",
            query_type="entity",
            retrieval_modes=["dense"],
            metadata_filters={},
            detected_entities=[],
            use_reranker=True,
            requires_graph_expansion=False,
            requires_api_lookup=False,
            max_iterations=2,
            route_confidence=0.7,
            rationale="test",
        )
        result = router.reformulate_query(plan, [])
        assert result == "original query"


class TestEvidenceAssessment:
    def test_to_dict(self):
        ea = EvidenceAssessment(
            enough_evidence=True,
            average_score=0.75,
            candidate_count=5,
            reasons=[],
        )
        d = ea.to_dict()
        assert d["enough_evidence"] is True
        assert d["average_score"] == 0.75
        assert d["candidate_count"] == 5

    def test_to_dict_with_reformulated_query(self):
        ea = EvidenceAssessment(
            enough_evidence=False,
            average_score=0.3,
            candidate_count=1,
            reasons=["low_average_retrieval_score"],
            reformulated_query="expanded query",
        )
        d = ea.to_dict()
        assert d["reformulated_query"] == "expanded query"


class TestQueryPlan:
    def test_to_dict(self):
        plan = QueryPlan(
            query="test",
            query_type="entity",
            retrieval_modes=["dense"],
            metadata_filters={},
            detected_entities=["TP53"],
            use_reranker=True,
            requires_graph_expansion=False,
            requires_api_lookup=False,
            max_iterations=2,
            route_confidence=0.7,
            rationale="test plan",
        )
        d = plan.to_dict()
        assert d["query"] == "test"
        assert d["query_type"] == "entity"
        assert d["detected_entities"] == ["TP53"]
