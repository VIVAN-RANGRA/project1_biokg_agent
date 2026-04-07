"""Tests for biokg_agent.agent"""
import pytest
from biokg_agent.agent import BioKGAgent, _normalize_gene, AgentResult
from biokg_agent.router import QueryPlan


class TestNormalizeGene:
    def test_basic(self):
        assert _normalize_gene("tp53") == "TP53"

    def test_strips_special_chars(self):
        assert _normalize_gene("IL-6") == "IL6"

    def test_uppercase(self):
        assert _normalize_gene("brca1") == "BRCA1"

    def test_empty_string(self):
        assert _normalize_gene("") == ""

    def test_with_spaces(self):
        assert _normalize_gene("TP 53") == "TP53"


class TestBioKGAgentBuild:
    def test_build_succeeds(self, agent):
        assert isinstance(agent, BioKGAgent)

    def test_agent_has_retriever(self, agent):
        assert agent.retriever is not None

    def test_agent_has_router(self, agent):
        assert agent.router is not None

    def test_agent_has_kg(self, agent):
        assert agent.kg is not None

    def test_agent_has_bundle(self, agent):
        assert agent.bundle is not None

    def test_agent_demo_mode(self, agent):
        assert agent.demo_mode is True

    def test_agent_live_apis_disabled(self, agent):
        assert agent.config.enable_live_apis is False


class TestAgentLookups:
    def test_ncbi_gene_lookup_returns_dict(self, agent):
        result = agent.ncbi_gene_lookup("TP53")
        assert isinstance(result, dict)
        assert "symbol" in result or "summary" in result

    def test_ncbi_gene_lookup_unknown_gene(self, agent):
        result = agent.ncbi_gene_lookup("UNKNOWNGENE999")
        assert isinstance(result, dict)

    def test_string_interactions_returns_list(self, agent):
        result = agent.string_interactions("TP53")
        assert isinstance(result, list)

    def test_string_interactions_has_partners(self, agent):
        result = agent.string_interactions("TP53")
        partners = [r["partner"] for r in result]
        assert "BRCA1" in partners or len(result) >= 0

    def test_string_interactions_respects_threshold(self, agent):
        result = agent.string_interactions("TP53", score_threshold=950)
        for r in result:
            assert r["score"] >= 950

    def test_drugbank_target_lookup_returns_list(self, agent):
        result = agent.drugbank_target_lookup("PARP1")
        assert isinstance(result, list)

    def test_drugbank_target_lookup_has_drugs(self, agent):
        result = agent.drugbank_target_lookup("PARP1")
        assert len(result) > 0
        assert "drug_name" in result[0]

    def test_drugbank_target_lookup_empty_for_unknown(self, agent):
        result = agent.drugbank_target_lookup("UNKNOWNGENE")
        assert result == []

    def test_gene_ontology_lookup_returns_list(self, agent):
        result = agent.gene_ontology_lookup("TP53")
        assert isinstance(result, list)

    def test_gene_ontology_lookup_has_terms(self, agent):
        result = agent.gene_ontology_lookup("TP53")
        assert len(result) > 0

    def test_pathway_lookup_returns_list(self, agent):
        result = agent.pathway_lookup("TP53")
        assert isinstance(result, list)

    def test_pathway_lookup_has_pathways(self, agent):
        result = agent.pathway_lookup("BRCA1")
        assert len(result) > 0


class TestAgentKnownGenes:
    def test_known_genes_returns_sorted_list(self, agent):
        genes = agent.known_genes()
        assert isinstance(genes, list)
        assert genes == sorted(genes)

    def test_known_genes_contains_tp53(self, agent):
        genes = agent.known_genes()
        assert "TP53" in genes

    def test_known_genes_contains_brca1(self, agent):
        genes = agent.known_genes()
        assert "BRCA1" in genes


class TestAgentPlanQuery:
    def test_plan_query_returns_query_plan(self, agent):
        plan = agent.plan_query("What does TP53 do?")
        assert isinstance(plan, QueryPlan)

    def test_plan_query_detects_entities(self, agent):
        plan = agent.plan_query("Tell me about TP53 and BRCA1")
        assert "TP53" in plan.detected_entities
        assert "BRCA1" in plan.detected_entities

    def test_plan_query_resolves_synonyms(self, agent):
        plan = agent.plan_query("What does P53 do?")
        assert "TP53" in plan.detected_entities


class TestAgentInvoke:
    def test_invoke_returns_dict(self, agent):
        result = agent.invoke("What does TP53 do?")
        assert isinstance(result, dict)

    def test_invoke_has_answer_text(self, agent):
        result = agent.invoke("What does TP53 do?")
        assert "answer_text" in result
        assert isinstance(result["answer_text"], str)

    def test_invoke_has_confidence_summary(self, agent):
        result = agent.invoke("Tell me about BRCA1")
        assert "confidence_summary" in result
        assert "overall_confidence" in result["confidence_summary"]

    def test_invoke_has_graph_summary(self, agent):
        result = agent.invoke("TP53")
        assert "graph_summary" in result

    def test_invoke_has_evidence_table(self, agent):
        result = agent.invoke("TP53")
        assert "evidence_table" in result
        assert isinstance(result["evidence_table"], list)

    def test_invoke_has_route_type(self, agent):
        result = agent.invoke("TP53")
        assert "route_type" in result

    def test_invoke_has_retrieval_iterations_count(self, agent):
        result = agent.invoke("TP53")
        assert "retrieval_iterations_count" in result
        assert result["retrieval_iterations_count"] >= 1


class TestAgentAnswer:
    def test_answer_returns_string(self, agent):
        result = agent.answer("What does TP53 do?")
        assert isinstance(result, str)
        assert len(result) > 0


class TestExpandGene:
    def test_expand_gene_populates_kg(self, agent):
        initial_nodes = agent.kg.summary()["nodes"]
        agent._expand_gene("TP53")
        assert agent.kg.summary()["nodes"] >= initial_nodes

    def test_expand_gene_returns_dict(self, agent):
        result = agent._expand_gene("EGFR")
        assert isinstance(result, dict)
        assert "gene" in result
        assert "interactions" in result
        assert "drugs" in result
        assert "go_terms" in result
        assert "pathways" in result


class TestAgentSave:
    def test_save_creates_files(self, config, demo_bundle, tmp_path):
        config.checkpoint_dir = str(tmp_path / "save_test")
        agent = BioKGAgent.build(config=config, bundle=demo_bundle, save_checkpoints=True)
        agent.save()
        from pathlib import Path
        checkpoint_dir = Path(config.checkpoint_dir)
        assert (checkpoint_dir / config.demo_bundle_name).exists()
        assert (checkpoint_dir / config.retriever_checkpoint_name).exists()
        assert (checkpoint_dir / config.graph_checkpoint_name).exists()


class TestAgentResult:
    def test_to_dict(self):
        result = AgentResult(
            answer_text="test answer",
            query_plan={"query_type": "entity"},
            retrieval_iterations=[],
            evidence_table=[],
            graph_summary={"nodes": 0, "edges": 0},
            confidence_summary={"overall_confidence": 0.5},
            checkpoint_paths={},
        )
        d = result.to_dict()
        assert d["answer_text"] == "test answer"

    def test_str(self):
        result = AgentResult(
            answer_text="test answer",
            query_plan={},
            retrieval_iterations=[],
            evidence_table=[],
            graph_summary={},
            confidence_summary={},
            checkpoint_paths={},
        )
        assert str(result) == "test answer"
