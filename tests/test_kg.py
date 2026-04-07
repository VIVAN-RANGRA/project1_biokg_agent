"""Tests for biokg_agent.kg"""
import pytest
from biokg_agent.kg import BioKnowledgeGraph


class TestBioKnowledgeGraph:
    def test_empty_graph_summary(self):
        kg = BioKnowledgeGraph()
        s = kg.summary()
        assert s["nodes"] == 0
        assert s["edges"] == 0

    def test_add_entity_creates_node(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        assert kg.summary()["nodes"] == 1

    def test_add_entity_returns_id(self):
        kg = BioKnowledgeGraph()
        result = kg.add_entity("TP53", "gene")
        assert result == "TP53"

    def test_add_entity_with_properties(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene", properties={"name": "tumor protein p53"})
        assert kg.summary()["nodes"] == 1

    def test_add_relationship_creates_edge(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        kg.add_entity("MDM2", "gene")
        kg.add_relationship("TP53", "MDM2", "INTERACTS_WITH")
        assert kg.summary()["edges"] == 1

    def test_add_relationship_returns_tuple(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        kg.add_entity("MDM2", "gene")
        result = kg.add_relationship("TP53", "MDM2", "INTERACTS_WITH")
        assert result == ("TP53", "MDM2", "INTERACTS_WITH")

    def test_summary_counts_node_types(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        kg.add_entity("olaparib", "drug")
        s = kg.summary()
        assert s["nodes"] == 2
        assert "gene" in s["node_types"]
        assert "drug" in s["node_types"]

    def test_summary_counts_relation_types(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        kg.add_entity("MDM2", "gene")
        kg.add_relationship("TP53", "MDM2", "INTERACTS_WITH")
        s = kg.summary()
        assert "INTERACTS_WITH" in s["relation_types"]

    def test_neighbors_returns_correct_neighbors(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        kg.add_entity("MDM2", "gene")
        kg.add_entity("BRCA1", "gene")
        kg.add_relationship("TP53", "MDM2", "INTERACTS_WITH")
        kg.add_relationship("TP53", "BRCA1", "INTERACTS_WITH")
        nbrs = kg.neighbors("TP53")
        assert "MDM2" in nbrs
        assert "BRCA1" in nbrs

    def test_neighbors_empty_for_isolated_node(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        assert kg.neighbors("TP53") == []

    def test_shortest_path_finds_path(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("A", "gene")
        kg.add_entity("B", "gene")
        kg.add_entity("C", "gene")
        kg.add_relationship("A", "B", "REL")
        kg.add_relationship("B", "C", "REL")
        path = kg.shortest_path("A", "C")
        assert len(path) == 3
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_shortest_path_empty_for_disconnected_nodes(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("A", "gene")
        kg.add_entity("Z", "gene")
        path = kg.shortest_path("A", "Z")
        assert path == []

    def test_shortest_path_missing_node_returns_empty(self):
        kg = BioKnowledgeGraph()
        path = kg.shortest_path("NONEXIST1", "NONEXIST2")
        assert path == []

    def test_query_entities_returns_scores(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        kg.add_entity("MDM2", "gene")
        kg.add_relationship("TP53", "MDM2", "INTERACTS_WITH")
        hits = kg.query_entities(["TP53"])
        assert "TP53" in hits
        assert isinstance(hits["TP53"], float)
        assert "MDM2" in hits

    def test_query_entities_empty_for_unknown(self):
        kg = BioKnowledgeGraph()
        hits = kg.query_entities(["NONEXIST"])
        assert hits == {}

    def test_query_subgraph_extracts_correct_subgraph(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("A", "gene")
        kg.add_entity("B", "gene")
        kg.add_entity("C", "gene")
        kg.add_entity("D", "gene")
        kg.add_relationship("A", "B", "REL")
        kg.add_relationship("B", "C", "REL")
        sub = kg.query_subgraph(["A"], hops=1)
        assert "A" in sub
        assert "B" in sub

    def test_export_html_creates_file(self, tmp_path):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        kg.add_entity("MDM2", "gene")
        kg.add_relationship("TP53", "MDM2", "INTERACTS_WITH")
        out = tmp_path / "test_graph.html"
        result = kg.export_html(out)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "<html" in content.lower()

    def test_save_and_load_round_trip(self, tmp_path):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        kg.add_entity("MDM2", "gene")
        kg.add_relationship("TP53", "MDM2", "INTERACTS_WITH")
        path = tmp_path / "kg_test.pkl"
        kg.save(path)
        assert path.exists()

        kg2 = BioKnowledgeGraph()
        kg2.load(path)
        assert kg2.summary()["nodes"] == 2
        assert kg2.summary()["edges"] == 1

    def test_from_checkpoint_nonexistent_returns_empty(self, tmp_path):
        kg = BioKnowledgeGraph.from_checkpoint(path=tmp_path / "no_such_file.pkl")
        assert kg.summary()["nodes"] == 0

    def test_from_checkpoint_no_args_returns_empty(self):
        kg = BioKnowledgeGraph.from_checkpoint()
        assert kg.summary()["nodes"] == 0

    def test_to_dict(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("TP53", "gene")
        d = kg.to_dict()
        assert isinstance(d, dict)

    def test_relation_evidence(self):
        kg = BioKnowledgeGraph()
        kg.add_entity("A", "gene")
        kg.add_entity("B", "gene")
        kg.add_relationship("A", "B", "REL")
        ev = kg.relation_evidence(["A"], hops=1)
        assert "seeds" in ev
        assert "nodes" in ev
        assert "edges" in ev
