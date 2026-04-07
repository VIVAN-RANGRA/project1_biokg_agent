"""Tests for biokg_agent.data"""
import pytest
from biokg_agent.data import DEMO_BUNDLE, DemoBundle, load_demo_bundle


class TestDemoBundle:
    def test_demo_bundle_has_gene_summaries(self):
        assert "gene_summaries" in DEMO_BUNDLE

    def test_demo_bundle_has_pubmed_records(self):
        assert "pubmed_records" in DEMO_BUNDLE

    def test_demo_bundle_has_string_ppi(self):
        assert "string_ppi" in DEMO_BUNDLE

    def test_demo_bundle_has_drugbank(self):
        assert "drugbank" in DEMO_BUNDLE

    def test_demo_bundle_has_pathways(self):
        assert "pathways" in DEMO_BUNDLE

    def test_demo_bundle_has_pathway_membership(self):
        assert "pathway_membership" in DEMO_BUNDLE

    def test_demo_bundle_has_go_terms(self):
        assert "go_terms" in DEMO_BUNDLE

    def test_demo_bundle_has_gene_annotations(self):
        assert "gene_annotations" in DEMO_BUNDLE

    def test_demo_bundle_has_gene_synonyms(self):
        assert "gene_synonyms" in DEMO_BUNDLE

    def test_gene_summaries_contains_tp53(self):
        assert "TP53" in DEMO_BUNDLE["gene_summaries"]

    def test_gene_summaries_contains_brca1(self):
        assert "BRCA1" in DEMO_BUNDLE["gene_summaries"]

    def test_gene_summaries_contains_egfr(self):
        assert "EGFR" in DEMO_BUNDLE["gene_summaries"]

    def test_gene_summary_has_expected_fields(self):
        tp53 = DEMO_BUNDLE["gene_summaries"]["TP53"]
        assert "gene_id" in tp53
        assert "symbol" in tp53
        assert "summary" in tp53

    def test_pubmed_records_is_list(self):
        assert isinstance(DEMO_BUNDLE["pubmed_records"], list)

    def test_pubmed_records_have_pmid(self):
        for rec in DEMO_BUNDLE["pubmed_records"]:
            assert "pmid" in rec

    def test_pubmed_records_have_title(self):
        for rec in DEMO_BUNDLE["pubmed_records"]:
            assert "title" in rec

    def test_pubmed_records_have_abstract(self):
        for rec in DEMO_BUNDLE["pubmed_records"]:
            assert "abstract" in rec

    def test_string_ppi_structure(self):
        for gene, interactions in DEMO_BUNDLE["string_ppi"].items():
            assert isinstance(interactions, list)
            for row in interactions:
                assert "partner" in row
                assert "score" in row

    def test_dataclass_creation(self):
        bundle = DemoBundle(
            gene_summaries=DEMO_BUNDLE["gene_summaries"],
            pubmed_records=DEMO_BUNDLE["pubmed_records"],
            string_ppi=DEMO_BUNDLE["string_ppi"],
            drugbank=DEMO_BUNDLE["drugbank"],
            pathways=DEMO_BUNDLE["pathways"],
            pathway_membership=DEMO_BUNDLE["pathway_membership"],
            go_terms=DEMO_BUNDLE["go_terms"],
            gene_annotations=DEMO_BUNDLE["gene_annotations"],
            gene_synonyms=DEMO_BUNDLE["gene_synonyms"],
        )
        assert bundle.gene_summaries == DEMO_BUNDLE["gene_summaries"]

    def test_as_dict_round_trip(self):
        bundle = DemoBundle(**DEMO_BUNDLE)
        d = bundle.as_dict()
        assert d["gene_summaries"] == DEMO_BUNDLE["gene_summaries"]
        assert d["pubmed_records"] == DEMO_BUNDLE["pubmed_records"]
        assert d["string_ppi"] == DEMO_BUNDLE["string_ppi"]

    def test_load_demo_bundle_no_checkpoint(self):
        bundle = load_demo_bundle(checkpoint_dir=None)
        assert isinstance(bundle, DemoBundle)
        assert isinstance(bundle.gene_summaries, dict)
        assert isinstance(bundle.pubmed_records, list)

    def test_load_demo_bundle_with_checkpoint_dir(self, tmp_path):
        bundle = load_demo_bundle(checkpoint_dir=tmp_path)
        assert isinstance(bundle, DemoBundle)
        assert len(bundle.gene_summaries) > 0

    def test_load_demo_bundle_force_rebuild(self, tmp_path):
        load_demo_bundle(checkpoint_dir=tmp_path)
        bundle = load_demo_bundle(checkpoint_dir=tmp_path, force_rebuild=True)
        assert isinstance(bundle, DemoBundle)
