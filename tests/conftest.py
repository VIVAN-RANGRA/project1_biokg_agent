"""Shared fixtures for BioKG-Agent tests."""
# NOTE: pytest should be added to requirements-kaggle.txt for test runs.
import pytest
from biokg_agent.config import ProjectConfig
from biokg_agent.data import DemoBundle, DEMO_BUNDLE


@pytest.fixture
def config(tmp_path):
    """Config with temp checkpoint dir."""
    cfg = ProjectConfig()
    cfg.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.enable_live_apis = False
    return cfg


@pytest.fixture
def demo_bundle():
    """Minimal demo bundle for testing."""
    return DemoBundle(
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


@pytest.fixture
def agent(config, demo_bundle):
    """Pre-built agent for testing."""
    from biokg_agent.agent import BioKGAgent
    return BioKGAgent.build(config=config, bundle=demo_bundle, save_checkpoints=False)
