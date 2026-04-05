"""Demo bundle helpers and compact biological seed data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .checkpoints import CheckpointStore, checkpoint_exists, load_json, save_json


DEMO_BUNDLE = {
    "gene_summaries": {
        "TP53": {
            "gene_id": "7157",
            "symbol": "TP53",
            "name": "tumor protein p53",
            "summary": "Master regulator of DNA damage response, apoptosis, and cell-cycle arrest.",
            "aliases": ["P53"],
        },
        "BRCA1": {
            "gene_id": "672",
            "symbol": "BRCA1",
            "name": "BRCA1 DNA repair associated",
            "summary": "Supports homologous recombination DNA repair and tumor suppression.",
            "aliases": [],
        },
        "EGFR": {
            "gene_id": "1956",
            "symbol": "EGFR",
            "name": "epidermal growth factor receptor",
            "summary": "Receptor tyrosine kinase driving proliferative signaling in many epithelial cancers.",
            "aliases": ["ERBB1"],
        },
        "IL6": {
            "gene_id": "3569",
            "symbol": "IL6",
            "name": "interleukin 6",
            "summary": "Cytokine involved in inflammatory and immune signaling.",
            "aliases": ["IL-6"],
        },
        "PARP1": {
            "gene_id": "142",
            "symbol": "PARP1",
            "name": "poly(ADP-ribose) polymerase 1",
            "summary": "DNA repair enzyme frequently targeted in homologous recombination-deficient tumors.",
            "aliases": [],
        },
    },
    "pubmed_records": [
        {
            "pmid": "1000001",
            "gene": "TP53",
            "title": "TP53 coordinates DNA damage response and apoptosis",
            "abstract": "TP53 activates DNA repair programs and apoptosis after genotoxic stress. BRCA1 and MDM2 are frequently discussed in TP53-centered cancer biology.",
        },
        {
            "pmid": "1000002",
            "gene": "BRCA1",
            "title": "BRCA1 participates in homologous recombination repair",
            "abstract": "BRCA1 supports homologous recombination, interacts with TP53-linked repair machinery, and is relevant to PARP inhibitor sensitivity.",
        },
        {
            "pmid": "1000003",
            "gene": "EGFR",
            "title": "EGFR signaling drives proliferation in epithelial tumors",
            "abstract": "EGFR activates downstream growth signaling and is targeted by approved kinase inhibitors such as erlotinib in selected cancers.",
        },
        {
            "pmid": "1000004",
            "gene": "IL6",
            "title": "IL6 and STAT3 mediate inflammatory signaling",
            "abstract": "IL6 promotes inflammatory signaling through STAT3 and is often discussed alongside immune regulation and cancer microenvironments.",
        },
        {
            "pmid": "1000005",
            "gene": "PARP1",
            "title": "PARP1 is a DNA repair enzyme targeted in cancer therapy",
            "abstract": "PARP1 supports single-strand break repair and is inhibited by olaparib in homologous recombination-deficient tumors.",
        },
    ],
    "string_ppi": {
        "TP53": [{"partner": "BRCA1", "score": 920}, {"partner": "MDM2", "score": 960}],
        "BRCA1": [{"partner": "TP53", "score": 920}, {"partner": "PARP1", "score": 880}],
        "EGFR": [{"partner": "AKT1", "score": 810}],
        "IL6": [{"partner": "STAT3", "score": 900}],
    },
    "drugbank": {
        "PARP1": [{"drug_name": "olaparib", "drugbank_id": "DB09074", "status": ["approved"], "mechanism": "PARP inhibitor"}],
        "EGFR": [{"drug_name": "erlotinib", "drugbank_id": "DB00530", "status": ["approved"], "mechanism": "EGFR tyrosine kinase inhibitor"}],
        "STAT3": [{"drug_name": "stattic", "drugbank_id": "DB12345", "status": ["investigational"], "mechanism": "STAT3 inhibitor"}],
    },
    "pathways": {
        "hsa03440": {
            "pathway_id": "hsa03440",
            "name": "Homologous recombination",
            "source": "KEGG",
            "description": "DNA repair pathway involving BRCA1 and PARP-associated repair context.",
        },
        "hsa04115": {
            "pathway_id": "hsa04115",
            "name": "p53 signaling pathway",
            "source": "KEGG",
            "description": "Stress-response signaling centered on TP53 and apoptosis/cell-cycle control.",
        },
        "hsa04012": {
            "pathway_id": "hsa04012",
            "name": "ERBB signaling pathway",
            "source": "KEGG",
            "description": "Growth signaling driven by EGFR family receptors.",
        },
        "hsa04060": {
            "pathway_id": "hsa04060",
            "name": "Cytokine-cytokine receptor interaction",
            "source": "KEGG",
            "description": "Immune and inflammatory signaling involving IL6.",
        },
    },
    "pathway_membership": {
        "TP53": ["hsa04115"],
        "BRCA1": ["hsa03440", "hsa04115"],
        "PARP1": ["hsa03440"],
        "EGFR": ["hsa04012"],
        "IL6": ["hsa04060"],
        "STAT3": ["hsa04060"],
    },
    "go_terms": {
        "GO:0006281": {
            "id": "GO:0006281",
            "name": "DNA repair",
            "namespace": "biological_process",
            "definition": "The process of restoring DNA after damage.",
        },
        "GO:0008283": {
            "id": "GO:0008283",
            "name": "cell proliferation",
            "namespace": "biological_process",
            "definition": "The multiplication or reproduction of cells.",
        },
        "GO:0006955": {
            "id": "GO:0006955",
            "name": "immune response",
            "namespace": "biological_process",
            "definition": "Any immune system process that functions in the response to a stimulus.",
        },
    },
    "gene_annotations": {
        "TP53": ["GO:0006281"],
        "BRCA1": ["GO:0006281"],
        "EGFR": ["GO:0008283"],
        "IL6": ["GO:0006955"],
        "PARP1": ["GO:0006281"],
    },
    "gene_synonyms": {
        "P53": "TP53",
        "IL-6": "IL6",
        "IL6": "IL6",
        "EGFR": "EGFR",
        "BRCA1": "BRCA1",
        "PARP1": "PARP1",
        "MDM2": "MDM2",
        "AKT1": "AKT1",
        "STAT3": "STAT3",
        "OLAPARIB": "olaparib",
    },
}


@dataclass(slots=True)
class DemoBundle:
    """Serializable bundle of demo data used by the CPU-safe agent."""

    gene_summaries: Dict[str, dict]
    pubmed_records: List[dict]
    string_ppi: Dict[str, list]
    drugbank: Dict[str, list]
    pathways: Dict[str, dict]
    pathway_membership: Dict[str, list]
    go_terms: Dict[str, dict]
    gene_annotations: Dict[str, list]
    gene_synonyms: Dict[str, str]

    def as_dict(self) -> dict:
        return {
            "gene_summaries": self.gene_summaries,
            "pubmed_records": self.pubmed_records,
            "string_ppi": self.string_ppi,
            "drugbank": self.drugbank,
            "pathways": self.pathways,
            "pathway_membership": self.pathway_membership,
            "go_terms": self.go_terms,
            "gene_annotations": self.gene_annotations,
            "gene_synonyms": self.gene_synonyms,
        }


def load_demo_bundle(checkpoint_dir: str | Path | None = None, force_rebuild: bool = False) -> DemoBundle:
    """Load the tiny bundled demo dataset."""

    if checkpoint_dir is None:
        payload = DEMO_BUNDLE
    else:
        store = CheckpointStore(checkpoint_dir)
        bundle_path = store.path("demo_bundle.json")
        if force_rebuild or not checkpoint_exists(bundle_path):
            payload = DEMO_BUNDLE
            save_json(payload, bundle_path)
        else:
            payload = load_json(bundle_path)

    payload = {**DEMO_BUNDLE, **payload}
    return DemoBundle(
        gene_summaries=dict(payload.get("gene_summaries", DEMO_BUNDLE["gene_summaries"])),
        pubmed_records=list(payload.get("pubmed_records", DEMO_BUNDLE["pubmed_records"])),
        string_ppi=dict(payload.get("string_ppi", DEMO_BUNDLE["string_ppi"])),
        drugbank=dict(payload.get("drugbank", DEMO_BUNDLE["drugbank"])),
        pathways=dict(payload.get("pathways", DEMO_BUNDLE["pathways"])),
        pathway_membership=dict(payload.get("pathway_membership", DEMO_BUNDLE["pathway_membership"])),
        go_terms=dict(payload.get("go_terms", DEMO_BUNDLE["go_terms"])),
        gene_annotations=dict(payload.get("gene_annotations", DEMO_BUNDLE["gene_annotations"])),
        gene_synonyms=dict(payload.get("gene_synonyms", DEMO_BUNDLE["gene_synonyms"])),
    )
