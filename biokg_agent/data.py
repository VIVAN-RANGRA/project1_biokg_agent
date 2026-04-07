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
        "GO:0006915": {
            "id": "GO:0006915",
            "name": "apoptotic process",
            "namespace": "biological_process",
            "definition": "A programmed cell death process initiated by a cell in response to intrinsic or extrinsic signals.",
        },
        "GO:0000077": {
            "id": "GO:0000077",
            "name": "DNA damage checkpoint",
            "namespace": "biological_process",
            "definition": "A cell cycle checkpoint that detects DNA damage and arrests the cell cycle.",
        },
        "GO:0045893": {
            "id": "GO:0045893",
            "name": "positive regulation of transcription",
            "namespace": "biological_process",
            "definition": "Any process that activates or increases the frequency of transcription.",
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
        "GO:0004713": {
            "id": "GO:0004713",
            "name": "protein tyrosine kinase activity",
            "namespace": "molecular_function",
            "definition": "Catalysis of ATP-dependent phosphorylation of tyrosine residues in proteins.",
        },
        "GO:0004871": {
            "id": "GO:0004871",
            "name": "signal transducer activity",
            "namespace": "molecular_function",
            "definition": "Transmitting a signal from one location to another.",
        },
        "GO:0043066": {
            "id": "GO:0043066",
            "name": "negative regulation of apoptotic process",
            "namespace": "biological_process",
            "definition": "Inhibition of programmed cell death.",
        },
    },
    "gene_annotations": {
        "TP53":  ["GO:0006915", "GO:0000077", "GO:0045893", "GO:0006281"],
        "BRCA1": ["GO:0006281", "GO:0006915"],
        "EGFR":  ["GO:0008283", "GO:0004713", "GO:0004871"],
        "IL6":   ["GO:0006955"],
        "PARP1": ["GO:0006281"],
        "BRAF":  ["GO:0008283", "GO:0004713"],
        "KRAS":  ["GO:0008283", "GO:0004871"],
        "BCL2":  ["GO:0043066", "GO:0006915"],
        "MDM2":  ["GO:0006915", "GO:0043066"],
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


# Curated approved drugs for top cancer genes.
# This supplements ChEMBL data which sometimes misses the most clinically
# important drugs due to binding-assay ordering in the API response.
CURATED_APPROVED_DRUGS: dict[str, list[dict]] = {
    "BRAF": [
        {"drug_name": "vemurafenib", "drugbank_id": "DB08881", "status": ["approved"], "mechanism": "BRAF V600E kinase inhibitor"},
        {"drug_name": "dabrafenib",  "drugbank_id": "DB09078", "status": ["approved"], "mechanism": "BRAF kinase inhibitor"},
        {"drug_name": "encorafenib", "drugbank_id": "DB11779", "status": ["approved"], "mechanism": "BRAF kinase inhibitor"},
        {"drug_name": "sorafenib",   "drugbank_id": "DB00398", "status": ["approved"], "mechanism": "multikinase inhibitor (BRAF/VEGFR/PDGFR)"},
    ],
    "EGFR": [
        {"drug_name": "osimertinib",  "drugbank_id": "DB09330", "status": ["approved"], "mechanism": "3rd-gen EGFR T790M inhibitor"},
        {"drug_name": "erlotinib",    "drugbank_id": "DB00530", "status": ["approved"], "mechanism": "EGFR TKI"},
        {"drug_name": "gefitinib",    "drugbank_id": "DB00317", "status": ["approved"], "mechanism": "EGFR TKI"},
        {"drug_name": "afatinib",     "drugbank_id": "DB08916", "status": ["approved"], "mechanism": "irreversible EGFR/HER2 inhibitor"},
        {"drug_name": "cetuximab",    "drugbank_id": "DB00002", "status": ["approved"], "mechanism": "anti-EGFR monoclonal antibody"},
    ],
    "ERBB2": [
        {"drug_name": "trastuzumab",  "drugbank_id": "DB00072", "status": ["approved"], "mechanism": "anti-HER2 monoclonal antibody"},
        {"drug_name": "pertuzumab",   "drugbank_id": "DB06366", "status": ["approved"], "mechanism": "anti-HER2 dimerisation inhibitor"},
        {"drug_name": "lapatinib",    "drugbank_id": "DB01259", "status": ["approved"], "mechanism": "dual EGFR/HER2 TKI"},
        {"drug_name": "neratinib",    "drugbank_id": "DB11878", "status": ["approved"], "mechanism": "irreversible HER2 TKI"},
    ],
    "ALK": [
        {"drug_name": "crizotinib",   "drugbank_id": "DB08865", "status": ["approved"], "mechanism": "ALK/MET/ROS1 inhibitor"},
        {"drug_name": "alectinib",    "drugbank_id": "DB11363", "status": ["approved"], "mechanism": "2nd-gen ALK inhibitor"},
        {"drug_name": "brigatinib",   "drugbank_id": "DB12267", "status": ["approved"], "mechanism": "ALK/EGFR inhibitor"},
        {"drug_name": "lorlatinib",   "drugbank_id": "DB13874", "status": ["approved"], "mechanism": "3rd-gen ALK/ROS1 inhibitor"},
    ],
    "ABL1": [
        {"drug_name": "imatinib",     "drugbank_id": "DB00619", "status": ["approved"], "mechanism": "BCR-ABL1/KIT/PDGFR inhibitor"},
        {"drug_name": "dasatinib",    "drugbank_id": "DB01254", "status": ["approved"], "mechanism": "BCR-ABL1/SRC inhibitor"},
        {"drug_name": "nilotinib",    "drugbank_id": "DB04868", "status": ["approved"], "mechanism": "BCR-ABL1 inhibitor (2nd-gen)"},
        {"drug_name": "ponatinib",    "drugbank_id": "DB08901", "status": ["approved"], "mechanism": "BCR-ABL1 T315I inhibitor"},
    ],
    "PARP1": [
        {"drug_name": "olaparib",     "drugbank_id": "DB09074", "status": ["approved"], "mechanism": "PARP1/2 inhibitor"},
        {"drug_name": "niraparib",    "drugbank_id": "DB12143", "status": ["approved"], "mechanism": "PARP1/2 inhibitor"},
        {"drug_name": "rucaparib",    "drugbank_id": "DB12332", "status": ["approved"], "mechanism": "PARP1/2/3 inhibitor"},
        {"drug_name": "talazoparib",  "drugbank_id": "DB12165", "status": ["approved"], "mechanism": "PARP1/2 trapping inhibitor"},
    ],
    "BCL2": [
        {"drug_name": "venetoclax",   "drugbank_id": "DB11581", "status": ["approved"], "mechanism": "BCL-2 selective inhibitor"},
    ],
    "MTOR": [
        {"drug_name": "everolimus",   "drugbank_id": "DB01590", "status": ["approved"], "mechanism": "mTORC1 inhibitor (rapamycin analogue)"},
        {"drug_name": "temsirolimus", "drugbank_id": "DB06287", "status": ["approved"], "mechanism": "mTOR inhibitor"},
    ],
    "CDK4": [
        {"drug_name": "palbociclib",  "drugbank_id": "DB09073", "status": ["approved"], "mechanism": "CDK4/6 inhibitor"},
        {"drug_name": "ribociclib",   "drugbank_id": "DB11730", "status": ["approved"], "mechanism": "CDK4/6 inhibitor"},
        {"drug_name": "abemaciclib",  "drugbank_id": "DB12001", "status": ["approved"], "mechanism": "CDK4/6 inhibitor"},
    ],
    "JAK2": [
        {"drug_name": "ruxolitinib",  "drugbank_id": "DB08877", "status": ["approved"], "mechanism": "JAK1/2 inhibitor"},
        {"drug_name": "fedratinib",   "drugbank_id": "DB14568", "status": ["approved"], "mechanism": "JAK2 inhibitor"},
    ],
    "FLT3": [
        {"drug_name": "midostaurin",  "drugbank_id": "DB06595", "status": ["approved"], "mechanism": "FLT3/PKC inhibitor"},
        {"drug_name": "gilteritinib", "drugbank_id": "DB14963", "status": ["approved"], "mechanism": "FLT3/AXL inhibitor"},
    ],
    "IDH1": [
        {"drug_name": "ivosidenib",   "drugbank_id": "DB14568", "status": ["approved"], "mechanism": "IDH1 R132 mutant inhibitor"},
    ],
    "IDH2": [
        {"drug_name": "enasidenib",   "drugbank_id": "DB13874", "status": ["approved"], "mechanism": "IDH2 R140/R172 mutant inhibitor"},
    ],
    "KIT": [
        {"drug_name": "imatinib",     "drugbank_id": "DB00619", "status": ["approved"], "mechanism": "KIT/BCR-ABL1/PDGFR inhibitor"},
        {"drug_name": "sunitinib",    "drugbank_id": "DB01268", "status": ["approved"], "mechanism": "KIT/VEGFR/PDGFR/FLT3 inhibitor"},
        {"drug_name": "ripretinib",   "drugbank_id": "DB15685", "status": ["approved"], "mechanism": "KIT/PDGFRA switch control inhibitor"},
    ],
    "PDGFRA": [
        {"drug_name": "imatinib",     "drugbank_id": "DB00619", "status": ["approved"], "mechanism": "PDGFRA/BCR-ABL1/KIT inhibitor"},
        {"drug_name": "avapritinib",  "drugbank_id": "DB15695", "status": ["approved"], "mechanism": "PDGFRA D842V inhibitor"},
    ],
    "MET": [
        {"drug_name": "capmatinib",   "drugbank_id": "DB12978", "status": ["approved"], "mechanism": "MET exon 14 inhibitor"},
        {"drug_name": "tepotinib",    "drugbank_id": "DB15393", "status": ["approved"], "mechanism": "MET inhibitor"},
        {"drug_name": "crizotinib",   "drugbank_id": "DB08865", "status": ["approved"], "mechanism": "ALK/MET/ROS1 inhibitor"},
    ],
    "RET": [
        {"drug_name": "selpercatinib","drugbank_id": "DB15685", "status": ["approved"], "mechanism": "RET kinase inhibitor"},
        {"drug_name": "pralsetinib",  "drugbank_id": "DB15823", "status": ["approved"], "mechanism": "RET inhibitor"},
        {"drug_name": "vandetanib",   "drugbank_id": "DB08764", "status": ["approved"], "mechanism": "RET/VEGFR/EGFR inhibitor"},
    ],
    "ROS1": [
        {"drug_name": "crizotinib",   "drugbank_id": "DB08865", "status": ["approved"], "mechanism": "ROS1/ALK/MET inhibitor"},
        {"drug_name": "entrectinib",  "drugbank_id": "DB14690", "status": ["approved"], "mechanism": "ROS1/NTRK/ALK inhibitor"},
    ],
    "VEGFA": [
        {"drug_name": "bevacizumab",  "drugbank_id": "DB00112", "status": ["approved"], "mechanism": "anti-VEGF-A monoclonal antibody"},
        {"drug_name": "sunitinib",    "drugbank_id": "DB01268", "status": ["approved"], "mechanism": "VEGFR/KIT/PDGFR inhibitor"},
    ],
    "TP53": [
        # No FDA-approved drugs DIRECTLY target TP53 — it's a tumor suppressor
        # Closest clinical compounds work by reactivating mutant p53 or blocking MDM2
        {"drug_name": "eprenetapopt (APR-246)", "drugbank_id": "DB12964", "status": ["phase 3"], "mechanism": "restores wild-type folding to mutant p53 — Phase 3 MDS/AML"},
        {"drug_name": "idasanutlin",            "drugbank_id": "DB12629", "status": ["phase 3"], "mechanism": "MDM2 inhibitor — prevents p53 degradation — Phase 3"},
        {"drug_name": "navtemadlin (AMG-232)",   "drugbank_id": "DB15441", "status": ["phase 2"], "mechanism": "MDM2 inhibitor — stabilizes WT p53"},
        {"drug_name": "milademetan",             "drugbank_id": "DB16111", "status": ["phase 2"], "mechanism": "MDM2 inhibitor — activates p53 pathway"},
    ],
    "MDM2": [
        {"drug_name": "idasanutlin",             "drugbank_id": "DB12629", "status": ["phase 3"], "mechanism": "MDM2 antagonist — blocks p53-MDM2 interaction"},
        {"drug_name": "navtemadlin (AMG-232)",    "drugbank_id": "DB15441", "status": ["phase 2"], "mechanism": "MDM2 inhibitor — picomolar potency"},
        {"drug_name": "milademetan",              "drugbank_id": "DB16111", "status": ["phase 2"], "mechanism": "MDM2/MDMX dual inhibitor"},
    ],
    "KRAS": [
        {"drug_name": "sotorasib",    "drugbank_id": "DB15858", "status": ["approved"], "mechanism": "KRAS G12C covalent inhibitor"},
        {"drug_name": "adagrasib",    "drugbank_id": "DB16049", "status": ["approved"], "mechanism": "KRAS G12C covalent inhibitor"},
    ],
    "BRCA1": [
        {"drug_name": "olaparib",     "drugbank_id": "DB09074", "status": ["approved"], "mechanism": "PARP inhibitor (exploits HRD)"},
        {"drug_name": "talazoparib",  "drugbank_id": "DB12165", "status": ["approved"], "mechanism": "PARP inhibitor (exploits HRD)"},
    ],
    "FGFR1": [
        {"drug_name": "erdafitinib",  "drugbank_id": "DB14973", "status": ["approved"], "mechanism": "pan-FGFR inhibitor"},
        {"drug_name": "pemigatinib",  "drugbank_id": "DB15685", "status": ["approved"], "mechanism": "FGFR1/2/3 inhibitor"},
    ],
    "FGFR2": [
        {"drug_name": "pemigatinib",  "drugbank_id": "DB15685", "status": ["approved"], "mechanism": "FGFR1/2/3 inhibitor"},
        {"drug_name": "infigratinib", "drugbank_id": "DB14928", "status": ["approved"], "mechanism": "pan-FGFR inhibitor"},
    ],
    "NRAS": [
        {"drug_name": "binimetinib",  "drugbank_id": "DB11528", "status": ["approved"], "mechanism": "MEK1/2 inhibitor for NRAS-mutant melanoma"},
    ],
    "MAP2K1": [
        {"drug_name": "trametinib",   "drugbank_id": "DB08911", "status": ["approved"], "mechanism": "MEK1/2 inhibitor"},
        {"drug_name": "cobimetinib",  "drugbank_id": "DB11610", "status": ["approved"], "mechanism": "MEK1 inhibitor"},
        {"drug_name": "binimetinib",  "drugbank_id": "DB11528", "status": ["approved"], "mechanism": "MEK1/2 inhibitor"},
    ],
    "PIK3CA": [
        {"drug_name": "alpelisib",    "drugbank_id": "DB12015", "status": ["approved"], "mechanism": "PI3Kα-selective inhibitor"},
    ],
    "NOTCH1": [
        {"drug_name": "nirogacestat", "drugbank_id": "DB15822", "status": ["approved"], "mechanism": "gamma-secretase inhibitor (NOTCH)"},
    ],
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

    # ------------------------------------------------------------------
    # Task 1: Merge CURATED_APPROVED_DRUGS into drugbank.
    # For each gene, prepend curated drugs before any existing ChEMBL
    # drugs, deduplicating by drug_name (curated wins on conflict).
    # ------------------------------------------------------------------
    drugbank = dict(payload.get("drugbank", DEMO_BUNDLE["drugbank"]))
    for gene, curated_drugs in CURATED_APPROVED_DRUGS.items():
        existing = drugbank.get(gene, [])
        # Collect drug_names already covered by curated list
        curated_names = {d["drug_name"] for d in curated_drugs}
        # Keep only existing entries whose name is NOT in the curated set
        deduped_existing = [d for d in existing if d["drug_name"] not in curated_names]
        drugbank[gene] = list(curated_drugs) + deduped_existing
    payload["drugbank"] = drugbank

    # ------------------------------------------------------------------
    # Task 2: Generate pathway_membership and pathways from GO
    # biological_process terms in gene_annotations / go_terms.
    # ------------------------------------------------------------------
    go_terms = dict(payload.get("go_terms", DEMO_BUNDLE["go_terms"]))
    gene_annotations = dict(payload.get("gene_annotations", DEMO_BUNDLE["gene_annotations"]))
    pathways = dict(payload.get("pathways", DEMO_BUNDLE["pathways"]))
    pathway_membership = dict(payload.get("pathway_membership", DEMO_BUNDLE["pathway_membership"]))

    for gene, go_ids in gene_annotations.items():
        bp_ids: list[str] = []
        for go_id in go_ids:
            term = go_terms.get(go_id)
            if term and term.get("namespace") == "biological_process":
                bp_ids.append(go_id)
                # Ensure a corresponding pathway entry exists
                if go_id not in pathways:
                    pathways[go_id] = {
                        "pathway_id": go_id,
                        "name": term.get("name", go_id),
                        "source": "GO:biological_process",
                        "description": term.get("definition", ""),
                    }
        if bp_ids:
            # Merge with any pre-existing pathway membership for this gene
            existing_ids = set(pathway_membership.get(gene, []))
            merged = list(pathway_membership.get(gene, [])) + [
                gid for gid in bp_ids if gid not in existing_ids
            ]
            pathway_membership[gene] = merged

    payload["pathways"] = pathways
    payload["pathway_membership"] = pathway_membership

    # ------------------------------------------------------------------
    # Task 3: Add cancer gene symbols from CURATED_APPROVED_DRUGS as
    # self-referencing synonyms in gene_synonyms.
    # ------------------------------------------------------------------
    gene_synonyms = dict(payload.get("gene_synonyms", DEMO_BUNDLE["gene_synonyms"]))
    for gene in CURATED_APPROVED_DRUGS:
        if gene not in gene_synonyms:
            gene_synonyms[gene] = gene
    payload["gene_synonyms"] = gene_synonyms

    return DemoBundle(
        gene_summaries=dict(payload.get("gene_summaries", DEMO_BUNDLE["gene_summaries"])),
        pubmed_records=list(payload.get("pubmed_records", DEMO_BUNDLE["pubmed_records"])),
        string_ppi=dict(payload.get("string_ppi", DEMO_BUNDLE["string_ppi"])),
        drugbank=payload["drugbank"],
        pathways=payload["pathways"],
        pathway_membership=payload["pathway_membership"],
        go_terms=dict(payload.get("go_terms", DEMO_BUNDLE["go_terms"])),
        gene_annotations=dict(payload.get("gene_annotations", DEMO_BUNDLE["gene_annotations"])),
        gene_synonyms=payload["gene_synonyms"],
    )
