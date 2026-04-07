"""Advanced retrieval and synthesis agent for BioKG-Agent."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

try:
    from .tools.uniprot import uniprot_protein_lookup
    from .tools.kegg import kegg_pathway_lookup
    from .tools.drugbank import drugbank_target_lookup as drugbank_api_lookup
except Exception:
    uniprot_protein_lookup = None  # type: ignore[assignment,misc]
    kegg_pathway_lookup = None  # type: ignore[assignment,misc]
    drugbank_api_lookup = None  # type: ignore[assignment,misc]

from .checkpoints import CheckpointStore
from .config import ProjectConfig, default_config
from .data import DemoBundle, load_demo_bundle
from .kg import BioKnowledgeGraph
from .retrieval import HybridRetrievalEngine, RetrievalBundle
from .router import EvidenceAssessment, QueryPlan, QueryRouter


def _normalize_gene(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


QUERY_ENTITY_HINTS: dict[str, list[str]] = {
    # Disease-centric cues
    "fanconi anemia": ["BRCA1", "BRCA2", "PALB2", "BRIP1", "RAD51C", "FANCA"],
    "melanoma and colorectal": ["BRAF", "KRAS", "NRAS", "PIK3CA", "APC", "TP53"],
    "glioblastoma": ["TP53", "PTEN", "RB1", "CDKN2A", "NF1"],
    "li-fraumeni": ["TP53", "CHEK2", "MDM2", "CDK4", "RB1"],
    "diffuse gastric cancer": ["CDH1", "CTNNA1", "TP53", "RHOA", "CTNNB1"],
    "myeloproliferative neoplasm": ["JAK2", "CALR", "MPL", "STAT5A", "STAT3"],
    "familial adenomatous polyposis": ["APC", "CTNNB1", "AXIN1", "GSK3B", "TCF7L2", "MYC"],
    "clear cell renal": ["VHL", "PBRM1", "BAP1", "SETD2", "HIF1A", "HIF2A"],
    "lynch syndrome": ["MLH1", "MSH2", "MSH6", "PMS2", "PDCD1", "CD274"],
    "her2-positive breast cancer": ["ERBB2", "GRB7", "TP53", "PIK3CA", "MYC", "TOP2A"],
    "castration-resistant prostate": ["AR", "PTEN", "TP53", "RB1", "BRCA2", "ERG"],
    "pancreatic ductal adenocarcinoma": ["KRAS", "TP53", "CDKN2A", "SMAD4", "BRCA2"],
    "acute myeloid leukemia": ["FLT3", "NPM1", "DNMT3A", "IDH1", "IDH2", "CEBPA"],
    "multiple myeloma": ["MYC", "CCND1", "FGFR3", "NFKB1", "TRAF3", "KRAS"],
    "mesothelioma": ["NF2", "BAP1", "CDKN2A", "LATS1", "LATS2", "YAP1"],
    "hereditary retinoblastoma": ["RB1", "CCND1", "CDK4", "CDKN2A", "E2F1"],
    "tuberous sclerosis": ["TSC1", "TSC2", "MTOR", "RHEB", "RPTOR"],
    "bladder cancer": ["FGFR3", "PIK3CA", "TP53", "RB1", "CDKN2A", "ERBB2"],
    "neurofibromatosis type 1": ["NF1", "KRAS", "NRAS", "MAPK1", "SPRED1"],
    "myelodysplastic syndromes": ["SF3B1", "SRSF2", "U2AF1", "ZRSR2", "TP53"],
    # Pathway/mechanism cues
    "p53 signaling pathway": ["TP53", "BAX", "BBC3", "FAS", "CASP9", "CASP8", "APAF1"],
    "wnt/beta-catenin": ["CTNNB1", "APC", "YAP1", "LATS1", "LATS2", "TAZ"],
    "pi3k-akt": ["PIK3CA", "AKT1", "BAD", "NFKB1", "BCL2", "CHUK"],
    "oncogenic kras": ["KRAS", "RAF1", "BRAF", "PIK3CA", "MAPK1", "AKT1"],
    "mtorc1": ["MTOR", "RPTOR", "RHEB", "ULK1", "TSC2", "AMPK"],
    "notch signaling pathway": ["NOTCH1", "RBPJ", "HES1", "MYC", "DTX1", "MAML1"],
    "mapk/erk": ["MAPK1", "MAPK3", "DUSP1", "DUSP6", "SPRY2", "RAF1"],
    "tgf-beta": ["TGFB1", "SMAD2", "SMAD3", "SMAD4", "SMAD7", "CDKN1A"],
    "jak-stat": ["JAK1", "JAK2", "STAT3", "STAT5A", "SOCS1", "SOCS3"],
    "hif-1alpha": ["HIF1A", "VHL", "VEGFA", "GLUT1", "LDHA", "PHD2"],
    "atm-chek2": ["ATM", "ATR", "CHEK1", "CHEK2", "CDC25A", "CDC25C", "TP53"],
    "nf-kb signaling": ["NFKB1", "RELA", "IKBKG", "MYD88", "CARD11", "BCL10"],
    "ferroptosis": ["GPX4", "SLC7A11", "ACSL4", "TFRC", "NFE2L2"],
    "ubiquitin-proteasome": ["CCNE1", "CCND1", "CDKN1A", "SKP2", "FBXW7", "APC/C"],
    "cgas-sting": ["CGAS", "TMEM173", "TBK1", "IRF3", "IFNB1"],
    "hedgehog pathway": ["SHH", "PTCH1", "SMO", "GLI1", "GLI2", "SUFU"],
    "polycomb": ["EZH2", "SUZ12", "EED", "BMI1", "RING1B", "CDKN2A"],
    "unfolded protein response": ["ERN1", "EIF2AK3", "ATF6", "XBP1", "DDIT3", "BCL2"],
    "yap/taz": ["MST1", "MST2", "LATS1", "LATS2", "YAP1", "WWTR1", "TEAD1"],
    # Common disease / condition queries (non-cancer)
    "ulcer": ["PTGS2", "PTGS1", "EGFR", "TGFA", "MUC5AC", "IL1B", "TNF", "NFKB1"],
    "peptic ulcer": ["PTGS2", "PTGS1", "IL1B", "TNF", "EGFR", "MUC5AC"],
    "gastric ulcer": ["PTGS2", "PTGS1", "EGFR", "TGFA", "MUC5AC", "IL1B"],
    "helicobacter pylori": ["PTGS2", "IL1B", "TNF", "NFKB1", "TP53", "CDH1"],
    "inflammation": ["PTGS2", "TNF", "IL6", "IL1B", "NFKB1", "STAT3", "RELA"],
    "diabetes": ["INS", "IGF1R", "IRS1", "PIK3CA", "AKT1", "FOXO1", "PPARG"],
    "type 2 diabetes": ["PPARG", "INS", "IRS1", "AKT1", "FOXO1", "AMPK", "GCGR"],
    "alzheimer": ["APP", "PSEN1", "PSEN2", "APOE", "BACE1", "MAPT", "CLU"],
    "parkinson": ["SNCA", "LRRK2", "PARK2", "PINK1", "DJ1", "GBA"],
    "hypertension": ["ACE", "AGT", "AGTR1", "NOS3", "EDN1", "NPPA", "ADRB1"],
    "heart failure": ["NPPA", "NPPB", "ACE", "ADRB1", "ADRB2", "PLN", "SERCA2A"],
    "asthma": ["IL4", "IL13", "IL5", "TSLP", "GATA3", "IFNG", "STAT6"],
    "rheumatoid arthritis": ["TNF", "IL6", "IL1B", "STAT3", "JAK1", "JAK2", "NFKB1"],
    "breast cancer": ["BRCA1", "BRCA2", "ERBB2", "ESR1", "PIK3CA", "TP53", "MYC"],
    "lung cancer": ["EGFR", "KRAS", "ALK", "ROS1", "MET", "TP53", "STK11"],
    "colorectal cancer": ["APC", "KRAS", "TP53", "SMAD4", "PIK3CA", "BRAF", "MLH1"],
    "prostate cancer": ["AR", "PTEN", "TP53", "RB1", "BRCA2", "ERG", "NKX3-1"],
    "leukemia": ["BCR", "ABL1", "FLT3", "JAK2", "NPM1", "CEBPA", "TP53"],
    "lymphoma": ["MYC", "BCL2", "BCL6", "TP53", "CD19", "CREBBP", "EP300"],
    "apoptosis": ["TP53", "BCL2", "BAX", "CASP3", "CASP9", "APAF1", "CYCS"],
    "cell cycle": ["CCND1", "CDK4", "CDK6", "RB1", "CDKN1A", "CDKN2A", "MYC"],
    "dna repair": ["BRCA1", "BRCA2", "ATM", "RAD51", "PARP1", "MLH1", "MSH2"],
    "angiogenesis": ["VEGFA", "KDR", "FLT1", "HIF1A", "ANGPT1", "PDGFB", "NRP1"],
    "metastasis": ["MMP9", "MMP2", "CDH1", "TWIST1", "SNAI1", "VIM", "CTNNB1"],
    "immunotherapy": ["PDCD1", "CD274", "CTLA4", "CD8A", "IFNG", "TNF", "LAG3"],
}


@dataclass(slots=True)
class RetrievalIteration:
    """One explicit retrieval round with decision traces."""

    iteration: int
    query: str
    plan: dict[str, Any]
    retrieval: dict[str, Any]
    assessment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentResult:
    """Structured agent output for demos and smoke evaluation."""

    answer_text: str
    query_plan: dict[str, Any]
    retrieval_iterations: list[dict[str, Any]]
    evidence_table: list[dict[str, Any]]
    graph_summary: dict[str, Any]
    confidence_summary: dict[str, Any]
    checkpoint_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.answer_text


@dataclass
class BioKGAgent:
    """Advanced RAG orchestration with routing, hybrid retrieval, and confidence traces."""

    config: ProjectConfig = field(default_factory=default_config)
    retriever: HybridRetrievalEngine | None = None
    router: QueryRouter | None = None
    kg: BioKnowledgeGraph | None = None
    knowledge_graph: BioKnowledgeGraph | None = None
    checkpoint_dir: str | None = None
    save_checkpoints: bool = True
    demo_mode: bool = True
    bundle: DemoBundle | None = None
    store: CheckpointStore | None = None
    planner: Callable[[str, list[str]], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_dir:
            self.config.checkpoint_dir = self.checkpoint_dir
        self.store = self.store or CheckpointStore(self.config.checkpoint_dir_path)
        self.bundle = self.bundle or load_demo_bundle(self.config.checkpoint_dir_path)
        if self.retriever is None:
            # Try to load from saved checkpoint (avoids re-embedding on every startup)
            _retriever_path = self.config.retriever_checkpoint_path
            if _retriever_path.exists():
                try:
                    self.retriever = HybridRetrievalEngine.load(_retriever_path)
                    print(f"[agent] Loaded retrieval engine from checkpoint (backend: {self.retriever.dense_backend})")
                    # Upgrade hashed backend → FAISS if pre-built index is available
                    if self.retriever.dense_backend == "hashed":
                        _faiss_path = self.config.checkpoint_dir_path / "faiss_index.bin"
                        if _faiss_path.exists():
                            try:
                                import faiss as _faiss
                                import numpy as _np
                                _loaded = _faiss.read_index(str(_faiss_path))
                                if _loaded.ntotal == len(self.retriever.records):
                                    self.retriever.faiss_index = _loaded
                                    self.retriever.dense_backend = "faiss"
                                    self.retriever.dense_dim = _loaded.d
                                    self.retriever._get_sentence_model()
                                    print(f"[agent] Upgraded retriever to FAISS ({_loaded.ntotal:,} vectors, dim={_loaded.d})")
                            except Exception as _fe:
                                print(f"[agent] FAISS upgrade failed: {_fe}")
                except Exception:
                    self.retriever = None
        if self.retriever is None:
            self.retriever = HybridRetrievalEngine.from_records(
                self.bundle.pubmed_records,
                config=self.config,
            )
        self.router = self.router or QueryRouter(config=self.config, planner=self.planner)
        self.kg = self.kg or self.knowledge_graph or BioKnowledgeGraph.from_checkpoint(
            checkpoint_dir=self.config.checkpoint_dir_path,
            graph_checkpoint_name=self.config.graph_checkpoint_name,
        )
        self._seed_graph()
        self._known_gene_set = set(self.known_genes())
        if self.config.enable_llm_synthesis and self.config.groq_api_key:
            try:
                from .llm import create_llm_backend
                self._llm = create_llm_backend(
                    groq_api_key=self.config.groq_api_key,
                    groq_model=self.config.groq_model,
                )
                self._llm.load()
                print(f"[agent] LLM backend: {self._llm}")
            except Exception as e:
                print(f"[agent] LLM init failed: {e}")
                self._llm = None
        else:
            self._llm = None
        if self.save_checkpoints:
            self.save()

    def _seed_graph(self) -> None:
        for gene, summary in self.bundle.gene_summaries.items():
            gene_id = _normalize_gene(gene)
            self.kg.add_entity(gene_id, "gene", properties={"label": gene_id, **summary})
        for pathway_id, pathway in self.bundle.pathways.items():
            self.kg.add_entity(
                pathway_id,
                "pathway",
                properties={"label": pathway.get("name", pathway_id), **pathway},
            )

    def known_genes(self) -> list[str]:
        genes = set(_normalize_gene(key) for key in self.bundle.gene_summaries)
        genes.update(_normalize_gene(key) for key in self.bundle.string_ppi.keys())
        genes.update(_normalize_gene(key) for key in self.bundle.drugbank.keys())
        genes.update(_normalize_gene(value) for value in self.bundle.gene_synonyms.values())
        genes.update(_normalize_gene(key) for key in self.bundle.gene_annotations.keys())
        return sorted(gene for gene in genes if gene)

    def ncbi_gene_lookup(self, gene: str) -> dict[str, Any]:
        gene = _normalize_gene(gene)
        if self.config.enable_live_apis and requests is not None:
            try:  # pragma: no cover - network dependent
                params = {
                    "db": "gene",
                    "retmode": "json",
                    "term": f"{gene}[Gene Name] AND Homo sapiens[Organism]",
                }
                if self.config.ncbi_api_key:
                    params["api_key"] = self.config.ncbi_api_key
                search_resp = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params=params,
                    timeout=15,
                )
                search_resp.raise_for_status()
                gene_ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
                if gene_ids:
                    summary_resp = requests.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                        params={
                            "db": "gene",
                            "retmode": "json",
                            "id": gene_ids[0],
                            **({"api_key": self.config.ncbi_api_key} if self.config.ncbi_api_key else {}),
                        },
                        timeout=15,
                    )
                    summary_resp.raise_for_status()
                    result = summary_resp.json().get("result", {})
                    doc = result.get(gene_ids[0], {})
                    if doc:
                        return {
                            "gene_id": doc.get("uid", gene_ids[0]),
                            "symbol": doc.get("name", gene),
                            "name": doc.get("description", ""),
                            "summary": doc.get("summary", ""),
                            "aliases": str(doc.get("otheraliases", "")).split(", ") if doc.get("otheraliases") else [],
                            "source": "ncbi_live",
                        }
            except Exception:
                pass
        payload = dict(
            self.bundle.gene_summaries.get(
                gene,
                {"symbol": gene, "summary": "No local summary available.", "aliases": []},
            )
        )
        payload["source"] = payload.get("source", "bundle")
        return payload

    def string_interactions(self, gene: str, score_threshold: int | None = None) -> list[dict[str, Any]]:
        gene = _normalize_gene(gene)
        score_threshold = score_threshold or self.config.string_score_threshold
        if self.config.enable_live_apis and requests is not None:
            try:  # pragma: no cover - network dependent
                response = requests.get(
                    "https://string-db.org/api/json/network",
                    params={
                        "identifiers": gene,
                        "species": 9606,
                        "required_score": score_threshold,
                    },
                    timeout=15,
                )
                response.raise_for_status()
                interactions = []
                for row in response.json():
                    partner = row.get("preferredName_B") if row.get("preferredName_A") == gene else row.get("preferredName_A")
                    if not partner:
                        continue
                    score = row.get("score", 0)
                    if isinstance(score, float):
                        score = int(score * 1000)
                    interactions.append(
                        {
                            "partner": _normalize_gene(str(partner)),
                            "score": int(score),
                            "source": "string_live",
                        }
                    )
                if interactions:
                    return interactions
            except Exception:
                pass
        return [
            {**dict(row), "source": row.get("source", "bundle")}
            for row in self.bundle.string_ppi.get(gene, [])
            if int(row.get("score", 0)) >= score_threshold
        ]

    def drugbank_target_lookup(self, gene: str) -> list[dict[str, Any]]:
        gene = _normalize_gene(gene)
        # Start with curated known approved drugs (highest quality data)
        from .data import CURATED_APPROVED_DRUGS
        curated = [
            {**d, "source": "curated"}
            for d in CURATED_APPROVED_DRUGS.get(gene, [])
        ]
        curated_names = {d["drug_name"].lower() for d in curated}
        # Supplement with ChEMBL/bundle data (deduplicated)
        bundle_drugs = [
            {**dict(row), "source": row.get("source", "drugbank_bundle")}
            for row in self.bundle.drugbank.get(gene, [])
            if str(row.get("drug_name", "")).lower() not in curated_names
        ]
        return curated + bundle_drugs

    def gene_ontology_lookup(self, gene: str) -> list[dict[str, Any]]:
        gene = _normalize_gene(gene)
        return [
            {**dict(self.bundle.go_terms.get(go_id, {"id": go_id, "name": go_id})), "source": "go_bundle"}
            for go_id in self.bundle.gene_annotations.get(gene, [])
        ]

    def pathway_lookup(self, gene: str) -> list[dict[str, Any]]:
        gene = _normalize_gene(gene)
        pw_ids = self.bundle.pathway_membership.get(gene, [])
        if pw_ids:
            return [
                {**dict(self.bundle.pathways.get(pid, {"pathway_id": pid, "name": pid})), "source": "pathway_bundle"}
                for pid in pw_ids
            ]
        # Fall back: use GO biological_process terms as pathway proxies
        go_ids = self.bundle.gene_annotations.get(gene, [])
        results = []
        for gid in go_ids:
            term = self.bundle.go_terms.get(gid, {})
            if term.get("namespace") == "biological_process":
                results.append({
                    "pathway_id": gid,
                    "name": term.get("name", gid),
                    "source": "go_biological_process",
                })
        return results

    def uniprot_protein_lookup(self, protein: str) -> dict:
        protein = _normalize_gene(protein)
        if self.config.enable_live_apis:
            try:
                return uniprot_protein_lookup(protein)
            except Exception:
                pass
        return {"accession": "", "name": protein, "function": "No data available", "source": "fallback"}

    def kegg_pathway_lookup(self, gene: str) -> list[dict]:
        gene = _normalize_gene(gene)
        if self.config.enable_live_apis:
            try:
                return kegg_pathway_lookup(gene)
            except Exception:
                pass
        return self.pathway_lookup(gene)  # fall back to bundle

    def plan_query(self, query: str) -> QueryPlan:
        plan = self.router.plan(query, self.known_genes())
        query_upper = query.upper()
        detected_entities = list(plan.detected_entities)
        for alias, canonical in self.bundle.gene_synonyms.items():
            if alias.upper() in query_upper:
                normalized = _normalize_gene(canonical)
                if normalized and normalized not in detected_entities:
                    detected_entities.append(normalized)
        for hinted in self._query_entity_hints(query, known_only=True):
            normalized = _normalize_gene(hinted)
            if normalized and normalized not in detected_entities:
                detected_entities.append(normalized)
        plan.detected_entities = detected_entities
        if detected_entities:
            plan.metadata_filters = {**plan.metadata_filters, "genes": detected_entities}
        if self.save_checkpoints:
            self.store.save_json(plan.to_dict(), self.config.query_plan_name)
        return plan

    def _query_entity_hints(self, query: str, known_only: bool = True) -> list[str]:
        lowered = query.lower()
        hints: list[str] = []

        for phrase, entities in QUERY_ENTITY_HINTS.items():
            if phrase in lowered:
                hints.extend(entities)

        # Generic biomedical cues that often imply canonical pathway actors.
        if "jak" in lowered and "stat" in lowered:
            hints.extend(["JAK1", "JAK2", "STAT3", "STAT5A", "SOCS1", "SOCS3"])
        if "hippo" in lowered:
            hints.extend(["YAP1", "WWTR1", "LATS1", "LATS2", "MST1", "MST2", "TEAD1"])
        if "hypoxia" in lowered:
            hints.extend(["HIF1A", "VHL", "VEGFA", "GLUT1", "LDHA", "PHD2"])
        if "apoptosis" in lowered:
            hints.extend(["BAX", "BBC3", "CASP8", "CASP9", "APAF1", "BCL2"])
        if "dna damage response" in lowered:
            hints.extend(["ATM", "ATR", "CHEK1", "CHEK2", "TP53"])

        dedup: list[str] = []
        seen: set[str] = set()
        for item in hints:
            normalized = _normalize_gene(item)
            if not normalized or normalized in seen:
                continue
            if known_only and normalized not in self._known_gene_set:
                continue
            seen.add(normalized)
            dedup.append(item.upper())
        return dedup

    def route_query(self, query: str) -> dict[str, Any]:
        return self.plan_query(query).to_dict()

    def pubmed_rag_search(
        self,
        query: str,
        top_k: int | None = None,
        strategy: list[str] | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> RetrievalBundle:
        plan = self.plan_query(query)
        bundle = self.retriever.retrieve(
            query=query,
            top_k=top_k or self.config.rag_top_k,
            metadata_filters=metadata_filters or plan.metadata_filters,
            strategy=strategy or plan.retrieval_modes,
            graph_hits=self.kg.query_entities(plan.detected_entities) if "graph" in (strategy or plan.retrieval_modes) else None,
        )
        if self.config.enable_reranker:
            bundle = self.retriever.rerank(
                query=query,
                bundle=bundle,
                top_n=min(top_k or self.config.rag_top_k, self.config.rerank_top_n),
            )
        return bundle

    def retrieve(
        self,
        plan: QueryPlan,
        query: str,
    ) -> RetrievalBundle:
        graph_hits = self.kg.query_entities(plan.detected_entities) if "graph" in plan.retrieval_modes else {}
        metadata_filters = plan.metadata_filters if self.config.enable_metadata_filters else {}
        bundle = self.retriever.retrieve(
            query=query,
            top_k=self.config.rag_top_k,
            metadata_filters=metadata_filters,
            strategy=plan.retrieval_modes,
            graph_hits=graph_hits if self.config.enable_graph_retrieval else None,
        )
        if plan.use_reranker and self.config.enable_reranker:
            bundle = self.retriever.rerank(
                query=query,
                bundle=bundle,
                top_n=min(self.config.rag_top_k, self.config.rerank_top_n),
            )
        if self.save_checkpoints:
            self.store.save_json(bundle.to_dict(), self.config.retrieval_trace_name)
            self.store.save_json(bundle.to_dict(), self.config.rerank_trace_name)
        return bundle

    def _expand_gene(self, gene: str) -> dict[str, Any]:
        gene = _normalize_gene(gene)
        summary = self.ncbi_gene_lookup(gene)
        interactions = self.string_interactions(gene)
        drugs = self.drugbank_target_lookup(gene)
        go_terms = self.gene_ontology_lookup(gene)
        pathways = self.pathway_lookup(gene)

        uniprot_data = {}
        if self.config.enable_live_apis:
            uniprot_data = self.uniprot_protein_lookup(gene)
            if uniprot_data.get("accession"):
                self.kg.add_entity(
                    uniprot_data["accession"], "protein",
                    properties={"label": uniprot_data.get("name", gene), **uniprot_data}
                )
                self.kg.add_relationship(gene, uniprot_data["accession"], "ENCODES", properties={"source": "uniprot"})

        self.kg.add_entity(gene, "gene", properties={"label": gene, **summary})
        for row in interactions:
            partner = _normalize_gene(str(row.get("partner", "")))
            if not partner:
                continue
            self.kg.add_entity(partner, "gene", properties={"label": partner})
            self.kg.add_relationship(
                gene,
                partner,
                "INTERACTS_WITH",
                properties={"score": row.get("score", 0), "source": row.get("source", "bundle")},
            )
        for row in drugs:
            drug_name = str(row.get("drug_name", "")).strip()
            if not drug_name:
                continue
            drug_id = str(row.get("drugbank_id", drug_name))
            self.kg.add_entity(drug_id, "drug", properties={"label": drug_name, **row})
            self.kg.add_relationship(
                drug_id,
                gene,
                "TARGETS",
                properties={
                    "status": row.get("status", []),
                    "mechanism": row.get("mechanism", ""),
                    "source": row.get("source", "drugbank_bundle"),
                },
            )
        for term in go_terms:
            go_id = str(term.get("id", ""))
            if not go_id:
                continue
            self.kg.add_entity(go_id, "go_term", properties={"label": term.get("name", go_id), **term})
            self.kg.add_relationship(gene, go_id, "ANNOTATED_WITH", properties={"source": term.get("source", "go_bundle")})
        for pathway in pathways:
            pathway_id = str(pathway.get("pathway_id", ""))
            if not pathway_id:
                continue
            self.kg.add_entity(
                pathway_id,
                "pathway",
                properties={"label": pathway.get("name", pathway_id), **pathway},
            )
            self.kg.add_relationship(
                gene,
                pathway_id,
                "IN_PATHWAY",
                properties={"source": pathway.get("source", "pathway_bundle")},
            )

        return {
            "gene": gene,
            "summary": summary,
            "interactions": interactions,
            "drugs": drugs,
            "go_terms": go_terms,
            "pathways": pathways,
        }

    def expand_graph(self, plan: QueryPlan, bundle: RetrievalBundle) -> dict[str, Any]:
        seed_genes = list(plan.detected_entities)
        for candidate in bundle.candidates[: self.config.rag_top_k]:
            gene = _normalize_gene(str(candidate.payload.get("gene", "")))
            if gene and gene not in seed_genes:
                seed_genes.append(gene)

        expansions = [self._expand_gene(gene) for gene in seed_genes[:8]]
        for candidate in bundle.candidates[: self.config.rag_top_k]:
            payload = candidate.payload
            pmid = str(payload.get("pmid", ""))
            if not pmid:
                continue
            pub_id = f"PMID:{pmid}"
            self.kg.add_entity(
                pub_id,
                "publication",
                properties={"label": payload.get("title", pub_id), "pmid": pmid},
            )
            gene = _normalize_gene(str(payload.get("gene", "")))
            if gene:
                self.kg.add_entity(gene, "gene", properties={"label": gene})
                self.kg.add_relationship(
                    gene,
                    pub_id,
                    "MENTIONED_IN",
                    properties={"score": candidate.final_score, "source": "retrieval"},
                )

        graph_summary = self.kg.summary()
        evidence_flags = {
            "has_relation_evidence": any(expansion["interactions"] or expansion["drugs"] for expansion in expansions),
            "has_process_evidence": any(expansion["go_terms"] or expansion["pathways"] for expansion in expansions),
        }
        if self.save_checkpoints:
            self.kg.save(self.config.graph_checkpoint_path)
            # Build a focused, query-specific visualization graph.
            # We deliberately limit the number of edges per category so the
            # browser renders a clean, readable graph (not the full accumulated
            # graph that can have thousands of nodes).
            vis_kg = BioKnowledgeGraph()
            for expansion in expansions:
                gene = expansion["gene"]
                vis_kg.add_entity(gene, "gene", properties={"label": gene})
                # Top 6 STRING interactions by score
                for iact in sorted(
                    expansion["interactions"],
                    key=lambda x: -int(x.get("score", 0)),
                )[:6]:
                    partner = _normalize_gene(str(iact.get("partner", "")))
                    if not partner:
                        continue
                    vis_kg.add_entity(partner, "gene", properties={"label": partner})
                    vis_kg.add_relationship(
                        gene, partner, "INTERACTS_WITH",
                        properties={"score": iact.get("score", 0)},
                    )
                # Top 5 drugs, approved-first
                def _dsort(d):
                    st = [s.lower() for s in d.get("status", [])]
                    return (0 if "approved" in st else 1)
                for drug in sorted(expansion["drugs"], key=_dsort)[:5]:
                    dname = str(drug.get("drug_name", "")).strip()
                    did = str(drug.get("drugbank_id", dname))
                    if not dname:
                        continue
                    vis_kg.add_entity(did, "drug", properties={"label": dname})
                    vis_kg.add_relationship(
                        did, gene, "TARGETS",
                        properties={"status": drug.get("status", [])},
                    )
                # Top 3 pathways
                for pw in expansion["pathways"][:3]:
                    pid = str(pw.get("pathway_id", ""))
                    if not pid:
                        continue
                    vis_kg.add_entity(
                        pid, "pathway",
                        properties={"label": pw.get("name", pid)},
                    )
                    vis_kg.add_relationship(gene, pid, "IN_PATHWAY")
                # Top 3 GO terms (biological_process only)
                bp_terms = [t for t in expansion["go_terms"]
                            if t.get("namespace") == "biological_process"]
                for term in bp_terms[:3]:
                    gid = str(term.get("id", ""))
                    if not gid:
                        continue
                    vis_kg.add_entity(
                        gid, "go_term",
                        properties={"label": term.get("name", gid)},
                    )
                    vis_kg.add_relationship(gene, gid, "ANNOTATED_WITH")
            vis_kg.export_html(self.config.graph_html_path)
        return {
            "expansions": expansions,
            "graph_summary": graph_summary,
            "evidence_flags": evidence_flags,
        }

    def _build_provenance_table(
        self,
        bundle: RetrievalBundle,
        graph_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        evidence_table: list[dict[str, Any]] = []
        for candidate in bundle.candidates[: self.config.rag_top_k]:
            evidence_table.append(
                {
                    "claim_id": f"literature:{candidate.candidate_id}",
                    "source_id": candidate.candidate_id,
                    "source_type": candidate.source_type,
                    "gene": candidate.payload.get("gene", ""),
                    "title": candidate.payload.get("title", ""),
                    "scores": {
                        "dense": candidate.dense_score,
                        "sparse": candidate.sparse_score,
                        "graph": candidate.graph_score,
                        "rerank": candidate.rerank_score,
                        "final": candidate.final_score,
                    },
                }
            )
        for expansion in graph_context["expansions"]:
            for interaction in expansion["interactions"]:
                evidence_table.append(
                    {
                        "claim_id": f"interaction:{expansion['gene']}:{interaction['partner']}",
                        "source_id": f"STRING:{expansion['gene']}:{interaction['partner']}",
                        "source_type": "string",
                        "relation": "INTERACTS_WITH",
                        "score": min(1.0, float(interaction.get("score", 0)) / 1000.0),
                    }
                )
            for drug in expansion["drugs"]:
                status = [item.lower() for item in drug.get("status", [])]
                evidence_table.append(
                    {
                        "claim_id": f"drug:{drug.get('drugbank_id', drug.get('drug_name', ''))}",
                        "source_id": drug.get("drugbank_id", drug.get("drug_name", "")),
                        "source_type": "drugbank",
                        "relation": "TARGETS",
                        "score": 1.0 if "approved" in status else 0.7,
                    }
                )
            for pathway in expansion["pathways"]:
                evidence_table.append(
                    {
                        "claim_id": f"pathway:{pathway.get('pathway_id', '')}:{expansion['gene']}",
                        "source_id": pathway.get("pathway_id", ""),
                        "source_type": "pathway",
                        "relation": "IN_PATHWAY",
                        "score": 0.8,
                    }
                )
            for term in expansion["go_terms"]:
                term_name = str(term.get("name", "")).lower()
                if term.get("namespace") == "biological_process":
                    evidence_table.append(
                        {
                            "claim_id": f"go_bp:{term.get('id', '')}:{expansion['gene']}",
                            "source_id": term.get("id", ""),
                            "source_type": "go",
                            "relation": "PARTICIPATES_IN",
                            "score": 0.7,
                        }
                    )
                if "phosphorylation" in term_name:
                    evidence_table.append(
                        {
                            "claim_id": f"go_phosphorylation:{term.get('id', '')}:{expansion['gene']}",
                            "source_id": term.get("id", ""),
                            "source_type": "go",
                            "relation": "PHOSPHORYLATES",
                            "score": 0.65,
                        }
                    )
                if "regulation" in term_name:
                    evidence_table.append(
                        {
                            "claim_id": f"go_regulation:{term.get('id', '')}:{expansion['gene']}",
                            "source_id": term.get("id", ""),
                            "source_type": "go",
                            "relation": "REGULATES",
                            "score": 0.6,
                        }
                    )
        return evidence_table

    def _compute_confidence(
        self,
        plan: QueryPlan,
        bundle: RetrievalBundle,
        graph_context: dict[str, Any],
        evidence_table: list[dict[str, Any]],
    ) -> dict[str, Any]:
        literature_confidence = sum(
            candidate.final_score for candidate in bundle.candidates[: self.config.rag_top_k]
        ) / max(len(bundle.candidates[: self.config.rag_top_k]), 1)
        graph_relation_score = sum(
            min(1.0, float(interaction.get("score", 0)) / 1000.0)
            for expansion in graph_context["expansions"]
            for interaction in expansion["interactions"]
        ) / max(
            len(
                [
                    interaction
                    for expansion in graph_context["expansions"]
                    for interaction in expansion["interactions"]
                ]
            ),
            1,
        )
        drug_confidence = sum(
            1.0 if "approved" in [item.lower() for item in drug.get("status", [])] else 0.7
            for expansion in graph_context["expansions"]
            for drug in expansion["drugs"]
        ) / max(
            len(
                [
                    drug
                    for expansion in graph_context["expansions"]
                    for drug in expansion["drugs"]
                ]
            ),
            1,
        )
        coverage = min(1.0, len(evidence_table) / max(self.config.min_supporting_evidence * 3, 1))
        source_agreement = min(1.0, len({item["source_type"] for item in evidence_table}) / 4.0)
        path_support = 1.0 / (1.0 + max(0, self.config.max_graph_hops - 1))
        # Weight components by query type so confidence varies meaningfully
        qt = plan.query_type
        if qt == "drug_target":
            # Drug queries: drug_confidence matters most
            overall = _clip(
                0.25 * literature_confidence
                + 0.15 * coverage
                + 0.15 * source_agreement
                + 0.10 * graph_relation_score
                + 0.25 * drug_confidence
                + 0.10 * path_support
            )
        elif qt == "relationship":
            # Relationship queries: graph_relation_score matters most
            overall = _clip(
                0.25 * literature_confidence
                + 0.15 * coverage
                + 0.15 * source_agreement
                + 0.25 * graph_relation_score
                + 0.10 * drug_confidence
                + 0.10 * path_support
            )
        else:
            # Default balanced weighting
            overall = _clip(
                0.25 * literature_confidence
                + 0.15 * coverage
                + 0.15 * source_agreement
                + 0.15 * graph_relation_score
                + 0.15 * drug_confidence
                + 0.15 * path_support
            )
        return {
            "literature_confidence": round(literature_confidence, 4),
            "graph_confidence": round(graph_relation_score, 4),
            "drug_confidence": round(drug_confidence, 4),
            "overall_confidence": round(overall, 4),
            "route_type": plan.query_type,
            "retrieval_channels": list(bundle.strategy),
            "reranker_backend": bundle.diagnostics.get("reranker_backend", "disabled"),
        }

    def synthesize(
        self,
        query: str,
        plan: QueryPlan,
        bundle: RetrievalBundle,
        graph_context: dict[str, Any],
        iterations: list[RetrievalIteration],
    ) -> AgentResult:
        evidence_table = self._build_provenance_table(bundle, graph_context)
        confidence = self._compute_confidence(plan, bundle, graph_context, evidence_table)

        def _drug_sort_key(drug):
            status = [s.lower() for s in drug.get("status", [])]
            if "approved" in status:
                return 0
            for s in status:
                m = re.match(r"phase (\d+)", s)
                if m:
                    return 5 - int(m.group(1))
            return 10

        # Build a clean plain-text answer (used by API callers and as LLM fallback)
        entities = plan.detected_entities
        gene_str = " and ".join(entities) if entities else "the queried gene(s)"
        qt = plan.query_type

        # Opening sentence
        if qt in ("drug_target", "mechanistic"):
            opening = f"{gene_str} is a well-studied therapeutic target."
        elif qt == "relationship":
            opening = f"Here is what is known about the relationship involving {gene_str}."
        elif qt == "pathway":
            opening = f"Here is what is known about {gene_str} in biological pathways."
        else:
            opening = f"Here is a summary of current knowledge about {gene_str}."

        parts = [opening]

        # Approved drugs — name + mechanism, no IDs
        all_drugs = []
        for exp in graph_context["expansions"]:
            for drug in sorted(exp["drugs"], key=_drug_sort_key)[:4]:
                st = [s.lower() for s in drug.get("status", [])]
                if "approved" in st:
                    mech = drug.get("mechanism", f"targets {exp['gene']}")
                    all_drugs.append(f"{drug.get('drug_name', '')} ({mech})")
        if all_drugs:
            parts.append(
                "Approved targeted therapies include: " + "; ".join(all_drugs[:5]) + "."
            )

        # Top protein partners — names only, no scores
        partner_bits: list[str] = []
        for exp in graph_context["expansions"]:
            top_iacts = sorted(exp["interactions"], key=lambda x: -int(x.get("score", 0)))[:4]
            for iact in top_iacts:
                p = iact.get("partner", "")
                if p:
                    partner_bits.append(f"{exp['gene']}–{p}")
        if partner_bits:
            parts.append(
                "Key protein interactions: " + ", ".join(list(dict.fromkeys(partner_bits))[:6]) + "."
            )

        # Pathways
        pw_bits: list[str] = []
        for exp in graph_context["expansions"]:
            for pw in exp.get("pathways", [])[:2]:
                name = pw.get("name", "")
                if name:
                    pw_bits.append(name)
        if pw_bits:
            parts.append(
                "Pathway involvement: " + "; ".join(list(dict.fromkeys(pw_bits))[:3]) + "."
            )

        # Literature
        lit_titles = [
            c.payload.get("title", "") for c in bundle.candidates[:3]
            if c.payload.get("title")
        ]
        if lit_titles:
            parts.append("Supporting literature: " + "; ".join(lit_titles) + ".")

        conf_pct = int(confidence["overall_confidence"] * 100)
        parts.append(f"Evidence confidence: {conf_pct}% (from {len(bundle.candidates)} retrieved records).")
        if confidence["overall_confidence"] < self.config.confidence_threshold:
            parts.append("Evidence is limited — treat this answer as preliminary.")

        # LLM synthesis override (when Groq API key or local model is set)
        if hasattr(self, '_llm') and self._llm is not None:
            try:
                from .llm import synthesize_answer
                llm_answer = synthesize_answer(
                    query=query,
                    evidence_table=evidence_table,
                    graph_summary=graph_context["graph_summary"],
                    confidence=confidence,
                    llm=self._llm,
                    expansions=graph_context.get("expansions", []),
                    lit_candidates=bundle.candidates[:8],
                )
                if llm_answer and len(llm_answer.strip()) > 50:
                    # UI already renders a confidence footer; avoid duplicate
                    # trailing confidence text in LLM answers.
                    parts = [llm_answer.strip()]
            except Exception as _llm_err:
                print(f"[agent] LLM synthesis failed: {_llm_err}")
                # fall back to template answer

        checkpoint_paths = {
            "query_plan": str(self.config.query_plan_path),
            "retrieval_trace": str(self.config.retrieval_trace_path),
            "rerank_trace": str(self.config.rerank_trace_path),
            "iteration_trace": str(self.config.iteration_trace_path),
            "confidence_report": str(self.config.confidence_report_path),
            "provenance_table": str(self.config.provenance_table_path),
            "graph": str(self.config.graph_checkpoint_path),
            "graph_html": str(self.config.graph_html_path),
        }

        result = AgentResult(
            answer_text=" ".join(parts),
            query_plan=plan.to_dict(),
            retrieval_iterations=[iteration.to_dict() for iteration in iterations],
            evidence_table=evidence_table,
            graph_summary=graph_context["graph_summary"],
            confidence_summary=confidence,
            checkpoint_paths=checkpoint_paths,
        )

        if self.save_checkpoints:
            self.store.save_json([iteration.to_dict() for iteration in iterations], self.config.iteration_trace_name)
            self.store.save_json(confidence, self.config.confidence_report_name)
            self.store.save_json(evidence_table, self.config.provenance_table_name)
        return result

    def invoke(self, query: str, top_k: int | None = None, **kwargs: Any) -> dict[str, Any]:
        if top_k:
            self.config.rag_top_k = top_k

        plan = self.plan_query(query)
        current_query = query
        iterations: list[RetrievalIteration] = []
        final_bundle: RetrievalBundle | None = None
        graph_context: dict[str, Any] | None = None

        for iteration_index in range(plan.max_iterations):
            bundle = self.retrieve(plan, current_query)
            graph_context = self.expand_graph(plan, bundle)
            assessment = self.router.assess(
                plan=plan,
                final_scores=[candidate.final_score for candidate in bundle.candidates],
                graph_summary=graph_context["graph_summary"],
                evidence_flags=graph_context["evidence_flags"],
                iteration=iteration_index,
            )
            iterations.append(
                RetrievalIteration(
                    iteration=iteration_index + 1,
                    query=current_query,
                    plan=plan.to_dict(),
                    retrieval=bundle.to_dict(),
                    assessment=assessment.to_dict(),
                )
            )
            final_bundle = bundle
            if assessment.enough_evidence or not assessment.reformulated_query:
                break
            current_query = assessment.reformulated_query

        result = self.synthesize(
            query=query,
            plan=plan,
            bundle=final_bundle or self.retrieve(plan, query),
            graph_context=graph_context or self.expand_graph(plan, final_bundle or self.retrieve(plan, query)),
            iterations=iterations,
        )
        payload = result.to_dict()
        payload["answer"] = payload["answer_text"]
        payload["route_type"] = result.query_plan["query_type"]
        payload["retrieval_channels"] = result.confidence_summary["retrieval_channels"]
        payload["reranker_used"] = result.confidence_summary["reranker_backend"] == "cross_encoder"
        payload["reranker_fallback_used"] = result.confidence_summary["reranker_backend"] == "heuristic"
        payload["retrieval_iterations_count"] = len(result.retrieval_iterations)
        payload["graph_html"] = str(self.config.graph_html_path)
        # Rich data for the Gradio UI to render structured HTML
        _gc = graph_context or {}
        payload["expansions"] = _gc.get("expansions", [])
        _fb = final_bundle or self.retrieve(plan, query)
        payload["lit_titles"] = [
            c.payload.get("title", "")
            for c in _fb.candidates[:6]
            if c.payload.get("title")
        ]
        # Show total records in the retrieval index, not just the top-K
        payload["lit_count"] = len(getattr(self.retriever, "records", [])) or len(_fb.candidates)
        if self.save_checkpoints:
            self.store.save_json(
                {
                    "query": query,
                    "route_type": payload["route_type"],
                    "confidence": result.confidence_summary,
                    "iterations": payload["retrieval_iterations_count"],
                },
                "last_query.json",
            )
            self.store.save_json(payload, "last_query_result.json")
        return payload

    def answer(self, query: str, top_k: int | None = None, **kwargs: Any) -> str:
        return str(self.invoke(query=query, top_k=top_k, **kwargs)["answer_text"])

    def run(self, query: str, top_k: int | None = None, **kwargs: Any) -> AgentResult:
        payload = self.invoke(query=query, top_k=top_k, **kwargs)
        return AgentResult(
            answer_text=payload["answer_text"],
            query_plan=payload["query_plan"],
            retrieval_iterations=payload["retrieval_iterations"],
            evidence_table=payload["evidence_table"],
            graph_summary=payload["graph_summary"],
            confidence_summary=payload["confidence_summary"],
            checkpoint_paths=payload["checkpoint_paths"],
        )

    def save(self) -> None:
        self.store.save_json(self.bundle.as_dict(), self.config.demo_bundle_name)
        self.retriever.to_checkpoint(self.store, self.config.retriever_checkpoint_name)
        self.kg.save(self.store.path(self.config.graph_checkpoint_name))

    @classmethod
    def build(
        cls,
        config: ProjectConfig | None = None,
        retriever: HybridRetrievalEngine | None = None,
        retrieval_index: HybridRetrievalEngine | None = None,
        router: QueryRouter | None = None,
        kg: BioKnowledgeGraph | None = None,
        knowledge_graph: BioKnowledgeGraph | None = None,
        checkpoint_dir: str | None = None,
        save_checkpoints: bool = True,
        demo_mode: bool = True,
        bundle: DemoBundle | None = None,
        store: CheckpointStore | None = None,
        planner: Callable[[str, list[str]], dict[str, Any]] | None = None,
    ) -> "BioKGAgent":
        return cls(
            config=config or default_config(),
            retriever=retriever or retrieval_index,
            router=router,
            kg=kg,
            knowledge_graph=knowledge_graph,
            checkpoint_dir=checkpoint_dir,
            save_checkpoints=save_checkpoints,
            demo_mode=demo_mode,
            bundle=bundle,
            store=store,
            planner=planner,
        )

    def attach_llm(self, llm) -> None:
        """Attach an LLM backend for synthesis and planning."""
        self._llm = llm
        try:
            from .llm import make_planner
            if self.config.enable_llm_planner:
                self.router.planner = make_planner(llm)
        except Exception:
            pass


def build_demo_agent(**kwargs: Any) -> BioKGAgent:
    return BioKGAgent.build(**kwargs)


def create_demo_agent(**kwargs: Any) -> BioKGAgent:
    return build_demo_agent(**kwargs)


def make_demo_agent(**kwargs: Any) -> BioKGAgent:
    return build_demo_agent(**kwargs)


BioKGDemoAgent = BioKGAgent
