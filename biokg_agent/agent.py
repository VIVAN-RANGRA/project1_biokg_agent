"""Advanced retrieval and synthesis agent for BioKG-Agent."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

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
        self.retriever = self.retriever or HybridRetrievalEngine.from_records(
            self.bundle.pubmed_records,
            config=self.config,
        )
        self.router = self.router or QueryRouter(config=self.config, planner=self.planner)
        self.kg = self.kg or self.knowledge_graph or BioKnowledgeGraph.from_checkpoint(
            checkpoint_dir=self.config.checkpoint_dir_path,
            graph_checkpoint_name=self.config.graph_checkpoint_name,
        )
        self._seed_graph()
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
        return [
            {**dict(row), "source": row.get("source", "drugbank_bundle")}
            for row in self.bundle.drugbank.get(gene, [])
        ]

    def gene_ontology_lookup(self, gene: str) -> list[dict[str, Any]]:
        gene = _normalize_gene(gene)
        return [
            {**dict(self.bundle.go_terms.get(go_id, {"id": go_id, "name": go_id})), "source": "go_bundle"}
            for go_id in self.bundle.gene_annotations.get(gene, [])
        ]

    def pathway_lookup(self, gene: str) -> list[dict[str, Any]]:
        gene = _normalize_gene(gene)
        return [
            {**dict(self.bundle.pathways.get(pathway_id, {"pathway_id": pathway_id, "name": pathway_id})), "source": "pathway_bundle"}
            for pathway_id in self.bundle.pathway_membership.get(gene, [])
        ]

    def plan_query(self, query: str) -> QueryPlan:
        plan = self.router.plan(query, self.known_genes())
        query_upper = query.upper()
        detected_entities = list(plan.detected_entities)
        for alias, canonical in self.bundle.gene_synonyms.items():
            if alias.upper() in query_upper:
                normalized = _normalize_gene(canonical)
                if normalized and normalized not in detected_entities:
                    detected_entities.append(normalized)
        plan.detected_entities = detected_entities
        if detected_entities:
            plan.metadata_filters = {**plan.metadata_filters, "genes": detected_entities}
        if self.save_checkpoints:
            self.store.save_json(plan.to_dict(), self.config.query_plan_name)
        return plan

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

        expansions = [self._expand_gene(gene) for gene in seed_genes[:4]]
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
            self.kg.export_html(self.config.graph_html_path)
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
        overall = _clip(
            0.40 * literature_confidence
            + 0.20 * coverage
            + 0.20 * source_agreement
            + 0.20 * path_support
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

        answer_parts = [
            f"Route: {plan.query_type}.",
            f"Detected entities: {', '.join(plan.detected_entities) if plan.detected_entities else 'none explicit'}.",
        ]
        if bundle.candidates:
            titles = [candidate.payload.get("title", "") for candidate in bundle.candidates[:3] if candidate.payload.get("title")]
            if titles:
                answer_parts.append("Top literature evidence: " + "; ".join(titles) + ".")
        relation_bits = []
        process_bits = []
        drug_bits = []
        for expansion in graph_context["expansions"]:
            for interaction in expansion["interactions"][:2]:
                relation_bits.append(f"{expansion['gene']} -> {interaction['partner']} (STRING {interaction.get('score', 0)})")
            for pathway in expansion["pathways"][:2]:
                process_bits.append(f"{expansion['gene']} in {pathway.get('name', pathway.get('pathway_id', 'pathway'))}")
            for drug in expansion["drugs"][:2]:
                drug_bits.append(f"{drug.get('drug_name', '')} targets {expansion['gene']}")
        if relation_bits:
            answer_parts.append("Graph relations: " + "; ".join(relation_bits[:6]) + ".")
        if process_bits:
            answer_parts.append("Pathway/process evidence: " + "; ".join(process_bits[:6]) + ".")
        if drug_bits:
            answer_parts.append("Drug evidence: " + "; ".join(drug_bits[:6]) + ".")
        answer_parts.append(f"Overall confidence: {confidence['overall_confidence']:.2f}.")
        if confidence["overall_confidence"] < self.config.confidence_threshold:
            answer_parts.append("Evidence is still limited, so treat this answer as tentative.")

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
            answer_text=" ".join(answer_parts),
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


def build_demo_agent(**kwargs: Any) -> BioKGAgent:
    return BioKGAgent.build(**kwargs)


def create_demo_agent(**kwargs: Any) -> BioKGAgent:
    return build_demo_agent(**kwargs)


def make_demo_agent(**kwargs: Any) -> BioKGAgent:
    return build_demo_agent(**kwargs)


BioKGDemoAgent = BioKGAgent
