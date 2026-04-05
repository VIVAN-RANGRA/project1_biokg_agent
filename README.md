# BioKG-Agent

Advanced RAG over biological knowledge sources with:

- hybrid retrieval: dense + BM25 + graph priors
- explicit query routing
- reranking with CPU-safe fallback
- iterative retrieval loops
- confidence scoring and provenance tracking
- checkpointed outputs for Gradio demos and Kaggle runs


## What It Does

Given a biology question like:

`What drugs target TP53 pathway proteins?`

the system can:

1. classify the query with an explicit route plan
2. run hybrid retrieval over literature records
3. expand graph evidence from STRING-like interactions, DrugBank-like targets, GO terms, and pathway membership
4. rerank the evidence
5. iterate if evidence is still weak
6. return an answer with confidence, provenance, and saved traces

## Main Files

- [run_demo.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\run_demo.py): CLI entrypoint for smoke evals and single-query runs
- [app.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\app.py): Gradio demo
- [requirements-kaggle.txt](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\requirements-kaggle.txt): Kaggle-friendly installs
- [biokg_agent/config.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\biokg_agent\config.py): feature flags, paths, weights, thresholds
- [biokg_agent/router.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\biokg_agent\router.py): query planning and iterative evidence checks
- [biokg_agent/retrieval.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\biokg_agent\retrieval.py): hybrid retrieval and reranking
- [biokg_agent/agent.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\biokg_agent\agent.py): orchestration and final synthesis
- [biokg_agent/kg.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\biokg_agent\kg.py): knowledge graph store and export
- [biokg_agent/data.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\biokg_agent\data.py): bundled seed data and checkpoint loading

## Quick Start

### 1. Install

```powershell
pip install -r requirements-kaggle.txt
```

### 2. Run the CPU-safe smoke eval

```powershell
python .\run_demo.py --smoke-eval --print-json
```

This validates:

- routing
- hybrid retrieval channels
- reranker fallback behavior
- iterative retrieval
- confidence fields
- provenance output
- alias normalization such as `IL-6 -> IL6`

### 3. Run one query

```powershell
python .\run_demo.py --query "What drugs target TP53 pathway proteins?" --print-json
```

### 4. Launch the app

```powershell
python .\app.py
```

## Kaggle Copy-Paste Flow

Use this sequence inside a Kaggle notebook terminal or cell:

```powershell
pip install -r requirements-kaggle.txt
python .\run_demo.py --smoke-eval --print-json
python .\run_demo.py --query "Which papers explain BRCA1 and PARP inhibitor resistance?" --print-json
python .\app.py
```

If you only want a fast verification pass, stop after the smoke eval.

## Default Runtime Behavior

The defaults are intentionally CPU-safe:

- `dense_backend = "hashed"`
- `reranker_backend = "heuristic"`
- `enable_live_apis = False`

That means:

- no heavy embedding model download is required for smoke tests
- no cross-encoder is required to verify the pipeline
- no live API dependency is needed for local correctness checks

## Switching To Heavier Kaggle Mode

Edit [biokg_agent/config.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\biokg_agent\config.py) if you want stronger but heavier retrieval:

- set `dense_backend = "auto"` or `"sentence_transformers"`
- set `reranker_backend = "cross_encoder"`
- keep `enable_reranker = True`
- optionally set `enable_live_apis = True`
- optionally set `ncbi_api_key`

Recommended pattern:

1. verify with CPU-safe mode
2. turn on dense models
3. turn on cross-encoder reranking
4. turn on live APIs only when the rest is stable

## Checkpoint Artifacts

Every run writes reusable artifacts to `checkpoints/`, including:

- `query_plan.json`
- `retrieval_trace.json`
- `rerank_trace.json`
- `iteration_trace.json`
- `confidence_report.json`
- `provenance_table.json`
- `smoke_eval.json`
- `last_query.json`
- `last_query_result.json`
- `kg_session.pkl`
- `kg_graph.html`
- `simple_retrieval_index.pkl`
- `demo_bundle.json`

These are what make the app demo and video workflow easy to replay without rerunning the entire flow.

## Public API Summary

Key methods in [biokg_agent/agent.py](C:\Users\AMIT\Desktop\BIO_PROJECTS\project1_biokg_agent\biokg_agent\agent.py):

- `plan_query(query)`
- `route_query(query)`
- `pubmed_rag_search(query, ...)`
- `string_interactions(gene, ...)`
- `drugbank_target_lookup(gene)`
- `gene_ontology_lookup(gene)`
- `pathway_lookup(gene)`
- `invoke(query)`
- `answer(query)`

Key classes:

- `BioKGAgent`
- `QueryRouter`
- `QueryPlan`
- `HybridRetrievalEngine`
- `RetrievalBundle`
- `BioKnowledgeGraph`

## Smoke Eval Scenarios

The built-in smoke suite currently checks:

- TP53 pathway drug question
- BRCA1 literature-focused question
- iterative TP53 therapy connection question
- IL-6 alias normalization question

The eval passes only if the output includes:

- a valid route type
- the expected retrieval channels
- at least the expected iteration count
- valid confidence values in `[0, 1]`
- a non-empty provenance table

## Notes

- The reranker may fall back to heuristic mode even when the advanced pipeline works correctly. That is expected in CPU-safe mode.
- The bundled data is a compact seed for verification, not the final large-scale corpus. The architecture is ready for larger real data later.
