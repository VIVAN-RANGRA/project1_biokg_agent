# BioKG-Agent

BioKG-Agent is a biology-focused RAG + knowledge graph assistant with:

- hybrid retrieval (hashed dense + BM25 + graph-aware signals)
- explicit query routing and iterative retrieval
- evidence table and source-aware answer synthesis
- reproducible benchmark tooling (`eval/run_eval.py`)
- checkpointed artifacts for fast app replay

## Repository Layout

- `app.py`: Gradio app
- `run_demo.py`: quick smoke checks and single-query CLI
- `biokg_agent/`: core agent, retrieval, routing, KG, and LLM clients
- `scripts/`: data ingestion and bundle/index build scripts
- `eval/benchmark.json`: 100-question evaluation set
- `eval/run_eval.py`: benchmark runner
- `checkpoints/`: generated runtime artifacts (not committed)
- `data/`: downloaded corpora and processed datasets (not committed)

## Setup

```powershell
pip install -r requirements-kaggle.txt
```

### API key setup (PowerShell)

Session only:

```powershell
$env:GROQ_API_KEY = "YOUR_KEY"
```

Persist for current user:

```powershell
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", "YOUR_KEY", "User")
```

Remove from current session:

```powershell
Remove-Item Env:\GROQ_API_KEY -ErrorAction SilentlyContinue
```

## Quick Validation

Smoke eval:

```powershell
python .\run_demo.py --smoke-eval --print-json
```

Single query:

```powershell
python .\run_demo.py --query "What drugs target TP53 pathway proteins?" --print-json
```

Launch app:

```powershell
python .\app.py
```

## Full Benchmark (100 Questions)

```powershell
python .\eval\run_eval.py --benchmark .\eval\benchmark.json --output .\eval\results.json
```

Resume interrupted run:

```powershell
python .\eval\run_eval.py --resume .\eval\results.json --output .\eval\results.json
```

Filter by category:

```powershell
python .\eval\run_eval.py --category pathway_mechanism --output .\eval\results_pathway.json
```

## Data Pipeline (No OpenTargets in Default Flow)

One-shot script:

```bash
bash scripts/run_all.sh
```

Manual steps:

```powershell
python .\scripts\ingest_pubmed.py --gene-list .\data\top_500_genes.txt --genes 500 --abstracts-per-gene 1200 --workers 4 --batch-size 100 --output .\data\pubmed_abstracts.jsonl
python .\scripts\ingest_string.py --output .\data\string_ppi.jsonl --score-threshold 700
python .\scripts\ingest_chembl_approved.py --output .\data\drugbank_targets.jsonl --target-approved-drugs 2600
python .\scripts\ingest_go.py --output-terms .\data\go_terms.jsonl --output-annotations .\data\gene_annotations.jsonl
python .\scripts\build_index.py --data-dir .\data\ --output-dir .\checkpoints\ --batch-size 512
```

For a smaller reproducible bundle build:

```powershell
python .\scripts\build_index.py --data-dir .\data\ --output-dir .\checkpoints\ --max-pubmed-records 300000 --max-partners-per-gene 80 --batch-size 512
```

## Reproducibility Notes

- `eval/run_eval.py` writes partial results after each question, so interrupted runs can resume safely.
- `checkpoints/`, `data/`, `logs/`, and evaluation result dumps are git-ignored by default.
- Defaults are CPU-safe (`hashed` dense backend + heuristic reranker), so local verification does not require large model downloads.

## Current Benchmark Snapshot

Latest hardened 100Q run (`eval/results_full_300k_hardened_v2.json`):

- pass rate: `0.87`
- entity recall: `0.7173`
- keyword recall: `0.4615`
- relationship recall: `0.3482`
- avg confidence: `0.7518`

## Notes

- The reranker may fall back to heuristic mode even when the advanced pipeline works correctly. That is expected in CPU-safe mode.
- The bundled data is a compact seed for verification, not the final large-scale corpus. The architecture is ready for larger real data later.
- The app is designed to degrade gracefully in CPU-only mode.
- If you want a clean run, remove `checkpoints/` before rebuilding.
- Data quality and corpus size still matter for rare, highly niche biomedical questions.
