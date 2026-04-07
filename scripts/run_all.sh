#!/bin/bash
# Full data ingestion pipeline for BioKG-Agent
# Run from project root: bash scripts/run_all.sh

set -e
mkdir -p data checkpoints

echo "=== Step 1/5: PubMed abstracts ==="
python scripts/ingest_pubmed.py --genes 50 --abstracts-per-gene 200 --output data/pubmed_abstracts.jsonl --batch-size 100

echo "=== Step 2/5: STRING PPI ==="
python scripts/ingest_string.py --output data/string_ppi.jsonl --score-threshold 700

echo "=== Step 3/5: Approved drug-target table (ChEMBL, no OpenTargets) ==="
python scripts/ingest_chembl_approved.py --output data/drugbank_targets.jsonl --target-approved-drugs 2600

echo "=== Step 4/5: Gene Ontology ==="
python scripts/ingest_go.py --output-terms data/go_terms.jsonl --output-annotations data/gene_annotations.jsonl

echo "=== Step 5/5: Build FAISS index + bundle ==="
python scripts/build_index.py --data-dir data/ --output-dir checkpoints/ --batch-size 512

echo "=== Done! Data ready in data/ and checkpoints/ ==="
