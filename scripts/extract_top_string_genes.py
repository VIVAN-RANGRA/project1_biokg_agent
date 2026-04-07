"""Extract top genes by STRING degree from a JSONL edge file.

Usage:
    python scripts/extract_top_string_genes.py \
        --input data/string_ppi.jsonl \
        --top-k 500 \
        --output data/top_500_string_genes.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top genes by STRING edge degree.")
    parser.add_argument("--input", type=str, default="data/string_ppi.jsonl", help="Input STRING JSONL path")
    parser.add_argument("--output", type=str, default="data/top_500_string_genes.txt", help="Output gene list path")
    parser.add_argument("--top-k", type=int, default=500, help="Number of genes to write")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    degree: Counter[str] = Counter()
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            g1 = str(rec.get("gene", "")).strip().upper()
            g2 = str(rec.get("partner", "")).strip().upper()
            if g1:
                degree[g1] += 1
            if g2:
                degree[g2] += 1

    top = degree.most_common(args.top_k)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for gene, _ in top:
            out.write(gene + "\n")

    print(f"Wrote {len(top):,} genes to {out_path.resolve()}")


if __name__ == "__main__":
    main()

