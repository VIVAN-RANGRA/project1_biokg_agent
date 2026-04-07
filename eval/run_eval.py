"""Evaluation runner for BioKG-Agent benchmark.

Usage:
    # Run full benchmark (100 questions)
    python eval/run_eval.py --output eval/results.json

    # Run subset
    python eval/run_eval.py --max-questions 10 --output eval/results.json

    # Run specific category
    python eval/run_eval.py --category drug_target --output eval/results.json

    # Run with live APIs
    python eval/run_eval.py --live-apis --output eval/results.json

    # Resume from partial results
    python eval/run_eval.py --resume eval/results.json --output eval/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Make script runnable from project root without external PYTHONPATH tweaks.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_benchmark(benchmark_path: str) -> dict[str, Any]:
    """Load the benchmark JSON file."""
    with open(benchmark_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_results(results: dict[str, Any], output_path: str) -> None:
    """Atomically save results to JSON (write to tmp then rename)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    tmp_path.replace(path)


def _entity_recall(expected_entities: list[str], answer_text: str) -> float:
    """Fraction of expected entities found in the answer text (case-insensitive)."""
    if not expected_entities:
        return 1.0
    answer_upper = answer_text.upper()
    found = sum(1 for entity in expected_entities if entity.upper() in answer_upper)
    return found / len(expected_entities)


def _keyword_recall(gold_keywords: list[str], answer_text: str) -> float:
    """Fraction of gold answer keywords found in the answer text (case-insensitive)."""
    if not gold_keywords:
        return 1.0
    answer_lower = answer_text.lower()
    found = sum(1 for kw in gold_keywords if kw.lower() in answer_lower)
    return found / len(gold_keywords)


def _relationship_recall(
    expected_relationships: list[str],
    graph_summary: dict[str, Any],
    evidence_table: list[dict[str, Any]],
    answer_text: str,
) -> float:
    """Fraction of expected relationship types found in graph_summary, evidence_table, or answer."""
    if not expected_relationships:
        return 1.0

    # Collect all text we can search through
    searchable = answer_text.upper()

    # Add relationship types from graph_summary
    for rel_type in graph_summary.get("relationship_types", []):
        searchable += " " + str(rel_type).upper()
    # Also check edge_count keys or any relationship mentions in graph_summary
    for key, value in graph_summary.items():
        searchable += " " + str(key).upper() + " " + str(value).upper()

    # Add relation fields from evidence_table
    for entry in evidence_table:
        rel = entry.get("relation", "")
        if rel:
            searchable += " " + str(rel).upper()
        claim_id = entry.get("claim_id", "")
        if claim_id:
            searchable += " " + str(claim_id).upper()
        source_id = entry.get("source_id", "")
        if source_id:
            searchable += " " + str(source_id).upper()

    found = sum(1 for rel in expected_relationships if rel.upper() in searchable)
    return found / len(expected_relationships)


def _source_coverage(
    expected_sources: list[str],
    retrieval_channels: list[str],
    evidence_table: list[dict[str, Any]],
    answer_text: str,
) -> float:
    """Fraction of expected data sources that were used or mentioned."""
    if not expected_sources:
        return 1.0

    # Build a searchable corpus from channels, evidence source types, and the answer
    searchable = " ".join(retrieval_channels).upper()
    searchable += " " + answer_text.upper()
    for entry in evidence_table:
        source_type = str(entry.get("source_type", "")).upper()
        source_id = str(entry.get("source_id", "")).upper()
        searchable += " " + source_type + " " + source_id

    # Map common expected_data_sources to what actually appears
    source_aliases: dict[str, list[str]] = {
        "STRING": ["STRING", "INTERACTS_WITH", "INTERACTION"],
        "KEGG": ["KEGG", "PATHWAY", "IN_PATHWAY"],
        "UNIPROT": ["UNIPROT", "PROTEIN"],
        "DRUGBANK": ["DRUGBANK", "DRUG", "TARGETS"],
        "NCBI": ["NCBI", "GENE", "PUBMED", "PMID", "LITERATURE"],
        "GO": ["GO_TERM", "GO:", "ANNOTATED_WITH", "GENE ONTOLOGY"],
        "PUBMED": ["PUBMED", "PMID", "LITERATURE"],
    }

    found = 0
    for src in expected_sources:
        src_upper = src.upper()
        aliases = source_aliases.get(src_upper, [src_upper])
        if any(alias in searchable for alias in aliases):
            found += 1
    return found / len(expected_sources)


def _progress_bar(current: int, total: int, width: int = 40, extra: str = "") -> str:
    """Build a simple text progress bar."""
    filled = int(width * current / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * current / max(total, 1)
    return f"[{bar}] {current}/{total} ({pct:5.1f}%) {extra}"


def _evaluate_question(
    agent: Any,
    question: dict[str, Any],
) -> dict[str, Any]:
    """Run agent on a single question and compute per-question metrics."""
    qid = question["id"]
    query = question["question"]
    t0 = time.time()

    try:
        payload = agent.invoke(query)
    except Exception as exc:
        return {
            "id": qid,
            "question": query,
            "category": question.get("category", ""),
            "difficulty": question.get("difficulty", ""),
            "answer": "",
            "entity_recall": 0.0,
            "keyword_recall": 0.0,
            "relationship_recall": 0.0,
            "confidence": 0.0,
            "iterations": 0,
            "sources_used": [],
            "expected_sources": question.get("expected_data_sources", []),
            "source_coverage": 0.0,
            "elapsed_seconds": round(time.time() - t0, 2),
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    elapsed = round(time.time() - t0, 2)
    answer_text = payload.get("answer_text", "")
    graph_summary = payload.get("graph_summary", {})
    evidence_table = payload.get("evidence_table", [])
    confidence_summary = payload.get("confidence_summary", {})
    retrieval_channels = payload.get("retrieval_channels", [])
    iterations_count = payload.get("retrieval_iterations_count", 1)
    confidence_val = confidence_summary.get("overall_confidence", 0.0)

    ent_recall = _entity_recall(question.get("expected_entities", []), answer_text)
    kw_recall = _keyword_recall(question.get("gold_answer_keywords", []), answer_text)
    rel_recall = _relationship_recall(
        question.get("expected_relationships", []),
        graph_summary,
        evidence_table,
        answer_text,
    )
    src_cov = _source_coverage(
        question.get("expected_data_sources", []),
        retrieval_channels,
        evidence_table,
        answer_text,
    )

    # A question passes if entity recall >= 0.3 and confidence is in valid range
    passed = ent_recall >= 0.3 and 0.0 <= confidence_val <= 1.0

    return {
        "id": qid,
        "question": query,
        "category": question.get("category", ""),
        "difficulty": question.get("difficulty", ""),
        "answer": answer_text,
        "entity_recall": round(ent_recall, 4),
        "keyword_recall": round(kw_recall, 4),
        "relationship_recall": round(rel_recall, 4),
        "confidence": round(confidence_val, 4),
        "iterations": iterations_count,
        "sources_used": retrieval_channels,
        "expected_sources": question.get("expected_data_sources", []),
        "source_coverage": round(src_cov, 4),
        "elapsed_seconds": elapsed,
        "pass": passed,
        "error": None,
    }


def _aggregate_metrics(per_question: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate metrics across all evaluated questions."""
    if not per_question:
        return {
            "entity_recall": 0.0,
            "relationship_recall": 0.0,
            "keyword_recall": 0.0,
            "avg_confidence": 0.0,
            "avg_iterations": 0.0,
            "source_coverage": 0.0,
            "pass_rate": 0.0,
            "error_rate": 0.0,
        }
    n = len(per_question)
    return {
        "entity_recall": round(sum(q["entity_recall"] for q in per_question) / n, 4),
        "relationship_recall": round(sum(q["relationship_recall"] for q in per_question) / n, 4),
        "keyword_recall": round(sum(q["keyword_recall"] for q in per_question) / n, 4),
        "avg_confidence": round(sum(q["confidence"] for q in per_question) / n, 4),
        "avg_iterations": round(sum(q["iterations"] for q in per_question) / n, 2),
        "source_coverage": round(sum(q["source_coverage"] for q in per_question) / n, 4),
        "pass_rate": round(sum(1 for q in per_question if q["pass"]) / n, 4),
        "error_rate": round(sum(1 for q in per_question if q.get("error")) / n, 4),
    }


def _group_metrics(
    per_question: list[dict[str, Any]], key: str
) -> dict[str, dict[str, float]]:
    """Group per-question results by a key and compute aggregates per group."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for q in per_question:
        group = q.get(key, "unknown")
        groups.setdefault(group, []).append(q)
    return {group: _aggregate_metrics(items) for group, items in sorted(groups.items())}


def _print_summary_table(results: dict[str, Any]) -> None:
    """Print a formatted summary table to stdout."""
    agg = results.get("aggregate_metrics", {})

    print("\n" + "=" * 72)
    print("  BioKG-Agent Benchmark Evaluation Summary")
    print("=" * 72)
    print(f"  Questions evaluated : {results['metadata']['num_questions']}")
    print(f"  Categories          : {', '.join(results['metadata']['categories_evaluated'])}")
    print()

    print("  AGGREGATE METRICS")
    print("  " + "-" * 40)
    print(f"  Entity recall       : {agg.get('entity_recall', 0):.4f}")
    print(f"  Keyword recall      : {agg.get('keyword_recall', 0):.4f}")
    print(f"  Relationship recall : {agg.get('relationship_recall', 0):.4f}")
    print(f"  Source coverage     : {agg.get('source_coverage', 0):.4f}")
    print(f"  Avg confidence      : {agg.get('avg_confidence', 0):.4f}")
    print(f"  Avg iterations      : {agg.get('avg_iterations', 0):.2f}")
    print(f"  Pass rate           : {agg.get('pass_rate', 0):.4f}")
    print(f"  Error rate          : {agg.get('error_rate', 0):.4f}")
    print()

    # By category
    by_cat = results.get("by_category", {})
    if by_cat:
        print("  BY CATEGORY")
        print("  " + "-" * 68)
        header = f"  {'Category':<30} {'EntR':>6} {'KwR':>6} {'RelR':>6} {'Pass':>6}"
        print(header)
        print("  " + "-" * 68)
        for cat, metrics in by_cat.items():
            print(
                f"  {cat:<30} "
                f"{metrics.get('entity_recall', 0):>6.3f} "
                f"{metrics.get('keyword_recall', 0):>6.3f} "
                f"{metrics.get('relationship_recall', 0):>6.3f} "
                f"{metrics.get('pass_rate', 0):>6.3f}"
            )
        print()

    # By difficulty
    by_diff = results.get("by_difficulty", {})
    if by_diff:
        print("  BY DIFFICULTY")
        print("  " + "-" * 68)
        header = f"  {'Difficulty':<30} {'EntR':>6} {'KwR':>6} {'AvgIt':>6} {'Pass':>6}"
        print(header)
        print("  " + "-" * 68)
        for diff, metrics in by_diff.items():
            print(
                f"  {diff:<30} "
                f"{metrics.get('entity_recall', 0):>6.3f} "
                f"{metrics.get('keyword_recall', 0):>6.3f} "
                f"{metrics.get('avg_iterations', 0):>6.2f} "
                f"{metrics.get('pass_rate', 0):>6.3f}"
            )
        print()

    # Errors
    per_q = results.get("per_question", [])
    errors = [q for q in per_q if q.get("error")]
    if errors:
        print(f"  ERRORS ({len(errors)} questions failed)")
        print("  " + "-" * 68)
        for q in errors[:10]:
            print(f"  Q{q['id']}: {q['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()

    print("=" * 72)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BioKG-Agent against the 100-question evaluation benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Path to benchmark.json. Defaults to eval/benchmark.json relative to this script.",
    )
    parser.add_argument(
        "--output",
        default="eval/results.json",
        help="Path to write results JSON (default: eval/results.json).",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit the number of questions to evaluate.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Only evaluate questions from this category.",
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        choices=["easy", "medium", "hard"],
        help="Only evaluate questions of this difficulty.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to partial results.json to resume from.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory for the agent.",
    )
    parser.add_argument(
        "--live-apis",
        action="store_true",
        help="Enable live API lookups (STRING, NCBI, UniProt, KEGG).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-question progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Resolve benchmark path
    if args.benchmark:
        benchmark_path = args.benchmark
    else:
        benchmark_path = str(Path(__file__).parent / "benchmark.json")

    print(f"Loading benchmark from {benchmark_path} ...")
    benchmark = _load_benchmark(benchmark_path)
    questions = benchmark.get("questions", [])

    # Filter by category
    if args.category:
        questions = [q for q in questions if q.get("category") == args.category]
        if not questions:
            print(f"ERROR: No questions found for category '{args.category}'.")
            print(f"Available categories: {list(benchmark.get('categories', {}).keys())}")
            return 1

    # Filter by difficulty
    if args.difficulty:
        questions = [q for q in questions if q.get("difficulty") == args.difficulty]
        if not questions:
            print(f"ERROR: No questions found for difficulty '{args.difficulty}'.")
            return 1

    # Limit count
    if args.max_questions is not None:
        questions = questions[: args.max_questions]

    # Resume support: load already-completed question IDs
    completed: dict[int, dict[str, Any]] = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from {resume_path} ...")
            with open(resume_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            for q in prev.get("per_question", []):
                completed[q["id"]] = q
            print(f"  Found {len(completed)} previously completed questions.")

    # Build agent
    print("Building BioKG-Agent ...")
    from biokg_agent.agent import BioKGAgent
    from biokg_agent.config import ProjectConfig

    config = ProjectConfig()
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.live_apis:
        config.enable_live_apis = True

    agent = BioKGAgent.build(
        config=config,
        checkpoint_dir=config.checkpoint_dir,
        save_checkpoints=False,
        demo_mode=True,
    )
    if args.live_apis:
        agent.config.enable_live_apis = True

    print(f"Evaluating {len(questions)} questions ...")
    print()

    per_question: list[dict[str, Any]] = []
    total = len(questions)
    start_time = time.time()

    for idx, question in enumerate(questions):
        qid = question["id"]

        # Skip if already completed (resume)
        if qid in completed:
            per_question.append(completed[qid])
            if not args.quiet:
                print(
                    _progress_bar(idx + 1, total, extra=f"Q{qid} [resumed]")
                )
            continue

        # Run evaluation
        if not args.quiet:
            sys.stdout.write(
                "\r" + _progress_bar(idx, total, extra=f"Q{qid} running...")
            )
            sys.stdout.flush()

        result = _evaluate_question(agent, question)
        per_question.append(result)

        if not args.quiet:
            status = "PASS" if result["pass"] else "FAIL"
            if result.get("error"):
                status = "ERR "
            extra = (
                f"Q{qid} {status} "
                f"ent={result['entity_recall']:.2f} "
                f"kw={result['keyword_recall']:.2f} "
                f"conf={result['confidence']:.2f} "
                f"({result['elapsed_seconds']:.1f}s)"
            )
            print("\r" + _progress_bar(idx + 1, total, extra=extra))

        # Save partial results after each question for resume support
        categories_seen = sorted(set(q.get("category", "") for q in per_question))
        partial_results = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "num_questions": len(per_question),
                "categories_evaluated": categories_seen,
                "config": {
                    "live_apis": args.live_apis,
                    "category_filter": args.category,
                    "difficulty_filter": args.difficulty,
                    "max_questions": args.max_questions,
                    "benchmark_path": benchmark_path,
                },
                "status": "in_progress",
            },
            "aggregate_metrics": _aggregate_metrics(per_question),
            "by_category": _group_metrics(per_question, "category"),
            "by_difficulty": _group_metrics(per_question, "difficulty"),
            "per_question": per_question,
        }
        _save_results(partial_results, args.output)

    elapsed_total = round(time.time() - start_time, 1)

    # Build final results
    categories_evaluated = sorted(set(q.get("category", "") for q in per_question))
    results: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_questions": len(per_question),
            "categories_evaluated": categories_evaluated,
            "total_elapsed_seconds": elapsed_total,
            "config": {
                "live_apis": args.live_apis,
                "category_filter": args.category,
                "difficulty_filter": args.difficulty,
                "max_questions": args.max_questions,
                "benchmark_path": benchmark_path,
            },
            "status": "complete",
        },
        "aggregate_metrics": _aggregate_metrics(per_question),
        "by_category": _group_metrics(per_question, "category"),
        "by_difficulty": _group_metrics(per_question, "difficulty"),
        "per_question": per_question,
    }
    _save_results(results, args.output)

    # Print summary
    _print_summary_table(results)
    print(f"  Results saved to: {Path(args.output).resolve()}")
    print(f"  Total time: {elapsed_total:.1f}s")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
