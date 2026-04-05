"""Unified runner for BioKG-Agent demo queries and smoke evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biokg_agent.agent import BioKGDemoAgent
from biokg_agent.config import ProjectConfig


SMOKE_CASES = [
    {
        "id": "tp53-pathway-drugs",
        "query": "What drugs target TP53 pathway proteins?",
        "must_contain": ["tp53", "confidence"],
        "expected_route": {"relationship", "hybrid", "mechanistic"},
        "min_iterations": 1,
        "expected_channels": {"dense", "bm25"},
        "expected_entities": {"TP53"},
    },
    {
        "id": "brca1-literature",
        "query": "Which papers explain BRCA1 and PARP inhibitor resistance?",
        "must_contain": ["literature", "brca1"],
        "expected_route": {"literature", "hybrid"},
        "min_iterations": 1,
        "expected_channels": {"dense", "bm25"},
        "expected_entities": {"BRCA1"},
    },
    {
        "id": "tp53-iterative",
        "query": "What connects TP53 to cancer therapy?",
        "must_contain": ["tp53", "confidence"],
        "expected_route": {"mechanistic", "hybrid"},
        "min_iterations": 2,
        "expected_channels": {"dense", "bm25", "graph"},
        "expected_entities": {"TP53"},
    },
    {
        "id": "il6-alias",
        "query": "What inflammatory signaling partners appear around IL-6?",
        "must_contain": ["il6"],
        "expected_route": {"relationship", "hybrid", "mechanistic"},
        "min_iterations": 1,
        "expected_channels": {"graph"},
        "expected_entities": {"IL6"},
    },
]


def build_agent(checkpoint_dir: str | None = None) -> BioKGDemoAgent:
    config = ProjectConfig()
    if checkpoint_dir:
        config.checkpoint_dir = checkpoint_dir
    return BioKGDemoAgent.build(config=config, checkpoint_dir=config.checkpoint_dir, save_checkpoints=True, demo_mode=True)


def run_query(agent: BioKGDemoAgent, query: str) -> dict:
    payload = agent.invoke(query)
    result_path = Path(agent.config.checkpoint_dir_path) / "last_query_result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload | {"result_path": str(result_path)}


def run_smoke_eval(agent: BioKGDemoAgent) -> dict:
    cases = []
    for case in SMOKE_CASES:
        payload = agent.invoke(case["query"])
        text = payload["answer_text"].lower()
        contains_expected_terms = all(term in text for term in case["must_contain"])
        route_ok = payload.get("route_type") in case["expected_route"]
        channels = set(payload.get("retrieval_channels", []))
        channels_ok = case["expected_channels"].issubset(channels)
        iterations_ok = payload.get("retrieval_iterations_count", 0) >= case["min_iterations"]
        entities_ok = case["expected_entities"].issubset(set(payload.get("query_plan", {}).get("detected_entities", [])))
        confidence = payload.get("confidence_summary", {}).get("overall_confidence", 0.0)
        confidence_ok = isinstance(confidence, (float, int)) and 0.0 <= float(confidence) <= 1.0
        provenance_ok = bool(payload.get("evidence_table"))
        passed = all([contains_expected_terms, route_ok, channels_ok, iterations_ok, entities_ok, confidence_ok, provenance_ok])
        cases.append(
            {
                "id": case["id"],
                "query": case["query"],
                "pass": passed,
                "answer": payload["answer_text"],
                "route_type": payload.get("route_type"),
                "retrieval_channels": payload.get("retrieval_channels", []),
                "retrieval_iterations_count": payload.get("retrieval_iterations_count", 0),
                "reranker_used": payload.get("reranker_used", False),
                "reranker_fallback_used": payload.get("reranker_fallback_used", False),
                "confidence": payload.get("confidence_summary", {}),
                "graph_summary": payload.get("graph_summary", {}),
                "assertions": {
                    "contains_expected_terms": contains_expected_terms,
                    "route_ok": route_ok,
                    "channels_ok": channels_ok,
                    "iterations_ok": iterations_ok,
                    "entities_ok": entities_ok,
                    "confidence_ok": confidence_ok,
                    "provenance_ok": provenance_ok,
                },
            }
        )
    report = {
        "num_cases": len(cases),
        "pass_rate": sum(1 for case in cases if case["pass"]) / max(len(cases), 1),
        "cases": cases,
    }
    report_path = Path(agent.config.checkpoint_dir_path) / "smoke_eval.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report | {"report_path": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular BioKG-Agent demo without heavy preprocessing.")
    parser.add_argument("--checkpoint-dir", default=None, help="Where to save checkpoints and smoke-eval artifacts.")
    parser.add_argument("--query", default=None, help="Single query to run through the demo agent.")
    parser.add_argument("--smoke-eval", action="store_true", help="Run the tiny CPU-safe smoke benchmark.")
    parser.add_argument("--print-json", action="store_true", help="Print results as JSON instead of short text.")
    parser.add_argument("--live-apis", action="store_true", help="Enable live API lookups where implemented.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agent = build_agent(checkpoint_dir=args.checkpoint_dir)
    if args.live_apis:
        agent.config.enable_live_apis = True

    if args.smoke_eval:
        report = run_smoke_eval(agent)
        if args.print_json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Smoke eval pass rate: {report['pass_rate']:.2f}")
            print(f"Report: {report['report_path']}")
        return 0

    query = args.query or "What drugs target TP53 pathway proteins?"
    result = run_query(agent, query)
    if args.print_json:
        print(json.dumps(result, indent=2))
    else:
        print(result["answer_text"])
        print(f"Saved query result: {result['result_path']}")
        print(f"Graph HTML: {result.get('graph_html', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
