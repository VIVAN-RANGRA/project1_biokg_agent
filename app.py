"""Gradio entrypoint for the BioKG-Agent demo."""

from __future__ import annotations

import json

try:  # pragma: no cover - optional runtime dependency
    import gradio as gr
except Exception as exc:  # pragma: no cover - optional runtime dependency
    gr = None
    _GRADIO_ERROR = exc
else:
    _GRADIO_ERROR = None

from biokg_agent.agent import BioKGDemoAgent
from biokg_agent.config import default_config


def build_app():
    if gr is None:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(f"Gradio is not available: {_GRADIO_ERROR}")

    agent = BioKGDemoAgent(config=default_config())

    def respond(query: str):
        payload = agent.invoke(query)
        evidence = json.dumps(payload.get("evidence_table", []), indent=2)
        summary = json.dumps(
            {
                "route_type": payload.get("route_type"),
                "retrieval_channels": payload.get("retrieval_channels"),
                "retrieval_iterations": payload.get("retrieval_iterations_count"),
                "confidence": payload.get("confidence_summary", {}),
            },
            indent=2,
        )
        return payload["answer_text"], evidence, payload.get("graph_html", ""), summary

    with gr.Blocks(title="BioKG-Agent Demo") as demo:
        gr.Markdown("# BioKG-Agent\nCPU-safe biological reasoning over a demo knowledge graph.")
        query = gr.Textbox(label="Ask a biology question", placeholder="What drugs target EGFR and which pathways connect to them?")
        run = gr.Button("Run query")
        answer = gr.Textbox(label="Answer", lines=5)
        evidence = gr.Code(label="Evidence table", language="json")
        graph = gr.HTML(label="Knowledge graph")
        summary = gr.Code(label="Route and confidence", language="json")
        run.click(respond, inputs=[query], outputs=[answer, evidence, graph, summary])
    return demo


def main() -> None:
    demo = build_app()
    cfg = default_config()
    demo.launch(share=cfg.gradio_share, server_port=cfg.gradio_port)


if __name__ == "__main__":  # pragma: no cover
    main()
