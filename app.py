"""Gradio entrypoint for BioKG-Agent.

Layout  : Left = chat + tool badges + answer  |  Right = Plotly network graph
Graph   : Plotly (bundled with Gradio — no CDN, no CSP issues, always renders)
Subgraph: ONLY seed genes + top-5 STRING partners + top-5 drugs + top-3 pathways
          → typically 15-30 nodes, clean and readable like the reference design
"""

from __future__ import annotations

import os

import json
import math
import re
import socket
import time
from collections import defaultdict
from pathlib import Path

try:
    import gradio as gr
except Exception as exc:
    gr = None
    _GRADIO_ERROR = exc
else:
    _GRADIO_ERROR = None

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from biokg_agent.agent import BioKGAgent
from biokg_agent.config import ProjectConfig

# ---------------------------------------------------------------------------
# Plotly graph builder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Graph visual config
# ---------------------------------------------------------------------------

_TYPE_COLOR = {
    "gene":        "#00d4aa",   # teal-green — main gene nodes
    "drug":        "#f39c12",   # vivid amber — drugs pop out
    "pathway":     "#3498db",   # blue
    "go_term":     "#9b59b6",   # purple
    "protein":     "#e74c3c",   # red
    "publication": "#7f8c8d",
}
# Seed genes get a special bright colour added in _build_plotly_graph
_SEED_COLOR   = "#00ff88"
_SEED_SIZE    = 34

_TYPE_SIZE = {
    "gene": 22, "drug": 18, "pathway": 16,
    "go_term": 15, "protein": 16, "publication": 11,
}
_TYPE_SYMBOL = {
    "gene": "circle", "drug": "square", "pathway": "diamond",
    "go_term": "triangle-up", "protein": "circle-open", "publication": "x",
}

# Edge colour by relationship label
_EDGE_COLOR = {
    "INTERACTS_WITH":  "rgba(0,212,170,0.55)",   # teal
    "TARGETS":         "rgba(243,156,18,0.65)",   # amber
    "IN_PATHWAY":      "rgba(52,152,219,0.55)",   # blue
    "PARTICIPATES_IN": "rgba(52,152,219,0.55)",
    "REGULATES":       "rgba(155,89,182,0.60)",   # purple
    "ASSOCIATED_WITH": "rgba(149,165,166,0.40)",
}
_EDGE_WIDTH = {
    "INTERACTS_WITH": 1.8, "TARGETS": 2.2, "IN_PATHWAY": 1.4,
    "PARTICIPATES_IN": 1.4, "REGULATES": 1.6, "ASSOCIATED_WITH": 1.0,
}
_DEFAULT_EDGE_COLOR = "rgba(120,120,140,0.30)"
_DEFAULT_EDGE_WIDTH = 1.0


def _concentric_layout(nodes: list, edges: list,
                       seed_genes: set) -> dict[str, tuple[float, float]]:
    """Four-ring concentric layout.

    Ring 0 (centre): query / seed genes
    Ring 1 (r=0.90): STRING partner genes
    Ring 2 (r=1.75): drugs
    Ring 3 (r=2.55): GO terms / pathways
    """
    gene_ids  = {n["id"] for n in nodes if n.get("title") == "gene"}
    drug_ids  = {n["id"] for n in nodes if n.get("title") == "drug"}
    other_ids = {n["id"] for n in nodes
                 if n.get("title") not in ("gene", "drug")}

    partner_genes = gene_ids - seed_genes
    pos: dict[str, tuple[float, float]] = {}

    def _ring(items: list, radius: float, phase: float = 0.0) -> None:
        n = len(items)
        for i, nid in enumerate(sorted(items)):
            a = 2 * math.pi * i / max(n, 1) + phase
            pos[nid] = (radius * math.cos(a), radius * math.sin(a))

    seeds_list = sorted(seed_genes)
    n_s = len(seeds_list)
    for i, nid in enumerate(seeds_list):
        a = 2 * math.pi * i / max(n_s, 1)
        r = 0.0 if n_s == 1 else 0.28
        pos[nid] = (r * math.cos(a), r * math.sin(a))

    _ring(list(partner_genes), 0.90, phase=-math.pi / 2)
    _ring(list(drug_ids),      1.75, phase= math.pi / 8)
    _ring(list(other_ids),     2.55, phase=-math.pi / 8)
    return pos


def _build_plotly_graph(data_path: str | Path | None):
    """Return a polished Plotly Figure from the JSON sidecar."""
    if not _HAS_PLOTLY:
        return None
    if not data_path:
        return None

    p = Path(str(data_path))
    if p.suffix == ".html":
        p = p.with_suffix(".json")
    if not p.exists():
        return None

    try:
        raw   = json.loads(p.read_text(encoding="utf-8"))
        nodes: list = raw.get("nodes", [])
        edges: list = raw.get("edges", [])
    except Exception:
        return None
    if not nodes:
        return None

    # ── identify seed genes (highest degree) ─────────────────────────────
    degree: dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        degree[e["from"]] = degree.get(e["from"], 0) + 1
        degree[e["to"]]   = degree.get(e["to"],   0) + 1

    gene_ids = [n["id"] for n in nodes if n.get("title") == "gene"]
    gene_ids_sorted = sorted(gene_ids, key=lambda x: -degree.get(x, 0))
    n_seeds  = max(1, len(gene_ids_sorted) // 5)
    seed_set = set(gene_ids_sorted[:n_seeds])

    pos      = _concentric_layout(nodes, edges, seed_set)
    node_ids = {n["id"] for n in nodes}

    # ── edge traces — one trace per relationship type ─────────────────────
    edge_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for e in edges:
        if e["from"] not in node_ids or e["to"] not in node_ids:
            continue
        lbl = e.get("label", "")
        x0, y0 = pos.get(e["from"], (0, 0))
        x1, y1 = pos.get(e["to"],   (0, 0))
        edge_groups[lbl][0].extend([x0, x1, None])
        edge_groups[lbl][1].extend([y0, y1, None])

    edge_traces = []
    for lbl, (ex, ey) in edge_groups.items():
        color = _EDGE_COLOR.get(lbl, _DEFAULT_EDGE_COLOR)
        width = _EDGE_WIDTH.get(lbl, _DEFAULT_EDGE_WIDTH)
        edge_traces.append(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="none", showlegend=False,
        ))

    # ── node traces — seeds separate, then by type ────────────────────────
    by_type: dict[str, list] = defaultdict(list)
    seed_nodes, non_seed_genes = [], []
    for node in nodes:
        ntype = node.get("title", "gene")
        if node["id"] in seed_set:
            seed_nodes.append(node)
        elif ntype == "gene":
            non_seed_genes.append(node)
        else:
            by_type[ntype].append(node)

    node_traces = []

    # Seed gene trace — large glowing nodes
    if seed_nodes:
        xs = [pos.get(n["id"], (0, 0))[0] for n in seed_nodes]
        ys = [pos.get(n["id"], (0, 0))[1] for n in seed_nodes]
        labels = [n.get("label", n["id"]) for n in seed_nodes]
        hover  = [
            f"<b>{n.get('label', n['id'])}</b><br>"
            f"Query gene · {degree.get(n['id'], 0)} connections"
            for n in seed_nodes
        ]
        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color="#00ff88", family="monospace"),
            marker=dict(
                size=_SEED_SIZE,
                color=_SEED_COLOR,
                symbol="circle",
                line=dict(width=3, color="#ffffff"),
                opacity=0.95,
            ),
            name="query gene",
            hovertext=hover,
            hoverinfo="text",
        ))

    # Partner gene trace
    if non_seed_genes:
        xs = [pos.get(n["id"], (0, 0))[0] for n in non_seed_genes]
        ys = [pos.get(n["id"], (0, 0))[1] for n in non_seed_genes]
        labels = [n.get("label", n["id"]) for n in non_seed_genes]
        hover  = [
            f"<b>{n.get('label', n['id'])}</b><br>"
            f"Partner gene · {degree.get(n['id'], 0)} connections"
            for n in non_seed_genes
        ]
        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=9, color="#a8e6cf"),
            marker=dict(
                size=_TYPE_SIZE["gene"],
                color=_TYPE_COLOR["gene"],
                symbol="circle",
                line=dict(width=1.5, color="#0a3a2a"),
                opacity=0.85,
            ),
            name="gene",
            hovertext=hover,
            hoverinfo="text",
        ))

    # Drug / GO / pathway traces
    for ntype, group in by_type.items():
        color  = _TYPE_COLOR.get(ntype, "#888")
        size   = _TYPE_SIZE.get(ntype, 14)
        symbol = _TYPE_SYMBOL.get(ntype, "circle")
        xs = [pos.get(n["id"], (0, 0))[0] for n in group]
        ys = [pos.get(n["id"], (0, 0))[1] for n in group]

        # Smart label: drugs keep full name (short), GO terms get last meaningful word(s)
        def _smart_label(n: dict, nt: str) -> str:
            raw = n.get("label", n["id"])
            if nt == "drug":
                return raw[:18]          # drug names are short, keep them
            if nt in ("go_term", "pathway"):
                # Strip generic prefixes like "regulation of", "positive regulation of"
                cleaned = re.sub(
                    r'^(positive |negative )?(regulation of |biosynthesis of |'
                    r'metabolic process|cellular )?', '', raw, flags=re.I
                ).strip()
                return (cleaned[:20] + "…") if len(cleaned) > 20 else (cleaned or raw[:18])
            return raw[:18]

        labels = [_smart_label(n, ntype) for n in group]
        hover  = [
            f"<b>{n.get('label', n['id'])}</b><br>"
            f"<span style='color:#aaa'>{ntype}</span>"
            for n in group
        ]
        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(
                size=9 if ntype == "drug" else 8,
                color="#f6c90e" if ntype == "drug" else "#c9b8f0",
            ),
            marker=dict(
                size=size, color=color, symbol=symbol,
                line=dict(width=1.5, color="#111"),
                opacity=0.90,
            ),
            name=ntype,
            hovertext=hover,
            hoverinfo="text",
        ))

    fig = go.Figure(edge_traces + node_traces)

    # Title = seed gene names | subtitle = edge breakdown
    seed_names = " · ".join(sorted(seed_set)[:4])
    etype_counts = {k: len(v[0]) // 3 for k, v in edge_groups.items() if v[0]}
    edge_parts   = [f"{v} {k.replace('_',' ').lower()}" for k, v in etype_counts.items()]
    edge_summary = " · ".join(edge_parts[:3]) if edge_parts else ""

    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=11),
        title=dict(
            text=(
                f"<b style='color:#00ff88'>{seed_names}</b>"
                f"<span style='color:#8b949e;font-size:11px;'> — interaction network</span>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=13, color="#c9d1d9"),
        ),
        legend=dict(
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#c9d1d9", size=10),
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1.0,
            itemsizing="constant",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x"),   # equal aspect ratio keeps circles round
        dragmode="pan",                # default drag = pan (intuitive for graphs)
        margin=dict(l=10, r=10, t=60, b=30),
        height=580,
        annotations=[
            dict(
                text=(
                    f"<b>{len(nodes)}</b> nodes · <b>{len(edges)}</b> edges"
                    + (f"  ({edge_summary})" if edge_summary else "")
                    + "  — hover any node for details"
                ),
                xref="paper", yref="paper", x=0.5, y=-0.04,
                showarrow=False,
                font=dict(size=10, color="#8b949e"),
                align="center",
            )
        ],
    )
    return fig


_PLACEHOLDER_FIG = None   # returned before first query


# ---------------------------------------------------------------------------
# Chat HTML helpers
# ---------------------------------------------------------------------------

_TOOL_COLORS: dict[str, str] = {
    "ncbi_gene_lookup":       "#1a6b9c",
    "string_interactions":    "#1a6b3c",
    "drugbank_target_lookup": "#6b1a4b",
    "gene_ontology_lookup":   "#6b5c1a",
    "pathway_lookup":         "#3c1a6b",
    "uniprot_protein_lookup": "#1a4b6b",
    "kegg_pathway_lookup":    "#6b3c1a",
}


def _tool_badge(name: str, args: str) -> str:
    color = _TOOL_COLORS.get(name, "#2c4a6b")
    return (
        f'<span style="display:inline-block;background:{color};color:#fff;'
        f'padding:3px 14px;border-radius:16px;font-size:12.5px;'
        f'font-family:Consolas,monospace;margin:3px 5px 3px 0;">'
        f'Tool: {name}(<span style="color:#a8d8ea">{args}</span>)</span>'
    )


def _infer_tool_calls(payload: dict) -> list[tuple[str, str]]:
    tools: list[tuple[str, str]] = []
    entities = payload.get("query_plan", {}).get("detected_entities", [])
    for g in entities[:3]:
        tools.append(("ncbi_gene_lookup", f'"{g}"'))
    if entities:
        tools.append(("string_interactions", f'"{entities[0]}"'))
    ev = payload.get("evidence_table", [])
    dg = list(dict.fromkeys(
        e.get("gene", "") for e in ev
        if e.get("source_type") == "drugbank" and e.get("gene")
    ))
    for g in dg[:2]:
        tools.append(("drugbank_target_lookup", f'"{g}"'))
    return tools


def _esc(s: str) -> str:
    """Minimal HTML escape."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _section(title: str, icon: str, color: str, body: str) -> str:
    return (
        f'<div style="margin-bottom:14px;border-radius:8px;overflow:hidden;">'
        f'<div style="background:{color};padding:7px 14px;font-size:12px;'
        f'font-weight:600;color:#fff;letter-spacing:.5px;">{icon}&nbsp; {title}</div>'
        f'<div style="background:#0d1b2a;padding:10px 14px;font-size:13.5px;'
        f'line-height:1.8;color:#d8eaf8;">{body}</div>'
        f'</div>'
    )


def _highlight(text: str, genes: list[str]) -> str:
    for g in sorted(genes, key=len, reverse=True):
        text = re.sub(r'\b' + re.escape(g) + r'\b',
                      f'<b style="color:#7ec8e3">{g}</b>', text)
    return text


def _build_chat_html(query: str, payload: dict) -> str:
    entities    = payload.get("query_plan", {}).get("detected_entities", [])
    expansions  = payload.get("expansions", [])
    lit_titles  = payload.get("lit_titles", [])
    lit_count   = payload.get("lit_count", 0)
    answer_text = payload.get("answer_text", "")
    conf        = float(payload.get("confidence_summary", {}).get("overall_confidence", 0))
    conf_detail = payload.get("confidence_summary", {})
    route       = payload.get("route_type", "hybrid")
    channels    = payload.get("retrieval_channels", [])
    badges      = "".join(_tool_badge(n, a) for n, a in _infer_tool_calls(payload))
    conf_color  = "#27ae60" if conf >= 0.70 else "#e67e22" if conf >= 0.50 else "#e74c3c"

    # ── Section 0: Narrative answer (most important) ──────────────────────
    # Split LLM output on blank lines → paragraphs; single newlines stay inline
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', answer_text) if p.strip()]
    if not paragraphs:
        paragraphs = [answer_text.strip()] if answer_text.strip() else []

    answer_html = "".join(
        f'<p style="margin:0 0 10px 0;color:#e2e8f0;font-size:13.5px;'
        f'line-height:1.75;padding-left:12px;border-left:2px solid #2563eb;">'
        f'{_highlight(_esc(p), entities)}</p>'
        for p in paragraphs
    ) if paragraphs else '<span style="color:#94a3b8">No answer generated.</span>'

    answer_section = _section("SUMMARY", "💡", "#1e40af", answer_html)

    # ── Section 1: Drugs — split approved vs clinical ─────────────────────
    approved_rows = []
    clinical_rows = []
    for exp in expansions:
        gene = exp.get("gene", "")
        drugs = exp.get("drugs", [])
        for d in drugs:
            name = _esc(d.get("drug_name", ""))
            if not name:
                continue
            mech = _esc(d.get("mechanism", f"targets {gene}"))
            statuses = [s.lower() for s in d.get("status", [])]
            status_str = ", ".join(d.get("status", ["investigational"]))
            if "approved" in statuses:
                approved_rows.append(
                    f'<div style="padding:5px 0;border-bottom:1px solid #1a2a40;">'
                    f'<b style="color:#c084fc;font-size:13.5px">{name}</b>'
                    f'<span style="background:#166534;color:#86efac;font-size:10px;'
                    f'padding:1px 6px;border-radius:8px;margin-left:8px;">FDA APPROVED</span>'
                    f'<span style="color:#475569;font-size:11px;margin-left:6px;">'
                    f'targets {_esc(gene)}</span>'
                    f'<br><span style="color:#94a3b8;font-size:12px;">{mech}</span>'
                    f'</div>'
                )
            else:
                # only show clinical-stage if it's a named compound
                phase_label = status_str.title()
                clinical_rows.append(
                    f'<div style="padding:5px 0;border-bottom:1px solid #1a2a40;">'
                    f'<b style="color:#94a3b8;font-size:13px">{name}</b>'
                    f'<span style="background:#1e293b;color:#64748b;font-size:10px;'
                    f'padding:1px 6px;border-radius:8px;margin-left:8px;">'
                    f'{_esc(phase_label)}</span>'
                    f'<span style="color:#374151;font-size:11px;margin-left:6px;">'
                    f'targets {_esc(gene)}</span>'
                    f'</div>'
                )

    drug_section = ""
    if approved_rows:
        drug_section = _section(
            "FDA-APPROVED TARGETED THERAPIES", "💊", "#6b21a8",
            "".join(approved_rows[:6])
        )
        if clinical_rows:
            drug_section += _section(
                "CLINICAL-STAGE COMPOUNDS", "🔬", "#1e3a5f",
                "".join(clinical_rows[:4])
            )
    elif clinical_rows:
        drug_section = _section(
            "CLINICAL-STAGE COMPOUNDS (no FDA-approved direct targets)", "🔬", "#1e3a5f",
            "".join(clinical_rows[:4])
        )
    else:
        drug_section = _section(
            "DRUG EVIDENCE", "💊", "#1f2937",
            '<span style="color:#4b5563;font-size:13px;">No direct drug targets found '
            'in curated database. Query may refer to a pathway, not a single druggable target.</span>'
        )

    # ── Section 2: Protein interactions ──────────────────────────────────
    iact_rows = []
    seen_pairs: set[tuple] = set()
    for exp in expansions:
        gene = exp.get("gene", "")
        for iact in sorted(exp.get("interactions", []),
                           key=lambda x: -int(x.get("score", 0)))[:4]:
            partner = _esc(iact.get("partner", ""))
            pair = tuple(sorted([gene, iact.get("partner","")]))
            if not partner or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            score     = int(iact.get("score", 0))
            bar_pct   = min(100, score // 10)
            bar_color = "#22c55e" if score >= 900 else "#eab308" if score >= 700 else "#f97316"
            quality   = "High confidence" if score >= 900 else "Medium confidence" if score >= 700 else "Low confidence"
            iact_rows.append(
                f'<div style="padding:5px 0;border-bottom:1px solid #1a2a40;">'
                f'<b style="color:#7dd3fc">{_esc(gene)}</b>'
                f'<span style="color:#475569;margin:0 6px;">interacts with</span>'
                f'<b style="color:#7dd3fc">{partner}</b>'
                f'<span style="float:right;font-size:11px;color:#64748b;">'
                f'{quality} &bull; {score}/1000</span>'
                f'<div style="height:3px;background:#1e293b;border-radius:2px;margin-top:5px;">'
                f'<div style="height:3px;width:{bar_pct}%;background:{bar_color};border-radius:2px;">'
                f'</div></div>'
                f'</div>'
            )

    iact_section = ""
    if iact_rows:
        iact_section = _section(
            "PROTEIN-PROTEIN INTERACTIONS (STRING DB)", "🔗", "#1e3a5f",
            "".join(iact_rows[:8])
        )

    # ── Section 3: Pathways & GO terms ────────────────────────────────────
    pw_chips: list[str] = []
    go_chips: list[str] = []
    seen_pw: set[str] = set()
    for exp in expansions:
        for pw in exp.get("pathways", [])[:5]:
            name = pw.get("name", pw.get("pathway_id", ""))
            if name and name not in seen_pw:
                seen_pw.add(name)
                pw_chips.append(
                    f'<span style="display:inline-block;background:#1e3a5f;color:#93c5fd;'
                    f'padding:4px 12px;border-radius:14px;font-size:12px;margin:3px;">'
                    f'{_esc(name)}</span>'
                )
        for term in exp.get("go_terms", []):
            if term.get("namespace") in ("biological_process", "molecular_function"):
                t = term.get("name", term.get("id", ""))
                if t and t not in seen_pw:
                    seen_pw.add(t)
                    color = "#86efac" if term.get("namespace") == "biological_process" else "#fde68a"
                    go_chips.append(
                        f'<span style="display:inline-block;background:#1e2a3a;color:{color};'
                        f'padding:4px 12px;border-radius:14px;font-size:12px;margin:3px;">'
                        f'{_esc(t)}</span>'
                    )

    pathway_section = ""
    all_chips = pw_chips[:4] + go_chips[:4]
    if all_chips:
        legend = (
            '<div style="font-size:11px;color:#475569;margin-bottom:6px;">'
            '<span style="color:#93c5fd">&#9632;</span> pathway &nbsp;'
            '<span style="color:#86efac">&#9632;</span> biological process &nbsp;'
            '<span style="color:#fde68a">&#9632;</span> molecular function</div>'
        )
        pathway_section = _section(
            "PATHWAYS & BIOLOGICAL PROCESSES", "🧬", "#14532d",
            legend + "".join(all_chips)
        )

    # ── Section 4: Literature ─────────────────────────────────────────────
    lit_rows = []
    for i, title in enumerate(lit_titles[:6], 1):
        lit_rows.append(
            f'<div style="padding:5px 0;border-bottom:1px solid #1a2a40;">'
            f'<span style="color:#475569;font-size:11px;font-weight:600;'
            f'margin-right:8px;">[{i}]</span>'
            f'<span style="color:#cbd5e1;font-size:13px;">'
            f'{_esc(title[:110])}{"..." if len(title) > 110 else ""}</span>'
            f'</div>'
        )
    if lit_rows:
        note = (f'<div style="color:#374151;font-size:11px;margin-top:8px;">'
                f'Searched <b style="color:#475569">{lit_count:,}</b> PubMed records '
                f'using FAISS semantic search + BM25</div>')
        lit_section = _section(
            "SUPPORTING LITERATURE (PubMed)", "📚", "#1e3a5f",
            "".join(lit_rows) + note
        )
    else:
        lit_section = ""

    # ── Footer: confidence breakdown ──────────────────────────────────────
    lit_conf   = int(conf_detail.get("literature_confidence", 0) * 100)
    graph_conf = int(conf_detail.get("graph_confidence", 0) * 100)
    drug_conf  = int(conf_detail.get("drug_confidence", 0) * 100)
    conf_bar   = int(conf * 100)

    def _mini_bar(pct: int, col: str) -> str:
        return (f'<div style="display:inline-block;width:50px;height:5px;vertical-align:middle;'
                f'background:#1e293b;border-radius:3px;margin:0 5px;">'
                f'<div style="height:5px;width:{pct}%;background:{col};border-radius:3px;">'
                f'</div></div>')

    conf_footer = (
        f'<div style="margin-top:12px;padding-top:10px;border-top:1px solid #1e2a3a;'
        f'font-size:11.5px;color:#475569;display:flex;flex-wrap:wrap;gap:10px 18px;">'
        f'<span>Route: <b style="color:#8ab4f8">{_esc(route)}</b></span>'
        f'<span>Sources: <b style="color:#a8d8ea">{_esc(", ".join(channels))}</b></span>'
        f'<span>Overall: <b style="color:{conf_color}">{conf_bar}%</b>'
        f'{_mini_bar(conf_bar, conf_color)}</span>'
        f'<span>Literature{_mini_bar(lit_conf,"#60a5fa")}{lit_conf}%</span>'
        f'<span>Graph{_mini_bar(graph_conf,"#34d399")}{graph_conf}%</span>'
        f'<span>Drugs{_mini_bar(drug_conf,"#c084fc")}{drug_conf}%</span>'
        f'<span style="color:#374151;font-size:10.5px;">'
        f'Confidence = weighted average of retrieval quality, evidence coverage, '
        f'and source diversity</span>'
        f'</div>'
    )

    return (
        f'<div style="background:#16213e;padding:18px 20px;border-radius:12px;'
        f'font-family:\'Segoe UI\',Arial,sans-serif;color:#e8e8e8;">'
        # Query bubble
        f'<div style="background:#0f3460;border-radius:10px 10px 10px 2px;'
        f'padding:12px 16px;margin-bottom:14px;font-size:14px;'
        f'color:#e8e8e8;line-height:1.5;font-style:italic;">{_esc(query)}</div>'
        # Tool badges
        f'<div style="margin-bottom:14px;line-height:2.4;">{badges}</div>'
        # Sections (answer first, then supporting data)
        + answer_section
        + drug_section
        + iact_section
        + pathway_section
        + lit_section
        + conf_footer
        + '</div>'
    )


# ---------------------------------------------------------------------------
# Agent cache — pre-warmed at startup
# ---------------------------------------------------------------------------
_AGENT: BioKGAgent | None = None


def _get_agent() -> BioKGAgent:
    global _AGENT
    if _AGENT is None:
        cfg = ProjectConfig.from_env()
        _AGENT = BioKGAgent.build(config=cfg)
        if cfg.enable_llm_synthesis:
            try:
                from biokg_agent.llm import create_llm_backend
                llm = create_llm_backend(
                    model_id=cfg.llm_model_id, device=cfg.llm_device,
                    groq_api_key=cfg.groq_api_key, groq_model=cfg.groq_model,
                    backend=cfg.llm_backend,
                )
                _AGENT.attach_llm(llm)
                print("[app] LLM attached:", cfg.llm_backend)
            except Exception as e:
                print(f"[app] LLM unavailable ({e}) — using template answers")
    return _AGENT


# ---------------------------------------------------------------------------
_EXAMPLES = [
    "What kinases phosphorylate BRCA1 that are targets of cancer drugs?",
    "What drugs target EGFR and which pathways are involved?",
    "How does TP53 regulate apoptosis and what drugs target this pathway?",
    "What is the relationship between KRAS and downstream signaling?",
    "Which approved drugs target BRAF mutations in cancer?",
]

# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app():
    if gr is None:
        raise RuntimeError(f"Gradio not available: {_GRADIO_ERROR}")

    def respond(query: str):
        if not query or not query.strip():
            return (
                '<div style="color:#888;padding:20px;text-align:center;">'
                'Please enter a question.</div>',
                _PLACEHOLDER_FIG,
            )
        agent = _get_agent()
        payload = agent.invoke(query.strip())
        chat_html = _build_chat_html(query.strip(), payload)
        fig = _build_plotly_graph(payload.get("graph_html", ""))
        return chat_html, fig

    with gr.Blocks(title="BioKG-Agent") as demo:
        gr.Markdown(
            "# 🔬 BioKG-Agent\n"
            "Multi-hop biological reasoning · PubMed · STRING PPI · "
            "ChEMBL drugs · Gene Ontology",
        )

        with gr.Row():
            query_box = gr.Textbox(
                label="Ask a biology question",
                placeholder="What kinases phosphorylate BRCA1 that are targets of cancer drugs?",
                lines=2,
            )
            run_btn = gr.Button("▶ Run", variant="primary", scale=0, min_width=100)

        gr.Examples(
            examples=[[q] for q in _EXAMPLES],
            inputs=[query_box],
            label="Example queries",
            examples_per_page=5,
        )

        with gr.Row():
            chat_out = gr.HTML(
                value='<div style="background:#16213e;color:#555;padding:60px;'
                      'text-align:center;border-radius:12px;min-height:540px;'
                      'display:flex;align-items:center;justify-content:center;">'
                      '<span>Enter a query above and click ▶ Run</span></div>',
                label="Answer",
            )
            graph_out = gr.Plot(label="Knowledge Graph")

        run_btn.click(fn=respond, inputs=[query_box], outputs=[chat_out, graph_out])
        query_box.submit(fn=respond, inputs=[query_box], outputs=[chat_out, graph_out])

    return demo


# ---------------------------------------------------------------------------

def _find_free_port(start=7860, end=7869):
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return start  # fallback


def main() -> None:
    cfg = ProjectConfig.from_env()
    port = int(os.environ.get("GRADIO_SERVER_PORT", 0)) or _find_free_port()

    print("[app] Pre-warming agent ...")
    t0 = time.time()
    _get_agent()
    print(f"[app] Agent ready in {time.time()-t0:.1f}s")

    demo = build_app()

    if cfg.ngrok_auth_token:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(cfg.ngrok_auth_token)
            tunnel = ngrok.connect(port)
            print(f"\n  Ngrok URL: {tunnel.public_url}\n")
        except Exception as e:
            print(f"[warn] Ngrok: {e}")

    demo.launch(server_port=port, inbrowser=True, share=cfg.gradio_share)


if __name__ == "__main__":
    main()
