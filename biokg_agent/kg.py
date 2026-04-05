"""NetworkX-backed biological knowledge graph."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Sequence

from .checkpoints import load_pickle, save_pickle

try:  # pragma: no cover - optional dependency
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    nx = None


def _build_graph():
    if nx is None:  # pragma: no cover - fallback path
        class _MiniGraph:
            def __init__(self):
                self.nodes = {}
                self.edges = []

            def add_node(self, node_id, **attrs):
                self.nodes[node_id] = {**self.nodes.get(node_id, {}), **attrs}

            def add_edge(self, source, target, **attrs):
                self.edges.append((source, target, dict(attrs)))

            def number_of_nodes(self):
                return len(self.nodes)

            def number_of_edges(self):
                return len(self.edges)

            def neighbors(self, node_id):
                for source, target, _ in self.edges:
                    if source == node_id:
                        yield target

            def predecessors(self, node_id):
                for source, target, _ in self.edges:
                    if target == node_id:
                        yield source

            def subgraph(self, nodes):
                return self

            def copy(self):
                return self

            def to_undirected(self):
                return self

        return _MiniGraph()
    return nx.MultiDiGraph()


@dataclass
class BioKnowledgeGraph:
    """Small persistent graph store for the demo agent."""

    graph: Any = field(default_factory=_build_graph)

    def add_entity(self, entity_id: str, entity_type: str, properties: dict | None = None, **attrs: Any) -> str:
        payload = {"type": entity_type, "label": entity_id}
        if properties:
            payload.update(properties)
        payload.update(attrs)
        self.graph.add_node(entity_id, **payload)
        return entity_id

    def add_relationship(self, source: str, target: str, rel_type: str, properties: dict | None = None, **attrs: Any) -> tuple[str, str, str]:
        payload = {"type": rel_type}
        if properties:
            payload.update(properties)
        payload.update(attrs)
        self.graph.add_edge(source, target, **payload)
        return source, target, rel_type

    def summary(self) -> dict:
        type_counts: dict[str, int] = {}
        relation_counts: dict[str, int] = {}
        try:
            for _, attrs in self.graph.nodes(data=True):
                node_type = str(attrs.get("type", "entity"))
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
        except Exception:
            pass
        try:
            for _, _, attrs in self.graph.edges(data=True):
                rel_type = str(attrs.get("type", "relation"))
                relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        except Exception:
            pass
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": type_counts,
            "relation_types": relation_counts,
        }

    def to_dict(self) -> dict:
        if nx is None:
            return {
                "nodes": self.graph.nodes,
                "edges": self.graph.edges,
            }
        return nx.node_link_data(self.graph, edges="links")

    def neighbors(self, entity_id: str) -> List[str]:
        if hasattr(self.graph, "neighbors"):
            return list(self.graph.neighbors(entity_id))
        return []

    def shortest_path(self, source: str, target: str) -> List[str]:
        if nx is None or source not in self.graph or target not in self.graph:
            return []
        try:
            return list(nx.shortest_path(self.graph.to_undirected(), source, target))
        except Exception:
            return []

    def query_entities(self, entities: Sequence[str]) -> dict[str, float]:
        hits: dict[str, float] = {}
        for entity in entities:
            if entity not in self.graph:
                continue
            degree = len(self.neighbors(entity))
            hits[entity] = min(1.0, 0.4 + 0.1 * degree)
            for neighbor in self.neighbors(entity):
                hits[neighbor] = max(hits.get(neighbor, 0.0), min(1.0, 0.2 + 0.05 * degree))
        return hits

    def relation_evidence(self, seeds: Sequence[str], hops: int = 2) -> dict[str, Any]:
        subgraph = self.query_subgraph(seeds, hops=hops)
        if nx is None:
            node_count = len(getattr(subgraph, "nodes", {}))
            edge_count = len(getattr(subgraph, "edges", []))
        else:
            node_count = subgraph.number_of_nodes()
            edge_count = subgraph.number_of_edges()
        return {"seeds": list(seeds), "hops": hops, "nodes": node_count, "edges": edge_count}

    def query_subgraph(self, seeds: Sequence[str], hops: int = 1):
        if nx is None:
            return self.graph
        seeds = [seed for seed in seeds if seed in self.graph]
        if not seeds:
            return nx.MultiDiGraph()
        nodes = set(seeds)
        frontier = set(seeds)
        for _ in range(max(hops, 0)):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.neighbors(node))
                next_frontier.update(self.graph.predecessors(node))
            next_frontier -= nodes
            nodes.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return self.graph.subgraph(nodes).copy()

    def subgraph(self, names: set[str]):
        return self.query_subgraph(list(names), hops=0)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if nx is None:
            save_pickle({"nodes": self.graph.nodes, "edges": self.graph.edges}, path)
        else:
            save_pickle(nx.node_link_data(self.graph, edges="links"), path)
        return path

    def load(self, path: str | Path) -> "BioKnowledgeGraph":
        payload = load_pickle(path)
        if nx is None:
            graph = _build_graph()
            graph.nodes.update(payload.get("nodes", {}))
            graph.edges.extend(payload.get("edges", []))
            self.graph = graph
            return self
        self.graph = nx.node_link_graph(payload, directed=True, multigraph=True, edges="links")
        return self

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        session_dir: str | Path | None = None,
        graph_checkpoint_name: str = "kg_session.pkl",
    ) -> "BioKnowledgeGraph":
        if path is None:
            base_dir = checkpoint_dir or session_dir
            if base_dir is None:
                return cls()
            path = Path(base_dir) / graph_checkpoint_name
        if not Path(path).exists():
            return cls()
        return cls().load(path)

    def export_html(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:  # pragma: no cover - optional dependency
            from pyvis.network import Network

            net = Network(height="650px", width="100%", directed=True)
            for node, attrs in self.graph.nodes(data=True):
                net.add_node(node, label=attrs.get("label", node), title=json.dumps(attrs))
            for source, target, attrs in self.graph.edges(data=True):
                net.add_edge(source, target, label=attrs.get("type", "relation"), title=json.dumps(attrs))
            net.write_html(str(path))
        except Exception:
            nodes = []
            try:
                node_iter = self.graph.nodes(data=True)
            except Exception:
                node_iter = []
            for node, attrs in node_iter:
                nodes.append({"id": node, **dict(attrs)})
            edges = []
            try:
                edge_iter = self.graph.edges(data=True)
            except Exception:
                edge_iter = []
            for source, target, attrs in edge_iter:
                edges.append({"source": source, "target": target, **dict(attrs)})
            path.write_text("<html><body><pre>" + json.dumps({"nodes": nodes, "edges": edges}, indent=2) + "</pre></body></html>", encoding="utf-8")
        return path
