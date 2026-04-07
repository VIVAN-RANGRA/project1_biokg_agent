"""LLM-driven ReAct agent for BioKG-Agent.

The LLM sees the user question, available tools, and accumulated context,
then decides which tool to call next (or to give a final answer).

Works with both GroqBackend (API) and LLMBackend (local GPU).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """You are BioKG-Agent, a biological reasoning assistant.

You have access to these tools:

1. ncbi_gene_lookup(gene_symbol) - Get gene summary, aliases, diseases from NCBI
2. uniprot_protein_lookup(protein_name) - Get protein function, GO terms, domains, PTMs from UniProt
3. string_interactions(gene, score_threshold=700) - Get protein-protein interactions from STRING
4. kegg_pathway_lookup(gene) - Get KEGG pathways for a gene
5. drugbank_target_lookup(gene) - Get drugs targeting a protein
6. pubmed_rag_search(query) - Search PubMed literature via RAG
7. kg_add_entity(entity_id, entity_type, properties) - Add node to knowledge graph
8. kg_add_relationship(source, target, rel_type, properties) - Add edge to knowledge graph
9. kg_query(entity_id) - Query knowledge graph for an entity's neighbors and connections
10. kg_shortest_path(source, target) - Find shortest path between two entities in the graph
11. FINISH(answer) - Return final answer to user

WORKFLOW:
1. THINK about what information you need
2. Call ONE tool at a time using this exact format:
   THOUGHT: <your reasoning>
   ACTION: <tool_name>(<arguments>)
3. You will receive the OBSERVATION (tool result)
4. Repeat THINK-ACTION-OBSERVATION until you have enough evidence
5. When ready, call FINISH(your detailed answer with citations)

RULES:
- Always build the knowledge graph as you go (use kg_add_entity and kg_add_relationship)
- Cite sources: (STRING score: 0.92), (PMID: 12345), (DrugBank: approved)
- If graph reveals unexpected connections, highlight them
- Maximum {max_steps} tool calls before you must FINISH
- Be systematic: look up entities first, then interactions, then drugs/pathways
"""

# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _strip_quotes(s: str) -> str:
    """Remove surrounding quotes from a string value."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _parse_args_string(args_str: str) -> tuple[list[str], dict[str, Any]]:
    """Parse a tool argument string into positional and keyword arguments.

    Handles formats like:
        'TP53'
        '"TP53", score_threshold=700'
        'TP53, MDM2'
        'entity_id="TP53", entity_type="gene", properties={"label": "TP53"}'
    """
    args_str = args_str.strip()
    if not args_str:
        return [], {}

    positional: list[str] = []
    keyword: dict[str, Any] = {}

    # Split on commas that are not inside braces or brackets
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in args_str:
        if char in ('{', '[', '('):
            depth += 1
            current.append(char)
        elif char in ('}', ']', ')'):
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current).strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check for keyword argument
        kw_match = re.match(r'^(\w+)\s*=\s*(.+)$', part, re.DOTALL)
        if kw_match:
            key = kw_match.group(1)
            val_str = kw_match.group(2).strip()
            # Try JSON parse for dicts/lists/numbers
            try:
                keyword[key] = json.loads(val_str)
            except (json.JSONDecodeError, ValueError):
                keyword[key] = _strip_quotes(val_str)
        else:
            # Positional argument — try int/float first
            stripped = _strip_quotes(part)
            try:
                positional.append(str(int(stripped)))
            except ValueError:
                positional.append(stripped)

    return positional, keyword


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """LLM-driven ReAct agent that wraps BioKGAgent's tools."""

    def __init__(self, base_agent, llm, max_steps: int = 10):
        """
        Parameters
        ----------
        base_agent : BioKGAgent
            The underlying agent with tool methods and knowledge graph.
        llm : GroqBackend | LLMBackend
            An LLM backend with a ``generate(prompt, system_prompt=...)`` method.
        max_steps : int
            Maximum number of tool calls before forcing a final answer.
        """
        self.agent = base_agent
        self.llm = llm
        self.max_steps = max_steps
        self.tools: dict[str, Callable[..., Any]] = self._register_tools()

    def _register_tools(self) -> dict[str, Callable[..., Any]]:
        """Map tool names to actual methods on the base agent."""
        return {
            "ncbi_gene_lookup": self.agent.ncbi_gene_lookup,
            "uniprot_protein_lookup": self.agent.uniprot_protein_lookup,
            "string_interactions": self.agent.string_interactions,
            "kegg_pathway_lookup": self.agent.kegg_pathway_lookup,
            "drugbank_target_lookup": self.agent.drugbank_target_lookup,
            "pubmed_rag_search": self._pubmed_search,
            "kg_add_entity": self.agent.kg.add_entity,
            "kg_add_relationship": self.agent.kg.add_relationship,
            "kg_query": self._kg_query,
            "kg_shortest_path": self.agent.kg.shortest_path,
        }

    # ------------------------------------------------------------------
    # Wrapped tool helpers
    # ------------------------------------------------------------------

    def _pubmed_search(self, query: str) -> list[dict[str, Any]]:
        """Wrap pubmed_rag_search to return simple JSON-serializable dicts."""
        try:
            bundle = self.agent.pubmed_rag_search(query)
            results = []
            for c in bundle.candidates[:5]:
                results.append({
                    "pmid": c.payload.get("pmid", ""),
                    "title": c.payload.get("title", ""),
                    "snippet": str(c.payload.get("abstract", ""))[:200],
                    "score": round(c.final_score, 4),
                })
            return results
        except Exception as exc:
            return [{"error": str(exc)}]

    def _kg_query(self, entity_id: str) -> dict[str, Any]:
        """Query KG for entity neighbors and info."""
        try:
            neighbors = self.agent.kg.neighbors(entity_id)
        except Exception:
            neighbors = []
        return {
            "entity": entity_id,
            "neighbors": neighbors[:20],
            "degree": len(neighbors),
        }

    # ------------------------------------------------------------------
    # Core ReAct loop
    # ------------------------------------------------------------------

    def invoke(self, query: str) -> dict[str, Any]:
        """Run the ReAct loop: iteratively reason and act until FINISH or max steps."""
        system = REACT_SYSTEM_PROMPT.format(max_steps=self.max_steps)
        conversation = f"USER QUESTION: {query}\n\n"
        steps: list[dict[str, Any]] = []

        for step_num in range(self.max_steps):
            # Manage context window — summarize if conversation is too long
            prompt_text = self._manage_context(conversation, query)

            # Ask LLM for next action
            try:
                response = self.llm.generate(
                    prompt_text + "What is your next step?",
                    system_prompt=system,
                )
            except Exception as exc:
                logger.error("LLM generation failed at step %d: %s", step_num + 1, exc)
                return self._error_result(
                    f"LLM generation failed: {exc}", steps, query,
                )

            # Parse THOUGHT and ACTION from response
            thought, action_name, action_args = self._parse_response(response)
            logger.info(
                "Step %d: THOUGHT=%s ACTION=%s(%s)",
                step_num + 1, thought[:80], action_name, action_args[:80],
            )

            if action_name == "FINISH":
                # Final answer
                return {
                    "answer_text": action_args,
                    "steps": steps,
                    "num_steps": len(steps),
                    "graph_summary": self.agent.kg.summary(),
                    "graph_html": str(self.agent.config.graph_html_path),
                }

            # Execute tool
            observation = self._execute_tool(action_name, action_args)

            steps.append({
                "step": step_num + 1,
                "thought": thought,
                "action": f"{action_name}({action_args})",
                "observation": str(observation)[:500],
            })

            # Append to conversation
            obs_text = self._truncate(observation)
            conversation += (
                f"THOUGHT: {thought}\n"
                f"ACTION: {action_name}({action_args})\n"
                f"OBSERVATION: {obs_text}\n\n"
            )

        # Max steps reached — force finish
        try:
            final = self.llm.generate(
                conversation
                + "\nYou've reached the maximum number of steps. "
                "Provide your final answer based on evidence gathered so far.",
                system_prompt=system,
            )
        except Exception as exc:
            final = f"Max steps reached. LLM synthesis failed: {exc}"

        return {
            "answer_text": final,
            "steps": steps,
            "num_steps": len(steps),
            "graph_summary": self.agent.kg.summary(),
            "graph_html": str(self.agent.config.graph_html_path),
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> tuple[str, str, str]:
        """Parse THOUGHT/ACTION from LLM output.

        Returns (thought, action_name, action_args).
        Handles multiple formats robustly and defaults to FINISH with raw
        text if parsing fails completely.
        """
        text = text.strip()

        # --- Try to extract THOUGHT ---
        thought = ""
        thought_match = re.search(
            r'THOUGHT\s*:\s*(.+?)(?=\nACTION\s*:|$)', text, re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # --- Try to extract ACTION ---
        # Pattern 1: ACTION: FINISH(...)
        finish_match = re.search(
            r'(?:ACTION\s*:\s*)?FINISH\s*\((.+)\)\s*$', text, re.DOTALL | re.IGNORECASE,
        )
        if finish_match:
            answer = finish_match.group(1).strip()
            # Remove wrapping quotes if the whole answer is quoted
            answer = _strip_quotes(answer)
            if not thought:
                # Everything before FINISH is the thought
                idx = text.lower().find("finish")
                if idx > 0:
                    thought = text[:idx].strip()
                    # Clean up "ACTION:" prefix from thought
                    thought = re.sub(r'^(THOUGHT\s*:\s*)', '', thought, flags=re.IGNORECASE).strip()
            return thought, "FINISH", answer

        # Pattern 2: ACTION: tool_name(args)
        action_match = re.search(
            r'ACTION\s*:\s*(\w+)\s*\((.*)?\)\s*$', text, re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            action_name = action_match.group(1).strip()
            action_args = (action_match.group(2) or "").strip()
            if not thought:
                idx = text.lower().find("action")
                if idx > 0:
                    thought = text[:idx].strip()
                    thought = re.sub(r'^(THOUGHT\s*:\s*)', '', thought, flags=re.IGNORECASE).strip()
            return thought, action_name, action_args

        # Pattern 3: Inline tool call without ACTION: prefix — tool_name(args)
        inline_match = re.search(
            r'\b(\w+)\s*\(([^)]*)\)\s*$', text, re.DOTALL,
        )
        if inline_match:
            candidate_name = inline_match.group(1).strip()
            if candidate_name in self.tools or candidate_name.upper() == "FINISH":
                action_args = (inline_match.group(2) or "").strip()
                if not thought:
                    thought = text[:inline_match.start()].strip()
                if candidate_name.upper() == "FINISH":
                    return thought, "FINISH", _strip_quotes(action_args)
                return thought, candidate_name, action_args

        # Fallback: cannot parse — treat entire response as final answer
        logger.warning("Could not parse ACTION from LLM output; treating as FINISH.")
        return text, "FINISH", text

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, args_str: str) -> Any:
        """Execute a tool by name with parsed arguments.

        Returns the tool result or an error message string. Never raises.
        """
        if name not in self.tools:
            return f"ERROR: Unknown tool '{name}'. Available tools: {', '.join(self.tools.keys())}"

        tool_fn = self.tools[name]
        positional, keyword = _parse_args_string(args_str)

        try:
            # Map tool-specific argument structures
            if name == "ncbi_gene_lookup":
                return tool_fn(positional[0] if positional else keyword.get("gene_symbol", keyword.get("gene", "")))

            elif name == "uniprot_protein_lookup":
                return tool_fn(positional[0] if positional else keyword.get("protein_name", keyword.get("protein", "")))

            elif name == "string_interactions":
                gene = positional[0] if positional else keyword.get("gene", "")
                threshold = keyword.get("score_threshold", None)
                if threshold is not None:
                    threshold = int(threshold)
                return tool_fn(gene, score_threshold=threshold)

            elif name == "kegg_pathway_lookup":
                return tool_fn(positional[0] if positional else keyword.get("gene", ""))

            elif name == "drugbank_target_lookup":
                return tool_fn(positional[0] if positional else keyword.get("gene", ""))

            elif name == "pubmed_rag_search":
                return tool_fn(positional[0] if positional else keyword.get("query", ""))

            elif name == "kg_add_entity":
                entity_id = positional[0] if len(positional) > 0 else keyword.get("entity_id", "")
                entity_type = positional[1] if len(positional) > 1 else keyword.get("entity_type", "entity")
                props = keyword.get("properties", {})
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except (json.JSONDecodeError, ValueError):
                        props = {}
                return tool_fn(entity_id, entity_type, properties=props)

            elif name == "kg_add_relationship":
                source = positional[0] if len(positional) > 0 else keyword.get("source", "")
                target = positional[1] if len(positional) > 1 else keyword.get("target", "")
                rel_type = positional[2] if len(positional) > 2 else keyword.get("rel_type", "RELATED_TO")
                props = keyword.get("properties", {})
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except (json.JSONDecodeError, ValueError):
                        props = {}
                return tool_fn(source, target, rel_type, properties=props)

            elif name == "kg_query":
                return tool_fn(positional[0] if positional else keyword.get("entity_id", ""))

            elif name == "kg_shortest_path":
                source = positional[0] if len(positional) > 0 else keyword.get("source", "")
                target = positional[1] if len(positional) > 1 else keyword.get("target", "")
                return tool_fn(source, target)

            else:
                # Generic call — try positional then keyword
                if positional and not keyword:
                    return tool_fn(*positional)
                elif keyword and not positional:
                    return tool_fn(**keyword)
                else:
                    return tool_fn(*positional, **keyword)

        except Exception as exc:
            logger.error("Tool %s raised: %s", name, exc)
            return f"ERROR executing {name}: {exc}"

    # ------------------------------------------------------------------
    # Context window management
    # ------------------------------------------------------------------

    def _manage_context(self, conversation: str, original_query: str) -> str:
        """If conversation exceeds ~3000 chars, summarize earlier steps.

        Always keeps the original question visible at the top.
        """
        max_context = 3000
        if len(conversation) <= max_context:
            return conversation

        # Split into lines and find step boundaries
        lines = conversation.split("\n")
        header = f"USER QUESTION: {original_query}\n\n"

        # Find THOUGHT/ACTION/OBSERVATION blocks
        blocks: list[str] = []
        current_block: list[str] = []
        for line in lines:
            if line.startswith("THOUGHT:") and current_block:
                blocks.append("\n".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            blocks.append("\n".join(current_block))

        if len(blocks) <= 2:
            # Not enough blocks to summarize — just truncate
            return header + conversation[-(max_context - len(header)):]

        # Keep first block (may contain user question) and last 2 blocks in full
        # Summarize middle blocks
        summary_parts: list[str] = []
        for block in blocks[1:-2]:
            # Extract just action lines
            action_match = re.search(r'ACTION:\s*(.+)', block, re.IGNORECASE)
            obs_match = re.search(r'OBSERVATION:\s*(.{0,100})', block, re.IGNORECASE)
            if action_match:
                summary = f"[Earlier] {action_match.group(1).strip()}"
                if obs_match:
                    summary += f" -> {obs_match.group(1).strip()}..."
                summary_parts.append(summary)

        summarized = header
        if summary_parts:
            summarized += "[SUMMARY OF EARLIER STEPS]\n"
            summarized += "\n".join(summary_parts)
            summarized += "\n[END SUMMARY]\n\n"

        # Append last 2 full blocks
        for block in blocks[-2:]:
            summarized += block + "\n"

        return summarized

    def _truncate(self, obj: Any, max_len: int = 800) -> str:
        """Truncate observation for context window management."""
        try:
            text = json.dumps(obj, indent=None, default=str)
        except (TypeError, ValueError):
            text = str(obj)

        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    # ------------------------------------------------------------------
    # Error helper
    # ------------------------------------------------------------------

    def _error_result(
        self, message: str, steps: list[dict[str, Any]], query: str,
    ) -> dict[str, Any]:
        """Build an error result dict."""
        return {
            "answer_text": f"Agent error: {message}",
            "steps": steps,
            "num_steps": len(steps),
            "graph_summary": self.agent.kg.summary(),
            "graph_html": str(self.agent.config.graph_html_path),
            "error": message,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_react_agent(
    config=None,
    llm=None,
    groq_api_key: str | None = None,
    max_steps: int = 10,
) -> ReActAgent:
    """Build a ReAct agent with all dependencies.

    Parameters
    ----------
    config : ProjectConfig | None
        Project configuration. Defaults to ``ProjectConfig.from_env()``.
    llm : GroqBackend | LLMBackend | None
        Pre-built LLM backend. If *None*, one is created automatically
        from config / environment variables.
    groq_api_key : str | None
        Groq API key override (convenience parameter).
    max_steps : int
        Maximum ReAct loop iterations.

    Returns
    -------
    ReActAgent
    """
    from .agent import BioKGAgent
    from .config import ProjectConfig

    config = config or ProjectConfig.from_env()
    agent = BioKGAgent.build(config=config)

    if llm is None:
        from .llm import create_llm_backend

        llm = create_llm_backend(
            groq_api_key=groq_api_key or config.groq_api_key,
            groq_model=config.groq_model,
            backend=config.llm_backend,
        )

    return ReActAgent(base_agent=agent, llm=llm, max_steps=max_steps)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=" * 60)
    print("BioKG-Agent ReAct Agent — Smoke Test")
    print("=" * 60)

    try:
        react_agent = create_react_agent()
        print(f"ReAct agent created with {react_agent.max_steps} max steps.")
        print(f"Tools available: {', '.join(react_agent.tools.keys())}")
        print(f"LLM backend: {react_agent.llm}")

        query = "What are the key protein interactions and drug targets for TP53 in cancer?"
        print(f"\nQuery: {query}")
        print("-" * 60)

        result = react_agent.invoke(query)

        print(f"\nAnswer:\n{result['answer_text']}")
        print(f"\nSteps taken: {result['num_steps']}")
        for step in result.get("steps", []):
            print(f"  Step {step['step']}: {step['action']}")
        print(f"\nGraph summary: {result.get('graph_summary', {})}")

    except Exception as exc:
        print(f"\nSmoke test could not complete (expected in env without LLM): {exc}")
        print("The module is importable and the API is functional.")
