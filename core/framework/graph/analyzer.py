"""Static Graph Analyzer for pre-execution validation.

Performs comprehensive static analysis on agent graphs to detect
problems BEFORE runtime execution.

Checks include:
- Dangerous cycles (without max_node_visits protection)
- Output key collisions on execution paths
- Missing inputs (input_keys not provided by predecessors)
- Dead-end paths (non-terminal nodes with no outgoing edges)
- Unreachable nodes (enhanced detection)
- Cost estimation (based on LLM node types)

Usage:
    from framework.graph.analyzer import GraphAnalyzer

    analyzer = GraphAnalyzer(graph_spec)
    report = analyzer.analyze()

    if report.has_errors:
        print("Errors:", report.errors)
    if report.has_warnings:
        print("Warnings:", report.warnings)
    print(f"Estimated cost: ~{report.estimated_tokens} tokens")
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from framework.graph.edge import GraphSpec


@dataclass
class AnalysisReport:
    """Result of static graph analysis.

    Contains categorized findings (errors, warnings, info) and detailed
    structural data for programmatic consumption.

    Attributes:
        errors: Critical issues that will likely cause runtime failures.
        warnings: Potential issues that may cause unexpected behavior.
        info: Informational findings (complexity, cost estimates).
        cycles: List of detected cycles as node ID sequences.
        unreachable_nodes: Node IDs with no path from any entry point.
        dead_end_nodes: Non-terminal node IDs with no outgoing edges.
        output_collisions: Dicts with 'key', 'first_writer', 'second_writer'.
        missing_inputs: Dicts with 'node', 'key', and resolution context.

    Example:
        report = AnalysisReport(errors=["Dangerous cycle detected: a → b → a"])
        assert not report.is_valid

        report = AnalysisReport(warnings=["Output collision on 'result'"])
        assert report.is_valid  # Warnings don't fail validation

        # Serialize for CI/CD pipelines
        data = report.to_dict()
        if not data["valid"]:
            sys.exit(1)
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    # Detailed findings
    cycles: list[list[str]] = field(default_factory=list)
    unreachable_nodes: list[str] = field(default_factory=list)
    dead_end_nodes: list[str] = field(default_factory=list)
    output_collisions: list[dict[str, Any]] = field(default_factory=list)
    missing_inputs: list[dict[str, Any]] = field(default_factory=list)

    # Cost estimation
    estimated_tokens: int = 0
    llm_node_count: int = 0
    total_node_count: int = 0
    total_edge_count: int = 0

    @property
    def has_errors(self) -> bool:
        """Check if analysis found critical errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if analysis found warnings."""
        return len(self.warnings) > 0

    @property
    def is_valid(self) -> bool:
        """Check if graph passed all critical checks."""
        return not self.has_errors

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "details": {
                "cycles": self.cycles,
                "unreachable_nodes": self.unreachable_nodes,
                "dead_end_nodes": self.dead_end_nodes,
                "output_collisions": self.output_collisions,
                "missing_inputs": self.missing_inputs,
            },
            "metrics": {
                "estimated_tokens": self.estimated_tokens,
                "llm_node_count": self.llm_node_count,
                "total_node_count": self.total_node_count,
                "total_edge_count": self.total_edge_count,
            },
        }


class GraphAnalyzer:
    """Static analyzer for agent execution graphs.

    Performs deep analysis of graph structure to detect problems
    that would only manifest at runtime, saving tokens and debugging time.

    Checks performed:
        - **Cycle detection**: DFS-based with max_node_visits awareness
        - **Reachability**: BFS from all entry points
        - **Dead ends**: Non-terminal nodes without outgoing edges
        - **Output collisions**: Same key written by sequential nodes
        - **Missing inputs**: input_keys not satisfied by predecessors
        - **Cost estimation**: ~800 tokens per LLM node (conservative)

    Example:
        from framework.graph import GraphAnalyzer

        analyzer = GraphAnalyzer(graph_spec)
        report = analyzer.analyze()

        # Check for critical issues
        if not report.is_valid:
            for error in report.errors:
                print(f"ERROR: {error}")
            sys.exit(1)

        # Review warnings
        for warning in report.warnings:
            print(f"WARNING: {warning}")

        # Cost estimation
        print(f"Estimated cost: ~{report.estimated_tokens} tokens")
    """

    # LLM node types that consume tokens
    LLM_NODE_TYPES = {"llm_generate", "llm_tool_use", "event_loop"}

    # Average tokens per LLM call (conservative estimate)
    AVG_TOKENS_PER_LLM_CALL = 800

    def __init__(self, graph: GraphSpec):
        """Initialize analyzer with a graph specification.

        Args:
            graph: The GraphSpec to analyze
        """
        self.graph = graph
        self._node_map: dict[str, Any] = {}
        self._adjacency: dict[str, list[str]] = {}
        self._reverse_adjacency: dict[str, list[str]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build internal indexes for efficient graph traversal.

        Creates:
            _node_map: O(1) lookup by node ID.
            _adjacency: Forward edges (node → successors).
            _reverse_adjacency: Backward edges (node → predecessors).
        """
        # Node lookup map
        for node in self.graph.nodes:
            self._node_map[node.id] = node

        # Adjacency lists (forward and reverse)
        for node in self.graph.nodes:
            self._adjacency[node.id] = []
            self._reverse_adjacency[node.id] = []

        for edge in self.graph.edges:
            if edge.source in self._adjacency:
                self._adjacency[edge.source].append(edge.target)
            if edge.target in self._reverse_adjacency:
                self._reverse_adjacency[edge.target].append(edge.source)

    def analyze(self) -> AnalysisReport:
        """Perform comprehensive static analysis on the graph.

        Runs all analysis passes in order: cycles, reachability, dead ends,
        output collisions, missing inputs, and cost estimation.

        Returns:
            AnalysisReport containing errors, warnings, info, and metrics.
        """
        report = AnalysisReport()

        # Basic metrics
        report.total_node_count = len(self.graph.nodes)
        report.total_edge_count = len(self.graph.edges)

        # Run all analysis passes
        self._analyze_cycles(report)
        self._analyze_reachability(report)
        self._analyze_dead_ends(report)
        self._analyze_output_collisions(report)
        self._analyze_missing_inputs(report)
        self._estimate_cost(report)

        # Add summary info
        report.info.append(
            f"Graph complexity: {report.total_node_count} nodes, {report.total_edge_count} edges"
        )
        if report.llm_node_count > 0:
            report.info.append(
                f"LLM nodes: {report.llm_node_count} (~{report.estimated_tokens} tokens estimated)"
            )

        return report

    def _analyze_cycles(self, report: AnalysisReport) -> None:
        """Detect cycles using DFS and check for proper loop protection.

        A cycle is dangerous if any node in it has max_node_visits <= 1,
        which could cause infinite loops at runtime. Protected cycles
        (feedback loops) are reported as info rather than errors.

        Args:
            report: AnalysisReport to append findings to.
        """
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node_id: str) -> bool:
            """DFS to detect cycles. Returns True if cycle found."""
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor in self._adjacency.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle - extract it
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    report.cycles.append(cycle)

                    # Check if cycle is protected
                    unprotected = []
                    for nid in cycle[:-1]:  # Exclude repeated node at end
                        node = self._node_map.get(nid)
                        if node:
                            max_visits = getattr(node, "max_node_visits", 1)
                            if max_visits <= 1:
                                unprotected.append(nid)

                    if unprotected:
                        cycle_str = " → ".join(cycle)
                        report.errors.append(
                            f"Dangerous cycle detected: {cycle_str}. "
                            f"Nodes without max_node_visits > 1: {unprotected}"
                        )
                    else:
                        cycle_str = " → ".join(cycle)
                        report.info.append(f"Protected cycle (feedback loop): {cycle_str}")
                    return True

            path.pop()
            rec_stack.remove(node_id)
            return False

        # Start DFS from entry node
        if self.graph.entry_node in self._node_map:
            dfs(self.graph.entry_node)

        # Also check from async entry points
        for entry_point in self.graph.async_entry_points:
            if entry_point.entry_node not in visited:
                if entry_point.entry_node in self._node_map:
                    path.clear()
                    rec_stack.clear()
                    dfs(entry_point.entry_node)

    def _analyze_reachability(self, report: AnalysisReport) -> None:
        """Find nodes unreachable from any entry point using BFS.

        Considers the main entry_node, named entry_points, and
        async_entry_points. Pause nodes and entry points themselves
        are excluded from unreachable warnings.

        Args:
            report: AnalysisReport to append findings to.
        """
        reachable: set[str] = set()
        to_visit: deque[str] = deque([self.graph.entry_node])

        # Add all entry points
        for ep_node in self.graph.entry_points.values():
            to_visit.append(ep_node)
        for async_ep in self.graph.async_entry_points:
            to_visit.append(async_ep.entry_node)

        # BFS to find all reachable nodes
        while to_visit:
            current = to_visit.popleft()
            if current in reachable:
                continue
            reachable.add(current)
            for neighbor in self._adjacency.get(current, []):
                if neighbor not in reachable:
                    to_visit.append(neighbor)

        # Find unreachable nodes
        async_entry_nodes = {ep.entry_node for ep in self.graph.async_entry_points}
        for node in self.graph.nodes:
            if node.id not in reachable:
                # Skip pause nodes and entry points (special cases)
                if (
                    node.id in self.graph.pause_nodes
                    or node.id in self.graph.entry_points.values()
                    or node.id in async_entry_nodes
                ):
                    continue
                report.unreachable_nodes.append(node.id)
                report.warnings.append(
                    f"Unreachable node: '{node.id}' has no path from any entry point"
                )

    def _analyze_dead_ends(self, report: AnalysisReport) -> None:
        """Find non-terminal nodes with no outgoing edges.

        A dead-end node causes execution to halt unexpectedly because
        the executor cannot determine the next step.

        Args:
            report: AnalysisReport to append findings to.
        """
        terminal_set = set(self.graph.terminal_nodes)

        for node in self.graph.nodes:
            # Skip terminal nodes
            if node.id in terminal_set:
                continue

            # Check if node has outgoing edges
            outgoing = self._adjacency.get(node.id, [])
            if not outgoing:
                report.dead_end_nodes.append(node.id)
                report.warnings.append(
                    f"Dead-end node: '{node.id}' has no outgoing edges "
                    f"but is not marked as terminal"
                )

    def _analyze_output_collisions(self, report: AnalysisReport) -> None:
        """Detect output key collisions on execution paths.

        When multiple nodes on the same path write to the same key,
        data can be unexpectedly overwritten.
        """
        # For each path from entry to terminal, track which keys are written
        # This is a simplified analysis - full path enumeration would be expensive

        # Build map of output_keys per node
        output_map: dict[str, list[str]] = {}
        for node in self.graph.nodes:
            output_keys = getattr(node, "output_keys", [])
            if output_keys:
                output_map[node.id] = output_keys

        # Check sequential paths using DFS
        # For efficiency, we check if any successor writes the same key
        for node_id, keys in output_map.items():
            self._check_output_collisions_for_node(node_id, keys, output_map, report)

    def _check_output_collisions_for_node(
        self,
        start_node: str,
        keys: list[str],
        output_map: dict[str, list[str]],
        report: AnalysisReport,
    ) -> None:
        """Check for output collisions starting from a specific node.

        Uses iterative DFS with a depth limit of 20 to avoid expensive
        full path enumeration on large graphs.

        Args:
            start_node: Node ID to start collision check from.
            keys: Output keys written by start_node.
            output_map: Pre-built map of node_id → output_keys.
            report: AnalysisReport to append findings to.
        """
        visited: set[str] = set()
        stack = [(start_node, 0)]

        while stack:
            current, depth = stack.pop()
            if depth > 20 or current in visited:
                continue
            visited.add(current)

            for successor in self._adjacency.get(current, []):
                successor_keys = output_map.get(successor, [])
                for key in keys:
                    if key in successor_keys:
                        report.output_collisions.append(
                            {
                                "key": key,
                                "first_writer": start_node,
                                "second_writer": successor,
                            }
                        )
                        report.warnings.append(
                            f"Output key collision: '{key}' written by "
                            f"'{start_node}' and later by '{successor}'"
                        )
                stack.append((successor, depth + 1))

    def _analyze_missing_inputs(self, report: AnalysisReport) -> None:
        """Detect nodes with input_keys not satisfied by predecessors.

        Resolution order for each node's input_keys:
            1. Output keys of direct predecessor nodes
            2. Graph-level memory_keys (always available)
            3. Initial context (entry node is skipped, gets input at runtime)

        Args:
            report: AnalysisReport to append findings to.
        """
        # Build map of what each node provides
        provides: dict[str, set[str]] = {}
        for node in self.graph.nodes:
            output_keys = getattr(node, "output_keys", [])
            provides[node.id] = set(output_keys)

        # Memory keys are always available
        memory_keys = set(self.graph.memory_keys)

        # Check each node's input requirements
        for node in self.graph.nodes:
            input_keys = getattr(node, "input_keys", [])
            if not input_keys:
                continue

            # Skip entry node (gets input from initial context)
            if node.id == self.graph.entry_node:
                continue

            # Get all predecessors
            predecessors = self._reverse_adjacency.get(node.id, [])
            if not predecessors:
                # Node has no predecessors but has input_keys
                # This is only OK if it's an entry point
                async_entries = {ep.entry_node for ep in self.graph.async_entry_points}
                if node.id not in async_entries and node.id not in self.graph.entry_points.values():
                    for key in input_keys:
                        if key not in memory_keys:
                            report.missing_inputs.append(
                                {
                                    "node": node.id,
                                    "key": key,
                                    "reason": "no_predecessors",
                                }
                            )
                            report.warnings.append(
                                f"Missing input: '{node.id}' expects '{key}' "
                                f"but has no predecessors"
                            )
                continue

            # Collect all keys provided by predecessors
            available_keys = set(memory_keys)
            for pred_id in predecessors:
                available_keys.update(provides.get(pred_id, set()))

            # Check if all input_keys are available
            for key in input_keys:
                if key not in available_keys:
                    # Check if any predecessor transitively provides it
                    # (simplified: just check direct predecessors)
                    report.missing_inputs.append(
                        {
                            "node": node.id,
                            "key": key,
                            "predecessors": predecessors,
                        }
                    )
                    pred_str = ", ".join(predecessors)
                    report.warnings.append(
                        f"Possible missing input: '{node.id}' expects '{key}' "
                        f"but predecessors ({pred_str}) may not provide it"
                    )

    def _estimate_cost(self, report: AnalysisReport) -> None:
        """Estimate token cost based on LLM node count.

        Uses a conservative average of ~800 tokens per LLM call.
        Actual cost varies with prompt length, tool usage, and
        which execution paths are taken at runtime.

        Args:
            report: AnalysisReport to append findings to.
        """
        llm_count = 0
        for node in self.graph.nodes:
            node_type = getattr(node, "node_type", "")
            if node_type in self.LLM_NODE_TYPES:
                llm_count += 1

        report.llm_node_count = llm_count
        report.estimated_tokens = llm_count * self.AVG_TOKENS_PER_LLM_CALL


def analyze_graph(graph: GraphSpec) -> AnalysisReport:
    """Convenience function to analyze a graph.

    Args:
        graph: The GraphSpec to analyze

    Returns:
        AnalysisReport with all findings
    """
    analyzer = GraphAnalyzer(graph)
    return analyzer.analyze()
