"""Tests for the GraphAnalyzer static analysis."""

import pytest

from framework.graph.analyzer import AnalysisReport, GraphAnalyzer, analyze_graph
from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
from framework.graph.node import NodeSpec


class TestAnalysisReport:
    """Tests for the AnalysisReport dataclass."""

    def test_empty_report_is_valid(self):
        """Empty report should be valid with no errors."""
        report = AnalysisReport()
        assert report.is_valid
        assert not report.has_errors
        assert not report.has_warnings

    def test_report_with_errors_is_invalid(self):
        """Report with errors should be invalid."""
        report = AnalysisReport(errors=["Some error"])
        assert not report.is_valid
        assert report.has_errors

    def test_report_with_warnings_is_valid(self):
        """Report with only warnings should still be valid."""
        report = AnalysisReport(warnings=["Some warning"])
        assert report.is_valid
        assert report.has_warnings

    def test_to_dict(self):
        """Test serialization to dictionary."""
        report = AnalysisReport(
            errors=["error1"],
            warnings=["warning1"],
            info=["info1"],
            estimated_tokens=1000,
            llm_node_count=2,
            total_node_count=5,
            total_edge_count=4,
        )
        result = report.to_dict()

        assert result["valid"] is False
        assert result["errors"] == ["error1"]
        assert result["warnings"] == ["warning1"]
        assert result["info"] == ["info1"]
        assert result["metrics"]["estimated_tokens"] == 1000
        assert result["metrics"]["llm_node_count"] == 2


class TestGraphAnalyzerBasic:
    """Basic tests for GraphAnalyzer."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple valid graph."""
        return GraphSpec(
            id="test-graph",
            goal_id="test-goal",
            entry_node="start",
            terminal_nodes=["end"],
            nodes=[
                NodeSpec(
                    id="start",
                    name="Start Node",
                    description="Entry point",
                    node_type="function",
                    output_keys=["data"],
                ),
                NodeSpec(
                    id="end",
                    name="End Node",
                    description="Exit point",
                    node_type="function",
                    input_keys=["data"],
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1",
                    source="start",
                    target="end",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
            ],
        )

    def test_analyze_simple_graph(self, simple_graph):
        """Simple valid graph should pass analysis."""
        analyzer = GraphAnalyzer(simple_graph)
        report = analyzer.analyze()

        assert report.is_valid
        assert report.total_node_count == 2
        assert report.total_edge_count == 1

    def test_convenience_function(self, simple_graph):
        """Test the analyze_graph convenience function."""
        report = analyze_graph(simple_graph)
        assert report.is_valid


class TestCycleDetection:
    """Tests for cycle detection."""

    @pytest.fixture
    def graph_with_dangerous_cycle(self):
        """Create a graph with an unprotected cycle."""
        return GraphSpec(
            id="cycle-graph",
            goal_id="test-goal",
            entry_node="a",
            terminal_nodes=["d"],
            nodes=[
                NodeSpec(id="a", name="A", description="Node A", node_type="function"),
                NodeSpec(id="b", name="B", description="Node B", node_type="function"),
                NodeSpec(id="c", name="C", description="Node C", node_type="function"),
                NodeSpec(id="d", name="D", description="Node D", node_type="function"),
            ],
            edges=[
                EdgeSpec(id="e1", source="a", target="b", condition=EdgeCondition.ON_SUCCESS),
                EdgeSpec(id="e2", source="b", target="c", condition=EdgeCondition.ON_SUCCESS),
                EdgeSpec(id="e3", source="c", target="a", condition=EdgeCondition.ON_SUCCESS),
                EdgeSpec(id="e4", source="c", target="d", condition=EdgeCondition.ON_FAILURE),
            ],
        )

    @pytest.fixture
    def graph_with_protected_cycle(self):
        """Create a graph with a properly protected feedback loop."""
        return GraphSpec(
            id="protected-cycle-graph",
            goal_id="test-goal",
            entry_node="worker",
            terminal_nodes=["done"],
            nodes=[
                NodeSpec(
                    id="worker",
                    name="Worker",
                    description="Does work",
                    node_type="event_loop",
                    max_node_visits=5,  # Protected!
                ),
                NodeSpec(
                    id="judge",
                    name="Judge",
                    description="Evaluates",
                    node_type="function",
                    max_node_visits=5,  # Protected!
                ),
                NodeSpec(
                    id="done",
                    name="Done",
                    description="Complete",
                    node_type="function",
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1", source="worker", target="judge", condition=EdgeCondition.ON_SUCCESS
                ),
                EdgeSpec(
                    id="e2", source="judge", target="worker", condition=EdgeCondition.CONDITIONAL
                ),
                EdgeSpec(
                    id="e3", source="judge", target="done", condition=EdgeCondition.ON_SUCCESS
                ),
            ],
        )

    def test_detects_dangerous_cycle(self, graph_with_dangerous_cycle):
        """Should detect unprotected cycles as errors."""
        analyzer = GraphAnalyzer(graph_with_dangerous_cycle)
        report = analyzer.analyze()

        assert report.has_errors
        assert len(report.cycles) > 0
        # Should mention "dangerous cycle" or similar
        assert any("cycle" in err.lower() for err in report.errors)

    def test_accepts_protected_cycle(self, graph_with_protected_cycle):
        """Should accept cycles with max_node_visits > 1."""
        analyzer = GraphAnalyzer(graph_with_protected_cycle)
        report = analyzer.analyze()

        # Should not have cycle-related errors
        cycle_errors = [e for e in report.errors if "cycle" in e.lower()]
        assert len(cycle_errors) == 0


class TestReachabilityAnalysis:
    """Tests for unreachable node detection."""

    @pytest.fixture
    def graph_with_unreachable_node(self):
        """Create a graph with an unreachable node."""
        return GraphSpec(
            id="unreachable-graph",
            goal_id="test-goal",
            entry_node="start",
            terminal_nodes=["end"],
            nodes=[
                NodeSpec(id="start", name="Start", description="Entry", node_type="function"),
                NodeSpec(id="end", name="End", description="Exit", node_type="function"),
                NodeSpec(
                    id="orphan", name="Orphan", description="Never reached", node_type="function"
                ),
            ],
            edges=[
                EdgeSpec(id="e1", source="start", target="end", condition=EdgeCondition.ON_SUCCESS),
            ],
        )

    def test_detects_unreachable_nodes(self, graph_with_unreachable_node):
        """Should warn about unreachable nodes."""
        analyzer = GraphAnalyzer(graph_with_unreachable_node)
        report = analyzer.analyze()

        assert report.has_warnings
        assert "orphan" in report.unreachable_nodes
        assert any("orphan" in w.lower() for w in report.warnings)


class TestDeadEndDetection:
    """Tests for dead-end node detection."""

    @pytest.fixture
    def graph_with_dead_end(self):
        """Create a graph with a dead-end node."""
        return GraphSpec(
            id="dead-end-graph",
            goal_id="test-goal",
            entry_node="start",
            terminal_nodes=["end"],
            nodes=[
                NodeSpec(id="start", name="Start", description="Entry", node_type="function"),
                NodeSpec(
                    id="dead_end",
                    name="Dead End",
                    description="No outgoing edges",
                    node_type="function",
                ),
                NodeSpec(id="end", name="End", description="Exit", node_type="function"),
            ],
            edges=[
                EdgeSpec(
                    id="e1", source="start", target="dead_end", condition=EdgeCondition.ON_SUCCESS
                ),
                EdgeSpec(id="e2", source="start", target="end", condition=EdgeCondition.ON_FAILURE),
            ],
        )

    def test_detects_dead_end_nodes(self, graph_with_dead_end):
        """Should warn about non-terminal nodes with no outgoing edges."""
        analyzer = GraphAnalyzer(graph_with_dead_end)
        report = analyzer.analyze()

        assert report.has_warnings
        assert "dead_end" in report.dead_end_nodes


class TestOutputCollisionDetection:
    """Tests for output key collision detection."""

    @pytest.fixture
    def graph_with_collision(self):
        """Create a graph where two nodes on the same path write to the same key."""
        return GraphSpec(
            id="collision-graph",
            goal_id="test-goal",
            entry_node="first",
            terminal_nodes=["last"],
            nodes=[
                NodeSpec(
                    id="first",
                    name="First Writer",
                    description="Writes result",
                    node_type="function",
                    output_keys=["result"],
                ),
                NodeSpec(
                    id="second",
                    name="Second Writer",
                    description="Also writes result",
                    node_type="function",
                    output_keys=["result"],  # Same key!
                ),
                NodeSpec(
                    id="last",
                    name="Last",
                    description="End",
                    node_type="function",
                    input_keys=["result"],
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1", source="first", target="second", condition=EdgeCondition.ON_SUCCESS
                ),
                EdgeSpec(
                    id="e2", source="second", target="last", condition=EdgeCondition.ON_SUCCESS
                ),
            ],
        )

    def test_detects_output_collision(self, graph_with_collision):
        """Should warn about output key collisions on the same path."""
        analyzer = GraphAnalyzer(graph_with_collision)
        report = analyzer.analyze()

        assert report.has_warnings
        assert len(report.output_collisions) > 0
        assert any("result" in w for w in report.warnings)


class TestMissingInputDetection:
    """Tests for missing input detection."""

    @pytest.fixture
    def graph_with_missing_input(self):
        """Create a graph where a node expects an input not provided by predecessors."""
        return GraphSpec(
            id="missing-input-graph",
            goal_id="test-goal",
            entry_node="start",
            terminal_nodes=["end"],
            nodes=[
                NodeSpec(
                    id="start",
                    name="Start",
                    description="Entry",
                    node_type="function",
                    output_keys=["data"],  # Only provides "data"
                ),
                NodeSpec(
                    id="end",
                    name="End",
                    description="Exit",
                    node_type="function",
                    input_keys=["data", "extra"],  # Expects "extra" which nobody provides
                ),
            ],
            edges=[
                EdgeSpec(id="e1", source="start", target="end", condition=EdgeCondition.ON_SUCCESS),
            ],
        )

    def test_detects_missing_inputs(self, graph_with_missing_input):
        """Should warn when input_keys cannot be satisfied by predecessors."""
        analyzer = GraphAnalyzer(graph_with_missing_input)
        report = analyzer.analyze()

        assert report.has_warnings
        assert len(report.missing_inputs) > 0
        assert any("extra" in str(m) for m in report.missing_inputs)


class TestCostEstimation:
    """Tests for token cost estimation."""

    @pytest.fixture
    def graph_with_llm_nodes(self):
        """Create a graph with multiple LLM nodes."""
        return GraphSpec(
            id="llm-graph",
            goal_id="test-goal",
            entry_node="start",
            terminal_nodes=["end"],
            nodes=[
                NodeSpec(
                    id="start",
                    name="Start",
                    description="Entry",
                    node_type="function",
                ),
                NodeSpec(
                    id="llm1",
                    name="LLM Node 1",
                    description="First LLM call",
                    node_type="llm_generate",
                ),
                NodeSpec(
                    id="llm2",
                    name="LLM Node 2",
                    description="Second LLM call",
                    node_type="event_loop",
                ),
                NodeSpec(
                    id="end",
                    name="End",
                    description="Exit",
                    node_type="function",
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1", source="start", target="llm1", condition=EdgeCondition.ON_SUCCESS
                ),
                EdgeSpec(id="e2", source="llm1", target="llm2", condition=EdgeCondition.ON_SUCCESS),
                EdgeSpec(id="e3", source="llm2", target="end", condition=EdgeCondition.ON_SUCCESS),
            ],
        )

    def test_estimates_token_cost(self, graph_with_llm_nodes):
        """Should estimate token cost based on LLM nodes."""
        analyzer = GraphAnalyzer(graph_with_llm_nodes)
        report = analyzer.analyze()

        assert report.llm_node_count == 2
        assert report.estimated_tokens > 0
        # Default is 800 tokens per LLM call
        assert report.estimated_tokens == 2 * 800


class TestComplexGraph:
    """Tests with more complex graph structures."""

    @pytest.fixture
    def complex_graph(self):
        """Create a complex graph with multiple features."""
        return GraphSpec(
            id="complex-graph",
            goal_id="test-goal",
            entry_node="input",
            terminal_nodes=["success", "failure"],
            nodes=[
                NodeSpec(
                    id="input",
                    name="Input Parser",
                    description="Parse input",
                    node_type="function",
                    output_keys=["parsed_data"],
                ),
                NodeSpec(
                    id="process",
                    name="Process",
                    description="Main processing",
                    node_type="llm_generate",
                    input_keys=["parsed_data"],
                    output_keys=["result"],
                ),
                NodeSpec(
                    id="validate",
                    name="Validate",
                    description="Validate result",
                    node_type="function",
                    input_keys=["result"],
                    output_keys=["is_valid"],
                ),
                NodeSpec(
                    id="success",
                    name="Success",
                    description="Success handler",
                    node_type="function",
                    input_keys=["result"],
                ),
                NodeSpec(
                    id="failure",
                    name="Failure",
                    description="Failure handler",
                    node_type="function",
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1", source="input", target="process", condition=EdgeCondition.ON_SUCCESS
                ),
                EdgeSpec(
                    id="e2", source="process", target="validate", condition=EdgeCondition.ON_SUCCESS
                ),
                EdgeSpec(
                    id="e3", source="validate", target="success", condition=EdgeCondition.ON_SUCCESS
                ),
                EdgeSpec(
                    id="e4", source="validate", target="failure", condition=EdgeCondition.ON_FAILURE
                ),
                EdgeSpec(
                    id="e5", source="process", target="failure", condition=EdgeCondition.ON_FAILURE
                ),
            ],
        )

    def test_complex_graph_analysis(self, complex_graph):
        """Complex but valid graph should pass analysis."""
        analyzer = GraphAnalyzer(complex_graph)
        report = analyzer.analyze()

        # Should be valid (no critical errors)
        assert report.is_valid
        assert report.total_node_count == 5
        assert report.total_edge_count == 5
        assert report.llm_node_count == 1


class TestEdgeCases:
    """Tests for edge cases and degenerate graph structures."""

    def test_single_node_graph(self):
        """Graph with only entry node (also terminal) should be valid."""
        graph = GraphSpec(
            id="single-node",
            goal_id="test-goal",
            entry_node="only",
            terminal_nodes=["only"],
            nodes=[
                NodeSpec(
                    id="only",
                    name="Only Node",
                    description="Entry and terminal",
                    node_type="function",
                ),
            ],
            edges=[],
        )
        report = GraphAnalyzer(graph).analyze()

        assert report.is_valid
        assert len(report.unreachable_nodes) == 0
        assert len(report.dead_end_nodes) == 0

    def test_self_loop_without_protection(self):
        """Self-loop (a -> a) without max_node_visits should be an error."""
        graph = GraphSpec(
            id="self-loop",
            goal_id="test-goal",
            entry_node="loop",
            terminal_nodes=["end"],
            nodes=[
                NodeSpec(
                    id="loop",
                    name="Self Loop",
                    description="Points to itself",
                    node_type="function",
                ),
                NodeSpec(
                    id="end",
                    name="End",
                    description="Terminal",
                    node_type="function",
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1",
                    source="loop",
                    target="loop",
                    condition=EdgeCondition.CONDITIONAL,
                ),
                EdgeSpec(
                    id="e2",
                    source="loop",
                    target="end",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
            ],
        )
        report = GraphAnalyzer(graph).analyze()

        assert report.has_errors
        assert len(report.cycles) > 0
        assert any("loop" in err.lower() for err in report.errors)

    def test_self_loop_with_protection(self):
        """Self-loop with max_node_visits > 1 should be reported as info, not error."""
        graph = GraphSpec(
            id="protected-self-loop",
            goal_id="test-goal",
            entry_node="retry",
            terminal_nodes=["end"],
            nodes=[
                NodeSpec(
                    id="retry",
                    name="Retry Node",
                    description="Retries itself",
                    node_type="event_loop",
                    max_node_visits=3,
                ),
                NodeSpec(
                    id="end",
                    name="End",
                    description="Terminal",
                    node_type="function",
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1",
                    source="retry",
                    target="retry",
                    condition=EdgeCondition.CONDITIONAL,
                ),
                EdgeSpec(
                    id="e2",
                    source="retry",
                    target="end",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
            ],
        )
        report = GraphAnalyzer(graph).analyze()

        cycle_errors = [e for e in report.errors if "cycle" in e.lower()]
        assert len(cycle_errors) == 0

    def test_multiple_issues_combined(self):
        """Graph with multiple problems should report all of them."""
        graph = GraphSpec(
            id="many-issues",
            goal_id="test-goal",
            entry_node="start",
            terminal_nodes=["end"],
            nodes=[
                NodeSpec(
                    id="start",
                    name="Start",
                    description="Entry",
                    node_type="function",
                    output_keys=["data"],
                ),
                NodeSpec(
                    id="a",
                    name="Node A",
                    description="In cycle",
                    node_type="function",
                    output_keys=["data"],
                ),
                NodeSpec(
                    id="b",
                    name="Node B",
                    description="In cycle",
                    node_type="function",
                    input_keys=["missing_key"],
                ),
                NodeSpec(
                    id="orphan",
                    name="Orphan",
                    description="Unreachable",
                    node_type="function",
                ),
                NodeSpec(
                    id="end",
                    name="End",
                    description="Terminal",
                    node_type="function",
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1",
                    source="start",
                    target="a",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
                EdgeSpec(
                    id="e2",
                    source="a",
                    target="b",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
                EdgeSpec(
                    id="e3",
                    source="b",
                    target="a",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
                EdgeSpec(
                    id="e4",
                    source="b",
                    target="end",
                    condition=EdgeCondition.ON_FAILURE,
                ),
            ],
        )
        report = GraphAnalyzer(graph).analyze()

        # Should detect cycle
        assert len(report.cycles) > 0
        # Should detect unreachable node
        assert "orphan" in report.unreachable_nodes
        # Should detect output collision (start and a both write 'data')
        assert len(report.output_collisions) > 0
        # Should detect missing input
        assert any(m["key"] == "missing_key" for m in report.missing_inputs)


class TestAsyncEntryPoints:
    """Tests for graphs with async entry points."""

    @pytest.fixture
    def multi_entry_graph(self):
        """Create a graph with multiple async entry points."""
        from framework.graph.edge import AsyncEntryPointSpec

        return GraphSpec(
            id="multi-entry-graph",
            goal_id="test-goal",
            entry_node="main_handler",
            terminal_nodes=["done"],
            async_entry_points=[
                AsyncEntryPointSpec(
                    id="webhook",
                    name="Webhook Handler",
                    entry_node="webhook_handler",
                    trigger_type="webhook",
                ),
                AsyncEntryPointSpec(
                    id="api",
                    name="API Handler",
                    entry_node="api_handler",
                    trigger_type="api",
                ),
            ],
            nodes=[
                NodeSpec(
                    id="main_handler",
                    name="Main Handler",
                    description="Main entry",
                    node_type="function",
                ),
                NodeSpec(
                    id="webhook_handler",
                    name="Webhook Handler",
                    description="Webhook entry",
                    node_type="function",
                ),
                NodeSpec(
                    id="api_handler",
                    name="API Handler",
                    description="API entry",
                    node_type="function",
                ),
                NodeSpec(
                    id="done",
                    name="Done",
                    description="Complete",
                    node_type="function",
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1",
                    source="main_handler",
                    target="done",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
                EdgeSpec(
                    id="e2",
                    source="webhook_handler",
                    target="done",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
                EdgeSpec(
                    id="e3",
                    source="api_handler",
                    target="done",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
            ],
        )

    def test_multi_entry_point_graph(self, multi_entry_graph):
        """Should handle multi-entry-point graphs correctly."""
        analyzer = GraphAnalyzer(multi_entry_graph)
        report = analyzer.analyze()

        # All nodes should be reachable via their entry points
        assert report.is_valid
        assert len(report.unreachable_nodes) == 0
