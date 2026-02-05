"""
Tests for token budget enforcement.

Verifies that TokenBudget correctly tracks consumption, warns at threshold,
raises on excess, and integrates with GraphExecutor for execution-level cost control.
"""

import pytest

from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
from framework.graph.executor import GraphExecutor
from framework.graph.goal import Goal
from framework.graph.node import NodeContext, NodeProtocol, NodeResult, NodeSpec
from framework.graph.token_budget import TokenBudget, TokenBudgetExceeded

# ---------------------------------------------------------------------------
# Unit tests: TokenBudget class
# ---------------------------------------------------------------------------


class TestTokenBudget:
    """Unit tests for the TokenBudget class."""

    def test_initial_state(self):
        budget = TokenBudget(limit=10_000)
        assert budget.limit == 10_000
        assert budget.consumed == 0
        assert budget.remaining == 10_000
        assert budget.utilization == 0.0

    def test_record_updates_consumed(self):
        budget = TokenBudget(limit=5000)
        budget.record(1200)
        assert budget.consumed == 1200
        assert budget.remaining == 3800

    def test_utilization_tracks_fraction(self):
        budget = TokenBudget(limit=10_000)
        budget.record(2500)
        assert budget.utilization == pytest.approx(0.25)

    def test_record_raises_on_exceed(self):
        budget = TokenBudget(limit=1000)
        budget.record(800)
        with pytest.raises(TokenBudgetExceeded) as exc_info:
            budget.record(300)
        assert exc_info.value.limit == 1000
        assert exc_info.value.consumed == 1100
        assert exc_info.value.last_node_tokens == 300

    def test_exact_limit_does_not_raise(self):
        budget = TokenBudget(limit=500)
        budget.record(500)  # Exactly at limit — should NOT raise
        assert budget.consumed == 500
        assert budget.remaining == 0

    def test_would_exceed_predicts_correctly(self):
        budget = TokenBudget(limit=1000)
        budget.record(800)
        assert budget.would_exceed(201) is True
        assert budget.would_exceed(200) is False
        assert budget.would_exceed(100) is False

    def test_invalid_limit_raises_value_error(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            TokenBudget(limit=0)
        with pytest.raises(ValueError, match="must be >= 1"):
            TokenBudget(limit=-5)

    def test_remaining_never_negative(self):
        budget = TokenBudget(limit=100)
        try:
            budget.record(200)
        except TokenBudgetExceeded:
            pass
        assert budget.remaining == 0

    def test_warning_at_threshold(self, caplog):
        """Budget logs a warning when crossing 80% utilization."""
        import logging

        budget = TokenBudget(limit=1000)
        with caplog.at_level(logging.WARNING):
            budget.record(799)
            assert "Token budget at" not in caplog.text
            budget.record(1)  # Now at exactly 80%
            assert "Token budget at" in caplog.text

    def test_warning_fires_only_once(self, caplog):
        """Warning should only fire once, not on every subsequent record."""
        import logging

        budget = TokenBudget(limit=1000)
        with caplog.at_level(logging.WARNING):
            budget.record(850)
            first_count = caplog.text.count("Token budget at")
            budget.record(50)
            second_count = caplog.text.count("Token budget at")
            assert first_count == 1
            assert second_count == 1  # Same count, no second warning

    def test_record_zero_tokens(self):
        """Recording zero tokens is valid and a no-op."""
        budget = TokenBudget(limit=100)
        budget.record(0)
        assert budget.consumed == 0
        assert budget.remaining == 100

    def test_record_negative_tokens_raises(self):
        """Negative tokens must be rejected to prevent budget bypass."""
        budget = TokenBudget(limit=1000)
        with pytest.raises(ValueError, match="must be >= 0"):
            budget.record(-1)

    def test_multiple_records_accumulate(self):
        """Multiple calls to record() accumulate correctly."""
        budget = TokenBudget(limit=10_000)
        for _ in range(10):
            budget.record(500)
        assert budget.consumed == 5000
        assert budget.utilization == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------


class TestTokenBudgetExceeded:
    def test_exception_message(self):
        exc = TokenBudgetExceeded(limit=5000, consumed=5500, last_node_tokens=800)
        assert "5,500" in str(exc)
        assert "5,000" in str(exc)
        assert "800" in str(exc)

    def test_exception_attributes(self):
        exc = TokenBudgetExceeded(limit=1000, consumed=1200, last_node_tokens=300)
        assert exc.limit == 1000
        assert exc.consumed == 1200
        assert exc.last_node_tokens == 300


# ---------------------------------------------------------------------------
# Integration tests: Executor + TokenBudget
# ---------------------------------------------------------------------------


class DummyRuntime:
    def start_run(self, **kwargs):
        return "run-1"

    def end_run(self, **kwargs):
        pass

    def report_problem(self, **kwargs):
        pass


class TokenConsumingNode(NodeProtocol):
    """Node that consumes a configurable number of tokens."""

    def __init__(self, tokens: int = 500):
        self._tokens = tokens

    def validate_input(self, ctx: NodeContext) -> list[str]:
        return []

    async def execute(self, ctx: NodeContext) -> NodeResult:
        output = dict.fromkeys(ctx.node_spec.output_keys, "done")
        return NodeResult(
            success=True,
            output=output,
            tokens_used=self._tokens,
            latency_ms=10,
        )


@pytest.mark.asyncio
class TestTokenBudgetExecutorIntegration:
    """Test token budget enforcement through the executor."""

    async def test_budget_allows_execution_within_limit(self):
        """Execution succeeds when total tokens stay within budget."""
        graph = GraphSpec(
            id="g1",
            goal_id="goal1",
            nodes=[
                NodeSpec(
                    id="n1",
                    name="node1",
                    description="test",
                    node_type="function",
                    input_keys=[],
                    output_keys=["result"],
                    max_retries=0,
                ),
            ],
            edges=[],
            entry_node="n1",
            token_budget=1000,
        )

        executor = GraphExecutor(
            runtime=DummyRuntime(),
            node_registry={"n1": TokenConsumingNode(tokens=500)},
        )
        goal = Goal(id="goal1", name="test", description="test")

        result = await executor.execute(graph=graph, goal=goal)
        assert result.success is True
        assert result.total_tokens == 500

    async def test_budget_aborts_on_exceed(self):
        """Execution fails when token budget is exceeded."""
        graph = GraphSpec(
            id="g1",
            goal_id="goal1",
            nodes=[
                NodeSpec(
                    id="n1",
                    name="step1",
                    description="first",
                    node_type="function",
                    input_keys=[],
                    output_keys=["intermediate"],
                    max_retries=0,
                ),
                NodeSpec(
                    id="n2",
                    name="step2",
                    description="second",
                    node_type="function",
                    input_keys=["intermediate"],
                    output_keys=["result"],
                    max_retries=0,
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1",
                    source="n1",
                    target="n2",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
            ],
            entry_node="n1",
            token_budget=800,  # 600 + 600 = 1200 > 800
        )

        executor = GraphExecutor(
            runtime=DummyRuntime(),
            node_registry={
                "n1": TokenConsumingNode(tokens=600),
                "n2": TokenConsumingNode(tokens=600),
            },
        )
        goal = Goal(id="goal1", name="test", description="test")

        result = await executor.execute(graph=graph, goal=goal)
        assert result.success is False
        assert "Token budget exceeded" in result.error

    async def test_no_budget_means_unlimited(self):
        """Without token_budget, execution is not limited."""
        graph = GraphSpec(
            id="g1",
            goal_id="goal1",
            nodes=[
                NodeSpec(
                    id="n1",
                    name="node1",
                    description="test",
                    node_type="function",
                    input_keys=[],
                    output_keys=["result"],
                    max_retries=0,
                ),
            ],
            edges=[],
            entry_node="n1",
            token_budget=None,  # No budget
        )

        executor = GraphExecutor(
            runtime=DummyRuntime(),
            node_registry={"n1": TokenConsumingNode(tokens=999_999)},
        )
        goal = Goal(id="goal1", name="test", description="test")

        result = await executor.execute(graph=graph, goal=goal)
        assert result.success is True

    async def test_budget_passed_to_node_context(self):
        """Token budget is available on NodeContext for event_loop nodes."""

        class ContextInspectorNode(NodeProtocol):
            token_budget_seen = None

            def validate_input(self, ctx):
                return []

            async def execute(self, ctx):
                ContextInspectorNode.token_budget_seen = ctx.token_budget
                return NodeResult(
                    success=True,
                    output={"result": "inspected"},
                    tokens_used=100,
                    latency_ms=1,
                )

        graph = GraphSpec(
            id="g1",
            goal_id="goal1",
            nodes=[
                NodeSpec(
                    id="n1",
                    name="inspector",
                    description="test",
                    node_type="function",
                    input_keys=[],
                    output_keys=["result"],
                    max_retries=0,
                ),
            ],
            edges=[],
            entry_node="n1",
            token_budget=5000,
        )

        executor = GraphExecutor(
            runtime=DummyRuntime(),
            node_registry={"n1": ContextInspectorNode()},
        )
        goal = Goal(id="goal1", name="test", description="test")

        await executor.execute(graph=graph, goal=goal)
        assert ContextInspectorNode.token_budget_seen is not None
        assert isinstance(ContextInspectorNode.token_budget_seen, TokenBudget)
        assert ContextInspectorNode.token_budget_seen.limit == 5000

    async def test_graphspec_token_budget_field(self):
        """GraphSpec accepts token_budget in constructor."""
        graph = GraphSpec(
            id="g1",
            goal_id="goal1",
            nodes=[],
            edges=[],
            entry_node="n1",
            token_budget=10_000,
        )
        assert graph.token_budget == 10_000

    async def test_graphspec_token_budget_defaults_to_none(self):
        """GraphSpec.token_budget defaults to None (unlimited)."""
        graph = GraphSpec(
            id="g1",
            goal_id="goal1",
            nodes=[],
            edges=[],
            entry_node="n1",
        )
        assert graph.token_budget is None

    async def test_budget_first_node_succeeds_second_exceeds(self):
        """First node completes within budget, second node triggers exceed."""
        graph = GraphSpec(
            id="g1",
            goal_id="goal1",
            nodes=[
                NodeSpec(
                    id="n1",
                    name="cheap",
                    description="low cost",
                    node_type="function",
                    input_keys=[],
                    output_keys=["intermediate"],
                    max_retries=0,
                ),
                NodeSpec(
                    id="n2",
                    name="expensive",
                    description="high cost",
                    node_type="function",
                    input_keys=["intermediate"],
                    output_keys=["result"],
                    max_retries=0,
                ),
            ],
            edges=[
                EdgeSpec(
                    id="e1",
                    source="n1",
                    target="n2",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
            ],
            entry_node="n1",
            token_budget=500,  # n1=200 OK, n2=400 → 600 > 500
        )

        executor = GraphExecutor(
            runtime=DummyRuntime(),
            node_registry={
                "n1": TokenConsumingNode(tokens=200),
                "n2": TokenConsumingNode(tokens=400),
            },
        )
        goal = Goal(id="goal1", name="test", description="test")

        result = await executor.execute(graph=graph, goal=goal)
        assert result.success is False
        assert "Token budget exceeded" in result.error
        # First node should have completed (200 tokens recorded)
        assert result.steps_executed >= 1

    async def test_budget_tracks_cumulative_across_nodes(self):
        """Budget correctly accumulates tokens across multiple nodes."""

        class TrackingNode(NodeProtocol):
            def validate_input(self, ctx):
                return []

            async def execute(self, ctx):
                output = dict.fromkeys(ctx.node_spec.output_keys, "ok")
                return NodeResult(success=True, output=output, tokens_used=100, latency_ms=1)

        graph = GraphSpec(
            id="g1",
            goal_id="goal1",
            nodes=[
                NodeSpec(
                    id="n1",
                    name="step1",
                    description="first",
                    node_type="function",
                    input_keys=[],
                    output_keys=["a"],
                    max_retries=0,
                ),
                NodeSpec(
                    id="n2",
                    name="step2",
                    description="second",
                    node_type="function",
                    input_keys=["a"],
                    output_keys=["b"],
                    max_retries=0,
                ),
                NodeSpec(
                    id="n3",
                    name="step3",
                    description="third",
                    node_type="function",
                    input_keys=["b"],
                    output_keys=["result"],
                    max_retries=0,
                ),
            ],
            edges=[
                EdgeSpec(id="e1", source="n1", target="n2", condition=EdgeCondition.ON_SUCCESS),
                EdgeSpec(id="e2", source="n2", target="n3", condition=EdgeCondition.ON_SUCCESS),
            ],
            entry_node="n1",
            token_budget=350,  # 100+100+100=300 < 350, should pass
        )

        executor = GraphExecutor(
            runtime=DummyRuntime(),
            node_registry={
                "n1": TrackingNode(),
                "n2": TrackingNode(),
                "n3": TrackingNode(),
            },
        )
        goal = Goal(id="goal1", name="test", description="test")

        result = await executor.execute(graph=graph, goal=goal)
        assert result.success is True
        assert result.total_tokens == 300
