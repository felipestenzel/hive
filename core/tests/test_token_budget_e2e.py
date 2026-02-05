"""
End-to-end tests for token budget enforcement during graph execution.

Covers:
- Multi-node sequential pipeline with budget exhaustion
- Budget enforcement across node retries
- Budget tracking across edges with data mapping
- Zero-budget / exact-limit edge cases
"""

from unittest.mock import MagicMock

import pytest

from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
from framework.graph.executor import GraphExecutor
from framework.graph.goal import Goal
from framework.graph.node import NodeContext, NodeProtocol, NodeResult, NodeSpec
from framework.runtime.core import Runtime

# ---------------------------------------------------------------------------
# Test node implementations
# ---------------------------------------------------------------------------


class TokenConsumingNode(NodeProtocol):
    """A node that succeeds and reports a configurable amount of token usage.

    Writes outputs to shared memory (matching FunctionNode behaviour) so that
    downstream nodes and the final ExecutionResult can see them.
    """

    def __init__(self, tokens: int, output: dict | None = None):
        self._tokens = tokens
        self._output = output or {}
        self.executed = False

    async def execute(self, ctx: NodeContext) -> NodeResult:
        self.executed = True
        # Populate output from spec keys if not explicitly provided
        output = dict(self._output)
        for key in ctx.node_spec.output_keys:
            if key not in output:
                output[key] = f"value_for_{key}"
        # Write to shared memory so downstream nodes / final output see these values
        for key, value in output.items():
            ctx.memory.write(key, value, validate=False)
        return NodeResult(success=True, output=output, tokens_used=self._tokens, latency_ms=1)


class FlakyTokenNode(NodeProtocol):
    """A node that fails N times then succeeds, using tokens on every attempt."""

    def __init__(self, fail_times: int, tokens_per_attempt: int, output: dict | None = None):
        self.fail_times = fail_times
        self.tokens_per_attempt = tokens_per_attempt
        self.attempt_count = 0
        self._output = output or {}

    async def execute(self, ctx: NodeContext) -> NodeResult:
        self.attempt_count += 1
        if self.attempt_count <= self.fail_times:
            return NodeResult(
                success=False,
                error=f"flaky failure #{self.attempt_count}",
                tokens_used=self.tokens_per_attempt,
                latency_ms=1,
            )
        output = dict(self._output)
        for key in ctx.node_spec.output_keys:
            if key not in output:
                output[key] = f"recovered_{key}"
        return NodeResult(
            success=True, output=output, tokens_used=self.tokens_per_attempt, latency_ms=1
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime():
    rt = MagicMock(spec=Runtime)
    rt.start_run = MagicMock(return_value="run_id")
    rt.decide = MagicMock(return_value="decision_id")
    rt.record_outcome = MagicMock()
    rt.end_run = MagicMock()
    rt.report_problem = MagicMock()
    rt.set_node = MagicMock()
    return rt


@pytest.fixture
def goal():
    return Goal(id="budget_test", name="Budget Test", description="Token budget e2e tests")


# ---------------------------------------------------------------------------
# Scenario 1: Multi-node pipeline with budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_budget_exhausted_at_third_node(runtime, goal):
    """Three sequential nodes each using 200 tokens. Budget=500 should fail at node 3.

    After node 1: 200 consumed, 300 remaining.
    After node 2: 400 consumed, 100 remaining.
    Node 3 uses 200 -> 600 consumed > 500 -> TokenBudgetExceeded.
    """
    node_a = NodeSpec(
        id="a", name="Node A", description="first", node_type="function", output_keys=["out_a"]
    )
    node_b = NodeSpec(
        id="b",
        name="Node B",
        description="second",
        node_type="function",
        output_keys=["out_b"],
    )
    node_c = NodeSpec(
        id="c",
        name="Node C",
        description="third",
        node_type="function",
        output_keys=["out_c"],
    )

    graph = GraphSpec(
        id="pipeline",
        goal_id=goal.id,
        name="Pipeline",
        entry_node="a",
        nodes=[node_a, node_b, node_c],
        edges=[
            EdgeSpec(id="a_b", source="a", target="b", condition=EdgeCondition.ON_SUCCESS),
            EdgeSpec(id="b_c", source="b", target="c", condition=EdgeCondition.ON_SUCCESS),
        ],
        terminal_nodes=["c"],
        token_budget=500,
    )

    executor = GraphExecutor(runtime=runtime)
    executor.register_node("a", TokenConsumingNode(tokens=200, output={"out_a": "data_a"}))
    executor.register_node("b", TokenConsumingNode(tokens=200, output={"out_b": "data_b"}))
    executor.register_node("c", TokenConsumingNode(tokens=200, output={"out_c": "data_c"}))

    result = await executor.execute(graph, goal, {})

    assert not result.success, "Should fail because 600 > 500"
    assert "Token budget exceeded" in result.error
    # The error message should reference the budget of 500
    assert "500" in result.error
    # total_tokens should reflect what was actually consumed (all three nodes ran)
    assert result.total_tokens == 600
    # Path should include all three nodes since node c executed before the budget check
    assert "c" in result.path


@pytest.mark.asyncio
async def test_pipeline_budget_sufficient(runtime, goal):
    """Three sequential nodes each using 200 tokens. Budget=700 should succeed."""
    node_a = NodeSpec(
        id="a", name="Node A", description="first", node_type="function", output_keys=["out_a"]
    )
    node_b = NodeSpec(
        id="b", name="Node B", description="second", node_type="function", output_keys=["out_b"]
    )
    node_c = NodeSpec(
        id="c", name="Node C", description="third", node_type="function", output_keys=["out_c"]
    )

    graph = GraphSpec(
        id="pipeline",
        goal_id=goal.id,
        name="Pipeline",
        entry_node="a",
        nodes=[node_a, node_b, node_c],
        edges=[
            EdgeSpec(id="a_b", source="a", target="b", condition=EdgeCondition.ON_SUCCESS),
            EdgeSpec(id="b_c", source="b", target="c", condition=EdgeCondition.ON_SUCCESS),
        ],
        terminal_nodes=["c"],
        token_budget=700,
    )

    executor = GraphExecutor(runtime=runtime)
    executor.register_node("a", TokenConsumingNode(tokens=200, output={"out_a": "data_a"}))
    executor.register_node("b", TokenConsumingNode(tokens=200, output={"out_b": "data_b"}))
    executor.register_node("c", TokenConsumingNode(tokens=200, output={"out_c": "data_c"}))

    result = await executor.execute(graph, goal, {})

    assert result.success
    assert result.total_tokens == 600
    assert result.path == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Scenario 2: Budget with node retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exceeded_on_retry(runtime, goal):
    """A flaky node that fails once then succeeds, using 300 tokens per attempt.

    Budget = 500.
    Attempt 1 (fail): 300 tokens recorded -> 300 consumed.
    Attempt 2 (success): 300 tokens recorded -> 600 consumed > 500 -> budget exceeded.
    """
    node_spec = NodeSpec(
        id="flaky",
        name="Flaky Node",
        description="fails once then succeeds",
        node_type="function",
        output_keys=["result"],
        max_retries=3,
    )

    graph = GraphSpec(
        id="retry_graph",
        goal_id=goal.id,
        name="Retry Graph",
        entry_node="flaky",
        nodes=[node_spec],
        edges=[],
        terminal_nodes=["flaky"],
        token_budget=500,
    )

    executor = GraphExecutor(runtime=runtime)
    flaky = FlakyTokenNode(fail_times=1, tokens_per_attempt=300)
    executor.register_node("flaky", flaky)

    result = await executor.execute(graph, goal, {})

    # The first attempt (failure) records 300 tokens.
    # The second attempt (success) records another 300 tokens -> 600 > 500.
    # Budget check happens after the node returns, so the second attempt
    # completes successfully but the budget check triggers.
    assert not result.success
    assert "Token budget exceeded" in result.error
    assert result.total_tokens == 600
    assert flaky.attempt_count == 2


@pytest.mark.asyncio
async def test_budget_survives_retries_when_sufficient(runtime, goal):
    """Flaky node with enough budget for both attempts should succeed."""
    node_spec = NodeSpec(
        id="flaky",
        name="Flaky Node",
        description="fails once then succeeds",
        node_type="function",
        output_keys=["result"],
        max_retries=3,
    )

    graph = GraphSpec(
        id="retry_graph",
        goal_id=goal.id,
        name="Retry Graph",
        entry_node="flaky",
        nodes=[node_spec],
        edges=[],
        terminal_nodes=["flaky"],
        token_budget=700,
    )

    executor = GraphExecutor(runtime=runtime)
    flaky = FlakyTokenNode(fail_times=1, tokens_per_attempt=300)
    executor.register_node("flaky", flaky)

    result = await executor.execute(graph, goal, {})

    assert result.success
    assert result.total_tokens == 600
    assert flaky.attempt_count == 2


# ---------------------------------------------------------------------------
# Scenario 3: Budget survives across edges with data mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_tracked_across_edges_with_data_mapping(runtime, goal):
    """Node A produces 'data', Node B reads 'data'. Both consume tokens.

    Node A uses 150 tokens.
    Node B uses 150 tokens.
    Budget = 400 should be sufficient (300 total).
    """
    node_a = NodeSpec(
        id="producer",
        name="Producer",
        description="produces data",
        node_type="function",
        output_keys=["data"],
    )
    # input_keys left empty: data flows via shared memory written by the
    # producer node.  Restricting input_keys would scope the memory view and
    # block reads to internal keys needed by the executor (same pattern as
    # test_context_handoff_between_nodes in test_event_loop_integration.py).
    node_b = NodeSpec(
        id="consumer",
        name="Consumer",
        description="consumes data",
        node_type="function",
        output_keys=["result"],
    )

    graph = GraphSpec(
        id="data_flow",
        goal_id=goal.id,
        name="Data Flow",
        entry_node="producer",
        nodes=[node_a, node_b],
        edges=[
            EdgeSpec(
                id="p_to_c",
                source="producer",
                target="consumer",
                condition=EdgeCondition.ON_SUCCESS,
            ),
        ],
        terminal_nodes=["consumer"],
        token_budget=400,
    )

    executor = GraphExecutor(runtime=runtime)
    executor.register_node("producer", TokenConsumingNode(tokens=150, output={"data": "payload"}))
    executor.register_node(
        "consumer", TokenConsumingNode(tokens=150, output={"result": "processed"})
    )

    result = await executor.execute(graph, goal, {})

    assert result.success
    assert result.total_tokens == 300
    assert result.path == ["producer", "consumer"]
    # Verify the data flowed correctly through memory
    assert result.output.get("data") == "payload"
    assert result.output.get("result") == "processed"


@pytest.mark.asyncio
async def test_budget_exhausted_across_edges(runtime, goal):
    """Node A uses 250, Node B uses 300. Budget=500 should fail at Node B."""
    node_a = NodeSpec(
        id="producer",
        name="Producer",
        description="produces data",
        node_type="function",
        output_keys=["data"],
    )
    node_b = NodeSpec(
        id="consumer",
        name="Consumer",
        description="consumes data",
        node_type="function",
        output_keys=["result"],
    )

    graph = GraphSpec(
        id="data_flow",
        goal_id=goal.id,
        name="Data Flow",
        entry_node="producer",
        nodes=[node_a, node_b],
        edges=[
            EdgeSpec(
                id="p_to_c",
                source="producer",
                target="consumer",
                condition=EdgeCondition.ON_SUCCESS,
            ),
        ],
        terminal_nodes=["consumer"],
        token_budget=500,
    )

    executor = GraphExecutor(runtime=runtime)
    executor.register_node("producer", TokenConsumingNode(tokens=250, output={"data": "payload"}))
    executor.register_node(
        "consumer", TokenConsumingNode(tokens=300, output={"result": "processed"})
    )

    result = await executor.execute(graph, goal, {})

    assert not result.success
    assert "Token budget exceeded" in result.error
    assert result.total_tokens == 550
    assert "consumer" in result.path


# ---------------------------------------------------------------------------
# Scenario 4: Zero-budget edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_one_node_uses_zero_tokens(runtime, goal):
    """token_budget=1 and a node using 0 tokens should pass."""
    node_spec = NodeSpec(
        id="zero",
        name="Zero Token Node",
        description="uses no tokens",
        node_type="function",
        output_keys=["result"],
    )

    graph = GraphSpec(
        id="zero_graph",
        goal_id=goal.id,
        name="Zero Graph",
        entry_node="zero",
        nodes=[node_spec],
        edges=[],
        terminal_nodes=["zero"],
        token_budget=1,
    )

    executor = GraphExecutor(runtime=runtime)
    executor.register_node("zero", TokenConsumingNode(tokens=0, output={"result": "ok"}))

    result = await executor.execute(graph, goal, {})

    assert result.success
    assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_budget_exact_limit(runtime, goal):
    """token_budget=1 and a node using exactly 1 token should pass (consumed == limit is OK)."""
    node_spec = NodeSpec(
        id="exact",
        name="Exact Limit Node",
        description="uses exactly the budget",
        node_type="function",
        output_keys=["result"],
    )

    graph = GraphSpec(
        id="exact_graph",
        goal_id=goal.id,
        name="Exact Graph",
        entry_node="exact",
        nodes=[node_spec],
        edges=[],
        terminal_nodes=["exact"],
        token_budget=1,
    )

    executor = GraphExecutor(runtime=runtime)
    executor.register_node("exact", TokenConsumingNode(tokens=1, output={"result": "ok"}))

    result = await executor.execute(graph, goal, {})

    # TokenBudget.record() raises only when consumed > limit, so exact == limit is fine.
    assert result.success
    assert result.total_tokens == 1


@pytest.mark.asyncio
async def test_budget_exceeded_by_one(runtime, goal):
    """token_budget=1 and a node using 2 tokens should fail."""
    node_spec = NodeSpec(
        id="over",
        name="Over Budget Node",
        description="exceeds budget by 1",
        node_type="function",
        output_keys=["result"],
    )

    graph = GraphSpec(
        id="over_graph",
        goal_id=goal.id,
        name="Over Graph",
        entry_node="over",
        nodes=[node_spec],
        edges=[],
        terminal_nodes=["over"],
        token_budget=1,
    )

    executor = GraphExecutor(runtime=runtime)
    executor.register_node("over", TokenConsumingNode(tokens=2, output={"result": "ok"}))

    result = await executor.execute(graph, goal, {})

    assert not result.success
    assert "Token budget exceeded" in result.error
    assert result.total_tokens == 2
