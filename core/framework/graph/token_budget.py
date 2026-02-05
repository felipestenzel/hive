"""Token Budget - Enforces token usage limits during graph execution.

Tracks cumulative token consumption across node executions and aborts
when a configured budget is exceeded. Works at two levels:

1. **Executor level** - checked after each node completes
2. **EventLoopNode level** - checked after each LLM turn within a node

The budget integrates naturally with the existing token tracking in
``NodeResult.tokens_used`` and ``ExecutionResult.total_tokens``.

Example:
    Budget enforcement via CLI::

        hive run exports/my-agent --max-tokens 10000

    Budget enforcement via GraphSpec::

        graph = GraphSpec(
            ...,
            token_budget=10000,
        )

    Programmatic usage::

        budget = TokenBudget(limit=10000)
        budget.record(500)   # OK
        budget.record(600)   # OK
        budget.remaining     # 8900
        budget.utilization   # 0.11
"""

import logging

logger = logging.getLogger(__name__)

_WARNING_THRESHOLD = 0.8  # Log warning at 80% utilization


class TokenBudgetExceeded(Exception):
    """Raised when cumulative token usage exceeds the configured budget.

    Attributes:
        limit: The configured token budget.
        consumed: Total tokens consumed when the limit was hit.
        last_node_tokens: Tokens used by the node that triggered the breach.
    """

    def __init__(self, limit: int, consumed: int, last_node_tokens: int = 0) -> None:
        self.limit = limit
        self.consumed = consumed
        self.last_node_tokens = last_node_tokens
        super().__init__(
            f"Token budget exceeded: {consumed:,} tokens used "
            f"(budget: {limit:,}, last node used {last_node_tokens:,})"
        )


class TokenBudget:
    """Tracks and enforces a token usage limit during execution.

    Args:
        limit: Maximum tokens allowed for the entire execution.
            Must be a positive integer.

    Raises:
        ValueError: If limit is not a positive integer.

    Example:
        >>> budget = TokenBudget(limit=5000)
        >>> budget.record(1200)
        >>> budget.remaining
        3800
        >>> budget.utilization
        0.24
        >>> budget.record(4000)  # doctest: +SKIP
        Traceback: TokenBudgetExceeded
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"token budget limit must be >= 1, got {limit}")
        self._limit = limit
        self._consumed = 0
        self._warned = False

    @property
    def limit(self) -> int:
        """The configured token budget."""
        return self._limit

    @property
    def consumed(self) -> int:
        """Total tokens consumed so far."""
        return self._consumed

    @property
    def remaining(self) -> int:
        """Tokens remaining before the budget is exhausted."""
        return max(0, self._limit - self._consumed)

    @property
    def utilization(self) -> float:
        """Fraction of budget consumed (0.0 to 1.0+)."""
        return self._consumed / self._limit

    def record(self, tokens: int) -> None:
        """Record token usage and enforce the budget.

        Args:
            tokens: Number of tokens consumed by the latest operation.
                Must be non-negative.

        Raises:
            ValueError: If tokens is negative.
            TokenBudgetExceeded: If cumulative usage exceeds the limit.
        """
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        self._consumed += tokens

        # Warn once when crossing 80% threshold
        if not self._warned and self._consumed >= self._limit * _WARNING_THRESHOLD:
            self._warned = True
            logger.warning(
                "Token budget at %.0f%% (%s/%s)",
                self.utilization * 100,
                f"{self._consumed:,}",
                f"{self._limit:,}",
            )

        if self._consumed > self._limit:
            raise TokenBudgetExceeded(
                limit=self._limit,
                consumed=self._consumed,
                last_node_tokens=tokens,
            )

    def would_exceed(self, estimated_tokens: int) -> bool:
        """Check whether an additional operation would exceed the budget.

        Args:
            estimated_tokens: Estimated tokens for the upcoming operation.

        Returns:
            True if consuming estimated_tokens would exceed the budget.
        """
        return (self._consumed + estimated_tokens) > self._limit
