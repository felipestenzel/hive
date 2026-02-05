"""Circuit Breaker pattern for resilient MCP tool calls.

Implements the Circuit Breaker pattern to prevent cascading failures
when MCP servers are degraded or unavailable. Each MCP server gets
an independent circuit that tracks its health.

State Machine::

    CLOSED ──(failures >= threshold)──► OPEN
      ▲                                   │
      │ (successes >= threshold)   (recovery_timeout)
      │                                   ▼
      └──────────── HALF_OPEN ◄───────────┘

States:
    CLOSED    - Normal operation, all requests flow through.
    OPEN      - Server is degraded, requests fail immediately (fail-fast).
    HALF_OPEN - Recovery probe: limited requests allowed to test if
                the server has recovered.

Usage:
    from framework.runner.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

    if not breaker.allow_request("tools-server"):
        raise CircuitOpenError("tools-server")

    try:
        result = call_mcp_server(...)
        breaker.record_success("tools-server")
    except Exception as e:
        breaker.record_failure("tools-server", e)
        raise

    # Health monitoring
    health = breaker.get_health_report()
    # {"tools-server": {"state": "closed", "failure_count": 0, ...}}
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    """Circuit breaker states.

    Transitions:
        CLOSED → OPEN: When failure_count >= failure_threshold.
        OPEN → HALF_OPEN: When recovery_timeout seconds have elapsed.
        HALF_OPEN → CLOSED: When consecutive successes >= success_threshold.
        HALF_OPEN → OPEN: On any failure during recovery probing.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a circuit is open and the request should fail fast.

    This avoids waiting for network timeouts on servers known to be degraded.
    The caller should handle this by skipping the tool or using a fallback.

    Attributes:
        server_name: The MCP server that is unavailable.
        message: Human-readable error description.
    """

    def __init__(self, server_name: str, message: str | None = None):
        self.server_name = server_name
        self.message = message or f"Circuit open for server '{server_name}'"
        super().__init__(self.message)


@dataclass
class CircuitStats:
    """Statistics for a single circuit."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    last_error: str | None = None
    consecutive_successes: int = 0
    half_open_calls: int = 0

    # Timestamps for state transitions
    opened_at: float | None = None
    half_opened_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for reporting."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "last_error": self.last_error,
            "consecutive_successes": self.consecutive_successes,
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Number of failures before opening circuit
    failure_threshold: int = 5

    # Seconds to wait before trying again (OPEN -> HALF_OPEN)
    recovery_timeout: float = 30.0

    # Number of successful calls in HALF_OPEN to close circuit
    success_threshold: int = 3

    # Maximum calls allowed in HALF_OPEN state
    half_open_max_calls: int = 3

    # Optional: Exceptions that should NOT count as failures
    # (e.g., validation errors are not server failures)
    ignored_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)


class CircuitBreaker:
    """Circuit Breaker for MCP server resilience.

    Tracks failures per server and prevents cascading failures by
    failing fast when a server is known to be degraded.

    Thread-safe implementation for concurrent agent executions.

    Example:
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            success_threshold=3,
        )

        # In MCP client
        if breaker.is_open("tools-server"):
            raise CircuitOpenError("tools-server")

        try:
            result = mcp_client.call_tool(...)
            breaker.record_success("tools-server")
            return result
        except Exception as e:
            breaker.record_failure("tools-server", e)
            raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
        half_open_max_calls: int = 3,
        ignored_exceptions: tuple[type[Exception], ...] | None = None,
    ):
        """Initialize circuit breaker with validation.

        Args:
            failure_threshold: Failures before opening circuit (must be >= 1).
            recovery_timeout: Seconds before trying recovery (must be > 0).
            success_threshold: Successes in HALF_OPEN to close (must be >= 1).
            half_open_max_calls: Max probe calls in HALF_OPEN (must be >= 1).
            ignored_exceptions: Exception types that don't count as failures
                (e.g., ValueError for input validation errors).

        Raises:
            ValueError: If any threshold or timeout is invalid.
        """
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if recovery_timeout <= 0:
            raise ValueError(f"recovery_timeout must be > 0, got {recovery_timeout}")
        if success_threshold < 1:
            raise ValueError(f"success_threshold must be >= 1, got {success_threshold}")
        if half_open_max_calls < 1:
            raise ValueError(f"half_open_max_calls must be >= 1, got {half_open_max_calls}")

        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            half_open_max_calls=half_open_max_calls,
            ignored_exceptions=ignored_exceptions or (),
        )

        # Per-server circuit stats
        self._circuits: dict[str, CircuitStats] = {}
        self._lock = threading.RLock()

    def _get_circuit(self, server_name: str) -> CircuitStats:
        """Get or create circuit stats for a server."""
        if server_name not in self._circuits:
            self._circuits[server_name] = CircuitStats()
        return self._circuits[server_name]

    def get_state(self, server_name: str) -> CircuitState:
        """Get current circuit state for a server.

        May transition from OPEN to HALF_OPEN if recovery timeout passed.

        Args:
            server_name: Name of the MCP server

        Returns:
            Current CircuitState
        """
        with self._lock:
            circuit = self._get_circuit(server_name)

            # Check for automatic transition OPEN -> HALF_OPEN
            if circuit.state == CircuitState.OPEN:
                if circuit.opened_at is not None:
                    elapsed = time.time() - circuit.opened_at
                    if elapsed >= self.config.recovery_timeout:
                        self._transition_to_half_open(server_name, circuit)

            return circuit.state

    def is_open(self, server_name: str) -> bool:
        """Check if circuit is open (should fail fast).

        Args:
            server_name: Name of the MCP server

        Returns:
            True if requests should fail fast
        """
        state = self.get_state(server_name)
        return state == CircuitState.OPEN

    def is_closed(self, server_name: str) -> bool:
        """Check if circuit is closed (normal operation).

        Args:
            server_name: Name of the MCP server

        Returns:
            True if requests should proceed normally
        """
        return self.get_state(server_name) == CircuitState.CLOSED

    def allow_request(self, server_name: str) -> bool:
        """Check if a request should be allowed through.

        In HALF_OPEN state, only allows limited requests for probing.

        Args:
            server_name: Name of the MCP server

        Returns:
            True if request should proceed
        """
        with self._lock:
            state = self.get_state(server_name)

            if state == CircuitState.CLOSED:
                return True

            if state == CircuitState.OPEN:
                return False

            # HALF_OPEN: allow limited probe requests
            circuit = self._get_circuit(server_name)
            if circuit.half_open_calls < self.config.half_open_max_calls:
                circuit.half_open_calls += 1
                return True

            return False

    def record_success(self, server_name: str) -> None:
        """Record a successful call to a server.

        In HALF_OPEN state, may transition to CLOSED after enough successes.

        Args:
            server_name: Name of the MCP server
        """
        with self._lock:
            circuit = self._get_circuit(server_name)
            circuit.success_count += 1
            circuit.last_success_time = time.time()
            circuit.consecutive_successes += 1

            if circuit.state == CircuitState.HALF_OPEN:
                if circuit.consecutive_successes >= self.config.success_threshold:
                    self._transition_to_closed(server_name, circuit)
                    logger.info(
                        "Circuit CLOSED for '%s' after %d consecutive successes",
                        server_name,
                        circuit.consecutive_successes,
                    )

    def record_failure(
        self,
        server_name: str,
        error: Exception | None = None,
    ) -> None:
        """Record a failed call to a server.

        May transition circuit to OPEN after threshold failures.

        Args:
            server_name: Name of the MCP server
            error: The exception that occurred
        """
        # Check if this exception type should be ignored
        if error is not None and isinstance(error, self.config.ignored_exceptions):
            logger.debug(
                "Ignoring %s for circuit breaker (server: %s)",
                type(error).__name__,
                server_name,
            )
            return

        with self._lock:
            circuit = self._get_circuit(server_name)
            circuit.failure_count += 1
            circuit.last_failure_time = time.time()
            circuit.last_error = str(error) if error else None
            circuit.consecutive_successes = 0

            if circuit.state == CircuitState.CLOSED:
                if circuit.failure_count >= self.config.failure_threshold:
                    self._transition_to_open(server_name, circuit)
                    logger.warning(
                        "Circuit OPEN for '%s' after %d failures. Last error: %s",
                        server_name,
                        circuit.failure_count,
                        circuit.last_error,
                    )

            elif circuit.state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN reopens the circuit
                self._transition_to_open(server_name, circuit)
                logger.warning(
                    "Circuit reopened for '%s' after failure in HALF_OPEN state",
                    server_name,
                )

    def _transition_to_open(self, server_name: str, circuit: CircuitStats) -> None:
        """Transition to OPEN: all requests will fail fast."""
        circuit.state = CircuitState.OPEN
        circuit.opened_at = time.time()
        circuit.half_opened_at = None
        circuit.half_open_calls = 0

    def _transition_to_half_open(self, server_name: str, circuit: CircuitStats) -> None:
        """Transition to HALF_OPEN: allow limited probe requests."""
        circuit.state = CircuitState.HALF_OPEN
        circuit.half_opened_at = time.time()
        circuit.half_open_calls = 0
        circuit.consecutive_successes = 0
        logger.info("Circuit HALF_OPEN for '%s', allowing probe requests", server_name)

    def _transition_to_closed(self, server_name: str, circuit: CircuitStats) -> None:
        """Transition to CLOSED: normal operation, counters reset."""
        circuit.state = CircuitState.CLOSED
        circuit.opened_at = None
        circuit.half_opened_at = None
        circuit.half_open_calls = 0
        circuit.failure_count = 0  # Reset failure count on close

    def reset(self, server_name: str) -> None:
        """Manually reset a circuit to CLOSED state.

        Useful for administrative recovery or testing.

        Args:
            server_name: Name of the MCP server
        """
        with self._lock:
            if server_name in self._circuits:
                circuit = self._circuits[server_name]
                self._transition_to_closed(server_name, circuit)
                logger.info("Circuit manually reset for '%s'", server_name)

    def reset_all(self) -> None:
        """Reset all circuits to CLOSED state."""
        with self._lock:
            for server_name, circuit in self._circuits.items():
                self._transition_to_closed(server_name, circuit)
                logger.info("Circuit manually reset for '%s'", server_name)

    def get_health_report(self) -> dict[str, dict[str, Any]]:
        """Get health report for all tracked servers.

        Returns:
            Dict mapping server_name to health stats
        """
        with self._lock:
            return {
                server_name: circuit.to_dict() for server_name, circuit in self._circuits.items()
            }

    def get_server_health(self, server_name: str) -> dict[str, Any]:
        """Get health report for a specific server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Health stats dictionary
        """
        with self._lock:
            circuit = self._get_circuit(server_name)
            return circuit.to_dict()


# Global circuit breaker instance for shared state across MCPClient instances
_global_circuit_breaker: CircuitBreaker | None = None


def get_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    success_threshold: int = 3,
    half_open_max_calls: int = 3,
    ignored_exceptions: tuple[type[Exception], ...] | None = None,
) -> CircuitBreaker:
    """Get or create the global circuit breaker instance.

    Uses singleton pattern to ensure all MCP clients share the same
    circuit breaker state.

    Args:
        failure_threshold: Failures before opening (only used on first call).
        recovery_timeout: Recovery timeout in seconds (only used on first call).
        success_threshold: Successes in HALF_OPEN to close (only used on first call).
        half_open_max_calls: Max probe calls in HALF_OPEN (only used on first call).
        ignored_exceptions: Exception types that don't count as failures
            (only used on first call).

    Returns:
        Global CircuitBreaker instance.
    """
    global _global_circuit_breaker
    if _global_circuit_breaker is None:
        _global_circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            half_open_max_calls=half_open_max_calls,
            ignored_exceptions=ignored_exceptions,
        )
    return _global_circuit_breaker


def reset_global_circuit_breaker() -> None:
    """Reset the global circuit breaker (useful for testing)."""
    global _global_circuit_breaker
    _global_circuit_breaker = None
