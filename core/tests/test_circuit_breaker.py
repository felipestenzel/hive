"""Tests for the Circuit Breaker pattern implementation."""

import threading
import time

import pytest

from framework.runner.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    get_circuit_breaker,
    reset_global_circuit_breaker,
)


class TestCircuitBreakerBasic:
    """Basic tests for CircuitBreaker functionality."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with low thresholds for testing."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2,
            half_open_max_calls=2,
        )

    def test_initial_state_is_closed(self, breaker):
        """Circuit should start in CLOSED state."""
        assert breaker.get_state("test-server") == CircuitState.CLOSED
        assert breaker.is_closed("test-server")
        assert not breaker.is_open("test-server")

    def test_allows_requests_when_closed(self, breaker):
        """Should allow requests when circuit is closed."""
        assert breaker.allow_request("test-server")

    def test_success_keeps_circuit_closed(self, breaker):
        """Recording successes should keep circuit closed."""
        for _ in range(10):
            breaker.record_success("test-server")

        assert breaker.is_closed("test-server")
        assert breaker.get_server_health("test-server")["success_count"] == 10

    def test_failures_below_threshold_keep_closed(self, breaker):
        """Failures below threshold should not open circuit."""
        breaker.record_failure("test-server", Exception("error1"))
        breaker.record_failure("test-server", Exception("error2"))

        assert breaker.is_closed("test-server")
        assert breaker.get_server_health("test-server")["failure_count"] == 2


class TestCircuitOpenTransition:
    """Tests for CLOSED -> OPEN transition."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with low threshold."""
        return CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    def test_opens_after_threshold_failures(self, breaker):
        """Circuit should open after reaching failure threshold."""
        for i in range(3):
            breaker.record_failure("test-server", Exception(f"error{i}"))

        assert breaker.is_open("test-server")
        assert breaker.get_state("test-server") == CircuitState.OPEN

    def test_blocks_requests_when_open(self, breaker):
        """Should block requests when circuit is open."""
        # Open the circuit
        for i in range(3):
            breaker.record_failure("test-server", Exception(f"error{i}"))

        assert not breaker.allow_request("test-server")

    def test_records_last_error(self, breaker):
        """Should record the last error message."""
        breaker.record_failure("test-server", ValueError("specific error"))

        health = breaker.get_server_health("test-server")
        assert "specific error" in health["last_error"]


class TestCircuitHalfOpenTransition:
    """Tests for OPEN -> HALF_OPEN transition."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with short recovery timeout."""
        return CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms for fast tests
            success_threshold=2,
            half_open_max_calls=2,
        )

    def test_transitions_to_half_open_after_timeout(self, breaker):
        """Circuit should transition to HALF_OPEN after recovery timeout."""
        # Open the circuit
        breaker.record_failure("test-server", Exception("e1"))
        breaker.record_failure("test-server", Exception("e2"))
        assert breaker.is_open("test-server")

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should now be HALF_OPEN
        assert breaker.get_state("test-server") == CircuitState.HALF_OPEN

    def test_allows_limited_requests_in_half_open(self, breaker):
        """Should allow limited probe requests in HALF_OPEN state."""
        # Open and wait for half-open
        breaker.record_failure("test-server", Exception("e1"))
        breaker.record_failure("test-server", Exception("e2"))
        time.sleep(0.15)

        # Should allow up to half_open_max_calls (2)
        assert breaker.allow_request("test-server")
        assert breaker.allow_request("test-server")
        assert not breaker.allow_request("test-server")  # Third should fail


class TestCircuitRecovery:
    """Tests for HALF_OPEN -> CLOSED recovery."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with fast settings."""
        return CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=2,
            half_open_max_calls=3,
        )

    def test_closes_after_success_threshold(self, breaker):
        """Circuit should close after enough successes in HALF_OPEN."""
        # Open and wait for half-open
        breaker.record_failure("test-server", Exception("e1"))
        breaker.record_failure("test-server", Exception("e2"))
        time.sleep(0.15)
        assert breaker.get_state("test-server") == CircuitState.HALF_OPEN

        # Record successes
        breaker.record_success("test-server")
        assert breaker.get_state("test-server") == CircuitState.HALF_OPEN

        breaker.record_success("test-server")
        assert breaker.get_state("test-server") == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self, breaker):
        """Circuit should reopen on any failure in HALF_OPEN state."""
        # Open and wait for half-open
        breaker.record_failure("test-server", Exception("e1"))
        breaker.record_failure("test-server", Exception("e2"))
        time.sleep(0.15)
        assert breaker.get_state("test-server") == CircuitState.HALF_OPEN

        # Fail during probe
        breaker.record_failure("test-server", Exception("probe failed"))

        assert breaker.is_open("test-server")

    def test_resets_failure_count_on_close(self, breaker):
        """Failure count should reset when circuit closes."""
        # Open and recover
        breaker.record_failure("test-server", Exception("e1"))
        breaker.record_failure("test-server", Exception("e2"))
        time.sleep(0.15)

        # Trigger transition to HALF_OPEN by checking state
        assert breaker.get_state("test-server") == CircuitState.HALF_OPEN

        # Record successes to close the circuit
        breaker.record_success("test-server")
        breaker.record_success("test-server")

        # Should be closed with reset failure count
        assert breaker.is_closed("test-server")
        health = breaker.get_server_health("test-server")
        assert health["failure_count"] == 0


class TestCircuitBreakerReset:
    """Tests for manual circuit reset."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker."""
        return CircuitBreaker(failure_threshold=2)

    def test_reset_closes_open_circuit(self, breaker):
        """Manual reset should close an open circuit."""
        # Open the circuit
        breaker.record_failure("test-server", Exception("e1"))
        breaker.record_failure("test-server", Exception("e2"))
        assert breaker.is_open("test-server")

        # Reset
        breaker.reset("test-server")

        assert breaker.is_closed("test-server")

    def test_reset_all(self, breaker):
        """reset_all should close all circuits."""
        # Open multiple circuits
        breaker.record_failure("server1", Exception("e1"))
        breaker.record_failure("server1", Exception("e2"))
        breaker.record_failure("server2", Exception("e1"))
        breaker.record_failure("server2", Exception("e2"))

        assert breaker.is_open("server1")
        assert breaker.is_open("server2")

        # Reset all
        breaker.reset_all()

        assert breaker.is_closed("server1")
        assert breaker.is_closed("server2")


class TestCircuitBreakerHealthReport:
    """Tests for health reporting."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker."""
        return CircuitBreaker(failure_threshold=3)

    def test_health_report_empty_initially(self, breaker):
        """Health report should be empty with no activity."""
        assert breaker.get_health_report() == {}

    def test_health_report_after_activity(self, breaker):
        """Health report should reflect circuit activity."""
        breaker.record_success("server1")
        breaker.record_failure("server2", Exception("error"))

        report = breaker.get_health_report()

        assert "server1" in report
        assert report["server1"]["state"] == "closed"
        assert report["server1"]["success_count"] == 1

        assert "server2" in report
        assert report["server2"]["failure_count"] == 1

    def test_get_server_health(self, breaker):
        """Should get health for specific server."""
        breaker.record_success("my-server")
        breaker.record_success("my-server")

        health = breaker.get_server_health("my-server")

        assert health["state"] == "closed"
        assert health["success_count"] == 2
        assert health["failure_count"] == 0


class TestCircuitBreakerIgnoredExceptions:
    """Tests for exception filtering."""

    def test_ignored_exceptions_not_counted(self):
        """Ignored exception types should not count as failures."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            ignored_exceptions=(ValueError,),
        )

        # These should not count
        breaker.record_failure("test-server", ValueError("validation error"))
        breaker.record_failure("test-server", ValueError("another validation"))
        breaker.record_failure("test-server", ValueError("third validation"))

        # Circuit should still be closed
        assert breaker.is_closed("test-server")

    def test_non_ignored_exceptions_still_counted(self):
        """Non-ignored exceptions should still be counted."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            ignored_exceptions=(ValueError,),
        )

        # These should count
        breaker.record_failure("test-server", RuntimeError("real error"))
        breaker.record_failure("test-server", RuntimeError("another error"))

        # Circuit should be open
        assert breaker.is_open("test-server")


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_failures(self):
        """Should handle concurrent failures correctly."""
        breaker = CircuitBreaker(failure_threshold=10)
        errors = []

        def record_failures():
            try:
                for i in range(5):
                    breaker.record_failure("test-server", Exception(f"error-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        health = breaker.get_server_health("test-server")
        assert health["failure_count"] == 20  # 4 threads × 5 failures

    def test_concurrent_mixed_operations(self):
        """Should handle mixed concurrent operations."""
        breaker = CircuitBreaker(failure_threshold=100)  # High threshold
        errors = []

        def mixed_operations():
            try:
                for _ in range(10):
                    breaker.record_success("test-server")
                    breaker.record_failure("test-server", Exception("err"))
                    breaker.get_state("test-server")
                    breaker.allow_request("test-server")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_operations) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_circuit_open_error_message(self):
        """CircuitOpenError should have informative message."""
        error = CircuitOpenError("my-server")
        assert "my-server" in str(error)
        assert error.server_name == "my-server"

    def test_circuit_open_error_custom_message(self):
        """CircuitOpenError should support custom message."""
        error = CircuitOpenError("my-server", "Custom message here")
        assert "Custom message here" in str(error)


class TestGlobalCircuitBreaker:
    """Tests for global circuit breaker singleton."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_global_circuit_breaker()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_global_circuit_breaker()

    def test_get_circuit_breaker_returns_singleton(self):
        """Should return same instance on multiple calls."""
        breaker1 = get_circuit_breaker()
        breaker2 = get_circuit_breaker()

        assert breaker1 is breaker2

    def test_get_circuit_breaker_uses_first_config(self):
        """First call's config should be used."""
        breaker1 = get_circuit_breaker(failure_threshold=5)
        breaker2 = get_circuit_breaker(failure_threshold=100)  # Ignored

        assert breaker1.config.failure_threshold == 5
        assert breaker2.config.failure_threshold == 5

    def test_reset_global_circuit_breaker(self):
        """reset_global_circuit_breaker should clear singleton."""
        breaker1 = get_circuit_breaker()
        reset_global_circuit_breaker()
        breaker2 = get_circuit_breaker()

        assert breaker1 is not breaker2


class TestCircuitBreakerConfigValidation:
    """Tests for configuration validation."""

    def test_rejects_zero_failure_threshold(self):
        """failure_threshold must be >= 1."""
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreaker(failure_threshold=0)

    def test_rejects_negative_failure_threshold(self):
        """failure_threshold must be >= 1."""
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreaker(failure_threshold=-1)

    def test_rejects_zero_recovery_timeout(self):
        """recovery_timeout must be > 0."""
        with pytest.raises(ValueError, match="recovery_timeout must be > 0"):
            CircuitBreaker(recovery_timeout=0)

    def test_rejects_negative_recovery_timeout(self):
        """recovery_timeout must be > 0."""
        with pytest.raises(ValueError, match="recovery_timeout must be > 0"):
            CircuitBreaker(recovery_timeout=-5.0)

    def test_rejects_zero_success_threshold(self):
        """success_threshold must be >= 1."""
        with pytest.raises(ValueError, match="success_threshold must be >= 1"):
            CircuitBreaker(success_threshold=0)

    def test_rejects_zero_half_open_max_calls(self):
        """half_open_max_calls must be >= 1."""
        with pytest.raises(ValueError, match="half_open_max_calls must be >= 1"):
            CircuitBreaker(half_open_max_calls=0)

    def test_accepts_valid_config(self):
        """Valid configuration should not raise."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=1,
            half_open_max_calls=1,
        )
        assert breaker.config.failure_threshold == 1


class TestCircuitBreakerMultipleServers:
    """Tests for handling multiple servers independently."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker."""
        return CircuitBreaker(failure_threshold=2)

    def test_independent_circuits_per_server(self, breaker):
        """Each server should have independent circuit state."""
        # Fail server1
        breaker.record_failure("server1", Exception("e1"))
        breaker.record_failure("server1", Exception("e2"))

        # server1 open, server2 closed
        assert breaker.is_open("server1")
        assert breaker.is_closed("server2")

    def test_success_on_one_does_not_affect_other(self, breaker):
        """Success on one server should not affect others."""
        breaker.record_failure("server1", Exception("e1"))
        breaker.record_success("server2")

        health1 = breaker.get_server_health("server1")
        health2 = breaker.get_server_health("server2")

        assert health1["failure_count"] == 1
        assert health2["success_count"] == 1
        assert health1["success_count"] == 0
        assert health2["failure_count"] == 0
