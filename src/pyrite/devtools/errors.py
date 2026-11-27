"""Shared exceptions for repository developer commands."""


class AgentToolingError(RuntimeError):
    """Raised when repository tooling detects invalid or inconsistent state."""
