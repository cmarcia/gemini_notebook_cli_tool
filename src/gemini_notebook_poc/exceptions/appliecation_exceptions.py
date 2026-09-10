
class NotebookOrchestratorError(Exception):
    """Base exception for all errors raised by NotebookOrchestrator."""
class NotebookAuthError(NotebookOrchestratorError):
    """Raised when authentication with NotebookLM or Gemini fails."""


class NotebookNotFoundError(NotebookOrchestratorError):
    """Raised when a requested notebook cannot be located."""


class NotebookQueryError(NotebookOrchestratorError):
    """Raised when an unrecoverable error occurs while querying a notebook."""


class SynthesisError(NotebookOrchestratorError):
    """Raised when cross-notebook LLM synthesis fails fatally."""
