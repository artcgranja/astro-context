"""Custom exceptions for astro-context."""


class AstroContextError(Exception):
    """Base exception for all astro-context errors."""


class TokenBudgetExceededError(AstroContextError):
    """Raised when token budget is exceeded and no overflow strategy can handle it."""


class RetrieverError(AstroContextError):
    """Raised when a retriever encounters an error."""


class StorageError(AstroContextError):
    """Raised when a storage backend encounters an error."""


class FormatterError(AstroContextError):
    """Raised when formatting context fails."""
