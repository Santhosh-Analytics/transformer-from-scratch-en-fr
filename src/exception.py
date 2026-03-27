import sys
from src.logger import logger


# ──────────────────────────────────────────────────────────────
# Base Exception
# ──────────────────────────────────────────────────────────────
class TransformerMTError(Exception):
    """
    Custom base exception for the Transformer MT project.
    All project-specific errors inherit from this.
    Automatically logs the formatted error on creation.
    """

    def __init__(self, message: str, sys_info=sys):
        self.error_message = self._format_error(message, sys_info)
        logger.error(self.error_message)  # auto-log on raise
        super().__init__(self.error_message)

    @staticmethod
    def _format_error(message: str, sys_info) -> str:
        _, _, tb = sys_info.exc_info()
        if tb:
            file_name = tb.tb_frame.f_code.co_filename
            line_no = tb.tb_lineno
            return f"[{file_name}  line {line_no}]  {message}"
        # raised without an active exception context (e.g. manual raise)
        return f"[no traceback]  {message}"

    def __str__(self) -> str:
        return self.error_message


# ──────────────────────────────────────────────────────────────
# Specific Exceptions
# ──────────────────────────────────────────────────────────────
class DataIngestionError(TransformerMTError):
    """Raised when downloading or reading raw data fails."""

    pass


class DataValidationError(TransformerMTError):
    """Raised when data doesn't match expected schema or format."""

    pass


class PreprocessingError(TransformerMTError):
    """Raised when tokenisation, vocab building, or DataLoader creation fails."""

    pass


class ModelBuildError(TransformerMTError):
    """Raised when the Transformer architecture cannot be instantiated."""

    pass


class ModelTrainingError(TransformerMTError):
    """Raised when the training loop fails mid-way."""

    pass


class ModelEvaluationError(TransformerMTError):
    """Raised when BLEU scoring or decoding fails."""

    pass


class CheckpointError(TransformerMTError):
    """Raised when saving or loading a model checkpoint fails."""

    pass


class ConfigError(TransformerMTError):
    """Raised when config values are missing or invalid."""

    pass
