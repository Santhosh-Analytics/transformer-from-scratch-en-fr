import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────────────────────
# Settings  (can be overridden via .env file)
# ──────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB per file
    LOG_BACKUP_COUNT: int = 3
    LOG_DIR: str = "logs"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


settings = Settings()

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
log_dir = Path(settings.LOG_DIR)
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{datetime.now():%Y_%m_%d_%H_%M_%S}.log"

# ──────────────────────────────────────────────────────────────
# Formatter
# ──────────────────────────────────────────────────────────────
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
formatter = logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S")

# ──────────────────────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────────────────────
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=settings.LOG_MAX_BYTES,
    backupCount=settings.LOG_BACKUP_COUNT,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)  # always capture everything to file

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))

# ──────────────────────────────────────────────────────────────
# Logger  ← renamed from churn_ann to transformer_mt
# ──────────────────────────────────────────────────────────────
logger = logging.getLogger("transformer_mt")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False


# ──────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
