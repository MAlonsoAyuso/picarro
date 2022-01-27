# https://docs.python.org/3/library/logging.config.html#configuration-dictionary-schema
from typing import Any, Dict


LogSettingsDict = Dict[str, Any]

DEFAULT_LOG_SETTINGS = {
    "formatters": {
        "picarro_default": {
            "format": "%(asctime)s %(levelname)-8s %(name)-18s %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "picarro_default",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "filename": "log.txt",
            "formatter": "picarro_default",
            "maxBytes": 1e6,
            "backupCount": 2,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": [
            "console",
            "file",
        ],
    },
    "version": 1,
    "disable_existing_loggers": False,
}
