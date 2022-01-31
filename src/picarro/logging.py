# https://docs.python.org/3/library/logging.config.html#configuration-dictionary-schema
from typing import Any, Dict


LogSettingsDict = Dict[str, Any]

DEFAULT_LOG_SETTINGS = {
    "formatters": {
        "detailed": {
            "format": "%(asctime)s %(levelname)-8s %(name)-22s %(message)s",
        },
        "brief": {
            "format": "%(levelname)-8s %(message)s",
        },
    },
    "filters": {
        "allow_picarro": {"name": "picarro"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "brief",
            "filters": ["allow_picarro"],
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "filename": "log.txt",
            "formatter": "detailed",
            "maxBytes": 1e6,
            "backupCount": 5,
            "filters": ["allow_picarro"],
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
