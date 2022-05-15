import logging

from colorlog import ColoredFormatter

from malib import settings


# from sample-factory: ....
Logger = logging.getLogger("MALib")
Logger.setLevel(settings.LOG_LEVEL)
Logger.handlers = []  # No duplicated handlers
Logger.propagate = False  # workaround for duplicated logs in ipython

stream_handler = logging.StreamHandler()
stream_handler.setLevel(settings.LOG_LEVEL)

stream_formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(levelname)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "white,bold",
        "INFOV": "cyan,bold",
        "WARNING": "yellow",
        "ERROR": "red,bold",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)
stream_handler.setFormatter(stream_formatter)
Logger.addHandler(stream_handler)
