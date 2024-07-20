import logging


def get_logger(name: str, level: int = logging.INFO):
    """
    Get a logger object.

    Args:
        name: Name of the logger
        level: Logging level

    Returns:
        Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
