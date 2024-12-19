import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def log_function(function):
    def decorator(*args, **kwargs):
        logger.INFO(f"Started {function.__name__}")
        output = function(*args, **kwargs)
        logger.INFO(f"Completed {function.__name__}")
        return output

    return decorator
