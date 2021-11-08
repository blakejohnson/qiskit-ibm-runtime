import logging


def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel("DEBUG")
    formatter = logging.Formatter(
        '%(module)s.%(funcName)s:%(levelname)s:%(asctime)s: %(message)s')
    logger.propagate = False
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


setup_logging()
