from unittest import TestCase
import logging
import os


class BaseTestCase(TestCase):
    """Helper class that contains common functionality."""

    @classmethod
    def setUpClass(cls):
        # Setup logging
        cls.log = logging.getLogger(cls.__name__)
        level = os.getenv("LOG_LEVEL", "DEBUG")
        cls._setup_test_logging(cls.log, level)

    @classmethod
    def _setup_test_logging(cls, logger, log_level):
        """Set logging to file and stdout for a logger.

        Args:
            logger (Logger): logger object to be updated.
            log_level (str): logging level.
        """
        # Set up formatter.
        log_fmt = "{}.%(funcName)s:%(levelname)s:%(asctime)s:" " %(message)s".format(logger.name)
        formatter = logging.Formatter(log_fmt)

        # Set up the stream handler.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.setLevel(log_level)
