# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper class that contains common functionality."""

from unittest import TestCase
import logging
import os
import warnings


class BaseTestCase(TestCase):
    """Helper class that contains common functionality."""

    @classmethod
    def setUpClass(cls):
        # Setup logging
        cls.log = logging.getLogger(cls.__name__)
        level = os.getenv("LOG_LEVEL", "DEBUG")
        cls._setup_test_logging(cls.log, level)

        # Suppress warnings caused by ibmq-provider still using deprecated paths in Terra to import
        # QPY.  ibmq-provider is itself deprecated, so we're avoiding pushing further changes to it.
        # TODO: remove all usage of ibmq-provider from this package before the complete removal of
        # that package and the release of Terra 0.25 in July 2023.
        warnings.filterwarnings(
            "ignore",
            r".*qiskit\.circuit\.qpy_serialization.*",
            DeprecationWarning,
            "qiskit.providers.ibmq",
        )

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

    @classmethod
    def simple_callback(cls, *args, **kwargs):
        """Simple callback function."""
        cls.log.debug("Callback args=%s, kwargs=%s", args, kwargs)
