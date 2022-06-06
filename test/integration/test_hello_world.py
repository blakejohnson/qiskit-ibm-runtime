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

"""Test hello world."""

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class MethodCallLogger:
    """Logger."""

    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, *args):
        self.func(*args)
        self.call_count += 1


class TestHelloWorld(BaseTestCase):
    """Test hello world."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name

    def setUp(self) -> None:
        """Test case setup."""
        interim_result_callback = MethodCallLogger(self.simple_callback)
        self.interim_result_callback = interim_result_callback

    def test_hello_world(self):
        """Test hello_world."""
        runtime_inputs = {"iterations": 2}
        options = {"backend_name": self.backend_name}
        job = self.provider.runtime.run(
            program_id="hello-world",
            options=options,
            inputs=runtime_inputs,
            callback=self.interim_result_callback,
        )
        self.log.debug("Job ID: %s", job.job_id())
        expected_result = "Hello, World!"
        self.assertEqual(job.result(), expected_result)
        self.assertEqual(self.interim_result_callback.call_count, runtime_inputs["iterations"] + 1)
