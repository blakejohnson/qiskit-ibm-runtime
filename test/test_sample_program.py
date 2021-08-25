from qiskit import IBMQ

from unittest import TestCase
from decorator import get_provider_and_backend


class MethodCallLogger:
    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, *args):
        self.func(*args)
        self.call_count += 1


class TestSampleProgram(TestCase):
    """Test sample_program."""

    @classmethod
    @get_provider_and_backend 
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        cls.provider = provider
        cls.backend_name = backend_name

    def setUp(self) -> None:
        """Test case setup."""
        def interim_result_callback(job_id, interim_result):
            pass
        interim_result_callback = MethodCallLogger(interim_result_callback)
        self.interim_result_callback = interim_result_callback

    def test_sample_program(self):
        """Test sample program."""
        input = {
            "iterations": 2
        }
        options = {"backend_name": self.backend_name}
        job = self.provider.runtime.run(program_id="sample-program",
                                   options=options,
                                   inputs=input,
                                   callback=self.interim_result_callback
                                   )
        expected_result = "All done!"
        self.assertEqual(job.result(), expected_result)
        self.assertEqual(self.interim_result_callback.call_count, input["iterations"])
