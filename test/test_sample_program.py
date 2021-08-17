from qiskit import IBMQ

import os
from unittest import TestCase, SkipTest


class MethodCallLogger:
    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, *args):
        self.func(*args)
        self.call_count += 1


class TestSampleProgram(TestCase):
    """Test sample_program."""
    def setUp(self) -> None:
        """Test case setup."""
        def interim_result_callback(job_id, interim_result):
            pass
        interim_result_callback = MethodCallLogger(interim_result_callback)
        self.interim_result_callback = interim_result_callback
        backend_name = os.getenv("QISKIT_IBM_DEVICE", None)
        if not backend_name:
            raise SkipTest("Runtime device not specified")
        self.backend_name = backend_name

    def test_sample_program(self):
        """Test sample program."""
        provider = IBMQ.load_account()
        input = {
            "iterations": 2
        }
        options = {"backend_name": self.backend_name}
        job = provider.runtime.run(program_id="sample-program",
                                   options=options,
                                   inputs=input,
                                   callback=self.interim_result_callback
                                   )
        expected_result = "All done!"
        self.assertEqual(job.result(), expected_result)
        self.assertEqual(self.interim_result_callback.call_count, input["iterations"])
