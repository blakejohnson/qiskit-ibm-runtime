from qiskit import IBMQ
from unittest import TestCase
class MethodCallLogger(object):
    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, *args):
        self.func(*args)
        self.call_count += 1
class TestSampleProgram(TestCase):
    """Test sample_program"""

    def setUp(self) -> None:
        """Test case setup."""
        def interim_result_callback(job_id, interim_result):
            print("Interim result:", job_id, interim_result)
        interim_result_callback= MethodCallLogger(interim_result_callback)
        self.interim_result_callback = interim_result_callback

    def test_run_program(self):
        """Test sample program"""
        provider = IBMQ.load_account()
        input = {
            "iterations": 2
        }
        options = {'backend_name': 'ibmq_qasm_simulator'}
        job = provider.runtime.run(program_id="sample-program",
                                options=options,
                                inputs=input,
                                callback=self.interim_result_callback
                                )
        expected_result = "All done!"
        self.assertEqual(job.result(), expected_result)
        self.assertTrue(self.interim_result_callback.call_count, input["iterations"])

