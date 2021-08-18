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

    @classmethod
    def setUpClass(cls):
        """Class setup."""
        hgp = os.getenv("QISKIT_IBM_HGP_STAGING", None) \
            if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True" \
            else os.getenv("QISKIT_IBM_HGP", None)
        if not hgp:
            raise SkipTest("Requires ibm provider.")
        hgp = hgp.split(",")
        backend_name = os.getenv("QISKIT_IBM_DEVICE_STAGING", None) \
            if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True" \
            else os.getenv("QISKIT_IBM_DEVICE", None)          
        if not backend_name:
            raise SkipTest("Runtime device not specified")
        cls.backend_name = backend_name
        if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True":
            print("Using staging creds")
            os.environ["QE_TOKEN"] = os.getenv("QE_TOKEN_STAGING", "")
            os.environ["QE_URL"] = os.getenv("QE_URL_STAGING", "")
        IBMQ.enable_account(os.getenv("QE_TOKEN", ""), os.getenv("QE_URL", ""))     
        cls.provider = IBMQ.get_provider(hub=hgp[0], group=hgp[1], project=hgp[2])

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
        input = {
            "iterations": 2
        }
        options = {"backend_name": self.__class__.backend_name}
        job = self.__class__.provider.runtime.run(program_id="sample-program",
                                   options=options,
                                   inputs=input,
                                   callback=self.interim_result_callback
                                   )
        expected_result = "All done!"
        self.assertEqual(job.result(), expected_result)
        self.assertEqual(self.interim_result_callback.call_count, input["iterations"])
