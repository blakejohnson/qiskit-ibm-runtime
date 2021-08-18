from qiskit import IBMQ, QuantumCircuit
from qiskit.providers.ibmq import RunnerResult
from qiskit.providers.jobstatus import JobStatus

import os
from unittest import TestCase, SkipTest


class TestCircuitRunner(TestCase):
    """Test circuit_runner."""

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
        N = 6
        qc = QuantumCircuit(N)
        qc.x(range(0, N))
        qc.h(range(0, N))
        self.qc = qc

    def test_circuit_runner(self):
        """Test circuit_runner program."""
        program_inputs = {
            'circuits': self.qc,
            'shots': 2048,
            'optimization_level': 0,
            'initial_layout': [0, 1, 4, 7, 10, 12],
            'measurement_error_mitigation': False
        }

        options = {"backend_name": self.__class__.backend_name}

        job = self.__class__.provider.runtime.run(program_id="circuit-runner",
                                        options=options,
                                        inputs=program_inputs,
                                        result_decoder=RunnerResult
                                        )
        self.assertEqual(job.status(),JobStatus.DONE, job.error_message())
