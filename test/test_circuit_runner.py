from qiskit import IBMQ, QuantumCircuit
from qiskit.providers.ibmq import RunnerResult
from qiskit.providers.jobstatus import JobStatus

import os
from unittest import TestCase, SkipTest

class TestCircuitRunner(TestCase):
    """Test circuit_runner."""

    def setUp(self) -> None:
        """Test case setup."""
        self.provider = IBMQ.load_account()
        N = 6
        qc = QuantumCircuit(N)
        qc.x(range(0, N))
        qc.h(range(0, N))
        self.qc = qc
        backend_name = os.getenv("QISKIT_IBM_DEVICE", None)
        if not backend_name:
            raise SkipTest("Runtime device not specified")
        self.backend_name = backend_name
        
    def test_circuit_runner(self):
        """Test circuit_runner program."""
        program_inputs = {
            'circuits': self.qc,
            'shots': 2048,
            'optimization_level': 0,
            'initial_layout': [0,1,4,7,10,12],
            'measurement_error_mitigation': False
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(program_id="circuit-runner",
                                    options=options,
                                    inputs=program_inputs,
                                    result_decoder=RunnerResult
                                    )
        self.assertTrue(job.status() in [JobStatus.QUEUED, JobStatus.DONE], job.error_message())
