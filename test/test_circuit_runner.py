from qiskit import QuantumCircuit
from qiskit.providers.ibmq import RunnerResult
from qiskit.providers.jobstatus import JobStatus

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class TestCircuitRunner(BaseTestCase):
    """Test circuit_runner."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name

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

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(program_id="circuit-runner",
                                        options=options,
                                        inputs=program_inputs,
                                        result_decoder=RunnerResult
                                        )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
