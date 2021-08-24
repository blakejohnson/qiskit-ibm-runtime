from qiskit import IBMQ, QuantumCircuit
from qiskit.providers.ibmq import RunnerResult
from qiskit.providers.jobstatus import JobStatus

from unittest import TestCase
from decorator import get_provider_and_backend


class TestCircuitRunner(TestCase):
    """Test circuit_runner."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        cls.provider = provider
        cls.backend_name = backend_name

    def setUp(self) -> None:
        """Test case setup."""
        N = 6
        qc = QuantumCircuit(N)
        qc.x(range(0, N))
        qc.h(range(0, N))
        self.qc = qc

    async def test_circuit_runner(self):
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
        await self.assertEqual(job.status(),JobStatus.DONE, job.error_message())
