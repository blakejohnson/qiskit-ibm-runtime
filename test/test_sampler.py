from qiskit import QuantumCircuit
from qiskit.providers.jobstatus import JobStatus

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class TestSampler(BaseTestCase):
    """Test sampler."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name

    def setUp(self) -> None:
        """Test case setup."""
        qc = QuantumCircuit(5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.measure_all()
        self.qc = qc

    def test_sampler(self):
        """Test sampler program."""
        program_inputs = {
            'circuits': self.qc,
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(program_id="sampler",
                                        options=options,
                                        inputs=program_inputs,
                                        )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
