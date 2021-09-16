from qiskit.providers.ibmq import RunnerResult
from qiskit.providers.jobstatus import JobStatus

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class TestCircuitRunnerQASM3(BaseTestCase):
    """Test circuit_runner_qasm3."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name

    def setUp(self) -> None:
        """Test case setup."""
        self.qc = '''OPENQASM 3;
        include "stdgates.inc";
        qubit[4] q;
        bit[4] c;
        reset q;
        x q[0];
        x q[2];
        h q[0];
        h q[1];
        h q[2];
        h q[3];'''

    def test_circuit_runner_qasm3(self):
        """Test circuit_runner program."""
        program_inputs = {
            'circuits': self.qc,
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(program_id="circuit-runner-qasm3",
                                        options=options,
                                        inputs=program_inputs,
                                        result_decoder=RunnerResult
                                        )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
