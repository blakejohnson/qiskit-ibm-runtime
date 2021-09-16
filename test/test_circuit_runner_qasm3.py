from unittest import skip

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
        self.qasm3_str ='''OPENQASM 3;
        include "stdgates.inc";

        def segment qubit[2] anc, qubit psi -> bit[2] {
        bit[2] b;
        reset anc;
        h anc;
        ccx anc[0], anc[1], psi;
        s psi;
        ccx anc[0], anc[1], psi;
        z psi;
        h anc;
        measure anc -> b;
        return b;
        }

        qubit input;
        qubit[2] ancilla;
        bit[2] flags = "11";
        bit output;

        reset input;
        h input;

        while(int(flags) != 0) {
        flags = segment ancilla, input;
        }
        rz(pi - arccos(3 / 5)) input;
        h input;
        output = measure input;'''

    def test_circuit_runner_qasm3(self):
        """Test circuit_runner_qasm3 program."""
        program_inputs = {
            'circuits': self.qasm3_str,
            'use_qasm3': True,
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(program_id="circuit-runner-qasm3",
                                        options=options,
                                        inputs=program_inputs,
                                        result_decoder=RunnerResult,
                                        )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
