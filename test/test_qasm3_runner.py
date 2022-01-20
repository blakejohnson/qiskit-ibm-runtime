from unittest import skip, SkipTest

from qiskit.providers.ibmq import RunnerResult
from qiskit.providers.jobstatus import JobStatus

from .decorator import get_provider_and_backend, IBMQ
from .base_testcase import BaseTestCase


QASM3_STR = """
OPENQASM 3;
include "stdgates.inc";
bit[2] qc;
qubit[2] qr;

h qr[0];
cx qr[0], qr[1];
qc[0] = measure qr[0];
qc[1] = measure qr[1];   
"""

QASM3_STR_WITH_ARGS = """
OPENQASM 3;
include "stdgates.inc";
input int[32] flip;
output bit result;
qubit q;

reset q;
if (flip == 1) x q;
result = measure q;
"""


class TestQASM3Runner(BaseTestCase):
    """Test circuit_runner_qasm3."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()

        _backend = None
        for provider in IBMQ.providers():
            backends = provider.backends(
                name="simulator_qasm3",
                operational=True)

            if backends:
                _backend = backends[0]
                break

        if not _backend:
            raise SkipTest("No access to QASM3 backend")

        cls.provider = _backend.provider()
        cls.runtime = provider.runtime
        cls.backend_name = _backend.name()
        cls.program_id = cls._find_program_id("qasm3-runner")

    @skip("Skip until backend supports qasm3")
    def test_circuit_runner_qasm3_real(self):
        """Test circuit_runner_qasm3 program on a real device."""
        program_inputs = {
            'circuits': QASM3_STR,
            'use_qasm3': True,
        }

        options = {"backend_name": self.backend_name}

        job = self.runtime.run(program_id="circuit-runner-qasm3",
                               options=options,
                               inputs=program_inputs,
                               result_decoder=RunnerResult,
                               )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())

    def test_sim_single_str(self):
        """Test the program on a simulator using a single QASM3 string."""
        result = self._run_program()[0]
        self.assertIsInstance(result, dict)
        self.assertEqual(len(list(result.values())[0]), 2)  # 2 classical bits

    def test_sim_single_str_shots(self):
        """Test the program on a simulator with multiple shots."""
        num_shots = 3
        result = self._run_program(run_config={"shots": num_shots})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), num_shots)
        for shot_res in result:
            self.assertEqual(len(list(shot_res.values())[0]), 2)  # 2 classical bits

    def test_sim_single_str_args(self):
        """Test the program on a simulator with args."""
        result = self._run_program(
            circuits=QASM3_STR_WITH_ARGS, qasm3_args={"flip": 1})
        self.assertTrue(list(result[0].values())[0], result)

    def test_sim_single_str_multi_args(self):
        """Test the program on a simulator with multiple args."""
        result = self._run_program(
            circuits=QASM3_STR_WITH_ARGS, qasm3_args=[{"flip": 1}, {"flip": 0}])
        self.assertTrue(list(result[0][0].values())[0], result)
        self.assertFalse(list(result[1][0].values())[0], result)

    def test_sim_single_str_multi_args_shot(self):
        num_shots = 3
        result = self._run_program(
            circuits=QASM3_STR_WITH_ARGS,
            qasm3_args=[{"flip": 1}, {"flip": 0}],
            run_config={"shots": num_shots})
        self.log.debug("test_sim_single_str_multi_args_shot result=%s", result)
        for idx, arg_result in enumerate(result):
            self.assertEqual(len(arg_result), num_shots)
            expected = idx == 0
            for shot_result in arg_result:
                self.assertEqual(expected, list(shot_result.values())[0])

    def _run_program(
            self,
            circuits=None,
            qasm3_args=None,
            run_config=None,
            block_for_result=True
    ):
        """Run the circuit-runner-qasm3 program.

        Args:
            circuits: Circuit(s) to run. Default is self.qasm3_str.
            qasm3_args: Args to pass to the QASM3 program.
            run_config: Execution time configuration.
            block_for_result: Whether to block for result.

        Returns:
            Job result if `block_for_result` is ``True``. Otherwise the job.
        """
        circuits = circuits or QASM3_STR
        program_inputs = {
            "circuits": circuits
        }
        if qasm3_args:
            program_inputs["qasm3_args"] = qasm3_args
        if run_config:
            program_inputs["run_config"] = run_config
        options = {"backend_name": self.backend_name}

        job = self.runtime.run(program_id=self.program_id,
                               options=options,
                               inputs=program_inputs
                               )
        self.log.debug("Job ID: %s", job.job_id())
        if block_for_result:
            return job.result()
        return job

    @classmethod
    def _find_program_id(cls, program_name):
        """Find ID of the program."""
        for pgm in cls.runtime.programs(refresh=True, limit=None):
            if pgm.name == program_name:
                return pgm.program_id
        raise ValueError("Unable to find ID for program %s", program_name)
