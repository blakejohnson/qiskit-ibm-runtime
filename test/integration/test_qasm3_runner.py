# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test circuit_runner_qasm3."""

from unittest import skip, SkipTest

from test.unit.test_circuit_merger import _create_test_circuits

from qiskit import QuantumCircuit
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
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ,unused-argument
        """Class setup."""
        super().setUpClass()

        _backend = None
        for hgp in IBMQ.providers():
            backends = hgp.backends(name="simulator_qasm3", operational=True)

            if backends:
                _backend = backends[0]
                break

        if not _backend:
            raise SkipTest("No access to QASM3 backend")

        cls.provider = _backend.provider()
        cls.runtime = provider.runtime
        cls.backend_name = _backend.name()
        cls.program_id = "qasm3-runner"
        cls.n_qubits = _backend.configuration().n_qubits

    @skip("Skip until backend supports qasm3")
    def test_circuit_runner_qasm3_real(self):
        """Test circuit_runner_qasm3 program on a real device."""
        program_inputs = {"circuits": QASM3_STR}

        options = {"backend_name": self.backend_name}

        job = self.runtime.run(
            program_id="circuit-runner-qasm3",
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
        result = self._run_program(circuits=QASM3_STR_WITH_ARGS, qasm3_args={"flip": 1})
        self.assertTrue(list(result[0].values())[0], result)

    def test_sim_single_str_multi_args(self):
        """Test the program on a simulator with multiple args."""
        result = self._run_program(
            circuits=QASM3_STR_WITH_ARGS, qasm3_args=[{"flip": 1}, {"flip": 0}]
        )
        self.assertTrue(list(result[0][0].values())[0], result)
        self.assertFalse(list(result[1][0].values())[0], result)

    def test_sim_single_str_multi_args_shot(self):
        """Test the program on a simulator with multiple args and shots."""
        num_shots = 3
        result = self._run_program(
            circuits=QASM3_STR_WITH_ARGS,
            qasm3_args=[{"flip": 1}, {"flip": 0}],
            run_config={"shots": num_shots},
        )
        self.log.debug("test_sim_single_str_multi_args_shot result=%s", result)
        for idx, arg_result in enumerate(result):
            self.assertEqual(len(arg_result), num_shots)
            expected = idx == 0
            for shot_result in arg_result:
                self.assertEqual(expected, list(shot_result.values())[0])

    def test_sim_circuit_merging(self):
        """Test multiple circuits to be merged on a simulator."""
        qc1, qc2 = _create_test_circuits()
        job = self._run_program(circuits=[qc1, qc2], block_for_result=False)
        self._check_job_completes(job)

    def test_sim_circuit_merging_many(self):
        """Test that many circuits can be merged and run."""
        qc1, _ = _create_test_circuits()
        circuits = [qc1 for i in range(10)]
        job = self._run_program(circuits=circuits, block_for_result=False, run_config={"shots": 10})
        self._check_job_completes(job)

    def test_sim_circuit_no_merging(self):
        """Test parameter for submitting circuits without merging."""
        qc1, qc2 = _create_test_circuits()
        job = self._run_program(circuits=[qc1, qc2], block_for_result=False, merge_circuits=False)
        self._check_job_completes(job)

    def test_sim_circuit_merging_reset(self):
        """Test that the number of resets can be influenced when merging
        multiple circuits."""
        qc1, qc2 = _create_test_circuits()
        job = self._run_program(circuits=[qc1, qc2], init_num_resets=2, block_for_result=False)
        self._check_job_completes(job)

    @skip("Skip until deployed qasm3 exporter can handle delays")
    def test_sim_circuit_merging_delays(self):
        """Test that the delay between circuits can be influenced when merging
        multiple circuits."""
        qc1, qc2 = _create_test_circuits()
        job = self._run_program(circuits=[qc1, qc2], init_delay=100, block_for_result=False)
        self._check_job_completes(job)

    def test_sim_circuit_merging_custom_init(self):
        """Test passing a custom circuit for initializing qubits."""
        qc1, qc2 = _create_test_circuits()
        custom_init = QuantumCircuit(self.n_qubits)
        custom_init.sx(range(self.n_qubits))
        custom_init.barrier(range(self.n_qubits))
        job = self._run_program(
            circuits=[qc1, qc2], init_circuit=custom_init, block_for_result=False
        )
        self._check_job_completes(job)

    def _run_program(
        self,
        circuits=None,
        qasm3_args=None,
        run_config=None,
        block_for_result=True,
        merge_circuits=None,
        init_num_resets=None,
        init_delay=None,
        init_circuit=None,
    ):
        """Run the circuit-runner-qasm3 program.

        Args:
            circuits: Circuit(s) to run. Default is self.qasm3_str.
            qasm3_args: Args to pass to the QASM3 program.
            run_config: Execution time configuration.
            block_for_result: Whether to block for result.
            merge_circuits: Whether to merge multiple submitted circuits into
                            one before execution (default is yes).
            init_num_resets: The number of reset to insert before each circuit
                         execution (default is 3).
            init_delay: The number of microseconds of delay to insert
                                 before each circuit execution.
        Returns:
            Job result if `block_for_result` is ``True``. Otherwise the job.
        """
        circuits = circuits or QASM3_STR
        program_inputs = {"circuits": circuits}
        if qasm3_args:
            program_inputs["qasm3_args"] = qasm3_args
        if run_config:
            program_inputs["run_config"] = run_config
        if merge_circuits is not None:
            program_inputs["merge_circuits"] = merge_circuits
        if init_num_resets is not None:
            program_inputs["init_num_resets"] = init_num_resets
        if init_delay is not None:
            program_inputs["init_delay"] = init_delay
        if init_circuit is not None:
            program_inputs["init_circuit"] = init_circuit
        options = {"backend_name": self.backend_name}

        job = self.runtime.run(program_id=self.program_id, options=options, inputs=program_inputs)
        self.log.debug("Job ID: %s", job.job_id())
        if block_for_result:
            return job.result()
        return job

    def _check_job_completes(self, job):
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
        result = job.result()
        self.assertIsInstance(result, dict)