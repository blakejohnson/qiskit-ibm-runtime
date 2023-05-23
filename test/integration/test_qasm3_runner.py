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

from unittest import SkipTest, expectedFailure

from test.unit.test_circuit_merger import _create_test_circuits

from qiskit import QuantumCircuit
from qiskit.providers.ibmq import RunnerResult  # pylint: disable=import-error
from qiskit.providers.jobstatus import JobStatus


from .decorator import get_provider_and_backend, IBMQ
from .base_testcase import BaseTestCase


class TestQASM3Runner(BaseTestCase):
    """Test circuit_runner_qasm3."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ,unused-argument
        """Class setup."""
        super().setUpClass()

        cls.provider = provider
        cls.backend_name = backend_name
        cls.runtime = None
        cls.program_id = "qasm3-runner"
        cls.n_qubits = None

    def _check_backend(self, backend_name):
        _backend = None
        for hgp in IBMQ.providers():
            backends = hgp.backends(name=backend_name, operational=True)

            if backends:
                _backend = backends[0]
                break
        if not _backend:
            raise SkipTest(f"No access to backend: {backend_name}")

        self.provider = _backend.provider if _backend.version == 2 else _backend.provider()
        self.runtime = self.provider.runtime
        self.backend_name = _backend.name if _backend.version == 2 else _backend.name()
        self.n_qubits = _backend.configuration().n_qubits
        return _backend

    def test_circuit_merging(self):
        """Test multiple circuits to be merged on a simulator."""
        qc1, qc2 = _create_test_circuits()
        job = self._run_program(circuits=[qc1, qc2], block_for_result=False)
        self._check_job_completes(job)

    def test_circuit_merging_many(self):
        """Test that many circuits can be merged and run."""
        qc1, _ = _create_test_circuits()
        circuits = [qc1 for i in range(10)]
        job = self._run_program(
            circuits=circuits,
            block_for_result=False,
            shots=10,
        )
        self._check_job_completes(job)

    def test_circuit_no_merging(self):
        """Test parameter for submitting circuits without merging."""
        qc1, qc2 = _create_test_circuits()
        job = self._run_program(circuits=[qc1, qc2], block_for_result=False, merge_circuits=False)
        self._check_job_completes(job)

    def test_circuit_merging_reset(self):
        """Test that the number of resets can be influenced when merging
        multiple circuits."""
        qc1, qc2 = _create_test_circuits()
        job = self._run_program(circuits=[qc1, qc2], init_num_resets=2, block_for_result=False)
        self._check_job_completes(job)

    def test_circuit_merging_init_delay(self):
        """Test that the delay between circuits can be influenced when merging
        multiple circuits with init_delay."""
        if self._check_backend(self.backend_name).configuration().simulator:
            self.skipTest(
                "Currently we cannot determine qubit initialization delay for a simulator."
            )

        qc1, qc2 = _create_test_circuits()
        job = self._run_program(
            circuits=[qc1, qc2],
            init_delay=100,
            block_for_result=False,
        )
        self._check_job_completes(job)

    def test_circuit_merging_rep_delay(self):
        """Test that the delay between circuits can be influenced when merging
        multiple circuits with rep_delay."""
        if self._check_backend(self.backend_name).configuration().simulator:
            self.skipTest(
                "Currently we cannot determine qubit initialization delay for a simulator."
            )

        qc1, qc2 = _create_test_circuits()
        job = self._run_program(
            circuits=[qc1, qc2],
            init_delay=100e-6,
            block_for_result=False,
        )
        self._check_job_completes(job)

    def test_init_delay_rep_delay_fails(self):
        """Test setting init_delay and rep_delay fails."""
        if self._check_backend(self.backend_name).configuration().simulator:
            self.skipTest(
                "Currently we cannot determine qubit initialization delay for a simulator."
            )

        qc1, qc2 = _create_test_circuits()
        job = self._run_program(
            circuits=[qc1, qc2],
            init_delay=100,
            rep_delay=200 - 6,
            block_for_result=False,
        )
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.ERROR)

    def test_circuit_merging_custom_init(self):
        """Test passing a custom circuit for initializing qubits."""
        qc1, qc2 = _create_test_circuits()
        custom_init = QuantumCircuit(2)
        custom_init.sx(range(2))
        custom_init.barrier(range(2))
        job = self._run_program(
            circuits=[qc1, qc2],
            init_circuit=custom_init,
            block_for_result=False,
        )
        self._check_job_completes(job)

    def test_no_init_qubits(self):
        """Test that qubit initialization may be disabled."""
        qc1, qc2 = _create_test_circuits()
        job = self._run_program(circuits=[qc1, qc2], block_for_result=False, init_qubits=False)
        self._check_job_completes(job)

    def test_meas_level(self):
        """Test setting the measurement level for an input circuit."""
        if self._check_backend(self.backend_name).configuration().simulator:
            self.skipTest(
                "Currently we cannot determine qubit initialization delay for a simulator."
            )
        qc1, qc2 = _create_test_circuits()

        result = self._run_program(
            circuits=[qc1, qc2],
            block_for_result=True,
            meas_level=1,
            decode=True,
        )
        self.assertIsNotNone(result.get_memory(0))

    def test_memory_true(self):
        """Test setting the memory true for an input circuit."""
        if self._check_backend(self.backend_name).configuration().simulator:
            self.skipTest(
                "Currently we cannot determine qubit initialization delay for a simulator."
            )
        qc1, qc2 = _create_test_circuits()

        result = self._run_program(
            circuits=[qc1, qc2],
            block_for_result=True,
            memory=True,
            decode=True,
        )
        self.assertIsNotNone(result.get_memory(0))

    @expectedFailure
    def test_memory_false(self):
        """Test setting the memory false for an input circuit.

        Currently expected failure as we do not support turning off memory at this point.
        """
        if self._check_backend(self.backend_name).configuration().simulator:
            self.skipTest(
                "Currently we cannot determine qubit initialization delay for a simulator."
            )
        qc1, qc2 = _create_test_circuits()

        result = self._run_program(
            circuits=[qc1, qc2],
            block_for_result=True,
            memory=False,
            decode=True,
        )
        self.assertIsNone(result.get_memory(0))

    def test_shots(self):
        """Test setting the shots for an input circuit."""
        shots = 1337
        qc1, _ = _create_test_circuits()

        result = self._run_program(
            circuits=[qc1],
            block_for_result=True,
            shots=shots,
            decode=True,
        )
        self.assertEqual(sum(result.get_counts(0).values()), shots)

    def _run_program(
        self,
        circuits,
        block_for_result=True,
        merge_circuits=None,
        init_qubits=True,
        init_num_resets=None,
        rep_delay=None,
        init_delay=None,
        init_circuit=None,
        meas_level=None,
        memory=None,
        shots=None,
        decode=False,
    ):
        """Run the circuit-runner-qasm3 program.

        Args:
            backend: Backend to run on.
            circuits: Circuit(s) to run.
            block_for_result: Whether to block for result.
            merge_circuits: Whether to merge multiple submitted circuits into
                            one before execution (default is yes).
            init_num_resets: Enable qubit initialization.
            init_num_resets: The number of reset to insert before each circuit
                         execution (default is 3).
            rep_delay: Specify a number of seconds between circuit executions
                to allow the qubits to idle for.
            init_delay: The number of microseconds of delay to insert
                                 before each circuit execution.
            meas_level: Set the measurement level for the program.
            decode: The result to the Qiskit result format.
        Returns:
            Job result if `block_for_result` is ``True``. Otherwise the job.
        """

        backend_name = self.backend_name

        self._check_backend(backend_name)
        program_inputs = {"circuits": circuits}
        if merge_circuits is not None:
            program_inputs["merge_circuits"] = merge_circuits
        if init_qubits is not None:
            program_inputs["init_qubits"] = init_qubits
        if init_num_resets is not None:
            program_inputs["init_num_resets"] = init_num_resets
        if rep_delay is not None:
            program_inputs["rep_delay"] = rep_delay
        if init_delay is not None:
            program_inputs["init_delay"] = init_delay
        if init_circuit is not None:
            program_inputs["init_circuit"] = init_circuit
        if meas_level is not None:
            program_inputs["meas_level"] = meas_level
        if memory is not None:
            program_inputs["memory"] = memory
        if shots is not None:
            program_inputs["shots"] = shots

        options = {"backend_name": self.backend_name}

        if decode:
            decoder = RunnerResult
        else:
            decoder = None

        job = self.runtime.run(
            program_id=self.program_id,
            options=options,
            inputs=program_inputs,
            result_decoder=decoder,
        )
        self.log.debug("Job ID: %s", job.job_id())
        if block_for_result:
            return job.result()
        return job

    def _check_job_completes(self, job):
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
        result = job.result()
        self.assertIsInstance(result, dict)
