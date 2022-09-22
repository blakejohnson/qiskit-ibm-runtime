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

"""Test circuit runner."""

from qiskit import QuantumCircuit
from qiskit.providers.ibmq import RunnerResult
from qiskit.providers.jobstatus import JobStatus

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class TestCircuitRunner(BaseTestCase):
    """Test circuit runner."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name

    def setUp(self) -> None:
        """Test case setup."""
        num_qubits = 6
        qc1 = QuantumCircuit(num_qubits)
        qc1.x(range(0, num_qubits))
        qc1.h(range(0, num_qubits))
        self.qc1 = qc1

    def test_circuit_runner(self):
        """Test circuit_runner program."""
        program_inputs = {
            "circuits": self.qc1,
            "shots": 2048,
            "optimization_level": 0,
            "initial_layout": [0, 1, 4, 7, 10, 12],
            "measurement_error_mitigation": False,
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(
            program_id="circuit-runner",
            options=options,
            inputs=program_inputs,
            result_decoder=RunnerResult,
        )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())

    def test_circuit_runner_qasm_2(self):
        """Test circuit_runner program with QASM 2.0 input."""
        qc_str = """OPENQASM 2.0;
            include "qelib1.inc";

            qreg q[2];
            creg c[2];

            h q[0];
            cx q[0], q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];"""
        program_inputs = {
            "circuits": qc_str,
            "shots": 2048,
            "optimization_level": 0,
            "measurement_error_mitigation": False,
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(
            program_id="circuit-runner",
            options=options,
            inputs=program_inputs,
            result_decoder=RunnerResult,
        )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
