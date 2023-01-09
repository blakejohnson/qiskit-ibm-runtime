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
from qiskit.compiler import assemble, transpile
from qiskit.test.reference_circuits import ReferenceCircuits
from qiskit.providers.ibmq import RunnerResult
from qiskit.providers.jobstatus import JobStatus
from qiskit.qobj import (
    PulseLibraryItem,
    PulseQobj,
    PulseQobjConfig,
    PulseQobjExperiment,
    PulseQobjInstruction,
    QobjHeader,
    QobjMeasurementOption,
)

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

    def test_circuit_runner_qasm_qobj(self):
        """Test circuit_runner program with QASMQobj Dict as input."""
        backend = self.provider.get_backend(self.backend_name)
        bell_in_qobj = assemble(
            transpile(ReferenceCircuits.bell(), backend=backend), backend=backend, shots=1024
        )
        program_inputs = {
            "circuits": bell_in_qobj.to_dict(),
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

    def test_circuit_runner_pulse_qobj(self):
        """Test circuit_runner program with PulseQobj Dict as input."""
        pulse_qobj = PulseQobj(
            qobj_id="12345",
            header=QobjHeader(),
            config=PulseQobjConfig(
                shots=1024,
                memory_slots=2,
                meas_level=1,
                memory_slot_size=8192,
                meas_return="avg",
                pulse_library=[
                    PulseLibraryItem(name="pulse0", samples=[0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j])
                ],
                qubit_lo_freq=[4.9],
                meas_lo_freq=[6.9],
                rep_time=1000,
            ),
            experiments=[
                PulseQobjExperiment(
                    header=QobjHeader(),
                    instructions=[
                        PulseQobjInstruction(name="pulse0", t0=0, ch="d0"),
                        PulseQobjInstruction(name="fc", t0=5, ch="d0", phase=1.57),
                        PulseQobjInstruction(name="fc", t0=5, ch="d0", phase=0.0),
                        PulseQobjInstruction(name="fc", t0=5, ch="d0", phase="P1"),
                        PulseQobjInstruction(name="setp", t0=10, ch="d0", phase=3.14),
                        PulseQobjInstruction(name="setf", t0=10, ch="d0", frequency=8.0),
                        PulseQobjInstruction(name="shiftf", t0=10, ch="d0", frequency=4.0),
                        PulseQobjInstruction(
                            name="acquire",
                            t0=15,
                            duration=5,
                            qubits=[0],
                            memory_slot=[0],
                            kernels=[
                                QobjMeasurementOption(
                                    name="boxcar", params={"start_window": 0, "stop_window": 5}
                                )
                            ],
                        ),
                    ],
                )
            ],
        )
        program_inputs = {
            "circuits": pulse_qobj.to_dict(),
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
