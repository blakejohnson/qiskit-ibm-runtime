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

"""Unit tests for Circuit Merger (in qasm3-runner)."""

from typing import Tuple

from qiskit import BasicAer, QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.test import QiskitTestCase

from programs.qasm3_runner import CircuitMerger


def _create_test_circuits() -> Tuple[QuantumCircuit, QuantumCircuit]:
    cl0 = ClassicalRegister(2)
    cl1 = ClassicalRegister(2)
    qr0 = QuantumRegister(2)
    qr1 = QuantumRegister(2)
    qc1 = QuantumCircuit(qr0, qr1, cl0, cl1)
    qc1.x(qr0)
    qc1.x(qr1[0])
    qc1.measure(qr0, cl0)
    qc1.measure(qr1, cl1)

    cl2 = ClassicalRegister(1)
    cl3 = ClassicalRegister(2)
    qr2 = QuantumRegister(1)
    qr3 = QuantumRegister(2)
    qc2 = QuantumCircuit(qr2, qr3, cl2, cl3)
    qc2.x(qr2[0])
    qc2.x(qr3[1])
    qc2.measure(qr2, cl2)
    qc2.measure(qr3, cl3)
    return qc1, qc2


class TestCircuitMerger(QiskitTestCase):
    """Test the Circuit Merger"""

    def setUp(self):
        super().setUp()
        self.sim_backend = BasicAer.get_backend("qasm_simulator")
        self.sim_backend.configuration().n_qubits = 4

    def test_basic_merge_two_circuits(self):
        """Test that circuits can be merged and that results are equal to those
        from singular execution."""
        qc1, qc2 = _create_test_circuits()
        merger = CircuitMerger([qc1, qc2], backend=self.sim_backend)
        merged_circuit = merger.merge_circuits()
        self.assertIsInstance(merged_circuit, QuantumCircuit)

        # check that merged circuit produces the same results
        result_qc1 = execute(qc1, backend=self.sim_backend, num_shots=100).result()
        result_qc2 = execute(qc2, backend=self.sim_backend, num_shots=100).result()
        result_merged = execute(merged_circuit, backend=self.sim_backend, num_shots=100).result()
        unwrapped_result = merger.unwrap_results(result_merged)

        self.assertEqual(result_qc1.get_counts(0), unwrapped_result.get_counts(0))
        self.assertEqual(result_qc2.get_counts(0), unwrapped_result.get_counts(1))

    def test_merge_multiple_resets(self):
        """Test that merging with parameterized initialization circuit inserts
        the correct number of delays."""
        qc1, qc2 = _create_test_circuits()
        num_used_qubits = max(len(qc1.qubits), len(qc2.qubits))
        num_qubit_reset = 10
        merger = CircuitMerger([qc1, qc2], backend=self.sim_backend)
        merged_circuit = merger.merge_circuits(init_num_resets=num_qubit_reset)
        self.assertEqual(merged_circuit.count_ops()["reset"], 2 * num_qubit_reset * num_used_qubits)

    def test_merge_many_circuits(self):
        """Test that merging a large number of circuits works as expected and
        that results are equal to those from singular execution."""
        num_circuits = 10
        num_shots = 100
        qc1, _ = _create_test_circuits()
        circuits = [qc1 for i in range(num_circuits)]
        merger = CircuitMerger(circuits, backend=self.sim_backend)
        merged_circuit = merger.merge_circuits()
        self.assertIsInstance(merged_circuit, QuantumCircuit)

        # check that merged circuit produces the same results
        result_qc1 = execute(qc1, backend=self.sim_backend, num_shots=num_shots).result()
        result_merged = execute(
            merged_circuit, backend=self.sim_backend, num_shots=num_shots
        ).result()
        unwrapped_result = merger.unwrap_results(result_merged)
        for i in range(num_circuits):
            self.assertEqual(result_qc1.get_counts(0), unwrapped_result.get_counts(i))

    def test_custom_init_circuit(self):
        """Test that merging with a custom circuit for qubit initialization
        works as expected."""
        qc1, qc2 = _create_test_circuits()
        n_qubits = self.sim_backend.configuration().n_qubits
        custom_init = QuantumCircuit(n_qubits)
        custom_init.y(range(n_qubits))
        custom_init.y(range(n_qubits))
        merger = CircuitMerger([qc1, qc2], backend=self.sim_backend)
        merged_circuit = merger.merge_circuits(init_circuit=custom_init)
        self.assertEqual(merged_circuit.count_ops()["y"], 2 * 2 * n_qubits)
