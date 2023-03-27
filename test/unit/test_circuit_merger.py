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

import unittest

from typing import Tuple

import numpy as np
from qiskit import BasicAer, QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile
from qiskit.circuit import Delay, Reset, Qubit, Clbit
from qiskit.providers.fake_provider import FakeBogota
from qiskit.result.models import ExperimentResultData

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


class TestCircuitMerger(unittest.TestCase):
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

    def test_init_delay(self):
        """Test circuit with init delay."""
        qc1, qc2 = _create_test_circuits()
        backend = FakeBogota()

        merger = CircuitMerger([qc1, qc2], backend=backend)
        merged_circuit = merger.merge_circuits(init_delay=1e-6, init_delay_unit="s")

        delays_found = 0
        for inst in merged_circuit.data:
            operation = inst.operation
            if isinstance(operation, Delay):
                delays_found += 1
                # 4500 cycle delay split between three resets
                # and aligned to mod 16 == 0 boundary.
                self.assertEqual(operation.duration, 1504)
                self.assertEqual(operation.unit, "dt")
        self.assertEqual(delays_found, 30)

    def test_reset_only_on_used_qubits(self):
        """Test that we only insert resets on qubits we use in the circuit."""

        backend = FakeBogota()

        qc0 = QuantumCircuit(1, 1)
        qc0.x(0)
        qc0.measure(0, 0)

        qc1 = transpile(qc0, backend, initial_layout=[0])
        qc2 = transpile(qc0, backend, initial_layout=[1])

        merger = CircuitMerger([qc1, qc2], backend=backend)
        merged_circuit = merger.merge_circuits(init_num_resets=1)

        reset_found = False
        for inst in merged_circuit.data:
            operation = inst.operation
            if isinstance(operation, Reset):
                reset_found = True
                self.assertIn(merged_circuit.find_bit(inst.qubits[0]).index, (0, 1))
        self.assertTrue(reset_found)

    def test_reset_only_on_used_qubits_non_nop(self):
        """Test that we only insert resets on qubits that have non-nop operations applied."""

        backend = FakeBogota()

        qc0 = QuantumCircuit(2, 2)
        qc0.x(0)
        qc0.barrier([0, 1])
        qc0.measure(0, 0)

        # Use ASAP scheduling to force delay padding of the program
        qc1 = transpile(qc0, backend, initial_layout=[0, 1], scheduling_method="asap")

        merger = CircuitMerger([qc1], backend=backend)
        merged_circuit = merger.merge_circuits(init_num_resets=1)

        reset_found = False
        for inst in merged_circuit.data:
            operation = inst.operation
            if isinstance(operation, Reset):
                reset_found = True
                self.assertIn(merged_circuit.find_bit(inst.qubits[0]).index, (0,))
        self.assertTrue(reset_found)

    def test_metadata(self):
        """Test adding of metadata to the result"""
        num_circuits = 2
        num_shots = 100
        qc1, _ = _create_test_circuits()
        circuits = [qc1.copy() for i in range(num_circuits)]
        circuits[0].metadata = {"foo": 1}
        circuits[1].metadata = {"bar": 1}
        merger = CircuitMerger(circuits, backend=self.sim_backend)
        merged_circuit = merger.merge_circuits()

        result_merged = execute(
            merged_circuit, backend=self.sim_backend, num_shots=num_shots
        ).result()
        unwrapped_result = merger.unwrap_results(result_merged)

        self.assertEqual(unwrapped_result.results[0].header.metadata["foo"], 1)
        self.assertEqual(unwrapped_result.results[1].header.metadata["bar"], 1)

    def test_merger_works_on_unowned_bits(self):
        """The Terra data model has bits as the fundamental object, which are not owned by
        registers.  This has not historically been the case, however, and the circuit merger used to
        make bad assumptions."""
        backend = FakeBogota()
        # This creates fundamental bits.  They are put _in_ a register, but the register doesn't own
        # them.
        qubits, clbits = [Qubit() for _ in [None] * 5], [Clbit() for _ in [None] * 5]
        qreg = QuantumRegister(name="q", bits=qubits)
        creg = ClassicalRegister(name="c", bits=clbits)

        qc1 = QuantumCircuit(qreg, creg)
        qc1.cx(0, 1)
        qc1.measure(0, 0)
        qc1.measure(1, 1)

        qc2 = QuantumCircuit(qreg, creg)
        qc2.cx(2, 3)
        qc2.cx(3, 4)
        qc2.measure(2, 2)
        qc2.measure(3, 3)
        qc2.measure(4, 4)

        merger = CircuitMerger([qc1, qc2], backend=backend)
        merged_circuit = merger.merge_circuits()

        ops = merged_circuit.count_ops()
        received = {"cx": ops.get("cx"), "measure": ops.get("measure")}
        expected = {"cx": 3, "measure": 5}
        self.assertEqual(received, expected)

    def test_single_circuit_with_init_delay_no_resets(self):
        """Test single circuit with initialization delay and no resets.

        A regression test for issue #343.
        """

        qc1, _ = _create_test_circuits()
        merger = CircuitMerger([qc1], backend=self.sim_backend)
        merged_circuit = merger.merge_circuits(
            init_num_resets=0,
            init_delay=1e-4,
            init_delay_unit="s",
        )

        delays_found = 0
        for inst in merged_circuit.data:
            operation = inst.operation
            if isinstance(operation, Delay):
                delays_found += 1
                self.assertEqual(operation.duration, 100016)
                self.assertEqual(operation.unit, "dt")
        self.assertEqual(delays_found, 4)

    def test_unwrap_results_meas_level_classified(self):
        """Test unwrapping circuits with measurement level 2."""
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

    def test_unwrap_results_meas_level_kerneled_avg(self):
        """Test unwrapping circuits with measurement level 1.

        Based on this test:
        https://github.com/Qiskit/qiskit-terra/blob/main/test/python/result/test_result.py#L478
        """
        qc0, qc1 = _create_test_circuits()
        merger = CircuitMerger([qc0, qc1], backend=self.sim_backend)
        merged_circuit = merger.merge_circuits()
        self.assertIsInstance(merged_circuit, QuantumCircuit)

        # Prepare a merged level 1 result.
        result_qc0 = execute(qc0, backend=self.sim_backend, num_shots=3).result()
        result_qc1 = execute(qc1, backend=self.sim_backend, num_shots=3).result()

        result_merged = execute(merged_circuit, backend=self.sim_backend, num_shots=3).result()

        # Dummy IQ data as stored in the transport format
        raw_memory_0 = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]]
        raw_memory_1 = [[1.0, 0.0], [0.0, 1.0]]

        result_qc0.results[0].meas_level = 1
        result_qc0.results[0].meas_return = "avg"
        result_qc0.results[0].data = ExperimentResultData.from_dict(
            dict(memory=raw_memory_0, **result_qc0.results[0].data.to_dict())
        )
        result_qc1.results[0].meas_level = 1
        result_qc1.results[0].meas_return = "avg"
        result_qc1.results[0].data = ExperimentResultData.from_dict(
            dict(memory=raw_memory_1, **result_qc1.results[0].data.to_dict())
        )

        raw_memory_merged = raw_memory_0 + raw_memory_1
        result_merged.results[0].meas_level = 1
        result_merged.results[0].meas_return = "avg"
        result_merged.results[0].data = ExperimentResultData.from_dict(
            dict(memory=raw_memory_merged)
        )

        # Unwrap and analyze unwrapped results.
        unwrapped_result = merger.unwrap_results(result_merged)

        np.testing.assert_almost_equal(result_qc0.get_memory(0), unwrapped_result.get_memory(0))
        np.testing.assert_almost_equal(result_qc1.get_memory(0), unwrapped_result.get_memory(1))

    def test_unwrap_results_meas_level_kerneled_single(self):
        """Test unwrapping circuits with measurement level 1.

        Based on this test:
        https://github.com/Qiskit/qiskit-terra/blob/main/test/python/result/test_result.py#L494
        """
        qc0, qc1 = _create_test_circuits()
        merger = CircuitMerger([qc0, qc1], backend=self.sim_backend)
        merged_circuit = merger.merge_circuits()
        self.assertIsInstance(merged_circuit, QuantumCircuit)

        # Prepare a merged level 1 result.
        result_qc0 = execute(qc0, backend=self.sim_backend, num_shots=3).result()
        result_qc1 = execute(qc1, backend=self.sim_backend, num_shots=3).result()

        result_merged = execute(merged_circuit, backend=self.sim_backend, num_shots=3).result()

        # Dummy IQ data as stored in the transport format
        raw_memory_0 = [
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]],
        ]
        raw_memory_1 = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]

        result_qc0.results[0].meas_level = 1
        result_qc0.results[0].meas_return = "single"
        result_qc0.results[0].data = ExperimentResultData.from_dict(
            dict(memory=raw_memory_0, **result_qc0.results[0].data.to_dict())
        )
        result_qc1.results[0].meas_level = 1
        result_qc1.results[0].meas_return = "single"
        result_qc1.results[0].data = ExperimentResultData.from_dict(
            dict(memory=raw_memory_1, **result_qc1.results[0].data.to_dict())
        )

        raw_memory_merged = []
        for i, _ in enumerate(raw_memory_0):
            raw_memory_merged.append(raw_memory_0[i] + raw_memory_1[i])
        result_merged.results[0].meas_level = 1
        result_merged.results[0].meas_return = "single"
        result_merged.results[0].data = ExperimentResultData.from_dict(
            dict(memory=raw_memory_merged)
        )

        # Unwrap and analyze unwrapped results.
        unwrapped_result = merger.unwrap_results(result_merged)

        np.testing.assert_almost_equal(result_qc0.get_memory(0), unwrapped_result.get_memory(0))
        np.testing.assert_almost_equal(result_qc1.get_memory(0), unwrapped_result.get_memory(1))
