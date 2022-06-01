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

"""Unit tests for Estimator."""

from ddt import ddt
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import EstimatorResult
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.test import QiskitTestCase

from programs.estimator import Estimator, main


class TestEstimator(QiskitTestCase):
    """Test Estimator"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    def test_init_observable_from_operator(self):
        """test to estimate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        matrix = Operator(
            [
                [-1.06365335, 0.0, 0.0, 0.1809312],
                [0.0, -1.83696799, 0.1809312, 0.0],
                [0.0, 0.1809312, -0.24521829, 0.0],
                [0.1809312, 0.0, 0.0, -1.06365335],
            ]
        )
        with Estimator(Aer.get_backend("aer_simulator"), [circuit], [matrix]) as est:
            est.set_run_options(seed_simulator=15, shots=10000)
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate(self):
        """test to estimate"""
        backend = Aer.get_backend("aer_simulator")
        with Estimator(backend, [self.ansatz], [self.observable]) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15, shots=10000)
            result = est(parameter_values=[0, 1, 1, 2, 3, 5])
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate_without_grouping(self):
        """test to estimate without grouping"""
        backend = Aer.get_backend("aer_simulator")
        with Estimator(backend, [self.ansatz], [self.observable], abelian_grouping=False) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15, shots=10000)
            result = est(parameter_values=[0, 1, 1, 2, 3, 5])
            self.assertEqual(len(est.transpiled_circuits), 5)
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate_multi_params(self):
        """test to estimate with multiple parameters"""
        backend = Aer.get_backend("aer_simulator")
        with Estimator(backend, [self.ansatz], [self.observable]) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15, shots=10000)
            result = est(parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]])
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.dtype, np.float64)
        np.testing.assert_allclose(result.values, [-1.283, -1.315], rtol=1e-03)

    def test_estimate_no_params(self):
        """test to estimate without parameters"""
        backend = Aer.get_backend("aer_simulator")
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        with Estimator(backend, [circuit], [self.observable]) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15, shots=10000)
            result = est()
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_run_with_multiple_observables_and_none_parameters(self):
        """test to estimate without parameters"""
        backend = Aer.get_backend("aer_simulator")
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        with Estimator(backend, circuit, ["ZZZ", "III"]) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15, shots=10000)
            result = est(circuit_indices=[0, 0])
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.dtype, np.float64)
        np.testing.assert_allclose(result.values, [0.0044, 1.0], rtol=1e-03)

    def test_estimate_with_ndarray(self):
        """test to estimate"""
        backend = Aer.get_backend("aer_simulator")
        param = np.array([[0, 1, 1, 2, 3, 5]])
        with Estimator(backend, [self.ansatz], [self.observable]) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15, shots=10000)
            result = est(parameter_values=param)
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_skip_transpilation(self):
        """test for ``skip_transpilation`` option"""
        backend = Aer.get_backend("aer_simulator")
        with Estimator(backend, [self.ansatz], [self.observable], skip_transpilation=False) as est:
            self.assertEqual(len(est.transpiled_circuits[0]), 12)

        with Estimator(backend, [self.ansatz], [self.observable], skip_transpilation=True) as est:
            self.assertEqual(len(est.transpiled_circuits[0]), 5)

    def test_call_validation(self):
        """Test for validations"""
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op1 = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        with Estimator(Aer.get_backend("aer_simulator"), [qc1, qc2], [op1, op2], [[]] * 2) as est:
            with self.assertRaises(QiskitError):
                est([0], [1], [[]])
            with self.assertRaises(QiskitError):
                est([1], [0], [[]])
            with self.assertRaises(QiskitError):
                est([0], [0], [[1e4]])
            with self.assertRaises(QiskitError):
                est([1], [1], [[1, 2]])


@ddt
class TestEstimatorMain(QiskitTestCase):
    """Test Estimator main"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    def test_main(self):
        """Test main"""
        backend = Aer.get_backend("aer_simulator")
        shots = 10000
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[self.ansatz],
            observables=[self.observable],
            circuit_indices=[0],
            observable_indices=[0],
            parameter_values=[[0, 1, 1, 2, 3, 5]],
            run_options={"shots": shots, "seed_simulator": 15},
        )
        np.testing.assert_allclose(result["values"], [-1.283], rtol=1e-3)
        self.assertEqual(len(result["metadata"]), 1)
        self.assertEqual(result["metadata"][0]["shots"], shots)

    def test_main2(self):
        """Test main 2"""
        backend = Aer.get_backend("aer_simulator")
        shots = 10000
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[self.ansatz],
            observables=[self.observable],
            circuit_indices=[0, 0],
            observable_indices=[0, 0],
            parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]],
            run_options={"shots": shots, "seed_simulator": 15},
        )
        np.testing.assert_allclose(result["values"], [-1.283, -1.315], rtol=1e-3)
        self.assertEqual(len(result["metadata"]), 2)
        self.assertEqual(result["metadata"][0]["shots"], shots)
        self.assertEqual(result["metadata"][1]["shots"], shots)
