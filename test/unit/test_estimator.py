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

from test.unit import combine
import unittest
from warnings import catch_warnings

from ddt import ddt
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator as RefEstimator
from qiskit.primitives import EstimatorResult
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.quantum_info.random import random_pauli_list
from qiskit.providers.fake_provider import FakeMontreal

from programs.estimator import Estimator, main


@ddt
class TestEstimator(unittest.TestCase):
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
            result = est(circuits=[0], observables=[0])
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate(self):
        """test to estimate"""
        backend = Aer.get_backend("aer_simulator")
        with Estimator(backend, [self.ansatz], [self.observable]) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15, shots=10000)
            result = est(circuits=[0], observables=[0], parameter_values=[[0, 1, 1, 2, 3, 5]])
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate_without_grouping(self):
        """test to estimate without grouping"""
        backend = Aer.get_backend("aer_simulator")
        with Estimator(backend, [self.ansatz], [self.observable], abelian_grouping=False) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15, shots=10000)
            result = est(circuits=[0], observables=[0], parameter_values=[[0, 1, 1, 2, 3, 5]])
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
            result = est(
                circuits=[0, 0],
                observables=[0, 0],
                parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]],
            )
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
            result = est(circuits=[0], observables=[0])
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
            result = est(circuits=[0, 0], observables=[0, 1])
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
            result = est(circuits=[0], observables=[0], parameter_values=param)
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

    @combine(noise=[False], grouping=[True, False], num_qubits=[2, 5])
    def test_compare_reference(self, noise, grouping, num_qubits):
        """Test to compare results of Estimator with those of the reference one"""
        size = 10
        seed = 123
        shots = 10000

        param_x = Parameter("x")
        circ = QuantumCircuit(num_qubits)
        circ.ry(param_x, range(num_qubits))
        obs = SparsePauliOp(
            random_pauli_list(num_qubits=num_qubits, size=size, seed=seed, phase=False)
        )
        params = np.linspace(0, np.pi, 5)[:, np.newaxis]
        with RefEstimator(circuits=[circ], observables=[obs]) as est:
            result = est([0] * len(params), [0] * len(params), params)
            targets = result.values
        backend = FakeMontreal() if noise else Aer.get_backend("aer_simulator")
        with Estimator(
            circuits=[circ],
            observables=[obs],
            backend=backend,
            abelian_grouping=grouping,
        ) as est:
            result = est(
                [0] * len(params), [0] * len(params), params, shots=shots, seed_simulator=15
            )
        np.testing.assert_allclose(result.values, targets, rtol=1e-1)


@ddt
class TestEstimatorMain(unittest.TestCase):
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

    @combine(resilience_level=[0, 1])
    def test_main(self, resilience_level):
        """Test main"""
        backend = Aer.get_backend("aer_simulator")
        shots = 10000

        with catch_warnings(record=True) as warn_cm:
            result = main(
                backend=backend,
                user_messenger=None,
                circuits=[self.ansatz],
                observables=[self.observable],
                circuit_indices=[0],
                observable_indices=[0],
                parameter_values=[[0, 1, 1, 2, 3, 5]],
                run_options={"shots": shots, "seed_simulator": 15},
                transpilation_settings={"seed_transpiler": 15},
                resilience_settings={"level": resilience_level},
            )
            self.assertEqual(len(warn_cm), resilience_level)
        np.testing.assert_allclose(result["values"], [-1.283], rtol=1e-3)
        self.assertEqual(len(result["metadata"]), 1)
        self.assertEqual(result["metadata"][0]["shots"], shots)

    @combine(resilience_level=[0, 1])
    def test_main2(self, resilience_level):
        """Test main 2"""
        backend = Aer.get_backend("aer_simulator")
        shots = 10000
        with catch_warnings(record=True) as warn_cm:
            result = main(
                backend=backend,
                user_messenger=None,
                circuits=[self.ansatz],
                observables=[self.observable],
                circuit_indices=[0, 0],
                observable_indices=[0, 0],
                parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]],
                run_options={"shots": shots, "seed_simulator": 15},
                transpilation_settings={"seed_transpiler": 15},
                resilience_settings={"level": resilience_level},
            )
            self.assertEqual(len(warn_cm), resilience_level)
        np.testing.assert_allclose(result["values"], [-1.283, -1.315], rtol=1e-3)
        self.assertEqual(len(result["metadata"]), 2)
        self.assertEqual(result["metadata"][0]["shots"], shots)
        self.assertEqual(result["metadata"][1]["shots"], shots)
