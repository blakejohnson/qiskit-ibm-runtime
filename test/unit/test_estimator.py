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

from math import ceil
from test.unit import combine
import unittest

from ddt import ddt
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator as RefEstimator
from qiskit.primitives import EstimatorResult
from qiskit.providers.fake_provider import FakeBogota, FakeMontreal
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.quantum_info.random import random_pauli_list

from programs.estimator import Estimator, PauliTwirledMitigation, main, CircuitCache
from .mock.mock_cache import MockCache

# TODO: remove this class when non-flexible interface is no longer supported in provider
@ddt
class TestEstimatorCircuitIndices(unittest.TestCase):
    """Test Estimator with circuit indices."""

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
        est = Estimator(Aer.get_backend("aer_simulator"), [circuit], [matrix])
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(circuit_indices=[0], observable_indices=[0])
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate(self):
        """test to estimate"""
        backend = Aer.get_backend("aer_simulator")
        est = Estimator(backend, [self.ansatz], [self.observable])
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(
            circuit_indices=[0], observable_indices=[0], parameter_values=[[0, 1, 1, 2, 3, 5]]
        )
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate_without_grouping(self):
        """test to estimate without grouping"""
        backend = Aer.get_backend("aer_simulator")
        est = Estimator(backend, [self.ansatz], [self.observable], abelian_grouping=False)
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(
            circuit_indices=[0], observable_indices=[0], parameter_values=[[0, 1, 1, 2, 3, 5]]
        )
        self.assertEqual(len(est.transpiled_circuits), 5)
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate_multi_params(self):
        """test to estimate with multiple parameters"""
        backend = Aer.get_backend("aer_simulator")
        est = Estimator(backend, [self.ansatz], [self.observable])
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(
            circuit_indices=[0, 0],
            observable_indices=[0, 0],
            parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]],
        )
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.dtype, np.float64)
        np.testing.assert_allclose(result.values, [-1.283, -1.315], rtol=1e-03)

    def test_estimate_no_params(self):
        """test to estimate without parameters"""
        backend = Aer.get_backend("aer_simulator")
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        est = Estimator(backend, [circuit], [self.observable])
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(circuit_indices=[0], observable_indices=[0])
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
        est = Estimator(backend, circuit, ["ZZZ", "III"])
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(circuit_indices=[0, 0], observable_indices=[0, 1])
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.dtype, np.float64)
        np.testing.assert_allclose(result.values, [0.0044, 1.0], rtol=1e-03)

    def test_estimate_with_ndarray(self):
        """test to estimate"""
        backend = Aer.get_backend("aer_simulator")
        param = np.array([[0, 1, 1, 2, 3, 5]])
        est = Estimator(backend, [self.ansatz], [self.observable])
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(circuit_indices=[0], observable_indices=[0], parameter_values=param)
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_skip_transpilation(self):
        """test for ``skip_transpilation`` option"""
        backend = Aer.get_backend("aer_simulator")
        est = Estimator(backend, [self.ansatz], [self.observable], skip_transpilation=False)
        self.assertEqual(len(est.transpiled_circuits[0]), 12)

        est = Estimator(backend, [self.ansatz], [self.observable], skip_transpilation=True)
        self.assertEqual(len(est.transpiled_circuits[0]), 5)

    def test_call_validation(self):
        """Test for validations"""
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op1 = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        est = Estimator(Aer.get_backend("aer_simulator"), [qc1, qc2], [op1, op2], [[]] * 2)
        with self.assertRaises(QiskitError):
            est.run([0], [1], [[]])
        with self.assertRaises(QiskitError):
            est.run([1], [0], [[]])
        with self.assertRaises(QiskitError):
            est.run([0], [0], [[1e4]])
        with self.assertRaises(QiskitError):
            est.run([1], [1], [[1, 2]])

    @combine(noise=[True, False], grouping=[True, False], num_qubits=[2, 5])
    def test_compare_reference(self, noise, grouping, num_qubits):
        """Test to compare results of Estimator with those of the reference one"""
        size = 10
        seed = 15
        shots = 10000
        num_twirled_circuits = 4
        shots_calibration = 8192

        param_x = Parameter("x")
        circ = QuantumCircuit(num_qubits)
        circ.ry(param_x, range(num_qubits))
        obs = SparsePauliOp(
            random_pauli_list(num_qubits=num_qubits, size=size, seed=seed, phase=False)
        )
        params = np.linspace(0, np.pi, 5)[:, np.newaxis]
        with RefEstimator(circuits=[circ], observables=[obs]) as ref_est:
            result = ref_est([0] * len(params), [0] * len(params), params)
            targets = result.values
        backend = FakeMontreal() if noise else Aer.get_backend("aer_simulator")
        mit = PauliTwirledMitigation(
            backend=backend,
            seed=seed,
            num_twirled_circuits=num_twirled_circuits,
            shots_calibration=shots_calibration,
        )
        est = Estimator(
            circuits=[circ],
            observables=[obs],
            backend=backend,
            abelian_grouping=grouping,
            pauli_twirled_mitigation=mit,
        )
        result = est.run(
            [0] * len(params), [0] * len(params), params, shots=shots, seed_simulator=seed
        )
        np.testing.assert_allclose(result.values, targets, rtol=1e-1, atol=1e-1)
        shots2 = int(ceil(shots / num_twirled_circuits)) * num_twirled_circuits
        for meta in result.metadata:
            self.assertEqual(meta["shots"], shots2)
            self.assertEqual(meta["readout_mitigation_num_twirled_circuits"], num_twirled_circuits)
            self.assertEqual(meta["readout_mitigation_shots_calibration"], shots_calibration)


@ddt
class TestEstimatorCircuitIds(unittest.TestCase):
    """Test Estimator with circuit ids."""

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
        self.circuit_id = str(id(self.ansatz))

    def test_init_observable_from_operator(self):
        """test to estimate without parameters"""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        circuit_id = str(id(circuit))
        matrix = Operator(
            [
                [-1.06365335, 0.0, 0.0, 0.1809312],
                [0.0, -1.83696799, 0.1809312, 0.0],
                [0.0, 0.1809312, -0.24521829, 0.0],
                [0.1809312, 0.0, 0.0, -1.06365335],
            ]
        )
        est = Estimator(
            backend=Aer.get_backend("aer_simulator"),
            circuits={circuit_id: circuit},
            observables=[matrix],
            circuit_ids=[circuit_id],
        )
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(observable_indices=[0])
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate(self):
        """test to estimate"""
        backend = Aer.get_backend("aer_simulator")
        est = Estimator(
            backend=backend,
            circuits={self.circuit_id: self.ansatz},
            observables=[self.observable],
            circuit_ids=[self.circuit_id],
        )
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(observable_indices=[0], parameter_values=[[0, 1, 1, 2, 3, 5]])
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate_without_grouping(self):
        """test to estimate without grouping"""
        backend = Aer.get_backend("aer_simulator")
        est = Estimator(
            backend=backend,
            circuits={self.circuit_id: self.ansatz},
            observables=[self.observable],
            abelian_grouping=False,
            circuit_ids=[self.circuit_id],
        )
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(observable_indices=[0], parameter_values=[[0, 1, 1, 2, 3, 5]])
        self.assertEqual(len(est.transpiled_circuits), 5)
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_estimate_multi_params(self):
        """test to estimate with multiple parameters"""
        backend = Aer.get_backend("aer_simulator")
        est = Estimator(
            backend=backend,
            circuits={self.circuit_id: self.ansatz},
            observables=[self.observable],
            circuit_ids=[self.circuit_id, self.circuit_id],
        )
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(
            observable_indices=[0, 0],
            parameter_values=[[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]],
        )
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.dtype, np.float64)
        np.testing.assert_allclose(result.values, [-1.283, -1.315], rtol=1e-03)

    def test_estimate_no_params(self):
        """test to estimate without parameters"""
        backend = Aer.get_backend("aer_simulator")
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        circuit_id = str(id(circuit))
        est = Estimator(
            backend=backend,
            circuits={circuit_id: circuit},
            observables=[self.observable],
            circuit_ids=[circuit_id],
        )
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(observable_indices=[0])
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
        circuit_id = str(id(circuit))
        est = Estimator(
            backend=backend,
            circuits={circuit_id: circuit},
            observables=["ZZZ", "III"],
            circuit_ids=[circuit_id, circuit_id],
        )
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(observable_indices=[0, 1])
        self.assertIsInstance(result, EstimatorResult)
        self.assertEqual(result.values.dtype, np.float64)
        np.testing.assert_allclose(result.values, [0.0044, 1.0], rtol=1e-03)

    def test_estimate_with_ndarray(self):
        """test to estimate"""
        backend = Aer.get_backend("aer_simulator")
        param = np.array([[0, 1, 1, 2, 3, 5]])
        est = Estimator(
            backend=backend,
            circuits={self.circuit_id: self.ansatz},
            observables=[self.observable],
            circuit_ids=[self.circuit_id],
        )
        est.set_transpile_options(seed_transpiler=15)
        est.set_run_options(seed_simulator=15, shots=10000)
        result = est.run(observable_indices=[0], parameter_values=param)
        self.assertIsInstance(result, EstimatorResult)
        self.assertIsInstance(result.values[0], float)
        self.assertAlmostEqual(result.values[0], -1.283, places=2)

    def test_skip_transpilation(self):
        """test for ``skip_transpilation`` option"""
        backend = Aer.get_backend("aer_simulator")
        est = Estimator(
            backend=backend,
            circuits={self.circuit_id: self.ansatz},
            observables=[self.observable],
            skip_transpilation=False,
        )
        self.assertEqual(len(est.transpiled_circuits[0]), 12)

        est = Estimator(
            backend=backend,
            circuits={self.circuit_id: self.ansatz},
            observables=[self.observable],
            skip_transpilation=True,
        )
        self.assertEqual(len(est.transpiled_circuits[0]), 5)

    def test_call_validation(self):
        """Test for validations"""
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)
        qc1_id = str(id(qc1))
        qc2_id = str(id(qc2))

        op1 = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        est1 = Estimator(
            backend=Aer.get_backend("aer_simulator"),
            circuits={qc1_id: qc1, qc2_id: qc2},
            observables=[op1, op2],
            parameters=[[]] * 2,
            circuit_ids=[qc1_id],
        )
        with self.assertRaises(QiskitError):
            est1.run(observable_indices=[1], parameter_values=[[]])
        with self.assertRaises(QiskitError):
            est1.run(observable_indices=[0], parameter_values=[[1e4]])
        est2 = Estimator(
            backend=Aer.get_backend("aer_simulator"),
            circuits={qc1_id: qc1, qc2_id: qc2},
            observables=[op1, op2],
            parameters=[[]] * 2,
            circuit_ids=[qc2_id],
        )
        with self.assertRaises(QiskitError):
            est2.run(observable_indices=[0], parameter_values=[[]])
        with self.assertRaises(QiskitError):
            est2.run(observable_indices=[1], parameter_values=[[1, 2]])

    @combine(noise=[True, False], grouping=[True, False], num_qubits=[2, 5])
    def test_compare_reference(self, noise, grouping, num_qubits):
        """Test to compare results of Estimator with those of the reference one"""
        size = 10
        seed = 15
        shots = 10000
        num_twirled_circuits = 4
        shots_calibration = 8192

        param_x = Parameter("x")
        circ = QuantumCircuit(num_qubits)
        circ.ry(param_x, range(num_qubits))
        circ_id = str(id(circ))
        obs = SparsePauliOp(
            random_pauli_list(num_qubits=num_qubits, size=size, seed=seed, phase=False)
        )
        params = np.linspace(0, np.pi, 5)[:, np.newaxis]
        with RefEstimator(circuits=[circ], observables=[obs]) as ref_est:
            result = ref_est([0] * len(params), [0] * len(params), params)
            targets = result.values
        backend = FakeMontreal() if noise else Aer.get_backend("aer_simulator")
        mit = PauliTwirledMitigation(
            backend=backend,
            seed=seed,
            num_twirled_circuits=num_twirled_circuits,
            shots_calibration=shots_calibration,
        )
        est = Estimator(
            circuits={circ_id: circ},
            observables=[obs],
            backend=backend,
            abelian_grouping=grouping,
            pauli_twirled_mitigation=mit,
            circuit_ids=[circ_id] * len(params),
        )
        result = est.run(
            observable_indices=[0] * len(params),
            parameter_values=params,
            shots=shots,
            seed_simulator=seed,
        )
        np.testing.assert_allclose(result.values, targets, rtol=1e-1, atol=1e-1)
        shots2 = int(ceil(shots / num_twirled_circuits)) * num_twirled_circuits
        for meta in result.metadata:
            self.assertEqual(meta["shots"], shots2)
            self.assertEqual(meta["readout_mitigation_num_twirled_circuits"], num_twirled_circuits)
            self.assertEqual(meta["readout_mitigation_shots_calibration"], shots_calibration)


# TODO: remove this class when non-flexible interface is no longer supported in provider
@ddt
class TestEstimatorMainCircuitIndices(unittest.TestCase):
    """Test Estimator main with circuit indices."""

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
    def test_no_noise_single_param(self, resilience_level):
        """Test for main without noise with single parameter set"""
        backend = Aer.get_backend("aer_simulator")
        shots = 10000
        circuits = [0]
        observables = [0]
        params = [[0, 1, 1, 2, 3, 5]]
        with RefEstimator([self.ansatz], [self.observable]) as estimator:
            target = estimator(circuits, observables, params).values
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[self.ansatz],
            observables=[self.observable],
            circuit_indices=circuits,
            observable_indices=observables,
            parameter_values=params,
            run_options={"shots": shots, "seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], target, rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 1)
        if resilience_level == 0:
            self.assertEqual(result["metadata"][0]["shots"], shots)
        else:
            div = result["metadata"][0]["readout_mitigation_num_twirled_circuits"]
            self.assertEqual(result["metadata"][0]["shots"], int(ceil(shots / div)) * div)

    @combine(resilience_level=[0, 1])
    def test_no_noise_multiple_params(self, resilience_level):
        """Test for main without noise with multiple parameter sets"""
        backend = Aer.get_backend("aer_simulator")
        shots = 10000
        circuits = [0, 0]
        observables = [0, 0]
        params = [[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]]
        with RefEstimator([self.ansatz], [self.observable]) as estimator:
            target = estimator(circuits, observables, params).values
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[self.ansatz],
            observables=[self.observable],
            circuit_indices=circuits,
            observable_indices=observables,
            parameter_values=params,
            run_options={"shots": shots, "seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], target, rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 2)
        if resilience_level == 0:
            self.assertEqual(result["metadata"][0]["shots"], shots)
            self.assertEqual(result["metadata"][1]["shots"], shots)
        else:
            div = result["metadata"][0]["readout_mitigation_num_twirled_circuits"]
            self.assertEqual(result["metadata"][0]["shots"], int(ceil(shots / div)) * div)
            self.assertEqual(result["metadata"][1]["shots"], int(ceil(shots / div)) * div)

    @combine(noise=[True, False], shots=[10000, 20000])
    def test_mitigation(self, noise, shots):
        """Test for mitigation w/ and w/o noise"""
        backend = FakeBogota() if noise else Aer.get_backend("aer_simulator")
        resilience_level = 1
        circuits = [0, 0]
        observables = [0, 0]
        params = [[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]]
        with RefEstimator([self.ansatz], [self.observable]) as estimator:
            target = estimator(circuits, observables, params).values
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[self.ansatz],
            observables=[self.observable],
            circuit_indices=circuits,
            observable_indices=observables,
            parameter_values=params,
            run_options={"shots": shots, "seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], target, rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 2)
        div = result["metadata"][0]["readout_mitigation_num_twirled_circuits"]
        self.assertEqual(result["metadata"][0]["shots"], int(ceil(shots / div)) * div)
        self.assertEqual(result["metadata"][1]["shots"], int(ceil(shots / div)) * div)

    @combine(noise=[True, False], resilience_level=[0, 1], shots=[100])
    def test_identity(self, noise, resilience_level, shots):
        """Test for identity observable"""
        backend = FakeBogota() if noise else Aer.get_backend("aer_simulator")
        circuit = RealAmplitudes(num_qubits=5, reps=2, entanglement="full")
        num_parameters = circuit.num_parameters
        observable = SparsePauliOp("IIIII")
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[circuit],
            observables=[observable],
            circuit_indices=[0, 0],
            observable_indices=[0, 0],
            parameter_values=[[0] * num_parameters, [1] * num_parameters],
            run_options={"shots": shots, "seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], [1, 1], rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 2)
        if resilience_level == 0:
            self.assertEqual(result["metadata"][0]["shots"], shots)
            self.assertEqual(result["metadata"][1]["shots"], shots)
        else:
            div = result["metadata"][0]["readout_mitigation_num_twirled_circuits"]
            self.assertEqual(result["metadata"][0]["shots"], int(ceil(shots / div)) * div)
            self.assertEqual(result["metadata"][1]["shots"], int(ceil(shots / div)) * div)

    @combine(noise=[True, False], resilience_level=[0, 1])
    def test_identity_wo_shots(self, noise, resilience_level):
        """Test for identity observable without `shots`"""
        backend = FakeBogota() if noise else Aer.get_backend("aer_simulator")
        circuit = RealAmplitudes(num_qubits=5, reps=2, entanglement="full")
        num_parameters = circuit.num_parameters
        observable = SparsePauliOp("IIIII")
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[circuit],
            observables=[observable],
            circuit_indices=[0, 0],
            observable_indices=[0, 0],
            parameter_values=[[0] * num_parameters, [1] * num_parameters],
            run_options={"seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], [1, 1], rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 2)


@ddt
class TestEstimatorMainCircuitIds(unittest.TestCase):
    """Test Estimator main with circuit ids."""

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
        self.circuit_id = str(id(self.ansatz))

    @combine(resilience_level=[0, 1])
    def test_no_noise_single_param(self, resilience_level):
        """Test for main without noise with single parameter set"""
        backend = Aer.get_backend("aer_simulator")
        shots = 10000
        circuits = [0]
        observables = [0]
        params = [[0, 1, 1, 2, 3, 5]]
        with RefEstimator([self.ansatz], [self.observable]) as estimator:
            target = estimator(circuits, observables, params).values
        result = main(
            backend=backend,
            user_messenger=None,
            circuits={self.circuit_id: self.ansatz},
            observables=[self.observable],
            circuit_ids=[self.circuit_id],
            observable_indices=observables,
            parameter_values=params,
            run_options={"shots": shots, "seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], target, rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 1)
        if resilience_level == 0:
            self.assertEqual(result["metadata"][0]["shots"], shots)
        else:
            div = result["metadata"][0]["readout_mitigation_num_twirled_circuits"]
            self.assertEqual(result["metadata"][0]["shots"], int(ceil(shots / div)) * div)

    @combine(resilience_level=[0, 1])
    def test_no_noise_multiple_params(self, resilience_level):
        """Test for main without noise with multiple parameter sets"""
        backend = Aer.get_backend("aer_simulator")
        shots = 10000
        circuits = [0, 0]
        observables = [0, 0]
        params = [[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]]
        with RefEstimator([self.ansatz], [self.observable]) as estimator:
            target = estimator(circuits, observables, params).values
        result = main(
            backend=backend,
            user_messenger=None,
            circuits={self.circuit_id: self.ansatz},
            observables=[self.observable],
            circuit_ids=[self.circuit_id, self.circuit_id],
            observable_indices=observables,
            parameter_values=params,
            run_options={"shots": shots, "seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], target, rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 2)
        if resilience_level == 0:
            self.assertEqual(result["metadata"][0]["shots"], shots)
            self.assertEqual(result["metadata"][1]["shots"], shots)
        else:
            div = result["metadata"][0]["readout_mitigation_num_twirled_circuits"]
            self.assertEqual(result["metadata"][0]["shots"], int(ceil(shots / div)) * div)
            self.assertEqual(result["metadata"][1]["shots"], int(ceil(shots / div)) * div)

    @combine(noise=[True, False], shots=[10000, 20000])
    def test_mitigation(self, noise, shots):
        """Test for mitigation w/ and w/o noise"""
        backend = FakeBogota() if noise else Aer.get_backend("aer_simulator")
        resilience_level = 1
        circuits = [0, 0]
        observables = [0, 0]
        params = [[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]]
        with RefEstimator([self.ansatz], [self.observable]) as estimator:
            target = estimator(circuits, observables, params).values
        result = main(
            backend=backend,
            user_messenger=None,
            circuits={self.circuit_id: self.ansatz},
            observables=[self.observable],
            circuit_ids=[self.circuit_id, self.circuit_id],
            observable_indices=observables,
            parameter_values=params,
            run_options={"shots": shots, "seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], target, rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 2)
        div = result["metadata"][0]["readout_mitigation_num_twirled_circuits"]
        self.assertEqual(result["metadata"][0]["shots"], int(ceil(shots / div)) * div)
        self.assertEqual(result["metadata"][1]["shots"], int(ceil(shots / div)) * div)

    @combine(noise=[True, False], resilience_level=[0, 1], shots=[100])
    def test_identity(self, noise, resilience_level, shots):
        """Test for identity observable"""
        backend = FakeBogota() if noise else Aer.get_backend("aer_simulator")
        circuit = RealAmplitudes(num_qubits=5, reps=2, entanglement="full")
        circuit_id = str(id(circuit))
        num_parameters = circuit.num_parameters
        observable = SparsePauliOp("IIIII")
        result = main(
            backend=backend,
            user_messenger=None,
            circuits={circuit_id: circuit},
            observables=[observable],
            circuit_ids=[circuit_id, circuit_id],
            observable_indices=[0, 0],
            parameter_values=[[0] * num_parameters, [1] * num_parameters],
            run_options={"shots": shots, "seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], [1, 1], rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 2)
        if resilience_level == 0:
            self.assertEqual(result["metadata"][0]["shots"], shots)
            self.assertEqual(result["metadata"][1]["shots"], shots)
        else:
            div = result["metadata"][0]["readout_mitigation_num_twirled_circuits"]
            self.assertEqual(result["metadata"][0]["shots"], int(ceil(shots / div)) * div)
            self.assertEqual(result["metadata"][1]["shots"], int(ceil(shots / div)) * div)

    @combine(noise=[True, False], resilience_level=[0, 1])
    def test_identity_wo_shots(self, noise, resilience_level):
        """Test for identity observable without `shots`"""
        backend = FakeBogota() if noise else Aer.get_backend("aer_simulator")
        circuit = RealAmplitudes(num_qubits=5, reps=2, entanglement="full")
        circuit_id = str(id(circuit))
        num_parameters = circuit.num_parameters
        observable = SparsePauliOp("IIIII")
        result = main(
            backend=backend,
            user_messenger=None,
            circuits={circuit_id: circuit},
            observables=[observable],
            circuit_ids=[circuit_id, circuit_id],
            observable_indices=[0, 0],
            parameter_values=[[0] * num_parameters, [1] * num_parameters],
            run_options={"seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pauli_twirled_mitigation": {"seed_mitigation": 15, "seed_simulator": 15},
            },
        )
        np.testing.assert_allclose(result["values"], [1, 1], rtol=1e-2)
        self.assertEqual(len(result["metadata"]), 2)


@ddt
class TestEstimatorCircuitCache(unittest.TestCase):
    """Test CircuitCache class in Estimator primitive"""

    def setUp(self):
        super().setUp()
        self._bell = QuantumCircuit(2)
        self._bell.h(0)
        self._bell.cx(0, 1)
        self._bell.measure_all()
        self._bell_id = "bell_12345"
        self._bell_digest = "b87d81d827c167fba859253fa5622e2538c07f3fa5745a525248ef3e26f65ed9"

    def test_initialize_transpiled_and_raw_circuits(self):
        """Test method to initialize a list of transpiled circuits and a list of raw circuits."""
        circuit_cache = CircuitCache(cache=MockCache())
        circuit_cache.initialize_transpiled_and_raw_circuits(
            circuits_map={self._bell_id: self._bell},
            circuit_ids=[self._bell_id, self._bell_id],
            backend_name="ibmq_qasm_simulator",
            transpile_options={"optimization_level": 1},
        )
        self.assertEqual(circuit_cache.transpiled_circuits, [self._bell_digest] * 2)
        self.assertEqual(circuit_cache.raw_circuits, [self._bell])

    def test_update_cache_and_merge_transpiled_circuits(self):
        """Test method to update cache with transpiled circuits and raw circuits and merge recently
        transpiled circuits with transpiled circuits retrieved from cache into a single list."""
        cache = MockCache()
        # Job 1
        circuit_cache_1 = CircuitCache(cache=cache)
        circuit_cache_1.initialize_transpiled_and_raw_circuits(
            circuits_map={self._bell_id: self._bell},
            circuit_ids=[self._bell_id, self._bell_id],
            backend_name="ibmq_qasm_simulator",
            transpile_options={"optimization_level": 1},
        )
        circuit_cache_1.update_cache_and_merge_transpiled_circuits(transpiled_circuits=[self._bell])

        # Job 2
        circuit_cache_2 = CircuitCache(cache=cache)
        circuit_cache_2.initialize_transpiled_and_raw_circuits(
            circuits_map={},
            circuit_ids=[self._bell_id, self._bell_id, self._bell_id],
            backend_name="ibmq_qasm_simulator",
            transpile_options={"optimization_level": 1},
        )
        self.assertEqual(circuit_cache_2.transpiled_circuits, [self._bell] * 3)
