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

"""Unit tests for Sampler."""

from test.unit import combine

from ddt import ddt
import numpy as np
from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.primitives import SamplerResult
from qiskit.test import QiskitTestCase

from programs.sampler import Sampler


@ddt
class TestSampler(QiskitTestCase):
    """Test Sampler"""

    def setUp(self):
        super().setUp()
        hadamard = QuantumCircuit(1, 1)
        hadamard.h(0)
        hadamard.measure(0, 0)
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure(0, 0)
        bell.measure(1, 1)
        self._circuit = [hadamard, bell]
        self._target = [
            {0: 0.5, 1: 0.5},
            {0: 0.5, 3: 0.5, 1: 0, 2: 0},
        ]
        self._run_config = {"seed_simulator": 15}
        self._pqc = QuantumCircuit(2, 2)
        self._pqc.compose(RealAmplitudes(num_qubits=2, reps=2), inplace=True)
        self._pqc.measure(0, 0)
        self._pqc.measure(1, 1)
        self._pqc_params = [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
        self._pqc_target = [{0: 1}, {0: 0.0148, 1: 0.3449, 2: 0.0531, 3: 0.5872}]

    def _generate_circuits_target(self, indices):
        if isinstance(indices, list):
            circuits = [self._circuit[j] for j in indices]
            target = [self._target[j] for j in indices]
        else:
            raise ValueError(f"invalid index {indices}")
        return circuits, target

    def _generate_params_target(self, indices):
        if isinstance(indices, list):
            params = [self._pqc_params[j] for j in indices]
            target = [self._pqc_target[j] for j in indices]
        else:
            raise ValueError(f"invalid index {indices}")
        return params, target

    def _compare_probs(self, prob, target):
        if not isinstance(target, list):
            target = [target]
        self.assertEqual(len(prob), len(target))
        for p, targ in zip(prob, target):
            for key, t_val in targ.items():
                if key in p:
                    self.assertAlmostEqual(p[key], t_val, places=1)
                else:
                    self.assertAlmostEqual(t_val, 0, places=1)

    @combine(indices=[[0], [1], [0, 1]], shots=[1000, 2000])
    def test_sample(self, indices, shots):
        """test to sample"""
        backend = BasicAer.get_backend("qasm_simulator")
        circuits, target = self._generate_circuits_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=circuits, backend=backend) as sampler:
                result = sampler(
                    parameter_values=[[] for _ in indices],
                    shots=shots,
                    **self._run_config,
                )
                self.assertEqual(result.metadata[0]["shots"], shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=circuits, backend=backend)
            result = sampler(
                parameter_values=[[] for _ in indices], shots=shots, **self._run_config
            )
            self.assertEqual(result.metadata[0]["shots"], shots)
            self._compare_probs(result.quasi_dists, target)

    @combine(
        indices=[[0], [1], [0, 1]],
        shots=[1000, 2000],
    )
    def test_sample_pqc(self, indices, shots):
        """test to sample a parametrized circuit"""
        backend = BasicAer.get_backend("qasm_simulator")
        params, target = self._generate_params_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=self._pqc, backend=backend) as sampler:
                sampler.set_run_options(shots=shots, **self._run_config)
                result: SamplerResult = sampler(parameter_values=params)
                self.assertEqual(result.metadata[0]["shots"], shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=self._pqc, backend=backend)
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler(parameter_values=params)
            self.assertEqual(result.metadata[0]["shots"], shots)
            self._compare_probs(result.quasi_dists, target)

    @combine(
        indices=[[0], [1], [0, 1]],
        shots=[1000, 2000],
    )
    def test_sample_with_ndarray(self, indices, shots):
        """test to sample a parametrized circuit"""
        backend = BasicAer.get_backend("qasm_simulator")
        params, target = self._generate_params_target(indices)
        params = np.asarray(params)
        with self.subTest("with-guard"):
            with Sampler(circuits=self._pqc, backend=backend) as sampler:
                sampler.set_run_options(shots=shots, **self._run_config)
                result: SamplerResult = sampler(parameter_values=params)
                self.assertEqual(result.metadata[0]["shots"], shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=self._pqc, backend=backend)
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler(parameter_values=params)
            self.assertEqual(result.metadata[0]["shots"], shots)
            self._compare_probs(result.quasi_dists, target)

    @combine(
        indices=[[0, 0], [0, 1], [1, 1]],
        shots=[1000, 2000],
    )
    def test_sample_two_pqcs(self, indices, shots):
        """test to sample two parametrized circuits"""
        backend = BasicAer.get_backend("qasm_simulator")
        circs = [self._pqc, self._pqc]
        params, target = self._generate_params_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=circs, backend=backend) as sampler:
                sampler.set_run_options(shots=shots, **self._run_config)
                result: SamplerResult = sampler(parameter_values=params)
                self.assertEqual(result.metadata[0]["shots"], shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=circs, backend=backend)
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler(parameter_values=params)
            self.assertEqual(result.metadata[0]["shots"], shots)
            self._compare_probs(result.quasi_dists, target)

    def test_call_validation(self):
        """Test for validations"""
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = RealAmplitudes(num_qubits=1, reps=1)
        qc2.measure_all()

        with Sampler(
            BasicAer.get_backend("qasm_simulator"),
            [qc1, qc2],
            [qc1.parameters, qc2.parameters],
        ) as sampler:
            with self.assertRaises(QiskitError):
                sampler([0], [[1e2]])
            with self.assertRaises(QiskitError):
                sampler([1], [[]])
            with self.assertRaises(QiskitError):
                sampler([1], [[1e2]])

    def test_empty_parameter(self):
        """Test for empty parameter"""
        n = 5
        qc = QuantumCircuit(n, n - 1)
        qc.measure(range(n - 1), range(n - 1))
        with Sampler(backend=BasicAer.get_backend("qasm_simulator"), circuits=[qc] * 10) as sampler:
            with self.subTest("one circuit"):
                result = sampler(circuit_indices=[0], shots=1000)
                self.assertEqual(len(result.metadata), 1)
                self.assertEqual(result.metadata[0]["shots"], 1000)
                self.assertEqual(len(result.quasi_dists), 1)
                for q_d in result.quasi_dists:
                    quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                    self.assertDictEqual(quasi_dist, {0: 1.0})

            with self.subTest("two circuits"):
                result = sampler(circuit_indices=[2, 4], shots=1000)
                self.assertEqual(len(result.metadata), 2)
                self.assertEqual(result.metadata[0]["shots"], 1000)
                self.assertEqual(result.metadata[1]["shots"], 1000)
                self.assertEqual(len(result.quasi_dists), 2)
                for q_d in result.quasi_dists:
                    quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                    self.assertDictEqual(quasi_dist, {0: 1.0})
