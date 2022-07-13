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
import unittest

from ddt import ddt
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.primitives import SamplerResult
from qiskit.providers.fake_provider import FakeBogota

from programs.sampler import Sampler, main


@ddt
class TestSampler(unittest.TestCase):
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

    def _compare_probs(self, probabilities, target):
        if not isinstance(target, list):
            target = [target]
        self.assertEqual(len(probabilities), len(target))
        for prob, targ in zip(probabilities, target):
            for key, t_val in targ.items():
                if key in prob:
                    self.assertAlmostEqual(prob[key], t_val, places=1)
                else:
                    self.assertAlmostEqual(t_val, 0, places=1)

    @combine(indices=[[0], [1], [0, 1]], shots=[1000, 2000])
    def test_sample(self, indices, shots):
        """test to sample"""
        backend = Aer.get_backend("aer_simulator")
        circuits, target = self._generate_circuits_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=circuits, backend=backend) as sampler:
                result = sampler(
                    circuits=circuits,
                    shots=shots,
                    **self._run_config,
                )
                self.assertEqual(result.metadata[0]["shots"], shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=circuits, backend=backend)
            result = sampler(
                circuits=circuits,
                shots=shots,
                **self._run_config,
            )
            self.assertEqual(result.metadata[0]["shots"], shots)
            self._compare_probs(result.quasi_dists, target)

    @combine(
        indices=[[0], [1], [0, 1]],
        shots=[1000, 2000],
    )
    def test_sample_pqc(self, indices, shots):
        """test to sample a parametrized circuit"""
        backend = Aer.get_backend("aer_simulator")
        params, target = self._generate_params_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=self._pqc, backend=backend) as sampler:
                sampler.set_run_options(shots=shots, **self._run_config)
                result: SamplerResult = sampler(
                    circuits=[0] * len(indices), parameter_values=params
                )
                self.assertEqual(result.metadata[0]["shots"], shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=self._pqc, backend=backend)
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler(circuits=[0] * len(indices), parameter_values=params)
            self.assertEqual(result.metadata[0]["shots"], shots)
            self._compare_probs(result.quasi_dists, target)

    @combine(
        indices=[[0], [1], [0, 1]],
        shots=[1000, 2000],
    )
    def test_sample_with_ndarray(self, indices, shots):
        """test to sample a parametrized circuit"""
        backend = Aer.get_backend("aer_simulator")
        params, target = self._generate_params_target(indices)
        params = np.asarray(params)
        with self.subTest("with-guard"):
            with Sampler(circuits=self._pqc, backend=backend) as sampler:
                sampler.set_run_options(shots=shots, **self._run_config)
                result: SamplerResult = sampler(
                    circuits=[0] * len(indices), parameter_values=params
                )
                self.assertEqual(result.metadata[0]["shots"], shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=self._pqc, backend=backend)
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler(circuits=[0] * len(indices), parameter_values=params)
            self.assertEqual(result.metadata[0]["shots"], shots)
            self._compare_probs(result.quasi_dists, target)

    @combine(
        indices=[[0, 0], [0, 1], [1, 1]],
        shots=[1000, 2000],
    )
    def test_sample_two_pqcs(self, indices, shots):
        """test to sample two parametrized circuits"""
        backend = Aer.get_backend("aer_simulator")
        circs = [self._pqc, self._pqc]
        params, target = self._generate_params_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=circs, backend=backend) as sampler:
                sampler.set_run_options(shots=shots, **self._run_config)
                result: SamplerResult = sampler(
                    circuits=[0] * len(indices), parameter_values=params
                )
                self.assertEqual(result.metadata[0]["shots"], shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=circs, backend=backend)
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler(circuits=[0] * len(indices), parameter_values=params)
            self.assertEqual(result.metadata[0]["shots"], shots)
            self._compare_probs(result.quasi_dists, target)

    def test_call_validation(self):
        """Test for validations"""
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = RealAmplitudes(num_qubits=1, reps=1)
        qc2.measure_all()

        with Sampler(
            Aer.get_backend("aer_simulator"),
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
        num_qubits = 5
        qc1 = QuantumCircuit(num_qubits, num_qubits - 1)
        qc1.measure(range(num_qubits - 1), range(num_qubits - 1))
        with Sampler(backend=Aer.get_backend("aer_simulator"), circuits=[qc1] * 10) as sampler:
            with self.subTest("one circuit"):
                result = sampler(circuits=[0], shots=1000)
                self.assertEqual(len(result.metadata), 1)
                self.assertEqual(result.metadata[0]["shots"], 1000)
                self.assertEqual(len(result.quasi_dists), 1)
                for q_d in result.quasi_dists:
                    quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                    self.assertDictEqual(quasi_dist, {0: 1.0})

            with self.subTest("two circuits"):
                result = sampler(circuits=[2, 4], shots=1000)
                self.assertEqual(len(result.metadata), 2)
                self.assertEqual(result.metadata[0]["shots"], 1000)
                self.assertEqual(result.metadata[1]["shots"], 1000)
                self.assertEqual(len(result.quasi_dists), 2)
                for q_d in result.quasi_dists:
                    quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                    self.assertDictEqual(quasi_dist, {0: 1.0})


@ddt
class TestSamplerMain(unittest.TestCase):
    """Test Sampler main"""

    def setUp(self):
        super().setUp()
        hadamard = QuantumCircuit(1, 1)
        hadamard.h(0)
        hadamard.measure(0, 0)
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure([0, 1], [0, 1])
        hadamard2 = QuantumCircuit(2, 2)
        hadamard2.h(0)
        hadamard2.x(1)
        hadamard2.measure([0, 1], [1, 0])
        self._circuits = [hadamard, bell, hadamard2]
        self._targets = [
            {"0": 0.5, "1": 0.5},
            {"00": 0.5, "11": 0.5, "01": 0, "10": 0},
            {"01": 0.5, "11": 0.5, "00": 0, "10": 0},
        ]
        self._pqc = QuantumCircuit(2, 2)
        self._pqc.compose(RealAmplitudes(num_qubits=2, reps=2), inplace=True)
        self._pqc.measure(0, 0)
        self._pqc.measure(1, 1)
        self._pqc_params = [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
        self._pqc_targets = [
            {"00": 1},
            {"00": 0.0148, "01": 0.3449, "10": 0.0531, "11": 0.5872},
        ]

    def _compare_probs(self, probabilities, target):
        if not isinstance(target, list):
            target = [target]
        self.assertEqual(len(probabilities), len(target))
        for prob, targ in zip(probabilities, target):
            for key, t_val in targ.items():
                if key in prob:
                    self.assertAlmostEqual(prob[key], t_val, places=1)
                else:
                    self.assertAlmostEqual(t_val, 0, places=1)

    def test_smaller_bitstring_length(self):
        """Test for smaller bitstring length than qubit size"""
        num_qubits = 4
        qc1 = QuantumCircuit(num_qubits, num_qubits - 1)
        qc1.x(num_qubits - 1)
        qc1.h(range(num_qubits))
        qc1.cx(range(num_qubits - 1), num_qubits - 1)
        qc1.h(range(num_qubits - 1))
        qc1.barrier()
        qc1.measure(range(num_qubits - 1), range(num_qubits - 1))

        qc2 = QuantumCircuit(num_qubits, num_qubits - 1)
        qc2.x(num_qubits - 1)
        qc2.h(range(num_qubits))
        qc2.h(range(num_qubits - 1))
        qc2.barrier()
        qc2.measure(range(num_qubits - 1), range(num_qubits - 1))

        qc3 = QuantumCircuit(num_qubits, num_qubits - 1)
        qc3.h(range(num_qubits))
        qc3.measure(0, 1)

        backend = Aer.get_backend("aer_simulator")
        results = main(
            backend=backend,
            user_messenger=None,
            circuits=[qc1, qc2, qc3],
            circuit_indices=[0, 1, 2],
            run_options={"shots": 1000, "seed_simulator": 123},
        )
        self._compare_probs(
            results["quasi_dists"], [{"111": 1.0}, {"000": 1.0}, {"000": 0.5, "010": 0.5}]
        )

    def test_sampler(self):
        """test sampler"""
        backend = Aer.get_backend("aer_simulator")
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=self._circuits,
            circuit_indices=[0, 1, 2],
            run_options={"shots": 1000, "seed_simulator": 123},
            transpilation_settings={"seed_transpiler": 15},
        )
        self._compare_probs(result["quasi_dists"], self._targets)

    def test_sampler_pqc(self):
        """test sampler with a parametrized circuit"""
        backend = Aer.get_backend("aer_simulator")
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[self._pqc],
            circuit_indices=[0, 0],
            parameter_values=self._pqc_params,
            run_options={"shots": 1000, "seed_simulator": 123},
            transpilation_settings={"seed_transpiler": 15},
        )
        self._compare_probs(result["quasi_dists"], self._pqc_targets)

    @combine(noise=[True, False], shots=[10000, 20000])
    def test_sampler_with_m3(self, noise, shots):
        """test sampler with M3"""
        backend = FakeBogota() if noise else Aer.get_backend("aer_simulator")
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=self._circuits,
            circuit_indices=[0, 1, 2],
            run_options={"shots": shots, "seed_simulator": 123},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={"level": 1},
        )
        self._compare_probs(result["quasi_dists"], self._targets)

    @combine(noise=[True, False], shots=[10000, 20000])
    def test_sampler_pqc_m3(self, noise, shots):
        """test sampler with a parametrized circuit and M3"""
        backend = FakeBogota() if noise else Aer.get_backend("aer_simulator")
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[self._pqc],
            circuit_indices=[0, 0],
            parameter_values=self._pqc_params,
            run_options={"shots": shots, "seed_simulator": 123},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={"level": 1},
        )
        self._compare_probs(result["quasi_dists"], self._pqc_targets)

    @combine(noise=[True, False], shots=[10000, 20000])
    def test_sampler_pqc_m3_2(self, noise, shots):
        """test sampler with a parametrized circuit and M3 (2)"""
        # Note: Bogota has a linear coupling map
        backend = FakeBogota() if noise else Aer.get_backend("aer_simulator")
        num_qubits = 3
        circ = QuantumCircuit(num_qubits)
        param = Parameter("x")
        circ.ry(param, 0)
        for i in range(num_qubits - 1):
            circ.cx(i, i + 1)
        circ.measure_all()
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[circ],
            circuit_indices=[0, 0, 0],
            parameter_values=[[0], [np.pi / 2], [np.pi]],
            run_options={"shots": shots, "seed_simulator": 123},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={"level": 1},
        )
        zeros = "0" * num_qubits
        ones = "1" * num_qubits
        targets = [{zeros: 1.0}, {zeros: 0.5, ones: 0.5}, {ones: 1.0}]
        self._compare_probs(result["quasi_dists"], targets)