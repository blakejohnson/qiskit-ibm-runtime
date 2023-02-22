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

"""Unit tests for noise model."""

import unittest
from unittest.mock import MagicMock

from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import FakeGuadalupe
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel

from programs.sampler import main as sampler_main
from programs.estimator import main as estimator_main


class TestNoiseModel(unittest.TestCase):
    """Test noise model."""

    def test_sampler_noise_model(self):
        """test noise model with sampler."""
        backend = Aer.get_backend("aer_simulator")
        fake_backend = FakeGuadalupe()
        noise_model = NoiseModel.from_backend(fake_backend)
        seed_simulator = 42
        shots = 100

        circ = QuantumCircuit(1, 1)
        circ.x(0)
        circ.measure(0, 0)
        aer_result = (
            backend.run(circ, noise_model=noise_model, seed_simulator=seed_simulator, shots=shots)
            .result()
            .get_counts()
        )

        sampler_result = sampler_main(
            backend=backend,
            user_messenger=MagicMock(),
            circuits=circ,
            circuit_indices=[0],
            run_options={
                "noise_model": noise_model,
                "seed_simulator": seed_simulator,
                "shots": shots,
            },
        )

        quasi = sampler_result["quasi_dists"][0]
        self.assertEqual(quasi["0"] * shots, aer_result["0"])
        self.assertEqual(quasi["1"] * shots, aer_result["1"])

    def test_sampler_m3(self):
        """test noise model with sampler and m3."""
        backend = Aer.get_backend("aer_simulator")
        fake_backend = FakeGuadalupe()
        noise_model = NoiseModel.from_backend(fake_backend)
        seed_simulator = 42
        shots = 100

        num_qubits = min(10, fake_backend.configuration().num_qubits)
        circ = QuantumCircuit(num_qubits, num_qubits)
        circ.x(range(0, num_qubits))
        circ.h(range(0, num_qubits))

        for idx in range(num_qubits // 2, 0, -1):
            circ.ch(idx, idx - 1)
        for idx in range(num_qubits // 2, num_qubits - 1):
            circ.ch(idx, idx + 1)
        circ.measure_all(add_bits=False)

        sampler_result = sampler_main(
            backend=backend,
            user_messenger=MagicMock(),
            circuits=circ,
            circuit_indices=[0],
            run_options={
                "noise_model": noise_model,
                "seed_simulator": seed_simulator,
                "shots": shots,
            },
            resilience_settings={"level": 1},
            transpilation_settings={
                "coupling_map": fake_backend.configuration().coupling_map,
                "basis_gates": fake_backend.configuration().basis_gates,
            },
        )

        self.assertGreater(sampler_result["metadata"][0]["readout_mitigation_overhead"], 2)

    def test_estimator_noise_model(self):
        """Test estimator with noise model."""
        circ = QuantumCircuit(2)
        circ.x(range(2))
        obs = SparsePauliOp.from_list([("IZ", 1)])

        fake_backend = FakeGuadalupe()
        noise_model = NoiseModel.from_backend(fake_backend)
        seed_simulator = 42
        seed_mitigation = 42
        shots = 100
        backend = Aer.get_backend("aer_simulator")

        result = estimator_main(
            backend=backend,
            user_messenger=MagicMock(),
            circuits=circ,
            circuit_indices=[0],
            observables=obs,
            observable_indices=[0],
            run_options={
                "noise_model": noise_model,
                "seed_simulator": seed_simulator,
                "shots": shots,
            },
            transpilation_settings={
                "coupling_map": fake_backend.configuration().coupling_map,
                "basis_gates": fake_backend.configuration().basis_gates,
            },
            resilience_settings={
                "pauli_twirled_mitigation": {
                    "seed_mitigation": seed_mitigation,
                    "seed_simulator": seed_simulator,
                },
            },
        )

        self.assertGreater(result["metadata"][0]["variance"], 0)
