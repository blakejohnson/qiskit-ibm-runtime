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

import unittest

from ddt import ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.fake_provider import FakeNairobi, FakeVigo
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from programs.estimator_dev import main, Constant
from .mock.mock_user_messenger import MockUserMessenger


@ddt
class TestPEC(unittest.TestCase):
    """Test PEC functionality."""

    @unittest.skip("Skip until fixed")
    def test_user_messenger(self):
        """Test for communication with primitives"""

        backend = AerSimulator.from_backend(FakeVigo())
        user_messenger = MockUserMessenger()
        observable = SparsePauliOp("III")
        qubits = 3
        circuit = QuantumCircuit(qubits)
        for i in range(10):
            circuit.cx(i % qubits, int(i + qubits / 2) % qubits)

        main(
            backend=backend,
            user_messenger=user_messenger,
            circuits=[circuit],
            observables=[observable],
            circuit_indices=[0],
            observable_indices=[0],
            run_options={"seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "enable_pec": True,
                "level": Constant.PEC_RESILIENCE_LEVEL,
            },
        )

        self.assertDictEqual(user_messenger.get_msg(0), {"layers_detected": 2})
        self.assertDictEqual(user_messenger.get_msg(1), {"layers_to_learn": 2})

    def test_num_samples(self):
        """Test option for specifying the number of samples to use"""

        backend = AerSimulator.from_backend(FakeVigo())
        user_messenger = MockUserMessenger()
        observable = SparsePauliOp("III")
        qubits = 3
        circuit = QuantumCircuit(qubits)
        for i in range(10):
            circuit.cx(i % qubits, int(i + qubits / 2) % qubits)

        result = main(
            backend=backend,
            user_messenger=user_messenger,
            circuits=[circuit],
            observables=[observable],
            circuit_indices=[0],
            observable_indices=[0],
            run_options={"seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "enable_pec": True,
                "level": Constant.PEC_RESILIENCE_LEVEL,
                "num_samples": 35,
            },
        )

        self.assertEqual(result["metadata"][0]["samples"], 35)

    def test_num_samples_default(self):
        """Test the default number of samples to use"""

        backend = AerSimulator.from_backend(FakeVigo())
        user_messenger = MockUserMessenger()
        observable = SparsePauliOp("III")
        qubits = 3
        circuit = QuantumCircuit(qubits)
        for i in range(10):
            circuit.cx(i % qubits, int(i + qubits / 2) % qubits)

        result = main(
            backend=backend,
            user_messenger=user_messenger,
            circuits=[circuit],
            observables=[observable],
            circuit_indices=[0],
            observable_indices=[0],
            run_options={"seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "enable_pec": True,
                "level": Constant.PEC_RESILIENCE_LEVEL,
            },
        )

        self.assertEqual(result["metadata"][0]["samples"], Constant.PEC_DEFAULT_NUM_SAMPLES)

    def test_max_learning_layers(self):
        """Test option for limiting the number of layers to learn"""
        backend = AerSimulator.from_backend(FakeNairobi())
        circuit = RealAmplitudes(num_qubits=5, reps=2, entanglement="full")
        num_parameters = circuit.num_parameters
        observable = SparsePauliOp("IIIII")
        user_messenger = MockUserMessenger()
        main(
            backend=backend,
            user_messenger=user_messenger,
            circuits=[circuit],
            observables=[observable],
            circuit_indices=[0],
            observable_indices=[0],
            parameter_values=[[0] * num_parameters],
            run_options={"seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "enable_pec": True,
                "level": Constant.PEC_RESILIENCE_LEVEL,
                "max_learning_layers": 3,
            },
        )
        self.assertDictEqual(user_messenger.get_msg(0), {"layers_detected": 6})
        self.assertDictEqual(user_messenger.get_msg(1), {"layers_to_learn": 3})
