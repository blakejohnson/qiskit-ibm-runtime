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
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator as RefEstimator
from qiskit.providers.fake_provider import FakeNairobi, FakeVigo, FakeMontreal
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

from programs.estimator import main, EstimatorConstant
from .mock.mock_user_messenger import MockUserMessenger


@ddt
class TestPEC(unittest.TestCase):
    """Test PEC functionality."""

    def test_result(self):
        """Test for main without noise with single parameter set"""
        resilience_level = EstimatorConstant.PEC_RESILIENCE_LEVEL
        backend = AerSimulator.from_backend(FakeMontreal(), noise_model=None)
        ansatz = RealAmplitudes(num_qubits=2, reps=2)
        observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        shots = 20
        circuits = [0]
        observables = [0]
        params = [[0, 1, 1, 2, 3, 5]]
        with RefEstimator([ansatz], [observable]) as estimator:
            target = estimator(circuits, observables, params).values
        pec_cache = {}
        trex_cache = {}
        result = main(
            backend=backend,
            user_messenger=None,
            circuits=[ansatz],
            observables=[observable],
            circuit_indices=circuits,
            observable_indices=observables,
            parameter_values=params,
            run_options={"shots": shots, "seed_simulator": 15},
            transpilation_settings={"seed_transpiler": 15},
            resilience_settings={
                "level": resilience_level,
                "pec_cache": pec_cache,
                "trex_cache": trex_cache,
            },
        )
        np.testing.assert_allclose(result["values"], target, rtol=2e-2)
        for metadata in result["metadata"]:
            self.assertGreater(metadata["standard_error"], 0)
            self.assertEqual(
                metadata["shots"], shots * EstimatorConstant.PEC_DEFAULT_SHOTS_PER_SAMPLE
            )
            self.assertEqual(metadata["samples"], shots)
            self.assertEqual(metadata["sampling_overhead"], 1.0)

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

        pec_cache = {}
        trex_cache = {}
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
                "level": EstimatorConstant.PEC_RESILIENCE_LEVEL,
                "pec_cache": pec_cache,
                "trex_cache": trex_cache,
            },
        )

        self.assertDictEqual(user_messenger.get_msg(0), {"layers_detected": 2})
        self.assertDictEqual(user_messenger.get_msg(1), {"layers_to_learn": 2})

    @unittest.skip("We are not letting the user pass number of samples for now")
    def test_num_samples(self):
        """Test option for specifying the number of samples to use"""

        backend = AerSimulator.from_backend(FakeVigo())
        user_messenger = MockUserMessenger()
        observable = SparsePauliOp("III")
        qubits = 3
        circuit = QuantumCircuit(qubits)
        for i in range(10):
            circuit.cx(i % qubits, int(i + qubits / 2) % qubits)

        pec_cache = {}
        trex_cache = {}
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
                "level": EstimatorConstant.PEC_RESILIENCE_LEVEL,
                "num_samples": 35,
                "pec_cache": pec_cache,
                "trex_cache": trex_cache,
            },
        )

        self.assertEqual(result["metadata"][0]["samples"], 35)

    def test_num_samples_default(self):
        """Test the default number of samples to use"""

        backend = AerSimulator.from_backend(FakeVigo(), noise_model=None)
        user_messenger = MockUserMessenger()
        observable = SparsePauliOp("III")
        qubits = 3
        circuit = QuantumCircuit(qubits)
        for i in range(10):
            circuit.cx(i % qubits, int(i + qubits / 2) % qubits)
        pec_cache = {}
        trex_cache = {}

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
                "level": EstimatorConstant.PEC_RESILIENCE_LEVEL,
                "pec_cache": pec_cache,
                "trex_cache": trex_cache,
            },
        )

        self.assertEqual(
            result["metadata"][0]["samples"], EstimatorConstant.PEC_DEFAULT_NUM_SAMPLES
        )

    def test_default_options_passed_to_pec(self):
        """Test default options passed to PEC estimator"""
        # very noisy model
        p_meas = 0.2
        p_gate1 = 0.1

        # # QuantumError objects
        error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
        error_gate1 = pauli_error([("X", p_gate1), ("I", 1 - p_gate1)])
        error_gate2 = error_gate1.tensor(error_gate1)

        # Add errors to noise model
        noise_bit_flip = NoiseModel(basis_gates=["rz", "sx", "cx"])
        noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
        noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["sx"])
        noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

        # NOTE: We should be able to set coupling map on simulator here
        # but that is currently broken with the stratify function which
        # requires it as a transpile option
        backend = AerSimulator(noise_model=noise_bit_flip)
        user_messenger = MockUserMessenger()

        qubits = 5
        observable = SparsePauliOp("Z" * qubits)
        circuit = QuantumCircuit(qubits)
        for i in range(10):
            circuit.cx(i % qubits, (i + 1) % qubits)

        pec_cache = {}
        trex_cache = {}
        shots = 10
        result = main(
            backend=backend,
            user_messenger=user_messenger,
            circuits=[circuit],
            observables=[observable],
            circuit_indices=[0],
            observable_indices=[0],
            parameter_values=[[]],
            run_options={"seed_simulator": 15, "shots": shots},
            transpilation_settings={
                "seed_transpiler": 15,
                "coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
            },
            resilience_settings={
                "level": EstimatorConstant.PEC_RESILIENCE_LEVEL,
                "pec_cache": pec_cache,
                "trex_cache": trex_cache,
            },
        )

        metadata = result["metadata"][0]
        result_samples = metadata.get("samples")
        result_shots = metadata.get("shots")
        self.assertEqual(
            result_samples,
            shots * EstimatorConstant.PEC_DEFAULT_MAX_SAMPLING_OVERHEAD,
            msg=f"Incorrect samples value in metadata: {metadata}",
        )
        self.assertEqual(
            result_shots,
            result_samples * EstimatorConstant.PEC_DEFAULT_SHOTS_PER_SAMPLE,
            msg=f"Incorrect shots value in metadata: {metadata}",
        )
        # assert on the default max learning layers
        self.assertGreater(
            user_messenger._messages[0]["unique_layers_detected"],
            EstimatorConstant.PEC_DEFAULT_MAX_LEARNING_LAYERS,
            msg="""Invalid test: the number of unique layers detected is not greater than the default
                   maximum. Please update the test to meet this requirement.""",
        )
        non_ones_count = sum(
            1
            for overhead in user_messenger._messages[-1]["sampling_overhead_by_layer"]
            if overhead != 1
        )
        self.assertEqual(
            non_ones_count,
            EstimatorConstant.PEC_DEFAULT_MAX_LEARNING_LAYERS,
            msg=f"Number of learned layers {non_ones_count} is greater than the default",
        )

    @unittest.skip("We are not letting the user pass the number of max layers for now")
    def test_max_learning_layers(self):
        """Test option for limiting the number of layers to learn"""
        backend = AerSimulator.from_backend(FakeNairobi())
        circuit = RealAmplitudes(num_qubits=5, reps=2, entanglement="full")
        num_parameters = circuit.num_parameters
        observable = SparsePauliOp("IIIII")
        user_messenger = MockUserMessenger()
        pec_cache = {}
        trex_cache = {}
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
                "level": EstimatorConstant.PEC_RESILIENCE_LEVEL,
                "max_learning_layers": 3,
                "pec_cache": pec_cache,
                "trex_cache": trex_cache,
            },
        )
        self.assertDictEqual(user_messenger.get_msg(0), {"layers_detected": 6})
        self.assertDictEqual(user_messenger.get_msg(1), {"layers_to_learn": 3})
