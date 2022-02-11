from unittest import skip

from qiskit import Aer
from qiskit.algorithms import NumPyMinimumEigensolver, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import Z, I
from qiskit_optimization.runtime import QAOAProgram

import numpy as np

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase
from .utils import find_program_id


@skip("Skip until decompose and qpy issues are fixed")
class TestQAOA(BaseTestCase):
    """Test QAOA."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend = cls.provider.get_backend(backend_name)
        # Use callback if on real device to avoid CI timeout
        cls.callback_func = (
            None if cls.backend.configuration().simulator else cls.simple_callback
        )

    def test_program(self):
        """Test qaqo program."""
        hamiltonian = (Z ^ Z ^ I ^ I) + (I ^ Z ^ Z ^ I) + (Z ^ I ^ I ^ Z)
        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
        self.log.info("Exact: %s", reference.eigenvalue)

        reps = 2
        initial_point = np.random.random(2 * reps)
        optimizer = SPSA(maxiter=300, learning_rate=0.01, perturbation=0.01)

        simulator = Aer.get_backend("qasm_simulator")
        local_qaoa = QAOA(
            optimizer=optimizer,
            reps=reps,
            initial_point=initial_point,
            quantum_instance=simulator,
        )
        local_result = local_qaoa.compute_minimum_eigenvalue(hamiltonian)
        self.log.info("Local simulator: %s", local_result.eigenvalue)

        qaoa = QAOAProgram(
            optimizer=optimizer,
            reps=reps,
            initial_point=initial_point,
            provider=self.provider,
            backend=self.backend,
            callback=self.callback_func,
        )
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        self.log.info("Runtime: %s", result.eigenvalue)
        if self.backend.configuration().simulator:
            self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 2)


class TestQAOARuntime(BaseTestCase):
    """Test the runtime program in the runtime/ directory."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend = cls.provider.get_backend(backend_name)
        # Use callback if on real device to avoid CI timeout
        cls.callback_func = (
            None if cls.backend.configuration().simulator else cls.simple_callback
        )

    def test_qaoa_runtime(self):
        """Test that the program runs."""

        hamiltonian = (Z ^ Z ^ I ^ I) + (I ^ Z ^ Z ^ I) + (Z ^ I ^ I ^ Z)

        maxiter = 40
        optimizer = SPSA(maxiter, learning_rate=0.01, perturbation=0.1)
        reps = 2
        initial_point = np.random.random(2 * reps)

        counter = {"count": 0}

        # pylint: disable=unused-argument
        def callback(*args):
            counter["count"] += 1

        # Runtime inputs
        runtime_inputs = {
            "operator": hamiltonian,
            "reps": reps,
            "optimizer": optimizer,
            "initial_point": initial_point,
            "shots": 1024,
            "use_pulse_efficient": False,
            "use_swap_strategies": False,
            "aux_operators": [Z ^ I ^ I ^ I, I ^ I ^ I ^ Z],
        }

        job = self.provider.runtime.run(
            program_id=find_program_id(self.provider.runtime, "qaoa"),
            options={"backend_name": self.backend.name()},
            inputs=runtime_inputs,
            callback=callback,
        )

        result = job.result()

        # assert the callback has been called
        self.assertTrue(counter["count"] > 0)

        # check the types and some basic properties of the result
        self.assertIsInstance(result["optimizer_time"], float)
        self.assertIsInstance(result["optimal_value"], float)
        self.assertTrue(len(result["optimal_point"]) == 2 * reps)
        self.assertIsNone(
            result["optimal_parameters"]
        )  # ParameterVectors not supported; this is None
        self.assertIsInstance(result["cost_function_evals"], int)
        self.assertIsInstance(result["eigenstate"], dict)
        self.assertIsInstance(result["eigenvalue"], complex)
        self.assertTrue(len(result["aux_operator_eigenvalues"]) == 2)

        # check history
        history = result["optimizer_history"]
        for key in ["nfevs", "params", "energy", "std"]:
            self.assertEqual(len(history[key]), counter["count"] - 1)

        # check inputs
        for key in [
            "operator",
            "reps",
            "optimizer",
            "initial_point",
            "shots",
            "use_pulse_efficient",
            "use_swap_strategies",
            "aux_operators",
        ]:
            self.assertTrue(key in result["inputs"])
