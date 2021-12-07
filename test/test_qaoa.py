from unittest import skip

from qiskit import Aer
from qiskit.algorithms import NumPyMinimumEigensolver, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import Z, I
from qiskit_optimization.runtime import QAOAProgram

import numpy as np

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


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
        cls.callback_func = None if cls.backend.configuration().simulator else cls.simple_callback

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
            quantum_instance=simulator
        )
        local_result = local_qaoa.compute_minimum_eigenvalue(hamiltonian)
        self.log.info("Local simulator: %s", local_result.eigenvalue)

        qaoa = QAOAProgram(
            optimizer=optimizer,
            reps=reps,
            initial_point=initial_point,
            provider=self.provider,
            backend=self.backend,
            callback=self.callback_func
        )
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        self.log.info("Runtime: %s", result.eigenvalue)
        if self.backend.configuration().simulator:
            self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 2)
