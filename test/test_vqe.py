from qiskit import IBMQ
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.opflow import X, Z, I
from qiskit_nature.runtime import VQEProgram

import numpy as np
from unittest import TestCase
from decorator import get_provider_and_backend


class TestVQE(TestCase):
    """Test VQE."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        cls.provider = provider
        cls.backend_name = backend_name
        cls.backend = cls.provider.get_backend(backend_name)
        
    def setUp(self) -> None:
        """Test case setup."""
        spin_coupling = (Z ^ Z ^ I) + (I ^ Z ^ Z)
        transverse_field = (X ^ I ^ I) + (I ^ X ^ I) + (I ^ I ^ X)
        hamiltonian = -0.5 * (spin_coupling + 0.5 * transverse_field)
        self.hamiltonian = hamiltonian
        
    def test_vqe_direct(self):
        """Test vqe script."""
        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(self.hamiltonian)
        print("Exact result:", reference.eigenvalue)
        ansatz = EfficientSU2(3, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        inputs = {
            "operator": self.hamiltonian,
            "ansatz": ansatz,
            "initial_point": initial_point,
            "optimizer": optimizer
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(
            program_id="vqe",
            inputs=inputs,
            options=options
        )

        result = job.result()
        print("Runtime:", result["eigenvalue"])
        if self.backend.configuration().simulator:
            self.assertTrue(abs(result["eigenvalue"] - reference.eigenvalue) <= 1)

    def test_nature_program(self):
        """Test vqe nature program."""
        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(self.hamiltonian)
        print("Exact result:", reference.eigenvalue)
        ansatz = EfficientSU2(3, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        vqe = VQEProgram(
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            provider=self.provider,
            backend=self.backend,
            store_intermediate=True,
        )
        result = vqe.compute_minimum_eigenvalue(self.hamiltonian)
        print("VQE program result:", result.eigenvalue)
        if self.backend.configuration().simulator:
            self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 1)

    def test_optimization_program(self):
        """Test vqe optimization program."""
        self.hamiltonian = (Z ^ Z ^ I ^ I) + (I ^ Z ^ Z ^ I) + (Z ^ I ^ I ^ Z)

        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(self.hamiltonian)
        print("Exact result:", reference.eigenvalue)
        ansatz = RealAmplitudes(4, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        vqe = VQEProgram(
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            provider=self.provider,
            backend=self.backend,
        )
        result = vqe.compute_minimum_eigenvalue(self.hamiltonian)
        print("VQE program result:", result.eigenvalue)
        if self.backend.configuration().simulator:
            self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 1)
