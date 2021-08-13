from qiskit import IBMQ
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.opflow import X, Z, I
from qiskit_nature.runtime import VQEProgram

from unittest import TestCase, SkipTest
import numpy as np
import os

class TestVQE(TestCase):
    """Test VQE."""

    def setUp(self) -> None:
        """Test case setup."""
        spin_coupling = (Z ^ Z ^ I) + (I ^ Z ^ Z)
        transverse_field = (X ^ I ^ I) + (I ^ X ^ I) + (I ^ I ^ X)
        hamiltonian = -0.5 * (spin_coupling + 0.5 * transverse_field)
        self.hamiltonian = hamiltonian
        hgp = os.getenv("QISKIT_IBM_HGP", None)
        if not hgp:
            raise SkipTest("Requires ibm provider.")
        self.hgp = hgp.split(",")
        backend_name = os.getenv("QISKIT_IBM_DEVICE", None)
        if not backend_name:
            raise SkipTest("Runtime device not specified")
        self.backend_name = backend_name

    def test_script(self):
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

        IBMQ.load_account()
        provider = IBMQ.get_provider(hub=self.hgp[0], group=self.hgp[1], project=self.hgp[2])

        job = provider.runtime.run(
            program_id="vqe",
            inputs=inputs,
            options=options
        )

        result = job.result()
        print("Runtime:", result["eigenvalue"])
        self.assertTrue(abs(result["eigenvalue"] - reference.eigenvalue) <= 1)

    def test_nature_program(self):
        """Test vqe nature program."""
        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(self.hamiltonian)
        print("Exact result:", reference.eigenvalue)
        ansatz = EfficientSU2(3, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        IBMQ.load_account()
        provider = IBMQ.get_provider(hub=self.hgp[0], group=self.hgp[1], project=self.hgp[2])
        backend = provider.get_backend(self.backend_name)

        vqe = VQEProgram(
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            provider=provider,
            backend=backend,
            store_intermediate=True,
        )
        result = vqe.compute_minimum_eigenvalue(self.hamiltonian)
        print("VQE program result:", result.eigenvalue)
        self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 1)

    def test_optimization_program(self):
        """Test vqe optimization program."""
        self.hamiltonian = (Z ^ Z ^ I ^ I) + (I ^ Z ^ Z ^ I) + (Z ^ I ^ I ^ Z)

        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(self.hamiltonian)
        print("Exact result:", reference.eigenvalue)
        ansatz = RealAmplitudes(4, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        IBMQ.load_account()
        provider = IBMQ.get_provider(hub=self.hgp[0], group=self.hgp[1], project=self.hgp[2])
        backend = provider.get_backend(self.backend_name)

        vqe = VQEProgram(
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            provider=provider,
            backend=backend,
        )
        result = vqe.compute_minimum_eigenvalue(self.hamiltonian)
        print("VQE program result:", result.eigenvalue)
        self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 1)
