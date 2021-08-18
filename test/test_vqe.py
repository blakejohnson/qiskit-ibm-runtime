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

    @classmethod
    def setUpClass(cls):
        """Class setup."""
        hgp = os.getenv("QISKIT_IBM_HGP_STAGING", None) \
            if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True" \
            else os.getenv("QISKIT_IBM_HGP", None)
        if not hgp:
            raise SkipTest("Requires ibm provider.")
        hgp = hgp.split(",")
        backend_name = os.getenv("QISKIT_IBM_DEVICE_STAGING", None) \
            if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True" \
            else os.getenv("QISKIT_IBM_DEVICE", None)          
        if not backend_name:
            raise SkipTest("Runtime device not specified")
        cls.backend_name = backend_name
        if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True":
            print("Using staging creds")
            os.environ["QE_TOKEN"] = os.getenv("QE_TOKEN_STAGING", "")
            os.environ["QE_URL"] = os.getenv("QE_URL_STAGING", "")
        IBMQ.enable_account(os.getenv("QE_TOKEN", ""), os.getenv("QE_URL", ""))
        cls.provider = IBMQ.get_provider(hub=hgp[0], group=hgp[1], project=hgp[2])
        cls.backend = cls.provider.get_backend(backend_name)
        
    def setUp(self) -> None:
        """Test case setup."""
        spin_coupling = (Z ^ Z ^ I) + (I ^ Z ^ Z)
        transverse_field = (X ^ I ^ I) + (I ^ X ^ I) + (I ^ I ^ X)
        hamiltonian = -0.5 * (spin_coupling + 0.5 * transverse_field)
        self.hamiltonian = hamiltonian
        
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

        options = {"backend_name": self.__class__.backend_name}

        job = self.__class__.provider.runtime.run(
            program_id="vqe",
            inputs=inputs,
            options=options
        )

        result = job.result()
        print("Runtime:", result["eigenvalue"])
        if self.__class__.backend.configuration().simulator:
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
            provider=self.__class__.provider,
            backend=self.__class__.backend,
            store_intermediate=True,
        )
        result = vqe.compute_minimum_eigenvalue(self.hamiltonian)
        print("VQE program result:", result.eigenvalue)
        if self.__class__.backend.configuration().simulator:
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
            provider=self.__class__.provider,
            backend=self.__class__.backend,
        )
        result = vqe.compute_minimum_eigenvalue(self.hamiltonian)
        print("VQE program result:", result.eigenvalue)
        if self.__class__.backend.configuration().simulator:
            self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 1)
