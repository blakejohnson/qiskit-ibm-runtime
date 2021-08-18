from unittest import TestCase, SkipTest
import numpy as np
import os

from qiskit import IBMQ, Aer
from qiskit.algorithms import NumPyMinimumEigensolver, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import Z, I
from qiskit_optimization.runtime import QAOAProgram

class TestQAOA(TestCase):
    """Test QAOA."""

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
        if os.getenv("QISKIT_IBM_USE_STAGING_CREDENTIALS", "") == "True":
            print("Using staging creds")
            os.environ["QE_TOKEN"] = os.getenv("QE_TOKEN_STAGING", "")
            os.environ["QE_URL"] = os.getenv("QE_URL_STAGING", "")
        IBMQ.load_account()
        cls.provider = IBMQ.get_provider(hub=hgp[0], group=hgp[1], project=hgp[2])
        cls.backend = cls.provider.get_backend(backend_name)

    def test_program(self):
        """Test qaqo program."""
        hamiltonian = (Z ^ Z ^ I ^ I) + (I ^ Z ^ Z ^ I) + (Z ^ I ^ I ^ Z)
        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
        print("Exact:", reference.eigenvalue)

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
        print("Local simulator:", local_result.eigenvalue)

        qaoa = QAOAProgram(
            optimizer=optimizer,
            reps=reps,
            initial_point=initial_point,
            provider=self.__class__.provider,
            backend=self.__class__.backend,
        )
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        print("Runtime:", result.eigenvalue)
        if self.__class__.backend.configuration().simulator:
            self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 2)
        