from unittest import TestCase
import numpy as np
from qiskit import IBMQ, Aer
from qiskit.algorithms import NumPyMinimumEigensolver, QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import Z, I
from qiskit_optimization.runtime import QAOAProgram

class TestQAOA(TestCase):
    """Test QAOA."""
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

        IBMQ.load_account()
        provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")
        backend = provider.get_backend("ibmq_qasm_simulator")

        qaoa = QAOAProgram(
            optimizer=optimizer,
            reps=reps,
            initial_point=initial_point,
            provider=provider,
            backend=backend,
        )
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        print("Runtime:", result.eigenvalue)
        self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 2)
        