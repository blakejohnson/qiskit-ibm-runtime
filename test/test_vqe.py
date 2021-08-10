from unittest import TestCase
import numpy as np
from qiskit import IBMQ
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, Z, I
from qiskit_nature.runtime import VQEProgram
from qiskit.circuit.library import RealAmplitudes

class TestVQE(TestCase):
    """Test VQE."""
    def test_script(self):
        """Test VQE script."""
        spin_coupling = (Z ^ Z ^ I) + (I ^ Z ^ Z)
        transverse_field = (X ^ I ^ I) + (I ^ X ^ I) + (I ^ I ^ X)
        hamiltonian = -0.5 * (spin_coupling + 0.5 * transverse_field)

        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
        print("Exact result:", reference.eigenvalue)
        ansatz = EfficientSU2(3, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        inputs = {
            "operator": hamiltonian,
            "ansatz": ansatz,
            "initial_point": initial_point,
            "optimizer": optimizer
        }

        options = {"backend_name": "ibmq_qasm_simulator"}

        IBMQ.load_account()
        provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")

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
        spin_coupling = (Z ^ Z ^ I) + (I ^ Z ^ Z)
        transverse_field = (X ^ I ^ I) + (I ^ X ^ I) + (I ^ I ^ X)
        hamiltonian = -0.5 * (spin_coupling + 0.5 * transverse_field)

        ansatz = EfficientSU2(3, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        IBMQ.load_account()
        provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")
        backend = provider.get_backend("ibmq_qasm_simulator")

        vqe = VQEProgram(
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            provider=provider,
            backend=backend,
            store_intermediate=True,
        )
        result = vqe.compute_minimum_eigenvalue(hamiltonian)

        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
        print("Exact result:", reference.eigenvalue)
        print("VQE Prgram result:", result.eigenvalue)
        self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 1)

    def test_optimization_program(self):
        """Test optimization program."""
        hamiltonian = (Z ^ Z ^ I ^ I) + (I ^ Z ^ Z ^ I) + (Z ^ I ^ I ^ Z)

        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
        print("Exact result:", reference.eigenvalue)
        ansatz = RealAmplitudes(4, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        IBMQ.load_account()
        provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")
        backend = provider.get_backend("ibmq_qasm_simulator")

        vqe = VQEProgram(
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            provider=provider,
            backend=backend,
        )
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        print("VQE program result:", result.eigenvalue)
        self.assertTrue(abs(result.eigenvalue - reference.eigenvalue) <= 1)