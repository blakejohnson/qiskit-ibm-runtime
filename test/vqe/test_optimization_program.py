import numpy as np

from qiskit import IBMQ
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import Z, I

from qiskit_optimization.runtime import VQEProgram

print("------VQE optimization program tests started------")
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
print(result)
print("-----VQE optimization program tests completed-----")
