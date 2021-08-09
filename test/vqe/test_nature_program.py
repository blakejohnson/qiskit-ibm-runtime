import numpy as np

from qiskit import IBMQ
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, Z, I

from qiskit_nature.runtime import VQEProgram

print("--------QAQO nature program tests started---------")
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
print(result)
print("-------QAQO nature program tests completed--------")
