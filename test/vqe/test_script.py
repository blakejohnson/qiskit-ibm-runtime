import numpy as np
from qiskit import IBMQ
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import X, Z, I

print("----------VQE script tests started----------")

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
print("----------VQE script tests completed----------")
