from qiskit.providers.ibmq import RunnerResult
from qiskit import IBMQ, QuantumCircuit
from unittest import TestCase

class TestCircuitRunner(TestCase):
    def test_circuit_runner(self):

        provider = IBMQ.load_account()
        N = 6
        qc = QuantumCircuit(N)
        qc.x(range(0, N))
        qc.h(range(0, N))

        program_inputs = {
            'circuits': qc,
            'shots': 2048,
            'optimization_level': 0,
            'initial_layout': [0,1,4,7,10,12],
            'measurement_error_mitigation': False
        }

        options = {'backend_name': "ibmq_qasm_simulator"}

        job = provider.runtime.run(program_id="circuit-runner",
                                    options=options,
                                    inputs=program_inputs,
                                    result_decoder=RunnerResult
                                    )
        result = job.result()
        print("CIRCUIT_RUNNER")
        print("Runtime:", result)
