from qiskit.providers import provider
from qiskit.providers.ibmq import RunnerResult
from qiskit import IBMQ, QuantumCircuit
from unittest import TestCase

class TestCircuitRunner(TestCase):
    """Test circuit_runner."""

    def setUp(self) -> None:
        """Test case setup."""
        self.provider = IBMQ.load_account()
        N = 6
        qc = QuantumCircuit(N)
        qc.x(range(0, N))
        qc.h(range(0, N))
        self.qc = qc
        
    def test_circuit_runner(self):
        """Test circuit_runner program."""
        program_inputs = {
            'circuits': self.qc,
            'shots': 2048,
            'optimization_level': 0,
            'initial_layout': [0,1,4,7,10,12],
            'measurement_error_mitigation': False
        }

        options = {'backend_name': "ibmq_qasm_simulator"}

        job = self.provider.runtime.run(program_id="circuit-runner",
                                    options=options,
                                    inputs=program_inputs,
                                    result_decoder=RunnerResult
                                    )
        result = job.result()
        print("Runtime:", result)
        expected_status = "JobStatus.DONE"
        self.assertEqual(str(job.status()), expected_status)

