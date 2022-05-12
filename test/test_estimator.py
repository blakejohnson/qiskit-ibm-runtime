import numpy as np

from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.ibmq.runtime.exceptions import RuntimeJobFailureError

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class TestEstimator(BaseTestCase):
    """Test hello_world."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        cls.service = provider.runtime
        cls.backend_name = backend_name
        cls.options = {"backend_name": backend_name}
        cls.program_id = "estimator"
        cls.observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        cls.ansatz = RealAmplitudes(num_qubits=2, reps=2)

    def test_run_without_params(self):
        """Test estimator without parameters."""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        inputs = {
            "circuits": circuit,
            "observables": self.observable,
            "circuit_indices": [0],
            "observable_indices": [0],
        }
        self._run_job_and_verify(inputs)

    def test_run_single_params(self):
        """Test estimator with a single parameter."""
        inputs = {
            "circuits": self.ansatz,
            "observables": self.observable,
            "parameter_values": [[0, 1, 1, 2, 3, 5]],
            "circuit_indices": [0],
            "observable_indices": [0],
        }

        self._run_job_and_verify(inputs)

    def test_run_multi_params(self):
        """Test estimator with multiple parameters."""
        inputs = {
            "circuits": self.ansatz,
            "observables": self.observable,
            "parameter_values": [[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]],
            "circuit_indices": [0, 0],
            "observable_indices": [0, 0],
        }
        self._run_job_and_verify(inputs)

    def test_numpy_params(self):
        """Test with parameters in numpy array."""
        param = np.random.rand(1, self.ansatz.num_parameters)
        inputs = {
            "circuits": [self.ansatz],
            "observables": self.observable,
            "circuit_indices": [0],
            "observable_indices": [0],
            "parameter_values": param,
        }
        self._run_job_and_verify(inputs)

    def test_bad_circuit_indices(self):
        """Test passing bad circuit indices."""
        inputs = {
            "circuits": self.ansatz,
            "observables": self.observable,
            "parameter_values": [[0, 1, 1, 2, 3, 5]],
            "circuit_indices": [0, 1],
            "observable_indices": [0],
        }
        job = self.service.run(
            program_id=self.program_id,
            options=self.options,
            inputs=inputs,
        )
        self.log.debug("Job ID: %s", job.job_id())
        with self.assertRaises(RuntimeJobFailureError):
            job.result()
        self.assertEqual(job.status(), JobStatus.ERROR, job.error_message())

    def _run_job_and_verify(self, inputs):
        """Run a job."""
        job = self.service.run(
            program_id=self.program_id,
            options=self.options,
            inputs=inputs,
        )
        self.log.debug("Job ID: %s", job.job_id())
        job.result()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
