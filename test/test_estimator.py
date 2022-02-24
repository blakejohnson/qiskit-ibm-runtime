from unittest import skip

from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.providers.jobstatus import JobStatus

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase
from .utils import find_program_id


class TestEstimator(BaseTestCase):
    """Test hello_world."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        cls.service = provider.runtime
        cls.backend_name = backend_name

    @skip("Skip until all setup")
    def test_estimator(self):
        """Test estimator."""
        program_id = find_program_id(self.service, "estimator")
        observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        ansatz = RealAmplitudes(num_qubits=2, reps=2)
        parameters = [0, 1, 1, 2, 3, 5]
        run_options = {"shots": 1000}

        program_inputs = {
            "circuits": ansatz,
            "observables": observable,
            "parameters": parameters,
            "run_options": run_options
        }

        options = {"backend_name": self.backend_name}

        job = self.service.run(program_id=program_id,
                               options=options,
                               inputs=program_inputs,
                               )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
