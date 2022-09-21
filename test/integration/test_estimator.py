# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test estimator."""

import numpy as np

from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.ibmq.runtime.exceptions import RuntimeJobFailureError

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


# TODO: remove this class when non-flexible interface is no longer supported in provider
class TestEstimatorCircuitIndices(BaseTestCase):
    """Test estimator with circuit indices."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ
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


class TestEstimatorCircuitIds(BaseTestCase):
    """Test estimator with circuit ids."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ
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
        cls.circuit_id = str(id(cls.ansatz))

    def test_run_without_params(self):
        """Test estimator without parameters."""
        circuit = self.ansatz.bind_parameters([0, 1, 1, 2, 3, 5])
        circuit_id = str(id(circuit))
        inputs = {
            "circuits": {circuit_id: circuit},
            "observables": self.observable,
            "circuit_ids": [circuit_id],
            "observable_indices": [0],
        }
        self._run_job_and_verify(inputs)

    def test_run_single_params(self):
        """Test estimator with a single parameter."""
        inputs = {
            "circuits": {self.circuit_id: self.ansatz},
            "observables": self.observable,
            "parameter_values": [[0, 1, 1, 2, 3, 5]],
            "circuit_ids": [self.circuit_id],
            "observable_indices": [0],
        }

        self._run_job_and_verify(inputs)

    def test_run_multi_params(self):
        """Test estimator with multiple parameters."""
        inputs = {
            "circuits": {self.circuit_id: self.ansatz},
            "observables": self.observable,
            "parameter_values": [[0, 1, 1, 2, 3, 5], [1, 1, 2, 3, 5, 8]],
            "circuit_ids": [self.circuit_id, self.circuit_id],
            "observable_indices": [0, 0],
        }
        self._run_job_and_verify(inputs)

    def test_numpy_params(self):
        """Test with parameters in numpy array."""
        param = np.random.rand(1, self.ansatz.num_parameters)
        inputs = {
            "circuits": {self.circuit_id: self.ansatz},
            "observables": self.observable,
            "circuit_ids": [self.circuit_id],
            "observable_indices": [0],
            "parameter_values": param,
        }
        self._run_job_and_verify(inputs)

    def _run_job_and_verify(self, inputs):
        """Run a job."""
        job = self.service.run(
            program_id=self.program_id,
            options=self.options,
            inputs=inputs,
        )
        # TODO: after switching these tests to use qiskit-ibm-runtime we can add
        # start_session=True to the first run call above and make another run call with
        # circuits={} and circuit_ids=[self.circuit_id], so we can test caching circuits in redis
        self.log.debug("Job ID: %s", job.job_id())
        job.result()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
