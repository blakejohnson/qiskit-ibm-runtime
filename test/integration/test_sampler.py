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

"""Tests sampler."""

from qiskit import QuantumCircuit
from qiskit.providers.jobstatus import JobStatus

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class TestSampler(BaseTestCase):
    """Test sampler."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name

    def setUp(self) -> None:
        """Test case setup."""
        qc1 = QuantumCircuit(5)
        qc1.h(2)
        qc1.cx(2, 1)
        qc1.cx(1, 0)
        qc1.cx(2, 3)
        qc1.cx(3, 4)
        qc1.measure_all()
        self.qc1 = qc1
        self.qc1_id = str(id(qc1))

    # TODO: remove this test when non-flexible interface is no longer supported in provider
    def test_sampler_circuit_indices(self):
        """Test sampler program with circuit indices."""
        program_inputs = {"circuits": self.qc1, "circuit_indices": [0]}

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(
            program_id="sampler",
            options=options,
            inputs=program_inputs,
        )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())

    def test_sampler(self):
        """Test sampler program with circuit ids."""
        program_inputs = {"circuits": {self.qc1_id: self.qc1}, "circuit_ids": [self.qc1_id]}

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(
            program_id="sampler",
            options=options,
            inputs=program_inputs,
        )
        self.log.debug("Job ID: %s", job.job_id())
        job.wait_for_final_state()
        self.assertEqual(job.status(), JobStatus.DONE, job.error_message())
        # TODO: after switching these tests to use qiskit-ibm-runtime we can add
        # start_session=True to the first run call above and make another run call with
        # circuits={} and circuit_ids=[self.qc1_id], so we can test caching circuits in redis
