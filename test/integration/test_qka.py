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

"""Test QKA program."""

import sys

import numpy as np
from qka.featuremaps import FeatureMap

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class TestQKA(BaseTestCase):
    """Test QKA program."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ
        """Class setup."""
        super().setUpClass()
        sys.path.insert(0, "..")
        cls.provider = provider
        cls.backend_name = backend_name

    def test_qka_random(self):
        """Test QKA using randomly generated data."""

        num_samples = 5  # number of samples per class in the input data
        num_features = 2  # number of features in the input data
        # pylint: disable=invalid-name
        C = 1  # SVM soft-margin penalty
        maxiters = 2

        # create random test data and labels:
        x_train = np.random.rand(2 * num_samples, num_features)
        y_train = np.concatenate((-1 * np.ones(num_samples), np.ones(num_samples)))

        # Define the feature map and its initial parameters:
        feature_map = FeatureMap(feature_dimension=num_features)
        initial_point = np.random.uniform(-1, 1, size=feature_map._num_parameters)

        runtime_inputs = {
            "feature_map": feature_map,
            "data": x_train,
            "labels": y_train,
            "initial_kernel_parameters": initial_point,
            "maxiters": maxiters,
            "C": C,
        }
        options = {"backend_name": self.backend_name}
        job = self.provider.runtime.run(
            program_id="quantum-kernel-alignment",
            options=options,
            inputs=runtime_inputs,
            callback=self.simple_callback,
        )
        self.log.debug("Job ID: %s", job.job_id())
        result = job.result()
        self.log.debug("Job result: %s", result)
