import sys

import numpy as np
from qka.featuremaps import FeatureMap

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class TestQKA(BaseTestCase):
    """Test QKA program."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        sys.path.insert(0, '..')
        cls.provider = provider
        cls.backend_name = backend_name

    def test_qka_random(self):
        """Test QKA using randomly generated data."""

        num_samples = 5   # number of samples per class in the input data
        num_features = 2  # number of features in the input data
        C = 1             # SVM soft-margin penalty
        maxiters = 2

        # create random test data and labels:
        x_train = np.random.rand(2*num_samples, num_features)
        y_train = np.concatenate((-1*np.ones(num_samples), np.ones(num_samples)))

        # Define the feature map and its initial parameters:
        fm = FeatureMap(feature_dimension=num_features)
        initial_point = np.random.uniform(-1, 1, size=fm._num_parameters)

        runtime_inputs = {
            'feature_map': fm,
            'data': x_train,
            'labels': y_train,
            'initial_kernel_parameters': initial_point,
            'maxiters': maxiters,
            'C': C,
        }
        options = {'backend_name': self.backend_name}
        job = self.provider.runtime.run(program_id="quantum-kernel-alignment",
                                        options=options,
                                        inputs=runtime_inputs,
                                        callback=self.simple_callback
                                        )
        self.log.debug("Job ID: %s", job.job_id())
        result = job.result()
        self.log.debug("Job result: %s", result)
