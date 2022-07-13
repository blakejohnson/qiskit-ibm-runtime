# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for TorchTrainer and TorchInferer."""

import base64
from typing import Any

import dill
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from .base_testcase import BaseTestCase
from .decorator import get_provider_and_backend

try:
    from torch import Tensor
    from torch.nn import MSELoss
    from torch.optim import Adam
except ImportError:

    class Dataset:
        """Empty Dataset class
        Replacement if torch.utils.data.Dataset is not present.
        """

        pass


# pylint: disable=line-too-long
LOADER = "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cmVmZXRjaF9mYWN0b3KUSwKMCnBpbl9tZW1vcnmUiYwRcGluX21lbW9yeV9kZXZpY2WUjACUjAd0aW1lb3V0lEsAjA53b3JrZXJfaW5pdF9mbpROjCRfRGF0YUxvYWRlcl9fbXVsdGlwcm9jZXNzaW5nX2NvbnRleHSUTowNX2RhdGFzZXRfa2luZJRLAIwKYmF0Y2hfc2l6ZZRLAYwJZHJvcF9sYXN0lImMB3NhbXBsZXKUjBh0b3JjaC51dGlscy5kYXRhLnNhbXBsZXKUjBFTZXF1ZW50aWFsU2FtcGxlcpSTlCmBlH2UjAtkYXRhX3NvdXJjZZRoWHNijA1iYXRjaF9zYW1wbGVylGh/jAxCYXRjaFNhbXBsZXKUk5QpgZR9lChofmiCaHxLAWh9iXVijAlnZW5lcmF0b3KUTowKY29sbGF0ZV9mbpSMH3RvcmNoLnV0aWxzLmRhdGEuX3V0aWxzLmNvbGxhdGWUjA9kZWZhdWx0X2NvbGxhdGWUk5SMEnBlcnNpc3RlbnRfd29ya2Vyc5SJjBhfRGF0YUxvYWRlcl9faW5pdGlhbGl6ZWSUiIwbX0l0ZXJhYmxlRGF0YXNldF9sZW5fY2FsbGVklE6MCV9pdGVyYXRvcpROdWIu"


class TestTorchTrainInfer(BaseTestCase):
    """Test Torch train."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name
        cls.backend = cls.provider.get_backend(backend_name)
        # Use callback if on real device to avoid CI timeout
        cls.callback_func = None if cls.backend.configuration().simulator else cls.simple_callback

    def setUp(self) -> None:
        """Test case setup."""
        # Construct simple feature map

        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)
        # Construct simple feature map
        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)
        # Construct QNN
        self.qnn = TwoLayerQNN(1, feature_map, ansatz)

    def test_torch_train_direct(self):
        """Test torch train script directly."""
        initial_weights = np.array([-0.08473834])
        model = TorchConnector(self.qnn, initial_weights)
        # Test torch-train
        inputs = {
            "model": obj_to_str(model),
            "optimizer": obj_to_str(Adam(model.parameters(), lr=0.1)),
            "loss_func": obj_to_str(MSELoss(reduction="sum")),
            "train_data": LOADER,
            "epochs": 1,
            "seed": 42,
        }
        options = {"backend_name": self.backend_name}
        program_id = "torch-train"
        job = self.provider.runtime.run(program_id=program_id, inputs=inputs, options=options)
        result = job.result()

        self.assertEqual(len(result["train_history"]["train"]), 1)
        self.assertEqual(len(result["train_history"]["validation"]), 0)
        self.assertTrue(isinstance(str_to_obj(result["model_state_dict"])["weight"], Tensor))
        self.assertTrue(isinstance(result["execution_time"], float))
        if self.backend.configuration().simulator:
            self.assertTrue(result["train_history"]["train"][0]["loss"] <= 1)
        # Test torch-train with a validation data set
        inputs = {
            "model": obj_to_str(model),
            "optimizer": obj_to_str(Adam(model.parameters(), lr=0.1)),
            "loss_func": obj_to_str(MSELoss(reduction="sum")),
            "train_data": LOADER,
            "val_data": LOADER,
            "epochs": 1,
            "seed": 42,
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(program_id=program_id, inputs=inputs, options=options)
        result = job.result()

        self.assertEqual(len(result["train_history"]["train"]), 1)
        self.assertEqual(len(result["train_history"]["validation"]), 1)
        self.assertTrue(isinstance(str_to_obj(result["model_state_dict"])["weight"], Tensor))
        self.assertTrue(isinstance(result["execution_time"], float))
        if self.backend.configuration().simulator:
            self.assertTrue(result["train_history"]["train"][0]["loss"] <= 1)

    def test_torch_infer_directly(self):
        """Test torch infer script directly."""
        model = TorchConnector(self.qnn, initial_weights=np.array([-1.5586]))
        # Test torch-infer for prediction
        inputs = {
            "model": obj_to_str(model),
            "data": LOADER,
        }

        options = {"backend_name": self.backend_name}
        program_id = "torch-infer"
        job = self.provider.runtime.run(program_id=program_id, inputs=inputs, options=options)
        result = job.result()
        self.assertEqual(len(result["prediction"]), 20)
        self.assertTrue(isinstance(result["execution_time"], float))
        # Test torch-infer for scoring
        inputs = {
            "model": obj_to_str(model),
            "data": LOADER,
            "score_func": obj_to_str(MSELoss()),
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(program_id=program_id, inputs=inputs, options=options)
        result = job.result()
        self.assertEqual(len(result["prediction"]), 20)
        self.assertTrue(isinstance(result["execution_time"], float))
        self.assertTrue(isinstance(result["score"], float))
        if self.backend.configuration().simulator:
            self.assertTrue(result["score"] <= 0.1)


def obj_to_str(obj: Any) -> str:
    """
    Encodes any object into a JSON-compatible string using dill. The intermediate
    binary data must be converted to base 64 to be able to decode into utf-8 format.

    Returns:
        The encoded string
    """
    string = base64.b64encode(dill.dumps(obj, byref=False)).decode("utf-8")
    return string


def str_to_obj(string: str) -> Any:
    """
    Decodes a previously encoded string using dill (with an intermediate step
    converting the binary data from base 64).

    Returns:
        The decoded object
    """
    obj = dill.loads(base64.b64decode(string.encode()))
    return obj