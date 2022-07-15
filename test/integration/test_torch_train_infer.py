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
    from torch import Tensor, tensor, float32
    from torch.nn import MSELoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader

except ImportError:

    class Dataset:
        """Empty Dataset class
        Replacement if torch.utils.data.Dataset is not present.
        """

        pass


DATA_X = [
    [0.3067],
    [1.3521],
    [0.6457],
    [0.2820],
    [-0.4797],
    [0.9167],
    [-0.3922],
    [2.4616],
    [2.9133],
    [-0.7324],
    [1.8330],
    [0.1816],
    [0.4275],
    [2.6741],
    [-2.6953],
    [-2.5941],
    [-3.0146],
    [2.0899],
    [1.7477],
    [2.3249],
]

DATA_Y = [
    [0.4934],
    [1.0958],
    [0.5863],
    [0.3905],
    [-0.6142],
    [0.8496],
    [-0.5248],
    [0.8067],
    [0.2351],
    [-0.7028],
    [0.8717],
    [0.2902],
    [0.3971],
    [0.4780],
    [-0.6241],
    [-0.4735],
    [-0.0819],
    [0.9150],
    [1.1619],
    [0.8016],
]


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

        # Mock torch dataset
        x_tensor = tensor(DATA_X, dtype=float32)
        y_tensor = tensor(DATA_Y, dtype=float32)
        dataset = [[x, y] for (x, y) in zip(x_tensor, y_tensor)]
        # Construct torch dataloader
        self.loader = DataLoader(dataset)  # type: ignore[arg-type,var-annotated]

    def test_torch_train_direct(self):
        """Test torch train script directly."""
        initial_weights = np.array([-0.08473834])
        model = TorchConnector(self.qnn, initial_weights)
        # Test torch-train
        inputs = {
            "model": obj_to_str(model),
            "optimizer": obj_to_str(Adam(model.parameters(), lr=0.1)),
            "loss_func": obj_to_str(MSELoss(reduction="sum")),
            "train_data": obj_to_str(self.loader),
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
            "train_data": obj_to_str(self.loader),
            "val_data": obj_to_str(self.loader),
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
            "data": obj_to_str(self.loader),
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
            "data": obj_to_str(self.loader),
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
