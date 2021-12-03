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
import dill
from typing import Any

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
    from torch.utils.data import DataLoader, Dataset
except ImportError:

    class Dataset:  # type: ignore
        """Empty Dataset class
        Replacement if torch.utils.data.Dataset is not present.
        """

        pass


LOADER = "gASVnggAAAAAAACMG3RvcmNoLnV0aWxzLmRhdGEuZGF0YWxvYWRlcpSMCkRhdGFMb2FkZXKUk5QpgZR9lCiMB2RhdGFzZXSUjApkaWxsLl9kaWxslIwMX2NyZWF0ZV90eXBllJOUKGgGjApfbG9hZF90eXBllJOUjAR0eXBllIWUUpSMDFRvcmNoRGF0YXNldJSMGHRvcmNoLnV0aWxzLmRhdGEuZGF0YXNldJSMB0RhdGFzZXSUk5SFlH2UKIwKX19tb2R1bGVfX5SMCF9fbWFpbl9flIwHX19kb2NfX5SMEU1hcC1zdHlsZSBkYXRhc2V0lIwIX19pbml0X1+UaAaMEF9jcmVhdGVfZnVuY3Rpb26Uk5QoaAaMDF9jcmVhdGVfY29kZZSTlChLA0sASwBLA0sCS0NDIHQAfAGDAaABoQB8AF8CdAB8AoMBoAGhAHwAXwNkAFMAlE6FlCiMBlRlbnNvcpSMBWZsb2F0lIwBWJSMAXmUdJSMBHNlbGaUaCFoIoeUjE0vdmFyL2ZvbGRlcnMvdnovYjFmeXBocmQxMDMxZGRmZDF4ZGhqcV9tMDAwMGduL1QvaXB5a2VybmVsXzQ4MTAvMzEzNDA0MzM4My5weZRoGEsKQwQAAQ4BlCkpdJRSlGNfX2J1aWx0aW5fXwpfX21haW5fXwpoGE5OfZROdJRSlIwHX19sZW5fX5RoGihoHChLAUsASwBLAUsCS0NDCnQAfABqAYMBUwCUaB6MA2xlbpRoIYaUaCSFlGgmaC1LDkMCAAGUKSl0lFKUY19fYnVpbHRpbl9fCl9fbWFpbl9fCmgtTk59lE50lFKUjAtfX2dldGl0ZW1fX5RoGihoHChLAksASwBLBUsDS0NDNmQBZABsAH0CfAKgAXwBoQFyGnwBoAKhAH0BfABqA3wBGQB9A3wAagR8ARkAfQR8A3wEZgJTAJROSwCGlCiMBXRvcmNolIwJaXNfdGVuc29ylIwGdG9saXN0lGghaCJ0lChoJIwDaWR4lGg7jANYX2mUjAN5X2mUdJRoJmg4SxFDDAABCAEKAQgCCgEKA5QpKXSUUpRjX19idWlsdGluX18KX19tYWluX18KaDhOTn2UTnSUUpSMDl9fcGFyYW1ldGVyc19flCmMDV9fc2xvdG5hbWVzX1+UXZR1dJRSlCmBlH2UKGghjAx0b3JjaC5fdXRpbHOUjBJfcmVidWlsZF90ZW5zb3JfdjKUk5QojA10b3JjaC5zdG9yYWdllIwQX2xvYWRfZnJvbV9ieXRlc5STlEJNAQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAAAHN0b3JhZ2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADE0MDIxNTQ2NDA2MzE1MnECWAMAAABjcHVxA0sUTnRxBFEugAJdcQBYDwAAADE0MDIxNTQ2NDA2MzE1MnEBYS4UAAAAAAAAAFgInT7IEK0/X0slP4hjkD4QmvW+hqtqPwjIyL6Rih1AKnM6QNl7O7+Enuo/zug5PhPm2j5/JCtAJH8swHEGJsB/7kDAH8EFQPq03z9syhRAlIWUUpRLAEsUSwGGlEsBSwGGlImMC2NvbGxlY3Rpb25zlIwLT3JkZXJlZERpY3SUk5QpUpR0lFKUaCJoUihoVUJNAQAAgAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAihYBwAAAHN0b3JhZ2VxAGN0b3JjaApGbG9hdFN0b3JhZ2UKcQFYDwAAADE0MDIxNTQ1MTc5MDI4OHECWAMAAABjcHVxA0sUTnRxBFEugAJdcQBYDwAAADE0MDIxNTQ1MTc5MDI4OHEBYS4UAAAAAAAAAG6a/D50RIw/BRoWP1bvxz7oOx2/YnxZP6RbBr/VgU4/YrdwPh3oM7+oJF8/opuUPnNPyz4hv/Q+78cfv/Vo8r4+pKe9hj9qP9a4lD+TOE0/lIWUUpRLAEsUSwGGlEsBSwGGlIloXSlSlHSUUpR1YowLbnVtX3dvcmtlcnOUSwCMD3ByZWZldGNoX2ZhY3RvcpRLAowKcGluX21lbW9yeZSJjAd0aW1lb3V0lEsAjA53b3JrZXJfaW5pdF9mbpROjCRfRGF0YUxvYWRlcl9fbXVsdGlwcm9jZXNzaW5nX2NvbnRleHSUTowNX2RhdGFzZXRfa2luZJRLAIwKYmF0Y2hfc2l6ZZRLAYwJZHJvcF9sYXN0lImMB3NhbXBsZXKUjBh0b3JjaC51dGlscy5kYXRhLnNhbXBsZXKUjBFTZXF1ZW50aWFsU2FtcGxlcpSTlCmBlH2UjAtkYXRhX3NvdXJjZZRoTnNijA1iYXRjaF9zYW1wbGVylGhzjAxCYXRjaFNhbXBsZXKUk5QpgZR9lChocmh2aHBLAWhxiXVijAlnZW5lcmF0b3KUTowKY29sbGF0ZV9mbpSMH3RvcmNoLnV0aWxzLmRhdGEuX3V0aWxzLmNvbGxhdGWUjA9kZWZhdWx0X2NvbGxhdGWUk5SMEnBlcnNpc3RlbnRfd29ya2Vyc5SJjBhfRGF0YUxvYWRlcl9faW5pdGlhbGl6ZWSUiIwbX0l0ZXJhYmxlRGF0YXNldF9sZW5fY2FsbGVklE6MCV9pdGVyYXRvcpROdWIu"


class TestTorchTrainInfer(BaseTestCase):
    """Test Torch train."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
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
        import torch

        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)
        # Construct simple feature map
        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)
        # Construct QNN
        self.qnn = TwoLayerQNN(1, feature_map, ansatz)

    def find_program_id(self, program_name):
        """Returns the actual program id"""
        potential_id = None
        for program in self.provider.runtime.programs():
            if program.name == program_name:
                return program.program_id
            elif program.name.startswith(program_name):
                potential_id = program.program_id
        return potential_id

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
        program_id = self.find_program_id("torch-train")
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
        program_id = self.find_program_id("torch-infer")
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
