# This code is part of qiskit-runtime.
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

"""Runtime program for (Hybrid) QNN inference using PyTorch."""
import base64
import dill
import json
import sys
import traceback
from time import time
from math import fsum
from typing import Callable, List, Tuple, Union, Any

from qiskit import Aer
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.ibmq.runtime.utils import RuntimeDecoder
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.connectors import TorchConnector

try:
    from torch import Tensor, no_grad
    from torch.nn import Module
    from torch.nn.modules.loss import _Loss
    from torch.utils.data import DataLoader
except ImportError:

    class DataLoader:  # type: ignore
        """Empty DataLoader class
        Replacement if torch.utils.data.DataLoader is not present.
        """

        pass

    class Tensor:  # type: ignore
        """Empty Tensor class
        Replacement if torch.Tensor is not present.
        """

        pass

    class _Loss:  # type: ignore
        """Empty _Loss class
        Replacement if torch.nn.modules.loss._Loss is not present.
        """

        pass

    class Module:  # type: ignore
        """Empty Module class
        Replacement if torch.nn.Module is not present.
        Always fails to initialize
        """

        pass


WAIT=0.2


class TorchInferer:
    """Class used to run predictions and scoring using PyTorch."""

    def __init__(
        self,
        model: Module,
        data_loader: DataLoader = None,
        score_func: Union[str, Callable] = None,
    ):

        """
        Args:
            model: (Hybrid) QNN model of type ``torch.nn.Module``
            data_loader: Inference/test data loader of type ``torch.utils.data.DataLoader``.
            score_func: If provided, this function will be used to calculate
            the model's performance score as ``score = score_func(data, target)``.
        """
        self.model = model
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.score_func = score_func

    def predict(self) -> List[Tensor]:
        """
        Predict the outputs using the trained model

        Returns:
            The predicted outputs
        """
        self.model.eval()
        with no_grad():
            outputs = []
            for data, _ in self.data_loader:
                # Forward pass
                out = self.model(data)
                outputs.append(out.tolist())
        return outputs

    def score(self) -> Tuple[List[Tensor], Union[Tensor, float]]:
        """
        Calculate the score of the trained model

        Returns:
            A tuple of the score and the predicted outputs
        """
        self.model.eval()
        with no_grad():
            outputs = []
            scores = []

            for data, target in self.data_loader:
                # Forward pass
                out = self.model(data)
                score = self.score_func(out, target)
                scores.append(score)
                outputs.append(out.tolist())
            score = fsum(scores) / len(self.data_loader)
            if not isinstance(self.score_func, _Loss):
                score /= self.batch_size
        return outputs, score


def main(backend, user_messenger, **kwargs):
    """Entry function."""
    # Define mandatory arguments
    mandatory = {"model", "data"}
    missing = mandatory - set(kwargs.keys())
    if len(missing) > 0:
        raise ValueError(f"The following mandatory arguments are missing: {missing}.")

    # Get inputs and deserialize
    model = str_to_obj(kwargs.get("model", None))
    data_loader = str_to_obj(kwargs.get("data", None))
    score_func = kwargs.get("score_func", None)
    if score_func is not None:
        score_func = str_to_obj(score_func)

    shots = kwargs.get("shots", 1024)
    measurement_error_mitigation = kwargs.get("measurement_error_mitigation", False)

    # Set up quantum instance for qnn layers
    if measurement_error_mitigation:
        _quantum_instance = QuantumInstance(
            backend,
            shots=shots,
            measurement_error_mitigation_shots=shots,
            measurement_error_mitigation_cls=CompleteMeasFitter,
            wait=WAIT,
        )
    else:
        _quantum_instance = QuantumInstance(backend, shots=shots, wait=WAIT)

    if isinstance(model, TorchConnector):
        # Case for single qnn that has been wrapped in a TorchConnector
        # i.e: `model = TorchConnector(qnn1)`
        model.neural_network.quantum_instance = _quantum_instance
    else:
        # Case for a multi-layer torch module that contains a layer
        # with a qnn wrapped in a TorchConnector.
        # i.e: `model = Sequential(layer1, layer2, TorchConnector(qnn2))`
        for layer in model.children():
            if isinstance(layer, TorchConnector):
                layer.neural_network.quantum_instance = _quantum_instance

    infer = TorchInferer(model=model, data_loader=data_loader, score_func=score_func)
    serialized_result = {}

    # If score_func is provided, call the score method.
    # Otherwise, call the predict method.
    start = time()
    if score_func is not None:
        out, score = infer.score()
        if isinstance(score, Tensor):
            score = score.tolist()
        serialized_result["score"] = score
    else:
        out = infer.predict()
    serialized_result["prediction"] = out
    serialized_result["execution_time"] = time() - start

    user_messenger.publish(serialized_result, final=True)


def str_to_obj(string: str) -> Any:
    """
    Decodes a previously encoded string using dill (with an intermediate step
    converting the binary data from base 64).

    Returns:
        The decoded object
    """
    obj = dill.loads(base64.b64decode(string.encode()))
    return obj


if __name__ == "__main__":
    # the code currently uses Aer instead of runtime provider
    _backend = Aer.get_backend("qasm_simulator")
    user_params = {}
    if len(sys.argv) > 1:
        # If there are user parameters.
        user_params = json.loads(sys.argv[1], cls=RuntimeDecoder)
    try:
        main(_backend, **user_params)
    except Exception:
        print(traceback.format_exc())
