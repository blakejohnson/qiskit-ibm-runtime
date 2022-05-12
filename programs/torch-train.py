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

"""Runtime program for (Hybrid) QNN training using PyTorch."""
import base64
import dill
import json
import sys
import traceback
from collections import OrderedDict
from math import fsum
from time import time
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

from qiskit import Aer
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.ibmq.runtime import UserMessenger
from qiskit.providers.ibmq.runtime.utils import RuntimeDecoder
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.connectors import TorchConnector

try:
    from torch import Tensor
    from torch import get_rng_state, manual_seed, set_rng_state, no_grad
    from torch.nn import Module
    from torch.nn.modules.loss import _Loss
    from torch.optim import LBFGS, Optimizer
    from torch.utils.data import DataLoader, Dataset
    from qiskit_machine_learning.runtime import HookBase
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

    class Optimizer:  # type: ignore
        """Empty Optimizer
        Replacement if torch.optim.Optimizer is not present.
        """

        pass

    class Module:  # type: ignore
        """Empty Module class
        Replacement if torch.nn.Module is not present.
        Always fails to initialize
        """

        pass

    class HookBase:  # type: ignore
        """Empty HookBase class
        Replacement if qiskit_machine_learning.runtime.HookBase is not present.
        Always fails to initialize
        """

        pass


WAIT = 0.2


class History:
    """Class used to save train history."""

    def __init__(self):
        self.metrics = []

    def log_metrics(
        self,
        loss: float = 0,
        epoch: int = 0,
        forward_time: float = 0,
        backward_time: float = 0,
        epoch_time: float = 0,
    ):
        """a log method for metrics"""
        self.metrics.append(
            {
                "epoch": epoch,
                "loss": loss,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "epoch_time": epoch_time,
            }
        )

    def get_metrics(self):
        """return the logged metrics"""
        return self.metrics


class TorchTrainer:
    """Class used to control the PyTorch training loop logic."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_func: Union[_Loss, Callable],
        train_loader: DataLoader,
        hooks: List[HookBase],
        val_loader: Optional[DataLoader] = None,
        max_epochs: Optional[int] = 10,
        log_every_n_epochs: Optional[int] = 1,
        start_epoch: int = 0,
        user_messenger: Optional[UserMessenger] = None,
    ) -> None:
        """
        Args:
            model: A PyTorch nn.Module to be trained.
            optimizer: A PyTorch optimizer initialized for the model parameters.
            loss_function: A PyTorch-compatible loss function. Can be one of the
                official PyTorch loss functions from ``torch.nn.loss`` or a custom
                function defined by the user.
            train_loader: A PyTorch data loader object containing the training dataset.
            hooks: List of custom hook functions.
            val_loader: A PyTorch data loader object containing the validation dataset.
                If no validation loader is provided, the validation step will be skipped
                during training.
            max_epochs: The maximum number of training epochs. By default, 100.
            log_every_n_epochs: Logging period for train history and validation. By default,
                there will be logs every epoch.
            start_epoch: initial epoch for warm-start training.
            user_messenger (UserMessenger): used to publish interim results.
        """
        # Use setters to check for valid inputs
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_func = loss_func
        self.max_epochs = max_epochs + start_epoch
        self.log_period = log_every_n_epochs

        self.start_epoch = start_epoch
        self.epoch = start_epoch  # warm start
        self.global_step = start_epoch * len(self.train_loader)

        # Only do validation if val_loader is provided
        self.validate = val_loader is not None

        # Initialize training/validation history loggers
        self.train_logger = History()
        self.val_logger = History()

        # Register hooks provided by user
        self.hooks = []
        self.register_hooks(hooks)

        self.user_messenger = user_messenger

    def register_hooks(self, hooks: List[HookBase]) -> None:
        """
        Register hooks to the trainer.
        Args:
            hooks: List of hook classes to interact with the training loop.
        """
        if len(hooks) > 0:
            for hook in hooks:
                hook.trainer = self
                self.hooks.append(hook)

    def train(self) -> Tuple[OrderedDict, Dict[str, History]]:
        """
        This function implements a basic training loop with a main training
        step per iteration and several placeholders for user-defined hooks.

        Returns:
            Tuple of the trained parameters in the model (model.state_dict())
            and the dict of loggers for training_data and val_data
        """
        # Enable grads + batchnorm + dropout
        self.model.train()

        # Main training loop
        # Placeholder for `before_train` hooks
        self.before_train()
        # While loop for easier control using hooks
        while self.epoch < self.max_epochs:
            # Keep track of loss, forward and backward times
            train_losses = []
            fwd_times = []
            bckwd_times = []
            # Save `start_time` to calculate total training time
            start_time = time()
            # Iterate over batches
            for train_batch in self.train_loader:
                # Placeholder for `before_step` hooks
                self.before_step()
                # Main train step
                loss, t_fwd, t_bckwd = self.train_step(train_batch)
                # Save results
                train_losses.append(loss)
                fwd_times.append(t_fwd)
                bckwd_times.append(t_bckwd)
                # Placeholder for `after_step` hooks
                self.after_step()
                # Keep track of global step (useful for hooks)
                self.global_step += 1
            # Calculate total training time
            end_time = time() - start_time
            # Logging and validation are performed in `self.after_epoch`.
            # It also executes any `after_epoch` hook.
            self.after_epoch(train_losses, fwd_times, bckwd_times, end_time)
            # Keep track of epoch (useful for logging, hooks)
            self.epoch += 1

        # Training results are returned in `self.after_train`. It also
        # executes any `after_train` hook.
        return self.after_train()

    def train_step(self, batch_data: List[Tensor]) -> Tuple[float, float, float]:
        """
        Basic training step for batch data without gradient accumulation:
        (1) Forward pass, (2) loss calculation, (3) backward pass,
        (4) optimize weights.

        Args:
            A list of batch data from ``train_loader``
        Returns:
            A tuple of loss value, forward pass time, and backward pass time
        """
        data, target = batch_data
        loss = 0
        fwd_time = 0
        bckwd_time = 0
        # LBFGS optimizer requires an explicit closure function to be
        # defined and executed. This case is treated separately
        if isinstance(self.optimizer, LBFGS):

            def closure():
                # Initialize/clear gradients
                self.optimizer.zero_grad(set_to_none=True)
                # Forward pass
                output = self.model(data)
                # Calculate loss
                loss = self.loss_func(output, target)
                # Backward pass
                loss.backward()
                return loss

            # Run optimization step
            self.optimizer.step(closure)
            # Calculate the loss again for monitoring
            loss = closure()
            # Set Forward pass time and backward pass time to 0 for a LBFGS optimizer
            fwd_time = 0
            bckwd_time = 0
        else:
            # Initialize/clear gradients
            self.optimizer.zero_grad(set_to_none=True)
            stamp1 = time()
            # Forward pass
            output = self.model(data)
            stamp2 = time()
            # Calculate loss
            loss = self.loss_func(output, target)
            stamp3 = time()
            # Backward pass
            loss.backward()
            stamp4 = time()
            # Run optimization step
            self.optimizer.step()
            fwd_time = stamp2 - stamp1
            bckwd_time = stamp4 - stamp3
        # Return loss value, forward pass time, and backward pass time
        return loss.item(), fwd_time, bckwd_time

    def do_eval(self):
        """Perform validation after each epoch"""
        val_losses = []
        # Store the random seed for the next epoch,
        # so the random seed will be similar with/without validation
        # Disable grads + batchnorm + dropout
        rng_state = get_rng_state()
        self.model.eval()

        with no_grad():
            for data, target in self.val_loader:
                stamp1 = time()
                output = self.model(data)  # Forward pass
                stamp2 = time()
                val_loss = self.loss_func(output, target)  # Calculate loss
                val_losses.append(val_loss.item())
        # Load the random seed
        # Enable grads + batchnorm + dropout
        self.model.train()
        set_rng_state(rng_state)
        # Return average validation loss, epoch index, and forward pass time
        return fsum(val_losses) / len(val_losses), (stamp2 - stamp1)

    def before_train(self) -> None:
        """Perform `before_train` actions."""
        # Execute hooks if provided
        for hook in self.hooks:
            hook.before_train()

    def after_train(self) -> Tuple[OrderedDict, Dict[str, History]]:
        """
        Perform `after_train` actions including returning the trained parameters in the model
        and the dict of loggers.

        Returns:
            Tuple of the trained parameters in the model (model.state_dict())
            and the dict of loggers for training_data and val_data
        """
        # Execute hooks if provided
        for hook in self.hooks:
            hook.after_train()

        # Gather and return training results
        train_history = {
            "train": self.train_logger.get_metrics(),
            "validation": self.val_logger.get_metrics(),
        }
        return self.model.state_dict(), train_history

    def before_step(self) -> None:
        """Perform `before_step` actions."""
        # Execute hooks if provided
        for hook in self.hooks:
            hook.before_step()

    def after_step(self) -> None:
        """Perform `after_step` actions."""
        # Execute hooks if provided
        for hook in self.hooks:
            hook.after_step()

    def before_epoch(self) -> None:
        """Perform `before_epoch` actions."""
        # Execute hooks if provided
        for hook in self.hooks:
            hook.before_epoch()

    def after_epoch(self, train_losses, fwd_times, bckwd_times, epoch_time) -> None:
        """Perform `after_epoch` actions. Including validation, logging and hooks."""
        # For the moment, the log period is fixed to 1 epoch, but in the future
        # it could be accepted as a program input defined by the user (i.e. log
        # every 5 epochs).

        # If it's the first epoch or (the current epoch - the start epoch)
        # is a multiple of log_period
        if (self.epoch - self.start_epoch) % self.log_period == 0:
            val_avg_loss = None
            val_avg_fwd_t = None

            # Calculate average loss per epoch
            train_avg_loss = fsum(train_losses) / len(train_losses)
            # Calculate average forward time per epoch
            avg_fwd_t = fsum(fwd_times) / len(fwd_times)
            # Calculate average backward time per epoch
            avg_bckwd_t = fsum(bckwd_times) / len(bckwd_times)
            # If validation data is provided, perform a validation step
            if self.validate:
                val_avg_loss, val_avg_fwd_t = self.do_eval()

            # Publish interim results
            interim_result = {
                "epoch": self.epoch,
                "training loss": train_avg_loss,
                "validation loss": val_avg_loss,
                "average forward pass time": avg_fwd_t,
                "average backward pass time": avg_bckwd_t,
                "epoch_time": epoch_time,
            }

            self.user_messenger.publish(interim_result)
            # Log train metrics
            self.train_logger.log_metrics(
                loss=train_avg_loss,
                epoch=self.epoch,
                forward_time=avg_fwd_t,
                backward_time=avg_bckwd_t,
                epoch_time=epoch_time,
            )
            # If validation data is provided, log validation metrics.
            if self.validate:
                self.val_logger.log_metrics(
                    loss=val_avg_loss,
                    epoch=self.epoch,
                    forward_time=val_avg_fwd_t,
                    backward_time=0,
                    epoch_time=val_avg_fwd_t,
                )

        # Execute hooks if provided
        for hook in self.hooks:
            hook.after_epoch()


def main(backend, user_messenger, **kwargs):
    """Entry function."""
    # parse inputs
    mandatory = {"model", "optimizer", "loss_func", "train_data"}
    missing = mandatory - set(kwargs.keys())
    if len(missing) > 0:
        raise ValueError(f"The following mandatory arguments are missing: {missing}.")

    # Get inputs and deserialize
    print(kwargs)
    model = str_to_obj(kwargs.get("model", None))
    loss_func = str_to_obj(kwargs.get("loss_func", None))
    optimizer = str_to_obj(kwargs.get("optimizer", None))

    train_data_loader = str_to_obj(kwargs.get("train_data", None))
    val_data_loader = kwargs.get("val_data", None)
    if val_data_loader is not None:
        val_data_loader = str_to_obj(val_data_loader)

    hooks = kwargs.get("hooks", [])
    if len(hooks) > 0:
        hooks = str_to_obj(hooks)
    shots = kwargs.get("shots", 1024)
    epochs = kwargs.get("epochs", 10)
    start_epoch = kwargs.get("start_epoch", 0)
    measurement_error_mitigation = kwargs.get("measurement_error_mitigation", False)

    # Set quantum instance
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
    # Restore optimizer parameters after setting quantum instance, as modifying the layer
    # detaches tha corresponding parameters. Note that I have not found an "official"
    # PyTorch way of doing this, and this way breaks if the optimizer comes with a pre-loaded
    # state (this can cause problems with trying to stop and resume trainings).
    # Related issue: https://github.com/pytorch/pytorch/issues/42428.
    # For "normal" use, this fix should work fine:
    optimizer.param_groups[0]["params"] = list(model.parameters())
    if isinstance(optimizer, LBFGS):
        optimizer._params = optimizer.param_groups[0]["params"]
        optimizer._numel_cache = None
    # Fix the manual_seed of Pytorch
    seed = kwargs.get("seed", None)
    if seed:
        manual_seed(seed)

    # Initialize trainer
    trainer = TorchTrainer(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        hooks=hooks,
        max_epochs=epochs,
        start_epoch=start_epoch,
        user_messenger=user_messenger,
    )

    # Run training and measure time
    start = time()
    ret = trainer.train()
    execution_time = time() - start
    serialized_result = {
        "model_state_dict": obj_to_str(ret[0]),
        "train_history": ret[1],
        "execution_time": execution_time,
    }
    # Return the result
    user_messenger.publish(serialized_result, final=True)


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


if __name__ == "__main__":
    # The code currently uses Aer instead of runtime provider
    _backend = Aer.get_backend("qasm_simulator")
    user_params = {}
    if len(sys.argv) > 1:
        # If there are user parameters.
        user_params = json.loads(sys.argv[1], cls=RuntimeDecoder)
    try:
        main(_backend, **user_params)
    except Exception:
        print(traceback.format_exc())
