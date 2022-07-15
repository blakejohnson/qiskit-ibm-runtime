# type: ignore

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

"""The Qiskit runtime VQE program."""

import logging
import sys
import json
import traceback

import numpy as np

from qiskit import Aer
from qiskit.algorithms.optimizers import SPSA, QNSPSA
from qiskit.algorithms import VQE
from qiskit.opflow import PauliExpectation
from qiskit.utils import QuantumInstance

from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit_ibm_runtime.utils import RuntimeDecoder

WAIT = 0.2

logger = logging.getLogger(__name__)


class Publisher:
    """Class used to publish interim results."""

    def __init__(self, messenger):
        self._messenger = messenger

    def callback(self, *args, **kwargs):
        """Publisher callback"""
        text = list(args)
        for key, value in kwargs.items():
            text.append({key: value})
        self._messenger.publish(text)


def _process_qnspsa(optimizer, backend, ansatz):
    """Provide the callable to calculate the fidelity for QNSPSA.

    This is required since the callable cannot be serialized and sent to the runtime server.
    Therefore we have to re-define it here.
    """
    fidelity = QNSPSA.get_fidelity(ansatz, backend, PauliExpectation())
    optimizer.fidelity = fidelity


def main(backend, user_messenger, **kwargs):
    """Entry function."""
    # parse inputs
    mandatory = {"ansatz", "operator"}
    missing = mandatory - set(kwargs.keys())
    if len(missing) > 0:
        raise ValueError(f"The following mandatory arguments are missing: {missing}.")

    ansatz = kwargs["ansatz"]
    operator = kwargs["operator"]
    aux_operators = kwargs.get("aux_operators", None)
    initial_point = kwargs.get("initial_point", None)

    # set the number of batched evaluations, with a maximum of 1000 parameter sets at once
    # for now to avoid any memory issues of allocating too many large vectors
    max_evals_grouped = kwargs.get("max_evals_grouped", min(2 * ansatz.num_parameters, 1000))

    optimizer = kwargs.get("optimizer", SPSA())
    if isinstance(optimizer, dict):
        # since we cannot properly deprecate we raise the warning here to break the program
        raise DeprecationWarning(
            "Passing optimizers as dictionaries is no longer supported. "
            "Pass them directly as optimizer objects instead."
        )

    # re-define the fidelity callable if the optimizer is QNSPSA
    if isinstance(optimizer, QNSPSA):
        _process_qnspsa(optimizer, backend, ansatz)

    shots = kwargs.get("shots", 1024)
    measurement_error_mitigation = kwargs.get("measurement_error_mitigation", False)

    # set up quantum instance
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

    publisher = Publisher(user_messenger)

    # verify the initial point
    if initial_point == "random" or initial_point is None:
        initial_point = np.random.random(ansatz.num_parameters)
    elif len(initial_point) != ansatz.num_parameters:
        raise ValueError("Mismatching number of parameters and initial point dimension.")

    # construct the VQE instance
    vqe = VQE(
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        expectation=PauliExpectation(),
        max_evals_grouped=max_evals_grouped,
        callback=publisher.callback,
        quantum_instance=_quantum_instance,
    )
    result = vqe.compute_minimum_eigenvalue(operator, aux_operators)

    aux_operator_values = (
        result.aux_operator_eigenvalues if result.aux_operator_eigenvalues is not None else None
    )

    serialized_result = {
        "optimizer_evals": result.optimizer_evals,
        "optimizer_time": result.optimizer_time,
        "optimal_value": result.optimal_value,
        "optimal_point": result.optimal_point,
        "optimal_parameters": None,  # ParameterVectorElement is not serializable
        "cost_function_evals": result.cost_function_evals,
        "eigenstate": result.eigenstate,
        "eigenvalue": result.eigenvalue,
        "aux_operator_eigenvalues": aux_operator_values,
        "optimizer_history": None,
    }

    user_messenger.publish(serialized_result, final=True)


if __name__ == "__main__":
    # the code currently uses Aer instead of runtime provider
    _backend = Aer.get_backend("qasm_simulator")
    user_params = {}
    if len(sys.argv) > 1:
        # If there are user parameters.
        user_params = json.loads(sys.argv[1], cls=RuntimeDecoder)
    try:
        main(_backend, **user_params)
    except Exception:  # pylint: disable=broad-except
        print(traceback.format_exc())
