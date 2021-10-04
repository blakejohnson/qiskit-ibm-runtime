import numpy as np
from dataclasses import asdict

from qiskit.quantum_info import SparsePauliOp


def main(backend, user_messenger, **kwargs):

    state = kwargs.pop("state")
    observable = SparsePauliOp.from_list(kwargs.pop("observable"))
    parameters = kwargs.pop("parameters")
    class_name = kwargs.pop("class_name", "PauliExpectationValue")
    transpile_options = kwargs.pop("transpile_options", {})
    run_options = kwargs.pop("run_options", {})

    try:
        if class_name == "PauliExpectationValue":
            from qiskit.evaluators import PauliExpectationValue as Evaluator
        elif class_name == "ExactExpectationValue":
            from qiskit.evaluators import ExactExpectationValue as Evaluator
    except ModuleNotFoundError:
        raise RuntimeError("You are not authorized to use this program.")

    expval = Evaluator(state, observable, backend)
    expval.set_transpile_options(**transpile_options)
    expval.set_run_options(**run_options)
    result = expval.evaluate(parameters)

    # for debug
    # print(result)

    ret = {
        key: val.tolist() if isinstance(val, np.ndarray) else val
        for key, val in asdict(result).items()
    }
    user_messenger.publish(ret, final=True)
