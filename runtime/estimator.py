from dataclasses import asdict

import numpy as np
from mthree.utils import final_measurement_mapping
from qiskit.evaluators.backends import ReadoutErrorMitigation
from qiskit.quantum_info import SparsePauliOp


def main(backend, user_messenger, **kwargs):
    state = kwargs.pop("state")
    observable = kwargs.pop("observable")
    if isinstance(observable, list):
        observable = SparsePauliOp.from_list(observable)
    parameters = kwargs.pop("parameters")
    class_name = kwargs.pop("class_name", "PauliExpectationValue")
    transpile_options = kwargs.pop("transpile_options", {})
    run_options = kwargs.pop("run_options", {})
    measurement_error_mitigation = kwargs.pop("measurement_error_mitigation", False)

    try:
        if class_name == "PauliExpectationValue":
            from qiskit.evaluators import PauliExpectationValue as Evaluator

            if measurement_error_mitigation:
                expval = Evaluator(state, observable, backend)
                expval.set_transpile_options(**transpile_options)
                mapping = final_measurement_mapping(expval.transpiled_circuits[0])
                backend = ReadoutErrorMitigation(
                    backend=backend,
                    mitigation="mthree",
                    refresh=1800,  # refresh the calibration data every 1800 seconds
                    shots=8192,  # use M3's default shot number
                    qubits=list(mapping),
                )
            else:
                # use backend as is
                pass
        elif class_name == "ExactExpectationValue":
            # Note: ExactExpectationValue works only with Aer backend
            from qiskit.evaluators import ExactExpectationValue as Evaluator
    except ModuleNotFoundError:
        raise RuntimeError("You are not authorized to use this program.")

    expval = Evaluator(state, observable, backend)
    expval.set_transpile_options(**transpile_options)
    expval.set_run_options(**run_options)
    result = expval.evaluate(parameters)

    ret = {
        key: val.tolist() if isinstance(val, np.ndarray) else val
        for key, val in asdict(result).items()
    }
    user_messenger.publish(ret, final=True)
