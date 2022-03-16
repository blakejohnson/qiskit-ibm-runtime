from qiskit_primitives.estimator import Estimator


def main(
        backend,
        user_messenger,
        circuits,
        circuit_indices,
        observables,
        observable_indices,
        parameters=None,
        parameter_values=None,
        skip_transpilation=False,
        run_options=None,
):
    """Estimator primitive.

    Args:
        backend: Backend to run the circuits.
        user_messenger: Used to communicate with the user.
        circuits: Quantum circuits that represent quantum states
        observables: Observables.
        parameters: Parameters of quantum circuits, specifying the order in which values
            will be bound.
        circuit_indices: List of circuit indices.
        observable_indices: List of observable indices.
        parameter_values: Concrete parameters to be bound.
        skip_transpilation: Skip transpiling of circuits, default=False.
        run_options: Execution time options.

    Returns: Expectation values and metadata.

    """
    if len(circuit_indices) != len(observable_indices):
        raise ValueError("The length of circuit_indices must "
                         "match the length of observable_indices.")
    if parameter_values and len(parameter_values) != len(circuit_indices):
        raise ValueError("The length of parameter_values must "
                         "match the length of circuit_indices")

    estimator = Estimator(
        backend=backend,
        circuits=circuits,
        observables=observables,
        parameters=parameters,
        skip_transpilation=skip_transpilation
    )
    run_options = run_options or {}
    shots = run_options.get("shots") or backend.options.shots
    raw_result = estimator(
        circuit_indices=circuit_indices,
        observable_indices=observable_indices,
        parameter_values=parameter_values,
        **run_options)

    result = raw_result.__dict__
    for metadata in result["metadata"]:
        metadata["shots"] = shots

    return result
