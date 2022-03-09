from dataclasses import asdict

from qiskit_primitives.estimator import Estimator


def main(
        backend,
        user_messenger,
        circuits,
        observables,
        parameters=None,
        circuit_indices=None,
        observable_indices=None,
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

    Returns:

    """
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

    result = asdict(raw_result)
    for metadata in result["metadata"]:
        metadata["shots"] = shots

    return result
