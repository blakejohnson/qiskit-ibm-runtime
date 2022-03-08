from dataclasses import asdict

from qiskit.quantum_info.primitives.estimator.pauli_estimator import PauliEstimator


def main(
        backend,
        user_messenger,
        circuits,
        observables,
        parameters=None,
        circuits_indices=None,
        observables_indices=None,
        parameters_values=None,
        transpile_options=None,
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
        circuits_indices: List of circuit indices.
        observables_indices: List of observable indices.
        parameters_values: Concrete parameters to be bound.
        transpile_options: Transpilation options.
        run_options: Execution time options.

    Returns:

    """
    estimator = PauliEstimator(
        backend=backend,
        circuits=circuits,
        observables=observables,
        parameters=parameters
    )
    if transpile_options:
        estimator.set_transpile_options(**transpile_options)
    run_options = run_options or {}
    shots = run_options.get("shots") or backend.options.shots
    raw_result = estimator(
        circuits=circuits_indices,
        observables=observables_indices,
        parameters=parameters_values,
        **run_options)

    result = asdict(raw_result)
    for metadata in result["metadata"]:
        metadata["shots"] = shots

    return result
