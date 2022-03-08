from dataclasses import asdict

from qiskit.quantum_info.primitives.estimator.pauli_estimator import PauliEstimator


def main(
        backend,
        user_messenger,
        circuits,
        observables,
        parameters=None,
        circuit_indices=None,
        observable_indices=None,
        parameter_values=None,
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
        circuit_indices: List of circuit indices.
        observable_indices: List of observable indices.
        parameter_values: Concrete parameters to be bound.
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
        circuits=circuit_indices,
        observables=observable_indices,
        parameters=parameter_values,
        **run_options)

    result = asdict(raw_result)
    for metadata in result["metadata"]:
        metadata["shots"] = shots

    return result
