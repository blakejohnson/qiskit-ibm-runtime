from qiskit.quantum_info.primitives.estimator.pauli_estimator import PauliEstimator


def main(
        backend,
        user_messenger,
        circuits,
        observables,
        transpile_options=None,
        parameters=None,
        grouping=None,
        run_options=None
):
    estimator = PauliEstimator(
        circuits=circuits,
        observables=observables,
        backend=backend,
    )
    if transpile_options:
        estimator.set_transpile_options(**transpile_options)
    run_options = run_options or {}
    shots = run_options.get("shots") or backend.options.shots
    raw_result = estimator.run(parameters=parameters, grouping=grouping, **run_options)
    # Select only fields we want, in case more are added in the future.
    result = {
        "values": raw_result.values,
        "variances": raw_result.variances,
        "shots": shots
    }
    return result
