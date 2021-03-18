import numpy as np

from qiskit import Aer
from qiskit.circuit import ParameterVector
from qiskit.opflow import StateFn, CircuitSampler, PauliExpectation

from generalized_spsa import SPSA


def get_objective(program_params, options):
    """Get the objective function."""

    problem = program_params['problem']

    if problem['type'] != 'energy':
        raise ValueError(f'Unsupported problem type {problem["type"]}. Available: energy')

    return _get_energy_objective(program_params, options)


def _get_energy_objective(program_params, options):
    """Get a handle for an objective function to compute the expected value, given parameters."""
    # problem parameters
    problem = program_params['problem']
    observable = problem['H']

    # circuit parameters
    circuit = program_params['circuit']
    parameters = get_circuit_parameters(circuit)
    ansatz = StateFn(circuit)
    expr = ~StateFn(observable) @ ansatz

    # backend info
    backend = _get_backend(options['backend_name'])
    sampler = CircuitSampler(backend)

    # get expectation converter to speedup computations or make them suitable for hardware
    expectation = PauliExpectation()
    expr = expectation.convert(expr)

    def objective_function(x):
        params = dict(zip(parameters, x))
        res = np.real(sampler.convert(expr, params=params).eval())
        return res

    return objective_function


def _get_backend(name):
    """Get the backend by name."""

    if name == 'qasm_simulator':
        return Aer.get_backend('qasm_simulator')

    raise ValueError(f'Unsupported backend {name}.')


def get_overlap(ansatz, parameters, sampler=None, expectation=None):
    """Get the Fubini-Study metric for this application."""
    left_params = ParameterVector('l', ansatz.primitive.num_parameters)
    right_params = ParameterVector('r', ansatz.primitive.num_parameters)
    left = ansatz.assign_parameters(dict(zip(parameters, left_params)))
    right = ansatz.assign_parameters(dict(zip(parameters, right_params)))
    expr = ~left @ right
    params = left_params[:] + right_params[:]

    if sampler is None:
        def overlap(x, y):
            bound = expr.bind_parameters({left_params: x, right_params: y})
            return -0.5 * np.abs(bound.eval()).real ** 2

    else:
        if expectation is not None:
            expr = expectation.convert(expr)

        def overlap(x, y):
            param_binds = dict(zip(params, x.tolist() + y.tolist()))
            overlap = sampler.convert(expr, params=param_binds).eval()
            return -0.5 * np.abs(overlap).real ** 2

    return overlap


def get_circuit_parameters(circuit):
    """Get the circuit parameters in a unique order.

    Not necessary with the current Terra dev version, just here for compatibility with older
    versions.
    """
    return sorted(circuit.parameters, key=lambda p: p.name)


def get_optimizer(program_params, callback=None):
    """Get an instance of the optimizer, provided with the program parameters."""
    name = program_params['optimizer']
    if name not in ['SPSA', 'QN-SPSA']:
        raise ValueError(f'Unsupported optimizer: {name}. Available: SPSA, QN-SPSA')

    # construct default parameters
    settings = {'learning_rate': 0.01,
                'perturbation': 0.01,
                'callback': callback}

    if name == 'QN-SPSA':
        circuit = program_params['circuit']
        parameters = get_circuit_parameters(circuit)
        overlap = get_overlap(StateFn(circuit), parameters)

        settings.update({'spd_bias': 0.001,
                         'preconditioner': overlap})

    # update with the parameters the user set
    settings.update(program_params['optimizer_params'])

    # build the optimizer and return
    spsa = SPSA(**settings)
    return spsa
