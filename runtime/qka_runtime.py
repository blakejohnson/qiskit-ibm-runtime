import numpy as np
import itertools
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.compiler import assemble
from cvxopt import matrix, solvers

import sys
import json
# from ntc_provider.programruntime import QuantumProgramProvider

from qiskit import Aer


class FeatureMapForrelation:
    """Mapping data with the forrelation feature map from
    Havlicek et al. https://www.nature.com/articles/s41586-019-0980-2.

    Data is encoded in entangling blocks (Z and ZZ terms) interleaved with
    layers of either phase gates parametrized by kernel parameters, lambda,
    or of Hadamard gates.
    """

    def __init__(self, feature_dimension, depth=2, entangler_map=None):
        """
        Args:
            feature_dimension (int): number of features
            depth (int): number of repeated layers in the feature map
            entangler_map (list[list]): connectivity of qubits with a list of [source, target], or None for full entanglement.
                                        Note that the order in the list is the order of applying the two-qubit gate.
        """

        self._feature_dimension = feature_dimension
        self._num_qubits = self._feature_dimension = feature_dimension
        self._depth = depth
        self._copies = 1

        if entangler_map is None:
            self._entangler_map = [[i, j] for i in range(self._feature_dimension) for j in range(i + 1, self._feature_dimension)]
        else:
            self._entangler_map = entangler_map

        self._num_parameters = self._num_qubits * self._depth

    def construct_circuit(self, x=None, parameters=None, q=None, inverse=False, name=None):
        """Construct the feature map circuit.

        Args:
            x (numpy.ndarray): data vector of size feature_dimension
            parameters (numpy.ndarray): optional parameters in feature map
            q (QauntumRegister): the QuantumRegister object for the circuit
            inverse (bool): whether or not to invert the circuit

        Returns:
            QuantumCircuit: a quantum circuit transforming data x
        """

        if parameters is not None:
            if len(parameters) != self._num_parameters:
                raise ValueError('The number of feature map parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')

        circuit=QuantumCircuit(q, name=name)

        param_idx = 0
        for layer in range(self._depth):
            for i in range(self._num_qubits):
                if parameters is not None:
                    circuit.p(np.pi + 2 * np.pi * parameters[param_idx], q[i])
                    circuit.h(q[i])
                    param_idx += 1
                else:
                    circuit.h(q[i])
                circuit.p(2 * x[i], q[i])
            for source, target in self._entangler_map:
                circuit.cx(q[source], q[target])
                circuit.p(2 * (np.pi - x[source]) * (np.pi - x[target]), q[target])
                circuit.cx(q[source], q[target])

        if inverse:
            return circuit.inverse()
        else:
            return circuit


class KernelMatrix:
    """Build the kernel matrix from a quantum feature map."""

    def __init__(self, feature_map, backend):
        """
        Args:
            feature_map: the feature map object
            backend (Backend): the backend instance
        """

        self._feature_map = feature_map
        self._feature_map_circuit = self._feature_map.construct_circuit  # the feature map circuit
        self._backend = backend
        self.results = {}  # store the results object (program_data)

    def construct_kernel_matrix(self, x1_vec, x2_vec, parameters=None):
        """Create the kernel matrix for a given feature map and input data.

        With the qasm simulator or real backends, compute order 'n^2'
        states Phi^dag(y)Phi(x)|0> for input vectors x and y.

        Args:
            x1_vec (numpy.ndarray): NxD array of training data or test data, where N is the number of samples and D is the feature dimension
            x2_vec (numpy.ndarray): MxD array of training data or support vectors, where M is the number of samples and D is the feature dimension
            parameters (numpy.ndarray): optional parameters in feature map

        Returns:
           mat (numpy.ndarray): the kernel matrix
       """

        is_identical = False
        if np.array_equal(x1_vec, x2_vec):
            is_identical = True

        experiments = []

        measurement_basis = '0' * np.shape(x1_vec)[1]

        if is_identical:

            my_product_list = list(itertools.combinations(range(len(x1_vec)), 2)) # all pairwise combos of datapoint indices
            for index_1, index_2 in my_product_list:

                circuit = self._feature_map_circuit(x=x1_vec[index_1], parameters=parameters, name='{}_{}'.format(index_1, index_2))
                circuit += self._feature_map_circuit(x=x1_vec[index_2], parameters=parameters, inverse=True)
                circuit.measure_all()
                experiments.append(circuit)

            experiments = transpile(experiments, backend=self._backend)
            qobj = assemble(experiments, backend=self._backend, shots=1024)
            program_data = self._backend.run(qobj).result()

            self.results['program_data'] = program_data

            mat = np.eye(len(x1_vec), len(x1_vec))  # kernel matrix element on the diagonal is always 1
            for experiment, [index_1, index_2] in enumerate(my_product_list):

                counts = program_data.get_counts(experiment = experiment)
                shots = sum(counts.values())

                mat[index_1][index_2] = counts.get(measurement_basis, 0) / shots # kernel matrix element is the probability of measuring all 0s
                mat[index_2][index_1] = mat[index_1][index_2] # kernel matrix is symmetric

            return mat ** self._feature_map._copies

        else:

            for index_1, point_1 in enumerate(x1_vec):
                for index_2, point_2 in enumerate(x2_vec):

                    circuit = self._feature_map_circuit(x=point_1, parameters=parameters, name='{}_{}'.format(index_1, index_2))
                    circuit += self._feature_map_circuit(x=point_2, parameters=parameters, inverse=True)
                    circuit.measure_all()
                    experiments.append(circuit)

            experiments = transpile(experiments, backend=self._backend)
            qobj = assemble(experiments, backend=self._backend, shots=1024)
            program_data = self._backend.run(qobj).result()

            self.results['program_data'] = program_data

            mat = np.zeros((len(x1_vec), len(x2_vec)))
            i = 0
            for index_1, _ in enumerate(x1_vec):
                for index_2, _ in enumerate(x2_vec):

                    counts = program_data.get_counts(experiment = i)
                    shots = sum(counts.values())

                    mat[index_1][index_2] = counts.get(measurement_basis, 0) / shots
                    i += 1

            return mat ** self._feature_map._copies


def cvxopt_solver(K, y, C, max_iters=10000, show_progress=False):
    """Convex optimization of SVM objective using cvxopt.

    Args:
        K (numpy.ndarray): nxn kernel (Gram) matrix
        y (numpy.ndarray): nx1 vector of labels +/-1
        C (float): soft-margin penalty
        max_iters (int): maximum iterations for the solver
        show_progress (bool): print progress of solver

    Returns;
        ret (dict): results from the solver
    """

    if y.ndim == 1:
        y = y[:, np.newaxis]
    H = np.outer(y, y) * K
    f = -np.ones(y.shape)

    n = K.shape[1] # number of training points

    y = y.astype('float')

    P = matrix(H)                                                 # (nxn) yy^T * K, element-wise multiplication
    q = matrix(f)                                                 # (nx1) -ones
    G = matrix(np.vstack((-np.eye((n)), np.eye((n)))))            # for soft-margin, (2nxn) matrix with -identity (+identity) in top (bottom) half
    h = matrix(np.vstack((np.zeros((n,1)), np.ones((n,1)) * C)))  # for soft-margin, (2nx1) vector with 0's (C's) in top (bottom) half
    A = matrix(y, y.T.shape)                                      # (1xn) y
    b = matrix(np.zeros(1), (1, 1))                               # (1x1) zero

    solvers.options['maxiters'] = max_iters
    solvers.options['show_progress'] = show_progress

    ret = solvers.qp(P, q, G, h, A, b, kktsolver='ldl')

    return ret


def align_kernel(backend, feature_map, data, labels, lambda_plus, lambda_minus, C=1):
    """Align the quantum kernel.

    Uses SPSA for minimization over kernel parameters (lambdas) and
    convex optimization for maximization over lagrange multipliers (alpha):

    min_lambda max_alpha 1^T * alpha - (1/2) * alpha^T * Y * K_lambda * Y * alpha

    Args:
        backend (Backend): Backend used to run the circuits
        feature_map (FeatureMapForrelation): Feature map
        data (numpy.ndarray): NxD array of training data, where N is the number of samples and D is the feature dimension
        labels (numpy.ndarray): Nx1 array of +/-1 labels of the N training samples
        lambda_plus (numpy.ndarray): (+) kernel parameters
        lambda_minus (numpy.ndarray): (-) kernel parameters
        C (float): penalty parameter for the soft-margin support vector machine

    Returns:
        cost_plus (float): SVM objective function evaluated at (alpha_+, lambda_+)
        cost_minus (float): SVM objective function evaluated at (alpha_-, lambda_-)
    """

    kernel_matrix = KernelMatrix(feature_map=feature_map, backend=backend)

    # (STEP 2 OF PSEUDOCODE)
    # Evaluate kernel matrix for the (+) and (-) kernel parameters.

    kernel_plus = kernel_matrix.construct_kernel_matrix(x1_vec=data, x2_vec=data, parameters=lambda_plus)
    kernel_minus = kernel_matrix.construct_kernel_matrix(x1_vec=data, x2_vec=data, parameters=lambda_minus)

    # (STEP 3 OF PSEUDOCODE)
    # Maximize SVM objective function over
    # support vectors in the (+) and (-) directions.

    ret_plus = cvxopt_solver(K=kernel_plus, y=labels, C=C)
    cost_plus = -1 * ret_plus['primal objective']

    ret_minus = cvxopt_solver(K=kernel_minus, y=labels, C=C)
    cost_minus = -1 * ret_minus['primal objective']

    return cost_plus, cost_minus


class NumpyDecoder(json.JSONDecoder):
    """JSON Decoder for Numpy arrays and complex numbers."""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if 'type' in obj:
            if obj['type'] == 'complex':
                val = obj['value']
                return val[0] + 1j * val[1]
            if obj['type'] == 'array':
                return np.array(obj['value'])
        return obj


def main(backend, *args, **kwargs):
    """Entry function."""
    # Reconstruct the feature map object.
    feature_map = kwargs.get('feature_map')
    kwargs['feature_map'] = FeatureMapForrelation(**feature_map)
    kwargs['backend'] = backend
    results = align_kernel(**kwargs)
    print(json.dumps({'results': results}))


if __name__ == '__main__':
    # provider = QuantumProgramProvider()
    # backend = provider.backends()[0]

    # the code currently uses Aer instead of runtime provider
    backend = Aer.get_backend('qasm_simulator')
    user_params = {}
    if len(sys.argv) > 1:
        # If there are user parameters.
        user_params = json.loads(sys.argv[1], cls=NumpyDecoder)
    main(backend, **user_params)