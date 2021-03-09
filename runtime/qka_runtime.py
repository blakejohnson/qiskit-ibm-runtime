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


class FeatureMapQuantumControl:
    """Mapping data with the "quantum control" feature map from
    Havlicek et al. https://www.nature.com/articles/s41586-019-0980-2.

    Data is encoded in entangling blocks (Z and ZZ terms) interleaved with
    layers of either (1) u2 gates parametrized by kernel parameters, lambda,
    or (2) Hadamard gates.

    """

    def __init__(self, feature_dimension, depth=2, entangler_map=None):
        """
        Args:
            feature_dimension (int): number of features
            depth (int): the number of repeated circuits
            entangler_map (list[list]): describe the connectivity of qubits, each list describes [source, target], or None for full entanglement.
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

        self._num_parameters = self._num_qubits * self._depth # only single-qubit layers are parametrized
        # print('entangler map: {}'.format(self._entangler_map))

    def construct_circuit(self, x=None, parameters=None, q=None, inverse=False, name=None):
        """Construct the feature map circuit.

        Args:
            x (numpy.ndarray): 1-D to-be-transformed data.
            parameters (numpy.ndarray): optional parameters in feature map.
            q (QauntumRegister): the QuantumRegister object for the circuit.
            inverse (bool): whether or not to invert the circuit.

        Returns:
            QuantumCircuit: a quantum circuit transforming data x.
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
                    circuit.u2(0, 2 * np.pi * parameters[param_idx], q[i])
                    # circuit.u2(0, 2 * np.pi * parameters[self._num_qubits * layer + i], q[i])
                    param_idx += 1
                else:
                    circuit.h(q[i])
                circuit.u1(2 * x[i], q[i])
            for source, target in self._entangler_map:
                circuit.cx(q[source], q[target])
                circuit.u1(2 * (np.pi - x[source]) * (np.pi - x[target]), q[target])
                circuit.cx(q[source], q[target])

        if inverse:
            return circuit.inverse()
        else:
            return circuit


class KernelMatrix:
    """Build the kernel matrix from a quantum feature map.
    """

    def __init__(self, feature_map, backend):
        """
        Args:
            feature_map: feature map object
            backend (Backend): the backend
        """
        self._feature_map = feature_map
        self._feature_map_circuit = self._feature_map.construct_circuit  # the feature map circuit
        self._backend = backend
        self.results = {}  # store the results object (program_data)

    def construct_kernel_matrix(self, x1_vec, x2_vec, parameters=None):
        """Create the kernel matrix for a given feature map and input data.

        If using statevector simulator,
        compute 'n' states Phi(x)|0>,
        and then compute inner products classically.

        If using qasm simulator or backends,
        compute order 'n^2' states Phi^dag(y)Phi(x)|0>.

                Args:
                    x1_vec (numpy.ndarray): first dataset (can be training or test data)
                    x2_vec (numpy.ndarray): second dataset (can be training or support vectors)
                    parameters (numpy.ndarray): optional parameters in feature map.

                Returns:
                   mat (array): the kernel matrix
               """

        is_identical = False
        if np.array_equal(x1_vec, x2_vec):
            is_identical = True

        experiments = []  # list of QuantumCircuits to execute

        measurement_basis = '0' * np.shape(x1_vec)[1]

        if is_identical:

            my_product_list = list(itertools.combinations(range(len(x1_vec)), 2)) # all pairwise combos of datapoint indices
            for index_1, index_2 in my_product_list:

                circuit = self._feature_map_circuit(x=x1_vec[index_1], parameters=parameters, name='{}_{}'.format(index_1, index_2))
                circuit += self._feature_map_circuit(x=x1_vec[index_2], parameters=parameters, inverse=True)
                circuit.measure_all() # add measurement to all qubits
                experiments.append(circuit)

            experiments = transpile(experiments, backend=self._backend)
            qobj = assemble(experiments, backend=self._backend, shots=1024)
            program_data = self._backend.run(qobj).result()

            self.results['program_data'] = program_data

            mat = np.eye(len(x1_vec), len(x1_vec))  # kernel matrix element on the diagonal is always 1: point*point=|point|^2
            for experiment, [index_1, index_2] in enumerate(my_product_list):

                counts = program_data.get_counts(experiment = experiment) # dictionary of counts
                shots = sum(counts.values())

                mat[index_1][index_2] = counts.get(measurement_basis, 0) / shots # kernel matrix element is the probability of measuring all 0s
                mat[index_2][index_1] = mat[index_1][index_2] # kernel matrix is symmetric

            return mat ** self._feature_map._copies

        else:

            for index_1, point_1 in enumerate(x1_vec):
                for index_2, point_2 in enumerate(x2_vec):

                    circuit = self._feature_map_circuit(x=point_1, parameters=parameters, name='{}_{}'.format(index_1, index_2))
                    circuit += self._feature_map_circuit(x=point_2, parameters=parameters, inverse=True)
                    circuit.measure_all() # add measurement to all qubits
                    experiments.append(circuit)

            experiments = transpile(experiments, backend=self._backend)
            qobj = assemble(experiments, backend=self._backend, shots=1024)
            program_data = self._backend.run(qobj).result()

            self.results['program_data'] = program_data

            mat = np.zeros((len(x1_vec), len(x2_vec)))
            i = 0
            for index_1, _ in enumerate(x1_vec):
                for index_2, _ in enumerate(x2_vec):

                    counts = program_data.get_counts(experiment = i) # dictionary of counts
                    shots = sum(counts.values())

                    mat[index_1][index_2] = counts.get(measurement_basis, 0) / shots
                    i += 1

            return mat ** self._feature_map._copies


def gradient_ascent_cvxopt(K, y, C, max_iters=10000, show_progress=False):
    """Convex optimization of SVM objective using cvxopt.
    Args:
        K: nxn kernel (Gram) matrix
        y: nx1 vector of labels +/-1
        C: Box parameter (aka regularization parameter / margin penalty)

    Returns;
        alpha: optimized variables in SVM objective
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

    # optional data from 'ret':
    # alpha = np.asarray(ret['x']) # optimized alphas (support vector weights)
    # obj = ret['primal objective'] # value of the primal obj (1/2)*x'*P*x + q'*x

    return ret


def align_kernel(backend, feature_map, data, labels, lambda_plus, lambda_minus, C=1):
    """Align the quantum kernel.

    Uses SPSA for minimization wrt kernel parameters (lambda) and
    gradient ascent for maximization wrt support vector weights (alpha):

    min max cost_function

    Args:
        backend (Backend): Backend used to run the circuits.
        feature_map (FeatureMapQuantumControl): Feature map.
        data (numpy.ndarray): NxD array, where N is the number of data points, D is the feature dimension.
        labels (numpy.ndarray): Nx1 array of +/-1, where N is the number of data points
        lambda_plus (numpy.ndarray): (+) kernel parameter
        lambda_minus (numpy.ndarray): (-) kernel parameter
        C (float): penalty parameter for soft-margin

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

    ret_plus = gradient_ascent_cvxopt(K=kernel_plus, y=labels, C=C)
    cost_plus = -1 * ret_plus['primal objective']

    ret_minus = gradient_ascent_cvxopt(K=kernel_minus, y=labels, C=C)
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
    kwargs['feature_map'] = FeatureMapQuantumControl(**feature_map)
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
