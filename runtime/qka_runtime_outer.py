import numpy as np
from numpy.random import RandomState
import itertools
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from cvxopt import matrix, solvers
from typing import Any

import sys
import json
# from ntc_provider.programruntime import QuantumProgramProvider

from qiskit import Aer

class FeatureMapACME:
    """Mapping data with the feature map.
    """

    def __init__(self, feature_dimension, entangler_map=None):
        """
        Args:
            feature_dimension (int): number of features
            entangler_map (list[list]): connectivity of qubits with a list of [source, target], or None for full entanglement.
                                        Note that the order in the list is the order of applying the two-qubit gate.
        """
        self._feature_dimension = feature_dimension
        self._num_qubits = self._feature_dimension = feature_dimension

        if entangler_map is None:
            self._entangler_map = [[i, j] for i in range(self._feature_dimension) for j in range(i + 1, self._feature_dimension)]
        else:
            self._entangler_map = entangler_map

        self._num_parameters = self._num_qubits


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
            if isinstance(parameters, int) or isinstance(parameters, float):
                raise ValueError('Parameters must be a list.')
            elif (len(parameters) == 1):
                parameters = parameters * np.ones(self._num_qubits)
            else:
                if len(parameters) != self._num_parameters:
                    raise ValueError('The number of feature map parameters must be {}.'.format(self._num_parameters))

        if len(x) != 2*self._num_qubits:
            raise ValueError('The input vector must be of length {}.'.format(2*self._num_qubits))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')

        circuit=QuantumCircuit(q, name=name)

        for i in range(self._num_qubits):
            circuit.ry(-parameters[i], q[i])

        for source, target in self._entangler_map:
            circuit.cz(q[source], q[target])

        for i in range(self._num_qubits):
            circuit.rz(-2*x[2*i+1], q[i])
            circuit.rx(-2*x[2*i], q[i])

        if inverse:
            return circuit.inverse()
        else:
            return circuit

    def to_dict(self):
        return {'feature_dimension': self._feature_dimension,
                'entangler_map': self._entangler_map}


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

        measurement_basis = '0' * self._feature_map._num_qubits

        if is_identical:

            my_product_list = list(itertools.combinations(range(len(x1_vec)), 2)) # all pairwise combos of datapoint indices
            for index_1, index_2 in my_product_list:

                circuit_1 = self._feature_map_circuit(x=x1_vec[index_1], parameters=parameters, name='{}_{}'.format(index_1, index_2))
                circuit_2 = self._feature_map_circuit(x=x1_vec[index_2], parameters=parameters, inverse=True)
                circuit = circuit_1.compose(circuit_2)
                circuit.measure_all()
                experiments.append(circuit)

            experiments = transpile(experiments, backend=self._backend)
            program_data = self._backend.run(experiments, shots=8192).result()

            self.results['program_data'] = program_data

            mat = np.eye(len(x1_vec), len(x1_vec))  # kernel matrix element on the diagonal is always 1
            for experiment, [index_1, index_2] in enumerate(my_product_list):

                counts = program_data.get_counts(experiment = experiment)
                shots = sum(counts.values())

                mat[index_1][index_2] = counts.get(measurement_basis, 0) / shots # kernel matrix element is the probability of measuring all 0s
                mat[index_2][index_1] = mat[index_1][index_2] # kernel matrix is symmetric

            return mat

        else:

            for index_1, point_1 in enumerate(x1_vec):
                for index_2, point_2 in enumerate(x2_vec):

                    circuit_1 = self._feature_map_circuit(x=point_1, parameters=parameters, name='{}_{}'.format(index_1, index_2))
                    circuit_2 = self._feature_map_circuit(x=point_2, parameters=parameters, inverse=True)
                    circuit = circuit_1.compose(circuit_2)
                    circuit.measure_all()
                    experiments.append(circuit)

            experiments = transpile(experiments, backend=self._backend)
            program_data = self._backend.run(experiments, shots=8192).result()

            self.results['program_data'] = program_data

            mat = np.zeros((len(x1_vec), len(x2_vec)))
            i = 0
            for index_1, _ in enumerate(x1_vec):
                for index_2, _ in enumerate(x2_vec):

                    counts = program_data.get_counts(experiment = i)
                    shots = sum(counts.values())

                    mat[index_1][index_2] = counts.get(measurement_basis, 0) / shots
                    i += 1

            return mat


class QKA:
    """The quantum kernel alignment algorithm."""

    def __init__(self, feature_map, backend):
        """Constructor.

        Args:
            feature_map (partial obj): the quantum feature map object
            backend (Backend): the backend instance
        """

        self.feature_map = feature_map
        self.feature_map_circuit = self.feature_map.construct_circuit # the feature map circuit not yet evaluated with input arguments
        self.backend = backend
        self.num_parameters = self.feature_map._num_parameters  # number of parameters (lambdas) in the feature map

        self.result = {}

        self.kernel_matrix = KernelMatrix(feature_map=self.feature_map, backend=self.backend)

    def SPSA_parameters(self):
        """Return array of precomputed SPSA parameters.

        Returns:
            SPSA_params (numpy.ndarray): [a, c, alpha, gamma, A]

        The i-th optimization step, i>=0, the parameters evolve as

            a_i = a / (i + 1 + A) ** alpha,
            c_i = c / (i + 1) ** gamma,

        for fixed coefficents a, c, alpha, gamma, A.

        Default Qiskit values are:
        SPSA_params = [2*np.pi*0.1, 0.1, 0.602, 0.101, 0]
        """

        SPSA_params = np.zeros((5))
        SPSA_params[0] = 0.05              # a
        SPSA_params[1] = 0.1               # c
        SPSA_params[2] = 0.602             # alpha  (alpha range [0.5 - 1.0])
        SPSA_params[3] = 0.101             # gamma  (gamma range [0.0 - 0.5])
        SPSA_params[4] = 0                 # A

        return SPSA_params

    def cvxopt_solver(self, K, y, C, max_iters=10000, show_progress=False):
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

    def spsa_step_one(self, lambdas, spsa_params, count):
        """Evaluate +/- perturbations of kernel parameters (lambdas).

        Args:
            lambdas (numpy.ndarray): kernel parameters at step 'count' in SPSA optimization loop
            spsa_params (numpy.ndarray): SPSA parameters
            count (int): the current step in the SPSA optimization loop
        Returns:
            lambda_plus (numpy.ndarray): kernel parameters in + direction
            lambda_minus (numpy.ndarray): kernel parameters in - direction
            delta (numpy.ndarray): random vector with elements {-1,1}
        """

        prng = RandomState(count)

        c_spsa = float(spsa_params[1])/np.power(count+1, spsa_params[3])
        delta = 2*prng.randint(0, 2, size=np.shape(lambdas)[0]) - 1

        lambda_plus = lambdas + c_spsa * delta
        lambda_minus = lambdas - c_spsa * delta

        return lambda_plus, lambda_minus, delta

    def spsa_step_two(self, cost_plus, cost_minus, lambdas, spsa_params, delta, count):
        """Evaluate one iteration of SPSA on SVM objective function F and
        return updated kernel parameters.

        F(alpha, lambda) = 1^T * alpha - (1/2) * alpha^T * Y * K * Y * alpha

        Args:
            cost_plus (float): objective function F(alpha_+, lambda_+)
            cost_minus (float): objective function F(alpha_-, lambda_-)
            lambdas (numpy.ndarray): kernel parameters at step 'count' in SPSA optimization loop
            spsa_params (numpy.ndarray): SPSA parameters
            delta (numpy.ndarray): random vector with elements {-1,1}
            count(int): the current step in the SPSA optimization loop

        Returns:
            cost_final (float): estimate of updated SVM objective function F using average of F(alpha_+, lambda_+) and F(alpha_-, lambda_-)
            lambdas_new (numpy.ndarray): updated values of the kernel parameters after one SPSA optimization step
        """

        a_spsa = float(spsa_params[0])/np.power(count+1+spsa_params[4], spsa_params[2])
        c_spsa = float(spsa_params[1])/np.power(count+1, spsa_params[3])

        g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)

        lambdas_new = lambdas - a_spsa * g_spsa
        lambdas_new = lambdas_new.flatten()

        cost_final = (cost_plus + cost_minus) / 2

        return cost_final, lambdas_new

    def align_kernel(self, data, labels, initial_kernel_parameters=None, maxiters=10, C=1):
        """Align the quantum kernel.

        Uses SPSA for minimization over kernel parameters (lambdas) and
        convex optimization for maximization over lagrange multipliers (alpha):

        min_lambda max_alpha 1^T * alpha - (1/2) * alpha^T * Y * K_lambda * Y * alpha

        Args:
            data (numpy.ndarray): NxD array of training data, where N is the number of samples and D is the feature dimension
            labels (numpy.ndarray): Nx1 array of +/-1 labels of the N training samples
            initial_kernel_parameters (numpy.ndarray): Initial parameters of the quantum kernel
            maxiters (int): number of SPSA optimization steps
            C (float): penalty parameter for the soft-margin support vector machine

        Returns:
            result (dict): the results of kernel alignment
        """

        if initial_kernel_parameters is not None:
            lambdas = initial_kernel_parameters
        else:
            lambdas = np.random.uniform(-1.0, 1.0, size=(self.num_parameters))

        # Pre-computed spsa parameters:
        spsa_params = self.SPSA_parameters()

        # Save data at each SPSA run in the following lists:
        lambda_save = []       # updated kernel parameters after each spsa step
        cost_final_save = []   # avgerage cost at each spsa step
        cost_plus_save = []    # (+) cost at each spsa step
        cost_minus_save = []   # (-) cost at each spsa step
        program_data = []


        # #####################
        # Start the alignment:

        for count in range(maxiters):

            # (STEP 1 OF PSEUDOCODE)
            # First stage of SPSA optimization.

            lambda_plus, lambda_minus, delta = self.spsa_step_one(lambdas=lambdas, spsa_params=spsa_params, count=count)

            # (STEP 2 OF PSEUDOCODE)
            # Evaluate kernel matrix for the (+) and (-) kernel parameters.

            kernel_plus = self.kernel_matrix.construct_kernel_matrix(x1_vec=data, x2_vec=data, parameters=lambda_plus)
            kernel_minus = self.kernel_matrix.construct_kernel_matrix(x1_vec=data, x2_vec=data, parameters=lambda_minus)

            # (STEP 3 OF PSEUDOCODE)
            # Maximize SVM objective function over
            # support vectors in the (+) and (-) directions.

            ret_plus = self.cvxopt_solver(K=kernel_plus, y=labels, C=C)
            cost_plus = -1 * ret_plus['primal objective']

            ret_minus = self.cvxopt_solver(K=kernel_minus, y=labels, C=C)
            cost_minus = -1 * ret_minus['primal objective']

            # (STEP 4 OF PSEUDOCODE)
            # Second stage of SPSA optimization:
            # (one iteration of SPSA on SVM objective function F
            #  and return updated kernel parameters).

            cost_final, lambda_best = self.spsa_step_two(cost_plus=cost_plus, cost_minus=cost_minus,
                                                         lambdas=lambdas, spsa_params=spsa_params, delta=delta, count=count)

            lambdas = lambda_best # updated kernel parameters

            intrim_result = {'cost': cost_final,
                             'kernel_parameters': lambdas}
            # intrim_result = {'cost': cost_final,
            #                  'lambda': lambdas, 'cost_plus': cost_plus,
            #                  'cost_minus': cost_minus, 'cost_final': cost_final}
            post_interim_result(intrim_result)

            lambda_save.append(lambdas)
            cost_plus_save.append(cost_plus)
            cost_minus_save.append(cost_minus)
            cost_final_save.append(cost_final)

            program_data.append(self.kernel_matrix.results)


        # Evaluate aligned kernel matrix with optimized set of parameters averaged over last 10% of SPSA steps:
        num_last_lambdas = int(len(lambda_save) * 0.10)
        if num_last_lambdas > 0:
            last_lambdas = np.array(lambda_save)[-num_last_lambdas:, :] # the last 10% of lambdas
            lambdas = np.sum(last_lambdas, axis=0) / num_last_lambdas   # average over last 10% lambdas
        else:
            lambdas = np.array(lambda_save)[-1,:]

        kernel_best = self.kernel_matrix.construct_kernel_matrix(x1_vec=data, x2_vec=data, parameters=lambdas)

        self.result['aligned_kernel_parameters'] = lambdas
        self.result['aligned_kernel_matrix'] = kernel_best

        # self.result['all_kernel_parameters'] = lambda_save
        # self.result['all_final_cost_evaluations'] = cost_final_save
        # self.result['all_positive_cost_evaluations'] = cost_plus_save
        # self.result['all_negative_cost_evaluations'] = cost_minus_save

        # self.result['program_data'] = program_data

        return self.result


def post_interim_result(text):
    print(json.dumps({'post': text}, cls=NumpyEncoder))


class NumpyEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy arrays and complex numbers."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, 'tolist'):
            return {'type': 'array', 'value': obj.tolist()}
        if isinstance(obj, complex):
            return {'type': 'complex', 'value': [obj.real, obj.imag]}
        return super().default(obj)


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
    feature_map = kwargs.pop('feature_map')
    fm = FeatureMapACME(**feature_map)
    qka = QKA(feature_map=fm, backend=backend)
    qka_results = qka.align_kernel(**kwargs)

    print(json.dumps({'results': qka_results}, cls=NumpyEncoder))


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
