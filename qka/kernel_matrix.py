import numpy as np
import itertools
from qiskit.compiler import transpile
from qiskit.compiler import assemble


class KernelMatrix:
    """Build the kernel matrix from a quantum feature map."""

    def __init__(self, feature_map, backend):
        """
        Args:
            feature_map (int): the feature map object
            backend (Backend): the backend instance
        """

        self._feature_map = feature_map
        self._feature_map_circuit = self._feature_map.construct_circuit # the feature map circuit
        self._backend = backend
        self.results = {} # store the results object (program_data)


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