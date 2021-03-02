import numpy as np
import itertools
from qiskit import execute


class KernelMatrix:
    """Build the kernel matrix from a quantum feature map.
    """

    def __init__(self, feature_map, quantum_instance):
        """
        Args:
            feature_map (int): feature map object
            quantum_instance (QuantumInstance): the Quantum Instance
        """
        self._feature_map = feature_map
        self._feature_map_circuit = self._feature_map.construct_circuit # the feature map circuit
        self._quantum_instance = quantum_instance
        self.results = {} # store the results object (program_data)


    def construct_kernel_matrix(self, x1_vec, x2_vec, parameters=None):
        """Create the kernel matrix for a given feature map and input data.

        If using statevector simulator,
        compute 'n' states Phi(x)|0>,
        and then compute inner products classically.

        If using qasm simulator or backends,
        compute order 'n^2' states Phi^dag(y)Phi(x)|0>.

                Args:
                    x1_vec (array): first dataset (can be training or test data)
                    x2_vec (array): second dataset (can be training or support vectors)
                    parameters (array): optional parameters in feature map.

                Returns:
                   mat (array): the kernel matrix
               """

        is_identical = False
        if np.array_equal(x1_vec, x2_vec):
            is_identical = True

        experiments = [] # list of QuantumCircuits to execute


        if self._quantum_instance.is_statevector:

            if is_identical:

                for index, point in enumerate(x1_vec):

                    circuit = self._feature_map_circuit(x=point, parameters=parameters, name='{}'.format(index)) # execute only Phi(x)|0>
                    experiments.append(circuit)

                program_data = self._quantum_instance.execute(circuits=experiments)
                self.results['program_data'] = program_data

                my_product_list = list(itertools.combinations(range(len(x1_vec)), 2)) # all pairwise combos of datapoint indices
                mat = np.eye(len(x1_vec), len(x1_vec))  # kernel matrix element on the diagonal is always 1: point*point=|point|^2
                for index_i, index_j in my_product_list:

                    feature_vector_i = program_data.get_statevector(experiment = index_i) # statevector for data point i
                    feature_vector_j = program_data.get_statevector(experiment = index_j) # statevector for data point j

                    inner_product = np.conj(feature_vector_i) @ feature_vector_j # classically compute inner product of feature vectors

                    mat[index_i][index_j] = np.real(inner_product * np.conj(inner_product)) # kernel matrix element is conj square of inner product
                    mat[index_j][index_i] = mat[index_i][index_j] # kernel matrix is symmetric

                return mat ** self._feature_map._copies

            else:

                for index_1, point_1 in enumerate(x1_vec):

                    circuit = self._feature_map_circuit(x=point_1, parameters=parameters, name = 'x1_{}'.format(index_1))
                    experiments.append(circuit)

                for index_2, point_2 in enumerate(x2_vec):

                    circuit = self._feature_map_circuit(x=point_2, parameters=parameters, name = 'x2_{}'.format(index_2))
                    experiments.append(circuit)

                program_data = self._quantum_instance.execute(circuits=experiments)
                self.results['program_data'] = program_data

                mat = np.zeros((len(x1_vec), len(x2_vec)))
                for index_1, _ in enumerate(x1_vec):
                    for index_2, _ in enumerate(x2_vec):

                        feature_vector_1 = program_data.get_statevector(experiment = index_1)
                        feature_vector_2 = program_data.get_statevector(experiment = len(x1_vec) + index_2)

                        inner_product = np.conj(feature_vector_1) @ feature_vector_2 # classically compute inner product of feature vectors

                        mat[index_1][index_2] = np.real(inner_product * np.conj(inner_product)) # kernel matrix element is conj square of inner product

                return mat ** self._feature_map._copies


        else:

            # for qasm simulator or real backend

            measurement_basis = '0' * np.shape(x1_vec)[1]

            if is_identical:

                my_product_list = list(itertools.combinations(range(len(x1_vec)), 2)) # all pairwise combos of datapoint indices
                for index_1, index_2 in my_product_list:

                    circuit = self._feature_map_circuit(x=x1_vec[index_1], parameters=parameters, name='{}_{}'.format(index_1, index_2))
                    circuit += self._feature_map_circuit(x=x1_vec[index_2], parameters=parameters, inverse=True)
                    circuit.measure_all() # add measurement to all qubits
                    experiments.append(circuit)

                program_data = self._quantum_instance.execute(circuits=experiments)
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

                program_data = self._quantum_instance.execute(circuits=experiments)
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
