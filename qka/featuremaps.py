import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, extensions


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
        print('entangler map: {}'.format(self._entangler_map))


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




class FeatureMapGaussian:
    """Mapping data with the "Gaussian" feature map.

    Data is encoded in layers of single-qubit R_x rotations, interleaved with
    entangling blocks (Z and ZZ terms) parametrized by kernel parameters, lambda.

    In the limit of zero entanglement, this kernel matrix is equivalent to the
    RBF, or Gaussian, kernel.

    """

    def __init__(self, feature_dimension, depth=2, copies=1, sigma=1, entangler_map=None):
        """
        Args:
            feature_dimension (int): number of features
            depth (int): the number of repeated layers of the feature map
            copies (int): the number of tensor product copies of the feature map
            sigma (float): the standard deviation, or width, of the Gaussian
            entangler_map (list[list]): describe the connectivity of qubits, each list describes [source, target], or None for full entanglement.
                                        Note that the order in the list is the order of applying the two-qubit gate.
        """
        self._feature_dimension = feature_dimension
        self._num_qubits = self._feature_dimension = feature_dimension
        self._depth = depth
        self._copies = copies
        self._sigma = sigma
        self._name = 'gaussian'

        if entangler_map is None:
            self._entangler_map = [[i, j] for i in range(self._feature_dimension) for j in range(i + 1, self._feature_dimension)]
        else:
            self._entangler_map = entangler_map

        self._num_parameters = (self._num_qubits + len(self._entangler_map)) * self._depth # Z and ZZ terms are parametrized
        print('entangler map: {}'.format(self._entangler_map))



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

        if len(x) != self._num_qubits:
            raise ValueError('The feature dimension, {}, does not match length of input vector, {}!'.format(self._num_qubits, len(x)))

        if parameters is not None:
            if len(parameters) != self._num_parameters:
                raise ValueError('The number of feature map parameters has to be {}!'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')

        circuit=QuantumCircuit(q, name=name)

        param_idx = 0
        for layer in range(self._depth):
            for i in range(self._num_qubits):
                circuit.rx(2 * x[i] / (np.sqrt(2 * self._copies) * self._sigma * self._depth), q[i])
                circuit.u1(2 * parameters[param_idx], q[i])
                param_idx += 1

            for source, target in self._entangler_map:
                circuit.cx(q[source], q[target])
                circuit.u1(2 * parameters[param_idx], q[target])
                circuit.cx(q[source], q[target])
                param_idx += 1

        if inverse:
            return circuit.inverse()
        else:
            return circuit
