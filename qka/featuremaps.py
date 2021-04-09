import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


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
