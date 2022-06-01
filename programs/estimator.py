# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Expectation value class
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
import copy
from itertools import accumulate
import logging
from typing import cast

import numpy as np
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.result import BaseReadoutMitigator, Counts, Result
from qiskit.result.mitigation.utils import str2diag
from qiskit.transpiler import PassManager
import retworkx as rx

logger = logging.getLogger(__name__)


def _grouping(sparse_pauli_operator: SparsePauliOp):
    """Partition a SparsePauliOp into sets of commuting Pauli strings.

    Returns:
        List[SparsePauliOp]: List of SparsePauliOp where each SparsePauliOp contains commutable
            Pauli operators.
    """
    edges = sparse_pauli_operator.paulis._noncommutation_graph()
    graph = rx.PyGraph()
    graph.add_nodes_from(range(sparse_pauli_operator.size))
    graph.add_edges_from_no_data(edges)
    # Keys in coloring_dict are nodes, values are colors
    coloring_dict = rx.graph_greedy_color(graph)
    groups = defaultdict(list)
    for idx, color in coloring_dict.items():
        groups[color].append(idx)
    return [sparse_pauli_operator[group] for group in groups.values()]


SparsePauliOp.grouping = _grouping


def init_observable(observable: BaseOperator | PauliSumOp) -> SparsePauliOp:
    """Initialize observable by converting the input to a :class:`~qiskit.quantum_info.SparsePauliOp`.
    Args:
        observable: The observable.
    Returns:
        The observable as :class:`~qiskit.quantum_info.SparsePauliOp`.
    Raises:
        TypeError: If the observable is a :class:`~qiskit.opflow.PauliSumOp` and has a parameterized
            coefficient.
    """
    if isinstance(observable, SparsePauliOp):
        return observable
    elif isinstance(observable, PauliSumOp):
        if isinstance(observable.coeff, ParameterExpression):
            raise TypeError(
                f"Observable must have numerical coefficient, not {type(observable.coeff)}."
            )
        return observable.coeff * observable.primitive
    elif isinstance(observable, BasePauli):
        return SparsePauliOp(observable)
    elif isinstance(observable, BaseOperator):
        return SparsePauliOp.from_operator(observable)
    else:
        return SparsePauliOp(observable)


class Estimator(BaseEstimator):
    """
    Evaluates expectation value using pauli rotation gates.
    """

    _trans = str.maketrans({"X": "Z", "Y": "Z"})

    def __init__(
        self,
        backend: Backend,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
        parameters: Iterable[Iterable[Parameter]] | None = None,
        readout_mitigator: BaseReadoutMitigator | None = None,
        abelian_grouping: bool = True,
        bound_pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
    ):
        if not isinstance(backend, Backend):
            raise TypeError(f"backend should be BackendV1, not {type(backend)}.")

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        if isinstance(observables, (PauliSumOp, BaseOperator)):
            observables = [observables]
        observables = [init_observable(observable) for observable in observables]

        super().__init__(
            circuits=circuits,
            observables=observables,
            parameters=parameters,
        )
        self._is_closed = False

        self._abelian_grouping = abelian_grouping

        self._backend = backend
        self._readout_mitigator = readout_mitigator
        self._run_options = Options()
        self._is_closed = False

        self._transpile_options = Options()
        self._bound_pass_manager = bound_pass_manager

        self._preprocessed_circuits: list[tuple[QuantumCircuit, list[QuantumCircuit]]] | None = None
        self._transpiled_circuits: list[QuantumCircuit] | None = None

        self._grouping = list(zip(range(len(self._circuits)), range(len(observables))))
        self._skip_transpilation = skip_transpilation

    @property
    def run_options(self) -> Options:
        """Return options values for the evaluator.
        Returns:
            run_options
        """
        return self._run_options

    def set_run_options(self, **fields) -> Estimator:
        """Set options values for the evaluator.

        Args:
            **fields: The fields to update the options
        Returns:
            self
        """
        self._check_is_closed()
        self._run_options.update_options(**fields)
        return self

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields) -> BaseEstimator:
        """Set the transpiler options for transpiler.

        Args:
            **fields: The fields to update the options
        Returns:
            self
        """
        self._check_is_closed()
        self._transpiled_circuits = None
        self._transpile_options.update_options(**fields)
        return self

    @property
    def preprocessed_circuits(
        self,
    ) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Transpiled quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        self._check_is_closed()
        self._preprocessed_circuits = self._preprocessing()
        return self._preprocessed_circuits

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.

        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._check_is_closed()
        self._transpile()
        return self._transpiled_circuits

    @property
    def backend(self) -> Backend:
        """
        Returns:
            The backend which this sampler object based on
        """
        return self._backend

    def _transpile(self):
        """Split Transpile"""
        self._transpiled_circuits = []
        for common_circuit, diff_circuits in self.preprocessed_circuits:
            # 1. transpile a common circuit
            common_circuit = common_circuit.copy()
            num_qubits = common_circuit.num_qubits
            common_circuit.measure_all()
            if not self._skip_transpilation:
                common_circuit = cast(
                    QuantumCircuit,
                    transpile(common_circuit, self.backend, **self.transpile_options.__dict__),
                )
            bit_map = {bit: index for index, bit in enumerate(common_circuit.qubits)}
            layout = [bit_map[qr[0]] for _, qr, _ in common_circuit[-num_qubits:]]
            common_circuit.remove_final_measurements()
            # 2. transpile diff circuits
            transpile_opts = copy.copy(self.transpile_options)
            transpile_opts.update_options(initial_layout=layout)
            diff_circuits = cast(
                "list[QuantumCircuit]",
                transpile(diff_circuits, self.backend, **transpile_opts.__dict__),
            )
            # 3. combine
            transpiled_circuits = []
            for diff_circuit in diff_circuits:
                transpiled_circuit = common_circuit.copy()
                for creg in diff_circuit.cregs:
                    if creg not in transpiled_circuit.cregs:
                        transpiled_circuit.add_register(creg)
                transpiled_circuit.compose(diff_circuit, inplace=True)
                transpiled_circuit.metadata = diff_circuit.metadata
                transpiled_circuits.append(transpiled_circuit)
            self._transpiled_circuits += transpiled_circuits

    def __call__(
        self,
        circuit_indices: Sequence[int] | None = None,
        observable_indices: Sequence[int] | None = None,
        parameter_values: Sequence[Sequence[float]] | Sequence[float] | None = None,
        **run_options,
    ) -> EstimatorResult:
        self._check_is_closed()
        if isinstance(parameter_values, np.ndarray):
            parameter_values = parameter_values.tolist()

        if parameter_values and not isinstance(parameter_values[0], (np.ndarray, Sequence)):
            parameter_values = cast("Sequence[float]", parameter_values)
            parameter_values = [parameter_values]
        if (
            circuit_indices is None
            and len(self.circuits) == 1
            and observable_indices is None
            and len(self.observables) == 1
            and parameter_values is not None
        ):
            circuit_indices = [0] * len(parameter_values)
            observable_indices = [0] * len(parameter_values)
        if circuit_indices is None:
            circuit_indices = list(range(len(self.circuits)))
        if observable_indices is None:
            observable_indices = list(range(len(self.observables)))
        if parameter_values is None:
            for i in circuit_indices:
                if len(self.circuits[i].parameters) != 0:
                    raise QiskitError(
                        f"The {i}-th circuit ({len(circuit_indices)}) is parametrised,"
                        "but parameter values are not given."
                    )

            parameter_values = [[]] * len(circuit_indices)
        parameter_values = cast("Sequence[Sequence[float]]", parameter_values)

        # Validation
        if len(circuit_indices) != len(parameter_values):
            raise QiskitError(
                f"The number of circuits ({len(circuit_indices)}) does not match "
                f"the number of parameter sets ({len(parameter_values)})."
            )

        for i, value in zip(circuit_indices, parameter_values):
            if len(value) != len(self.parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self.parameters[i])}) for the {i}-th circuit."
                )

        for circ_i, obs_i in zip(circuit_indices, observable_indices):
            circuit_num_qubits = self.circuits[circ_i].num_qubits
            observable_num_qubits = self.observables[obs_i].num_qubits
            if circuit_num_qubits != observable_num_qubits:
                raise QiskitError(
                    f"The number of qubits of the {circ_i}-th circuit ({circuit_num_qubits}) does "
                    f"not match the number of qubits of the {obs_i}-th observable "
                    f"({observable_num_qubits})."
                )

        # Transpile
        self._grouping = list(zip(circuit_indices, observable_indices))
        transpiled_circuits = self.transpiled_circuits
        num_observables = [len(m) for (_, m) in self.preprocessed_circuits]
        accum = [0] + list(accumulate(num_observables))

        # Bind parameters
        parameter_dicts = [
            dict(zip(self._parameters[i], value))
            for i, value in zip(circuit_indices, parameter_values)
        ]
        bound_circuits = [
            transpiled_circuits[circuit_index].bind_parameters(p)
            for i, (p, n) in enumerate(zip(parameter_dicts, num_observables))
            for circuit_index in range(accum[i], accum[i] + n)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        results = self._backend.run(bound_circuits, **run_opts.__dict__).result()

        results.num_observables = num_observables
        return self._postprocessing(results)

    def close(self):
        self._is_closed = True

    def _preprocessing(self) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Preprocessing for evaluation of expectation value using pauli rotation gates.
        """
        preprocessed_circuits = []
        for group in self._grouping:
            circuit = self._circuits[group[0]]
            observable = self._observables[group[1]]
            diff_circuits: list[QuantumCircuit] = []
            if self._abelian_grouping:
                for o_p in observable.grouping():
                    coeff_dict = {
                        key: val.real.item() if np.isreal(val) else val.item()
                        for key, val in o_p.label_iter()
                    }
                    lst = []
                    for paulis in zip(*coeff_dict.keys()):
                        pauli_set = set(paulis)
                        pauli_set.discard("I")
                        lst.append(pauli_set.pop() if pauli_set else "I")
                    pauli = "".join(lst)

                    meas_circuit = QuantumCircuit(circuit.num_qubits, observable.num_qubits)
                    for i, val in enumerate(reversed(pauli)):
                        if val == "Y":
                            meas_circuit.sdg(i)
                        if val in ["Y", "X"]:
                            meas_circuit.h(i)
                        meas_circuit.measure(i, i)
                    meas_circuit.metadata = {"basis": pauli, "coeff": coeff_dict}
                    diff_circuits.append(meas_circuit)
            else:
                for pauli, coeff in observable.label_iter():
                    meas_circuit = QuantumCircuit(circuit.num_qubits, observable.num_qubits)
                    for i, val in enumerate(reversed(pauli)):
                        if val == "Y":
                            meas_circuit.sdg(i)
                        if val in ["Y", "X"]:
                            meas_circuit.h(i)
                        meas_circuit.measure(i, i)
                    coeff = coeff.real.item() if np.isreal(coeff) else coeff.item()
                    meas_circuit.metadata = {"basis": pauli, "coeff": coeff}
                    diff_circuits.append(meas_circuit)

            preprocessed_circuits.append((circuit.copy(), diff_circuits))
        return preprocessed_circuits

    def _postprocessing(self, result: Result) -> EstimatorResult:
        """
        Postprocessing for evaluation of expectation value using pauli rotation gates.
        """

        counts = result.get_counts()
        if not isinstance(counts, list):
            counts = [counts]
        metadata = [res.header.metadata for res in result.results]
        num_observables = result.num_observables
        accum = [0] + list(accumulate(num_observables))
        expval_list = []
        var_list = []

        for i in range(len(num_observables)):

            combined_expval = 0.0
            combined_var = 0.0
            combined_stderr = 0.0

            for count, meta in zip(
                counts[accum[i] : accum[i + 1]], metadata[accum[i] : accum[i + 1]]
            ):
                basis = meta.get("basis", None)
                coeff = meta.get("coeff", 1)
                basis_coeff = coeff if isinstance(coeff, dict) else {basis: coeff}
                for basis, coeff in basis_coeff.items():
                    diagonal = str2diag(basis.translate(self._trans)) if basis is not None else None
                    # qubits = meta.get("qubits", None)
                    shots = sum(count.values())

                    # Compute expval component
                    if self._readout_mitigator is None:
                        expval, var = _expval_with_variance(count, diagonal)
                    else:
                        expval, stddev = self._readout_mitigator.expectation_value(count, diagonal)
                        var = stddev**2 * shots
                    # Accumulate
                    combined_expval += expval * coeff
                    combined_var += var * coeff**2
                    combined_stderr += np.sqrt(max(var * coeff**2 / shots, 0.0))
            expval_list.append(combined_expval)
            var_list.append(combined_var)
        metadata = [{"variance": var} for var in var_list]

        return EstimatorResult(np.array(expval_list, np.float64), metadata)

    def _check_is_closed(self):
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

    def _bound_pass_manager_run(self, circuits):
        if self._bound_pass_manager is None:
            return circuits
        else:
            return cast("list[QuantumCircuit]", self._bound_pass_manager.run(circuits))

    @staticmethod
    def result_to_dict(result: EstimatorResult, shots: int):
        """Convert ``EstimatorResult`` to a dictionary

        Args:
            result: The result of ``Sampler``
            shots: The number of shots

        Returns:
            A dictionary representing the result.

        """
        ret = result.__dict__
        for metadata in ret["metadata"]:
            metadata["shots"] = shots
        return ret


def _expval_with_variance(
    counts: Counts,
    diagonal: np.ndarray | None = None,
) -> tuple[float, float]:

    probs = np.fromiter(counts.values(), dtype=float)
    shots = probs.sum()
    probs = probs / shots

    # Get diagonal operator coefficients
    if diagonal is None:
        coeffs = np.array([(-1) ** key.count("1") for key in counts.keys()], dtype=probs.dtype)
    else:
        keys = [int("0b" + key, 0) for key in counts.keys()]
        coeffs = np.asarray(diagonal[keys], dtype=probs.dtype)

    # Compute expval
    expval = coeffs.dot(probs)

    # Compute variance
    if diagonal is None:
        # The square of the parity diagonal is the all 1 vector
        sq_expval = np.sum(probs)
    else:
        sq_expval = (coeffs**2).dot(probs)
    variance = sq_expval - expval**2

    # Compute standard deviation
    if variance < 0:
        if not np.isclose(variance, 0):
            logger.warning(
                "Encountered a negative variance in expectation value calculation."
                "(%f). Setting standard deviation of result to 0.",
                variance,
            )
        variance = np.float64(0.0)
    return expval.item(), variance.item()


def main(
    backend,
    user_messenger,  # pylint: disable=unused-argument
    circuits,
    circuit_indices,
    observables,
    observable_indices,
    parameters=None,
    parameter_values=None,
    skip_transpilation=False,
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
        skip_transpilation: Skip transpiling of circuits, default=False.
        run_options: Execution time options.

    Returns: Expectation values and metadata.

    """
    if len(circuit_indices) != len(observable_indices):
        raise ValueError(
            f"The length of circuit_indices {len(circuit_indices)} must "
            f"match the length of observable_indices {len(observable_indices)}."
        )
    if parameter_values is not None and len(parameter_values) != len(circuit_indices):
        raise ValueError(
            f"The length of parameter_values {len(parameter_values)} must "
            f"match the length of circuit_indices {len(circuit_indices)}."
        )

    estimator = Estimator(
        backend=backend,
        circuits=circuits,
        observables=observables,
        parameters=parameters,
        skip_transpilation=skip_transpilation,
    )
    run_options = run_options or {}
    shots = run_options.get("shots") or backend.options.shots
    result = estimator(
        circuit_indices=circuit_indices,
        observable_indices=observable_indices,
        parameter_values=parameter_values,
        **run_options,
    )

    result_dict = estimator.result_to_dict(result, shots)

    return result_dict
