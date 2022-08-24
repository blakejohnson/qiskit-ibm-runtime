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

from mthree.utils import final_measurement_mapping
import numpy as np
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.providers import Backend, Options
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.result import Counts, Result
from qiskit.tools.monitor import job_monitor
from qiskit.transpiler import PassManager


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
        ret = observable
    elif isinstance(observable, PauliSumOp):
        if isinstance(observable.coeff, ParameterExpression):
            raise TypeError(
                f"Observable must have numerical coefficient, not {type(observable.coeff)}."
            )
        ret = observable.coeff * observable.primitive
    elif isinstance(observable, BasePauli):
        ret = SparsePauliOp(observable)
    elif isinstance(observable, BaseOperator):
        ret = SparsePauliOp.from_operator(observable)
    else:
        ret = SparsePauliOp(observable)

    return ret.simplify(atol=0)


def run_circuits(
    circuits: QuantumCircuit | list[QuantumCircuit],
    backend: Backend,
    monitor: bool = False,
    **run_options,
) -> tuple[Result, list[dict]]:
    """Remove metadata of circuits and run the circuits on a backend.

    Args:
        circuits: The circuits
        backend: The backend
        monitor: Enable job minotor if True
        **run_options: run_options

    Returns:
        The result and the metadata of the circuits
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    metadata = []
    for circ in circuits:
        metadata.append(circ.metadata)
        circ.metadata = {}

    job = backend.run(circuits, **run_options)
    if monitor:
        job_monitor(job)
    return job.result(), metadata


class Estimator(BaseEstimator):
    """
    Evaluates expectation value using pauli rotation gates.
    """

    def __init__(
        self,
        backend: Backend,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
        parameters: Iterable[Iterable[Parameter]] | None = None,
        abelian_grouping: bool = True,
        bound_pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
        pauli_twirled_mitigation: PauliTwirledMitigation | None = None,
    ):
        if not isinstance(backend, Backend):
            raise TypeError(f"backend should be BackendV1, not {type(backend)}.")

        if isinstance(observables, (PauliSumOp, BaseOperator)):
            observables = (observables,)
        observables = tuple(init_observable(observable) for observable in observables)

        super().__init__(
            circuits=circuits,
            observables=observables,
            parameters=parameters,
        )
        self._is_closed = False

        self._abelian_grouping = abelian_grouping

        self._backend = backend
        self._run_options = Options()
        self._is_closed = False

        self._transpile_options = Options()
        self._bound_pass_manager = bound_pass_manager

        self._preprocessed_circuits: list[tuple[QuantumCircuit, list[QuantumCircuit]]] | None = None
        self._transpiled_circuits: list[QuantumCircuit] | None = None

        self._grouping = list(zip(range(len(self._circuits)), range(len(self._observables))))
        self._skip_transpilation = skip_transpilation
        self._mitigation = pauli_twirled_mitigation

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
            The backend which this estimator object based on
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
                common_circuit = transpile(
                    common_circuit, self.backend, **self.transpile_options.__dict__
                )
            bit_map = {bit: index for index, bit in enumerate(common_circuit.qubits)}
            layout = [bit_map[qr[0]] for _, qr, _ in common_circuit[-num_qubits:]]
            common_circuit.remove_final_measurements()
            # 2. transpile diff circuits
            transpile_opts = copy.copy(self.transpile_options)
            transpile_opts.update_options(initial_layout=layout)
            diff_circuits = transpile(diff_circuits, self.backend, **transpile_opts.__dict__)
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

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        self._check_is_closed()

        # Transpile
        self._grouping = list(zip(circuits, observables))
        transpiled_circuits = self.transpiled_circuits
        num_observables = [len(m) for (_, m) in self.preprocessed_circuits]
        accum = [0] + list(accumulate(num_observables))

        # Bind parameters
        parameter_dicts = [
            dict(zip(self._parameters[i], value)) for i, value in zip(circuits, parameter_values)
        ]
        bound_circuits = [
            transpiled_circuits[circuit_index]
            if len(p) == 0
            else transpiled_circuits[circuit_index].bind_parameters(p)
            for i, (p, n) in enumerate(zip(parameter_dicts, num_observables))
            for circuit_index in range(accum[i], accum[i] + n)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)

        if self._mitigation:
            self._mitigation.cals_from_system(
                [final_measurement_mapping(circ) for circ in bound_circuits]
            )
            # run
            result, metadata = self._mitigation.run_circuits(bound_circuits, **run_opts.__dict__)
            accum = [e * self._mitigation.num_twirled_circuits for e in accum]
        else:
            result, metadata = run_circuits(bound_circuits, self._backend, **run_opts.__dict__)

        return self._postprocessing(result, accum, metadata)

    def close(self):
        self._is_closed = True

    @staticmethod
    def _measurement_circuit(num_qubits: int, pauli: Pauli):
        # Note: if pauli is I for all qubits, this function generates a circuit to measure only
        # the first qubit.
        # Although such an operator can be optimized out by interpreting it as a constant (1),
        # this optimization requires changes in various methods. So it is left as future work.
        qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
        if not np.any(qubit_indices):
            qubit_indices = [0]
        meas_circuit = QuantumCircuit(num_qubits, len(qubit_indices))
        for clbit, i in enumerate(qubit_indices):
            if pauli.x[i]:
                if pauli.z[i]:
                    meas_circuit.sdg(i)
                meas_circuit.h(i)
            meas_circuit.measure(i, clbit)
        return meas_circuit, qubit_indices

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
                for obs in observable.group_commuting(qubit_wise=True):
                    basis = Pauli(
                        (np.logical_or.reduce(obs.paulis.z), np.logical_or.reduce(obs.paulis.x))
                    )
                    meas_circuit, indices = self._measurement_circuit(circuit.num_qubits, basis)
                    paulis = PauliList.from_symplectic(
                        obs.paulis.z[:, indices],
                        obs.paulis.x[:, indices],
                        obs.paulis.phase,
                    )
                    meas_circuit.metadata = {
                        "paulis": paulis,
                        "coeffs": np.real_if_close(obs.coeffs),
                    }
                    diff_circuits.append(meas_circuit)
            else:
                for basis, obs in zip(observable.paulis, observable):
                    meas_circuit, indices = self._measurement_circuit(circuit.num_qubits, basis)
                    paulis = PauliList.from_symplectic(
                        obs.paulis.z[:, indices],
                        obs.paulis.x[:, indices],
                        obs.paulis.phase,
                    )
                    meas_circuit.metadata = {
                        "paulis": paulis,
                        "coeffs": np.real_if_close(obs.coeffs),
                    }
                    diff_circuits.append(meas_circuit)

            preprocessed_circuits.append((circuit.copy(), diff_circuits))
        return preprocessed_circuits

    def _postprocessing(
        self, result: Result, accum: list[int], metadata: list[dict]
    ) -> EstimatorResult:
        """
        Postprocessing for evaluation of expectation value using pauli rotation gates.
        """

        counts = result.get_counts()
        if not isinstance(counts, list):
            counts = [counts]
        expval_list = []
        var_list = []
        shots_list = []

        for i, j in zip(accum, accum[1:]):

            combined_expval = 0.0
            combined_var = 0.0
            step = self._mitigation.num_twirled_circuits if self._mitigation else 1

            for k in range(i, j, step):
                meta = metadata[k]
                paulis = meta["paulis"]
                coeffs = meta["coeffs"]

                if self._mitigation:
                    flips = [datum["flip"] for datum in metadata[k : k + step]]
                    count = self._mitigation.combine_counts(counts[k : k + step], flips)
                else:
                    count = counts[k]

                expvals, variances = _pauli_expval_with_variance(count, paulis)

                if self._mitigation:
                    count2 = self._mitigation.calibrated_counts[meta["qubits"]]
                    div, _ = _pauli_expval_with_variance(count2, paulis)
                    expvals /= div
                    # TODO: this variance is a rough estimation. Need more accurate one in the future.
                    variances /= div**2

                # Accumulate
                combined_expval += np.dot(expvals, coeffs)
                combined_var += np.dot(variances, coeffs**2)

            expval_list.append(combined_expval)
            var_list.append(combined_var)
            shots_list.append(sum(counts[i].values()) * step)

        metadata = [{"variance": var, "shots": shots} for var, shots in zip(var_list, shots_list)]
        if self._mitigation:
            for meta in metadata:
                meta.update(
                    {
                        "readout_mitigation_num_twirled_circuits": self._mitigation.num_twirled_circuits,
                        "readout_mitigation_shots_calibration": self._mitigation.shots_calibration,
                    }
                )

        return EstimatorResult(np.real_if_close(expval_list), metadata)

    def _check_is_closed(self):
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

    def _bound_pass_manager_run(self, circuits):
        if self._bound_pass_manager is None:
            return circuits
        else:
            return self._bound_pass_manager.run(circuits)


def _paulis2inds(paulis: PauliList) -> list[int]:
    """Convert PauliList to diagonal integers.

    These are integer representations of the binary string with a
    1 where there are Paulis, and 0 where there are identities.
    """
    # Treat Z, X, Y the same
    nonid = paulis.z | paulis.x

    inds = [0] * paulis.size
    # bits are packed into uint8 in little endian
    # e.g., i-th bit corresponds to coefficient 2^i
    packed_vals = np.packbits(nonid, axis=1, bitorder="little")
    for i, vals in enumerate(packed_vals):
        for j, val in enumerate(vals):
            inds[i] += val.item() * (1 << (8 * j))
    return inds


def _parity(integer: int) -> int:
    """Return the parity of an integer"""
    return bin(integer).count("1") % 2


def _pauli_expval_with_variance(counts: Counts, paulis: PauliList) -> tuple[np.ndarray, np.ndarray]:
    """Return array of expval and variance pairs for input Paulis.

    Note: All non-identity Pauli's are treated as Z-paulis, assuming
    that basis rotations have been applied to convert them to the
    diagonal basis.
    """
    # Diag indices
    size = len(paulis)
    diag_inds = _paulis2inds(paulis)

    expvals = np.zeros(size, dtype=float)
    denom = 0  # Total shots for counts dict
    for bin_outcome, freq in counts.items():
        outcome = int(bin_outcome, 2)
        denom += freq
        for k in range(size):
            coeff = (-1) ** _parity(diag_inds[k] & outcome)
            expvals[k] += freq * coeff

    # Divide by total shots
    expvals /= denom

    # Compute variance
    variances = 1 - expvals**2
    return expvals, variances


class PauliTwirledMitigation:
    """
    Pauli twirled readout error mitigation (T-Rex)
    """

    def __init__(
        self,
        backend: Backend,
        num_twirled_circuits: int = 16,
        shots_calibration: int = 8192,
        seed: np.random.Generator | int | None = None,
        **cal_run_options,
    ):
        self._backend = backend
        self._num_twirled_circuits = num_twirled_circuits
        self._shots_calibration = shots_calibration
        if seed is None or isinstance(seed, int):
            self._rng = np.random.default_rng(seed)
        elif isinstance(seed, np.random.Generator):
            self._rng = seed
        else:
            raise QiskitError(f"Invalid random number seed: {seed}")
        self._cal_run_options = cal_run_options
        self._counts_identity: dict[Sequence[int], Counts] = {}

    @property
    def num_twirled_circuits(self):
        """Number of Pauli twirled circuits for each circuit"""
        return self._num_twirled_circuits

    @property
    def calibrated_counts(self):
        """Calibrate counts for identity circuits"""
        return self._counts_identity

    @property
    def shots_calibration(self):
        """Number of shots for calibration"""
        return self._num_twirled_circuits * self._subdivide_shots(
            self._shots_calibration, self._num_twirled_circuits
        )

    def cals_from_system(self, mappings: list[dict[int, int]]):
        """Calibrate count data

        Args:
            mappings: The qubit mapping
        """
        for mapping in mappings:
            qubits_to_measure = tuple(mapping)
            if qubits_to_measure not in self._counts_identity:
                self._counts_identity[qubits_to_measure] = self._calibrate(qubits_to_measure)

    @staticmethod
    def _bitflip(bitstring: str, flip_qubits: list[int]):
        lst = list(bitstring[::-1])
        conv = {"0": "1", "1": "0"}
        for i in flip_qubits:
            lst[i] = conv[lst[i]]
        return "".join(lst[::-1])

    @classmethod
    def combine_counts(cls, counts: list[Counts], flips: list[list[int]]) -> Counts:
        """Combine count data by flipping bits

        Args:
            counts: The count data
            flips: The flip data

        Returns:
            The sum of the count data that are flipped at corresponding bits to the flip data.

        """
        total: dict[str, int] = defaultdict(int)
        for count, flip in zip(counts, flips):
            for key, num in count.items():
                total[cls._bitflip(key, flip)] += num
        return Counts(total)

    def _append_random_x_and_measure(self, circ: QuantumCircuit, qubits: Sequence[int]):
        flip = np.where(self._rng.choice(2, len(qubits)) == 1)[0]
        if len(flip) > 0:
            circ.x(np.asarray(qubits)[flip])
        meas = QuantumCircuit(circ.num_qubits, len(qubits))
        meas.measure(qubits, range(len(qubits)))
        for creg in meas.cregs:
            circ.add_register(creg)
        circ.compose(meas, inplace=True)
        qubits = tuple(qubits)
        if circ.metadata:
            circ.metadata["flip"] = flip
            circ.metadata["qubits"] = qubits
        else:
            circ.metadata = {"flip": flip, "qubits": qubits}
        return circ

    @staticmethod
    def _subdivide_shots(shots: int, div: int) -> int:
        """Subdivide shots

        Args:
            shots: The total number of shots to be subdivided
            div: The divisor

        Returns:
            The number of subdivided shots. The sum of the shots should be equal to or larger than
            the total number of shots.

        Reference:
            https://datagy.io/python-ceiling/ ("Python Ceiling Division" section)
        """
        return -(-shots // div)

    def _calibrate(self, qubits: Sequence[int]):
        circuits = []
        for _ in range(self._num_twirled_circuits):
            circ = QuantumCircuit(max(qubits) + 1)
            circ = self._append_random_x_and_measure(circ, qubits)
            circuits.append(circ)
        shots = self._subdivide_shots(self._shots_calibration, self._num_twirled_circuits)
        result, metadata = run_circuits(
            circuits, self._backend, shots=shots, **self._cal_run_options
        )

        counts = result.get_counts()
        if not isinstance(counts, list):
            counts = [counts]

        flips = [meta["flip"] for meta in metadata]

        return self.combine_counts(counts, flips)

    def run_circuits(self, circuits, shots, **options):
        """Generate Pauli twirled circuits and run them

        Args:
            circuits: The circuits to run.
            shots: The number of shots.
            **options: The run options of the backend.

        Returns:
            The result data of the circuits generated by applying Pauli twirling to the input circuits.

        """
        circuits2 = []
        for circ in circuits:
            qubits = list(final_measurement_mapping(circ))
            circ.remove_final_measurements(inplace=True)
            for _ in range(self._num_twirled_circuits):
                circ2 = self._append_random_x_and_measure(circ.copy(), qubits)
                circuits2.append(circ2)
        shots2 = self._subdivide_shots(shots, self._num_twirled_circuits)
        return run_circuits(circuits2, backend=self._backend, shots=shots2, **options)


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
    transpilation_settings=None,
    resilience_settings=None,
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
        skip_transpilation: (Deprecated) Skip transpiling of circuits, default=False.
        run_options: Execution time options.
        transpilation_settings: Transpilation settings.
        resilience_settings: Resilience settings.

    Returns: Expectation values and metadata.

    """
    transpilation_settings = transpilation_settings or {}
    skip_transpilation = transpilation_settings.pop("skip_transpilation", skip_transpilation)
    optimization_settings = transpilation_settings.pop("optimization_settings", {})
    resilience_settings = resilience_settings or {}

    mitigation = None
    if resilience_settings.get("level", 0) == 1:
        options = resilience_settings.pop("pauli_twirled_mitigation", {})
        seed = options.pop("seed_mitigation", None)
        mitigation = PauliTwirledMitigation(backend=backend, seed=seed, **options)

    estimator = Estimator(
        backend=backend,
        circuits=circuits,
        observables=observables,
        parameters=parameters,
        skip_transpilation=skip_transpilation,
        pauli_twirled_mitigation=mitigation,
    )

    transpile_options = transpilation_settings.copy()
    transpile_options["optimization_level"] = optimization_settings.get("level", 1)
    estimator.set_transpile_options(**transpile_options)

    run_options = run_options or {}
    if "shots" not in run_options:
        run_options["shots"] = backend.options.shots

    result = estimator(
        circuits=circuit_indices,
        observables=observable_indices,
        parameter_values=parameter_values,
        **run_options,
    )

    return result.__dict__
