# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
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

import copy
import hashlib
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import accumulate
from datetime import timedelta
from os import environ
from typing import Dict, List, Optional, cast, Union

import numpy as np
import qiskit
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RZGate, XGate
from qiskit.circuit.parametertable import ParameterView
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BackendEstimator, EstimatorResult
from qiskit.primitives.utils import init_observable, final_measurement_mapping
from qiskit.providers import Backend, BackendV1, BackendV2, Options
from qiskit.qasm3 import loads as qasm3_loads
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts, Result
from qiskit.tools.monitor import job_monitor
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    ConstrainedReschedule,
    InstructionDurationCheck,
    PadDynamicalDecoupling,
    TimeUnitConversion,
)
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder

from zne import zne, ZNEStrategy
from zne.noise_amplification import NoiseAmplifier, NOISE_AMPLIFIER_LIBRARY
from zne.extrapolation import Extrapolator, EXTRAPOLATOR_LIBRARY

from pec_runtime.primitives import Estimator as PEC_Estimator

QuantumProgram = Union[QuantumCircuit, str]

logger = logging.getLogger(__name__)

# If PRIMITIVES_DEBUG is True, metadata includes bound circuits, coeffs and, paulis.
# Only for internal development and test.
DEBUG = environ.get("PRIMITIVES_DEBUG", "false") == "true"


################################################################################
## ESTIMATOR
################################################################################
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
        monitor: Enable job monitor if True
        **run_options: run_options

    Returns:
        The result and the metadata of the circuits
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    metadata = []
    for circ in circuits:
        metadata.append(circ.metadata)
        metadata[-1]["bound_circuit"] = circ
        circ.metadata = {}

    job = backend.run(circuits, **run_options)
    if monitor:
        job_monitor(job)
    return job.result(), metadata


class Estimator:
    """
    Evaluates expectation value using pauli rotation gates.
    """

    def __init__(
        self,
        backend: Backend,
        circuits: QuantumProgram | Iterable[QuantumProgram] | Mapping[str, QuantumProgram],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
        parameters: Iterable[Iterable[Parameter]] | None = None,
        abelian_grouping: bool = True,
        bound_pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
        pauli_twirled_mitigation: PauliTwirledMitigation | None = None,
        circuit_ids: Sequence[str] | None = None,
    ):
        if isinstance(observables, (PauliSumOp, BaseOperator)):
            observables = (observables,)
        observables = tuple(init_observable(observable) for observable in observables)
        self._backend = backend
        self._circuit_ids: Sequence[str] = circuit_ids
        self._transpile_options = Options()
        self._circuits_map: Dict[str, QuantumProgram] = {}
        self._circuits = self._get_circuits(circuits=circuits)
        self._observables = self._get_observables(observables=observables)
        self._parameters = parameters
        self._circuit_cache = CircuitCache(cache=self._get_cache())
        self._is_closed = False
        self._abelian_grouping = abelian_grouping
        self._run_options = Options()
        self._bound_pass_manager = bound_pass_manager
        self._preprocessed_circuits: list[tuple[QuantumCircuit, list[QuantumCircuit]]] | None = None
        self._transpiled_circuits: list[QuantumCircuit] | None = None
        self._grouping = list(zip(range(len(self._circuits)), range(len(self._observables))))
        self._skip_transpilation = skip_transpilation
        self._mitigation = pauli_twirled_mitigation

    def _get_circuits(
        self, circuits: QuantumProgram | Iterable[QuantumProgram] | Mapping[str, QuantumProgram]
    ) -> list[QuantumCircuit]:
        """Return list of circuits."""

        # Convert from QASM to QauntumCircuit, if needed.
        if isinstance(circuits, str):
            circuits = (str_to_quantum_circuit(circuits),)
        elif isinstance(circuits, Dict):
            circuits = {
                k: (str_to_quantum_circuit(v) if isinstance(v, str) else v)
                for k, v in circuits.items()
            }
        elif isinstance(circuits, Iterable):
            circuits = [
                str_to_quantum_circuit(circuit) if isinstance(circuit, str) else circuit
                for circuit in circuits
            ]

        # Return list of QuantumCircuit objects.
        if isinstance(circuits, QuantumCircuit):
            return [circuits]
        elif isinstance(circuits, dict) or self._circuit_ids is not None:
            self._circuits_map = circuits  # type: ignore
            return list(circuits.values())  # type: ignore
        else:
            return [] if circuits is None else list(circuits)

    def _get_observables(
        self, observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp]
    ):
        """Return list of observables."""
        if isinstance(observables, SparsePauliOp):
            observables = (observables,)
        return [] if observables is None else list(observables)

    def _get_cache(self):
        """Return instance of Cache class."""
        try:
            _provider = (
                self._backend.provider if self._backend.version == 2 else self._backend.provider()
            )
            return _provider.cache()
        except AttributeError:
            # Unit tests use AerProvider which doesn't have cache() method
            return None

    def run(
        self,
        circuit_indices: Sequence[int] = None,
        observable_indices: Sequence[int] = None,
        parameter_values: Sequence[Sequence[float]] = None,
        **run_options,
    ) -> EstimatorResult:
        """Evaluate expectation values."""
        observable_indices = cast("list[int]", observable_indices)
        # TODO Remove this else block when removing support for non-flexible sessions
        if not self._circuit_ids:
            circuit_indices = cast("list[int]", circuit_indices)
            self._grouping = list(zip(circuit_indices, observable_indices))
            self._validate_circuits_observables_for_circuit_indices_path(
                circuits=self._circuits,
                observables=self._observables,
            )
            self._validate_circuit_indices_observable_indices(
                circuit_indices=circuit_indices,
                observable_indices=observable_indices,
            )
            self._parameters = self._initialize_parameters(
                circuits=self._circuits,
                parameters=self._parameters,
            )
            self._validate_parameters(
                circuits=self._circuits,
                parameters=self._parameters,
            )
            parameter_values = self._initialize_parameter_values(
                circuits_len=len(circuit_indices),
                circuits=self._circuits,
                parameter_values=parameter_values,
            )
            self._validate_parameter_values_circuit_indices(
                circuit_indices=circuit_indices,
                parameters=self._parameters,
                parameter_values=parameter_values,
            )
            transpiled_circuits = self.transpiled_circuits
            # Bind parameters
            parameter_dicts = [
                dict(zip(self._parameters[i], value))  # type: ignore
                for i, value in zip(circuit_indices, parameter_values)
            ]
        else:
            self._grouping = list(zip(self._circuit_ids, observable_indices))  # type: ignore
            transpiled_circuits = self.transpiled_circuits
            self._validate_circuits_observables_for_circuit_ids_path(
                circuits=transpiled_circuits,
                observables=self._observables,
            )
            self._validate_circuit_ids_observable_indices(
                circuit_ids=self._circuit_ids,
                observable_indices=observable_indices,
            )
            self._parameters = self._initialize_parameters(
                circuits=transpiled_circuits,
                parameters=self._parameters,
            )
            self._validate_parameters(
                circuits=transpiled_circuits,
                parameters=self._parameters,
            )
            parameter_values = self._initialize_parameter_values(
                circuits_len=len(self._circuit_ids),
                circuits=transpiled_circuits,
                parameter_values=parameter_values,
            )
            self._validate_parameter_values_circuit_ids(
                circuit_ids=self._circuit_ids,
                parameters=self._parameters,
                parameter_values=parameter_values,
            )
            # Bind parameters
            parameter_dicts = [
                dict(zip(self._parameters[i], value))  # type: ignore
                for i, value in zip(range(len(self._circuit_ids)), parameter_values)
            ]
        num_observables = [len(m) for (_, m) in self.preprocessed_circuits]
        accum = [0] + list(accumulate(num_observables))
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
        paulis_list = []
        coeffs_list = []
        circ_list = []

        for i, j in zip(accum, accum[1:]):

            combined_expval = 0.0
            combined_var = 0.0
            step = self._mitigation.num_twirled_circuits if self._mitigation else 1

            for k in range(i, j, step):
                meta = metadata[k]
                paulis = meta["paulis"]
                paulis_list.append(paulis)
                coeffs = meta["coeffs"]
                coeffs_list.append(coeffs)

                if self._mitigation:
                    flips = [datum["flip"] for datum in metadata[k : k + step]]
                    count = self._mitigation.combine_counts(counts[k : k + step], flips)
                    circ_list.append([datum["bound_circuit"] for datum in metadata[k : k + step]])
                else:
                    count = counts[k]
                    circ_list.append([metadata[k]["bound_circuit"]])

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

        if DEBUG:
            metadata = [
                {
                    "variance": var,
                    "shots": shots,
                    "paulis": paulis,
                    "coeffs": coeffs,
                    "bound_circuits": circ,
                }
                for var, shots, paulis, coeffs, circ in zip(
                    var_list, shots_list, paulis_list, coeffs_list, circ_list
                )
            ]
        else:
            metadata = [
                {"variance": var, "shots": shots} for var, shots in zip(var_list, shots_list)
            ]

        if self._mitigation:
            for meta in metadata:
                meta.update(
                    {
                        "readout_mitigation_num_twirled_circuits": self._mitigation.num_twirled_circuits,
                        "readout_mitigation_shots_calibration": self._mitigation.shots_calibration,
                    }
                )

        return EstimatorResult(np.real_if_close(expval_list), metadata)

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.

        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        _backend_name = self._backend.name if self._backend.version == 2 else self._backend.name()
        if not self._transpiled_circuits:
            if self._circuit_ids:
                # 1. Initialize a list transpiled circuits from cache and another list of
                # raw circuits whose transpiled versions were not found in cache
                self._circuit_cache.initialize_transpiled_and_raw_circuits(
                    circuits_map=self._circuits_map,
                    circuit_ids=self._circuit_ids,
                    backend_name=_backend_name,
                    transpile_options=self._transpile_options.__dict__,
                )
                # 2. Transpile the raw circuits whose transpiled versions were not found in cache
                raw_circuits = []
                for raw_circuit in self._circuit_cache.raw_circuits:
                    raw_circuits.append(self._duplicate_and_measure_all_qubits(circuit=raw_circuit))
                if not self._skip_transpilation:
                    transpiled_circuits = transpile(
                        raw_circuits, self.backend, **self.transpile_options.__dict__
                    )
                else:
                    transpiled_circuits = raw_circuits
                # 3. Update cache with transpiled and raw circuits and merge transpiled circuits
                # from step 2 with transpiled circuits retrieved from cache in step 1
                self._circuit_cache.update_cache_and_merge_transpiled_circuits(
                    transpiled_circuits=transpiled_circuits
                )
            self._split_transpile()
        return self._transpiled_circuits

    def _duplicate_and_measure_all_qubits(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Duplicate circuit and add measure gate to all qubits."""
        circuit = circuit.copy()
        if self._circuit_ids:
            circuit = self._update_circuit_metadata(
                circuit, {"raw_circuit_num_qubits": circuit.num_qubits}
            )
        circuit.measure_all()
        return circuit

    def _update_circuit_metadata(self, circuit: QuantumCircuit, metadata: dict) -> QuantumCircuit:
        """Update circuit metadata and return circuit"""
        if circuit.metadata:
            circuit.metadata.update(metadata)
        else:
            circuit.metadata = metadata
        return circuit

    def _split_transpile(self):
        """Split Transpile"""
        self._transpiled_circuits = []
        for common_circuit, diff_circuits in self.preprocessed_circuits:
            if not self._circuit_ids:
                # 1. transpile a common circuit
                num_qubits = common_circuit.num_qubits
                common_circuit = self._duplicate_and_measure_all_qubits(circuit=common_circuit)
                if not self._skip_transpilation:
                    common_circuit = transpile(
                        common_circuit, self.backend, **self.transpile_options.__dict__
                    )
            else:
                num_qubits = common_circuit.metadata.get("raw_circuit_num_qubits")
            bit_map = {bit: index for index, bit in enumerate(common_circuit.qubits)}
            layout = [bit_map[qr[0]] for _, qr, _ in common_circuit[-num_qubits:]]
            common_circuit.remove_final_measurements()
            # 2. transpile diff circuits
            transpile_opts = copy.copy(self.transpile_options)
            transpile_opts.update_options(initial_layout=layout)
            diff_circuits = transpile(diff_circuits, self.backend, **transpile_opts.__dict__)
            # 3. combine
            self._transpiled_circuits += self._combine(
                common_circuit=common_circuit, diff_circuits=diff_circuits
            )

    def _combine(
        self, common_circuit: QuantumCircuit, diff_circuits: list[QuantumCircuit]
    ) -> list[QuantumCircuit]:
        """Combine common circuit and diff circuits."""
        transpiled_circuits = []
        for diff_circuit in diff_circuits:
            transpiled_circuit = common_circuit.copy()
            for creg in diff_circuit.cregs:
                if creg not in transpiled_circuit.cregs:
                    transpiled_circuit.add_register(creg)
            transpiled_circuit.compose(diff_circuit, inplace=True)
            transpiled_circuit = self._update_circuit_metadata(
                circuit=transpiled_circuit, metadata=diff_circuit.metadata
            )
            transpiled_circuits.append(transpiled_circuit)
        return transpiled_circuits

    @property
    def preprocessed_circuits(
        self,
    ) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Transpiled quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        if not self._preprocessed_circuits:
            self._preprocessed_circuits = self._preprocessing()
        return self._preprocessed_circuits

    def _preprocessing(self) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Preprocessing for evaluation of expectation value using pauli rotation gates.
        """
        preprocessed_circuits = []
        for group in self._grouping:
            circuit = self._get_circuit_by_index_or_id(circuit_index_or_id=group[0])
            num_qubits = self._get_circuit_num_qubits(circuit=circuit)
            observable = self._observables[group[1]]
            diff_circuits: list[QuantumCircuit] = []
            if self._abelian_grouping:
                for obs in observable.group_commuting(qubit_wise=True):
                    basis = Pauli(
                        (np.logical_or.reduce(obs.paulis.z), np.logical_or.reduce(obs.paulis.x))
                    )
                    meas_circuit, paulis = self._measurement_circuit(num_qubits, basis, obs)
                    meas_circuit.metadata = {
                        "paulis": paulis,
                        "coeffs": np.real_if_close(obs.coeffs),
                    }
                    diff_circuits.append(meas_circuit)
            else:
                for basis, obs in zip(observable.paulis, observable):
                    meas_circuit, paulis = self._measurement_circuit(num_qubits, basis, obs)
                    meas_circuit.metadata = {
                        "paulis": paulis,
                        "coeffs": np.real_if_close(obs.coeffs),
                    }
                    diff_circuits.append(meas_circuit)

            preprocessed_circuits.append((circuit.copy(), diff_circuits))
        return preprocessed_circuits

    def _get_circuit_by_index_or_id(self, circuit_index_or_id: int | str):
        """Get circuit by index or id"""
        if self._circuit_ids:
            return self._circuit_cache.get_transpiled_circuit(
                circuit_id=circuit_index_or_id  # type: ignore
            )
        else:
            return self._circuits[circuit_index_or_id]  # type: ignore

    def _get_circuit_num_qubits(self, circuit: QuantumCircuit):
        """Get raw circuit number of qubits (before transpilation)"""
        if self._circuit_ids:
            return circuit.metadata.get("raw_circuit_num_qubits")
        else:
            return circuit.num_qubits

    @staticmethod
    def _measurement_circuit(
        num_qubits: int, pauli: Pauli, obs: SparsePauliOp
    ) -> tuple[QuantumCircuit, PauliList]:
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
        paulis = PauliList.from_symplectic(
            obs.paulis.z[:, qubit_indices],
            obs.paulis.x[:, qubit_indices],
            obs.paulis.phase,
        )
        return meas_circuit, paulis

    def _bound_pass_manager_run(self, circuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
        if self._bound_pass_manager is None:
            return circuits
        circs = self._bound_pass_manager.run(circuits)
        return circs if isinstance(circs, list) else [circs]

    def _validate_circuits_observables_for_circuit_indices_path(
        self,
        circuits: List[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
    ) -> None:
        for i, (circuit, observable) in enumerate(zip(circuits, observables)):
            if circuit.num_qubits != observable.num_qubits:
                raise QiskitError(
                    f"The number of qubits of the {i}-th circuit ({circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable "
                    f"({observable.num_qubits})."
                )

    def _validate_circuits_observables_for_circuit_ids_path(
        self,
        circuits: List[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
    ) -> None:
        for i, (circuit, observable) in enumerate(zip(circuits, observables)):
            circuit_num_qubits = circuit.metadata.get("raw_circuit_num_qubits")
            if circuit_num_qubits != observable.num_qubits:
                raise QiskitError(
                    f"The number of qubits of the {i}-th circuit ({circuit_num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable "
                    f"({observable.num_qubits})."
                )

    def _validate_circuit_indices_observable_indices(
        self,
        circuit_indices: Sequence[int],
        observable_indices: Sequence[int],
    ) -> None:
        """Validate circuits and observables indices."""
        if len(circuit_indices) != len(observable_indices):
            raise QiskitError(
                f"The number of circuits ({len(circuit_indices)}) does not match "
                f"the number of observables ({len(observable_indices)})."
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
        if max(circuit_indices) >= len(self.circuits):
            raise QiskitError(
                f"The number of circuits is {len(self.circuits)}, "
                f"but the index {max(circuit_indices)} is given."
            )
        if max(observable_indices) >= len(self.observables):
            raise QiskitError(
                f"The number of circuits is {len(self.observables)}, "
                f"but the index {max(observable_indices)} is given."
            )

    def _validate_circuit_ids_observable_indices(
        self,
        circuit_ids: Sequence[str],
        observable_indices: Sequence[int],
    ) -> None:
        """Validate circuit ids and observables indices."""
        if len(circuit_ids) != len(observable_indices):
            raise QiskitError(
                f"The number of circuits ({len(circuit_ids)}) does not match "
                f"the number of observables ({len(observable_indices)})."
            )

    def _initialize_parameters(
        self, circuits: List[QuantumCircuit], parameters: Iterable[Iterable[Parameter]] = None
    ) -> Iterable[Iterable[Parameter]]:
        if parameters is None:
            return [circ.parameters for circ in circuits]
        else:
            return [ParameterView(par) for par in parameters]

    def _validate_parameters(
        self, circuits: List[QuantumCircuit], parameters: Iterable[Iterable[Parameter]] = None
    ) -> None:
        if len(parameters) != len(circuits):  # type: ignore
            raise QiskitError(
                f"Different number of parameters ({len(parameters)}) "  # type: ignore
                f"and circuits ({len(circuits)})."
            )
        for i, (circ, params) in enumerate(zip(circuits, parameters)):
            if circ.num_parameters != len(params):  # type: ignore
                raise QiskitError(
                    f"Different numbers of parameters of {i}-th circuit: "
                    f"expected {circ.num_parameters}, actual {len(params)}."  # type: ignore
                )

    def _initialize_parameter_values(
        self,
        circuits_len: int,
        circuits: List[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]] = None,
    ) -> Sequence[Sequence[float]]:
        # Support ndarray
        if isinstance(parameter_values, np.ndarray):
            parameter_values = parameter_values.tolist()
        # Allow optional
        if parameter_values is None:
            for i, circuit in enumerate(circuits):
                if circuit.num_parameters != 0:
                    raise QiskitError(
                        f"The {i}-th circuit ({len(circuits)}) is parameterized,"
                        "but parameter values are not given."
                    )
            return [[]] * circuits_len
        return parameter_values

    def _validate_parameter_values_circuit_indices(
        self,
        circuit_indices: Sequence[int] = None,
        parameters: Iterable[Iterable[Parameter]] = None,
        parameter_values: Sequence[Sequence[float]] = None,
    ) -> None:
        if len(circuit_indices) != len(parameter_values):
            raise QiskitError(
                f"The number of circuits ({len(circuit_indices)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )
        for i, value in zip(circuit_indices, parameter_values):
            if len(value) != len(parameters[i]):  # type: ignore
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(parameters[i])}) "  # type: ignore
                    f"for the {i}-th circuit."
                )

    def _validate_parameter_values_circuit_ids(
        self,
        circuit_ids: Sequence[str] = None,
        parameters: Iterable[Iterable[Parameter]] = None,
        parameter_values: Sequence[Sequence[float]] = None,
    ) -> None:
        if len(circuit_ids) != len(parameter_values):
            raise QiskitError(
                f"The number of circuits ({len(circuit_ids)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )
        for i, value in zip(range(len(circuit_ids)), parameter_values):
            if len(value) != len(parameters[i]):  # type: ignore
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(parameters[i])}) "  # type: ignore
                    f"for the circuit {i}."
                )

    @property
    def circuits(self) -> tuple[QuantumCircuit, ...]:
        """Quantum circuits used for evaluating expectation values.
        Returns:
            List of Quantum Circuits.
        """
        return tuple(self._circuits)

    @property
    def observables(self) -> tuple[SparsePauliOp, ...]:
        """Observables to be estimated.
        Returns:
            The observables.
        """
        return tuple(self._observables)

    @property
    def parameters(self) -> tuple[ParameterView, ...]:
        """Parameters of the quantum circuits.
        Returns:
            Parameters, where ``parameters[i][j]`` is the j-th parameter of the i-th circuit.
        """
        return tuple(self._parameters)

    @property
    def backend(self) -> Backend:
        """
        Returns:
            The backend which this estimator object based on
        """
        return self._backend

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
        self._run_options.update_options(**fields)
        return self

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields):
        """Set the transpiler options for transpiler.

        Args:
            **fields: The fields to update the options
        Returns:
            self
        """
        self._transpiled_circuits = None
        self._transpile_options.update_options(**fields)
        return self

    def close(self):
        """Close the session and free resources"""
        self._is_closed = True


def _paulis2inds(paulis: PauliList) -> list[int]:
    """Convert PauliList to diagonal integers.

    These are integer representations of the binary string with a
    1 where there are Paulis, and 0 where there are identities.
    """
    # Treat Z, X, Y the same
    nonid = paulis.z | paulis.x

    # bits are packed into uint8 in little endian
    # e.g., i-th bit corresponds to coefficient 2^i
    packed_vals = np.packbits(nonid, axis=1, bitorder="little").astype(object)
    power_uint8 = 1 << (8 * np.arange(packed_vals.shape[1], dtype=object))
    inds = packed_vals @ power_uint8
    return inds.tolist()


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


class CircuitCache:
    """
    Class to cache circuits in in-memory data store
    """

    def __init__(
        self,
        cache,
    ):
        """
        Args:
            cache: An instance of Cache class
        """
        self._cache = cache
        self._circuit_id_digest_map: Dict[str, str] = {}
        self._transpiled_circuits_map: Dict[str, QuantumCircuit | str] = {}
        self._transpiled_circuits: list[QuantumCircuit | str] = []
        self._raw_circuits = []
        self._circuit_ids = []
        self._circuit_digests = []
        self._backend_name: str = ""
        self._transpile_options: Dict = {}

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit | str]:
        """
        Returns a list of transpiled circuits or
        an intermediate list of transpiled circuits and circuit digests
        """
        return self._transpiled_circuits

    @property
    def raw_circuits(self) -> list[QuantumCircuit]:
        """
        Returns a list of raw (not transpiled) circuits
        """
        return self._raw_circuits

    def initialize_transpiled_and_raw_circuits(
        self,
        circuits_map: Dict[str, QuantumCircuit],
        circuit_ids: Sequence[str],
        backend_name: str,
        transpile_options: Dict,
    ) -> None:
        """
        Get transpiled circuits from in-memory datastore and also get raw circuits (not transpiled)
        """
        self._backend_name = backend_name
        self._transpile_options = transpile_options
        for circuit_id in circuit_ids:
            hash_str = self._construct_hash_str(
                circuit_id=circuit_id,
                backend_name=backend_name,
                transpile_options=transpile_options,
            )
            circuit_digest = self._hash(hash_str=hash_str)
            self._circuit_id_digest_map.update({circuit_id: circuit_digest})
            # Option 1: Try to get transpiled circuit from local map
            # (helps not to transpile repeated circuits in same job)
            transpiled_circuit = self._transpiled_circuits_map.get(circuit_digest)
            if transpiled_circuit:
                if transpiled_circuit == circuit_digest:
                    self._transpiled_circuits.append(circuit_digest)
                    continue
                self._transpiled_circuits.append(transpiled_circuit)
                continue
            # Option 2: Try to get transpiled circuit from in-memory data store
            # (helps not to transpile repeated circuits across jobs in a session)
            if self._cache:
                try:
                    transpiled_circuit = self._cache.get(key=circuit_digest)
                    self._transpiled_circuits.append(transpiled_circuit)
                    self._transpiled_circuits_map.update({circuit_digest: transpiled_circuit})
                    continue
                except Exception as exception:  #  pylint: disable=broad-except
                    logger.warning("Could not get transpiled circuit from cache. %s", exception)
            # Option 3: Try to get raw circuit from local map so it can be transpiled
            circuit = circuits_map.get(circuit_id)
            # Option 4: Try to get raw circuit from in-memory data store so it can be transpiled
            if not circuit:
                try:
                    hash_str = self._cache.get(circuit_id)
                    hash_obj = json.loads(hash_str, cls=RuntimeDecoder)
                    circuit = hash_obj.get("circuit")
                except Exception as exception:  #  pylint: disable=broad-except
                    logger.warning("Could not get raw circuit from cache. %s", exception)
            self._raw_circuits.append(circuit)
            self._circuit_ids.append(circuit_id)
            self._circuit_digests.append(circuit_digest)
            self._transpiled_circuits.append(circuit_digest)
            self._transpiled_circuits_map.update({circuit_digest: circuit_digest})

    def _construct_hash_str(
        self, circuit_id: str, backend_name: str, transpile_options: Dict
    ) -> str:
        """Construct str to hash using circuit_id, backend name and transpile_options."""
        hash_obj = {
            "circuit_id": circuit_id,
            "backend": backend_name,
            "transpile_options": transpile_options,
        }
        return json.dumps(hash_obj)

    def _hash(self, hash_str: str) -> str:
        """Hashes and returns a digest.
        blake2s is supposedly faster than SHAs.
        """
        return hashlib.blake2s(hash_str.encode()).hexdigest()

    def update_cache_and_merge_transpiled_circuits(
        self,
        transpiled_circuits: List[QuantumCircuit],
    ) -> None:
        """Update cache with
        * transpiled circuit
        * raw circuit + backend name + transpile_options.

        We put circuit_digest as placeholder in self._transpiled_circuits list for
        circuits not found in cache. This method also updates those placeholders in the list
        with the transpiled circuits from the map."""
        if self._circuit_digests:
            for i, transpiled_circuit in enumerate(transpiled_circuits):
                circuit_digest = self._circuit_digests[i]
                # Save transpiled circuit in local map
                self._transpiled_circuits_map.update({circuit_digest: transpiled_circuit})
                if self._cache:
                    # Save transpiled circuit in in-memory data store
                    self._cache.set(key=circuit_digest, value=transpiled_circuit)
                    # Save raw circuit in in-memory data store so it can be used later for transpilation
                    # if different transpile options are passed
                    raw_circuit_hash_str = self._construct_raw_circuit_hash_str(
                        circuit=self._raw_circuits[i],
                        backend_name=self._backend_name,
                        transpile_options=self._transpile_options,
                    )
                    self._cache.set(key=self._circuit_ids[i], value=raw_circuit_hash_str)
        for i, transpiled_circuit in enumerate(self._transpiled_circuits):
            # Check if it is a circuit digest string
            if isinstance(transpiled_circuit, str):
                # Replace circuit digest string with
                self._transpiled_circuits[i] = self._transpiled_circuits_map.get(transpiled_circuit)

    def _construct_raw_circuit_hash_str(
        self,
        circuit: QuantumCircuit,
        backend_name: str,
        transpile_options: Dict,
    ) -> str:
        """Construct hash string based on circuit, backend name and transpile options."""
        hash_obj = {
            "circuit": circuit,
            "backend": backend_name,
            "transpile_options": transpile_options,
        }
        return json.dumps(hash_obj, cls=RuntimeEncoder)

    def get_transpiled_circuit(self, circuit_id: str) -> QuantumCircuit:
        """Return QuantumCircuit for circuit id"""
        circuit_digest = self._circuit_id_digest_map.get(circuit_id)
        return self._transpiled_circuits_map.get(circuit_digest)


################################################################################
## DYNAMICAL DECOUPLING
################################################################################
def dynamical_decoupling_pass(backend: Union[BackendV1, BackendV2]) -> Optional[PassManager]:
    """Generates a pass manager of the dynamical decoupling

    Note that this pass is supposed to be applied to bound circuits

    Args:
        backend: the backend to execute the input circuits

    Returns:
        PassManager: the pass manager of the dynamical decoupling
    """
    # Source:
    # https://github.ibm.com/IBM-Q-Software/ntc-ibm-programs/issues/213
    # https://github.ibm.com/IBM-Q-Software/pec-runtime/blob/f8f0a49ee18eda9754734dd3260ea8c8812ee342/pec_runtime/utils/dynamical_decoupling.py#L47
    #
    # Note: ProgramBackend used to be BackendV1
    # https://github.ibm.com/IBM-Q-Software/ntc-provider/blob/efa7eaedc92a7a022aba237a00c63886678c1ac4/programruntime/runtime_backend.py#L31
    # https://github.com/Qiskit/qiskit-ibm-runtime/blob/af308caeb7c261a1fb1a7ca7a45c49f55df02215/qiskit_ibm_runtime/program/program_backend.py#L20

    try:
        if isinstance(backend, BackendV2):
            target = backend.target
            durations = target.durations()
            timing_constraints = target.timing_constraints()
        else:
            target = None
            durations = InstructionDurations.from_backend(backend)
            timing_constraints = TimingConstraints(**backend.configuration().timing_constraints)
    except AttributeError:
        logger.warning("Backend (%s) does not support dynamical decoupling.", backend)
        return None

    # This can be removed after qiskit-terra 0.24 is released and installed.
    extra_params = {}
    version_parts = qiskit.__version__.split(".")
    if (int(version_parts[0]) == 0 and int(version_parts[1]) >= 24) or int(version_parts[0]) > 0:
        extra_params["target"] = target

    dd_sequence = [XGate(), RZGate(np.pi), XGate(), RZGate(-np.pi)]
    spacing = [1 / 4, 1 / 2, 0, 0, 1 / 4]
    schedule_pass = ALAPScheduleAnalysis(durations, **extra_params)

    return PassManager(
        [
            TimeUnitConversion(durations, **extra_params),
            schedule_pass,
            InstructionDurationCheck(
                acquire_alignment=timing_constraints.acquire_alignment,
                pulse_alignment=timing_constraints.pulse_alignment,
            ),
            ConstrainedReschedule(
                acquire_alignment=timing_constraints.acquire_alignment,
                pulse_alignment=timing_constraints.pulse_alignment,
            ),
            PadDynamicalDecoupling(
                durations=durations,
                dd_sequence=dd_sequence,
                spacing=spacing,
                pulse_alignment=timing_constraints.pulse_alignment,
                **extra_params,
            ),
        ]
    )


################################################################################
## PEC
################################################################################


################################################################################
## T-REX (logic in Estimator as well)
################################################################################
class PauliTwirledMitigation:
    """
    Pauli twirled readout error mitigation (T-Rex)
    """

    def __init__(
        self,
        backend: Backend,
        num_twirled_circuits: int = 16,
        shots_calibration: int = 8192,
        seed: int | None = None,
        **cal_run_options,
    ):
        self._backend = backend
        if num_twirled_circuits % 2 == 1:
            logger.warning(
                "Number of twirled circuits should be even, but it is (%d). It will be increased by 1.",
                num_twirled_circuits,
            )
            num_twirled_circuits += 1
        self._num_twirled_circuits = num_twirled_circuits
        self._shots_calibration = shots_calibration
        if seed is None or isinstance(seed, int):
            self._rng = np.random.default_rng(seed)
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

    def _random_bits(self, qubits: Sequence[int]):
        bits = self._rng.choice(2, (self._num_twirled_circuits // 2, len(qubits)))
        bits = np.concatenate([bits, 1 - bits])
        return bits

    def _append_x_and_measure(self, circ: QuantumCircuit, qubits: Sequence[int], bits: np.ndarray):
        # `bits` represents a list of integers (0 or 1)
        flip = np.where(bits == 1)[0]
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
        for bits in self._random_bits(qubits):
            circ = QuantumCircuit(max(qubits) + 1)
            circ = self._append_x_and_measure(circ, qubits, bits)
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
            for bits in self._random_bits(qubits):
                circ2 = self._append_x_and_measure(circ.copy(), qubits, bits)
                circuits2.append(circ2)
        shots2 = self._subdivide_shots(shots, self._num_twirled_circuits)
        return run_circuits(circuits2, backend=self._backend, shots=shots2, **options)


################################################################################
## ZNE
################################################################################
DEFAULT_NOISE_FACTORS: Sequence[float] = (1, 3, 5)
DEFAULT_AMPLIFIER_SETTING: str = "TwoQubitAmplifier"
DEFAULT_EXTRAPOLATOR_SETTING: str = "LinearExtrapolator"


def zne_strategy_from_settings(settings: dict) -> ZNEStrategy:
    """Build ZNEStrategy object from client settings."""
    noise_factors: Sequence[float] = settings.get("noise_factors") or DEFAULT_NOISE_FACTORS
    amplifier_setting: str = settings.get("noise_amplifier") or DEFAULT_AMPLIFIER_SETTING
    noise_amplifier: NoiseAmplifier = noise_amplifier_from_setting(amplifier_setting)
    extrapolator_setting: str = settings.get("extrapolator") or DEFAULT_EXTRAPOLATOR_SETTING
    extrapolator: Extrapolator = extrapolator_from_setting(extrapolator_setting)
    if len(noise_factors) < extrapolator.min_points:
        raise ValueError(
            f"Invalid setting: {extrapolator_setting!r} requires at least "
            f"{extrapolator.min_points} noise factors but only {len(noise_factors)} were given: "
            f"{noise_factors}."
        )
    return ZNEStrategy(
        noise_factors=noise_factors,
        noise_amplifier=noise_amplifier,
        extrapolator=extrapolator,
    )


def noise_amplifier_from_setting(setting: str) -> NoiseAmplifier:
    """Build NoiseAmplifier object from str setting."""
    if not isinstance(setting, str):
        raise TypeError(
            f"Expected `str` type for noise amplifier setting, got `{type(setting)}` instead."
        )
    cls = NOISE_AMPLIFIER_LIBRARY.get(setting)
    if cls is None:
        raise ValueError(f"Invalid noise amplifier setting '{setting}'.")
    return cls()


def extrapolator_from_setting(setting: str) -> Extrapolator:
    """Build Extrapolator object from str setting."""
    if not isinstance(setting, str):
        raise TypeError(
            f"Expected `str` type for extrapolator setting, got `{type(setting)}` instead."
        )
    cls = EXTRAPOLATOR_LIBRARY.get(setting)
    if cls is None:
        raise ValueError(f"Invalid extrapolator setting '{setting}'.")
    return cls()


################################################################################
## ESTIMATOR CONSTANTS
################################################################################
class EstimatorConstant:
    """Class to capture Primitives constants"""

    TREX_RESILIENCE_LEVEL: int = 1
    ZNE_RESILIENCE_LEVEL: int = 2
    PEC_RESILIENCE_LEVEL: int = 3

    DEFAULT_RESILIENCE_LEVEL: int = TREX_RESILIENCE_LEVEL

    PEC_DEFAULT_NUM_SAMPLES: int = 1024
    PEC_DEFAULT_MAX_CIRCUITS: int = 100
    PEC_DEFAULT_MAX_LEARNING_LAYERS: int = 4
    PEC_DEFAULT_SHOTS_PER_SAMPLE: int = 128
    PEC_DEFAULT_MAX_SAMPLING_OVERHEAD: int = 100
    PEC_DEFAULT_CACHE_TIMEOUT: timedelta = timedelta(hours=2)

    INVALID_QASM_VERSION_MESSAGE = (
        "OpenQASM version invalid or not specified in program, will use OpenQASM 3."
    )


################################################################################
## MAIN
################################################################################

# Move this function to a "commons" file when in new repo
def str_to_quantum_circuit(program: str) -> QuantumCircuit:
    """Converts a QASM program to a QuantumCircuit object. Depending on the
    OpenQASM version of the program, it will use either
    `QuantumCircuit.from_qasm_str` or `qiskit.qasm3.loads`.
    If no OpenQASM version is specified in the header of the program, then it's
    assumed to be an OpenQASM3 program.
    Args:
        program: a OpenQASM program as a string
    Returns:
        QuantumCircuit: the input OpenQASM program as a quantum circuit object
    """
    match = re.search(r"OPENQASM\s+(\d+)(\.(\d+))*", program)
    try:
        if match is None:
            # Issue a warning and try using OpenQASM3 if version was invalid or not specified
            logger.warning(EstimatorConstant.INVALID_QASM_VERSION_MESSAGE)
            return qasm3_loads(program)
        else:
            qasm_version = match.group(1)
            if float(qasm_version) == 2:
                # OpenQASM2
                return QuantumCircuit.from_qasm_str(program)
            else:  # version 3 and other versions
                # use default OpenQASM3 loads
                return qasm3_loads(program)
    # catch all exceptions from openqasm3.parser.*, qiskit.qasm.exceptions.*, qiskit.qasm3.exceptions.*
    except Exception as qasm_error:  #  pylint: disable=broad-except
        raise QiskitError(f"Error parsing OpenQASM program. {getattr(qasm_error, 'msg', '')}")


def _restore_circuits(circuits, circuit_indices, circuit_ids, backend) -> list[QuantumCircuit]:
    if isinstance(circuits, dict):
        circuit_list = _circuit_dict_to_list(circuits, backend, circuit_ids)
    elif circuit_indices and isinstance(circuits, list):
        circuit_list = [circuits[i] for i in circuit_indices]
    else:
        circuit_list = [circuits]
    return circuit_list


def _circuit_dict_to_list(circuits: dict, backend, circuit_ids) -> list:
    try:
        _provider = backend.provider if backend.version == 2 else backend.provider()
        cache = _provider.cache()

        for circuit_id in circuit_ids:
            if circuit_id not in circuits:
                hash_str = cache.get(circuit_id)
                hash_obj = json.loads(hash_str, cls=RuntimeDecoder)
                circuit = hash_obj.get("circuit")
                circuits[circuit_id] = circuit
    except AttributeError:
        # Unit tests use AerProvider which doesn't have cache() method
        pass

    return [circuits[circuit_id] for circuit_id in circuit_ids]


# TODO: Remove this function when decoder supports numpy types.
# See https://github.ibm.com/IBM-Q-Software/ntc-provider/issues/600
def _scrub_numpy(obj):
    """Remove numpy from an object or container"""
    # pylint: disable = too-many-return-statements
    if isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return _scrub_numpy(obj.tolist())
        else:
            return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _scrub_numpy(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [_scrub_numpy(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(_scrub_numpy(i) for i in obj)
    return obj


def result_to_dict(result: EstimatorResult):
    """Convert ``EstimatorResult`` to a dictionary

    Args:
        result: The result of ``Estimator``

    Returns:
        A dictionary representing the result.

    """
    values = tuple(result.values.tolist())
    ret = _scrub_numpy(result.__dict__)
    ret["values"] = values
    return ret


def main(
    backend,
    user_messenger,  # pylint: disable=unused-argument
    circuits,
    observables,
    observable_indices,
    circuit_indices=None,
    parameters=None,
    parameter_values=None,
    skip_transpilation=False,
    run_options=None,
    transpilation_settings=None,
    resilience_settings=None,
    circuit_ids=None,
):
    """Estimator primitive.

    Args:
        backend: Backend to run the circuits.
        user_messenger: Used to communicate with the user.
        circuits: (QuantumCircuit or list or dict): A single or list or dictionary of QuantumCircuits.
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
        circuit_ids (list): A list of unique IDs of QuantumCircuits.

    Returns: Expectation values and metadata.

    """
    if DEBUG:
        logger.info("Debug mode")

    # Settings and options
    transpilation_settings = transpilation_settings or {}
    skip_transpilation = transpilation_settings.pop("skip_transpilation", skip_transpilation)
    optimization_settings = transpilation_settings.pop("optimization_settings", {})
    resilience_settings = resilience_settings or {}
    run_options = run_options or {}
    if "shots" not in run_options:
        run_options["shots"] = backend.options.shots

    # Configure noise model
    # TODO: Is this necessary? Aer can be passed seed and noise model as run options
    # to backend.run, unless the IBM hosted Aer sim does something weird with run
    # options it should work for it too
    noise_model = run_options.pop("noise_model", getattr(backend.options, "noise_model", None))
    seed_simulator = run_options.pop(
        "seed_simulator", getattr(backend.options, "seed_simulator", None)
    )
    if hasattr(backend, "configuration") and backend.configuration().simulator:
        backend.set_options(noise_model=noise_model, seed_simulator=seed_simulator)

    # Configure transpilation
    transpile_options = transpilation_settings.copy()
    optimization_level = optimization_settings.get("level", 1)
    transpile_options["optimization_level"] = optimization_level
    if optimization_level >= 1 and not skip_transpilation:
        bound_pass_manager = dynamical_decoupling_pass(backend)
    else:
        bound_pass_manager = None

    # Execute for different resilience levels
    resilience_level = resilience_settings.get("level", EstimatorConstant.DEFAULT_RESILIENCE_LEVEL)
    if resilience_level == 0:  # None
        estimator = Estimator(
            backend=backend,
            circuits=circuits,
            observables=observables,
            parameters=parameters,
            skip_transpilation=skip_transpilation,
            pauli_twirled_mitigation=None,
            circuit_ids=circuit_ids,
            bound_pass_manager=bound_pass_manager,
        )
        estimator.set_transpile_options(**transpile_options)
        result = estimator.run(
            circuit_indices=circuit_indices,
            observable_indices=observable_indices,
            parameter_values=parameter_values,
            **run_options,
        )
    elif resilience_level == EstimatorConstant.TREX_RESILIENCE_LEVEL:  # T-REX
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
            circuit_ids=circuit_ids,
            bound_pass_manager=bound_pass_manager,
        )
        estimator.set_transpile_options(**transpile_options)
        result = estimator.run(
            circuit_indices=circuit_indices,
            observable_indices=observable_indices,
            parameter_values=parameter_values,
            **run_options,
        )
    elif resilience_level == EstimatorConstant.ZNE_RESILIENCE_LEVEL:  # ZNE
        ZNEEstimator = zne(BackendEstimator)  # pylint: disable=invalid-name
        zne_strategy = zne_strategy_from_settings(resilience_settings)
        estimator = ZNEEstimator(
            backend=backend,
            skip_transpilation=skip_transpilation,
            bound_pass_manager=bound_pass_manager,
            zne_strategy=zne_strategy,
        )
        estimator.set_transpile_options(**transpile_options)
        circuit_list = _restore_circuits(circuits, circuit_indices, circuit_ids, backend)
        observable_list = [observables[i] for i in observable_indices]
        job = estimator.run(
            circuits=circuit_list,
            observables=observable_list,
            parameter_values=parameter_values,
            **run_options,
        )
        result = job.result()
        for metadatum in result.metadata:
            zne_md = metadatum.get("zne")
            if zne_md:
                zne_md["noise_amplification"]["noise_amplifier"] = repr(
                    zne_md["noise_amplification"]["noise_amplifier"]
                )
                zne_md["extrapolation"]["extrapolator"] = repr(
                    zne_md["extrapolation"]["extrapolator"]
                )
    elif resilience_level == EstimatorConstant.PEC_RESILIENCE_LEVEL:  # PEC
        if isinstance(backend, BackendV1):
            transpilation_settings["basis_gates"] = backend.configuration().basis_gates
        else:
            # BackendV2
            transpilation_settings["basis_gates"] = list(backend.target.operation_names)
        circuit_list = _restore_circuits(circuits, circuit_indices, circuit_ids, backend)
        observable_list = [observables[i] for i in observable_indices]

        # PEC default values
        pec_resilience_settings = {
            "level": EstimatorConstant.PEC_RESILIENCE_LEVEL,
            "max_circuits": EstimatorConstant.PEC_DEFAULT_MAX_CIRCUITS,
            "max_learning_layers": EstimatorConstant.PEC_DEFAULT_MAX_LEARNING_LAYERS,
            "shots_per_sample": EstimatorConstant.PEC_DEFAULT_SHOTS_PER_SAMPLE,
            "max_sampling_overhead": EstimatorConstant.PEC_DEFAULT_MAX_SAMPLING_OVERHEAD,
            "cache_timeout": EstimatorConstant.PEC_DEFAULT_CACHE_TIMEOUT,
        }
        # Override with any supplied values
        pec_resilience_settings.update(**resilience_settings)

        estimator = PEC_Estimator(
            backend=backend,
            circuits=circuit_list,
            observables=observable_list,
            parameters=parameters,
            skip_transpilation=skip_transpilation,
            resilience_settings=pec_resilience_settings,
            transpilation_settings=transpilation_settings,
            user_messenger=user_messenger,
        )

        # No need to set traspile options here, PEC transpile options are set in
        # the class constructor
        shots = run_options.pop("shots", EstimatorConstant.PEC_DEFAULT_NUM_SAMPLES)
        result = estimator.run(
            circuit_indices=circuit_indices,
            observable_indices=observable_indices,
            parameter_values=parameter_values,
            shots=shots,
            **run_options,
        )

        result = EstimatorResult(result["values"], result["metadata"])
    else:
        raise QiskitError(f"Resilience level {resilience_level} not supported.")

    return result_to_dict(result)
