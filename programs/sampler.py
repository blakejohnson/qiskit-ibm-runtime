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
Sampler class
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from collections.abc import Iterable, Sequence
from os import environ
from typing import Dict, List, Optional, cast, Union

import numpy as np
from mthree import M3Mitigation
from mthree.classes import QuasiDistribution as M3QuasiDistribution
from mthree.utils import final_measurement_mapping, marginal_distribution
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RZGate, XGate
from qiskit.circuit.parametertable import ParameterView
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.primitives import SamplerResult
from qiskit.providers import BackendV1, BackendV2, Options
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.qasm3 import loads as qasm3_loads
from qiskit.result import Counts, QuasiDistribution, Result
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

QuantumProgram = Union[QuantumCircuit, str]

# Number of effective shots per measurement error rate
DEFAULT_SHOTS = 25000

logger = logging.getLogger(__name__)

# If PRIMITIVES_DEBUG is True, metadata includes bound circuits.
# Only for internal development and test.
DEBUG = environ.get("PRIMITIVES_DEBUG", "false") == "true"


class Sampler:
    """
    Sampler class
    """

    def __init__(
        self,
        backend: Backend,
        circuits: QuantumProgram | Iterable[QuantumProgram] | Dict[str, QuantumProgram],
        parameters: Iterable[Iterable[Parameter]] | None = None,
        bound_pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
        circuit_ids: Sequence[str] = None,
    ):
        """
        Args:
            circuits: circuits to be executed
        Raises:
            TypeError: The given type of arguments is invalid.
        """
        self._backend = backend
        self._circuit_ids: Sequence[str] = circuit_ids
        self._transpile_options = Options()
        self._circuits_map: Dict[str, QuantumProgram] = {}
        self._circuits = self._get_circuits(circuits=circuits)
        self._parameters = parameters
        self._circuit_cache = CircuitCache(cache=self._get_cache())
        self._run_options = Options()
        self._is_closed = False
        self._bound_pass_manager = bound_pass_manager
        self._transpiled_circuits: list[QuantumCircuit] | None = None
        self._skip_transpilation = skip_transpilation
        self._m3_mitigation: M3Mitigation | None = None

    def _get_circuits(
        self, circuits: QuantumProgram | Iterable[QuantumProgram] | Dict[str, QuantumProgram]
    ):
        """Return list of QuanutmCircuit circuits."""

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
            circuits = (circuits,)
            return list(circuits)
        elif isinstance(circuits, Dict) or self._circuit_ids is not None:
            self._circuits_map = circuits  # type: ignore
            return list(circuits.values())  # type: ignore
        else:
            return [] if circuits is None else list(circuits)

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

    @property
    def preprocessed_circuits(self) -> list[QuantumCircuit]:
        """
        Preprocessed quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        # This method is no longer used in the new flexible session flow
        # with circuits and circuit_ids, and hence can be removed when
        # we remove support for circuit_indices
        return list(self._circuits)

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
                # 1. Initialize a list of transpiled circuits from cache and another list of
                # raw circuits whose transpiled versions were not found in cache
                self._circuit_cache.initialize_transpiled_and_raw_circuits(
                    circuits_map=self._circuits_map,
                    circuit_ids=self._circuit_ids,
                    backend_name=_backend_name,
                    transpile_options=self._transpile_options.__dict__,
                )
                if self._circuit_cache.raw_circuits:
                    if not self._skip_transpilation:
                        # 2. Transpile the raw circuits whose transpiled versions were not found in cache
                        transpiled_circuits = self._transpile(
                            circuits=self._circuit_cache.raw_circuits
                        )
                    else:
                        transpiled_circuits = self._circuit_cache.raw_circuits
                    # 3. Update cache with transpiled and raw circuits and merge transpiled circuits
                    # from step 2 with transpiled circuits retrieved from cache in step 1
                    self._circuit_cache.update_cache_and_merge_transpiled_circuits(
                        transpiled_circuits=transpiled_circuits,
                    )
                self._transpiled_circuits = self._circuit_cache.transpiled_circuits
            else:
                if not self._skip_transpilation:
                    self._transpiled_circuits = self._transpile(circuits=self.preprocessed_circuits)
                else:
                    self._transpiled_circuits = list(self._circuits)
        return self._transpiled_circuits

    def _transpile(self, circuits: List[QuantumCircuit]):
        """Transpile given circuits in parallel. Calling transpile on multiple circuits is faster
        than calling transpile once for each circuit."""
        transpiled_circuits = cast(
            "list[QuantumCircuit]",
            transpile(
                circuits,
                self._backend,
                **self.transpile_options.__dict__,
            ),
        )
        return transpiled_circuits

    @property
    def backend(self) -> Backend:
        """
        Returns:
            The backend which this sampler object based on
        """
        return self._backend

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """Quantum circuits to be sampled.

        Returns:
            The quantum circuits to be sampled.
        """
        return self._circuits

    @property
    def run_options(self) -> Options:
        """Return options values for the evaluator.
        Returns:
            run_options
        """
        return self._run_options

    def set_run_options(self, **fields) -> Sampler:
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

    def set_transpile_options(self, **fields) -> Sampler:
        """Set the transpiler options for transpiler.
        Args:
            **fields: The fields to update the options.
        Returns:
            self.
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._transpile_options.update_options(**fields)
        return self

    def run(
        self,
        circuit_indices: Sequence[int] = None,
        parameter_values: Sequence[Sequence[float]] = None,
        **run_options,
    ) -> SamplerResult:
        """Run the sampling of bitstrings."""
        # TODO remove this if block when non-flexible sessions interface is no longer supported
        if not self._circuit_ids:
            self._parameters = self._initialize_parameters(
                circuits=self._circuits,
                parameters=self._parameters,
            )
            self._validate_parameters(
                circuits=self._circuits,
                parameters=self._parameters,
            )
            self._validate_circuit_indices(
                circuits=self._circuits,
                circuit_indices=circuit_indices,
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
            # This line does the actual transpilation
            transpiled_circuits = self.transpiled_circuits
            bound_circuits = [
                transpiled_circuits[i]
                if len(value) == 0
                else transpiled_circuits[i].bind_parameters(
                    (dict(zip(self._parameters[i], value)))  # type: ignore
                )
                for i, value in zip(circuit_indices, parameter_values)
            ]
        else:
            # This line does the actual transpilation
            transpiled_circuits = self.transpiled_circuits
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
            bound_circuits = [
                transpiled_circuits[i]
                if len(value) == 0
                else transpiled_circuits[i].bind_parameters(
                    (dict(zip(self._parameters[i], value)))  # type: ignore
                )
                for i, value in zip(range(len(self._circuit_ids)), parameter_values)
            ]

        bound_circuits = self._bound_pass_manager_run(bound_circuits)

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        result = self._backend.run(bound_circuits, **run_opts.__dict__).result()

        return self._postprocessing(result, bound_circuits)

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

    def _validate_circuit_indices(
        self,
        circuits: List[QuantumCircuit],
        circuit_indices: Sequence[int] = None,
    ) -> None:
        if max(circuit_indices) >= len(circuits):
            raise QiskitError(
                f"The number of circuits is {len(circuits)}, "
                f"but the index {max(circuit_indices)} is given."
            )

    def _initialize_parameter_values(
        self,
        circuits_len: int,
        circuits: List[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]] = None,
    ) -> Sequence[Sequence[float]]:
        # Support ndarray
        if isinstance(parameter_values, np.ndarray):
            return parameter_values.tolist()
        # Allow optional
        elif parameter_values is None:
            for i, circuit in enumerate(circuits):
                if circuit.num_parameters != 0:
                    raise QiskitError(
                        f"The {i}-th circuit ({len(circuits)}) is parameterized, "
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

    def close(self):
        """Close the session and free resources"""
        self._is_closed = True

    def _postprocessing(self, result: Result, circuits: list[QuantumCircuit]) -> SamplerResult:
        if not isinstance(result, Result):
            raise TypeError("result must be an instance of Result.")

        counts = result.get_counts()
        if not isinstance(counts, list):
            counts = [counts]

        shots = sum(counts[0].values())

        quasis = []
        mitigation_overheads = []
        mitigation_times = []
        for count, circ in zip(counts, circuits):
            # hack for separated cregs
            count = Counts({key.replace(" ", ""): val for key, val in count.items()})

            if self._m3_mitigation is None:
                quasis.append(QuasiDistribution({k: v / shots for k, v in count.items()}))
            else:
                quasi, details = self._apply_correction(count, circ)
                quasis.append(QuasiDistribution(quasi))
                mitigation_overheads.append(quasi.mitigation_overhead)
                mitigation_times.append(details["time"])

        metadata = []
        for idx, _ in enumerate(result.results):
            _temp_dict = {"shots": shots}
            if self._m3_mitigation:
                _temp_dict["readout_mitigation_overhead"] = mitigation_overheads[idx]
                _temp_dict["readout_mitigation_time"] = mitigation_times[idx]
            if DEBUG:
                _temp_dict["bound_circuits"] = [circuits[idx]]
            metadata.append(_temp_dict)

        return SamplerResult(quasi_dists=quasis, metadata=metadata)

    def _apply_correction(
        self, counts: Counts, circuit: QuantumCircuit
    ) -> tuple[M3QuasiDistribution, dict]:
        mapping = final_measurement_mapping(circuit)
        used_clbits = set(mapping.keys())
        all_clbits = set(range(circuit.num_clbits))
        if used_clbits != all_clbits:
            unused_clbits = list(all_clbits - used_clbits)
            reduced_counts, reduced_mapping = marginal_distribution(
                counts, sorted(used_clbits), mapping
            )
            quasi, details = self._m3_mitigation.apply_correction(
                reduced_counts, reduced_mapping, return_mitigation_overhead=True, details=True
            )
            quasi = self._expand_keys(quasi, unused_clbits)
        else:
            quasi, details = self._m3_mitigation.apply_correction(
                counts, mapping, return_mitigation_overhead=True, details=True
            )
        return quasi, details

    def _expand_keys(
        self, quasi: M3QuasiDistribution, unused_clbits: list[int]
    ) -> M3QuasiDistribution:
        """fill '0' to unused qubits"""

        def _expand(key: str):
            lst = list(key[::-1])
            for i in sorted(unused_clbits):
                lst.insert(i, "0")
            return "".join(lst[::-1])

        return M3QuasiDistribution(
            {_expand(key): val for key, val in quasi.items()},
            shots=quasi.shots,
            mitigation_overhead=quasi.mitigation_overhead,
        )

    def _bound_pass_manager_run(self, circuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
        if self._bound_pass_manager is None:
            return circuits
        circs = self._bound_pass_manager.run(circuits)
        return circs if isinstance(circs, list) else [circs]

    def calibrate_m3_mitigation(self, backend) -> None:
        """Calibrate M3 mitigation

        Args:
            backend: The backend.
        """
        mappings = [final_measurement_mapping(circ) for circ in self.transpiled_circuits]
        self._m3_mitigation = M3Mitigation(backend)
        self._m3_mitigation.cals_from_system(mappings, shots=DEFAULT_SHOTS)

    @staticmethod
    def result_to_dict(
        result: SamplerResult, circuits, circuit_indices, transpiled_circuits, circuit_ids
    ):
        """Convert ``SamplerResult`` to a dictionary

        Args:
            result: The result of ``Sampler``
            circuits: The circuits
            circuit_indices: The circuit indices

        Returns:
            A dictionary representing the result.

        """
        ret = result.__dict__
        if circuit_indices:
            ret["quasi_dists"] = [
                dist.binary_probabilities(circuits[index].num_clbits)
                for index, dist in zip(circuit_indices, result.quasi_dists)
            ]
        else:
            ret["quasi_dists"] = [
                dist.binary_probabilities(transpiled_circuits[index].num_clbits)
                for index, dist in zip(range(len(circuit_ids)), result.quasi_dists)
            ]
        return ret


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
            durations = InstructionDurations.from_backend(backend)
            timing_constraints = TimingConstraints(**backend.configuration().timing_constraints)
    except AttributeError:
        logger.warning("Backend (%s) does not support dynamical decoupling.", backend)
        return None
    dd_sequence = [XGate(), RZGate(np.pi), XGate(), RZGate(-np.pi)]
    spacing = [1 / 4, 1 / 2, 0, 0, 1 / 4]

    schedule_pass = ALAPScheduleAnalysis(durations)

    return PassManager(
        [
            TimeUnitConversion(durations),
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
            ),
        ]
    )


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
            logger.warning(SamplerConstant.INVALID_QASM_VERSION_MESSAGE)
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


################################################################################
## SAMPLER CONSTANTS
################################################################################
class SamplerConstant:
    """Class to capture Sampler constants"""

    INVALID_QASM_VERSION_MESSAGE = (
        "OpenQASM version invalid or not specified in program, will use OpenQASM 3."
    )


def main(
    backend,
    user_messenger,  # pylint: disable=unused-argument
    circuits,
    circuit_indices=None,
    parameters=None,
    parameter_values=None,
    skip_transpilation=False,
    run_options=None,
    transpilation_settings=None,
    resilience_settings=None,
    circuit_ids=None,
    **kwargs,  # pylint: disable=unused-argument
):

    """Sample distributions generated by given circuits executed on the target backend.

    Parameters:
        backend (ProgramBackend): Qiskit backend instance.
        user_messenger (UserMessenger): Used to communicate with the program user.
        circuits: (QuantumCircuit or list or dict): A single or list or dictionary of QuantumCircuits.
        parameters (list): Parameters of the quantum circuits.
        circuit_indices (list): Indexes of the circuits to evaluate.
        parameter_values (list): Concrete parameters to be bound.
        skip_transpilation (bool): (Deprecated) Skip transpiling of circuits, default=False.
        run_options (dict): A collection of kwargs passed to backend.run().
        transpilation_settings (dict): Transpilation settings.
        resilience_settings (dict): Resilience settings.
        circuit_ids (list): A list of unique IDs of QuantumCircuits.
        kwargs (dict): Temporary solution to make flexible session work. TO BE REMOVED.

    Returns:
        dict: A dictionary with quasi-probabilities and metadata.
    """
    transpilation_settings = transpilation_settings or {}
    optimization_settings = transpilation_settings.pop("optimization_settings", {})
    skip_transpilation = transpilation_settings.pop("skip_transpilation", skip_transpilation)
    run_options = run_options or {}

    # Configure noise model.
    noise_model = run_options.pop("noise_model", None)
    seed_simulator = run_options.pop("seed_simulator", None)
    if hasattr(backend, "configuration") and backend.configuration().simulator:
        backend.set_options(noise_model=noise_model, seed_simulator=seed_simulator)

    transpile_options = transpilation_settings.copy()
    optimization_level = optimization_settings.get("level", 1)
    transpile_options["optimization_level"] = optimization_level

    if optimization_level >= 1 and not skip_transpilation:
        bound_pass_manager = dynamical_decoupling_pass(backend)
    else:
        bound_pass_manager = None

    sampler = Sampler(
        backend=backend,
        circuits=circuits,
        parameters=parameters,
        skip_transpilation=skip_transpilation,
        circuit_ids=circuit_ids,
        bound_pass_manager=bound_pass_manager,
    )

    sampler.set_transpile_options(**transpile_options)
    # Must transpile circuits before calibrating M3
    transpiled_circuits = sampler.transpiled_circuits

    resilience_settings = resilience_settings or {}

    if resilience_settings.get("level", 0) == 1:
        sampler.calibrate_m3_mitigation(backend)

    result = sampler.run(
        circuit_indices=circuit_indices,
        parameter_values=parameter_values,
        **run_options,
    )

    result_dict = sampler.result_to_dict(
        result, sampler.circuits, circuit_indices, transpiled_circuits, circuit_ids
    )

    return result_dict
