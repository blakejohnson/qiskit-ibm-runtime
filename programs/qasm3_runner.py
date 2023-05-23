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
"""A runtime program that takes one or more circuits, converts them to OpenQASM3,
compiles them, executes them, and optionally applies measurement error mitigation.
By default, multiple circuits will be merge into one before converting the
combined circuit into OpenQASM3.
This program can also take and execute one or more OpenQASM3 strings. Note that this
program can only run on a backend that supports OpenQASM3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from qiskit.circuit.library import Barrier
from qiskit.circuit.quantumcircuit import ClassicalRegister, Delay, QuantumCircuit, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.qasm3 import Exporter, ExperimentalFeatures
from qiskit.result import Result
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import TimeUnitConversion
from qiskit_ibm_runtime.utils import RuntimeEncoder
import numpy as np


logger = logging.getLogger(__name__)


# fix rep_delay in shot loop to 0 since we manually insert
# TODO: while we await https://github.ibm.com/IBM-Q-Software/ibm-qss-compiler/issues/889
# set to 200us see:
# https://ibm-quantumcomputing.slack.com/archives/G01C867NNT1/p1667482191568199
QSS_COMPILER_REP_DELAY = 200e-6

QASM3_SIM_NAME = "simulator_qasm3"
QASM2_SIM_NAME = "qasm_simulator"
SIMULATORS = (QASM3_SIM_NAME, QASM2_SIM_NAME)


class ConvertNearestMod16Delay(TransformationPass):
    """Convert delay to the nearest mod 16 delay."""

    def run(self, dag: DAGCircuit):
        for node in dag.op_nodes():
            if isinstance(node.op, Delay):
                node.op = node.op.copy()
                node.op.duration = self._nearest_mod_16_duration(node.op.duration)

        return dag

    def _nearest_mod_16_duration(self, duration: int) -> int:
        remainder = duration % 16
        return duration + (16 - remainder)


class CircuitMerger:
    """Utility class for submitting multiple QuantumCircuits for execution as
    a single circuit, and splitting the results from the joint execution.
    """

    def __init__(
        self,
        circuits: List[QuantumCircuit],
        backend,
    ):
        self.circuits = circuits
        self.backend = backend

    class InitStrategy(Enum):
        """Available initialization strategies.

        See this experiment notebook for how this strategy and value
        was chosen.
        https://github.ibm.com/IBM-Q-Software/ws-dynamic-circuits/blob/c6f1f4995c3311b5cf3cd64d48c7f8f19f02aaf8/docs/experiments/qubit_initialization_strategies.ipynb
        """

        BEFORE = "before"
        INTERSPERSED_AFTER = "interspersed_after"

    def _create_init_circuit(
        self,
        used_qubits: Iterable[int],
        init_num_resets: int,
        init_delay: float,
        init_delay_unit: str,
        init_strategy: InitStrategy = InitStrategy.INTERSPERSED_AFTER,
    ) -> QuantumCircuit:
        """Create a parameterized initialization circuit or return the
        user-provided initialization circuit.
        """
        if init_strategy == self.InitStrategy.BEFORE:
            return self._init_strategy_before(
                used_qubits, init_num_resets, init_delay, init_delay_unit
            )
        elif init_strategy == self.InitStrategy.INTERSPERSED_AFTER:
            return self._init_strategy_interspersed_after(
                used_qubits, init_num_resets, init_delay, init_delay_unit
            )
        else:
            raise ValueError(f"Initialization strategy of {init_strategy} is not recognized.")

    def _get_dt(self) -> float:
        """Get the dt conversion factor for the target backend."""
        try:
            return self.backend.configuration().dt
        except AttributeError:
            return 1e-9

    def _init_strategy_interspersed_after(
        self,
        used_qubits: Iterable[int],
        init_num_resets: int,
        init_delay: float,
        init_delay_unit: str,
    ) -> QuantumCircuit:
        """Intersperse delay after resets."""
        n_qubits = self.backend.configuration().n_qubits
        circuit = QuantumCircuit(n_qubits)

        # Only reset qubits that are used in the experiment to reduce
        # initialization time and replicate current backend behaviour
        circuit.barrier(used_qubits)
        if init_num_resets > 0:
            delay_per_round = init_delay / init_num_resets
            for _ in range(0, init_num_resets):
                circuit.reset(used_qubits)
                circuit.barrier(used_qubits)
                if delay_per_round > 0:
                    for qubit in used_qubits:
                        circuit.delay(delay_per_round, qubit, unit=init_delay_unit)
                    circuit.barrier(used_qubits)
        elif init_delay:
            for qubit in used_qubits:
                circuit.delay(init_delay, qubit, unit=init_delay_unit)
            circuit.barrier(used_qubits)

        if init_delay:
            instruction_durations = InstructionDurations(dt=self._get_dt())
            pm_ = PassManager(
                [TimeUnitConversion(instruction_durations), ConvertNearestMod16Delay()]
            )
            circuit = pm_.run(circuit)

        return circuit

    def _init_strategy_before(
        self,
        used_qubits: Iterable[int],
        init_num_resets: int,
        init_delay: float,
        init_delay_unit: str,
    ) -> QuantumCircuit:
        """Add full delay before reset rounds."""
        n_qubits = self.backend.configuration().n_qubits
        circuit = QuantumCircuit(n_qubits)

        # Only reset qubits that are used in the experiment to reduce
        # initialization time and replicate current backend behaviour
        circuit.barrier(used_qubits)
        if init_delay:
            for qubit in used_qubits:
                circuit.delay(init_delay, qubit, unit=init_delay_unit)
            circuit.barrier(used_qubits)
        for _ in range(0, init_num_resets):
            circuit.reset(used_qubits)
            circuit.barrier(used_qubits)

        if init_delay:
            instruction_durations = InstructionDurations(dt=self._get_dt())
            pm_ = PassManager(
                [TimeUnitConversion(instruction_durations), ConvertNearestMod16Delay()]
            )
            circuit = pm_.run(circuit)

        return circuit

    def _used_qubits(self, circuits: List[QuantumCircuit]) -> Union[Set[int], range]:
        """Find all qubits used across the circuits to be merged."""
        qubits: Set[int] = set()
        for circuit in circuits:
            if len(circuit.qregs) > 1:
                # circuit is not working on physical qubits, fallback to resetting all qubits
                return range(self.backend.configuration().n_qubits)
            for data in circuit.data:
                if isinstance(data.operation, Delay):
                    continue
                if isinstance(data.operation, Barrier):
                    continue
                qubits.update(circuit.find_bit(qubit).index for qubit in data.qubits)
        return qubits

    def _compose_circuits(
        self, merged_circuit: QuantumCircuit, init_circuit: QuantumCircuit, init: bool
    ) -> QuantumCircuit:
        """Compose merged circuit."""
        bit_offset = 0
        for circuit in self.circuits:
            if init:
                merged_circuit.compose(init_circuit, inplace=True)

            # assign the circuits classical bits in sequence, tracking the latest offset
            merged_circuit.compose(
                circuit,
                clbits=merged_circuit.clbits[bit_offset : (bit_offset + circuit.num_clbits)],
                inplace=True,
            )
            bit_offset += circuit.num_clbits
        return merged_circuit

    def merge_circuits(
        self,
        init: bool = True,
        init_num_resets: int = 3,
        init_delay: int = 0,
        init_delay_unit: str = "s",
        init_circuit: Optional[QuantumCircuit] = None,
    ) -> QuantumCircuit:
        """Merge circuits into one and return the result.

        Merge all the circuits in this instance into a single circuit. Between
        the circuits, initialize qubits with init_num_resets times a qubit reset
        operation and a delay of duration init_delay (unit in parameter
        init_delay_unit). If a custom circuit init_circuit is provided, use
        that as an alternative.

        Args:
            init: Enable initialization of qubits
            init_num_resets: Number of qubit initializations (resets) to
                perform in a row.
            init_delay: Delay to insert between circuits.
            init_delay_unit: Unit of the delay.
            init_circuit: Custom circuit for initializing qubits.

        Returns: All circuits in this instance merged into one and separated
        by qubit initialization circuits.
        """

        # Collect all classical registers and mangle their names (to handle potential duplicates)
        def mangle_register_name(idx, register):
            return "qc" + str(idx) + "_" + register.name

        regs = []
        for idx, circuit in enumerate(self.circuits):
            # Qiskit now supports allocating the same clbit to multiple registers
            # this requires careful handling now. A clbit may also *not*
            # belong to a register. This is complicated as the result format
            # requires registers which we must therefore generate.
            for clbit in circuit.clbits:
                # Check if clbit belongs to a register
                bit_locations = circuit.find_bit(clbit)
                if not bit_locations.registers:
                    # if we do not belong to a register generate the register for the clbit
                    # and add it to the circuit
                    generated_creg = ClassicalRegister(bits=[clbit])
                    circuit.add_register(generated_creg)

            # now that we have generated registers for clbits without them, we continue.
            for creg in circuit.cregs:
                regs.append(ClassicalRegister(creg.size, mangle_register_name(idx, creg)))

        regs.insert(0, QuantumRegister(self.backend.configuration().n_qubits))

        # create empty circuit into which to merge all others;
        # use transpile for mapping to physical qubits
        merged_circuit = QuantumCircuit(*regs)
        # Set the layout to physical qubits.
        # This avoids having to call transpile.
        # This sets a private method, but this is how transpiler passes also set this attribute.
        merged_circuit._layout = Layout(
            {qubit: idx for idx, qubit in enumerate(merged_circuit.qubits)}
        )

        used_qubits = self._used_qubits(self.circuits)

        if not init_circuit and init:
            init_circuit = self._create_init_circuit(
                used_qubits, init_num_resets, init_delay, init_delay_unit
            )

        return self._compose_circuits(merged_circuit, init_circuit, init)

    def unwrap_results(self, result: Result):
        """Unwrap results from executing a merged circuit.

        Postprocess the result of executing the merged circuit and separate
        the result data per circuit. Create a corresponding result object
        that allows retrieving counts and memory individually, such as if the
        circuits in this instance had been executed separately.

        Args:
            result: Result of the execution of the merged circuit.

        Returns: Result object that behaves as if the circuits in this
            instance had been executed separately.
        """
        return QiskitQobjResultUnwrapper().prepare_execution_result(self.circuits, result)


class QiskitQobjResultUnwrapper:
    """Unwraps a result from the backend and populates its metadata."""

    def prepare_execution_result(self, circuits: List[QuantumCircuit], result: Result):
        """Unwrap results from executing a merged circuit.

        Postprocess the result of executing the merged circuit and separate
        the result data per circuit. Create a corresponding result object
        that allows retrieving counts and memory individually, such as if the
        circuits in this instance had been executed separately.

        Args:
            circuits: Quantum circuits which were executed.
            result: Result of the execution of the merged circuit.

        Returns: Result object that behaves as if the circuits in this
            instance had been executed separately.
        """
        combined_res = result.results[0].to_dict()
        meas_level = combined_res.get("meas_level")
        meas_levels = set(combined_res.get("meas_levels", []))
        meas_levels.add(meas_level)
        meas_return = combined_res.get("meas_return", MeasurementReturnType.AVG.value)

        unwrapped_results = []
        bit_offset = 0
        for circuit in circuits:
            circuit_result = self.prepare_circuit_result(
                combined_res, circuit, bit_offset, meas_levels, meas_return
            )
            bit_offset += len(circuit.clbits)
            unwrapped_results.append(circuit_result)

        res_dict = result.to_dict()
        res_dict["results"] = unwrapped_results
        return Result.from_dict(res_dict)

    def prepare_circuit_result(
        self,
        orig_result: Dict[str, Any],
        circuit: QuantumCircuit,
        bit_offset: int,
        meas_levels: List[MeasurementReportingLevel],
        meas_return: MeasurementReturnType,
    ) -> Dict[str, Any]:
        """Prepare the Qiskit result from the Qobj result that was returned
        which contains no circuit metadata."""
        res = orig_result.copy()
        combined_data = orig_result["data"]
        assert "data" in res

        res["header"] = res["header"].copy()
        res["data"] = res["data"].copy()

        header = res["header"]
        header["name"] = res["name"] = circuit.name

        header["metadata"] = circuit.metadata

        # see qiskit/assembler/assemble_circuits.py for how Qiskit builds
        # the information about classical bits and registers in a
        # result's header.
        header["creg_sizes"] = [[creg.name, creg.size] for creg in circuit.cregs]
        num_clbits = len(circuit.clbits)
        header["memory_slots"] = num_clbits

        header["clbit_labels"] = [
            [creg.name, i] for creg in circuit.cregs for i in range(creg.size)
        ]

        if "counts" in combined_data:
            res["data"]["counts"] = self.extract_counts(
                combined_data["counts"], bit_offset, num_clbits
            )

        if MeasurementReportingLevel.KERNELED.value in meas_levels:
            # Handle measurement level 1 memory as KERNELED data.
            if meas_return == MeasurementReturnType.AVG.value:
                res["data"]["memory"] = self.extract_kernels_avg(
                    combined_data["memory"], bit_offset, num_clbits
                )
            elif meas_return == "single":
                res["data"]["memory"] = self.extract_kernels_single(
                    combined_data["memory"], bit_offset, num_clbits
                )
            else:
                raise ValueError(f"Measurement return type {meas_return} is not supported.")

        elif "memory" in combined_data:
            # Handle measurement level 2 memory as counts.
            extracted_memory = [
                self.extract_bits(bitstring, bit_offset, num_clbits)
                for bitstring in combined_data["memory"]
            ]
            res["data"]["memory"] = extracted_memory

        return res

    def extract_bits(self, bitstring: str, bit_position: int, num_bits: int) -> str:
        """Extract selected bits from the bitstring"""
        assert bitstring.startswith("0x") or bitstring.startswith("0b")
        bitstring_as_int = int(bitstring, 0)
        bitstring_as_int >>= bit_position
        mask = (1 << num_bits) - 1
        return hex(bitstring_as_int & mask)

    def extract_kernels_avg(
        self, memory: List[Tuple[float, float]], bit_position: int, num_bits: int
    ) -> List[Tuple[float, float]]:
        """Extract selected bits from the average kernel memory"""
        return memory[bit_position : bit_position + num_bits]

    def extract_kernels_single(
        self, memory: List[Tuple[float, float]], bit_position: int, num_bits: int
    ) -> List[Tuple[float, float]]:
        """Extract selected bits from the single kernel memory"""
        extracted_results = []
        for shot_result in memory:
            extracted_results.append(self.extract_kernels_avg(shot_result, bit_position, num_bits))
        return extracted_results

    def extract_counts(
        self, combined_counts: Dict[str, int], bit_offset: int, num_clbits: int
    ) -> Dict[str, int]:
        """Extract selected bits from the binned counts"""
        extracted_counts: Dict[str, int] = {}
        for bitstring, count in combined_counts.items():
            extracted_hex = self.extract_bits(bitstring, bit_offset, num_clbits)
            if extracted_hex in extracted_counts:
                extracted_counts[extracted_hex] += count
            else:
                extracted_counts[extracted_hex] = count
        return extracted_counts


class Qasm3Encoder(RuntimeEncoder):
    """QASM3 Encoder"""

    def default(self, obj):  # pylint: disable=arguments-differ
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class MeasurementReportingLevel(Enum):
    """Measurement result reporting level."""

    KERNELED = 1
    CLASSIFIED = 2


class MeasurementReturnType(Enum):
    """Measurement return type."""

    AVERAGE = "average"
    AVG = "avg"
    # Same as average
    SINGLE = "single"


@dataclass
class QASM3Options:
    """Options for the qasm3-runner."""

    circuits: Union[QuantumCircuit, List[QuantumCircuit]] = None
    merge_circuits: bool = True
    shots: int = 4000
    # Number of repetitions of each circuit, for sampling.
    meas_level: MeasurementReportingLevel = MeasurementReportingLevel.CLASSIFIED
    # meas_level: The reporting level for measurements results:
    #    Level 2: Discriminated measurement counts
    #   Level 1: KERNELED measurement kernel values.
    meas_return: MeasurementReturnType = MeasurementReturnType.AVERAGE
    init_circuit: Optional[QuantumCircuit] = None
    rep_delay: float = 100.0e-6
    # The number of seconds of delay to insert before each circuit execution.
    # These will be interspersed with resets.
    # See https://github.ibm.com/IBM-Q-Software/ws-dynamic-circuits/blob/c6f1f4995c3311b5cf3cd64d48c7f8f19f02aaf8/docs/experiments/qubit_initialization_strategies.ipynb # pylint: disable=line-too-long
    # for how this default value was chosen.
    init_num_resets: int = 3
    # The number of qubit resets to insert before each circuit execution.
    memory: bool = True
    # Whether to request the individual shot outcomes rather than just the binned counts (if `False`)
    # Currently this option is always set to `True` even if set to `False`.

    @classmethod
    def build_from_runtime(cls, backend, **kwargs) -> "QASM3Options":
        """Built the options class from the default runtime input
        overriding the fields that are set to ``None`` with their
        defaults.
        """
        non_none = (
            "shots",
            "meas_level",
            "meas_return",
            "init_delay",
            "init_num_resets",
            "run_config",
            "skip_transpilation",
            "use_measurement_mitigation",
        )

        for key in non_none:
            if key in kwargs and kwargs[key] is None:
                del kwargs[key]

        if kwargs.get("run_config", None):
            raise RuntimeError(
                "Setting the qasm3-runner `run_config` has completed its deprecation period "
                "and is no longer available for usage. If you are using `rep_delay` or "
                "`shots` as a `run_config` setting, these are now top-level program "
                " arguments. If you did not set this argument "
                "please update your provider to the latest `qiskit_ibm_provider`."
            )

        if "init_delay" in kwargs:
            raise RuntimeError(
                "The `init_delay` argument is no longer available. "
                "Please use `rep_delay` instead. Units are in seconds."
            )

        if not kwargs.get("skip_transpilation", True):
            raise RuntimeError(
                "Transpilation within qasm3-runner has completed its deprecation period "
                "and is no longer available for usage. If you did not set this argument "
                "please update your provider to the latest `qiskit_ibm_provider`."
            )

        if kwargs.get("use_measurement_mitigation", False):
            raise RuntimeError(
                "Enabling measurement error mitigation through `use_measurement_mitigation` "
                "has completed its deprecation period "
                "and is no longer available for usage. If you did not set this argument "
                "please update your provider to the latest `qiskit_ibm_provider`."
            )

        if kwargs.get("exporter_config", False):
            raise RuntimeError(
                "Setting the qasm3 exporter config has completed its deprecation period "
                "and is no longer available for usage. If you did not set this argument "
                "please update your provider to the latest `qiskit_ibm_provider`."
            )

        if kwargs.get("memory", False):
            logger.warning(
                "Currently the qasm3-runner does not support `memory=False`. "
                "Bitstrings will be reported for each shot."
            )

        # Use the default rep_delay from the backend
        try:
            # Simulators do not have this available so only set if is present
            # if not fall back to the default
            kwargs.setdefault("rep_delay", backend.configuration().default_rep_delay)
        except (KeyError, AttributeError):
            pass

        # Configure reset settings for the "init_qubits" argument.
        # To disable qubit initialization.
        if not kwargs.pop("init_qubits", True):
            kwargs["rep_delay"] = 0.0
            kwargs["init_num_resets"] = 0.0
            kwargs["init_circuit"] = None
            kwargs["merge_circuits"] = False

        QASM3Options.are_valid_options(**kwargs)
        return QASM3Options(**kwargs)

    @classmethod
    def are_valid_options(cls, **kwargs) -> None:
        """Check if supplied options are valid."""
        arg_names = set(kwargs.keys())
        valid_args = set(cls.__annotations__.keys())
        if not_in := arg_names - valid_args:
            raise RuntimeError(
                f'The following arguments are not valid for the "qasm3-runner" runtime program: {str(list(not_in))}'
            )

    def prepare_run_config(self):
        """Prepare an externally safe run configuration."""
        extra_compile_args = []

        # Counts is the default so don't set unless overridden
        # as older compiler versions do not support.
        if MeasurementReportingLevel(self.meas_level) != MeasurementReportingLevel.CLASSIFIED:
            extra_compile_args.append(f"--rta-measure-report-level={int(self.meas_level)}")

        # Average is the default so don't set unless overridden
        # as older compiler versions do not support.
        measurement_return_type = MeasurementReturnType(self.meas_return)
        if measurement_return_type == MeasurementReturnType.AVG:
            measurement_return_type = MeasurementReturnType.AVERAGE

        if measurement_return_type != MeasurementReturnType.AVERAGE:
            extra_compile_args.append(
                f"--rta-measure-report-type={str(measurement_return_type.value)}"
            )

        filtered_run_config = {
            "extra_compile_args": extra_compile_args,
            "shots": self.shots,
            "rep_delay": QSS_COMPILER_REP_DELAY,
        }

        return filtered_run_config


def main(
    backend,
    user_messenger,  # pylint: disable=unused-argument
    circuits,
    **kwargs,
):
    """Execute

    Args:
        backend: Backend to execute circuits on.
        user_messenger (UserMessenger): Used to communicate with the program user.
        circuits: Circuits to execute.

    Returns:
        Program result.
    """
    if circuits and not isinstance(circuits, list):
        circuits = [circuits]

    if not circuits or (
        not all(isinstance(circ, QuantumCircuit) for circ in circuits)
        and not all(isinstance(circ, str) for circ in circuits)
    ):
        raise RuntimeError(
            "Circuit need to be of type QuantumCircuit or str and \
            circuit types need to be consistent in a list of circuits."
        )

    # TODO Better validation once we can query for input_allowed
    _backend_name = backend.name if backend.version == 2 else backend.name()
    if backend.configuration().simulator and _backend_name not in SIMULATORS:
        raise RuntimeError(
            f"The selected backend ({_backend_name}) does not support dynamic circuit capabilities"
        )

    is_qc = isinstance(circuits[0], QuantumCircuit)

    options = QASM3Options.build_from_runtime(backend, **kwargs)

    use_merging = False

    # Submit circuits for testing of standard circuit merger
    qasm2_sim = _backend_name == QASM2_SIM_NAME
    if is_qc:
        if options.merge_circuits:
            if options.init_circuit is not None and not isinstance(
                options.init_circuit, QuantumCircuit
            ):
                raise RuntimeError(
                    "init_circuit must be of type QuantumCircuit but is "
                    f"{options.init_circuit.__class__}."
                )
            merger = CircuitMerger(
                circuits,
                backend=backend,
            )
            use_merging = True
            circuits = [
                merger.merge_circuits(
                    init_num_resets=options.init_num_resets,
                    init_delay=options.rep_delay,
                    init_delay_unit="s",
                    init_circuit=options.init_circuit,
                )
            ]
        else:
            # If not circuit merging we must still modify the circuit
            # to workaround https://github.com/Qiskit/qiskit-terra/issues/10112
            # which is throwing out physical qubit information.
            for circuit in circuits:
                # Set the layout to physical qubits.
                # This avoids having to call transpile.
                # This sets a private method, but this is how transpiler passes also set this attribute.
                circuit._layout = Layout({qubit: idx for idx, qubit in enumerate(circuit.qubits)})

        # Convert circuits to qasm3
        qasm3_strs = []
        exporter_config = {
            "includes": (),
            "disable_constants": True,
            "basis_gates": backend.configuration().basis_gates,
            "experimental": ExperimentalFeatures.SWITCH_CASE_V1,
        }
        for circ in circuits:
            qasm3_strs.append(Exporter(**exporter_config).dumps(circ))

        if not qasm2_sim:
            payload = qasm3_strs
        else:
            payload = circuits
    else:
        if _backend_name == QASM2_SIM_NAME:
            raise RuntimeError(
                "This simulator backend does not support OpenQASM 3 source strings as input. "
                "Please submit a quantum circuit instead."
            )
        payload = circuits

    # Prepare safe run_config
    filtered_run_config = options.prepare_run_config()

    result = backend.run(payload, **filtered_run_config).result()

    if use_merging:
        result = merger.unwrap_results(result)
    elif is_qc:
        # If a quantum circuit was submitted without merging we may still inject metadata
        result = QiskitQobjResultUnwrapper().prepare_execution_result(circuits, result)

    return result.to_dict()


def final_measurement_mapping(circuit):
    """Returns the final measurement mapping for a circuit that
    has been transpiled (flattened registers) or has flat registers.

    Parameters:
        circuit (QuantumCircuit): Input quantum circuit.

    Returns:
        dict: Mapping of qubits to classical bits for final measurements.

    Raises:
        ValueError: More than one quantum or classical register.
    """
    if len(circuit.qregs) > 1 or len(circuit.qregs) > 1:
        raise ValueError("Number of quantum or classical registers is greater than one.")
    active_qubits = set(circuit.qubits)
    active_cbits = set(circuit.clbits)
    qmap = []
    cmap = []
    for instruction in circuit.data[::-1]:
        if instruction.operation.name == "measure":
            cbit = instruction.clbits[0]
            qbit = instruction.qubits[0]
            if cbit in active_cbits and qbit in active_qubits:
                qmap.append(qbit)
                cmap.append(cbit)
                active_cbits.remove(cbit)
                active_qubits.remove(qbit)
        elif instruction.operation.name != "barrier":
            active_qubits -= instruction.qubits

        if len(active_cbits) == 0 or len(active_qubits) == 0:
            break
    if cmap and qmap:
        mapping = {}
        for idx, qubit in enumerate(qmap):
            mapping[circuit.find_bit(qubit).index] = cmap[idx]
    else:
        raise ValueError("Measurement not found in circuits.")

    # Sort so that classical bits are in numeric order low->high.
    mapping = dict(sorted(mapping.items(), key=lambda item: item[1]))
    return mapping


def quasi_to_hex(quasi_dict):
    """Converts a quasi-prob dict with bitstrings to hex

    Parameters:
        quasi_dict (QuasiDistribution): Input quasi dict

    Returns:
        dict: hex dict.
    """
    hex_quasi = {}
    for key, val in quasi_dict.items():
        hex_quasi[hex(int(key, 2))] = val
    return hex_quasi
