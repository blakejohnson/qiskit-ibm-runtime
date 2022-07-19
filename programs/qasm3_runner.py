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

from time import perf_counter
from typing import Dict, Iterable, List, Optional, Set, Union

from qiskit.circuit.quantumcircuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.qasm3 import Exporter
from qiskit.result import marginal_counts, Result
from qiskit_ibm_runtime.utils import RuntimeEncoder
import mthree
import numpy as np


QASM3_SIM_NAME = "simulator_qasm3"
QASM2_SIM_NAME = "qasm_simulator"
SIMULATORS = (QASM3_SIM_NAME, QASM2_SIM_NAME)


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

    def _create_init_circuit(
        self,
        used_qubits: Iterable[int],
        init_num_resets: int,
        init_delay: int,
        init_delay_unit: str,
    ) -> QuantumCircuit:
        """Create a parameterized initialization circuit or return the
        user-provided initialization circuit.
        """
        # reset circuit must span all qubits
        n_qubits = self.backend.configuration().n_qubits
        circuit = QuantumCircuit(n_qubits)

        # Only reset qubits that are used in the experiment to reduce
        # initialization time and replicate current backend behaviour
        circuit.barrier(used_qubits)
        for _ in range(0, init_num_resets):
            circuit.reset(used_qubits)
            circuit.barrier(used_qubits)
        if init_delay:
            circuit.delay(init_delay, range(n_qubits), unit=init_delay_unit)
        circuit.barrier(used_qubits)

        if init_delay:
            # need to schedule circuit
            return transpile(
                circuit,
                backend=self.backend,
                scheduling_method="as_late_as_possible",
                optimization_level=0,
            )
        else:
            return circuit

    def _used_qubits(self, circuits: List[QuantumCircuit]) -> Union[Set[int], range]:
        """Find all qubits used across the circuits to be merged."""
        qubits: Set[int] = set()
        for circuit in circuits:
            if len(circuit.qregs) > 1:
                # circuit is not working on physical qubits, fallback to resetting all qubits
                return range(self.backend.configuration().n_qubits)
            for data in circuit.data:
                qubits.update(qubit.index for qubit in data[1])
        return qubits

    def _compose_circuits(
        self, merged_circuit: QuantumCircuit, init_circuit: QuantumCircuit, init: bool
    ):
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
        init_delay_unit: str = "us",
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

        regs = [
            ClassicalRegister(creg.size, mangle_register_name(idx, creg))
            for idx, circuit in enumerate(self.circuits)
            for creg in circuit.cregs
        ]
        regs.insert(0, QuantumRegister(self.backend.configuration().n_qubits))

        # create empty circuit into which to merge all others;
        # use transpile for mapping to physical qubits
        merged_circuit = transpile(
            QuantumCircuit(*regs), backend=self.backend, optimization_level=0
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
        combined_res = result.results[0].to_dict()
        combined_data = combined_res["data"]
        unwrapped_results = []
        bit_offset = 0

        def extract_bits(bitstring: str, bit_position: int, num_bits: int) -> str:
            assert bitstring.startswith("0x") or bitstring.startswith("0b")
            bitstring_as_int = int(bitstring, 0)
            bitstring_as_int >>= bit_position
            mask = (1 << num_bits) - 1
            return hex(bitstring_as_int & mask)

        def extract_counts(
            combined_counts: Dict[str, int], bit_offset: int, num_clbits: int
        ) -> Dict[str, int]:
            extracted_counts: Dict[str, int] = {}
            for bitstring, count in combined_counts.items():
                extracted_hex = extract_bits(bitstring, bit_offset, num_clbits)
                if extracted_hex in extracted_counts:
                    extracted_counts[extracted_hex] += count
                else:
                    extracted_counts[extracted_hex] = count
            return extracted_counts

        for circuit in self.circuits:
            res = combined_res.copy()
            assert "header" in res
            assert "data" in res

            res["header"] = res["header"].copy()
            res["data"] = res["data"].copy()

            header = res["header"]
            header["name"] = res["name"] = circuit.name

            # see qiskit/assembler/assemble_circuits.py for how Qiskit builds
            # the information about classical bits and registers in a
            # result's header.
            header["creg_sizes"] = [[creg.name, creg.size] for creg in circuit.cregs]
            num_clbits = sum([creg.size for creg in circuit.cregs])
            header["memory_slots"] = num_clbits

            header["clbit_labels"] = [
                [creg.name, i] for creg in circuit.cregs for i in range(creg.size)
            ]

            if "counts" in combined_data:
                res["data"]["counts"] = extract_counts(
                    combined_data["counts"], bit_offset, num_clbits
                )

            if "memory" in combined_data:
                extracted_memory = [
                    extract_bits(bitstring, bit_offset, num_clbits)
                    for bitstring in combined_data["memory"]
                ]
                res["data"]["memory"] = extracted_memory

            bit_offset += num_clbits
            unwrapped_results.append(res)

        res_dict = result.to_dict()
        res_dict["results"] = unwrapped_results
        return Result.from_dict(res_dict)


class Qasm3Encoder(RuntimeEncoder):
    """QASM3 Encoder"""

    def default(self, obj):  # pylint: disable=arguments-differ
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def main(
    backend,
    user_messenger,  # pylint: disable=unused-argument
    circuits,
    transpiler_config=None,
    exporter_config=None,
    run_config=None,
    qasm3_args=None,
    skip_transpilation=False,
    use_measurement_mitigation=False,
    merge_circuits=True,
    init_num_resets=3,
    init_delay=0,
    init_circuit=None,
):
    """Execute

    Args:
        backend: Backend to execute circuits on.
        user_messenger (UserMessenger): Used to communicate with the program user.
        circuits: Circuits to execute.
        transpiler_config: Transpiler configurations.
        exporter_config: QASM3 exporter configurations.
        run_config: Execution time configurations.
        qasm3_args: Arguments to pass to the QASM3 program loop.
        skip_transpilation: Skip transpiling of circuits.
        use_measurement_mitigation: Whether to perform measurement error
            mitigation.
        merge_circuits: whether to merge submitted QuantumCircuits into one
            before execution.
        init_num_resets: The number of qubit resets to insert before each
            circuit execution.
        init_delay: The number of microseconds of delay to insert before each
            circuit execution.
        init_circuit: Custom circuit for initializing qubits between merged
            circuits.

    Returns:
        Program result.
    """
    if circuits and not isinstance(circuits, list):
        circuits = [circuits]

    if not circuits or (
        not all(isinstance(circ, QuantumCircuit) for circ in circuits)
        and not all(isinstance(circ, str) for circ in circuits)
    ):
        raise ValueError(
            "Circuit need to be of type QuantumCircuit or str and \
            circuit types need to be consistent in a list of circuits."
        )

    # TODO Better validation once we can query for input_allowed
    if backend.configuration().simulator and backend.name() not in SIMULATORS:
        raise ValueError(
            f"The selected backend ({backend.name()}) does not support dynamic circuit capabilities"
        )

    is_qc = isinstance(circuits[0], QuantumCircuit)

    if use_measurement_mitigation and (not is_qc or backend.name() != QASM3_SIM_NAME):
        raise NotImplementedError(
            "Measurement error mitigation is only supported for "
            "QuantumCircuit inputs and non-simulator backends."
        )

    run_config = run_config or {}
    use_merging = False

    if is_qc:
        if merge_circuits and use_measurement_mitigation:
            raise NotImplementedError(
                "Measurement error mitigation cannot be used together with circuit merging."
            )

        if merge_circuits:
            if init_circuit is not None and not isinstance(init_circuit, QuantumCircuit):
                raise ValueError(
                    f"init_circuit must be of type QuantumCircuit but is {init_circuit.__class__}."
                )
            merger = CircuitMerger(
                circuits,
                backend=backend,
            )
            use_merging = True
            circuits = [
                merger.merge_circuits(
                    init_num_resets=init_num_resets,
                    init_delay=init_delay,
                    init_delay_unit="us",
                    init_circuit=init_circuit,
                )
            ]

        if not skip_transpilation:
            # Transpile the circuits using given transpile options
            transpiler_config = transpiler_config or {}
            circuits = transpile(circuits, backend=backend, **transpiler_config)

        # Convert circuits to qasm3
        qasm3_strs = []
        qasm3_metadata = []
        exporter_config = exporter_config or {}
        exporter_config["disable_constants"] = exporter_config.get("disable_constants", True)
        for circ in circuits:
            qasm3_strs.append(Exporter(**exporter_config).dumps(circ))
            qasm3_metadata.append(get_circuit_metadata(circ))

        # Submit circuits for testing of standard circuit merger
        if backend.name() != QASM2_SIM_NAME:
            payload = qasm3_strs
            run_config["qasm3_metadata"] = qasm3_metadata
        else:
            payload = circuits

        if use_measurement_mitigation:
            # get final meas mappings
            mappings = []
            all_meas_qubits = []
            for idx, circ in enumerate(circuits):
                mappings.append(final_measurement_mapping(circ))
                all_meas_qubits.extend(list(mappings[idx].keys()))

            # Collect set of active measured qubits over which to
            # mitigate.
            all_meas_qubits = list(set(all_meas_qubits))

            # Get an M3 mitigator and calibrate over all measured qubits
            mit = mthree.M3Mitigation(backend)
            mit.tensored_cals_from_system(all_meas_qubits)
    else:
        if backend.name() == QASM2_SIM_NAME:
            raise ValueError(
                "This simulator backend does not support OpenQASM 3 source strings as input. "
                "Please submit a quantum circuit instead."
            )
        payload = circuits

    if backend.name() == QASM3_SIM_NAME:
        if len(payload) > 1:
            raise ValueError("QASM3 simulator only supports a single circuit.")
        result = backend.run(payload[0], args=qasm3_args, shots=run_config.get("shots", None))
        return Qasm3Encoder().encode(result)

    result = backend.run(payload, **run_config).result()

    if use_merging:
        result = merger.unwrap_results(result)

    # Do the actual mitigation here
    if use_measurement_mitigation:
        quasi_probs = []
        mit_times = []
        for idx, circ in enumerate(circuits):
            num_cbits = circ.num_clbits
            num_measured_bits = len(mappings[idx])
            raw_counts = result.get_counts(idx)
            # check if more bits than measured so need to marginalize
            if num_cbits > num_measured_bits:
                raw_counts = marginal_counts(raw_counts, list(mappings[idx].values()))
            _qubits = list(mappings[idx].keys())
            start_time = perf_counter()
            quasi = mit.apply_correction(raw_counts, _qubits)
            stop_time = perf_counter()
            mit_times.append(stop_time - start_time)
            # Convert quasi dist with bitstrings to hex version and append
            quasi_probs.append(quasi_to_hex(quasi))

        # Attach to results.
        for idx, res in enumerate(result.results):
            res.data.quasiprobabilities = quasi_probs[idx]
            res.data._data_attributes.append("quasiprobabilities")
            res.header.final_measurement_mapping = mappings[idx]
            res.header.measurement_mitigation_time = mit_times[idx]

    return result.to_dict()


def get_circuit_metadata(circuit: QuantumCircuit):
    """Get the circuit metadata."""

    # header data
    num_qubits = 0
    memory_slots = 0
    qubit_labels = []
    clbit_labels = []

    qreg_sizes = []
    creg_sizes = []
    for qreg in circuit.qregs:
        qreg_sizes.append([qreg.name, qreg.size])
        for j in range(qreg.size):
            qubit_labels.append([qreg.name, j])
        num_qubits += qreg.size
    for creg in circuit.cregs:
        creg_sizes.append([creg.name, creg.size])
        for j in range(creg.size):
            clbit_labels.append([creg.name, j])
        memory_slots += creg.size

    return {
        "qubit_labels": qubit_labels,
        "n_qubits": num_qubits,
        "qreg_sizes": qreg_sizes,
        "clbit_labels": clbit_labels,
        "memory_slots": memory_slots,
        "creg_sizes": creg_sizes,
        "name": circuit.name,
        "global_phase": float(circuit.global_phase),
        "metadata": circuit.metadata or {},
    }


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
    num_qubits = circuit.num_qubits
    num_clbits = circuit.num_clbits
    active_qubits = list(range(num_qubits))
    active_cbits = list(range(num_clbits))
    qmap = []
    cmap = []
    for item in circuit._data[::-1]:
        if item[0].name == "measure":
            cbit = item[2][0].index
            qbit = item[1][0].index
            if cbit in active_cbits and qbit in active_qubits:
                qmap.append(qbit)
                cmap.append(cbit)
                active_cbits.remove(cbit)
                active_qubits.remove(qbit)
        elif item[0].name != "barrier":
            for q_q in item[1]:
                if q_q.index in active_qubits:
                    active_qubits.remove(q_q.index)

        if len(active_cbits) == 0 or len(active_qubits) == 0:
            break
    if cmap and qmap:
        mapping = {}
        for idx, qubit in enumerate(qmap):
            mapping[qubit] = cmap[idx]
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
