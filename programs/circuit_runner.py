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
A runtime program that takes one or more circuits, compiles them, executes them,
and optionally applies measurement error mitigation.
"""

import json
import sys
from time import perf_counter
from typing import Dict

from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile, schedule
from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.result import marginal_counts
from qiskit_ibm_runtime.utils import RuntimeDecoder
from qiskit_ibm_runtime.program import UserMessenger
import mthree


def main(
    backend,
    user_messenger,  # pylint: disable=unused-argument
    circuits,
    initial_layout=None,
    seed_transpiler=None,
    optimization_level=None,
    transpiler_options=None,
    scheduling_method=None,
    schedule_circuit=False,
    inst_map=None,
    meas_map=None,
    measurement_error_mitigation=False,
    **kwargs,
):
    """Circuit runner program"""

    # Support Qobj
    if isinstance(circuits, Dict) and "type" in circuits:
        if circuits["type"] == "PULSE":
            circuits = PulseQobj.from_dict(circuits)
        else:
            circuits = QasmQobj.from_dict(circuits)
        kwargs["skip_transpilation"] = True
    elif not isinstance(circuits, list):
        circuits = [circuits]

    if isinstance(circuits, list):
        for i, circuit in enumerate(circuits):
            # Support QASM 2.0 strings
            if isinstance(circuit, str) and "OPENQASM 2.0" in circuit:
                circuits[i] = QuantumCircuit.from_qasm_str(circuit)

    noise_model = kwargs.pop("noise_model", None)
    seed_simulator = kwargs.pop("seed_simulator", None)
    if backend.configuration().simulator:
        backend.set_options(noise_model=noise_model, seed_simulator=seed_simulator)

    # transpiling the circuits using given transpile options (deprecated).
    skip_transpilation = kwargs.pop("skip_transpilation", False)
    if not skip_transpilation:
        transpiler_options = transpiler_options or {}
        circuits = transpile(
            circuits,
            initial_layout=initial_layout,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
            backend=backend,
            **transpiler_options,
        )
        _provider = backend.provider if backend.version == 2 else backend.provider()
        job_object_storage = _provider.job_object_storage()
        if job_object_storage:
            job_object_storage.save_transpiled_circuits(
                transpiled_circuits={
                    "transpiled_circuits": {
                        "qpy": circuits,
                        "qasm2": [circuit.qasm() for circuit in circuits],
                    }
                },
            )

    if schedule_circuit:
        circuits = schedule(
            circuits=circuits,
            backend=backend,
            inst_map=inst_map,
            meas_map=meas_map,
            method=scheduling_method,
        )

    if measurement_error_mitigation:
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

    # Compute raw results
    result = backend.run(circuits, **kwargs).result()

    # Do the actual mitigation here
    if measurement_error_mitigation:
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


if __name__ == "__main__":
    # Test using Aer
    simulator = Aer.get_backend("qasm_simulator")
    user_params = {}
    if len(sys.argv) > 1:
        # If there are user parameters.
        user_params = json.loads(sys.argv[1], cls=RuntimeDecoder)
    main(simulator, UserMessenger(), **user_params)
