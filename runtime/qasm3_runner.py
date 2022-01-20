# Circuit runner QASM3 runtime program

from time import perf_counter

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.qasm3 import Exporter
from qiskit.result import marginal_counts
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder
import mthree
import numpy as np

QASM3_SIM_NAME = "simulator_qasm3"


class Qasm3Encoder(RuntimeEncoder):

    def default(self, obj):  # pylint: disable=arguments-differ
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def main(backend, user_messenger,
         circuits,
         transpiler_config=None,
         exporter_config=None,
         run_config=None,
         qasm3_args=None,
         skip_transpilation=False,
         use_measurement_mitigation=False
         ):
    """Execute

    Args:
        backend: Backend to execute circuits on.
        user_messenger: Used to communicate with the users.
        circuits: Circuits to execute.
        transpiler_config: Transpiler configurations.
        exporter_config: QASM3 exporter configurations.
        run_config: Execution time configurations.
        qasm3_args: Arguments to pass to the QASM3 program loop.
        skip_transpilation: Skip transpiling of circuits.
        use_measurement_mitigation: Whether to perform measurement error mitigation.

    Returns:
        Program result.
    """

    if circuits and not isinstance(circuits, list):
        circuits = [circuits]

    if not circuits or \
        (not all(isinstance(circ, QuantumCircuit) for circ in circuits) and
            not all(isinstance(circ, str) for circ in circuits)):
        raise ValueError('Circuit need to be of type QuantumCircuit or str and \
            circuit types need to be consistent in a list of circuits.')

    # TODO Better validation once we can query for input_allowed
    if backend.configuration().simulator and backend.name() != QASM3_SIM_NAME:
        raise ValueError("This backend does not support QASM3")

    is_qc = isinstance(circuits[0], QuantumCircuit)

    if use_measurement_mitigation and (not is_qc or backend.name() != QASM3_SIM_NAME):
        raise NotImplementedError("Measurement error mitigation is only supported for "
                                  "QuantumCircuit inputs and non-simulator backends.")

    if is_qc:
        raise RuntimeError(
            "You are not authorized to use this program with QuantumCircuit inputs.")

    run_config = run_config or {}

    if is_qc:
        if not skip_transpilation:
            # Transpile the circuits using given transpile options
            transpiler_config = transpiler_config or {}
            circuits = transpile(circuits, backend=backend, **transpiler_config)

        # Convert circuits to qasm3
        qasm3_strs = []
        qasm3_metadata = []
        exporter_config = exporter_config or {}
        for circ in circuits:
            qasm3_strs.append(Exporter(**exporter_config).dumps(circ))
            qasm3_metadata.append(get_circuit_metadata(circ))

        payload = qasm3_strs
        run_config["qasm3_metadata"] = qasm3_metadata

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
        payload = circuits

    if backend.name() == QASM3_SIM_NAME:
        if len(payload) > 1:
            raise ValueError("QASM3 simulator only supports a single circuit.")
        result = backend.run(payload[0],
                             args=qasm3_args,
                             shots=run_config.get("shots", None))
        user_messenger.publish(result, final=True, encoder=Qasm3Encoder)
        return

    result = backend.run(payload, **run_config).result()

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
                raw_counts = marginal_counts(raw_counts,
                                             list(mappings[idx].values()))
            _qubits = list(mappings[idx].keys())
            start_time = perf_counter()
            quasi = mit.apply_correction(raw_counts, _qubits)
            stop_time = perf_counter()
            mit_times.append(stop_time-start_time)
            # Convert quasi dist with bitstrings to hex version and append
            quasi_probs.append(quasi_to_hex(quasi))

        # Attach to results.
        for idx, res in enumerate(result.results):
            res.data.quasiprobabilities = quasi_probs[idx]
            res.data._data_attributes.append('quasiprobabilities')
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
        "metadata": circuit.metadata or {}
    }


def final_measurement_mapping(qc):
    """Returns the final measurement mapping for a circuit that
    has been transpiled (flattened registers) or has flat registers.
    
    Parameters:
        qc (QuantumCircuit): Input quantum circuit.
    
    Returns:
        dict: Mapping of qubits to classical bits for final measurements.

    Raises:
        ValueError: More than one quantum or classical register.
    """
    if len(qc.qregs) > 1 or len(qc.qregs) > 1:
        raise ValueError('Number of quantum or classical registers is greater than one.')
    num_qubits = qc.num_qubits
    num_clbits = qc.num_clbits
    active_qubits = list(range(num_qubits))
    active_cbits = list(range(num_clbits))
    qmap = []
    cmap = []
    for item in qc._data[::-1]:
        if item[0].name == 'measure':
            cbit = item[2][0].index
            qbit = item[1][0].index
            if cbit in active_cbits and qbit in active_qubits:
                qmap.append(qbit)
                cmap.append(cbit)
                active_cbits.remove(cbit)
                active_qubits.remove(qbit)
        elif item[0].name != 'barrier':
            for qq in item[1]:
                if qq.index in active_qubits:
                    active_qubits.remove(qq.index)

        if not len(active_cbits) or not len(active_qubits):
            break
    if cmap and qmap:
        mapping = {}
        for idx, qubit in enumerate(qmap):
            mapping[qubit] = cmap[idx]
    else:
        raise ValueError('Measurement not found in circuits.')

    # Sort so that classical bits are in numeric order low->high.
    mapping = dict(sorted(mapping.items(), key=lambda item: item[1])) 
    return mapping


def quasi_to_hex(qp):
    """Converts a quasi-prob dict with bitstrings to hex

    Parameters:
        qp (QuasiDistribution): Input quasi dict

    Returns:
        dict: hex dict.
    """
    hex_quasi = {}
    for key, val in qp.items():
        hex_quasi[hex(int(key, 2))] = val     
    return hex_quasi  
