# Circuit runner runtime program

import json
import sys

from qiskit import Aer
from qiskit.compiler import transpile, schedule
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder, RuntimeDecoder
from qiskit.providers.ibmq.runtime import UserMessenger
import mthree


def main(backend, user_messenger, circuits,
         initial_layout=None, seed_transpiler=None, optimization_level=None,
         transpiler_options=None, scheduling_method=None,
         schedule_circuit=False, inst_map=None, meas_map=None,
         measurement_error_mitigation=False,
         **kwargs):

    # transpiling the circuits using given transpile options
    transpiler_options = transpiler_options or {}
    circuits = transpile(circuits,
                         initial_layout=initial_layout,
                         seed_transpiler=seed_transpiler,
                         optimization_level=optimization_level,
                         backend=backend, **transpiler_options)

    if schedule_circuit:
        circuits = schedule(circuits=circuits,
                            backend=backend,
                            inst_map=inst_map,
                            meas_map=meas_map,
                            method=scheduling_method)

    
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

    
    
    result = backend.run(circuits, **kwargs).result()
    print(json.dumps(result, cls=RuntimeEncoder))


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

    # Sort so that classical bits are in numeric order low->high.
    mapping = dict(sorted(mapping.items(), key=lambda item: item[1])) 
    return mapping


if __name__ == '__main__':
    # Test using Aer
    backend = Aer.get_backend('qasm_simulator')
    user_params = {}
    if len(sys.argv) > 1:
        # If there are user parameters.
        user_params = json.loads(sys.argv[1], cls=RuntimeDecoder)
    main(backend, UserMessenger(), **user_params)
