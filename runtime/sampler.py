import mthree
from qiskit import transpile


def main(backend, user_messenger,
         circuits,
         transpiler_config={'optimization_level': 3, 
                           'layout_method': 'sabre',
                           'routing_method': 'sabre'},
         run_config={'shots': 8192},
         skip_transpilation=False,
         use_measurement_mitigation=False,
         return_mitigation_overhead=False
        ):
    
    """Sample distributions generated by given circuits executed on the target backend.
    
    Parameters:
        backend (ProgramBackend): Qiskit backend instance.
        user_messenger (UserMessenger): Used to communicate with the program user.
        circuits: (QuantumCircuit or list): A single list of QuantumCircuits.
        transpiler_config (dict): A collection of kwargs passed to transpile().
        run_config (dict): A collection of kwargs passed to backend.run().
        skip_transpilation (bool): Skip transpiling of circuits, default=False.
        use_measurement_mitigation (bool): Improve resulting using measurement
                                           error mitigation, default=False.
        return_mitigation_overhead (bool): Return mitigation overhead factor,
                                           default=False.
                                           
    Notes:
        Default values for transpiler_config selected as they give the best results
        overall.
                                           
    Returns:
        dict: Dictionary with keys 'raw_counts', 'quasiprobabilities', and
              'mitigation_overhead', containing lists of counts data, quasiprobabilites,
              and mitigation overheads.  Only one of either `raw_counts` or 
              `quasiprobabilities` is returned at a time.
    """
    # transpiling the circuits using given transpile options
    if not skip_transpilation:
        trans_circuits = transpile(circuits, backend=backend,
                                   **transpiler_config)
        # Make sure everything is a list
        if not isinstance(trans_circuits, list):
            trans_circuits = [trans_circuits]
    # If skipping set circuits -> trans_circuits
    else:
        if not isinstance(circuits, list):
            trans_circuits = [circuits]
        else:
            trans_circuits = circuits

    # If doing measurement mitigation we must build and calibrate a
    # mitigator object.  Will also determine which qubits need to be
    # calibrated.
    quasi_dists = []
    if use_measurement_mitigation:
        # Get an the measurement mappings at end of circuits
        meas_maps = mthree.utils.final_measurement_mapping(trans_circuits)
        # Get an M3 mitigator
        mit = mthree.M3Mitigation(backend)
        # Calibrate over the set of qubits measured in the transpiled circuits.
        mit.cals_from_system(meas_maps)

    # Compute raw results
    result = backend.run(trans_circuits, **run_config).result()
    raw_counts = result.get_counts()
    if not isinstance(raw_counts, list):
        raw_counts = [raw_counts]

    # When using measurement mitigation we need to apply the correction and then
    # compute the expectation values from the computed quasi-probabilities.
    mitigation_overhead = []
    if use_measurement_mitigation:
        quasi_dists = mit.apply_correction(raw_counts, meas_maps,
                                           return_mitigation_overhead=return_mitigation_overhead)
        if return_mitigation_overhead:
            if not isinstance(quasi_dists, mthree.classes.QuasiCollection):
                mitigation_overhead = [quasi_dists.mitigation_overhead]
            else:
                mitigation_overhead = list(quasi_dists.mitigation_overhead)
                
        if not isinstance(quasi_dists, mthree.classes.QuasiCollection):
            quasi_dists = [quasi_dists]

    out = ([] if use_measurement_mitigation else [rc.hex_outcomes() for rc in raw_counts],
           [quasi_to_hex(qp) for qp in quasi_dists],
           mitigation_overhead)
    
    return out

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
