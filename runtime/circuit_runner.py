# Circuit runner runtime program

import json
import sys

from qiskit import Aer
from qiskit.compiler import transpile, schedule
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder, RuntimeDecoder
from qiskit.providers.ibmq.runtime import UserMessenger


def main(backend, user_messenger, circuits,
         initial_layout=None, seed_transpiler=None, optimization_level=None,
         transpiler_options=None, scheduling_method=None,
         schedule_circuit=False, inst_map=None, meas_map=None,
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

    result = backend.run(circuits, **kwargs).result()
    user_messenger.publish(result, final=True)


if __name__ == '__main__':
    # Test using Aer
    backend = Aer.get_backend('qasm_simulator')
    user_params = {}
    if len(sys.argv) > 1:
        # If there are user parameters.
        user_params = json.loads(sys.argv[1], cls=RuntimeDecoder)
    main(backend, UserMessenger(), **user_params)
