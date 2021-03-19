# Circuit execution runtime program

import json
import sys
from typing import Any

import numpy as np

from qiskit import Aer
from qiskit import transpile
from qiskit.compiler import assemble
from qiskit.assembler.disassemble import disassemble
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.qobj import QasmQobj
from qiskit.result import Result


# Encoder and decoder will be available for import in qiskit-ibmq-provider

class RuntimeEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy arrays, complex numbers, and circuits."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, 'tolist'):
            return {'type': 'array', 'value': obj.tolist()}
        if isinstance(obj, complex):
            return {'type': 'complex', 'value': [obj.real, obj.imag]}
        if isinstance(obj, QuantumCircuit):
            return {'type': 'circuits', 'value': assemble(obj).to_dict()}
        if isinstance(obj, Result):
            return {'type': 'result', 'value': obj.to_dict()}
        return super().default(obj)


class RuntimeDecoder(json.JSONDecoder):
    """JSON Decoder for Numpy arrays, complex numbers, and circuits."""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if 'type' in obj:
            if obj['type'] == 'complex':
                val = obj['value']
                return val[0] + 1j * val[1]
            if obj['type'] == 'array':
                return np.array(obj['value'])
            if obj['type'] == 'circuits':
                circuits, _, _ = disassemble(QasmQobj.from_dict(obj['value']))
                if len(circuits) == 1:
                    return circuits[0]
                return circuits
            if obj['type'] == 'result':
                return Result.from_dict(obj['value'])
        return obj


def post_interim_result(text):
    print(json.dumps({'post': text}, cls=RuntimeEncoder))


def main(backend, **kwargs):
    circs = kwargs.pop('circuits', None)
    if not circs:
        raise ValueError("Circuits are required.")
    transpiled = transpile(circuits=circs, backend=backend, **kwargs)
    qobj = assemble(transpiled, backend=backend)  # TODO - remove when not using Aer
    result = backend.run(qobj).result()

    print(json.dumps({'results': result}, cls=RuntimeEncoder))


if __name__ == '__main__':
    # provider = QuantumProgramProvider()
    # backend = provider.backends()[0]

    # the code currently uses Aer instead of runtime provider
    backend = Aer.get_backend('qasm_simulator')
    user_params = {}
    if len(sys.argv) > 1:
        # If there are user parameters.
        user_params = json.loads(sys.argv[1], cls=RuntimeDecoder)
    main(backend, **user_params)
