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
Sampler class
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import copy
from typing import cast

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.primitives import BaseSampler, SamplerResult
from qiskit.providers import Options
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import BaseReadoutMitigator, QuasiDistribution, Result
from qiskit.transpiler import PassManager


class Sampler(BaseSampler):
    """
    Sampler class
    """

    def __init__(
        self,
        backend: Backend,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        parameters: Iterable[Iterable[Parameter]] | None = None,
        readout_mitigator: BaseReadoutMitigator | None = None,
        bound_pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
    ):
        """
        Args:
            circuits: circuits to be executed
        Raises:
            TypeError: The given type of arguments is invalid.
        """
        if not isinstance(backend, Backend):
            raise TypeError(f"backend should be BackendV1, not {type(backend)}.")

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        super().__init__(circuits, parameters)

        self._backend = backend
        self._readout_mitigator = readout_mitigator
        self._run_options = Options()
        self._is_closed = False

        self._transpile_options = Options()
        self._bound_pass_manager = bound_pass_manager

        self._preprocessed_circuits: list[QuantumCircuit] | None = None
        self._transpiled_circuits: list[QuantumCircuit] | None = None
        self._skip_transpilation = skip_transpilation

    @property
    def preprocessed_circuits(self) -> list[QuantumCircuit]:
        """
        Preprocessed quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._check_is_closed()
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
        self._check_is_closed()
        if self._skip_transpilation:
            self._transpiled_circuits = list(self._circuits)
        else:
            self._transpile()
        return self._transpiled_circuits

    @property
    def backend(self) -> Backend:
        """
        Returns:
            The backend which this sampler object based on
        """
        return self._backend

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
        self._check_is_closed()
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
        self._check_is_closed()

        self._transpile_options.update_options(**fields)
        return self

    def __call__(
        self,
        circuit_indices: Sequence[int] | None = None,
        parameter_values: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **run_options,
    ) -> SamplerResult:
        self._check_is_closed()
        if isinstance(parameter_values, np.ndarray):
            parameter_values = parameter_values.tolist()

        if parameter_values and not isinstance(parameter_values[0], (np.ndarray, Sequence)):
            parameter_values = cast("Sequence[float]", parameter_values)
            parameter_values = [parameter_values]
        if circuit_indices is None and parameter_values is not None and len(self.circuits) == 1:
            circuit_indices = [0] * len(parameter_values)
        if circuit_indices is None:
            circuit_indices = list(range(len(self.circuits)))
        if parameter_values is None:
            parameter_values = [[]] * len(circuit_indices)
        parameter_values = cast("Sequence[Sequence[float]]", parameter_values)

        # Validation
        if len(circuit_indices) != len(parameter_values):
            raise QiskitError(
                f"The number of circuits ({len(circuit_indices)}) does not match "
                f"the number of parameter sets ({len(parameter_values)})."
            )

        for i, value in zip(circuit_indices, parameter_values):
            if len(value) != len(self.parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self.parameters[i])}) for the {i}-th circuit."
                )

        transpiled_circuits = self.transpiled_circuits
        bound_circuits = [
            transpiled_circuits[i].bind_parameters((dict(zip(self._parameters[i], value))))
            for i, value in zip(circuit_indices, parameter_values)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        result = self._backend.run(bound_circuits, **run_opts.__dict__).result()

        return self._postprocessing(result)

    def close(self):
        self._is_closed = True

    def _postprocessing(self, result: Result) -> SamplerResult:
        if not isinstance(result, Result):
            raise TypeError("result must be an instance of Result.")

        counts = result.get_counts()
        if not isinstance(counts, list):
            counts = [counts]

        shots = sum(counts[0].values())

        quasis = []
        for count in counts:
            if self._readout_mitigator is None:
                quasis.append(QuasiDistribution({k: v / shots for k, v in count.items()}))
            else:
                quasis.append(self._readout_mitigator.quasi_probabilities(count))

        metadata = [
            {"header_metadata": res.header.metadata, "shots": shots} for res in result.results
        ]

        return SamplerResult(quasi_dists=quasis, metadata=metadata)

    def _transpile(self):
        self._transpiled_circuits = cast(
            "list[QuantumCircuit]",
            transpile(
                self.preprocessed_circuits,
                self.backend,
                **self.transpile_options.__dict__,
            ),
        )

    def _check_is_closed(self):
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

    def _bound_pass_manager_run(self, circuits):
        if self._bound_pass_manager is None:
            return circuits
        else:
            return cast("list[QuantumCircuit]", self._bound_pass_manager.run(circuits))

    @staticmethod
    def result_to_dict(result: SamplerResult, circuits, circuit_indices):
        """Convert ``SamplerResult`` to a dictionary

        Args:
            result: The result of ``Sampler``
            circuits: The circuits
            circuit_indices: The circuit indices

        Returns:
            A dictionary representing the result.

        """
        ret = result.__dict__
        ret["quasi_dists"] = [
            dist.binary_probabilities(circuits[index].num_clbits)
            for index, dist in zip(circuit_indices, result.quasi_dists)
        ]
        return ret


def main(
    backend,
    user_messenger,  # pylint: disable=unused-argument
    circuits,
    circuit_indices,
    parameters=None,
    parameter_values=None,
    skip_transpilation=False,
    run_options=None,
):

    """Sample distributions generated by given circuits executed on the target backend.

    Parameters:
        backend (ProgramBackend): Qiskit backend instance.
        user_messenger (UserMessenger): Used to communicate with the program user.
        circuits: (QuantumCircuit or list): A single list of QuantumCircuits.
        parameters (list): Parameters of the quantum circuits.
        circuit_indices (list): Indexes of the circuits to evaluate.
        parameter_values (list): Concrete parameters to be bound.
        skip_transpilation (bool): Skip transpiling of circuits, default=False.
        run_options (dict): A collection of kwargs passed to backend.run().

    Returns:
        dict: A dictionary with quasi-probabilities and metadata.
    """
    sampler = Sampler(
        backend=backend,
        circuits=circuits,
        parameters=parameters,
        skip_transpilation=skip_transpilation,
    )

    run_options = run_options or {}
    result = sampler(
        circuit_indices=circuit_indices,
        parameter_values=parameter_values,
        **run_options,
    )

    result_dict = sampler.result_to_dict(result, circuits, circuit_indices)

    return result_dict
