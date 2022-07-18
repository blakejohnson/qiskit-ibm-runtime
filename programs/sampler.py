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

from mthree import M3Mitigation
from mthree.utils import final_measurement_mapping
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.primitives import BaseSampler, SamplerResult
from qiskit.providers import Options
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import QuasiDistribution, Result
from qiskit.transpiler import PassManager

# Number of effective shots per measurement error rate
DEFAULT_SHOTS = 25000


class Sampler(BaseSampler):
    """
    Sampler class
    """

    def __init__(
        self,
        backend: Backend,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        parameters: Iterable[Iterable[Parameter]] | None = None,
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

        super().__init__(circuits, parameters)

        self._backend = backend
        self._run_options = Options()
        self._is_closed = False

        self._transpile_options = Options()
        self._bound_pass_manager = bound_pass_manager

        self._preprocessed_circuits: list[QuantumCircuit] | None = None
        self._transpiled_circuits: list[QuantumCircuit] | None = None
        self._skip_transpilation = skip_transpilation
        self._m3_mitigation: M3Mitigation | None = None

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
        elif self._transpiled_circuits is None:
            # Only transpile if have not done so yet
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

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        self._check_is_closed()

        # This line does the actual transpilation
        transpiled_circuits = self.transpiled_circuits
        bound_circuits = [
            transpiled_circuits[i].bind_parameters((dict(zip(self._parameters[i], value))))
            for i, value in zip(circuits, parameter_values)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        result = self._backend.run(bound_circuits, **run_opts.__dict__).result()

        return self._postprocessing(result, bound_circuits)

    def close(self):
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
            if self._m3_mitigation is None:
                quasis.append(QuasiDistribution({k: v / shots for k, v in count.items()}))
            else:
                mapping = final_measurement_mapping(circ)
                quasi, details = self._m3_mitigation.apply_correction(
                    count, mapping, return_mitigation_overhead=True, details=True
                )
                quasis.append(QuasiDistribution(quasi))
                mitigation_overheads.append(quasi.mitigation_overhead)
                mitigation_times.append(details["time"])

        metadata = []
        for idx, res in enumerate(result.results):
            _temp_dict = {"header_metadata": res.header.metadata, "shots": shots}
            if self._m3_mitigation:
                _temp_dict["readout_mitigation_overhead"] = mitigation_overheads[idx]
                _temp_dict["readout_mitigation_time"] = mitigation_times[idx]
            metadata.append(_temp_dict)

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

    def calibrate_m3_mitigation(self, backend) -> None:
        """Calibrate M3 mitigation

        Args:
            backend: The backend.
        """
        mappings = [final_measurement_mapping(circ) for circ in self.transpiled_circuits]
        self._m3_mitigation = M3Mitigation(backend)
        self._m3_mitigation.cals_from_system(mappings, shots=DEFAULT_SHOTS)

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
    transpilation_settings=None,
    resilience_settings=None,
):

    """Sample distributions generated by given circuits executed on the target backend.

    Parameters:
        backend (ProgramBackend): Qiskit backend instance.
        user_messenger (UserMessenger): Used to communicate with the program user.
        circuits: (QuantumCircuit or list): A single list of QuantumCircuits.
        parameters (list): Parameters of the quantum circuits.
        circuit_indices (list): Indexes of the circuits to evaluate.
        parameter_values (list): Concrete parameters to be bound.
        skip_transpilation (bool): (Deprecated) Skip transpiling of circuits, default=False.
        run_options (dict): A collection of kwargs passed to backend.run().
        transpilation_settings (dict): Transpilation settings.
        resilience_settings (dict): Resilience settings.

    Returns:
        dict: A dictionary with quasi-probabilities and metadata.
    """
    transpilation_settings = transpilation_settings or {}
    optimization_settings = transpilation_settings.pop("optimization_settings", {})
    skip_transpilation = transpilation_settings.pop("skip_transpilation", skip_transpilation)

    sampler = Sampler(
        backend=backend,
        circuits=circuits,
        parameters=parameters,
        skip_transpilation=skip_transpilation,
    )

    transpile_options = transpilation_settings.copy()
    transpile_options["optimization_level"] = optimization_settings.get("level", 1)

    sampler.set_transpile_options(**transpile_options)
    # Must transpile circuits before calibrating M3
    _ = sampler.transpiled_circuits

    resilience_settings = resilience_settings or {}

    if resilience_settings.get("level", 0) == 1:
        sampler.calibrate_m3_mitigation(backend)

    run_options = run_options or {}
    result = sampler(
        circuits=circuit_indices,
        parameter_values=parameter_values,
        **run_options,
    )

    result_dict = sampler.result_to_dict(result, sampler.circuits, circuit_indices)

    return result_dict
