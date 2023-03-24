"""Unit tests for QASM3 integration with the primitives."""

import copy
import logging
from test.unit import combine
from typing import Iterable, Union
import unittest

from ddt import ddt
import numpy as np

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers.fake_provider import FakeMontreal
from qiskit.qasm3 import loads as qasm3_loads
from qiskit.quantum_info import SparsePauliOp

from programs.sampler import Sampler, SamplerConstant
from programs.estimator import Estimator, EstimatorConstant

from .test_estimator import get_simulator


class PrimitivesExperiment:
    """Represents an execution of the primitives using OpenQasm or QuantumCircuit"""

    def __init__(self, program, params, observables, expected_quasi_prob, expected_values) -> None:
        self._program = program
        self._params = params
        self._observables = observables
        self._expected_quasi_prob = expected_quasi_prob
        self._expected_values = expected_values

    @property
    def program(self):
        """Returns the program."""
        return self._program

    @program.setter
    def program(self, new_program):
        self._program = new_program

    @property
    def params(self):
        """Returns the experiment params."""
        return self._params

    @property
    def observables(self):
        """Returns the observables used in the (estimator) experiment"""
        return self._observables

    @property
    def expected_quasi_prob(self):
        """Returns the exptected quasi-probability distribution."""
        return self._expected_quasi_prob

    @property
    def expected_values(self):
        """Return the expected expectation values."""
        return self._expected_values


# TODO: remove this class when non-flexible interface is no longer supported in provider
@ddt
class TestQASMPrimitives(unittest.TestCase):
    """Test the use of OpenQASM programs with the primitives"""

    def setUp(self):
        super().setUp()
        self._run_config = {"seed_simulator": 15}
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()

        self._qc_bell = PrimitivesExperiment(
            program=bell,
            params=[],
            observables=SparsePauliOp("ZZ"),
            expected_quasi_prob={0: 0.5, 3: 0.5},
            expected_values=1,
        )
        self._qasm2_no_params = PrimitivesExperiment(
            program="""
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[3];
                creg c[3];
                h q[0];
                cz q[0],q[1];
                cx q[0],q[2];
                measure q[0] -> c[0];
                measure q[1] -> c[1];
                measure q[2] -> c[2];
            """,
            params=[],
            observables=SparsePauliOp("ZZZ"),
            expected_quasi_prob={0: 0.5, 5: 0.5},
            expected_values=1,
        )

        self._qasm3_no_params = PrimitivesExperiment(
            program="""
                OPENQASM 3;
                include "stdgates.inc";
                qubit[3] q;
                bit[3] c;
                h q[0];
                cz q[0], q[1];
                cx q[0], q[2];
                c[0] = measure q[0];
                c[1] = measure q[1];
                c[2] = measure q[2];
            """,
            params=[],
            observables=SparsePauliOp("ZZZ"),
            expected_quasi_prob={0: 0.5, 5: 0.5},
            expected_values=1,
        )

        self._qasm3_wtih_params = PrimitivesExperiment(
            program="""
                OPENQASM 3;
                include "stdgates.inc";
                input angle theta1;
                input angle theta2;
                bit[3] c;
                qubit[3] q;
                rz(theta1) q[0];
                sx q[0];
                rz(theta2) q[0];
                cx q[0], q[1];
                h q[1];
                cx q[1], q[2];
                c[0] = measure q[0];
                c[1] = measure q[1];
                c[2] = measure q[2];
            """,
            params=[1.5, 1],
            observables=SparsePauliOp("ZZZ"),
            expected_quasi_prob={0: 0.25, 1: 0.25, 6: 0.25, 7: 0.25},
            expected_values=0,
        )

        self._qasm3_subroutine = PrimitivesExperiment(
            program="""
                OPENQASM 3.0;
                include "stdgates.inc";

                input float[64] a;
                qubit[3] q;
                bit[3] out;

                let aliased = q[0:1];

                gate my_gate(a) c, t {
                gphase(a / 2);
                ry(a) c;
                cx c, t;
                }
                gate my_phase(a) c {
                ctrl @ inv @ gphase(a) c;
                }

                my_gate(a * 2) aliased[0], q[{1, 2}][0];
                h q[2];

                out[0] = measure q[0];
                out[1] = measure q[1];
                out[2] = measure q[2];

            """,
            params=[np.pi / 2],
            observables=SparsePauliOp("ZZZ"),
            expected_quasi_prob={3: 0.5, 7: 0.5},
            expected_values=0,
        )

        self._qasm3_dynamic = PrimitivesExperiment(
            program="""
                OPENQASM 3;
                include "stdgates.inc";
                bit[2] c;
                qubit[3] q;
                h q[0];
                cx q[0], q[1];
                c[0] = measure q[0];
                h q[0];
                cx q[0], q[1];
                c[1] = measure q[0];
                if (c[0]) {
                    x q[2];
                } else {
                    h q[2];
                    z q[2];
                }
            """,
            params=[],
            observables=SparsePauliOp("ZZZ"),
            expected_quasi_prob={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
            expected_values=None,
        )

        self._qasm3_subroutine_dynamic = PrimitivesExperiment(
            program="""
                OPENQASM 3.0;
                include "stdgates.inc";

                input float[64] a;
                qubit[3] q;
                bit[2] mid;
                bit[3] out;

                let aliased = q[0:1];

                gate my_gate(a) c, t {
                gphase(a / 2);
                ry(a) c;
                cx c, t;
                }
                gate my_phase(a) c {
                ctrl @ inv @ gphase(a) c;
                }

                my_gate(a * 2) aliased[0], q[{1, 2}][0];
                measure q[0] -> mid[0];
                measure q[1] -> mid[1];

                while (mid == "00") {
                reset q[0];
                reset q[1];
                my_gate(a) q[0], q[1];
                my_phase(a - pi/2) q[1];
                mid[0] = measure q[0];
                mid[1] = measure q[1];
                }

                if (mid[0]) {
                let inner_alias = q[{0, 1}];
                reset inner_alias;
                }

                out = measure q;
            """,
            params=[np.pi / 2],
            observables=SparsePauliOp("ZZZ"),
            expected_quasi_prob={3: 1},
            expected_values=None,
        )

    def _extract_experiment_data(
        self, experiments: Union[PrimitivesExperiment, Iterable[PrimitivesExperiment]]
    ):
        """
        Aux function to extract experiments data from a list of experiments

        Args:
            experiments: The `PrimitivesExperiment` experiments to run.

        Returns:
            A tuple containing the programs, indices, parameter values, observables, expected quasi
            probabilities, and expected values.

        """
        if isinstance(experiments, PrimitivesExperiment):
            # only one experiment that tests passing a single qasm program
            programs = experiments.program
            params = [experiments.params]
            observables = experiments.observables
            expected_quasi_prob = experiments.expected_quasi_prob
            expected_values = [experiments.expected_values]
            indices = [0]
            param_values = {"parameter_values": params} if len(experiments.params) != 0 else {}
        else:
            # many experiments that test passing a list of qasm program and quantum circuits
            programs, params, observables, expected_quasi_prob, expected_values = map(
                list,
                zip(
                    *[
                        (
                            e.program,
                            e.params,
                            e.observables,
                            e.expected_quasi_prob,
                            e.expected_values,
                        )
                        for e in experiments
                    ]
                ),
            )
            indices = list(range(len(experiments)))
            param_values = {"parameter_values": params}
        return (
            programs,
            indices,
            param_values,
            observables,
            expected_quasi_prob,
            expected_values,
        )

    def _run_primitive_test(
        self,
        primitive: type,
        experiments: Union[PrimitivesExperiment, Iterable[PrimitivesExperiment]],
    ) -> None:
        """Aux function to run both Sampler and Estimator experiments using OpenQASM inputs.

        Args:
            primitive: either `programs.sampler.Sampler` or `programs.estimator.Estimator` type object
            experiments: The `PrimitivesExperiment` experiments to run

        Returns:
            None
        """
        backend = get_simulator(FakeMontreal())
        shots = 10000

        (
            programs,
            indices,
            param_values,
            observables,
            expected_quasi_prob,
            expected_values,
        ) = self._extract_experiment_data(experiments)

        if primitive == Sampler:
            sampler = Sampler(circuits=programs, backend=backend)
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler.run(indices, **param_values)

            # Assert quasi probabilites
            self._compare_probs(result.quasi_dists, expected_quasi_prob)

        elif primitive == Estimator:
            estimator = Estimator(circuits=programs, backend=backend, observables=observables)
            estimator.set_run_options(shots=shots, **self._run_config)
            result = estimator.run(indices, indices, **param_values)

            # Assert expectation values
            np.testing.assert_allclose(result.values, expected_values, atol=5e-2)
        else:
            raise Exception("Only sampler and estimator tests supported.")

        # Assert number of shots
        for metadata in result.metadata:
            self.assertEqual(metadata["shots"], shots)

    def _compare_probs(self, probabilities, target):
        if not isinstance(target, list):
            target = [target]
        self.assertEqual(len(probabilities), len(target))
        for prob, targ in zip(probabilities, target):
            for key, t_val in targ.items():
                if key in prob:
                    self.assertAlmostEqual(prob[key], t_val, places=1)
                else:
                    self.assertAlmostEqual(t_val, 0, places=1)

    @combine(primitive=[Sampler, Estimator])
    def test_qasm2_no_params(self, primitive: type):
        """Test a QASM2 program that doesn't take any params"""
        self._run_primitive_test(primitive, self._qasm2_no_params)

    @combine(primitive=[Sampler, Estimator])
    def test_qasm3_no_params(self, primitive: type):
        """Test a QASM3 program that doesn't take any params"""
        self._run_primitive_test(primitive, self._qasm3_no_params)

    @combine(primitive=[Sampler, Estimator])
    def test_qasm3_with_params_angle(self, primitive):
        """Test a QASM3 program that takes two angle params"""
        self._run_primitive_test(primitive, self._qasm3_wtih_params)

    @combine(primitive=[Sampler, Estimator])
    def test_qasm3_subroutines(self, primitive):
        """Test a QASM3 program that contains subroutines"""
        self._run_primitive_test(primitive, self._qasm3_subroutine)

    @combine(primitive=[Sampler, Estimator])
    def test_qasm3_multiple_circuits(self, primitive):
        """Test passing a list of QASM3 programs to the Sampler"""

        experiments = [
            self._qasm3_no_params,
            self._qasm3_wtih_params,
            self._qasm3_subroutine,
        ]
        self._run_primitive_test(primitive, experiments)

    @combine(primitive=[Sampler, Estimator])
    def test_qasm3_mix_qasm_and_quantum_circuit(self, primitive):
        """Test passing a mixed list of QASM3 programs and QuantumCircut objects to Sampler"""

        experiments = [
            self._qasm3_no_params,
            self._qc_bell,
            self._qasm3_wtih_params,
        ]
        self._run_primitive_test(primitive, experiments)

    @combine(primitive=[Sampler, Estimator])
    def test_qasm_circuit_ids(self, primitive):
        """Test passing a circuit ids for QASM3 programs to Sampler"""
        backend = get_simulator(FakeMontreal())
        shots = 1000

        id_qasm2 = str(id(QuantumCircuit.from_qasm_str(self._qasm2_no_params.program)))
        id_qasm3 = str(id(qasm3_loads(self._qasm3_no_params.program)))

        if primitive == Sampler:
            sampler = Sampler(
                circuits={
                    id_qasm2: self._qasm2_no_params.program,
                    id_qasm3: self._qasm3_no_params.program,
                },
                circuit_ids=[id_qasm2, id_qasm3],
                backend=backend,
            )
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler.run()

            # Assert quasi probabilities
            self._compare_probs(
                result.quasi_dists,
                [
                    self._qasm2_no_params.expected_quasi_prob,
                    self._qasm3_no_params.expected_quasi_prob,
                ],
            )
        else:
            estimator = Estimator(
                circuits={
                    id_qasm2: self._qasm2_no_params.program,
                    id_qasm3: self._qasm3_no_params.program,
                },
                circuit_ids=[id_qasm2, id_qasm3],
                observables=[self._qasm2_no_params.observables, self._qasm3_no_params.observables],
                backend=backend,
            )
            estimator.set_run_options(shots=shots, **self._run_config)
            result = estimator.run(observable_indices=[0, 0])

            # Assert expectation values
            np.testing.assert_allclose(
                result.values,
                [self._qasm2_no_params.expected_values, self._qasm3_no_params.expected_values],
                atol=5e-2,
            )

        self.assertEqual(result.metadata[0]["shots"], shots)
        self.assertEqual(result.metadata[1]["shots"], shots)

    @combine(primitive=[Sampler, Estimator])
    def test_no_qasm_version(self, primitive):
        """Test a QASM program that doesn't have a QASM version header"""
        experiment = copy.deepcopy(self._qasm3_no_params)
        experiment.program = experiment.program.replace("OPENQASM 3;", "")

        # retrieve "programs.sampler" or "programs.estimator" logger accordingly
        logger = logging.getLogger(f"programs.{primitive.__name__.lower()}")
        with self.assertLogs(logger=logger, level="WARNING") as log_records:

            # Run primitive experiment
            self._run_primitive_test(primitive, experiment)

            # In the future, INVALID_QASM_VERSION_MESSAGE constant will live in a common constants file
            # and there will be no need for finding the right constant class
            constant_class = SamplerConstant if primitive == Sampler else EstimatorConstant

            # Assert warning in logs
            self.assertIn(
                constant_class.INVALID_QASM_VERSION_MESSAGE,
                log_records.output[0],
            )

    @unittest.skip("Skip until dynamic circuits are rejected")
    @combine(primitive=[Sampler, Estimator])
    def test_qasm3_dynamic(self, primitive):
        """Test a dynamic QASM3 program"""
        self._run_primitive_test(primitive, self._qasm3_dynamic)

    @unittest.skip("Skip until dynamic circuits are rejected")
    @combine(primitive=[Sampler, Estimator])
    def test_qasm3_subroutines_dynamic(self, primitive):
        """Test a dynamic QASM3 program that contains subroutines"""
        self._run_primitive_test(primitive, self._qasm3_subroutine_dynamic)

    @combine(primitive=[Sampler, Estimator])
    def test_invalid_qasm(self, primitive):
        """Test invalid QASM inputs"""
        backend = get_simulator()
        invalid_qasm2 = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cz q[0];
        """
        invalid_qasm3 = """
            OPENQASM 3.0;
            c[0] = measure qr[0];
            c[1] = measure qr[1];
        """
        invalid_qasm2_random_text = """
            OPENQASM 2;
            Lorem Ipsum
        """
        invalid_qasm3_random_text = """
            OPENQASM 3.0;
            Lorem Ipsum
        """
        invalid_no_qasm_version_random_text = """
            Lorem;
            ipsum;
            dolor;
            sit amet,
        """

        qasm_inputs = (
            invalid_qasm2,
            invalid_qasm3,
            invalid_qasm2_random_text,
            invalid_qasm3_random_text,
            invalid_no_qasm_version_random_text,
        )

        for qasm_input in qasm_inputs:
            args = {"circuits": qasm_input, "backend": backend}
            if primitive == Estimator:
                args["observables"] = SparsePauliOp("XYZ")
            self.assertRaises(QiskitError, primitive, **args)
