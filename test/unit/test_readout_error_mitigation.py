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

"""Unit tests for ReadoutErrorMitigation."""


from ddt import ddt
import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.result.mitigation.local_readout_mitigator import (
    LocalReadoutMitigator,
)
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeBogota

from programs.estimator import Estimator


@ddt
class TestReadoutErrorMitigation(QiskitTestCase):
    """Test ReadoutErrorMitigation"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    def test_readout_error_mitigation(self):
        """test for local readout error mitigation"""
        backend = FakeBogota()
        backend.set_options(seed_simulator=15)
        mitigator = LocalReadoutMitigator(backend=backend, qubits=[0, 1])

        theta1 = [0, 1, 1, 2, 3, 5]
        exp1 = -1.2788
        with self.subTest("theta1"):
            with Estimator(
                backend, [self.ansatz], [self.observable], readout_mitigator=mitigator
            ) as est:
                est.set_transpile_options(seed_transpiler=15)
                est.set_run_options(seed_simulator=15, shots=10000)
                result = est(parameter_values=theta1)
            self.assertAlmostEqual(result.values[0], exp1, places=2)

        theta2 = [1, 2, 3, 4, 5, 6]
        exp2 = -0.6099
        with self.subTest("theta2"):
            with Estimator(
                backend, [self.ansatz], [self.observable], readout_mitigator=mitigator
            ) as est:
                est.set_transpile_options(seed_transpiler=15)
                est.set_run_options(seed_simulator=15, shots=10000)
                result = est(parameter_values=theta2)
            self.assertAlmostEqual(result.values[0], exp2, places=2)

        with self.subTest("theta1 and theta2"):
            with Estimator(
                backend, [self.ansatz], [self.observable], readout_mitigator=mitigator
            ) as est:
                est.set_transpile_options(seed_transpiler=15)
                est.set_run_options(seed_simulator=15, shots=10000)
                result = est(parameter_values=[theta1, theta2])
            np.testing.assert_almost_equal(result.values, [exp1, exp2], decimal=2)
