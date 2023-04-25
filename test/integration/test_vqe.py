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

"""Test VQE."""

import numpy as np
import qiskit_nature
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import I, X, Z
from qiskit.providers.basicaer import QasmSimulatorPy

from programs.vqe import main

from .base_testcase import BaseTestCase
from .decorator import get_provider_and_backend


class CountingBackend(QasmSimulatorPy):
    """A version of Terra's built-in QasmSimulator that counts how often ``run`` is called."""

    def __init__(self, configuration=None, provider=None, **fields):
        super().__init__(configuration, provider, **fields)
        self.run_count = 0

    def run(self, qobj, **backend_options):
        self.run_count += 1
        return super().run(qobj, **backend_options)


class BlackHoleMessenger:
    """A fake user messenger swallowing everything thrown at it."""

    # pylint: disable=unused-argument
    def publish(self, msg, final=False):
        """Fakes the publish method, does not return anything or have any effect."""
        pass


class TestVQE(BaseTestCase):
    """Test VQE."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):  # pylint: disable=arguments-differ
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name
        cls.backend = cls.provider.get_backend(backend_name)
        # Use callback if on real device to avoid CI timeout
        cls.callback_func = None if cls.backend.configuration().simulator else cls.simple_callback

    def setUp(self) -> None:
        """Test case setup."""
        # avoid deprecation warnings from Qiskit Nature (this has to be set even if no
        # auxiliary operators are used anywhere)
        qiskit_nature.settings.dict_aux_operators = True

        spin_coupling = (Z ^ Z ^ I) + (I ^ Z ^ Z)
        transverse_field = (X ^ I ^ I) + (I ^ X ^ I) + (I ^ I ^ X)
        hamiltonian = -0.5 * (spin_coupling + 0.5 * transverse_field)
        self.hamiltonian = hamiltonian
        self.observables = [Z ^ Z ^ Z, Z ^ I ^ I]

    def test_vqe_direct(self):
        """Test vqe script."""
        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(self.hamiltonian)
        self.log.info("Exact result: %s", reference.eigenvalue)
        ansatz = EfficientSU2(3, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        # test aux operators as dictionaries
        aux_operators = {"magn0": self.observables[0], "magn1": self.observables[1]}

        inputs = {
            "operator": self.hamiltonian,
            "ansatz": ansatz,
            "initial_point": initial_point,
            "optimizer": optimizer,
            "aux_operators": aux_operators,
        }

        options = {"backend_name": self.backend_name}

        job = self.provider.runtime.run(
            program_id="vqe",
            inputs=inputs,
            options=options,
            callback=self.callback_func,
        )
        self.log.debug("Job ID: %s", job.job_id())

        result = job.result()
        self.log.info("Runtime: %s", result["eigenvalue"])

        with self.subTest(msg="check auxops is dict"):
            self.assertIsInstance(result["aux_operator_eigenvalues"], dict)

        with self.subTest(msg="check history shape"):
            self.assertIsInstance(result["optimizer_history"], dict)
            # 300x 2 evaluations plus 25x 2 for the initial calibration and a final one
            self.assertEqual(len(result["optimizer_history"]["nfevs"]), 651)

        if self.backend.configuration().simulator:
            with self.subTest(msg="check eigenvalue"):
                self.assertLess(abs(result["eigenvalue"] - reference.eigenvalue), 1)

    def test_batched_evaluations(self):
        """Test circuit evaluations are batched per default."""
        backend = CountingBackend()

        ansatz = EfficientSU2(3, entanglement="linear", reps=1)
        initial_point = np.random.random(ansatz.num_parameters)
        maxiter = 10
        optimizer = SPSA(maxiter, learning_rate=0.01, perturbation=0.1)

        main(
            backend,
            user_messenger=BlackHoleMessenger(),
            operator=self.hamiltonian,
            ansatz=ansatz,
            initial_point=initial_point,
            optimizer=optimizer,
        )

        # one evaluation per iteration plus a final one from SPSA and a final one from VQE
        self.assertEqual(backend.run_count, maxiter + 2)
