import os
import numpy as np

from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA, COBYLA, QNSPSA
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit.opflow import X, Z, I

from qiskit_nature.runtime import VQEClient
from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import HDF5Driver
from qiskit_nature.problems.second_quantization.electronic import (
    ElectronicStructureProblem,
)
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber
from qiskit_nature.transformers.second_quantization.electronic import (
    ActiveSpaceTransformer,
)

from .decorator import get_provider_and_backend
from .base_testcase import BaseTestCase


class TestVQE(BaseTestCase):
    """Test VQE."""

    @classmethod
    @get_provider_and_backend
    def setUpClass(cls, provider, backend_name):
        """Class setup."""
        super().setUpClass()
        cls.provider = provider
        cls.backend_name = backend_name
        cls.backend = cls.provider.get_backend(backend_name)
        # Use callback if on real device to avoid CI timeout
        cls.callback_func = (
            None if cls.backend.configuration().simulator else cls.simple_callback
        )

    def setUp(self) -> None:
        """Test case setup."""
        spin_coupling = (Z ^ Z ^ I) + (I ^ Z ^ Z)
        transverse_field = (X ^ I ^ I) + (I ^ X ^ I) + (I ^ I ^ X)
        hamiltonian = -0.5 * (spin_coupling + 0.5 * transverse_field)
        self.hamiltonian = hamiltonian
        self.observables = [Z ^ Z ^ Z, Z ^ I ^ I]

    def test_vqe_direct(self):
        """Test vqe script."""
        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
            self.hamiltonian
        )
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

        self.assertIsInstance(result["aux_operator_eigenvalues"], dict)

        if self.backend.configuration().simulator:
            self.assertLess(abs(result["eigenvalue"] - reference.eigenvalue), 1)

    def test_nature_program(self):
        """Test vqe nature program."""
        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
            self.hamiltonian
        )
        self.log.info("Exact result: %s", reference.eigenvalue)
        ansatz = EfficientSU2(3, entanglement="linear", reps=3)
        initial_point = np.zeros(ansatz.num_parameters)
        optimizer = COBYLA()

        vqe = VQEClient(
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            provider=self.provider,
            backend=self.backend,
            store_intermediate=True,
            callback=self.callback_func,
            shots=2048,
        )
        result = vqe.compute_minimum_eigenvalue(
            self.hamiltonian, aux_operators=self.observables
        )
        self.log.info("VQE program result: %s", result.eigenvalue)

    def test_nature_full_workflow(self):
        """Test the ground state search workflow from Qiskit Nature."""
        molecule = Molecule(
            geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 2.5]]],
            charge=0,
            multiplicity=1,
        )
        current_dir = os.path.dirname(__file__)
        hdf5_file = os.path.join(current_dir, "lih_sto3g.hdf5")
        driver = HDF5Driver(hdf5_file)

        properties = driver.run()
        particle_number = properties.get_property(ParticleNumber)
        active_space_trafo = ActiveSpaceTransformer(
            num_electrons=particle_number.num_particles, num_molecular_orbitals=3
        )

        problem = ElectronicStructureProblem(driver, transformers=[active_space_trafo])
        qubit_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)

        ansatz = EfficientSU2(4, reps=1, entanglement="linear")

        optimizer = QNSPSA(None, maxiter=300, learning_rate=0.01, perturbation=0.1)
        solver = VQEClient(
            ansatz, optimizer, provider=self.provider, backend=self.backend
        )

        reference_solver = NumPyMinimumEigensolver()

        gse = GroundStateEigensolver(qubit_converter, solver)
        result = gse.solve(problem)

        reference_gse = GroundStateEigensolver(qubit_converter, reference_solver)
        reference_result = reference_gse.solve(problem)

        if self.backend.configuration().simulator:
            self.assertLess(
                abs(result.eigenenergies[0] - reference_result.eigenenergies[0]), 2
            )

    def test_optimization_program(self):
        """Test vqe optimization program."""
        self.hamiltonian = (Z ^ Z ^ I ^ I) + (I ^ Z ^ Z ^ I) + (Z ^ I ^ I ^ Z)

        reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
            self.hamiltonian
        )
        self.log.info("Exact result: %s", reference.eigenvalue)
        ansatz = RealAmplitudes(4, entanglement="linear", reps=3)
        initial_point = np.random.random(ansatz.num_parameters)
        optimizer = SPSA(maxiter=300)

        vqe = VQEClient(
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            provider=self.provider,
            backend=self.backend,
            callback=self.callback_func,
        )
        result = vqe.compute_minimum_eigenvalue(
            self.hamiltonian, aux_operators=self.observables
        )
        self.log.info("VQE program result: %s", result.eigenvalue)

        self.assertIsNotNone(result.eigenvalue)
        self.assertIsNotNone(result.aux_operator_eigenvalues)

        if self.backend.configuration().simulator:
            self.assertLess(abs(result.eigenvalue - reference.eigenvalue), 1)
