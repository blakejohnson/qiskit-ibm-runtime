"""Test dynamical decoupling insertion pass."""

from os import environ
from test.unit import combine
import unittest

from ddt import ddt
import numpy as np
from qiskit import transpile
from qiskit.circuit import Delay, QuantumCircuit
from qiskit.circuit.library import RZGate, XGate
from qiskit.providers.fake_provider import FakeCairo
from qiskit.quantum_info import SparsePauliOp

from programs.estimator import main as mainEstimator
from programs.sampler import main as mainSampler


# TO-DO: add combinations of different primitives, circuits and observables, optimization levels,
# reslience levels, etc.
@ddt
class TestDynamicalDecouplingEstimator(unittest.TestCase):
    """Tests Dynamical Decoupling in Estimator."""

    def setUp(self):
        super().setUp()
        self._backend = FakeCairo()
        self._ghz4 = QuantumCircuit(4, 4)
        self._ghz4.h(0)
        self._ghz4.cx(0, 1)
        self._ghz4.cx(1, 2)
        self._ghz4.cx(2, 3)
        self._observable = SparsePauliOp("ZZZZ")

    def test_dd_opt0(self):
        """Test for dynamical decoupling w/ optimization level=0"""
        self.assertTrue("PRIMITIVES_DEBUG" in environ)
        self.assertEqual(environ["PRIMITIVES_DEBUG"], "true")
        optimization_level = 0
        result = mainEstimator(
            backend=self._backend,
            user_messenger=None,
            circuits=[self._ghz4],
            observables=[self._observable],
            circuit_indices=[0],
            observable_indices=[0],
            parameter_values=[[]],
            transpilation_settings={
                "optimization_settings": {"level": optimization_level},
                "seed_transpiler": 123,
            },
            resilience_settings={"level": 0},
        )
        transpiled_ghz4 = transpile(
            self._ghz4, self._backend, optimization_level=optimization_level, seed_transpiler=123
        )
        transpiled_ghz4.measure([0, 1, 2, 3], [0, 1, 2, 3])
        self.assertEqual(len(result["metadata"]), 1)
        self.assertEqual(len(result["metadata"][0]["bound_circuits"]), 1)
        self.assertEqual(result["metadata"][0]["bound_circuits"][0], transpiled_ghz4)

    def test_dd_opt1(self):
        """Test for dynamical decoupling w/ optimization level=1"""
        self.assertTrue("PRIMITIVES_DEBUG" in environ)
        self.assertEqual(environ["PRIMITIVES_DEBUG"], "true")
        optimization_level = 1
        result = mainEstimator(
            backend=self._backend,
            user_messenger=None,
            circuits=[self._ghz4],
            observables=[self._observable],
            circuit_indices=[0],
            observable_indices=[0],
            parameter_values=[[]],
            transpilation_settings={
                "optimization_settings": {"level": optimization_level},
                "seed_transpiler": 123,
            },
            resilience_settings={"level": 0},
        )

        transpiled_ghz4 = transpile(
            self._ghz4, self._backend, optimization_level=optimization_level, seed_transpiler=123
        )
        qubits = [0, 1, 2, 3]

        transpiled_ghz4_dd = transpiled_ghz4.copy()
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(96), [qubits[1]], front=True)
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(2592), [qubits[2]], front=True)
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(4032), [qubits[3]], front=True)
        for i in set(range(27)) - set(qubits):
            transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(8320), [i], front=True)

        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(560), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(1120), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(np.pi), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(-np.pi), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(560), [qubits[0]])

        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(200), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(400), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(np.pi), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(-np.pi), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(200), [qubits[1]])
        transpiled_ghz4_dd.measure(qubits, [0, 1, 2, 3])

        self.assertEqual(len(result["metadata"]), 1)
        self.assertEqual(len(result["metadata"][0]["bound_circuits"]), 1)
        self.assertEqual(result["metadata"][0]["bound_circuits"][0], transpiled_ghz4_dd)

    @combine(optimization_level=[2, 3])
    def test_dd_opt2_3(self, optimization_level):
        """Test for dynamical decoupling w/ optimization level=2 and 3"""
        self.assertTrue("PRIMITIVES_DEBUG" in environ)
        self.assertEqual(environ["PRIMITIVES_DEBUG"], "true")
        result = mainEstimator(
            backend=self._backend,
            user_messenger=None,
            circuits=[self._ghz4],
            observables=[self._observable],
            circuit_indices=[0],
            observable_indices=[0],
            parameter_values=[[]],
            transpilation_settings={
                "optimization_settings": {"level": optimization_level},
                "seed_transpiler": 123,
            },
            resilience_settings={"level": 0},
        )

        transpiled_ghz4 = transpile(
            self._ghz4, self._backend, optimization_level=optimization_level, seed_transpiler=123
        )
        qubits = [3, 5, 8, 9]

        transpiled_ghz4_dd = transpiled_ghz4.copy()
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(96), [qubits[1]], front=True)
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(1344), [qubits[2]], front=True)
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(2128), [qubits[3]], front=True)
        for i in set(range(27)) - set(qubits):
            transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(6576), [i], front=True)

        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(436), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(872), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(np.pi), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(-np.pi), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(436), [qubits[0]])

        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(240), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(480), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(np.pi), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(-np.pi), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(240), [qubits[1]])
        transpiled_ghz4_dd.measure(qubits, [0, 1, 2, 3])

        self.assertEqual(len(result["metadata"]), 1)
        self.assertEqual(len(result["metadata"][0]["bound_circuits"]), 1)
        self.assertEqual(result["metadata"][0]["bound_circuits"][0], transpiled_ghz4_dd)


@ddt
class TestDynamicalDecouplingSampler(unittest.TestCase):
    """Tests Dynamical Decoupling in Sampler."""

    def setUp(self):
        super().setUp()
        self._backend = FakeCairo()
        self._ghz4 = QuantumCircuit(4, 4)
        self._ghz4.h(0)
        self._ghz4.cx(0, 1)
        self._ghz4.cx(1, 2)
        self._ghz4.cx(2, 3)

    def test_dd_opt0(self):
        """Test for dynamical decoupling w/ optimization level=0"""
        self.assertTrue("PRIMITIVES_DEBUG" in environ)
        self.assertEqual(environ["PRIMITIVES_DEBUG"], "true")
        optimization_level = 0
        ghz4 = self._ghz4.copy()
        ghz4.measure([0, 1, 2, 3], [0, 1, 2, 3])
        result = mainSampler(
            backend=self._backend,
            user_messenger=None,
            circuits=[ghz4],
            circuit_indices=[0],
            observable_indices=[0],
            parameter_values=[[]],
            transpilation_settings={
                "optimization_settings": {"level": optimization_level},
                "seed_transpiler": 123,
            },
            resilience_settings={"level": 0},
        )
        transpiled_ghz4 = transpile(
            ghz4, self._backend, optimization_level=optimization_level, seed_transpiler=123
        )
        self.assertEqual(len(result["metadata"]), 1)
        self.assertEqual(len(result["metadata"][0]["bound_circuits"]), 1)
        self.assertEqual(result["metadata"][0]["bound_circuits"][0], transpiled_ghz4)

    def test_dd_opt1(self):
        """Test for dynamical decoupling w/ optimization level=1"""
        self.assertTrue("PRIMITIVES_DEBUG" in environ)
        self.assertEqual(environ["PRIMITIVES_DEBUG"], "true")
        optimization_level = 1
        ghz4 = self._ghz4.copy()
        ghz4.measure([0, 1, 2, 3], [0, 1, 2, 3])
        result = mainSampler(
            backend=self._backend,
            user_messenger=None,
            circuits=[ghz4],
            circuit_indices=[0],
            observable_indices=[0],
            parameter_values=[[]],
            transpilation_settings={
                "optimization_settings": {"level": optimization_level},
                "seed_transpiler": 123,
            },
            resilience_settings={"level": 0},
        )

        transpiled_ghz4 = transpile(
            self._ghz4, self._backend, optimization_level=optimization_level, seed_transpiler=123
        )
        qubits = [0, 1, 2, 3]

        transpiled_ghz4_dd = transpiled_ghz4.copy()
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(96), [qubits[1]], front=True)
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(2592), [qubits[2]], front=True)
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(4032), [qubits[3]], front=True)
        for i in set(range(27)) - set(qubits):
            transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(8320), [i], front=True)

        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(560), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(1120), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(np.pi), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(-np.pi), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(560), [qubits[0]])

        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(200), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(400), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(np.pi), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(-np.pi), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(200), [qubits[1]])
        transpiled_ghz4_dd.measure(qubits, [0, 1, 2, 3])

        self.assertEqual(len(result["metadata"]), 1)
        self.assertEqual(len(result["metadata"][0]["bound_circuits"]), 1)
        self.assertEqual(result["metadata"][0]["bound_circuits"][0], transpiled_ghz4_dd)

    @combine(optimization_level=[2, 3])
    def test_dd_opt2_3(self, optimization_level):
        """Test for dynamical decoupling w/ optimization level=2 and 3"""
        self.assertTrue("PRIMITIVES_DEBUG" in environ)
        self.assertEqual(environ["PRIMITIVES_DEBUG"], "true")
        ghz4 = self._ghz4.copy()
        ghz4.measure([0, 1, 2, 3], [0, 1, 2, 3])
        result = mainSampler(
            backend=self._backend,
            user_messenger=None,
            circuits=[ghz4],
            circuit_indices=[0],
            observable_indices=[0],
            parameter_values=[[]],
            transpilation_settings={
                "optimization_settings": {"level": optimization_level},
                "seed_transpiler": 123,
            },
            resilience_settings={"level": 0},
        )

        ghz4 = self._ghz4.remove_final_measurements()
        transpiled_ghz4 = transpile(
            self._ghz4, self._backend, optimization_level=optimization_level, seed_transpiler=123
        )
        qubits = [3, 5, 8, 9]

        transpiled_ghz4_dd = transpiled_ghz4.copy()
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(96), [qubits[1]], front=True)
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(1344), [qubits[2]], front=True)
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(2128), [qubits[3]], front=True)
        for i in set(range(27)) - set(qubits):
            transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(6576), [i], front=True)

        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(436), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(872), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(np.pi), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(-np.pi), [qubits[0]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(436), [qubits[0]])

        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(240), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(480), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(np.pi), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(XGate(), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(RZGate(-np.pi), [qubits[1]])
        transpiled_ghz4_dd = transpiled_ghz4_dd.compose(Delay(240), [qubits[1]])
        transpiled_ghz4_dd.measure(qubits, [0, 1, 2, 3])

        self.assertEqual(len(result["metadata"]), 1)
        self.assertEqual(len(result["metadata"][0]["bound_circuits"]), 1)
        self.assertEqual(result["metadata"][0]["bound_circuits"][0], transpiled_ghz4_dd)
