# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for swap strategies."""

from qiskit.test import QiskitTestCase
from qiskit.transpiler import CouplingMap
from qiskit.providers.aer import AerSimulator
from qiskit.exceptions import QiskitError

from programs.qaoa import (
    SwapStrategy,
    LineSwapStrategy,
    get_swap_strategy,
    swap_pass_manager_creator,
)


class TestSwapStrategy(QiskitTestCase):
    """A class to test the swap strategies."""

    def setUp(self):
        """Setup the tests."""
        super().setUp()
        self.line_coupling_map = CouplingMap(
            couplinglist=[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (1, 0),
                (2, 1),
                (3, 2),
                (4, 3),
            ]
        )

        self.line_swap_layers = [
            [(0, 1), (2, 3)],
            [(1, 2), (3, 4)],
            [(0, 1), (2, 3)],
            [(1, 2), (3, 4)],
            [(0, 1), (2, 3)],
        ]

        self.line_edge_coloring = {(0, 1): 0, (1, 2): 1, (2, 3): 0, (3, 4): 1}
        self.line_strategy = SwapStrategy(
            coupling_map=self.line_coupling_map,
            swap_layers=self.line_swap_layers,
            edge_coloring=self.line_edge_coloring,
        )

    def test_invert_permutation(self):
        """Test the invert permutation of the swap strat."""
        permutation = [0, 2, 3, 4, 1]
        inverse_permutation = [0, 4, 1, 2, 3]
        self.assertEqual(SwapStrategy.invert_permutation(permutation), inverse_permutation)
        self.assertEqual(SwapStrategy.invert_permutation(inverse_permutation), permutation)
        with self.assertRaises(ValueError):
            SwapStrategy.invert_permutation([0, 1, 42])
        with self.assertRaises(ValueError):
            SwapStrategy.invert_permutation([0, 1, "test"])

    def test_composed_permutation(self):
        """Test the permutation at several layers."""
        self.assertEqual(self.line_strategy.composed_permutation(0), [0, 1, 2, 3, 4])
        self.assertEqual(self.line_strategy.composed_permutation(1), [1, 0, 3, 2, 4])
        self.assertEqual(self.line_strategy.composed_permutation(2), [2, 0, 4, 1, 3])
        self.assertEqual(self.line_strategy.composed_permutation(3), [3, 1, 4, 0, 2])
        self.assertEqual(self.line_strategy.composed_permutation(4), [4, 2, 3, 0, 1])
        self.assertEqual(self.line_strategy.composed_permutation(5), [4, 3, 2, 1, 0])

    def test_inverse_composed_permutation(self):
        """Test the inverse permutation at several swap layers."""
        self.assertEqual(self.line_strategy.inverse_composed_permutation(0), [0, 1, 2, 3, 4])
        self.assertEqual(self.line_strategy.inverse_composed_permutation(1), [1, 0, 3, 2, 4])
        self.assertEqual(self.line_strategy.inverse_composed_permutation(2), [1, 3, 0, 4, 2])
        self.assertEqual(self.line_strategy.inverse_composed_permutation(3), [3, 1, 4, 0, 2])
        self.assertEqual(self.line_strategy.inverse_composed_permutation(4), [3, 4, 1, 2, 0])
        self.assertEqual(self.line_strategy.inverse_composed_permutation(5), [4, 3, 2, 1, 0])

    def test_apply_swap_layer(self):
        """Test the swapping on a list."""
        list_to_swap = [0, 10, 20, 30, 40]

        self.assertEqual(
            self.line_strategy.apply_swap_layer(list_to_swap, 0),
            [10, 0, 30, 20, 40],
        )

        self.assertEqual(
            self.line_strategy.apply_swap_layer(list_to_swap, 1),
            [0, 20, 10, 40, 30],
        )

    def test_length(self):
        """Test the __len__ operator."""
        self.assertEqual(len(self.line_strategy), 5)

    def test_swapped_coupling_map(self):
        """Test that edge set is properly swapped."""
        edge_set = {(2, 0), (0, 4), (4, 1), (1, 3), (3, 1), (1, 4), (4, 0), (0, 2)}
        swapped_map = self.line_strategy.swapped_coupling_map(3)
        self.assertEqual(edge_set, set(swapped_map.get_edges()))

    def test_check_configuration(self):
        """Test the configuration."""
        strategy = SwapStrategy(
            coupling_map=self.line_coupling_map,
            swap_layers=[[(0, 1), (1, 2)], [(1, 3), (2, 4)]],
        )
        with self.assertRaises(RuntimeError):
            distance_matrix = strategy.distance_matrix  # pylint: disable=unused-variable

    def test_raises_embed_in_smaller_graph(self):
        """Test embedding."""
        small_line = CouplingMap(couplinglist=[(0, 1), (1, 0), (1, 2), (2, 1)])
        with self.assertRaises(RuntimeError):
            self.line_strategy.embed_in(coupling_map=small_line)

    def test_distance_matrix(self):
        """Test distance matrix."""
        line_distance_matrix = [
            [0, 0, 3, 1, 2],
            [0, 0, 0, 2, 3],
            [3, 0, 0, 0, 1],
            [1, 2, 0, 0, 0],
            [2, 3, 1, 0, 0],
        ]
        self.assertEqual(line_distance_matrix, self.line_strategy.distance_matrix)

    def test_reaches_full_connectivity(self):
        """Test that we reach full connectivity on the longest line of Mumbai."""

        # The longest line on e.g. Mumbai has the qubits
        ll27 = [
            0,
            1,
            2,
            3,
            5,
            8,
            11,
            14,
            16,
            19,
            22,
            25,
            24,
            23,
            21,
            18,
            15,
            12,
            10,
            7,
            6,
        ]

        ll27_map = [[ll27[idx], ll27[idx + 1]] for idx in range(len(ll27) - 1)]
        ll27_map += [[ll27[idx + 1], ll27[idx]] for idx in range(len(ll27) - 1)]

        # Create a line swap strategy on this line
        layer1 = [(ll27[idx], ll27[idx + 1]) for idx in range(0, len(ll27) - 1, 2)]
        layer2 = [(ll27[idx], ll27[idx + 1]) for idx in range(1, len(ll27), 2)]

        num = len(ll27)
        for n_layers, result in [
            (num - 4, False),
            (num - 3, False),
            (num - 2, True),
            (num - 1, True),
        ]:
            swap_strat_ll = []
            for idx in range(n_layers):
                if idx % 2 == 0:
                    swap_strat_ll.append(layer1)
                else:
                    swap_strat_ll.append(layer2)

            strat = SwapStrategy(CouplingMap(ll27_map), swap_strat_ll)
            self.assertEqual(strat.reaches_full_connectivity(), result)


class TestLineSwapStrategy(QiskitTestCase):
    """A class to test the line swap strategy."""

    def test_full_line(self):
        """Test that we reach full connectivity on a line."""

        n_nodes = 5
        strategy = LineSwapStrategy(list(range(n_nodes)))

        self.assertEqual(len(strategy.swap_layers), n_nodes - 2)

        # The LineSwapStrategy will apply the following permutations
        layers = [
            [0, 1, 2, 3, 4],  # coupling map
            [1, 0, 3, 2, 4],  # layer 1
            [1, 3, 0, 4, 2],  # layer 2
            [3, 1, 4, 0, 2],  # layer 3 <-- full connectivity is reached.
        ]

        for layer_idx, layer in enumerate(layers):
            expected = set()
            for idx in range(len(layer) - 1):
                expected.add((layer[idx], layer[idx + 1]))
                expected.add((layer[idx + 1], layer[idx]))

            strat_edges = strategy.swapped_coupling_map(layer_idx).get_edges()
            self.assertEqual(len(strat_edges), len(expected))
            for edge in strat_edges:
                self.assertTrue(edge in expected)

        self.assertEqual(strategy.swap_layers[0], [(0, 1), (2, 3)])
        self.assertEqual(strategy.swap_layers[1], [(1, 2), (3, 4)])
        self.assertEqual(strategy.swap_layers[2], [(0, 1), (2, 3)])

        self.assertTrue(strategy.reaches_full_connectivity())

    def test_line(self):
        """Test the creation of a line swap strategy."""

        n_nodes = 5
        strategy = LineSwapStrategy(list(range(n_nodes)))

        self.assertEqual(strategy.swap_layers[0], [(0, 1), (2, 3)])
        self.assertEqual(strategy.swap_layers[1], [(1, 2), (3, 4)])
        self.assertEqual(strategy.swap_layers[2], [(0, 1), (2, 3)])

        self.assertTrue(strategy.reaches_full_connectivity())


class TestSpecializations(QiskitTestCase):
    """Test the special swap strategies."""

    def test_five_qubits_tee(self):
        """Test the swap strats for five qubit devices."""

        strat = get_swap_strategy("ibmq_belem")[0]

        self.assertTrue(strat.reaches_full_connectivity())

    def test_seven_qubits(self):
        """Test the swap strategy for devices like lagos."""

        strat = get_swap_strategy("ibm_lagos", 7)[0]

        self.assertTrue(strat.reaches_full_connectivity())

        strat = get_swap_strategy("ibm_lagos", 6)[0]

        self.assertTrue(strat.reaches_full_connectivity())

    def test_double_ring(self):
        """Test that the double ring swap strategies reach full connectivity."""

        for n_qubits in [22, 23, 24, 25, 26, 27]:
            strat = get_swap_strategy("ibmq_mumbai", n_qubits)[0]

            self.assertTrue(strat.reaches_full_connectivity())


class TestSwapStrategyCreator(QiskitTestCase):
    """Test the swap strategy creator."""

    def test_coupling_map_error(self):
        """Test that an error is raised of the backend does not have a coupling map."""

        with self.assertRaises(QiskitError):
            swap_pass_manager_creator(AerSimulator())
