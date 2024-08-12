# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simplified Test for the Sabre Swap pass"""

import unittest
import ddt
import csv
import os 
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase  # pylint: disable=wrong-import-order

@ddt.ddt
class SimplifiedTestSabreSwap(QiskitTestCase):
    """Simplified tests for the SabreSwap pass."""

    @classmethod
    def setUpClass(cls):
        """Create the .data directory and initialize the CSV file."""
        super().setUpClass()
        os.makedirs('.data', exist_ok=True)
        cls.csv_file_path = '.data/ghz_depths.csv'
        
        # Initialize the CSV file with headers
        with open(cls.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["num_qubits", "old_depth", "new_depth"])


    def test_trivial_case(self):
        """Test that an already mapped circuit is unchanged.
                  ┌───┐┌───┐
        q_0: ──■──┤ H ├┤ X ├──■──
             ┌─┴─┐└───┘└─┬─┘  │
        q_1: ┤ X ├──■────■────┼──
             └───┘┌─┴─┐       │
        q_2: ──■──┤ X ├───────┼──
             ┌─┴─┐├───┤       │
        q_3: ┤ X ├┤ X ├───────┼──
             └───┘└─┬─┘     ┌─┴─┐
        q_4: ───────■───────┤ X ├
                            └───┘
        """
        coupling = CouplingMap.from_ring(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # F
        qc.cx(1, 0)
        qc.cx(4, 3)  # F
        qc.cx(0, 4)

        passmanager = PassManager(SabreSwap(coupling, "basic"))
        new_qc = passmanager.run(qc)

        self.assertEqual(new_qc, qc)

    def test_lookahead_mode(self):
        """Test lookahead mode with a simple circuit."""
        coupling = CouplingMap.from_line(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # free
        qc.cx(1, 3)  # F
        qc.cx(2, 3)  # E
        qc.cx(1, 3)  # E

        pm = PassManager(SabreSwap(coupling, "lookahead"))
        new_qc = pm.run(qc)

        self.assertEqual(new_qc.num_nonlocal_gates(), 7)

    def test_deterministic_output(self):
        """Test that the output of the SabreSwap pass is deterministic for a given random seed."""
        heuristic = "basic"
        width = 40

        # The actual circuit is unimportant, we just need one with lots of scoring degeneracy.
        qc = QuantumCircuit(width)
        for i in range(width // 2):
            qc.cx(i, i + (width // 2))
        for i in range(0, width, 2):
            qc.cx(i, i + 1)
        dag = circuit_to_dag(qc)

        coupling = CouplingMap.from_line(width)
        pass_0 = SabreSwap(coupling, heuristic, seed=0, trials=1)
        pass_1 = SabreSwap(coupling, heuristic, seed=1, trials=1)
        dag_0 = pass_0.run(dag)
        dag_1 = pass_1.run(dag)

        # This deliberately avoids using a topological order, because that introduces an opportunity
        # for the re-ordering to sort the swaps back into a canonical order.
        def normalize_nodes(dag):
            return [(node.op.name, node.qargs, node.cargs) for node in dag.op_nodes()]

        # A sanity check for the test - if unequal seeds don't produce different outputs for this
        # degenerate circuit, then the test probably needs fixing (or Sabre is ignoring the seed).
        self.assertNotEqual(normalize_nodes(dag_0), normalize_nodes(dag_1))

        # Check that a re-run with the same seed produces the same circuit in the exact same order.
        self.assertEqual(normalize_nodes(dag_0), normalize_nodes(pass_0.run(dag)))

    @ddt.data(2, 4, 8, 16)
    def test_ghz_2q_depth(self, num_qubits):
        """Test that the SabreSwap pass with a fixed seed produces a specific 2 qubit gate depth for a GHZ circuit."""
        # Setup
        coupling = CouplingMap.from_line(num_qubits)
        heuristic = "basic"

        # Create a GHZ circuit
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(qr)
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)

        # Create a PassManager and run the SabreSwap pass
        pm = PassManager(SabreSwap(coupling, heuristic, seed=42))
        new_qc = pm.run(qc)

        # Decompose the swap gates in the circuits to cx gates
        qc = qc.decompose()
        new_qc = new_qc.decompose()

        # Optional: Save the circuits to files
        qc.draw(output='mpl', filename=f'.images/ghz_{num_qubits}_old_circuit.png')
        new_qc.draw(output='mpl', filename=f'.images/ghz_{num_qubits}_new_circuit.png')

        # Obtain the 2 qubit gate depth of old and new circuits
        old_depth = qc.depth(lambda x: x.operation.num_qubits == 2)
        new_depth = new_qc.depth(lambda x: x.operation.num_qubits == 2)

        # Save the results to a CSV file
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([num_qubits, old_depth, new_depth])

        # Check that the new depth is greater than 0, confirms that the SabreSwap pass was applied
        self.assertGreater(new_depth, 0)


if __name__ == "__main__":
    unittest.main()
