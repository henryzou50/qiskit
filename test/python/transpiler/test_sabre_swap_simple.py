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
import csv
import os 
import ddt
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap
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

    @ddt.data(27)
    def test_ghz_2q_depth(self, num_qubits):
        """Test that the SabreSwap pass with a fixed seed produces a specific 2 qubit gate depth for a GHZ circuit."""
        # Setup
        coupling_list = [[6, 7],
 [7, 6],
 [12, 13],
 [13, 12],
 [7, 10],
 [10, 7],
 [8, 11],
 [11, 8],
 [3, 2],
 [2, 3],
 [4, 7],
 [7, 4],
 [9, 8],
 [8, 9],
 [17, 18],
 [18, 17],
 [1, 0],
 [0, 1],
 [15, 12],
 [12, 15],
 [1, 4],
 [4, 1],
 [20, 19],
 [19, 20],
 [26, 25],
 [25, 26],
 [21, 23],
 [23, 21],
 [21, 18],
 [18, 21],
 [16, 14],
 [14, 16],
 [13, 14],
 [14, 13],
 [24, 25],
 [25, 24],
 [11, 14],
 [14, 11],
 [24, 23],
 [23, 24],
 [10, 12],
 [12, 10],
 [22, 19],
 [19, 22],
 [5, 8],
 [8, 5],
 [22, 25],
 [25, 22],
 [3, 5],
 [5, 3],
 [16, 19],
 [19, 16],
 [18, 15],
 [15, 18],
 [1, 2],
 [2, 1]]
        coupling = CouplingMap(coupling_list)
        heuristic = "decay"

        # Create a GHZ circuit
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(qr)

        for i in range(1, num_qubits):
            qc.cx(0, i)

        # Create a PassManager and run the SabreSwap pass
        pm = PassManager(SabreSwap(coupling, heuristic, seed=42, trials=10))
        new_qc = pm.run(qc)

        # Decompose the swap gates in the circuits to cx gates
        qc = qc.decompose()
        new_qc = new_qc.decompose()

        # Optional: Save the circuits to files
        #qc.draw(output='mpl', filename=f'.images/ghz_{num_qubits}_old_circuit.png')
        #new_qc.draw(output='mpl', filename=f'.images/ghz_{num_qubits}_new_circuit.png')

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