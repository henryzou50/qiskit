""" This files contains the ulti functions to set up the experiments. """

from qiskit import QuantumCircuit 
from qiskit.transpiler import CouplingMap
import os
import ast

def file_to_circuit_list(directory):
    """ Get a list of circuits from the given directory.

    Args:
        directory (str): Path to the directory containing the QASM files.
    Returns:
        list: A list of QuantumCircuit objects.
    """
    qc_list = []
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        qc = QuantumCircuit.from_qasm_file(file_path)
        qc_list.append(qc)
        qc_list.sort(key=lambda x: x.num_qubits)
    return qc_list


def file_to_coupling_map(file_path):
    """ Converts a file to a coupling map.

    Args: 
        file_path (str): The path to the file to convert.
    Returns:
        CouplingMap: The coupling map from the file.  
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()
        return CouplingMap(ast.literal_eval(content))