""" Contains the functions for the circuits used in the experiments. """

import os
from os import listdir
from os.path import isfile, join
from qiskit import QuantumCircuit

def directory_to_circuits(directory):
    """ Return a list of QuantumCircuits from the files in the directory, along with 
    the list of file names. The files are sorted by the number of lines in the file. 
     
    Args: 
        directory (str): The path to the directory containing the files.

    Returns:
        (list, list): a list of QuantumCircuits and a list of file names.
    """

    # Check if the directory exists, if not raise an error
    if not os.path.exists(directory):
        raise ValueError(f'The directory {directory} does not exist')

    # Get all of the files in the directory
    files = [f for f in listdir(directory) if isfile(join(directory, f))]

    # Sort the files by the number of lines in the file
    files.sort(key=lambda f: sum(1 for line in open(join(directory, f))))

    # Read the files and store the circuits
    circuits = []
    for file in files:
        with open(join(directory, file), 'r') as f:
            circuits.append(QuantumCircuit.from_qasm_str(f.read()))

    return circuits, files

def get_circuits_depth(circuits):
    """ Return the list of the depths of the circuits. In our experiments, 
    we will record the depth as the 2-qubit gate depth of the circuits. 
    Note that we are not decomposing the circuits, and if we want to get the 
    decomposed depth, we need to decompose the circuits first.

    Args:
        circuits (list): a list of QuantumCircuits.

    Returns:
        list: a list of the depths of the circuits.
    """

    return [circuit.depth(lambda x: x.operation.num_qubits == 2) for circuit in circuits]