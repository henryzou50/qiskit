import os
from os import listdir
from os.path import isfile, join
from qiskit import QuantumCircuit


def get_circuits_from_directory(directory):
    """ Return a list of QuantumCircuits from the files in the directory. 
    The files are sorted by the number of lines in the file.
     
    Args: 
        directory (str): The path to the directory containing the files.

    Returns:
        list[QuantumCircuit]: The QuantumCircuits from the files in the directory.
    """

    # Check if the directory exists, if not raise an error
    if not os.path.exists(directory):
        raise ValueError(f'The directory {directory} does not exist')

    # Get all of the files in the directory
    files = [f for f in listdir(directory) if isfile(join(directory, f))]

    # Sort the files by the number of lines in the file
    files.sort(key=lambda f: sum(1 for line in open(join(directory, f))))

    # Read the files and return the QuantumCircuits
    circuits = []
    counter = 2
    for file in files:
        with open(join(directory, file), 'r') as f:
            # print the file name
            print(f"{counter}. {file}")
            counter += 1

            circuits.append(QuantumCircuit.from_qasm_str(f.read()))

    return circuits
