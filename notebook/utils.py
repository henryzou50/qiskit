from qiskit.transpiler import CouplingMap
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, QuantumVolume
import random
import ast
import os
from os import listdir
from os.path import isfile, join
from qiskit.qasm2 import dump

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
    

def directory_to_circuits(directory):
    """ Return a list of QuantumCircuits from the files in the directory, along with 
    the list of file names. The files are sorted by the number of lines in the file. 
     
    Args: 
        directory (str): The path to the directory containing the files.
    Returns:
        circuits (list): a list of QuantumCircuits and a list of file names.
    """

    # Check if the directory exists, if not raise an error
    if not os.path.exists(directory):
        raise ValueError(f'The directory {directory} does not exist')

    # Get all of the files in the directory
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    # Sort the files by the number of lines in the file
    files.sort(key=lambda f: sum(1 for line in open(join(directory, f), 'r', errors='ignore')))
    # Remove any files that are not .qasm files
    files = [f for f in files if f.endswith('.qasm')]

    # Read the files and store the circuits
    circuits = []
    for file in files:
        with open(join(directory, file), 'r') as f:
            circuits.append(QuantumCircuit.from_qasm_str(f.read()))

    # Rename the circuits to the file names
    for i in range(len(circuits)):
        circuits[i].name = files[i]

    return circuits


def create_and_save_ghz_circuits(num_qubits_list, save_dir="ghz"):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for i in num_qubits_list:
        # Create a GHZ circuit with i qubits
        qc = QuantumCircuit(i)
        qc.h(0)
        for qubit in range(1, i):
            qc.cx(qubit - 1, qubit)
        
        # Define the filename for saving
        filename = f"ghz_{i}.qasm"
        filepath = os.path.join(save_dir, filename)
        
        # Save the circuit as a QASM file
        with open(filepath, "w") as f:
            dump(qc, f)
    
    print(f"GHZ circuits saved to {save_dir}")

def create_and_save_qft_circuits(num_qubits_list, save_dir="qft"):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for i in num_qubits_list:
        # Create a QFT circuit with i qubits
        qc = QFT(i).decompose()
        
        # Define the filename for saving
        filename = f"qft_{i}.qasm"
        filepath = os.path.join(save_dir, filename)
        
        # Save the circuit as a QASM file
        with open(filepath, "w") as f:
            dump(qc, f)
    
    print(f"QFT circuits saved to {save_dir}")

def create_and_save_qv_circuits(num_qubits_list, depth_list, seed=42, save_dir="qv"):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Obtain a seed list from the seed
    random.seed(seed)
    seed_list = random.sample(range(1000), len(num_qubits_list) * len(depth_list))
    
    for n in num_qubits_list:
        for d in depth_list:
            curr_seed = seed_list.pop(0)
            # Create a Quantum Volume circuit with n qubits and depth d
            qc = QuantumVolume(n, d, seed=curr_seed)
            qc = qc.decompose()
            
            # Define the filename for saving
            filename = f"qv_{n}_{d}_{curr_seed}.qasm"
            filepath = os.path.join(save_dir, filename)
            
            # Save the circuit as a QASM file
            with open(filepath, "w") as f:
                dump(qc, f)
    
    print(f"Quantum Volume circuits saved to {save_dir}")


