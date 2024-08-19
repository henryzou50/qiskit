from qiskit.transpiler import CouplingMap
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, \
                                     EnlargeWithAncilla
from qiskit import QuantumCircuit
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


def build_pass_manager(coupling_map, layout_pass, routing_pass):
    """ Build a pass manager for the given routing and layout passes. 
    
    Args:
        coupling_map (CouplingMap): The coupling map of the device.
        layout_pass (LayoutPass): The layout pass to use.
        routing_pass (RoutingPass): The routing pass to use. (can be None)
    
    Returns:
        PassManager: The pass manager to use.
    """
    pass_manager = PassManager([
        layout_pass,
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout()
    ])
    if routing_pass is not None:
        pass_manager.append(routing_pass)

    return pass_manager