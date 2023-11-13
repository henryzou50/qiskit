''' This file includes functions to create the circuits used for the experiments. '''

from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
import os

def get_circuit_list(directory):
    """
    Get a list of circuits from the given directory.

    Parameters:
    directory (list of a str): Directory where the circuits are stored
    """
    qc_list = [] 
    for path in directory:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            qc = QuantumCircuit.from_qasm_file(file_path)
            qc_list.append(qc)
            qc_list.sort(key=lambda x: x.num_qubits)
    return qc_list

def ghz_circuit(n): 
    """
    Generate a GHZ circuit of size n.

    Parameters:
    n (int): Size of the GHZ state
    """   
    if n <= 0:
        raise ValueError("n should be a positive integer")

    qc = QuantumCircuit(n)
    qc.h(0)
    for qubit in range(1, n):
        qc.cx(0, qubit)

    return qc

def generate_and_store_ghz_circuits(start, end, directory="circuits/ghz"):
    """
    Generate GHZ circuits from 'start' to 'end' qubits and save them as QASM files 
    in the specified directory.

    Parameters:
    start (int): Start size of the GHZ state
    end (int): End size of the GHZ state
    directory (str): Directory where the QASM files should be saved
    """

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    for n in range(start, end + 1):
        qc = ghz_circuit(n)
        qasm_str = qc.qasm()
    
        file_path = os.path.join(directory, f"ghz_{n}.qasm")
        with open(file_path, 'w') as file:
            file.write(qasm_str)
            
    print(f"QASM files for GHZ circuits saved in {directory}!")

def generate_and_store_qv_circuits(qubit_sizes=[5], 
                                   depth_ranges=[(1, 10)],
                                   increments=[1], 
                                   base_path="circuits",
                                   seed=42):
    """
    Generates Quantum Volume circuits with specified qubit sizes and depth ranges, 
    and stores them in the specified directory structure.

    Parameters:
    qubit_sizes (list of int): Sizes of the Quantum Volume circuits
    depth_ranges (list of tuples): Depth ranges for the Quantum Volume circuits
    increments (list of int): Increments for the depth ranges
    base_path (str): Base directory where the circuits should be stored
    seed (int): Seed for generating the Quantum Volume circuits
    """

    os.makedirs(base_path, exist_ok=True)
    
    for size in qubit_sizes:
        for (start, stop), increment in zip(depth_ranges, increments):
            # Create directory for the current qubit size and depth range
            depth_range_str = f"{start}-{stop}"
            dir_path = os.path.join(base_path, f"size{size}_depth_{depth_range_str}")
            os.makedirs(dir_path, exist_ok=True)
            
            # Generate and store circuits for each depth within the range
            for depth in range(start, stop, increment):
                qv_circuit = QuantumVolume(size, depth, seed=seed)  
                qasm_str = qv_circuit.qasm()
                file_path = os.path.join(dir_path, f"qv_size{size}_depth{depth}.qasm")
                
                with open(file_path, "w") as file:
                    file.write(qasm_str)
                
                print(f"Stored Quantum Volume circuit for size {size} and depth {depth} at {file_path}")

def sort_circuits_by_depth(qc_list):
    """
    Sorts an array of Qiskit QuantumCircuits by their depth.
    The smallest depth circuit will be the first in the list.
    
    Parameters:
    qc_list (list): A list of Qiskit QuantumCircuit objects.
    
    Returns:
    list: A new list with QuantumCircuit objects sorted by depth.
    """
    # Use the sorted function with a key that calls the depth method on each circuit
    sorted_qc_list = sorted(qc_list, key=lambda x: x.decompose().depth())
    return sorted_qc_list