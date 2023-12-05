''' This file includes functions to create the circuits used for the experiments. '''

from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
import os
import re


def get_circuit_list(directory):
    """ Get a list of circuits from the given directory.

    Args:
        directory (str): Path to the directory containing the QASM files.
    Returns:
        list: A list of QuantumCircuit objects.
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
    """ Generate a GHZ circuit of size n.

    Args:
        n (int): Size of the GHZ state.
    Returns:    
        QuantumCircuit: A GHZ circuit of size n.
    """   
    if n <= 0:
        raise ValueError("n should be a positive integer")

    qc = QuantumCircuit(n)
    qc.h(0)
    for qubit in range(1, n):
        qc.cx(0, qubit)

    return qc


def generate_and_store_ghz_circuits(start, end, directory="circuits/ghz"):
    """ Generate GHZ circuits from 'start' to 'end' qubits and save them as QASM files 
    in the specified directory.

    Args:
        start (int): Starting number of qubits.
        end (int): Ending number of qubits.
        directory (str): Directory to save the QASM files.
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
                                   depth_ranges=[(100, 501)],
                                   increments=[100], 
                                   base_path="circuits",
                                   seed=42):
    """ Generate Quantum Volume circuits for the given qubit sizes and depth ranges and save them
    as QASM files in the specified directory.

    Args:
        qubit_sizes (list): A list of qubit sizes.
        depth_ranges (list): A list of tuples representing the start and stop values for the depth
            of the circuits.
        increments (list): A list of integers representing the increment between the start and stop
            values for the depth of the circuits.
        base_path (str): Directory to save the QASM files.
        seed (int): Seed for the random number generator.
    Returns:    
        None
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
                
                print(f"Stored Quantum Volume circuit for size {size} and depth {depth} at \
                        {file_path}")


def sort_circuits_by_depth(qc_list):
    """ Sorts an array of Qiskit QuantumCircuits by their depth. The smallest depth circuit will 
    be the first in the list.

    Args:
        qc_list (list): A list of QuantumCircuit objects.   
    Returns:
        list: A list of QuantumCircuit objects sorted by their depth.
    """
    # Use the sorted function with a key that calls the depth method on each circuit
    sorted_qc_list = sorted(qc_list, key=lambda x: x.decompose().depth())
    return sorted_qc_list


def update_qasm_file(file_path):
    """ Updates the .qasm file by adding an underscore and a number to the unitary gate name.

    Args:   
        file_path (str): Path to the .qasm file to update.
    Returns:
        None
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    unitary_counter = 0
    updated_lines = []
    for line in lines:
        if line.startswith("gate unitary"):
            updated_line = re.sub(r"gate unitary[_\d]*", f"gate unitary_{unitary_counter}", line)
            unitary_counter += 1
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

    print(f"Updated .qasm file: {file_path}")