from qiskit import QuantumCircuit
import numpy as np
import random

random.seed(42)

def add_randomized_non_overlapping_layer(qc, coupling_list, gate_type='cx'):
    # Shuffle the coupling list to ensure random exploration
    shuffled_coupling_list = coupling_list.copy()
    random.shuffle(shuffled_coupling_list)
    
    selected_edges = []
    used_qubits = set()
    
    # Select non-overlapping edges from the shuffled list
    for edge in shuffled_coupling_list:
        if edge[0] not in used_qubits and edge[1] not in used_qubits:
            selected_edges.append(edge)
            used_qubits.update(edge)

    min_edges = len(selected_edges) // 2 if len(selected_edges) % 2 == 0 else (len(selected_edges) // 2) + 1
    num_edges_to_include = random.randint(min_edges, len(selected_edges))
    edges_to_include = random.sample(selected_edges, num_edges_to_include)
    
    # if gate_type is 'swap', then include only one edge
    if gate_type == 'swap':
        edges_to_include = [edges_to_include[0]]
    
    # Add CNOT or SWAP gates for each selected edge to the existing quantum circuit
    for edge in edges_to_include:
        if gate_type == 'cx':
            qc.cx(edge[0], edge[1])
        elif gate_type == 'swap':
            qc.swap(edge[0], edge[1])
        else:
            raise ValueError('Invalid gate type')
        
def create_parallel_circuit(num_qubits, num_layers, coupling_map, barrier=False):
    # Create a quantum circuit with the given number of qubits
    qc = QuantumCircuit(num_qubits)
    coupling_list = list(coupling_map.get_edges())
    
    # Add layers of randomized non-overlapping CNOT gates and SWAP gates
    for _ in range(num_layers):
        add_randomized_non_overlapping_layer(qc, coupling_list, gate_type='cx')
        if barrier:
            qc.barrier()
        add_randomized_non_overlapping_layer(qc, coupling_list, gate_type='swap')
        if barrier:
            qc.barrier()
    
    return qc

def apply_swaps_and_get_matching_circuit(qc):
    # Initialize a new quantum circuit with the same number of qubits
    new_qc = QuantumCircuit(qc.num_qubits)
    
    # Mapping of qubit indices to their positions after swaps
    qubit_map = {i: i for i in range(qc.num_qubits)}
    
    # Iterate through the circuit instructions in reverse order
    for instr, qargs, _ in reversed(qc.data):
        if instr.name == 'swap':
            # Update the qubit mapping based on the swap
            qubit_map[qargs[0]._index], qubit_map[qargs[1]._index] = qubit_map[qargs[1]._index], qubit_map[qargs[0]._index]
        elif instr.name == 'cx':
            # Apply the current qubit mapping to the CNOT gate and add it to the new circuit
            new_qc.cx(qubit_map[qargs[0]._index], qubit_map[qargs[1]._index])
    
    # Reverse the order of gates in the new circuit to reflect the original order
    new_qc = new_qc.reverse_ops()
    
    return new_qc


def scramble_qubits(circuit, seed=None):
    """
    Creates a new circuit with qubits scrambled according to a random permutation.
    The permutation can be made reproducible by specifying a seed value.

    Parameters:
    - circuit: The input QuantumCircuit object to be scrambled.
    - seed: An optional seed value for the random number generator for reproducibility.

    Returns:
    - A new QuantumCircuit object with qubits scrambled.
    """
    # Set the seed for numpy's random number generator only if it is not None
    if seed is not None:
        np.random.seed(seed)
    
    num_qubits = circuit.num_qubits
    # Generate a random permutation of qubit indices
    perm = np.random.permutation(num_qubits)
    
    # Create a new quantum circuit with the same number of qubits
    new_circuit = QuantumCircuit(num_qubits)
    
    # Mapping of original qubit indices to their new positions
    qubit_mapping = {original: permuted for original, permuted in enumerate(perm)}
    
    # Iterate through each instruction in the original circuit
    for instruction, qargs, cargs in circuit._data:
        # Correctly map the qubits according to the permutation
        new_qargs = [new_circuit.qubits[qubit_mapping[qarg._index]] for qarg in qargs]
        # Apply the same instruction to the new circuit with permuted qubits
        new_circuit.append(instruction, new_qargs, cargs)
    
    return new_circuit