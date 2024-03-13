""" This file will build and store the circuits for the experiments. 
The types of circuits are:
    - GHZ
    - QVOL
"""

from qiskit.qasm2 import dump
from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from circuits import directory_to_circuits
import random 
import os


def build_ghz_circuit(num_qubits):
    """ Build the GHZ circuit with num_qubits. 
    
    Args: 
        num_qubits (int): number of qubits in the circuit.
    Returns:
        QuantumCircuit: the GHZ circuit.
    """
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    return qc

def build_ghz_circuits(start=2, end=127):
    """ Build the GHZ circuits with num_qubits from start to end. 
    
    Args:
        start (int): the smallest number of qubits.
        end (int): the largest number of qubits.
    Returns:
        list: a list of GHZ circuits.
    """
    circuits = []
    for i in range(start, end+1):
        circuits.append(build_ghz_circuit(i))
    return circuits

def build_qvol_circuits(num_qubits, num_circuits, seeds):
    """ Build the Quantum Volume circuits with num_qubits, 
    each circuit will have different depth (from 1 to num_circuits),
    each circuit will have different seed.
    
    Args:
        num_qubits (int): number of qubits in the circuit.
        num_circuits (int): number of circuits to build.
        seeds (list): a list of seeds for the circuits.
    Returns:
        list: a list of Quantum Volume circuits."""
    
    circuits = []

    for i in range(num_circuits):
        # each circuit will have different depth
        qv = QuantumVolume(num_qubits, i+1, seed=seeds[i])
        circuits.append(qv)
    return circuits

def build_decomposed_circuits(circuits, num_times):
    """ Decompose the circuits num_times. Used to prepare the circuits like QVOL and red 
    queen for the experiments. 
    
    Args:
        circuits (list): a list of QuantumCircuits.
        num_times (int): the number of times to decompose the circuits.
    Returns:
        list: a list of decomposed circuits.
    """
    decomposed_circuits = []
    for circuit in circuits:
        for _ in range(num_times):
            circuit = circuit.decompose()
        decomposed_circuits.append(circuit)
    return decomposed_circuits


# Run the file to build and store the circuits
if __name__ == "__main__":
    directory = "sabre_mods/experiments/circuits/"

    # set the flags to run the circuits
    run_ghz  = False
    run_qvol = False
    run_qvol_decomposed_1 = False
    run_qvol_decomposed_2 = True

    if run_ghz:
        # build GHZ circuits of 2 to 127 qubits
        start = 2
        end   = 127
        print("Building GHZ circuits from", start, "to", end, "qubits.")

        ghz_circuits = build_ghz_circuits(start, end)
        circuit_type = directory + "ghz/"
        # check if the directory exists, if not create it
        if not os.path.exists(circuit_type):
            os.makedirs(circuit_type)

        # create a list of file names for the circuits, based on the number of qubits in the circuits
        file_names = [circuit_type + "ghz_" + str(i) + ".qasm" for i in range(start, end+1)]
        # store the circuits
        for i in range(len(ghz_circuits)):
            dump(ghz_circuits[i], file_names[i])
            print("     GHZ circuit with", i+2, "qubits is stored.")
        print("All GHZ circuits are stored.\n")
        
    if run_qvol:
        # build QVOL circuits of 10 qubits and 100 circuits
        num_qubits   = 10
        num_circuits = 100
        seed         = 42  
        random.seed(seed)
        # generate random seeds for reproducibility
        seeds = random.sample(range(1, 1000), num_circuits)

        print("Building", num_circuits, "QVOL circuits with", num_qubits, "qubits.")

        qvol_circuits = build_qvol_circuits(num_qubits, num_circuits, seeds)
        circuit_type  = directory + "qvol/"
        # check if the directory exists, if not create it
        if not os.path.exists(circuit_type):
            os.makedirs(circuit_type)
        
        # create a list of file names for the circuits, based on the num_qubits, depth, and seed where seed is the form of 0001 to 9999
        file_names = [circuit_type + "qvol_" + str(num_qubits) + "_" + str(i+1) + "_" + str(seeds[i]).zfill(4) + ".qasm" for i in range(num_circuits)]
        # store the circuits
        for i in range(len(qvol_circuits)):
            dump(qvol_circuits[i], file_names[i])
        print("     QVOL circuit id:", i+1, "is stored.")
        print("All QVOL circuits are stored.\n")

    if run_qvol_decomposed_1:
        print("Decomposing the QVOL circuits 1 time.")

        # build the QVOL circuits from the directory from the previous step
        prev_directory = "sabre_mods/experiments/circuits/qvol/"
        circuits, files = directory_to_circuits(prev_directory)

        new_directory = "sabre_mods/experiments/circuits/qvol_d1/"
        # check if the directory exists, if not create it
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        
        # decompose the circuits 1
        decomposed_circuits = build_decomposed_circuits(circuits, 1)
        # store the circuits
        for i in range(len(decomposed_circuits)):
            dump(decomposed_circuits[i], new_directory + files[i])
            print("     Decomposed QVOL circuit id:", i+1, "is stored.")

    if run_qvol_decomposed_2:
        print("Decomposing the QVOL circuits 2 times.")

        # build the QVOL circuits from the directory from the original step
        prev_directory = "sabre_mods/experiments/circuits/qvol/"
        circuits, files = directory_to_circuits(prev_directory)

        new_directory = "sabre_mods/experiments/circuits/qvol_d2/"
        # check if the directory exists, if not create it
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        
        # decompose the circuits 2
        decomposed_circuits = build_decomposed_circuits(circuits, 2)
        # store the circuits
        for i in range(len(decomposed_circuits)):
            dump(decomposed_circuits[i], new_directory + files[i])
            print("     Decomposed QVOL circuit id:", i+1, "is stored.")

    print("All circuits are stored.")
