from qiskit.circuit.library import QuantumVolume
from qiskit.qasm2 import dump
import os

def generate_qv_circuits(num_circuits, depth, num_qubits=10, directory="circuits/qvol"):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    for i in range(num_circuits):
        # Generate a Quantum Volume circuit with a different seed each time
        qvol = QuantumVolume(num_qubits, depth, seed=i)
        qc = qvol.decompose()
        
        # Define the QASM file path
        filename = f"qv_circuit_{num_qubits}q_{depth}d_{i}.qasm"
        filepath = os.path.join(directory, filename)
        
        # Save the circuit to a QASM file
        dump(qc, filepath)
        
        print(f"Circuit {i+1} saved to {filepath}")
