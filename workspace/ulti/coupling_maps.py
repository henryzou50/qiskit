''' This file includes functions that involve the coupling maps for the experiments '''

from qiskit.transpiler.coupling import CouplingMap
import ast

def file_to_coupling_map(file_path):
    """
    Converts a file to a coupling map.

    Parameters:
    file_path (str): Path to the file
    """
    with open(file_path, 'r') as f:
        content = f.read().strip()
        return CouplingMap(ast.literal_eval(content))