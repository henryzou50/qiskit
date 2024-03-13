""" Contains the functions for the coupling maps used in the experiments. """

from qiskit.transpiler import CouplingMap
import ast

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