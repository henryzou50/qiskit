from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, EnlargeWithAncilla, TrivialLayout
import time
import numpy as np

def build_pm(routing_pass, layout_pass, coupling_map, seed=42, lookahead=0):
    """
    Builds a pass manager with the given routing pass, layout pass, coupling map.
    
    Parameters:
    routing_pass (TransformationPass): Routing pass to use
    layout_pass (TransformationPAss): Layout pass to use
    coupling_map (CouplingMap): Coupling map to use
    seed (int): Seed to use for the layout pass and routing pass

    Returns:
    PassManager: A PassManager object with the given routing pass and layout pass
    """

    # constructs a pass manager, does not consider lookahead
    if lookahead == 0:
        layout = layout_pass(coupling_map, routing_pass=routing_pass(coupling_map, fake_run=True, seed=seed), seed=seed)
        return PassManager([
            layout,
            FullAncillaAllocation(coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            routing_pass(coupling_map,seed=seed)
        ])
    else:
        layout = layout_pass(coupling_map, routing_pass=routing_pass(coupling_map, fake_run=True, seed=seed, lookahead_depth=lookahead), seed=seed)
        return PassManager([
            layout,
            FullAncillaAllocation(coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            routing_pass(coupling_map,seed=seed, lookahead_depth=lookahead)
        ])
    
    
def build_pm_fast(routing_pass, layout_pass, coupling_map, seed=42, lookahead=0):
    """
    Builds a pass manager with the given routing pass, layout pass, coupling map.
    
    Parameters:
    routing_pass (TransformationPass): Routing pass to use
    layout_pass (TransformationPAss): Layout pass to use
    coupling_map (CouplingMap): Coupling map to use
    seed (int): Seed to use for the layout pass and routing pass

    Returns:
    PassManager: A PassManager object with the given routing pass and layout pass
    """

    # constructs a pass manager, does not consider lookahead
    if lookahead == 0:
        layout = layout_pass(coupling_map, routing_pass=SabreSwap(coupling_map, fake_run=True, seed=seed), seed=seed)
        return PassManager([
            layout,
            FullAncillaAllocation(coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            routing_pass(coupling_map,seed=seed)
        ])
    else:
        layout = layout_pass(coupling_map, routing_pass=SabreSwap(coupling_map, fake_run=True, seed=seed), seed=seed)
        return PassManager([
            layout,
            FullAncillaAllocation(coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            routing_pass(coupling_map,seed=seed, lookahead_depth=lookahead)
        ])

def generate_pass_managers(num_shots, routing_pass, layout_pass, coupling_map, initial_seed=42, lookahead=0):
    """
    Generates a list of PassManager objects with different seeds.
    
    Parameters:
    num_shots (int): The number of PassManager instances to create.
    routing_pass (callable): Routing pass class to use
    layout_pass (callable): Layout pass class to use
    coupling_map (CouplingMap): Coupling map to use
    init_seed (int): Initial seed to use for the PassManager objects
    
    Returns:
    list: A list of PassManager objects with incrementing seeds starting from 42.
    """
    pass_managers = []
    
    for i in range(num_shots):
        seed = initial_seed + i
        pm = build_pm(routing_pass, layout_pass, coupling_map, seed, lookahead)
        pass_managers.append(pm)
    
    return pass_managers

def generate_pass_managers_fast(num_shots, routing_pass, layout_pass, coupling_map, initial_seed=42, lookahead=0):
    """
    Generates a list of PassManager objects with different seeds.
    
    Parameters:
    num_shots (int): The number of PassManager instances to create.
    routing_pass (callable): Routing pass class to use
    layout_pass (callable): Layout pass class to use
    coupling_map (CouplingMap): Coupling map to use
    init_seed (int): Initial seed to use for the PassManager objects
    
    Returns:
    list: A list of PassManager objects with incrementing seeds starting from 42.
    """
    pass_managers = []
    
    for i in range(num_shots):
        seed = initial_seed + i
        pm = build_pm_fast(routing_pass, layout_pass, coupling_map, seed, lookahead)
        pass_managers.append(pm)
    
    return pass_managers


def transpiled_data(qc, pass_managers):
    """
    Returns a dictionary of transpiled data for the best transpiled circuit 
    (with the lowest depth) after transpiling it with each pass manager in the list.

    Parameters:
    qc (QuantumCircuit): Quantum circuit to transpile
    pass_managers (list): List of PassManager objects to use for transpilation

    Returns:
    dict: A dictionary containing the best transpiled data and standard deviations of the transpiled data
    """
    
    best_data = {
        'depth': float('inf'),
        'depth_ratio': 0,
        'time': 0,
        'cx_gates': 0,
        'num_gates': 0
    }
    qc = qc.decompose()
    depth_orig = qc.depth()
    
    depths = []
    depth_ratios = []
    times = []
    cx_gates = []
    num_gates_list = []

    for pm in pass_managers:
        time_start = time.time()
        qc_tr = pm.run(qc)
        time_end = time.time()
        time_elapsed = time_end - time_start

        qc_tr = qc_tr.decompose(["swap"])
        depth = qc_tr.depth()
        depth_ratio = round((depth / depth_orig), 2)

        ops = qc_tr.count_ops()
        
        # Add data to lists
        depths.append(depth)
        depth_ratios.append(depth_ratio)
        times.append(time_elapsed)
        cx_gates.append(ops.get('cx', 0))
        num_gates_list.append(sum(ops.values()))

        # Check if this transpiled circuit has lower depth than the best found so far
        if depth < best_data['depth']:
            best_data = {
                'depth': depth,
                'depth_ratio': depth_ratio,
                'time': time_elapsed,
                'cx_gates': ops.get('cx', 0),
                'num_gates': sum(ops.values())
            }

    result = {
        'best_data': best_data,
        'std_dev': {
            'depth': round_to_significant_figures(np.std(depths)),
            'depth_ratio': round_to_significant_figures(np.std(depth_ratios)),
            'time': round_to_significant_figures(np.std(times)),
            'cx_gates': round_to_significant_figures(np.std(cx_gates)),
            'num_gates': round_to_significant_figures(np.std(num_gates_list))
        }
    }

    return result

def round_to_significant_figures(num, n=4):
    """ 
    Round a number to n significant figures.
    
    Parameters:
    num (float): Number to round
    
    Returns:
    float: Rounded number"""
    if num == 0:
        return 0
    else:
        return round(num, -int(np.floor(np.log10(abs(num))) - (n - 1)))