""" This file contains the functions to create the pass managers used in the experiments. """

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, \
                                     EnlargeWithAncilla, TrivialLayout
from qiskit.transpiler.passes.routing.sabre_swap import SabreSwap
from qiskit.transpiler.passes.layout.sabre_layout import SabreLayout
import time
import numpy as np
import pandas as pd

def build_pm(routing_pass, layout_pass, coupling_map, heuristic="basic", seed=42, look=0, beam=1,
             triv_layout=False, fast_layout=False):
    """ Builds a pass manager with the given parameters. 
    
    Args: 
        routing_pass (str): The routing pass to use. 
        layout_pass (str): The layout pass to use. 
        coupling_map (CouplingMap): The coupling map to use. 
        heuristic (str): The heuristic to use (for regular Sabre). 
        seed (int): The seed to use. 
        look (int): The lookahead steps to use (for Sabre Look). 
        beam (int): The beam width to use (for Sabre Look, Sabre Dive).
    
    Return:
        PassManager: The pass manager with the five passes for transpilation.
    """


    routing_args = {'coupling_map': coupling_map, 'seed': seed}
    if look > 0:
        routing_args['look'] = look
    if beam > 1:
        routing_args['beam_width'] = beam
    if heuristic != "basic":
        routing_args['heuristic'] = heuristic

    if triv_layout:
        layout = TrivialLayout(coupling_map)
    elif fast_layout:
        return PassManager([
            SabreLayout(coupling_map, seed=seed),
            EnlargeWithAncilla(),
            ApplyLayout(),
            SabreSwap(**routing_args)
        ])
    else:
        layout_routing_pass = routing_pass(fake_run=True, **routing_args)
        layout = layout_pass(coupling_map, routing_pass=layout_routing_pass, 
                        seed=seed)

    return PassManager([
        layout,
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        routing_pass(**routing_args)
    ])

def build_pm_list(routing_pass, layout_pass, coupling_map, num_pm=4, heuristic="basic", seed=42, 
                  look=0, beam=1, triv_layout=False, fast_layout=False):
    """ Builds a list of pass managers with the given parameters, and where each pm has the 
    same parameters except for the seed.

    Args: 
        num_pm (int): The number of pass managers to build. 
        routing_pass (str): The routing pass to use. 
        layout_pass (str): The layout pass to use. 
        coupling_map (CouplingMap): The coupling map to use. 
        heuristic (str): The heuristic to use (for regular Sabre). 
        seed (int): The seed to use. 
        look (int): The lookahead steps to use (for Sabre Look). 
        beam (int): The beam width to use (for Sabre Look, Sabre Dive).
    """

    pm_list = []

    for i in range(num_pm):
        pm_list.append(build_pm(routing_pass, layout_pass, coupling_map, heuristic, 
                                seed+i, look, beam, triv_layout=triv_layout, fast_layout=fast_layout))
    return pm_list

def run_one_circuit(qc, pm_list):
    """ Runs the experiment for the given pass managers. and returns a dictionary of the 
    transpilation data for th best transpiled circuit (determined as th one with the lowest 
    depth) after transpiling it with each pass maanger in the list.
     
    In these experiments, the goal is to run the same type of pass managers (only different 
    in the seeds) and run the transpilation for the same circuit, and then return the best 
    transpiled circuit data, but also the standard deviation of each of the parameters.
    
    Args:
        qc (QuantumCircuit): The quantum circuit to transpile. 
        pm_list (list): The list of pass managers to use.
        
    Returns:
        dict: A dictionary with the best transpiled circuit data, and the standard deviation 
              of each of the parameters. 
    """
    best_data = {
        'depth': np.inf,
        'cx gates': np.inf,
        'time': np.inf,
    }
    qc_decomposed = qc.decompose()
    
    depths = []
    cx_gates = []
    times = []
    
    for pm in pm_list:
        # Timing the transpilation
        start = time.time()
        qc_tr = pm.run(qc_decomposed)
        end = time.time()

        # Decompose the swaps gate in the circuit
        qc_tr = qc_tr.decompose()

        # Obtaining data
        depth = qc_tr.depth()
        cx = qc_tr.count_ops()['cx']
        time_ = end - start

        depths.append(depth)
        cx_gates.append(cx)
        times.append(time_)

        if depth < best_data['depth']:
            best_data['depth'] = depth
            best_data['cx gates'] = cx
            best_data['time'] = time_

    best_data['depth_std'] = round_to_sig_figures(np.std(depths))
    best_data['cx_std']    = round_to_sig_figures(np.std(cx_gates))
    best_data['time_std']  = round_to_sig_figures(np.std(times))

    return best_data

def round_to_sig_figures(num, n=4):
    """ Rounds the given number to n significant figures. 
    
    Args:
        num (float): The number to round. 
        n (int): The number of significant figures to round to.
    
    Returns:
        float: The number rounded to n significant figures.
    """
    if num == 0:
        return 0
    return round(num, -int(np.floor(np.log10(abs(num))) - (n - 1)))

def run_experiment(filename, qc_list, routing_pass, layout_pass, coupling_map, num_pm=4, 
                   heuristic="basic", seed=42, look=0, beam=1, triv_layout=False, fast_layout=False):
    """ Runs the experiment for the given parameters and saves the results to a CSV file.

    Args:
        filename (str): The name of the file to save the results.
        qc_list: list of quantum circuits to transpile.
        routing_pass (str): The routing pass to use.
        layout_pass (str): The layout pass to use.
        coupling_map (CouplingMap): The coupling map to use.
        num_pm (int): The number of pass managers to build.
        heuristic (str): The heuristic to use (for regular Sabre).
        seed (int): The seed to use.
        look (int): The lookahead steps to use (for Sabre Look).
        beam (int): The beam width to use (for Sabre Look, Sabre Dive).
    Returns:
        df (pd.DataFrame): A dataframe with the results of the experiment.
    """ 

    # Build the pass managers
    pm_list = build_pm_list(routing_pass, layout_pass, coupling_map, num_pm, heuristic, 
                            seed, look, beam, triv_layout=False, fast_layout=fast_layout)

    # Initialize an empty list to hold the data frames
    data_frames = []

    # Run the experiment for each of the qc in the list
    counter = 0
    for qc in qc_list:
        data = run_one_circuit(qc, pm_list)
        data['look'] = look
        data['beam'] = beam
        data['heuristic'] = heuristic
        
        # Convert the data to a DataFrame and append it to the list
        data_df = pd.DataFrame([data])
        data_frames.append(data_df)

        # Concatenate all the DataFrames and save to CSV after each iteration
        all_data_df = pd.concat(data_frames, ignore_index=True)
        all_data_df.to_csv(filename, index=False)

        print("Finished: ", counter, " out of ", len(qc_list))
        counter += 1

    return all_data_df


def run_experiment_beam(qc, routing_pass, layout_pass, coupling_map, beam_list,
                        num_pm=4, heuristic="basic", seed=42, look=0): 
    """ Runs the experiment for the given parameters for the beam list.
    
    Args:
        filename (str): The name of the file to save the results.
        qc_list: list of qunatum circuits to transpile.
        routing_pass (str): The routing pass to use.
        layout_pass (str): The layout pass to use.
        coupling_map (CouplingMap): The coupling map to use.
        beam_list (list): The list of beam values to use.
        num_pm (int): The number of pass managers to build.
        heuristic (str): The heuristic to use (for regular Sabre).
        seed (int): The seed to use.
        look (int): The lookahead steps to use (for Sabre Look).

    Returns:
        df (pd.DataFrame): A dataframe with the results of the experiment.
    """
    data_list = []
    counter = 0
    for beam in beam_list:
        pm_list = build_pm_list(routing_pass, layout_pass, coupling_map, num_pm, heuristic, 
                                seed, look, beam)
        data = run_one_circuit(qc, pm_list)
        data['look'] = look
        data['beam'] = beam
        data['heuristic'] = heuristic
        data_list.append(data)
        print("Finished: ", counter, " out of ", len(beam_list))
        counter += 1
    return pd.DataFrame(data_list)



