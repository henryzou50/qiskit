""" This file contains the functions to create the pass managers used in the experiments. """

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, \
                                     EnlargeWithAncilla, TrivialLayout, SetLayout
from qiskit.transpiler.passes.routing.sabre_swap     import SabreSwap as Sabre
from qiskit.transpiler.passes.routing.sabre_swap_025 import SabreSwap as Sabre025
from qiskit.transpiler.passes.routing.sabre_depth    import SabreSwap as SabreDepth
from qiskit.transpiler.passes.routing.sabre_crit     import SabreSwap as SabreCrit
from qiskit.transpiler.passes.routing.sabre_dive     import SabreSwap as SabreDive
from qiskit.transpiler.passes.routing.sabre_look     import SabreSwap as SabreLook

from qiskit.transpiler.passes.layout.sabre_layout    import SabreLayout

import time
import numpy as np
import pandas as pd

def build_routing_pass(rp_str, coupling_map, seed=42, look=0, beam=5, crit=1, num_iter=1):
    """ Builds a routing pass with the given parameters. 
    
    Args: 
        rp_str (str): The routing pass to use. 
        coupling_map (CouplingMap): The coupling map to use.
        seed (int): The seed to use. 
        look (int): The lookahead steps to use (for Sabre Look). 
        beam (int): The beam width to use (for Sabre Look, Sabre Dive).
        crit (int): The criticality to use (for Sabre Crit).
        num_iter (int): The number of iterations to use (for Sabre Dive).    
    """
    routing_pass = None
    if rp_str == "sabre":
        print("Successfully built Sabre. Seed: ", seed)
        routing_pass = Sabre(coupling_map=coupling_map, seed=seed)
    elif rp_str == "sabre_025":
        print("Successfully built Sabre_025. Seed: ", seed)
        routing_pass = Sabre025(coupling_map=coupling_map, seed=seed)
    elif rp_str == "sabre_025_extended":
        print("Successfully built Sabre_Extended. Seed: ", seed)
        routing_pass = Sabre025(coupling_map=coupling_map, heuristic="lookahead", seed=seed)
    elif rp_str == "sabre_025_depth":
        print("Successfully built Sabre_Depth. Seed: ", seed)
        routing_pass = SabreDepth(coupling_map=coupling_map, seed=seed)
    elif rp_str == "sabre_025_crit":
        print("Successfully built Sabre_Crit. Seed: ", seed, " Crit: ", crit)
        routing_pass = SabreCrit(coupling_map=coupling_map, seed=seed, crit_weight=crit)
    elif rp_str == "sabre_025_dive":
        print("Successfully built Sabre_Dive. Seed: ", seed, " Beam: ", beam, " Iter: ", num_iter)
        routing_pass = SabreDive(coupling_map=coupling_map, seed=seed, 
                                 beam_width=beam, num_iterations=num_iter)
    elif rp_str == "sabre_025_look":
        print("Successfully built Sabre_Look. Seed: ", seed, " Look: ", look, " Beam: ", beam)
        routing_pass = SabreLook(coupling_map=coupling_map, seed=seed, look=look, beam=beam)
    else:
        raise ValueError("Invalid routing pass string")
    return routing_pass

def build_layout_pass(lp_str, coupling_map, routing_pass, seed=42):
    """ Builds a layout pass with the given parameters. 
    
    Args: 
        lp_str (str): The layout pass to use. 
        coupling_map (CouplingMap): The coupling map to use. 
    """
    layout_pass = None
    if lp_str == "sabre_layout":
        print("Successfully built Sabre_Layout. Seed: ", seed)
        layout_pass = SabreLayout(coupling_map=coupling_map, routing_pass=routing_pass, seed=seed)
    elif lp_str == "fast_layout":
        print("Successfully built Fast_Layout")
        layout_pass = SabreLayout(coupling_map=coupling_map, 
                                  routing_pass=Sabre(coupling_map=coupling_map, seed=seed), seed=seed)
    elif lp_str == "trivial_layout":
        print("Successfully built Trivial_Layout")
        layout_pass = TrivialLayout(coupling_map=coupling_map)
    elif lp_str == "set_layout":
        print("Successfully built Set_Layout")
        layout_pass = SetLayout([1,2,5,6,9,10])
    else:
        raise ValueError("Invalid layout pass string")
    return layout_pass


def build_pm(routing_pass, layout_pass, coupling_map):
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

    return PassManager([
        layout_pass,
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        routing_pass
    ])

def build_pm_list(rp_str, lp_str, coupling_map, num_pm=4, seed=42, look=0, beam=5, crit=1, num_iter=1):
    """ Builds a list of pass managers with the given parameters, and where each pm has the 
    same parameters except for the seed.

    Args: 
        rp_str (str): The routing pass to use. 
        lp_str (str): The layout pass to use. 
        coupling_map (CouplingMap): The coupling map to use. 
        num_pm (int): The number of pass managers to
        seed (int): The seed to use.
        look (int): The lookahead steps to use (for Sabre Look).
        beam (int): The beam width to use (for Sabre Look, Sabre Dive).
        crit (int): The criticality to use (for Sabre Crit).
        num_iter (int): The number of iterations to use (for Sabre Dive).
    """

    pm_list = []

    for i in range(num_pm):
        routing_pass = build_routing_pass(rp_str, coupling_map, seed+i, look, beam, crit, num_iter)
        layout_pass = build_layout_pass(lp_str, coupling_map, routing_pass, seed+i)
        pm_list.append(build_pm(routing_pass, layout_pass, coupling_map))

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
        depth = qc_tr.depth(lambda x: x.operation.num_qubits == 2)
        ops = qc_tr.count_ops()
        if 'cx' not in ops:
            cx = 0
        else:
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

def run_experiment(filename, qc_list, rp_str, lp_str, coupling_map, num_pm=4, seed=42, 
                   look=0, beam=5, crit=1, num_iter=1):
    """ Runs the experiment for the given parameters and saves the results to a CSV file.

    Args:
        filename (str): The name of the file to save the results.
        qc_list: list of quantum circuits to transpile.
        rp_str (str): The routing pass to use.
        lp_str (str): The layout pass to use.
        coupling_map (CouplingMap): The coupling map to use.
        num_pm (int): The number of pass managers to build.
        seed (int): The seed to use.
        look (int): The lookahead steps to use (for Sabre Look).
        beam (int): The beam width to use (for Sabre Look, Sabre Dive).
        crit (int): The criticality to use (for Sabre Crit).
        num_iter (int): The number of iterations to use (for Sabre Dive).
    Returns:
        df (pd.DataFrame): A dataframe with the results of the experiment.
    """ 

    # Build the pass managers
    pm_list = build_pm_list(rp_str, lp_str, coupling_map, num_pm, seed, look, beam, crit, num_iter)

    # Initialize an empty list to hold the data frames
    data_frames = []

    # Run the experiment for each of the qc in the list
    counter = 0
    for qc in qc_list:
        data = run_one_circuit(qc, pm_list)
        data['look'] = look
        data['beam'] = beam
        data['crit'] = crit
        data['rp'] = rp_str
        data['lp'] = lp_str
        data['num_iter'] = num_iter
        
        # Convert the data to a DataFrame and append it to the list
        data_df = pd.DataFrame([data])
        data_frames.append(data_df)

        # Concatenate all the DataFrames and save to CSV after each iteration
        all_data_df = pd.concat(data_frames, ignore_index=True)
        all_data_df.to_csv(filename, index=False)

        print("Finished: ", counter, " out of ", len(qc_list))
        counter += 1

    return all_data_df


def run_one_circuit_info(qc, pm_list):
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
    #qc_decomposed = qc.decompose()
    
    depths = []
    cx_gates = []
    times = []

    qc_tr_list = []
    
    for pm in pm_list:
        # Timing the transpilation
        start = time.time()
        qc_tr = pm.run(qc)
        end = time.time()

        # Decompose the swaps gate in the circuit
        #qc_tr = qc_tr.decompose()
        # qc_tr = qc_tr.decompose()

        # Obtaining data
        depth = qc_tr.depth()
        ops = qc_tr.count_ops()
        if 'cx' not in ops:
            cx = 0
        else:
            cx = qc_tr.count_ops()['cx']
        time_ = end - start

        depths.append(depth)
        cx_gates.append(cx)
        times.append(time_)

        if depth < best_data['depth']:
            best_data['depth'] = depth
            best_data['cx gates'] = cx
            best_data['time'] = time_
        qc_tr_list.append(qc_tr)

    best_data['depth_std'] = round_to_sig_figures(np.std(depths))
    best_data['cx_std']    = round_to_sig_figures(np.std(cx_gates))
    best_data['time_std']  = round_to_sig_figures(np.std(times))

    return best_data, qc_tr_list

def run_experiment_info(filename, qc_list, rp_str, lp_str, coupling_map, num_pm=4, seed=42, 
                   look=0, beam=1, crit=1):
    """ Runs the experiment for the given parameters and saves the results to a CSV file.

    Args:
        filename (str): The name of the file to save the results.
        qc_list: list of quantum circuits to transpile.
        rp_str (str): The routing pass to use.
        lp_str (str): The layout pass to use.
        coupling_map (CouplingMap): The coupling map to use.
        num_pm (int): The number of pass managers to build.
        seed (int): The seed to use.
        look (int): The lookahead steps to use (for Sabre Look).
        beam (int): The beam width to use (for Sabre Look, Sabre Dive).
        crit (int): The criticality to use (for Sabre Crit).
    Returns:
        df (pd.DataFrame): A dataframe with the results of the experiment.
    """ 

    # Build the pass managers
    pm_list = build_pm_list(rp_str, lp_str, coupling_map, num_pm, seed, look, beam, crit)

    # Initialize an empty list to hold the data frames
    data_frames = []

    # Run the experiment for each of the qc in the list
    counter = 0
    
    qc_tr_list = []
    for qc in qc_list:
        data, qc_tr = run_one_circuit_info(qc, pm_list)
        data['look'] = look
        data['beam'] = beam
        data['crit'] = crit
        data['rp'] = rp_str
        data['lp'] = lp_str
        
        # Convert the data to a DataFrame and append it to the list
        data_df = pd.DataFrame([data])
        data_frames.append(data_df)

        # Concatenate all the DataFrames and save to CSV after each iteration
        all_data_df = pd.concat(data_frames, ignore_index=True)
        all_data_df.to_csv(filename, index=False)

        print("Finished: ", counter, " out of ", len(qc_list))
        counter += 1
        qc_tr_list += qc_tr

    return all_data_df, qc_tr_list


# TODO Adjust below functions

def run_experiment_beam(filename, qc, routing_pass, layout_pass, coupling_map, beam_list,
                        num_pm=4, heuristic="basic", seed=42, look=0, triv_layout=False): 
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
    counter = 0
    data_frames = []
    for beam in beam_list:
        pm_list = build_pm_list(routing_pass, layout_pass, coupling_map, num_pm, heuristic, 
                                seed, look, beam, triv_layout=triv_layout)
        data = run_one_circuit(qc, pm_list)
        data['look'] = look
        data['beam'] = beam
        data['heuristic'] = heuristic

        data_df = pd.DataFrame([data])
        data_frames.append(data_df)

        all_data_df = pd.concat(data_frames, ignore_index=True)
        all_data_df.to_csv(filename, index=False)

        print("Finished: ", counter, " out of ", len(beam_list))
        counter += 1
    return all_data_df

def run_experiment_look(filename, qc, routing_pass, layout_pass, coupling_map, look_list,
                        num_pm=4, heuristic="basic", seed=42, beam=1, triv_layout=False):
    """ Runs the experiment for the given parameters for the look list.
    
    Args:
        filename (str): The name of the file to save the results.
        qc_list: list of qun
        routing_pass (str): The routing pass to use.
        layout_pass (str): The layout pass to use.
        coupling_map (CouplingMap): The coupling map to use.
        look_list (list): The list of look values to use.
        num_pm (int): The number of pass managers to build.
        heuristic (str): The heuristic to use (for regular Sabre).
        seed (int): The seed to use.
        beam (int): The beam width to use (for Sabre Look, Sabre Dive).
    
    Returns:
        df (pd.DataFrame): A dataframe with the results of the experiment.
    """
    counter = 0
    data_frames = []
    for look in look_list:
        pm_list = build_pm_list(routing_pass, layout_pass, coupling_map, num_pm, heuristic, 
                                seed, look, beam, triv_layout=triv_layout)
        data = run_one_circuit(qc, pm_list)
        data['look'] = look
        data['beam'] = beam
        data['heuristic'] = heuristic

        data_df = pd.DataFrame([data])
        data_frames.append(data_df)

        all_data_df = pd.concat(data_frames, ignore_index=True)
        all_data_df.to_csv(filename, index=False)

        print("Finished: ", counter, " out of ", len(look_list))
        counter += 1
    return all_data_df


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
