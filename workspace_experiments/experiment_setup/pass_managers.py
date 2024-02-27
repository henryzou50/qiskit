from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, \
                                     EnlargeWithAncilla, TrivialLayout, SetLayout
from qiskit.transpiler.passes.layout.sabre_layout    import SabreLayout
import time
import numpy as np
import pandas as pd

# routing passes
from qiskit.transpiler.passes.routing.sabre_swap             import SabreSwap as Sabre
from qiskit.transpiler.passes.routing.sabre_swap_v0_20_      import SabreSwap as SabreSwap_v0_20
from qiskit.transpiler.passes.routing.sabre_swap_v0_20_depth import SabreSwap as SabreSwap_v0_20_depth
from qiskit.transpiler.passes.routing.sabre_swap_v0_20_dive  import SabreSwap as SabreSwap_v0_20_dive


def build_rp(rp_str, cm, seed=42, look=0, beam=1, num_iter=1, crit=1):
    """ Build a routing pass based on the routing pass string rp_str. 
    
    Args: 
        rp_str (str): routing pass string
        cm (list): coupling map
        seed     (int): seed for the routing pass
        look     (int): look ahead (only for Sabre Look)
        beam     (int): beam width (only for Sabre Look and Sabre Dive)
        num_iter (int): number of iterations (only for Sabre Dive)
        crit     (int): criteral path weight (only for Sabre Crit) 
    """

    rp = None
    print(f"Using seed {seed} for routing pass.")

    if rp_str   == "sabre":
        print(f"    Building Sabre routing pass")
        rp = Sabre(cm, seed=seed)
    elif rp_str == "sabre_v0_20_":
        print(f"    Building Sabre v0.20 routing pass")
        rp = SabreSwap_v0_20(cm, seed=seed)
    elif rp_str == "sabre_v0_20_extended":
        print(f"    Building Sabre v0.20 extended routing pass")
        rp = SabreSwap_v0_20(cm, seed=seed, heuristic="lookahead")
    elif rp_str == "sabre_v0_20_depth":
        print(f"    Building Sabre v0.20 depth routing pass")
        rp = SabreSwap_v0_20_depth(cm, seed=seed)
    elif rp_str == "sabre_v0_20_dive":
        print(f"    Building Sabre v0.20 dive routing pass with beam {beam}")
        rp = SabreSwap_v0_20_dive(cm, seed=seed, beam_width=beam)
    else:
        raise ValueError(f"Unknown routing pass {rp_str}")
    
    return rp 

def build_lp(lp_str, cm, rp, seed=42, max_iter=1):
    """ Build a layout pass based on the layout pass string lp_str. 
    
    Args: 
        lp_str (str): layout pass string
        cm (list): coupling map
        rp (obj): routing pass
        seed     (int): seed for the layout pass
    """

    lp = None
    print(f"        Using seed {seed} and max iterations {max_iter} for layout pass.")

    if lp_str   == "sabre_layout":
        print(f"            Building Sabre layout pass")
        lp = SabreLayout(cm, rp, seed=seed, max_iterations=max_iter)
    elif lp_str == "fast_layout":
        print(f"            Building Fast layout pass")
        lp = SabreLayout(cm, Sabre(cm, seed=seed), seed=seed, max_iterations=max_iter)
    elif lp_str == "trivial_layout":
        print(f"            Building Trivial layout pass")
        lp = TrivialLayout()
    else:
        raise ValueError(f"Unknown layout pass {lp_str}")
    
    
    return lp

def build_pm(rp, lp, cm):
    """ Build a pass manager based on the routing and layout passes. 
    
    Args: 
        rp (obj): routing pass
        lp (obj): layout pass
        cm (list): coupling map
    """

    return PassManager([
        lp,
        FullAncillaAllocation(cm),
        EnlargeWithAncilla(),
        ApplyLayout(),
        rp
    ])

def build_pm_list(rp_str, lp_str, cm, num_pm=4, seed=42, look=0, beam=1, num_iter=1, crit=1, max_iter=1):
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
        max_iter (int): The number of iterations for the layout pass.
    """

    pm_list = []

    for i in range(num_pm):
        rp = build_rp(rp_str, cm, seed+i, look, beam, num_iter, crit)
        lp = build_lp(lp_str, cm, rp, seed+i, max_iter)
        pm_list.append(build_pm(rp, lp, cm))

    return pm_list


def run_one_circuit(qc, pm_list):
    """ Runs the experiment for the given pass managers. and returns a dictionary of the 
    transpilation data for th best transpiled circuit (determined as the one with the lowest 
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
    
    depths   = []
    cx_gates = []
    times    = []
    seed_idx = 0
    
    for pm in pm_list:
        # Timing the transpilation
        start = time.time()
        qc_tr = pm.run(qc)
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
            best_data['seed'] = seed_idx
        elif depth == best_data['depth'] and cx < best_data['cx gates']:
            best_data['depth'] = depth
            best_data['cx gates'] = cx
            best_data['time'] = time_
            best_data['seed'] = seed_idx
        seed_idx += 1

    best_data['depth_std'] = round_to_sig_figures(np.std(depths))
    best_data['cx_std']    = round_to_sig_figures(np.std(cx_gates))
    best_data['time_std']  = round_to_sig_figures(np.std(times))

    return best_data


def run_experiment(filename, qc_list, rp_str, lp_str, coupling_map, num_pm=4, seed=42, 
                   look=0, beam=1, num_iter=1, crit=1, max_iter=1):
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
        num_iter (int): The number of iterations to use (for Sabre Dive).
        crit (int): The criticality to use (for Sabre Crit).
        max_iter (int): The number of iterations for the layout pass.
    Returns:
        df (pd.DataFrame): A dataframe with the results of the experiment.
    """ 

    # Build the pass managers


    pm_list = build_pm_list(rp_str, lp_str, coupling_map, num_pm, seed, look, beam, num_iter, crit, max_iter)

    # Initialize an empty list to hold the data frames
    data_frames = []

    # Run the experiment for each of the qc in the list
    counter = 0
    for qc in qc_list:
        data = run_one_circuit(qc, pm_list)
        data['rp'] = rp_str
        data['lp'] = lp_str
        data['look'] = look
        data['beam'] = beam
        data['num_iter'] = num_iter
        data['crit'] = crit
        data['max_iter'] = max_iter
        
        # Convert the data to a DataFrame and append it to the list
        data_df = pd.DataFrame([data])
        data_frames.append(data_df)

        # Concatenate all the DataFrames and save to CSV after each iteration
        all_data_df = pd.concat(data_frames, ignore_index=True)
        all_data_df.to_csv(filename, index=False)

        depth = data['depth']
        time_ = data['time']
        print(f"Finished: {counter} out of {len(qc_list)} with depth {depth} and time {time_}")
        counter += 1

    return all_data_df

def run_beam_experiment(filename, qc, rp_str, lp_str, coupling_map, beam_list, seed=42, num_pm=1,
                   look=0, num_iter=1, crit=1, max_iter=1):
    """ Runs the experiment for the given parameters and saves the results to a CSV file.

    Args:
        filename (str): The name of the file to save the results.
        qc: quantum circuit to transpile.
        rp_str (str): The routing pass to use.
        lp_str (str): The layout pass to use.
        coupling_map (CouplingMap): The coupling map to use.
        beam_list (list): The list of beam widths to use.
        seed (int): The seed to use.
        num_pm (int): The number of pass managers to build.
        look (int): The lookahead steps to use (for Sabre Look).
        num_iter (int): The number of iterations to use (for Sabre Dive).
        crit (int): The criticality to use (for Sabre Crit).
        max_iter (int): The number of iterations for the layout pass.
    """

    counter = 0
    data_frames = []
    for beam in beam_list:
        pm_list = build_pm_list(rp_str, lp_str, coupling_map, num_pm, seed, look, beam, 
                                num_iter, crit, max_iter)
        data = run_one_circuit(qc, pm_list)
        data['rp'] = rp_str
        data['lp'] = lp_str
        data['look'] = look
        data['beam'] = beam
        data['num_iter'] = num_iter
        data['crit'] = crit
        data['max_iter'] = max_iter

        data_df = pd.DataFrame([data])
        data_frames.append(data_df)

        all_data_df = pd.concat(data_frames, ignore_index=True)
        all_data_df.to_csv(filename, index=False)

        depth = data['depth']
        time_ = data['time']

        print(f"Finished: {counter} out of {len(beam_list)} with depth {depth} and time {time_}")
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