""" Contains the functions for the pass managers used in the experiments. """

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, \
                                     EnlargeWithAncilla
import time
import numpy as np
import pandas as pd
import os

def build_pm(coupling_map, layout_pass, routing_pass):
    """ Build a pass manager for the given routing and layout passes. 
    
    Args:
        coupling_map (CouplingMap): The coupling map of the device.
        layout_pass (LayoutPass): The layout pass to use.
        routing_pass (RoutingPass): The routing pass to use. (can be None)
    
    Returns:
        PassManager: The pass manager to use.
    """
    pass_manager = PassManager([
        layout_pass,
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout()
    ])
    if routing_pass is not None:
        pass_manager.append(routing_pass)

    return pass_manager

def run_circuit(qc, pm, num_times):
    """ Run the experiment for the given pass manager num_times and returns a 
    dictionary of the transpialtion data for the best transpiled circuit (determined 
    as the one with the lowest depth, if there are multiple results with the same 
    depth, then the one with the lowest number of cx gates.)
    
    For each num_times transpilation, we will change the seed for the pass manager.

    Args:
        qc (QuantumCircuit): The quantum circuit to transpile.
        pm (PassManager): The pass manager to use.
        num_times (int): The number of times to transpile the circuit.

    Returns
        dict: The transpilation data for the best transpiled circuit. 
    """
    best_data = {
        'depth': np.inf,
        'cx': np.inf,
        'time': np.inf,
    }

    depths    = []
    cx_counts = []
    times     = []

    print(f'    Running the experiment for circuit {qc.name}...')
    print(f'        Depths: ', end='')
    for _ in range(num_times):
        start = time.time()
        qc_tr = pm.run(qc)
        end = time.time()

        # Decompose the circuit to decompose the swaps gate into cx gates
        qc_tr = qc_tr.decompose()

        # Obtain data
        depths.append(qc_tr.depth(lambda x: x.operation.num_qubits == 2))
        print(depths[-1], end=', ')
        times.append(end - start)
        ops = qc_tr.count_ops()
        if 'cx' in ops:
            cx_counts.append(ops['cx'])
        else:
            cx_counts.append(0)
        
        # Update best data
        if depths[-1] < best_data['depth']:
            best_data['depth'] = depths[-1]
            best_data['cx'] = cx_counts[-1]
            best_data['time'] = times[-1]
        elif depths[-1] == best_data['depth']:
            if cx_counts[-1] < best_data['cx']:
                best_data['depth'] = depths[-1]
                best_data['cx'] = cx_counts[-1]
                best_data['time'] = times[-1]

        # Change the seed of the pass manager, pm._tasks[0][0] is the layout pass
        # pm._tasks[0][4] is the routing pass (note the routing pass does not have to exist)
        if hasattr(pm._tasks[0][0], 'seed'):
            pm._tasks[0][0].seed += 1
        # check if pm._tasks[0][4] is occupied
        if len(pm._tasks[0]) > 4:
            if hasattr(pm._tasks[0][4], 'seed'):
                pm._tasks[0][4].seed +=1
    print()

    best_data['depth_std'] = np.std(depths)
    best_data['cx_std'] = np.std(cx_counts)
    best_data['time_std'] = np.std(times)

    return best_data
        
def run_circuits(filename, circuits, pm, num_times=4, max_iter=3, beam=1, look=1, crit=1):
    """ Run the experiment for the given circuits and pass manager num_times, 
    and store the results in a pandas DataFrame. After running 1 circuit, 
    update the filename (a csv file) each time. 

    Args:
        filename (str): The name of the file to store the results.
        circuits (list): The list of QuantumCircuits to transpile.
        pm (PassManager): The pass manager to use.
        num_times (int): The number of times to transpile the circuits.
        max_iter (int): The maximum number of iterations for the routing pass.
        beam (int): The beam width for the SabreDive and SabreLook
        look (int): The number of look ahead for the SabreLook
        crit (int): The critical path weight for the SabreCrit

    Returns:
        df (DataFrame): The DataFrame containing the results of the experiment.
    """

    # Create the DataFrame
    df = []

    # Get the string of layout pass from pm
    lp_str = pm._tasks[0][0].__class__.__name__

    # check if directory of the filename exists, if not create it
    directory = '/'.join(filename.split('/')[:-1])
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    for qc in circuits:
        data = run_circuit(qc, pm, num_times)
        data['circuit'] = qc.name
        data['layout_pass'] = lp_str
        data['num_times'] = num_times
        data['max_iter'] = max_iter
        data['beam'] = beam
        data['look'] = look
        data['crit'] = crit
        df.append(data)

        pd.DataFrame(df).to_csv(filename, index=False)

        depth = data['depth']
        time_ = data['time']

        print(f'    Circuit {qc.name} transpiled with best depth {depth} and time {time_}.')

    return pd.DataFrame(df)