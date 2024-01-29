""" This file contains the functions to create the pass managers used in the experiments. """

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, \
                                     EnlargeWithAncilla, TrivialLayout  

def build_pm(routing_pass, layout_pass, coupling_map, heuristic="basic", seed=42, look=0, beam=1):
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
        routing_args['beam_'] = beam
    if heuristic != "basic":
        routing_args['heuristic'] = heuristic

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

def build_pm_list(num_pm, routing_pass, layout_pass, coupling_map, heuristic="basic", seed=42, 
                  look=0, beam=1):
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
                                seed+i, look, beam))
    return pm_list





    




