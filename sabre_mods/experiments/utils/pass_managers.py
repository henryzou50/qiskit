""" Contains the functions for the pass managers used in the experiments. """

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, \
                                     EnlargeWithAncilla
import time
import numpy as np
import pandas as pd

def build_pm(coupling_map, routing_pass, layout_pass):
    """ Build a pass manager for the given routing and layout passes. 
    
    Args:
        coupling_map (CouplingMap): The coupling map of the device.
        routing_pass (RoutingPass): The routing pass to use.
        layout_pass (LayoutPass): The layout pass to use.
    
    Returns:
        PassManager: The pass manager to use.
    """

    return PassManager([
        layout_pass,
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        routing_pass
    ])
