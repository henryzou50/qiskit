from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, \
                                     EnlargeWithAncilla, TrivialLayout, SetLayout
from qiskit.transpiler.passes.layout.sabre_layout    import SabreLayout
import time
import numpy as np
import pandas as pd

# routing passes
from qiskit.transpiler.passes.routing.sabre_swap import SabreSwap as Sabre
from sabre_mods.sabre_swap_v0_20_   import SabreSwap as SabreSwap_v0_20

def build_rp(rp_str, cm, seed=42, look=0, beam=0, num_iter=1, crit=1):
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
    else:
        raise ValueError(f"Unknown routing pass {rp_str}")
    
    return rp 

def build_lp(lp_str, cm, rp, seed=42):
    """ Build a layout pass based on the layout pass string lp_str. 
    
    Args: 
        lp_str (str): layout pass string
        cm (list): coupling map
        rp (obj): routing pass
        seed     (int): seed for the layout pass
    """

    lp = None
    print(f"Using seed {seed} for layout pass.")


    if lp_str   == "sabre_layout":
        print(f"    Building Sabre layout pass")
        lp = SabreLayout(cm, rp, seed=seed)
    elif lp_str == "fast_layout":
        print(f"    Building Fast layout pass")
        lp = SabreLayout(cm, Sabre(cm, seed=seed), seed=seed)
    elif lp_str == "trivial_layout":
        print(f"    Building Trivial layout pass")
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