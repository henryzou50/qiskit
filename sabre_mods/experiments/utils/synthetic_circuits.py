import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from typing import Optional, Union


class SyntheticCircuit(QuantumCircuit):
    """ A synthetic circuit model. 
    
    The model circuit represents a circuit that respects the edges of the coupling 
    map of device and is built synthetically from a list of layers. Each layer 
    exhibit a depth of 1, and all of the gates in one specific layer are the
    same gate type. 

    i.e., if there are n layers, then the original depth of the circuit is n.
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        layers: list,
        seed: Optional[Union[int, np.random.Generator]] = None,
        barriers: bool = False,
    ) -> None:
        """ Initializes the synthetic circuit from the coupling map and layers.

        Args:
            coupling_map (CouplingMap): The coupling map of the device
            layers: A list of tuples where each tuple contains the gate type and density
                    of the gate type. The gate type can be either 'cx' or 'swap'. The density
                    is a float between 0 and 1.
            seed: The seed for the random number generator
            barriers: If True, a barrier is added after each layer
        """ 
        # Initialize RNG
        if seed is None:
            rng_set = np.random.default_rng()
            seed = rng_set.integers(low=1, high=1000)
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)
        unitary_seeds = rng.integers(low=1, high=1000, size=len(layers))

        # Build circuit
        circuit = QuantumCircuit(coupling_map.size())
        # Respects the edges of the coupling map
        coupling_list = list(coupling_map.get_edges())

        for i, layer in enumerate(layers):
            gate_type, density = layer
            self._build_layer(circuit, coupling_list, gate_type, density, unitary_seeds[i])
            if barriers:
                circuit.barrier()

        
        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_instruction(), qubits=self.qubits, inplace=True)

    def _build_layer(self, circuit, coupling_list, gate_type, density, seed):
        """ Builds a layer of gates in the quantum circuit. 

        Args:
            circuit (QuantumCircuit): The quantum circuit to add the gates to
            coupling_list: A list of tuples that represent the coupling map of the device
            gate_type: The type of gate to add. Can be either 'cx' or 'swap'
            density: The density of the gate type to add. A float between 0 and 1
            seed: The seed for the random number generator

        Returns:
            None
        """
        # Ensure density is within the valid range
        if not (0 <= density <= 1):
            raise ValueError("Density parameter must be between 0 and 1")
        
        # Set the seed for the random number generator only if it is not None
        rng = np.random.default_rng(seed)

        # Shuffle the coupling list to ensure random exploration
        shuffled_coupling_list = coupling_list.copy()
        rng.shuffle(shuffled_coupling_list)

        # Select non-overlapping edges from the shuffled list
        selected_edges = []
        used_qubits = set()

        for edge in shuffled_coupling_list:
            if edge[0] not in used_qubits and edge[1] not in used_qubits:
                selected_edges.append(edge)
                used_qubits.add(edge[0])
                used_qubits.add(edge[1])

            if len(used_qubits) == circuit.num_qubits:
                break

        # Add the gates to the circuit
        for edge in selected_edges:
            if gate_type == "cx":
                circuit.cx(edge[0], edge[1])
            elif gate_type == "swap":
                circuit.swap(edge[0], edge[1])
            else:
                raise ValueError("Gate type must be either 'cx' or 'swap'")
            
    def _scramble_qubits(self, circuit, seed):
        """ Scrambles the qubits in the quantum circuit. 

        Args:
            circuit (QuantumCircuit): The quantum circuit to scramble the qubits
            seed: The seed for the random number generator

        Returns:
            None
        """
        # Set the seed for the random number generator only if it is not None
        rng = np.random.default_rng(seed)
        qubits = list(range(circuit.num_qubits))
        rng.shuffle(qubits)
        circuit.reorder_qubits(qubits)
        return circuit



