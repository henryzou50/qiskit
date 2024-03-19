import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from typing import Optional, Union
import networkx as nx


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
            layers: A list of tuples where each tuple contains (gate_type, param).
                    The gate_type can be either 'cx' or 'swap'. 
                    The param is can either be a float between 0 and 1 (represents density,
                    where the number of gates in the layer is the density * len(coupling_map)),
                    or a integer (represents the number of gates in the layer, cannot be 
                    greater than the maximum matching of the coupling map).
            seed: The seed for the random number generator
            barriers: If True, a barrier is added after each layer

        Example Layers:
            - Circuit that alternates between cx and swap gates, and each layer has a density of 0.5
            layers = [
                ('cx', 0.5),
                ('swap', 0.5),
                ('cx', 0.5),
                ('swap', 0.5)
            ]
            - Circuit that has 10 cx gates in the first layer and 5 swap gates in the second layer
            layers = [
                ('cx', 10),
                ('swap', 5)
            ]
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
            gate_type, param = layer

            # Verify if param is a valid float or an integer
            max_matching = self._max_matching(coupling_list)
            if isinstance(param, float):
                if not (0 <= param <= 1):
                    raise ValueError("Density parameter must be between 0 and 1")
            elif isinstance(param, int):
                if param > max_matching:
                    raise ValueError("The number of gates in the layer cannot be greater than the maximum matching of the coupling map")

            self._build_layer(circuit, coupling_list, gate_type, param, unitary_seeds[i])
            if barriers:
                circuit.barrier()

        
        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_instruction(), qubits=self.qubits, inplace=True)

    def _build_layer(self, circuit, coupling_list, gate_type, param, seed):
        """ Builds a layer of gates in the quantum circuit. 

        Args:
            circuit (QuantumCircuit): The quantum circuit to add the gates to
            coupling_list: A list of tuples that represent the coupling map of the device
            gate_type: The type of gate to add. Can be either 'cx' or 'swap'
            param: The parameter of the gate type to add. If param is a float, it represents
                     the density of the gate type to add. If param is an integer, it represents
                     the number of gates in the layer.
            seed: The seed for the random number generator

        Returns:
            None
        """
        # Set the number of gates in the layer according to the param
        if isinstance(param, float):
            num_gates = int(param * circuit.num_qubits)
        else:
            num_gates = param
        
        # Shuffle the coupling list to ensure random exploration
        rng = np.random.default_rng(seed)
        shuffled_coupling_list = coupling_list.copy()
        rng.shuffle(shuffled_coupling_list)

        # Select non-overlapping edges from the shuffled list
        selected_edges = []
        used_qubits = set()

        for edge in shuffled_coupling_list:
            # Check if the number of gates has been reached
            if len(selected_edges) >= num_gates:
                break
            # Check if the edge is not in the used qubits
            if edge[0] not in used_qubits and edge[1] not in used_qubits:
                selected_edges.append(edge)
                used_qubits.add(edge[0])
                used_qubits.add(edge[1])

        # Add the gates to the circuit
        for edge in selected_edges:
            if gate_type == "cx":
                circuit.cx(edge[0], edge[1])
            elif gate_type == "swap":
                circuit.swap(edge[0], edge[1])
            else:
                raise ValueError("Gate type must be either 'cx' or 'swap'")
            
    def _max_matching(self, coupling_list):
        """ Returns the maximum matching of the coupling map, i.e. the maximum number of
        gates that can be executed in parallel for 1 layer. This will be calculated using 
        the Blossom algorithm from networkx.
        
        Args:
            coupling_list: A list of tuples that represent the coupling map of the device
            
        Returns:
            int: The maximum matching of the coupling map
        """
        graph = nx.Graph()
        graph.add_edges_from(coupling_list)
        return len(nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True))
            
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



