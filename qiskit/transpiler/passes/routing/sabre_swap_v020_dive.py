# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Routing via SWAP insertion using the SABRE method from Li et al."""

import logging
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import retworkx

from qiskit.circuit.library.standard_gates import SwapGate, CXGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode


class SabreSwap(TransformationPass):
    r"""Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of the SWAP-based heuristic search from the SABRE qubit
    mapping paper [1] (Algorithm 1). The heuristic aims to minimize the number
    of lossy SWAPs inserted and the depth of the circuit.

    This algorithm starts from an initial layout of virtual qubits onto physical
    qubits, and iterates over the circuit DAG until all gates are exhausted,
    inserting SWAPs along the way. It only considers 2-qubit gates as only those
    are germane for the mapping problem (it is assumed that 3+ qubit gates are
    already decomposed).

    In each iteration, it will first check if there are any gates in the
    ``front_layer`` that can be directly applied. If so, it will apply them and
    remove them from ``front_layer``, and replenish that layer with new gates
    if possible. Otherwise, it will try to search for SWAPs, insert the SWAPs,
    and update the mapping.

    The search for SWAPs is restricted, in the sense that we only consider
    physical qubits in the neighborhood of those qubits involved in
    ``front_layer``. These give rise to a ``swap_candidate_list`` which is
    scored according to some heuristic cost function. The best SWAP is
    implemented and ``current_layout`` updated.

    **References:**

    [1] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(
        self,
        coupling_map,
        seed=None,
        fake_run=False,
    ):
        r"""SabreSwap initializer.

        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            seed (int): random seed used to tie-break among candidate swaps.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.

        Additional Information:

            The search space of possible SWAPs on physical qubits is explored
            by assigning a score to the layout that would result from each SWAP.
            The goodness of a layout is evaluated based on how viable it makes
            the remaining virtual gates that must be applied. A few heuristic
            cost functions are supported
        """

        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if coupling_map is None or coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()
        self.seed = seed
        self.fake_run = fake_run
        self.required_predecessors = None
        self._bit_indices = None
        self.dist_matrix = None
        self.dag = None
        self.beam_width = 100
        self.rng = None

    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")
        
        # Initialize self variables
        self.dist_matrix = self.coupling_map.distance_matrix
        self._bit_indices = {bit: idx for idx, bit in enumerate(dag.qregs["q"])}
        self.required_predecessors = self._build_required_predecessors(dag)
        self.dag = dag
        self.rng = np.random.default_rng(self.seed)


        if self.beam_width >= 1:
            return self._dive()

        rng = self.rng

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None
        if not self.fake_run:
            mapped_dag = dag.copy_empty_like()

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)

        # Start algorithm from the front layer and iterate until all gates done.
        front_layer = dag.front_layer()

        while front_layer:
            execute_gate_list = []

            # Remove as many immediately applicable gates as possible
            new_front_layer = []
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    # Accessing layout._v2p directly to avoid overhead from __getitem__ and a
                    # single access isn't feasible because the layout is updated on each iteration
                    if self.coupling_map.graph.has_edge(
                        current_layout._v2p[v0], current_layout._v2p[v1]
                    ):
                        execute_gate_list.append(node)
                    else:
                        new_front_layer.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)
            front_layer = new_front_layer

            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(mapped_dag, node, current_layout, canonical_register)
                    for successor in self._successors(node, dag):
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)
                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            swap_scores = {}
            for swap_qubits in self._obtain_swaps(front_layer, current_layout):
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)
                score = self._score_heuristic(
                    front_layer, trial_layout 
                )
                swap_scores[swap_qubits] = score
            min_score = min(swap_scores.values())
            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
            best_swap = rng.choice(best_swaps)
            self._apply_gate(
                mapped_dag,
                DAGOpNode(op=SwapGate(), qargs=best_swap),
                current_layout,
                canonical_register,
            )
            current_layout.swap(*best_swap)

        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return dag
    
    def _dive(self):
        """ Run SabreSwap with beam search to get all the beam states, and then performs 
        the regular algorithm on each of the beam states to get a finished circuit mapping. 
        The beam states represents the initial mappings of the circuit that are promising. 
        
        Args:
            None
        Returns:
            DAGCircuit: The dag that has the lowest circuit depth. 
        """

        # Phase 1) Create the mapped_dag, current layout, and front layer 
        rng = np.random.default_rng(self.seed)
        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = self.dag.copy_empty_like()
        current_layout = Layout.generate_trivial_layout(self.dag.qregs["q"])
        front_layer = self.dag.front_layer()

        initial_state = {
            "mapped_dag": mapped_dag,
            "layout": current_layout,
            "front_layer": front_layer,
            "predecessors": self.required_predecessors.copy(),
            "gates_done": 0
        }

        # Phase 2) Update mapped_dag, current layout and front layer until all gates are exhausted
        self._update_state(initial_state)

        # if front layer is not empty, then we need to perform beam search
        if initial_state["front_layer"]:
            # Phase 3) Use bfs to get the beam states
            beam_states = self._get_beam_states(initial_state)

            # Phase 4) Perform the regular algorithm on each of the beam states
            for state in beam_states:
                while state["front_layer"]:
                    self._find_and_apply_swaps(state)
                    self._update_state(state)
        
            # Phase 5) Set the final dag to be the one with the lowest depth
            beam_states.sort(key=lambda x: x["mapped_dag"].depth())
            mapped_dag = beam_states[0]["mapped_dag"]
        

        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return self.dag
    
    def _update_state(self, state):
        """ Updates the state of the circuit mapping until all gates are exhausted
        
        Args:
            state (Dict): the current state of the circuit mapping. 
        Returns:
            None
        """
        mapped_dag     = state["mapped_dag"]
        current_layout = state["layout"]
        front_layer    = state["front_layer"]
        predecessors   = state["predecessors"]
        gates_done     = state["gates_done"]


        while True:
            execute_gate_list = []
            # Remove as many immediately applicable gates as possible
            new_front_layer = []

            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    # Accessing layout._v2p directly to avoid overhead from __getitem__ and a
                    # single access isn't feasible because the layout is updated on each iteration
                    if self.coupling_map.graph.has_edge(
                        current_layout._v2p[v0], current_layout._v2p[v1]
                    ):
                        execute_gate_list.append(node)
                    else:
                        new_front_layer.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)
            front_layer = new_front_layer


            if not execute_gate_list: # no more gates to execute
                break
            else:
                for node in execute_gate_list:
                    self._apply_gate(mapped_dag, node, current_layout, self.dag.qregs["q"])
                    gates_done += 1
                    for successor in self._successors(node, self.dag):
                        predecessors[successor] -= 1
                        if predecessors[successor] == 0:
                            front_layer.append(successor)
        state["front_layer"] = front_layer
        state["predecessors"] = predecessors
        state["gates_done"] = gates_done
    
    def _get_beam_states(self, initial_state): 
        """ Gets the list of the beam states, which are possible initial mappings of the circuit. 
        The beam states include the dag, the layout, and the front layer. The number of beam states
        is equal to the beam width.
        
        Args:
            initial_state (Dict): the initial state of the circuit mapping.
        Returns:
            List: A list of the beam states. 
        """

        # perform bfs to get the beam states, once the nodes at the current level is greater than 
        # the beam width, then stop the bfs and collect the beam states 

        current_level_nodes = 0
        current_level = [initial_state]
        end_solution = []

        while current_level_nodes < self.beam_width: 
            current_level_nodes = 0
            next_level = []
            for state in current_level:
                # Get list of swap candidates 
                swap_candidates = self._obtain_swaps(state["front_layer"], state["layout"])
                current_level_nodes += len(swap_candidates)

                # Each swap represents a trial new state
                for swap in swap_candidates: 
                    # Make a copy of the layout and apply swap to the copy
                    trial_layout = state["layout"].copy()
                    trial_layout.swap(*swap)

                    # Make a copy of the dag and apply swap to the copy
                    # Note since dag does not have copy, we have to use deepcopy
                    trial_dag = deepcopy(state["mapped_dag"])
                    self._apply_gate(
                        trial_dag, 
                        DAGOpNode(op=CXGate(), qargs=swap),
                        trial_layout,
                        self.dag.qregs["q"],
                    )
                    self._apply_gate(
                        trial_dag,
                        DAGOpNode(op=CXGate(), qargs=(swap[1], swap[0])),
                        trial_layout,
                        self.dag.qregs["q"],
                    )
                    self._apply_gate(
                        trial_dag,
                        DAGOpNode(op=CXGate(), qargs=swap),
                        trial_layout,
                        self.dag.qregs["q"],
                    )

                    # Make a copy of the front layer and predecessors
                    trial_front_layer = state["front_layer"].copy() 
                    trial_predecessors = state["predecessors"].copy()

                    # Create trial state and update it until no more gates can be applied
                    trial_state = {
                        "mapped_dag": trial_dag,
                        "layout": trial_layout,
                        "front_layer": trial_front_layer,
                        "predecessors": trial_predecessors,
                        "gates_done": state["gates_done"]
                    }

                    self._update_state(trial_state)
                    if not trial_state["front_layer"]:
                        end_solution.append(trial_state)
                    next_level.append(trial_state)
            current_level = next_level
        # add end solutions to the current level
        current_level.extend(end_solution)
        # organize current_level by first the number of gates done, then by mapped_dag depth
        current_level.sort(key=lambda x: (-x["gates_done"], x["mapped_dag"].depth()))
        # prune the current_level to only include the beam_width number of states
        current_level = current_level[:self.beam_width]
        return current_level

    def _find_and_apply_swaps(self, state):
        """ Finds the best swap and applies it to the mapped_dag and current layout.
        
        Args:
            state (Dict): the current state of the circuit mapping. 
        Returns:
            None
        """
        mapped_dag = state["mapped_dag"]
        current_layout = state["layout"]
        front_layer = state["front_layer"]

        swap_scores = {}
        for swap_qubits in self._obtain_swaps(front_layer, current_layout):
            trial_layout = current_layout.copy()
            trial_layout.swap(*swap_qubits)
            score = self._score_heuristic(
                front_layer, trial_layout 
            )
            swap_scores[swap_qubits] = score
        min_score = min(swap_scores.values())
        best_swaps = [k for k, v in swap_scores.items() if v == min_score]
        best_swaps.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
        best_swap = self.rng.choice(best_swaps)

        # instead of applying the swap gate, we will apply 3 cnot gates to simulate the swap gate
        self._apply_gate(
            mapped_dag,
            DAGOpNode(op=CXGate(), qargs=best_swap),
            current_layout,
            self.dag.qregs["q"],
        )
        self._apply_gate(
            mapped_dag,
            DAGOpNode(op=CXGate(), qargs=(best_swap[1], best_swap[0])),
            current_layout,
            self.dag.qregs["q"],
        )
        self._apply_gate(
            mapped_dag,
            DAGOpNode(op=CXGate(), qargs=best_swap),
            current_layout,
            self.dag.qregs["q"],
        )

        current_layout.swap(*best_swap)
        state["layout"] = current_layout
        state["front_layer"] = front_layer


    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)

    def _build_required_predecessors(self, dag):
        out = defaultdict(int)
        # We don't need to count in- or out-wires: outs can never be predecessors, and all input
        # wires are automatically satisfied at the start.
        for node in dag.op_nodes():
            for successor in self._successors(node, dag):
                out[successor] += 1
        return out

    def _successors(self, node, dag):
        """Return an iterable of the successors along each wire from the given node.

        This yields the same successor multiple times if there are parallel wires (e.g. two adjacent
        operations that have one clbit and qubit in common), which is important in the swapping
        algorithm for detecting if each wire has been accounted for."""
        for _, successor, _ in dag.edges(node):
            if isinstance(successor, DAGOpNode):
                yield successor

    def _is_resolved(self, node):
        """Return True if all of a node's predecessors in dag are applied."""
        return self.required_predecessors[node] == 0


    def _obtain_swaps(self, front_layer, current_layout):
        """Return a set of candidate swaps that affect qubits in front_layer.

        For each virtual qubit in front_layer, find its current location
        on hardware and the physical qubits in that neighborhood. Every SWAP
        on virtual qubits that corresponds to one of those physical couplings
        is a candidate SWAP.

        Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
        """
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted([virtual, virtual_neighbor], key=lambda q: self._bit_indices[q])
                    candidate_swaps.add(tuple(swap))
        return candidate_swaps

    def _add_greedy_swaps(self, front_layer, dag, layout, qubits):
        """Mutate ``dag`` and ``layout`` by applying greedy swaps to ensure that at least one gate
        can be routed."""
        layout_map = layout._v2p
        target_node = min(
            front_layer,
            key=lambda node: self.dist_matrix[layout_map[node.qargs[0]], layout_map[node.qargs[1]]],
        )
        for pair in _shortest_swap_path(tuple(target_node.qargs), self.coupling_map, layout):
            self._apply_gate(dag, DAGOpNode(op=SwapGate(), qargs=pair), layout, qubits)
            layout.swap(*pair)

    def _compute_cost(self, layer, layout):
        cost = 0
        layout_map = layout._v2p
        for node in layer:
            cost += self.dist_matrix[layout_map[node.qargs[0]], layout_map[node.qargs[1]]]
        return cost

    def _score_heuristic(self, front_layer, layout):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        return self._compute_cost(front_layer, layout)

    def _undo_operations(self, operations, dag, layout):
        """Mutate ``dag`` and ``layout`` by undoing the swap gates listed in ``operations``."""
        if dag is None:
            for operation in reversed(operations):
                layout.swap(*operation.qargs)
        else:
            for operation in reversed(operations):
                dag.remove_op_node(operation)
                p0 = self._bit_indices[operation.qargs[0]]
                p1 = self._bit_indices[operation.qargs[1]]
                layout.swap(p0, p1)


def _transform_gate_for_layout(op_node, layout, device_qreg):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = copy(op_node)
    mapped_op_node.qargs = tuple(device_qreg[layout._v2p[x]] for x in op_node.qargs)
    return mapped_op_node


def _shortest_swap_path(target_qubits, coupling_map, layout):
    """Return an iterator that yields the swaps between virtual qubits needed to bring the two
    virtual qubits in ``target_qubits`` together in the coupling map."""
    v_start, v_goal = target_qubits
    start, goal = layout._v2p[v_start], layout._v2p[v_goal]
    # TODO: remove the list call once using retworkx 0.12, as the return value can be sliced.
    path = list(retworkx.dijkstra_shortest_paths(coupling_map.graph, start, target=goal)[goal])
    # Swap both qubits towards the "centre" (as opposed to applying the same swaps to one) to
    # parallelise and reduce depth.
    split = len(path) // 2
    forwards, backwards = path[1:split], reversed(path[split:-1])
    for swap in forwards:
        yield v_start, layout._p2v[swap]
    for swap in backwards:
        yield v_goal, layout._p2v[swap]