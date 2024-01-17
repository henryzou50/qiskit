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

from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import retworkx

from qiskit.circuit.library.standard_gates import SwapGate
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
        lookahead_steps=1,
        beam_width=1
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
        self.qubits_depth = None
        self._bit_indices = None
        self.dist_matrix = None
        self.lookahead_steps = lookahead_steps
        self.beam_width = beam_width
        self.end_solution_found = False

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
        self.end_solution_found = False
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        max_iterations_without_progress = 10 * len(dag.qubits)  # Arbitrary.
        ops_since_progress = []

        self.dist_matrix = self.coupling_map.distance_matrix

        rng = np.random.default_rng(self.seed)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None
        if not self.fake_run:
            mapped_dag = dag.copy_empty_like()

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)

        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}

        # A depth tracker for each qubit
        self.qubits_depth = dict.fromkeys(dag.qubits, 0)

        # Start algorithm from the front layer and iterate until all gates done.
        self.required_predecessors = self._build_required_predecessors(dag)
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

            if not execute_gate_list and len(ops_since_progress) > max_iterations_without_progress:
                # Backtrack to the last time we made progress, then greedily insert swaps to route
                # the gate with the smallest distance between its arguments.  This is a release
                # valve for the algorithm to avoid infinite loops only, and should generally not
                # come into play for most circuits.
                self._undo_operations(ops_since_progress, mapped_dag, current_layout)
                self._add_greedy_swaps(front_layer, mapped_dag, current_layout, canonical_register)
                continue

            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(mapped_dag, node, current_layout, canonical_register)
                    for successor in self._successors(node, dag):
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)

                ops_since_progress = []
                continue

            # Start bfs search for lookahead
            initial_node = Node(current_layout, front_layer, self.required_predecessors, 
                                self.qubits_depth, [])
            
            # get the first node in the lookahead search
            gate_seq  = self.lookahead_search(initial_node, dag).gate_seq
            
            # if end solution is found, loop to apply all of the gates to the mapped_dag
            if self.end_solution_found:
                for gate in gate_seq:
                    # if gate is a swap, we need to update the layout
                    if gate.name == "swap":
                        current_layout.swap(*gate.qargs)
                    self._apply_gate(mapped_dag, gate, current_layout, canonical_register)
                break

            best_swap = gate_seq[0].qargs
            swap_node = self._apply_gate(
                mapped_dag,
                DAGOpNode(op=SwapGate(), qargs=best_swap),
                current_layout,
                canonical_register,
            )
            current_layout.swap(*best_swap)
            ops_since_progress.append(swap_node)
        
        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return dag
    
    def lookahead_search(self, initial_node, dag):
        """ Performs a breadth-first search of the swap exploration to find the best swap. 
        The length of the swap sequence explore is equal to the length of self.lookahead_steps.

        Args:
            initial_node (Node): The initial node to start the search from. Includes:
                layout (Layout): The current layout of the circuit.
                front_layer (list): The current front layer of the circuit.
                successors (dict): The current successors of the circuit.
                qubit_depth (dict): The current depth of the qubits.
                gate_seq (list): The current gate sequence (excluding swaps).
                swap_seq (list): The current swap sequence.
        Returns:
            Node: The node with the best score.
        """
        all_gate_seq = set() # exclude swaps
        end_solutions = []
        current_level = [initial_node]
        for i in range(self.lookahead_steps):
            next_level = []
            for node in current_level:
                if node.front_layer == []: # already added, avoid double counting
                    continue
                # Find the swap candidates for this node's front layer and current_layout
                swap_candidates = list(self._obtain_swaps(node.front_layer, node.layout))
                # sorting so that we always get the same order of swaps, so there is no randomness from order
                swap_candidates.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
                for swap_qubits in swap_candidates:
                    # Create a new layout and apply the swap
                    trial_layout = node.layout.copy()
                    trial_layout.swap(*swap_qubits)

                    # Update score front
                    if i == 0:
                        trial_score_front = self._score_heuristic(node.front_layer, trial_layout)
                    else:
                        trial_score_front = node.score_front

                    # Create a new swap sequence and add the swap
                    swap = DAGOpNode(op=SwapGate(), qargs=swap_qubits)
                    trial_gate_seq    = node.gate_seq + [swap]

                    # Create variables that need to be updated when applying gates
                    trial_successors  = node.successors.copy()
                    trial_front_layer = node.front_layer.copy()
                    trial_qubit_depth = node.qubit_depth.copy()

                    # Update depth with the swap
                    trial_qubit_depth = self._update_qubit_depth(swap, trial_qubit_depth)

                    while True: # continue to update until no more gates can be applied
                        new_front_layer = []
                        execute_gate_list = []
                        for gate in trial_front_layer:
                            if len(gate.qargs) == 2:
                                v0, v1 = gate.qargs
                                if self.coupling_map.graph.has_edge(
                                    trial_layout._v2p[v0], trial_layout._v2p[v1]
                                ):
                                    execute_gate_list.append(gate)
                                else:
                                    new_front_layer.append(gate) # later fix for single and barrier
                        trial_front_layer = new_front_layer
                        if not execute_gate_list: # no more gates can be applied
                            break
                        else:
                            # Apply the gates that can be applied
                            for gate in execute_gate_list:
                                trial_gate_seq.append(gate)
                                all_gate_seq.add(gate)
                                trial_qubit_depth = self._update_qubit_depth(gate, trial_qubit_depth)
                                
                                for successor in self._successors(gate, dag):
                                    trial_successors[successor] -= 1
                                    if trial_successors[successor] == 0:
                                        trial_front_layer.append(successor)

                    trial_node = Node(trial_layout, trial_front_layer, trial_successors, 
                                    trial_qubit_depth, trial_gate_seq)
                    trial_node.score_front = trial_score_front
                    trial_node.score_depth = max(trial_node.qubit_depth.values())
                    # i represents number of swaps, 
                    # so the number of gates is i - len(trial_node.gate_seq)
                    # the more gates done, the more negative the score_gates
                    trial_node.score_gates = i - len(trial_node.gate_seq) 
                    
                    # Reached a potential point of end of the lookahead
                    if trial_front_layer == []:
                        end_solutions.append(trial_node)
                        # end of this branch, don't add to next level
                        continue
                    next_level.append(trial_node)
                    
                # Update the lookahead score for each node 
                for node in next_level:
                    gates_remaining = []
                    for gate in all_gate_seq:
                        if gate not in node.gate_seq:
                            gates_remaining.append(gate)
                    node.score_looka = self._compute_cost(gates_remaining, node.layout)
                    node.score_total_1 = node.score_front + node.score_looka 
                    node.score_total_2 = node.score_depth + node.score_gates

            # Sort the next level by score first, then by depth, if equal, then random tie-break
            # print each node's score front with their score_depth
            next_level.sort(key=lambda x: (x.score_total_1, x.score_total_2))
            current_level = next_level[:self.beam_width]
        
        # If there is an end solution, return the one with the best score
        if end_solutions:
            end_solutions.sort(key=lambda x: (x.score_depth))
            self.end_solution_found = True
            return end_solutions[0]

        return current_level[0]

    def _update_qubit_depth(self, node, qubit_depth):
        """Update the depth of the qubits after applying a gate.

        Args:
            node (DAGNode): The node to apply.
            current_layout (Layout): The current layout of the circuit.
            qubit_depth (dict): The current depth of the qubits.
        Returns:
            dict: The updated depth of the qubits.
        """
        if len(node.qargs) == 2:
            depth = max(qubit_depth[node.qargs[0]], qubit_depth[node.qargs[1]]) + 1
            if node.name == "swap": # Treat swaps as 3 CNOTS
                depth += 2
            qubit_depth[node.qargs[0]] = depth
            qubit_depth[node.qargs[1]] = depth
        return qubit_depth
        


    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)

        # Updates the depth of the wires, only considers 2 qubit gates
        if len(node.qargs) == 2:
            depth = max(self.qubits_depth[node.qargs[0]], self.qubits_depth[node.qargs[1]]) + 1
            if node.name == "swap": # Treat swaps as 3 CNOTS
                depth += 2
            self.qubits_depth[node.qargs[0]] = depth
            self.qubits_depth[node.qargs[1]] = depth

        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)

    def _reset_qubits_depth(self):
        """ Reset all qubit depthto 0 upon request """
        self.qubits_depth = {k: 0 for k in self.qubits_depth.keys()}

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

    def _obtain_extended_set(self, dag, front_layer):
        """Populate extended_set by looking ahead a fixed number of gates.
        For each existing element add a successor until reaching limit.
        """
        extended_set = []
        decremented = []
        tmp_front_layer = front_layer
        done = False
        while tmp_front_layer and not done:
            new_tmp_front_layer = []
            for node in tmp_front_layer:
                for successor in self._successors(node, dag):
                    decremented.append(successor)
                    self.required_predecessors[successor] -= 1
                    if self._is_resolved(successor):
                        new_tmp_front_layer.append(successor)
                        if len(successor.qargs) == 2:
                            extended_set.append(successor)
                if len(extended_set) >= 20:
                    done = True
                    break
            tmp_front_layer = new_tmp_front_layer
        for node in decremented:
            self.required_predecessors[node] += 1
        return extended_set

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

        # normalize by the number of gates
        if len(layer) == 0:
            return 0
        cost /= len(layer)
        cost *= 3
        return cost

    def _score_heuristic(self, front_layer, layout):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        first_cost = self._compute_cost(front_layer, layout) / len (front_layer)

        return first_cost

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

class Node():
    def __init__(self, layout, front_layer, successors, qubit_depth, gate_seq):
        self.layout        = layout # layout of the circuit
        self.front_layer   = front_layer # front layer of the circuit
        self.successors    = successors # successors of the front layer
        self.qubit_depth   = qubit_depth # depth of each qubit
        self.gate_seq      = gate_seq # sequence of all of the gates applied
        self.score_front   = 0
        self.score_depth   = 0
        self.score_looka   = 0
        self.score_gates   = 0
        self.score_total_1 = 0
        self.score_total_2 = 0