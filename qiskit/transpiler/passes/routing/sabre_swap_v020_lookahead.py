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
import itertools

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
        heuristic="lookahead",
        seed=None,
        fake_run=False,
        lookahead_depth=0,
        beam_width=5,
        alpha=1,
        beta=1,
        charlie=1,
    ):
        r"""SabreSwap initializer.

        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic').
            seed (int): random seed used to tie-break among candidate swaps.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.

        Additional Information:

            The search space of possible SWAPs on physical qubits is explored
            by assigning a score to the layout that would result from each SWAP.
            The goodness of a layout is evaluated based on how viable it makes
            the remaining virtual gates that must be applied. A few heuristic
            cost functions are supported

            - 'basic':

            The sum of distances for corresponding physical qubits of
            interacting virtual qubits in the front_layer.

            .. math::

                H_{basic} = \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]
        """

        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if coupling_map is None or coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()

        self.heuristic = heuristic
        self.seed = seed
        self.fake_run = fake_run
        self.required_predecessors = None
        self._bit_indices = None
        self.dist_matrix = None
        self.lookahead_depth = lookahead_depth
        self.gates_committed = []
        self.beam_width = beam_width
        self.gates_found = set()
        self.alpha = alpha
        self.beta = beta
        self.charlie = charlie

    def run(self, dag):
        self.gates_committed = []
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

        self.gates_committed = []
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

            # After all free gates are exhausted, initialize BFS to perform lookahead exploration
            # BFS tree will be represented by the queue
            queue = [(front_layer, current_layout, [], 0, self.required_predecessors, self.gates_committed, [])] # (front_layer, current_layout, sequence of swaps, depth)
            best_swap_sequences = None
            best_score = float("inf")
            best_score_depth = float("inf")
            
            # Start BFS lookahead
            swap_scores = {}
            previous_depth = calculate_circuit_depth(self.gates_committed)
            self.gates_found = set()
            while queue:
                #print("--------------------")
                # represents the current node in the tree
                queue_front_layer, queue_layout, swap_sequence, depth, successors, explored_gates, gates = queue.pop(0)
                # print all items of queue
                qubit_indices = []
                for node in queue_front_layer:
                    qubit_indices.append(tuple(qubit.index for qubit in node.qargs))
                #index_order = []
                #for index in queue_layout.get_virtual_bits().keys():
                #    index_order.append(index.index)
                #print("Current queue front layer: ", qubit_indices) 
                #print("Current queue layout: ", index_order)
                #print("Current queue swap sequence: ", swap_sequence)
                #print("Current queue depth: ", depth)
                #print("Current queue explored gates: ", explored_gates)

                if depth <= self.lookahead_depth:
                    # obtaining the swaps for the current lookahead layer
                    swap_candidates = list(self._obtain_swaps(queue_front_layer, queue_layout))
                    swap_candidates.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))

                    # beam search to prune the number of swap candidates
                    scored_swaps = []
                    for swap_qubits in swap_candidates:
                        trial_layout = queue_layout.copy()
                        trial_layout.swap(*swap_qubits)
                        score = self._score_heuristic(queue_front_layer, trial_layout)
                        scored_swaps.append((score, swap_qubits))
                    
                    # sort the scored swaps by score (ascending)
                    scored_swaps.sort(key=lambda x: x[0])

                    for _, swap_qubits in scored_swaps[:self.beam_width]:
                        # for each exploration, we need a new copy of:
                        # layout, successors, swap_sequence, explored_gates, front_layer, explored_gates, gates
                        trial_layout = queue_layout.copy()
                        trial_layout.swap(*swap_qubits)
                        trial_explored_gates = explored_gates.copy()
                        trial_explored_gates.append(self._fake_apply_gate(DAGOpNode(op=SwapGate(), qargs=swap_qubits),
                                                                          trial_layout, canonical_register))


                        trial_swap_sequence = swap_sequence + [swap_qubits]
                        trial_successors = successors.copy()
                        trial_explored_gates = explored_gates.copy()
                        trial_gates = gates.copy()

                        trial_front_layer = []
                        execute_gate_list = []

                        # update front layer with gates that now can be executed
                        for node in queue_front_layer:
                            # checks only two qubit gates
                            if len(node.qargs) == 2:
                                v0, v1 = node.qargs
                                if self.coupling_map.graph.has_edge(
                                    trial_layout._v2p[v0], trial_layout._v2p[v1]
                                ):
                                    execute_gate_list.append(node)
                                else:
                                    trial_front_layer.append(node)
                        
                        # execute any excutable gates, and update front layer with the successor gates
                        if execute_gate_list:
                            for node in execute_gate_list:
                                trial_gates.append(node)
                                self.gates_found.add(node)
                                trial_explored_gates.append(self._fake_apply_gate(node, trial_layout, canonical_register))
                                for successor in self._successors(node, dag): # may need to check this out later
                                    trial_successors[successor] -= 1 
                                    if trial_successors[node] == 0:
                                        trial_front_layer.append(successor)

                        queue.append((trial_front_layer, trial_layout, trial_swap_sequence, 
                                      depth + 1, trial_successors, trial_explored_gates, trial_gates))
                else:
                    # Reached lookahead depth, score this seqeuence
                    # Calculate score depth
                    current_depth = calculate_circuit_depth(explored_gates)
                    score_depth = current_depth - previous_depth

                    # Calculate lookahead score
                    gates_to_apply = []
                    for gate in self.gates_found:
                        if gate not in gates:
                            gates_to_apply.append(gate)
                    score_lookahead = self._compute_cost(gates_to_apply, queue_layout)

                    # Calculate front score
                    normalization_factor = len(queue_front_layer) 
                    if normalization_factor == 0:
                        score_front = 0
                    else:
                        score_front = self._compute_cost(queue_front_layer, queue_layout) / normalization_factor
                    #score_front = 0
                    #score_lookahead = 0
                    score = (score_depth * self.alpha) + (score_lookahead * self.beta) + (score_front * self.charlie)
                    if score < best_score:
                        best_score = score
                        best_swap_sequences = [swap_sequence]
                    elif score == best_score:
                        best_swap_sequences.append(swap_sequence)
                    # Continue loop to explore all squences at the current depth
            # Apply only the first swap of the best sequence found after lookahead 
            if best_swap_sequences is not None:
                best_swap_sequence = rng.choice(best_swap_sequences)
                first_swap = best_swap_sequence[0]
                swap_node = self._apply_gate(
                    mapped_dag,
                    DAGOpNode(op=SwapGate(), qargs=first_swap),
                    current_layout,
                    canonical_register,
                )
                current_layout.swap(*first_swap)
                ops_since_progress.append(swap_node)
                # If the front_layer is empty, the circuit is done, and we can apply the whole sequence
            if not front_layer:
                for swap_qubits in best_swap_sequences[1:]:  # Apply the rest of the sequence
                    swap_node = self._apply_gate(
                        mapped_dag,
                        DAGOpNode(op=SwapGate(), qargs=swap_qubits),
                        current_layout,
                        canonical_register,
                    )
                    current_layout.swap(*swap_qubits)
        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return dag

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        #print("     Executing node:" , new_node.qargs[0].index, new_node.qargs[1].index)
        if len(new_node.qargs) == 2:
            # if node is a swap, then need to add it 3 times, so 1 swap is 3 gates
            if new_node.name == "swap":
                self.gates_committed.append((new_node.qargs[0].index, new_node.qargs[1].index))
                self.gates_committed.append((new_node.qargs[0].index, new_node.qargs[1].index))
            self.gates_committed.append((new_node.qargs[0].index, new_node.qargs[1].index))
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
    
    def _fake_apply_gate(self, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        # if node is a swap, then need to add it 3 times, so 1 swap is 3 gates
        if new_node.name == "swap":
            self.gates_committed.append((new_node.qargs[0].index, new_node.qargs[1].index))
            self.gates_committed.append((new_node.qargs[0].index, new_node.qargs[1].index))
        # assume that input is a two qubit gate
        return (new_node.qargs[0].index, new_node.qargs[1].index)

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

def calculate_circuit_depth(gates):
    # Get all unique qubits in the circuit
    nodes = set(itertools.chain(*gates))
    
    # Initialize depth 0 for each qubit
    depths = {node: 0 for node in nodes} 

    for gate in gates:
        
        # Get the max depth of the two qubits in this gate
        depth = max(depths[gate[0]], depths[gate[1]])

        # Increment the depth by 1
        depth += 1

        # Update the depth for both qubits
        depths[gate[0]] = depth  
        depths[gate[1]] = depth
        
    # Return the max depth overall 
    if depths:
        return max(depths.values())
    else:
        return 0