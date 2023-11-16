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
import itertools
import random

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
        lookahead_depth=0,
        weight_front=1,
        weight_depth=1,
        weight_looka=1,
        weight_gates=1,
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
                H_{basic} = \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]
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
        self.lookahead_depth = lookahead_depth # amount of more swaps to explore
        self.gates_order = [] # list of all 2-qubit gates committed in order, depth of the circuit.decompose(["swap"])
        self.gates_explored = set() # keep track of all gates explored in the lookahead branching
        self.found_end = False # flag to indicate if we have found an end solution
        self.end_gates_info = [] # list of all end solutions found, with their depth and sequence of gates
        random.seed(self.seed)
        self.weight_front = weight_front
        self.weight_depth = weight_depth
        self.weight_looka = weight_looka
        self.weight_gates = weight_gates

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

        # resetting variables for a new run
        self.gates_order = []
        self.gates_explored = set()
        self.found_end = False
        self.end_gates_info = []

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

            # Remove as many immediately applicable gates as possible, and update front layer
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

            # Used to detect infinite loops
            if not execute_gate_list and len(ops_since_progress) > max_iterations_without_progress:
                # Backtrack to the last time we made progress, then greedily insert swaps to route
                # the gate with the smallest distance between its arguments.  This is a release
                # valve for the algorithm to avoid infinite loops only, and should generally not
                # come into play for most circuits.
                self._undo_operations(ops_since_progress, mapped_dag, current_layout)
                self._add_greedy_swaps(front_layer, mapped_dag, current_layout, canonical_register)
                continue

            # If there are gates to apply, do so and update the front layer, and return to top of loop
            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(mapped_dag, node, current_layout, canonical_register)
                    for successor in self._successors(node, dag):
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)


                ops_since_progress = []
                continue

            # After all free gates are exhausted, initialize BFS queue to perform lookahead exploration.
            # We use a BFS queue to explore the search space of SWAPs. 
            # In the queue, we store the following:
            # (front_layer, current_layout, swap_sequence, predecessors, gates_to_execute, gate_order,
            #  score_front, gates_to_execute, all_gates, depth)
            queue = [(front_layer, current_layout, [], self.required_predecessors, self.gates_order,
                      float("inf"), [], [], 0)] 
            
            # start of new exploration, so need resets
            best_swap_sequences = None # used to get the first swap, can be adjusted later to reduce time
            self.gates_explored = set() 
            prev_depth = calculate_circuit_depth(self.gates_order) # used to measure change in depth

            # setting scores to worse than any possible score
            best_score = float("inf")
            while queue:
                # length of gates_to_execute is the gate score (does not contain swaps)
                # all_gate contains all of the gates in DAGOpNode form
                q_front_layer, q_current_layout, q_swap_sequence, predecessors, gate_order, \
                    score_front, gates_to_execute, all_gates, depth = queue.pop(0)

                # exploring all swap candidates at this depth and then adding the next layer to the queue
                if depth <= self.lookahead_depth:
                    swap_candidates = list(self._obtain_swaps(front_layer, current_layout))
                    # sorting so that we always get the same order of swaps, so there is no randomness from order
                    swap_candidates.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))

                    for swap in swap_candidates:
                        # need copies so that we don't mutate the original objects
                        trial_layout = q_current_layout.copy()
                        trial_layout.swap(*swap)
                        trial_front_layer = q_front_layer.copy()
                        trial_predecessors = predecessors.copy() 
                        trial_gates_to_execute = gates_to_execute.copy()
                        trial_all_gates = all_gates.copy()

                        # adding gates
                        trial_all_gates.append(DAGOpNode(op=SwapGate(), qargs=swap))
                        trial_gate_order = gate_order + self._fake_apply_gate(
                            DAGOpNode(op=SwapGate(), qargs=swap), trial_layout, canonical_register
                        )
                        trial_swap_sequence = q_swap_sequence + [swap]

                        trial_score_front = score_front
                        if depth == 0:
                            # only getting the front score at the initial step
                            trial_score_front = self._score_heuristic(trial_front_layer, trial_layout)
                        
                        # changing front_layer to reflect the swap
                        while True: # note may need to think about single-qubit gates later
                            execute_gate_list = []
                            new_front_layer = []
                            for node in trial_front_layer:
                                if len(node.qargs) == 2:
                                    v0, v1 = node.qargs
                                    if self.coupling_map.graph.has_edge(
                                        trial_layout._v2p[v0], trial_layout._v2p[v1]
                                    ):
                                        execute_gate_list.append(node)
                                    else:
                                        new_front_layer.append(node) 
                            trial_front_layer = new_front_layer
                            if not execute_gate_list: # done with updating front layer
                                break
                            else:
                                # update front layer with successors and fake apply gates
                                for node in execute_gate_list:
                                    trial_gate_order += self._fake_apply_gate(node, trial_layout, canonical_register)
                                    trial_gates_to_execute.append(node)
                                    self.gates_explored.add(node)

                                    trial_all_gates.append(node)
                                    for successor in self._successors(node, dag):
                                        trial_predecessors[successor] -= 1
                                        if trial_predecessors[successor] == 0:
                                            trial_front_layer.append(successor)

                         # reached a potential point of end of the lookahead
                        if trial_front_layer == []:
                            self.found_end = True
                            curr_depth = calculate_circuit_depth(trial_gate_order)
                            self.end_gates_info.append({"sequence": trial_all_gates, "depth": curr_depth - prev_depth})

                        queue.append((trial_front_layer, trial_layout, trial_swap_sequence, trial_predecessors, trial_gate_order,
                                      trial_score_front, trial_gates_to_execute, trial_all_gates, depth + 1))
                # reached the end of the lookahead, now we score what we have
                else:
                    # calculate lookahead score
                    curr_depth = calculate_circuit_depth(gate_order)
                    score_depth = curr_depth - prev_depth

                    # calculate gate score
                    score_gates = len(gates_to_execute)

                    # calculate lookahead score
                    gates_remaining = []
                    for gate in self.gates_explored:
                        if gate not in gates_to_execute:
                            gates_remaining.append(gate)
                    score_looka = self._compute_cost(gates_remaining, q_current_layout)
                    
                    # considering weightings
                    score_front *= self.weight_front
                    score_depth *= self.weight_depth
                    score_looka *= self.weight_looka
                    score_gates *= self.weight_gates
                    
                    score = score_front + score_depth + score_looka + score_gates
                    # if the score is better than the current best, then update the best
                    if score < best_score:
                        best_score = score
                        best_swap_sequences = [q_swap_sequence]
                    elif score == best_score:
                        best_swap_sequences.append(q_swap_sequence)
                    
            # we have found the end solution from at least one of the branch, now 
            # we pick the one with the lowest depth and apply it 
            if self.found_end:
                min_depth = min(item['depth'] for item in self.end_gates_info)

                # find all sequences with the minimum depth
                lowest_depth_sequences = [item['sequence'] for item in self.end_gates_info if item['depth'] == min_depth]
                sequence = random.choice(lowest_depth_sequences)

                for gate in sequence:
                    self._apply_gate(mapped_dag, gate, current_layout, canonical_register)
                    if gate.name == "swap": # need to change the layout when swapping
                        current_layout.swap(*gate.qargs) 
                break # done with routing

            # no end solution found, so we apply the best swap sequence
            if best_swap_sequences is not None:
                # randomly choose one of the best swap sequences
                best_swap_sequence = rng.choice(best_swap_sequences)
                # apply and commit to the first swap in the sequence
                first_swap = best_swap_sequence[0]
                swap_node = self._apply_gate(
                    mapped_dag,
                    DAGOpNode(op=SwapGate(), qargs=first_swap),
                    current_layout,
                    canonical_register,
                )
                current_layout.swap(*first_swap)
                ops_since_progress.append(swap_node)
            else:
                raise TranspilerError("No valid swap sequence found")

        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return dag

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        if len(new_node.qargs) == 2:
            # may need to change this later to account for single-qubit gates
            # if node is a swap, then need to add it 3 times to represent how 1 swap is 3 gates
            if new_node.name == "swap":
                self.gates_order.append((new_node.qargs[0].index, new_node.qargs[1].index))
                self.gates_order.append((new_node.qargs[0].index, new_node.qargs[1].index))
            self.gates_order.append((new_node.qargs[0].index, new_node.qargs[1].index))
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)

    def _fake_apply_gate(self, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        # if node is a swap, then need to add it 3 times, so 1 swap is 3 gates
        if new_node.name == "swap":
            return [(new_node.qargs[0].index, new_node.qargs[1].index),
                    (new_node.qargs[0].index, new_node.qargs[1].index),
                    (new_node.qargs[0].index, new_node.qargs[1].index)]
        else: 
            return [(new_node.qargs[0].index, new_node.qargs[1].index)]

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
        if len(front_layer) == 0:
            return 0
        normalization = len(front_layer)
        return self._compute_cost(front_layer, layout) / normalization

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