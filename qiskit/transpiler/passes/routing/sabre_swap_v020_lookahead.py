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
        seed=None,
        fake_run=False,
    ):
        r"""SabreSwap initializer.

        Args:
            coupling_map (CouplingMap): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic' or 'lookahead' or 'decay').
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
        self.qubits_decay = None
        self._bit_indices = None
        self.dist_matrix = None
        # carries the list of all the two-qubit gates commited in order, swaps are counted 3 times
        # used to get the depth of the circuit.decompose(["swap"])
        self.gates_depth = [] 

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
    
        max_iterations_without_progress = 10 * len(dag.qubits)  # Arbitrary.
        ops_since_progress = []
        self.gates_depth = []

        # Normally this isn't necessary, but here we want to log some objects that have some
        # non-trivial cost to create.

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

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            swap_scores_front = {}
            swap_scores_depth = {}
            swap_scores_gates = {}
            #swap_scores_indices = {}
            prev_depth = calculate_circuit_depth(self.gates_depth)

            swap_candidates = list(self._obtain_swaps(front_layer, current_layout))
            swap_candidates.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))

            for swap_qubits in swap_candidates:
                #print("   Evaluating swap: ", swap_qubits[0].index, swap_qubits[1].index)

                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)

                # calculating front score
                score_front = self._score_heuristic(
                    front_layer, trial_layout
                )

                # Need copy of front layer to update it for the current swap
                trial_front_layer = front_layer.copy()
                # Need copy of successors to update it for the current swap
                trial_required_predecessors = self.required_predecessors.copy()
                # Need copy of current committed gates
                trial_gates_depth = self.gates_depth + self._fake_apply_gate(
                    DAGOpNode(op=SwapGate(), qargs=swap_qubits),
                    trial_layout,
                    canonical_register
                )

                # Need to reset the count of gates executed for each branch
                count_gate_executed = 0

                # In process of updating front layer
                while True:
                    new_front_layer = []
                    execute_gate_list = []
                    for node in trial_front_layer:
                        if len(node.qargs) == 2:
                            v0, v1 = node.qargs
                            if self.coupling_map.graph.has_edge(
                                trial_layout._v2p[v0], trial_layout._v2p[v1]
                            ):
                                execute_gate_list.append(node)
                            else:
                                new_front_layer.append(node)
                        else:  # Single-qubit gates as well as barriers are free
                            execute_gate_list.append(node)
                    trial_front_layer = new_front_layer

                    # Done with updating front layer
                    if not execute_gate_list:
                        break
                    else:
                        # Update front layer with successors and fake apply gates
                        for node in execute_gate_list:
                            trial_gates_depth += self._fake_apply_gate(node, trial_layout, canonical_register)
                            count_gate_executed += 1
                            for successor in self._successors(node, dag):
                                trial_required_predecessors[successor] -= 1
                                if trial_required_predecessors[successor] == 0:
                                    trial_front_layer.append(successor)
                # calculating depth score
                curr_depth = calculate_circuit_depth(trial_gates_depth)
                delta_depth = curr_depth - prev_depth   
                
                swap_scores_front[swap_qubits] = score_front
                swap_scores_depth[swap_qubits] = delta_depth
                swap_scores_gates[swap_qubits] = count_gate_executed
                #swap_scores_indices[swap_qubits] = (swap_qubits[0].index, swap_qubits[1].index)
            # print out all infomation from swap_scores, swap_scores_depth, swap_scores_gates
            
            #for i in range(len(swap_scores_front)):
            #    print(f"score: {list(swap_scores_front.values())[i]}, depth: {list(swap_scores_depth.values())[i]}, gates: {list(swap_scores_gates.values())[i]}, indices: {list(swap_scores_indices.values())[i]}")
            sorted_swaps = sorted(swap_scores_front.keys(), 
                      key=lambda x: (-swap_scores_gates[x], 
                                     swap_scores_depth[x],
                                     swap_scores_front[x]))
            #sorted_swaps = sorted(swap_scores_front.keys(), 
            #          key=lambda x: (-swap_scores_gates[x], 
            #                         swap_scores_depth[x],
            #                         swap_scores_front[x]))

            sorted_swaps = sorted(swap_scores_front.keys(), 
                      key=lambda x: (-swap_scores_gates[x], 
                                     swap_scores_front[x],
                                     swap_scores_depth[x]))


            #for swaps in sorted_swaps:
            #    print("sorted swaps: ", swaps[0].index, swaps[1].index)

            #min_score = min(swap_scores_front.values())
            #best_swaps = [k for k, v in swap_scores_front.items() if v == min_score]
            #best_swaps.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
            #best_swap = rng.choice(best_swaps)

            best_swaps = [s for s in sorted_swaps if 
              swap_scores_gates[s] == swap_scores_gates[sorted_swaps[0]] and
              swap_scores_depth[s] == swap_scores_depth[sorted_swaps[0]] and
              swap_scores_front[s] == swap_scores_front[sorted_swaps[0]]]

            best_swap = rng.choice(best_swaps)
            swap_node = self._apply_gate(
                mapped_dag,
                DAGOpNode(op=SwapGate(), qargs=best_swap),
                current_layout,
                canonical_register,
            )
            current_layout.swap(*best_swap)
            ops_since_progress.append(swap_node)


        self.property_set["final_layout"] = current_layout
        #print("Final depth of circuit: ", calculate_circuit_depth(self.gates_depth))
        if not self.fake_run:
            return mapped_dag
        return dag

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        if len(new_node.qargs) == 2:
            # if node is a swap, then need to add it 3 times, so 1 swap is 3 gates
            if new_node.name == "swap":
                self.gates_depth.append((new_node.qargs[0].index, new_node.qargs[1].index))
                self.gates_depth.append((new_node.qargs[0].index, new_node.qargs[1].index))
            self.gates_depth.append((new_node.qargs[0].index, new_node.qargs[1].index))
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)

    def _fake_apply_gate(self, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        #print("     Executing gate: ", new_node.name)
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
        front_len = len(front_layer)
        if front_len == 0:
            return 0
        else:
            return self._compute_cost(front_layer, layout) / front_len 

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