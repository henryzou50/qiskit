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
        lookahead_steps=5,
        beam_width=6,
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
        self.qubits_depth = None
        self._bit_indices = None
        self.dist_matrix = None
        self.lookahead_steps = lookahead_steps
        self.beam_width = beam_width

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

        self.dist_matrix = self.coupling_map.distance_matrix


        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None
        if not self.fake_run:
            mapped_dag = dag.copy_empty_like()

        canonical_register = dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
    

        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}

        # Initialize the depth of each qubit to 0.
        self.qubits_depth = dict.fromkeys(dag.qubits, 0)

        # Start algorithm from the front layer and iterate until all gates done.
        predecessors = self._build_required_predecessors(dag)
        front_layer = dag.front_layer()

        gate_seq, front_layer, predecessors = self._update_front_layer(
            current_layout, dag, front_layer, predecessors)
        
        # Step 1: apply the gates that were executed when updating front layer
        for gate in gate_seq:
            self._apply_gate(mapped_dag, gate, current_layout, canonical_register)

        initial_node = Node(current_layout, front_layer, 
                            predecessors, self.qubits_depth,
                            mapped_dag)
        mapped_dag = self._lookahead_search(initial_node, dag, canonical_register)
    
        circuit_depth = max(self.qubits_depth.values())
        print("Circuit depth: ", circuit_depth)
        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return dag
    
    def _lookahead_search(self, initial_node, dag, canonical_register):
        current_level = [initial_node]
        end_solutions = []
        for i in range(self.lookahead_steps):
            print("Lookahead step: ", i)
            next_level = []
            for node in current_level:
                if node.front_layer == []:
                    print("     End solution")
                    continue
                # Obtain next set of candidates 
                print("Front layer: ", node.front_layer)
                swap_candidates = self._obtain_swaps(node.front_layer, node.layout)
                # For each candidate, apply the sabre algorithm
                for swap_qubits in swap_candidates:
                    trial_layout = node.layout.copy()
                    trial_layout.swap(*swap_qubits)
                    # Create a new node with the new layout
                    trial_node =  Node(trial_layout.copy(), node.front_layer.copy(), 
                                        node.predecessors.copy(), node.qubit_depth.copy(),
                                        node.mapped_dag)
                    # Run the sabre algorithm on the new node
                    trial_gate_seq, trial_depth = self._get_sabre_result(trial_node, 
                                                                         dag, 
                                                                         swap_qubits)
                    trial_node.gate_seq = trial_gate_seq
                    trial_node.depth = trial_depth
                    next_level.append(trial_node)
                    print("     Trial front layer", trial_node.front_layer)
                    print("     Trial depth: ", trial_node.depth)
            # Sort the branches by depth
            next_level.sort(key=lambda x: x.depth)
            # Create a beam of the best branches
            beam = next_level[:self.beam_width]
            # Update the nodes mapped dag with the gates that were executed
            for branch in beam:
                initial_layout = branch.layout
                initial_dag = deepcopy(branch.mapped_dag)
                gate_seq = branch.gate_seq
                for gate in gate_seq:
                    # if swap need to update the layout
                    if gate.name == "swap":
                        initial_layout.swap(*gate.qargs)
                    self._branch_apply_gate(initial_dag, gate, initial_layout, canonical_register)
                # note that we do not need to update the qubit depth as it was already calculated
                branch.mapped_dag = initial_dag
                # confirm depth with mapped_dag
            
            current_level = beam
        
        # Select number 1 branch
        final_mapped_dag = beam[0].mapped_dag
        return final_mapped_dag
    
    def _get_sabre_result(self, node, dag, swap_qubit):
        """ Run the sabre algorithm on the given node and return the gate sequence and depth
        
        Args:
            node (Node): node containing the layout, front layer, predecessors, and qubit depth
            dag (DAGCircuit): the directed acyclic graph to be mapped.
            swap_qubit (tuple): the qubits to swap
        Returns:
            gate_seq (list): the gate sequence of the sabre algorithm
            trial_depth (int): the depth of the sabre algorithm
        """
        rng = np.random.default_rng(self.seed)
        current_layout = node.layout
        front_layer    = node.front_layer
        qubits_depth   = node.qubit_depth
        predecessors   = node.predecessors
        swap = DAGOpNode(op=SwapGate(), qargs=swap_qubit)
        gate_seq = [swap]
        first_run = True

        first_layout = None
        first_front_layer = None
        first_predecessors = None

        self._fake_apply_gate(qubits_depth, swap)
        while front_layer:
            execute_gate_list = []
            # Remove as many immediately applicable gates as possible
            new_front_layer = []
            for gate in front_layer:
                if len(gate.qargs) == 2:
                    v0, v1 = gate.qargs
                    # Accessing layout._v2p directly to avoid overhead from __getitem__ and a
                    # single access isn't feasible because the layout is updated on each iteration
                    if self.coupling_map.graph.has_edge(
                        current_layout._v2p[v0], current_layout._v2p[v1]
                    ):
                        execute_gate_list.append(gate)
                    else:
                        new_front_layer.append(gate)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(gate)
            front_layer = new_front_layer

            if execute_gate_list:
                for gate in execute_gate_list:
                    gate_seq.append(gate)
                    self._fake_apply_gate(qubits_depth, gate)
                    for successor in self._successors(gate, dag):
                        predecessors[successor] -= 1
                        if predecessors[successor] == 0:
                            front_layer.append(successor)
                continue
            
            # update front layer, predecessors, current layout, and qubit depth of the node
            if first_run:
                first_front_layer  = front_layer.copy()
                first_predecessors = predecessors.copy()
                first_layout       = current_layout.copy()
                first_run = False
            # end of first run

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            swap_scores = {}
            swap_candidates = self._obtain_swaps(front_layer, current_layout)
            for swap_qubits in swap_candidates:
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
            swap = DAGOpNode(op=SwapGate(), qargs=best_swap)
            gate_seq.append(swap)
            self._fake_apply_gate(qubits_depth, swap)
            current_layout.swap(*best_swap)

        trial_depth = max(qubits_depth.values())

        # update node info with the first run

        node.layout       = first_layout
        node.front_layer  = first_front_layer
        node.predecessors = first_predecessors

        print("     First front layer: ", first_front_layer)

        return gate_seq, trial_depth
    
    def _update_front_layer(self, current_layout, dag, front_layer, predecessors):
        gate_seq = []

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
                    gate_seq.append(node)
                    for successor in self._successors(node, dag):
                        predecessors[successor] -= 1
                        if predecessors[successor] == 0:
                            front_layer.append(successor)
                continue
            # done updating front layer and predecessors
            break

        return gate_seq, front_layer, predecessors

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
    
    def _branch_apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)
    
    def _fake_apply_gate(self, qubits_depth, node):
        # Updates the depth of the wires, only considers 2 qubit gates
        if len(node.qargs) == 2:
            depth = max(qubits_depth[node.qargs[0]], qubits_depth[node.qargs[1]]) + 1
            if node.name == "swap": # Treat swaps as 3 CNOTS
                depth += 2
            qubits_depth[node.qargs[0]] = depth
            qubits_depth[node.qargs[1]] = depth

    def _reset_qubits_depth(self):
        """Reset the depth of all qubits to 0."""
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

class Node():
    def __init__(self, layout, front_layer, predecessors, qubit_depth, mapped_dag):
        self.layout        = layout # layout of the circuit
        self.front_layer   = front_layer # front layer of the circuit
        self.predecessors  = predecessors # successors of the front layer
        self.qubit_depth   = qubit_depth # depth of each qubit
        self.mapped_dag    = mapped_dag # mapped dag
        self.depth         = None
        self.gate_seq      = None