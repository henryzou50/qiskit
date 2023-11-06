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
import pprint

logger = logging.getLogger(__name__)

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
        heuristic="basic",
        lookahead_depth=3,
        alpha=1,
        beta=1,
        seed=None, 
        fake_run=False,
        beam_width=5,
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
        self.branch_id = 0
        self.nodes = 0
        self.skips = 0
        self.zeros = 0
        self.total_branches = 0
        self.total_nodes = 0
        self.total_skips = 0 
        self.total_zeros = 0
        self.alpha = alpha
        self.beta = beta
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

        gates_committed = []
        swaps_committed = []
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
            
            # execuate any executable gates, and update the front layer with the successor gates 
            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(mapped_dag, node, current_layout, canonical_register)
                    gates_committed.append(node)
                    for successor in self._successors(node, dag):
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)

                ops_since_progress = []
                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            swap_scores = {}
            lookahead_info = []
            for swap_qubits in self._obtain_swaps(front_layer, current_layout):
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)
                self._lookahead(
                    self.lookahead_depth, front_layer, trial_layout, dag, self.required_predecessors,
                    [], [swap_qubits], lookahead_info
                )

            #self.print_lookahead_info(lookahead_info)
            score_lookahead = self._score_heuristic_lookahead(lookahead_info)

            # update scores
            for swap in score_lookahead:
                swap_scores[swap] = score_lookahead[swap] 

            min_score = min(swap_scores.values())
            print("min_score: ", min_score)
            best_swaps = [k for k, v in swap_scores.items() if v == min_score]
            best_swaps.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
            best_swap = rng.choice(best_swaps)
            print("best_swap: ", best_swap[0].index, best_swap[1].index)
            swap_node = self._apply_gate(
                mapped_dag,
                DAGOpNode(op=SwapGate(), qargs=best_swap),
                current_layout,
                canonical_register,
            )
            swaps_committed.append(best_swap)
            current_layout.swap(*best_swap)
            ops_since_progress.append(swap_node)
            self.nodes = 0   
            self.skips = 0
            self.zeros = 0
            self.branch_id = 0
        self.property_set["final_layout"] = current_layout
        print("--------------------------------------------------------------")
        if not self.fake_run:
            return mapped_dag
        return dag

    def _lookahead(self, lookahead, front_layer, layout, dag, successors,
                   gates, swaps, lookahead_info):
        """Recursive lookahead function that does exploration of depth lookahead. Will 
        append the lookahead info to lookahead_info once we reach the base case.

        Parameter:
            lookahead (int): the depth of lookahead
            front_layer (list): the front layer of the current branch
            layout (Layout): the current layout
            dag (DAGCircuit): the dag
            successors (dict): a dict of successors of each node in dag
            gates (list): the list of gates in the current branch
            swaps (list): the list of swaps in the current branch
            lookahead_info (list): a list of tuples (gates, swaps, layout, front_layer, branch_id)
        """
        self.nodes += 1
        self.total_nodes += 1

        # Step 1: Apply executable gates and update front_layer
        trial_front_layer = front_layer.copy()
        trial_successors = successors.copy()
        while True:
            # Find executable gates and remove them from front layer
            new_front_layer = []
            execute_gate_list = []
            for node in trial_front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    if self.coupling_map.graph.has_edge(
                        layout._v2p[v0], layout._v2p[v1]
                    ):
                        execute_gate_list.append(node)
                    else:
                        new_front_layer.append(node)
            trial_front_layer = new_front_layer
            #  Repeat until there are no executable gates
            if not execute_gate_list:
                break
            # Updates front layer and successors
            gates += execute_gate_list
            for node in execute_gate_list:
                    for successor in self._successors(node, dag):
                        trial_successors[successor] -= 1
                        if trial_successors[node] == 0:
                            trial_front_layer.append(successor)
      
        # Base case
        if lookahead == 0 or trial_front_layer == []:
            lookahead_info.append((gates, swaps, layout, trial_front_layer, self.branch_id))
            self.branch_id += 1
            self.total_branches += 1
            return 

        # Step 2: Find candidate swaps and check scores
        scored_swaps = []
        for swap_qubits in self._obtain_swaps(trial_front_layer, layout):
            # Skip the last swap from the swap order
            if swaps and swap_qubits == swaps[-1]:
                self.skips += 1
                self.total_skips += 1
                continue
            
            trial_layout = layout.copy()
            trial_layout.swap(*swap_qubits)
            score = self._score_heuristic(trial_front_layer, trial_layout)
            scored_swaps.append((score, swap_qubits, trial_layout))
        
        # Sort the scored swaps by their scores (ascending)
        scored_swaps.sort(key=lambda x: x[0])

        # Process only the best candidates as defined by the beam width
        for score, swap_qubits, trial_layout in scored_swaps[:self.beam_width]:
            trial_swaps = swaps + [swap_qubits]
            trial_gates = gates.copy()
            
            self._lookahead(lookahead-1, trial_front_layer, trial_layout, dag, 
                            successors, trial_gates, trial_swaps, lookahead_info)
        
        return

    
    def _score_heuristic_lookahead(self, lookahead_info):
        # get a list of all of the gates in the lookahead_info
        all_gates = []
        for i in range(len(lookahead_info)):
            gates = lookahead_info[i][0]
            for gate in gates:
                if gate not in all_gates:
                    all_gates.append(gate)

        swap_to_score = {}

        for i in range(len(lookahead_info)):
            current_gates = lookahead_info[i][0]
            layout = lookahead_info[i][2]
            front_layer = lookahead_info[i][3]
            first_swap = lookahead_info[i][1][0]
            swaps = lookahead_info[i][1]

            # gates_to_apply is all of the gates excluding the current gates
            gates_to_apply = []
            for gate in all_gates:
                if gate not in current_gates:
                    gates_to_apply.append(gate)

            # score_front is the cost of applying the current gates divided by the size of front_layer
            if len(front_layer) == 0:
                score_front = 0
            else:
                score_front = self._compute_cost(front_layer, layout)/ len(front_layer)
            score_lookahead = self._compute_cost(gates_to_apply, layout) * self.beta
            score_depth = len(swaps) * self.alpha

            score = score_front + score_depth
            if i == 0:
                track_front = score_front
                track_lookahead = score_lookahead

            # update first swap score with small score
            if first_swap not in swap_to_score:
                swap_to_score[first_swap] = score
            else:
                if score < swap_to_score[first_swap]:
                    swap_to_score[first_swap] = score
                    track_front = score_front
                    track_lookahead = score_lookahead

        #print("             Score Front: ", track_front, " Score Lookahead: ", track_lookahead)
        #print("swap_to_score: ", swap_to_score)
        return swap_to_score

    def print_lookahead_info(self, lookahead_info):
        """ Print the lookahead info

        Parameters:
            lookahead_info (list): a list of tuples (gates, swaps, layout, branch_id)
        """
        for i in range(len(lookahead_info)):
            gates = lookahead_info[i][0]
            swaps = lookahead_info[i][1]
            layout = lookahead_info[i][2]
            branch = lookahead_info[i][3]
            print("             Branch: ", branch)
            print("             Gate Order: ", [node.name for node in gates])
            print("             Swap Order: ", [(node[0].index, node[1].index) for node in swaps])
            print("             Layout:     ", [layout[i]._index for i in range(len(layout))])
            print("**************************************************************")
    
    def print_step_info(self, gates_committed, swaps_committed):
        """ Print the step info 
        Parameters:
            gates_committed (list): the list of gates committed
            swaps_committed (list): the list of swaps committed 
        """
        print("Gates Committed: ", [node.name for node in gates_committed])
        print("Swaps Committed: ", [(node[0].index, node[1].index) for node in swaps_committed])
        print("Number of Branches: ", self.branch_id)
        print("Number of Nodes: ", self.nodes)
        print("Number of Skips: ", self.skips)
        print("Number of Zeros: ", self.zeros)
        print("--------------------------------------------------------------")

    def print_final_info(self):
        print("Total Branches: ", self.total_branches)
        print("Total Nodes: ", self.total_nodes)
        print("Total Skips: ", self.total_skips)
        print("Total Zeros: ", self.total_zeros)


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