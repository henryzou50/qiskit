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

from qiskit.circuit.library.standard_gates import SwapGate, CXGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode
from qiskit.converters import dag_to_circuit

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
        self._bit_indices = None
        self.dist_matrix = None
        self.dag = None
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

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None
        if not self.fake_run:
            mapped_dag = dag.copy_empty_like()

        self.dist_matrix = self.coupling_map.distance_matrix
        self.dag = dag
        self.rng = np.random.default_rng(self.seed)
        canonical_register = dag.qregs["q"]
        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}

        # Initialize state variables
        front_layer    = dag.front_layer()
        current_layout = Layout.generate_trivial_layout(canonical_register)
        predecessors   = self._build_required_predecessors(dag)
        qubits_depth   = dict.fromkeys(dag.qubits, 0)
        gate_seq       = []
        gates_count    = 0
        
        # Phase 1: Create initial state
        initial_state = State(current_layout, front_layer, predecessors, 
                              qubits_depth, gate_seq, gates_count)

        # Phase 2: Update current layout and front layer until all gates are exhausted
        self._update_state(initial_state)

        # Phase 3: Get swap candidates and apply them
        if initial_state.front_layer:
            self._apply_swap(initial_state)
            self._update_state(initial_state)

            # Phase 4: If there are still gates in the front layer, repeat phase 2 and 3
            while initial_state.front_layer:
                self._apply_swap(initial_state)
                self._update_state(initial_state)

        # Phase 5: Use gate sequence to apply the gates to the mapped dag
        current_layout = Layout.generate_trivial_layout(canonical_register) # Reset layout
        for node in initial_state.gates_seq:
            if node.name == "swap":
                current_layout.swap(*node.qargs)
                # Instead of applying the swap gate mapped_dag, apply 3 CXGates to mapped_dag
                self._apply_gate(mapped_dag, DAGOpNode(op=CXGate(), qargs=node.qargs),
                                    current_layout, canonical_register)
                self._apply_gate(mapped_dag, DAGOpNode(op=CXGate(), qargs=node.qargs[::-1]),
                                    current_layout, canonical_register)
                self._apply_gate(mapped_dag, DAGOpNode(op=CXGate(), qargs=node.qargs),
                                    current_layout, canonical_register)
            else:
                self._apply_gate(mapped_dag, node, current_layout, canonical_register)

        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return dag
    
    def _validate_state(self, state):
        """ Validate the state of the algorithm by printing out the information for the state
        and then manually checking the state for correctness.
        """
        print(f"{'Layout:':<20} {state.layout._v2p.values()}")
        print(f"{'Front Layer:':<20}")
        for node in state.front_layer:
            qargs = node.qargs
            indices = [qubit.index for qubit in qargs]
            print(f"    {'Node :':<16} {node.name}, {', '.join(map(str, indices))}")
        print(f"{'Predecessors:':<20} {state.predecessors.values()}")
        print(f"{'Qubit Depth:':<20} {state.qubit_depth.values()}")
        print(f"{'Gates Sequence:':<20}")
        for node in state.gates_seq:
            qargs = node.qargs
            indices = [qubit.index for qubit in qargs]
            print(f"    {'Node :':<16} {node.name}, {', '.join(map(str, indices))}")
        print(f"{'Gates Count:':<20} {state.gates_count}")

    def _update_state(self, state, trial=False):
        """ Update the state by removing as many immediately applicable gates from the front layer, 
        until no more gates can be immediately applied. The gates removed are added to the 
        gate sequence. 
        Here the variables that can be updated are:
            Front layer
            Predecessors
            Qubit depth
            Gates sequence
            Gates count
        Args:
            state (State): the state of the algorithm
            trial (bool): if true, no gate information is updated
        """
        # While there are still gates in the front layer
        while True:
            execute_gate_list = []
            new_front_layer = []

            for node in state.front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    # Accessing layout._v2p directly to avoid overhead from __getitem__ and a 
                    # single access isn't feasible because the layout is updated on each iteration
                    if self.coupling_map.graph.has_edge(
                        state.layout._v2p[v0], state.layout._v2p[v1]
                    ):
                        execute_gate_list.append(node)
                    else:
                        new_front_layer.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)
            state.front_layer = new_front_layer

            if not execute_gate_list:
                break
            else:
                for node in execute_gate_list:
                    # Update gate info
                    if not trial:
                        state.gates_seq.append(node)
                        state.gates_count += 1
                    # Update qubit depth
                    self._update_qubit_depth(node, state.qubit_depth)

                    # Update predecessors and front layer
                    for successor in self._successors(node, self.dag):
                        state.predecessors[successor] -= 1
                        if state.predecessors[successor] == 0:
                            state.front_layer.append(successor)
    
    def _update_qubit_depth(self, node, qubit_depth):
        """ Update the qubit depth of the qubits involved in the gate.
        Note this considers the depth of ONLY 2 qubit gates. 
        
        Arg: 
            node (DAGOpNode): the gate to be applied
            qubit_depth (dict): a dictionary of qubit depths
        """
        if len(node.qargs) == 2:
            depth = max(qubit_depth[node.qargs[0]], qubit_depth[node.qargs[1]]) + 1
            if node.name == "swap": # Treat swaps as 3 CNOTS
                depth += 2
            qubit_depth[node.qargs[0]] = depth
            qubit_depth[node.qargs[1]] = depth
    
    def _apply_swap(self, state):
        """ Apply a swap to the state. The swap is chosen from a list of candidate swaps. 
        The candidate swaps are chosen from the neighborhood of the qubits involved in the 
        front layer. The swap chosen is the one that minimizes the heuristic cost function.
        The swap chosen is added to the gate sequence. 
        Here the variables that can be updated are:
            Layout
            Qubit depth
            Gates sequence
        Args:
            state (State): the state of the algorithm
        """
        # Find the swap candidates for this state's front layer and current_layout
        swap_candidates = list(self._obtain_swaps(state.front_layer, state.layout))
        # sorting so that we always get the same order of swaps, so there is no randomness from order
        swap_candidates.sort(key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]]))
        # Explores each swap candidate and scores it

        best_swap = None
        best_score = float('inf')
        best_depth = float('inf')

        for swap in swap_candidates:
            trial_layout = state.layout.copy()
            trial_layout.swap(*swap)
            score = self._score_heuristic(state.front_layer, trial_layout)

            if score > best_score:
                continue

            trial_qubit_depth = state.qubit_depth.copy()
            # update depth 
            self._update_qubit_depth(DAGOpNode(op=SwapGate(), qargs=swap), trial_qubit_depth)
            trial_state = State(trial_layout, state.front_layer.copy(), state.predecessors.copy(),
                                trial_qubit_depth, None, 0)
            self._update_state(trial_state, trial=True)
            depth = max(trial_state.qubit_depth.values())

            score += (depth / 3)

            if score < best_score :
                best_swap = swap
                best_score = score

        # Update layout
        state.layout.swap(*best_swap)
        # Update gate sequence (note that gate count is not updated)
        best_swap_node = DAGOpNode(op=SwapGate(), qargs=best_swap)
        state.gates_seq.append(best_swap_node)
        # Update qubit depth
        self._update_qubit_depth(best_swap_node, state.qubit_depth)

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(new_node.op, new_node.qargs, new_node.cargs)

    def _reset_qubits_depth(self):
        """Reset all qubit decay factors to 1 upon request (to forget about
        past penalizations).
        """
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

class State():
    def __init__(self, layout, front_layer, predecessors, qubit_depth, gates_seq, gates_count):
        self.layout       = layout        # a Layout object
        self.front_layer  = front_layer   # a list of gates in the front layer
        self.predecessors = predecessors  # a dict of predecessors
        self.qubit_depth  = qubit_depth   # a dict of qubit depths
        self.gates_seq    = gates_seq     # a list of gates applied (including swaps)
        self.gates_count  = gates_count   # a int of gates applied (excluding swaps)
