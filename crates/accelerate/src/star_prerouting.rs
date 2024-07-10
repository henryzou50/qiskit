// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

/// Type alias for a node representation.
/// Each node is represented as a tuple containing:
/// - Operation index (usize)
/// - List of involved qubit indices (Vec<VirtualQubit>)
/// - Set of involved classical bit indices (HashSet<usize>)
/// - Directive flag (bool)
type Nodes = (usize, Vec<VirtualQubit>, HashSet<usize>, bool);

/// Type alias for a block representation.
/// Each block is represented by a tuple containing:
/// - A boolean indicating the presence of a center (bool)
/// - A list of nodes (Vec<Nodes>)
type Block = (bool, Vec<Nodes>);

use hashbrown::HashMap;
use hashbrown::HashSet;
use pyo3::prelude::*;
use numpy::IntoPyArray;
use crate::nlayout::PhysicalQubit;
use crate::sabre::sabre_dag::{SabreDAG, DAGNode};
use crate::nlayout::VirtualQubit;
use crate::sabre::swap_map::SwapMap;
use crate::sabre::NodeBlockResults;
use crate::sabre::SabreResult;
use crate::sabre::BlockResult;

/// Python function to perform star prerouting on a SabreDAG.
/// This function processes star blocks and updates the DAG and qubit mapping.
#[pyfunction]
#[pyo3(text_signature = "(dag, blocks, processing_order, /)")]
fn star_preroute(
    py: Python,
    dag: &mut SabreDAG,
    blocks: Vec<Block>,
    processing_order: Vec<Nodes>,
) -> (SwapMap, PyObject, NodeBlockResults, PyObject) {
    // Initialize qubit mapping to identity
    let mut qubit_mapping: Vec<usize> = (0..dag.num_qubits).collect();
    // Set to keep track of processed block IDs
    let mut processed_block_ids: HashSet<usize> = HashSet::new();
    // Determine the last two-qubit gate in the processing order
    let last_2q_gate = processing_order.iter().rev().find(|node| node.1.len() > 1);

    // Initialize the is_first_star flag
    let mut is_first_star = true;

    // Initialize structures for SabreResult
    let mut out_map: HashMap<usize, Vec<[PhysicalQubit; 2]>> = HashMap::new();
    let mut gate_order: Vec<usize> = Vec::with_capacity(dag.dag.node_count());
    let node_block_results: HashMap<usize, Vec<BlockResult>> = HashMap::new();

    // Process each node in the given processing order
    for node in processing_order.iter()
    {
        // Directly match the result of find_block_id
        if let Some(block_id) = find_block_id(&blocks, &node) {
            // Skip if the block has already been processed
            if processed_block_ids.contains(&block_id) {
                continue;
            }
            // Mark the block as processed and process the entire block
            processed_block_ids.insert(block_id);
            process_block(&mut qubit_mapping, dag, &blocks[block_id], last_2q_gate, &mut is_first_star, &mut gate_order, &mut out_map);
        } else {
            // Apply operation for nodes not part of any block
            apply_operation(&mut qubit_mapping, dag, &node, &mut gate_order, &mut out_map);
        }
    }

    let res = SabreResult {
        map: SwapMap { map: out_map },
        node_order: gate_order,
        node_block_results: NodeBlockResults { results: node_block_results },
    };

    let final_res = (
        res.map,
        res.node_order.into_pyarray_bound(py).into(),
        res.node_block_results,
        qubit_mapping.into_pyarray_bound(py).into(),
    );
    final_res
}

/// Finds the block ID for a given node.
/// 
/// # Arguments
/// 
/// * `blocks` - A vector of blocks to search for the node.
/// * `node` - The node for which the block ID needs to be found.
/// 
/// # Returns
/// 
/// An option containing the block ID if the node is part of a block, otherwise None.
fn find_block_id(blocks: &[Block], node: &Nodes) -> Option<usize> {
    for (i, block) in blocks.iter().enumerate() {
        if block.1.iter().any(|n| n.0 == node.0) {
            return Some(i);
        }
    }
    None
}

/// Processes a star block, applying operations and handling swaps.
/// 
/// # Arguments
/// 
/// * `qubit_mapping` - A mutable reference to the qubit mapping vector.
/// * `dag` - A mutable reference to the SabreDAG being modified.
/// * `block` - A tuple containing a boolean indicating the presence of a center and a vector of nodes representing the star block.
/// * `last_2q_gate` - The last two-qubit gate in the processing order.
/// * `is_first_star` - A mutable reference to a boolean indicating if this is the first star block being processed.
fn process_block(
    qubit_mapping: &mut Vec<usize>,
    dag: &mut SabreDAG,
    block: &Block,
    last_2q_gate: Option<&Nodes>,
    is_first_star: &mut bool,
    gate_order: &mut Vec<usize>,
    out_map: &mut HashMap<usize, Vec<[PhysicalQubit; 2]>>,
) {
    let (has_center, sequence) = block;

    // If the block contains exactly 2 nodes, apply them directly
    if sequence.len() == 2 {
        for inner_node in sequence {
            apply_operation(qubit_mapping, dag, inner_node, gate_order, out_map);
        }
        return;
    }

    let mut prev_qargs = None;
    let mut swap_source = false;

    // Process each node in the block
    for (i, inner_node) in sequence.iter().enumerate() {
        // Apply operation directly if it's a single-qubit operation or the same as previous qargs
        if inner_node.1.len() == 1 || prev_qargs == Some(&inner_node.1) {
            apply_operation(qubit_mapping, dag, inner_node, gate_order, out_map);
            continue;
        }

        // If this is the first star and no swap source has been identified, set swap_source
        if *is_first_star && !swap_source {
            swap_source = *has_center;
            apply_operation(qubit_mapping, dag, inner_node, gate_order, out_map);
            prev_qargs = Some(&inner_node.1);
            continue;
        }

        // Place 2q-gate and subsequent swap gate
        apply_operation(qubit_mapping, dag, inner_node, gate_order, out_map);

        if inner_node != last_2q_gate.unwrap() && inner_node.1.len() == 2 {
            if let Some(next_node) = sequence.get(i + 1).cloned() {
                // Use the node ID of the next node in the sequence
                apply_swap(qubit_mapping, &inner_node.1, next_node.0, out_map);
            }
        }

        prev_qargs = Some(&inner_node.1);
    }

}

/// Applies an operation to the DAG using the current qubit mapping.
/// 
/// # Arguments
/// 
/// * `qubit_mapping` - A mutable reference to the qubit mapping vector.
/// * `dag` - A mutable reference to the SabreDAG being modified.
/// * `node` - The node representing the operation to be applied.
/// * `gate_order` - A mutable reference to the gate order vector.
/// * `out_map` - A mutable reference to the output map.
fn apply_operation(
    qubit_mapping: &mut Vec<usize>,
    dag: &mut SabreDAG,
    node: &Nodes,
    gate_order: &mut Vec<usize>, 
    out_map: &mut HashMap<usize, Vec<[PhysicalQubit; 2]>>,
) {
    // Remap the qubits based on the current qubit mapping
    let mapped_qubits: Vec<VirtualQubit> = node.1.iter()
    .map(|q| VirtualQubit::new(qubit_mapping[q.index()].try_into().unwrap()))
    .collect();

    // Create a new DAGNode with the mapped qubits
    let new_node = DAGNode {
        py_node_id: node.0,
        qubits: mapped_qubits,
        directive: node.3,
    };

    // Add the new node to the DAG
    let new_index = dag.dag.add_node(new_node);
    gate_order.push(node.0);

    // Update edges based on the predecessors of the current qubits
    for q in &node.1 {
        if let Some(predecessor) = dag.dag.node_indices().find(|&i| {
            dag.dag.node_weight(i).unwrap().qubits.contains(q)
        }) {
            dag.dag.add_edge(predecessor, new_index, ());
        }
    }

    // if out_map does not contain the node id, insert it with an empty vector
    out_map.entry(node.0).or_insert_with(Vec::new);
}

/// Applies a swap operation to the DAG and updates the qubit mapping.
/// 
/// # Arguments
/// 
/// * `qubit_mapping` - A mutable reference to the qubit mapping vector.
/// * `dag` - A mutable reference to the SabreDAG being modified.
/// * `qargs` - A slice containing the qubit indices for the swap operation.
/// * `next_node_id` - The ID of the next node in the sequence.
/// * `out_map` - A mutable reference to the output map.
fn apply_swap(
    qubit_mapping: &mut Vec<usize>,
    qargs: &[VirtualQubit],
    next_node_id: usize,
    out_map: &mut HashMap<usize, Vec<[PhysicalQubit; 2]>>,
) {
    // Apply the swap operation in the `dag`
    // Update the `qubit_mapping` to reflect the swap
    if qargs.len() == 2 {
        let idx0 = qargs[0].index();
        let idx1 = qargs[1].index();

        qubit_mapping.swap(idx0, idx1);
        out_map.insert(next_node_id, vec![
            [
                PhysicalQubit::new(qubit_mapping[idx0].try_into().unwrap()), 
                PhysicalQubit::new(qubit_mapping[idx1].try_into().unwrap())
            ]
        ]);
    }
}

#[pymodule]
pub fn star_prerouting(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(star_preroute))?;
    Ok(())
}
