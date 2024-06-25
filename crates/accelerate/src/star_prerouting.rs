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

#![allow(unused_variables)]
#![warn(unused_mut)]


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

use hashbrown::HashSet;
use pyo3::prelude::*;
use crate::sabre::sabre_dag::{SabreDAG, DAGNode};
use crate::nlayout::VirtualQubit;


/// Python function to perform star prerouting on a SabreDAG.
/// This function processes star blocks and updates the DAG and qubit mapping.
#[pyfunction]
#[pyo3(text_signature = "(dag, blocks, processing_order, /)")]
fn star_preroute(
    py: Python,
    dag: &mut SabreDAG,
    blocks: Vec<Block>,
    processing_order: Vec<Nodes>,
) -> PyResult<()> {
    // Initialize qubit mapping to identity
    let mut qubit_mapping: Vec<usize> = (0..dag.num_qubits).collect();
    // Set to keep track of processed block IDs
    let mut processed_block_ids: HashSet<usize> = HashSet::new();
    // Determine the last two-qubit gate in the processing order
    let last_2q_gate = processing_order.iter().rev().find(|node| node.1.len() > 1).cloned();

    // Initialize the is_first_star flag
    let mut is_first_star = true;

    // Process each node in the given processing order
    for node in processing_order {
        // Directly match the result of find_block_id
        if let Some(block_id) = find_block_id(&blocks, &node) {
            // Skip if the block has already been processed
            if processed_block_ids.contains(&block_id) {
                continue;
            }
            // Mark the block as processed and process the entire block
            processed_block_ids.insert(block_id);
            process_block(&mut qubit_mapping, dag, &blocks[block_id], last_2q_gate.as_ref(), &mut is_first_star);
        } else {
            // Apply operation for nodes not part of any block
            apply_operation(&mut qubit_mapping, dag, &node);
        }
    }
    println!("Rust Qubit Mapping: {:?}", qubit_mapping);

    Ok(())
}

/// Finds the block ID for a given node.
/// 
/// # Arguments
/// 
/// * `blocks` - A vector of star blocks, where each block is represented by a tuple containing a boolean indicating the presence of a center and a list of nodes.
/// * `node` - The node for which the block ID needs to be found.
/// 
/// # Returns
/// 
/// An option containing the block ID if the node is part of a block, otherwise None.
fn find_block_id(blocks: &Vec<Block>, node: &Nodes) -> Option<usize> {
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
) {
    let (has_center, nodes) = block;

    // If the block contains exactly 2 nodes, apply them directly
    if nodes.len() == 2 {
        for node in nodes {
            apply_operation(qubit_mapping, dag, node);
        }
        return;
    }

    let mut prev_qargs = None;
    let mut swap_source = false;

    // Process each node in the block
    for node in nodes {
        // Apply operation directly if it's a single-qubit operation or the same as previous qargs
        if node.1.len() == 1 || prev_qargs == Some(&node.1) {
            apply_operation(qubit_mapping, dag, node);
            continue;
        }

        // If this is the first star and no swap source has been identified, set swap_source
        if *is_first_star && !swap_source {
            swap_source = *has_center;
            apply_operation(qubit_mapping, dag, node);
            prev_qargs = Some(&node.1);
            continue;
        }

        // Place 2q-gate and subsequent swap gate
        apply_operation(qubit_mapping, dag, node);

        if let Some(last) = last_2q_gate {
            apply_operation(qubit_mapping, dag, node);
            if node != last && node.1.len() == 2 {
                apply_swap(qubit_mapping, dag, &node.1);
            }
        }

        prev_qargs = Some(&node.1);
        *is_first_star = false;
    }

}

/// Applies an operation to the DAG using the current qubit mapping.
/// 
/// # Arguments
/// 
/// * `qubit_mapping` - A mutable reference to the qubit mapping vector.
/// * `dag` - A mutable reference to the SabreDAG being modified.
/// * `node` - The node representing the operation to be applied.
fn apply_operation(qubit_mapping: &mut Vec<usize>, dag: &mut SabreDAG, node: &Nodes) {
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

    // Update edges based on the predecessors of the current qubits
    for q in &node.1 {
    if let Some(predecessor) = dag.dag.node_indices().find(|&i| {
        dag.dag.node_weight(i).unwrap().qubits.contains(q)
    }) {
        dag.dag.add_edge(predecessor, new_index, ());
    }
    }
}


/// Applies a swap operation to the DAG and updates the qubit mapping.
/// 
/// # Arguments
/// 
/// * `qubit_mapping` - A mutable reference to the qubit mapping vector.
/// * `dag` - A mutable reference to the SabreDAG being modified.
/// * `qargs` - The qubit arguments for the swap operation.
fn apply_swap(qubit_mapping: &mut Vec<usize>, dag: &mut SabreDAG, qargs: &Vec<VirtualQubit>) {
    // Apply the swap operation in the `dag`
    // Update the `qubit_mapping` to reflect the swap
    if qargs.len() == 2 {
        let idx0 = qargs[0].index();
        let idx1 = qargs[1].index();
        qubit_mapping.swap(idx0, idx1);
    }
}



#[pymodule]
pub fn star_prerouting(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(star_preroute))?;
    Ok(())
} 