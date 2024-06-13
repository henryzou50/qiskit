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

use pyo3::prelude::*;

use crate::sabre::sabre_dag::SabreDAG;
use crate::star_block::StarBlockInfo;

#[pyfunction]
#[pyo3(
    text_signature = "(sabre_dag)"
)]
pub fn print_info(
    dag: &SabreDAG,
) -> PyResult<()> {
    // Iterate through the nodes and print the qargs and cargs
    println!("  Nodes:");
    for node in &dag.nodes {
        let qargs = node.1.iter().map(|q| q.index()).collect::<Vec<_>>();
        let cargs = node.2.iter().cloned().collect::<Vec<_>>();
        println!("Node: {}, qargs: {:?}, cargs: {:?}, directive: {}", node.0, qargs, cargs, node.3);
    }

    Ok(())

    
}

// Takes in a SabreDAG and a list of StarBlockInfo in order to apply the star prerouting algorithm
// by using the information from the StarBlockInfo to obtain the swaps to apply to SabreDAG.
// Also takes in a qubit mapping, which is an array of ints like [0, 1, 2, 3, 4]
// Returns the new qubit mapping after the swaps have been applied, like [3, 1, 0, 2, 4]
#[pyfunction]
#[pyo3(
    text_signature = "(dag, star_blocks, qubit_mapping)"
)]
pub fn apply_star_prerouting(
    dag: &SabreDAG,
    star_blocks: Vec<StarBlockInfo>,
    qubit_mapping: Vec<u32>,
) -> PyResult<()> {
    // For now to compile, just print basic info
    println!("  Star Blocks:");
    for star_block in &star_blocks {
        println!("      num2q: {}", star_block.num2q);
    }
    println!("  Qubit Mapping:");
    println!("    {:?}", qubit_mapping);


    // Loop through each node in the dag and prints its id
    for node in dag.dag.node_indices() {
        println!("Node: {:?}", node.index());
    }

    // Give each of the nodes in the star_blocks an id
    // For instance suppose, we have star_blocks with the nodes:
    // [[(10, [0], []), (11, [0, 1], [])], [(12, [0, 2], [])]] 
    // Then the star_blocks with the nodes [[(10, [0], []), (11, [0, 1], [])] will have id 0
    // and the star_blocks with the nodes [(12, [0, 2], [])] will have id 1
    // Make a dictionary with the nodes as keys and the ids as values
    let mut node_to_id = std::collections::HashMap::new();
    let mut id = 0;
    for star_block in &star_blocks {
        for node in &star_block.nodes {
            node_to_id.insert(node, id);
        }
        id += 1;
    }

    // Confirm that we were able to assign ids to all the nodes by printing the dictionary
    println!("  Node to ID:");
    for (node, id) in &node_to_id {
        println!("Node: {:?}, ID: {:?}", node, id);
    }
    Ok(())
}


#[pymodule]
pub fn star_prerouting(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(print_info))?;
    m.add_wrapped(wrap_pyfunction!(apply_star_prerouting))?;
    Ok(())
}