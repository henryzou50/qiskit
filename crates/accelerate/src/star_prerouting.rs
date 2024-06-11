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

#[pyfunction]
#[pyo3(
    text_signature = "(sabre_dag)"
)]
pub fn print_info(
    dag: &SabreDAG,
) -> PyResult<()> {
    let num_qubits = dag.num_qubits;
    let num_clbits = dag.num_clbits;
    let dag_nodes_count = dag.dag.node_count();
    let first_layer_len = dag.first_layer.len();
    let nodes_len = dag.nodes.len();
    let node_blocks_len = dag.node_blocks.len();

    println!("num_qubits: {}", num_qubits);
    println!("num_clbits: {}", num_clbits);
    println!("dag_nodes_count: {}", dag_nodes_count);
    println!("first_layer_len: {}", first_layer_len);
    println!("nodes_len: {}", nodes_len);
    println!("node_blocks_len: {}", node_blocks_len);

    Ok(())
}

#[pymodule]
pub fn star_prerouting(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(print_info))?;
    Ok(())
}