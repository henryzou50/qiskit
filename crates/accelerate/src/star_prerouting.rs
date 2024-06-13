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
    // Iterate through the nodes and print the qargs and cargs
    println!("  Nodes:");
    for node in &dag.nodes {
        let qargs = node.1.iter().map(|q| q.index()).collect::<Vec<_>>();
        let cargs = node.2.iter().cloned().collect::<Vec<_>>();
        println!("Node: {}, qargs: {:?}, cargs: {:?}, directive: {}", node.0, qargs, cargs, node.3);
    }

    Ok(())

    
}

#[pymodule]
pub fn star_prerouting(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(print_info))?;
    Ok(())
}