//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.


use pyo3::prelude::*;

#[pyclass(module = "qiskit._accelerate.star_block")]
#[derive(Clone, Debug)]
pub struct StarBlockInfo {
    // number of two-qubit gates in the block
    pub num2q: u32,
    // First element is physcial qubit number, second element is qargs, third element is cargs
    pub nodes: Vec<(u8, Vec<u32>, Vec<u32>)>,
}

#[pymethods]
impl StarBlockInfo {
    #[new]
    #[pyo3(text_signature = "(num2q, phys_qubit_nums, qargs_indices_list, cargs_indices_list/)")]
    // Example input from python
    // phys_qubit_nums: [10, 11, 12, 13, 14]
    // qargs_indices_list: [[0], [0, 1], [0, 2], [0, 3], [0, 4]]
    // cargs_indices_list: [[], [], [], [], []]
    // Expected nodes output:
    // nodes: [(10, [0], []), (11, [0, 1], []), (12, [0, 2], []), (13, [0, 3], []), (14, [0, 4], [])]
    pub fn new(
        num2q: u32, 
        phys_qubit_nums: Vec<u8>,
        qargs_indices_list: Vec<Vec<u32>>,
        cargs_indices_list: Vec<Vec<u32>>,
    ) -> PyResult<Self> {
        let mut nodes: Vec<(u8, Vec<u32>, Vec<u32>)> = Vec::new();
        for i in 0..qargs_indices_list.len() {
            let phys_qubit_num = phys_qubit_nums[i];
            let qargs = qargs_indices_list[i].clone();
            let cargs = cargs_indices_list[i].clone();
            nodes.push((phys_qubit_num, qargs, cargs));
        }

        Ok(StarBlockInfo {
            num2q,
            nodes,
        })
    }
} 

#[pyfunction]
#[pyo3(
    text_signature = "(star_block_info)"
)]
pub fn print_info(
    star_block_info: &StarBlockInfo,
) -> PyResult<()> {
    println!("  Star Block Info:");
    println!("num2q: {}", star_block_info.num2q);
    println!("nodes:");
    for node in &star_block_info.nodes {
        println!("Node: {}, qargs: {:?}, cargs: {:?}", node.0, node.1, node.2);
    }
    Ok(())
}

#[pymodule]
pub fn star_block(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<StarBlockInfo>()?;
    m.add_function(wrap_pyfunction!(print_info, m)?)?;
    Ok(())
}