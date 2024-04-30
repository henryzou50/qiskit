from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager, TransformationPass
from qiskit.providers.fake_provider import Fake27QPulseV1
from qiskit.transpiler.passes import ApplyLayout, FullAncillaAllocation, EnlargeWithAncilla

def get_two_qubit_depth(circuit: QuantumCircuit) -> int:
    """Get two qubit gate depth of a circuit."""
    return circuit.depth(lambda x: x.operation.num_qubits == 2)

def get_27_qubit_coupling_map() -> CouplingMap:
    """Get the coupling map of the 27 qubit device."""
    return CouplingMap(Fake27QPulseV1().configuration().coupling_map)

def build_ghz_circuit(n: int) -> QuantumCircuit:
    """Build a GHZ circuit of n qubits."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

def build_pass_manager(coupling_map: CouplingMap, 
                       layout_pass: TransformationPass, 
                       routing_pass: TransformationPass) -> PassManager:
    """Build a pass manager with layout and routing passes."""
    pm = PassManager([
        layout_pass,
        FullAncillaAllocation(coupling_map),
        EnlargeWithAncilla(),
        ApplyLayout(),
        routing_pass
    ])
    return pm

