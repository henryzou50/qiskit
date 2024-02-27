OPENQASM 2.0;
include "qelib1.inc";
qreg q0[25];
creg c0[1];
rx(-3.6924814) q0[1];
rx(-3.2923147) q0[13];
rx(5.4291652) q0[2];
rx(5.6875289) q0[14];
rx(1.3594796) q0[3];
rx(1.2065807) q0[15];
rx(-5.9123043) q0[4];
rx(-6.0041031) q0[16];
rx(-0.13186279) q0[5];
rx(0.50271205) q0[17];
rx(-4.3869008) q0[6];
rx(-4.1172873) q0[18];
rx(4.9830092) q0[7];
rx(4.8261369) q0[19];
rx(-1.4181518) q0[8];
rx(-1.5885531) q0[20];
rx(3.9058792) q0[9];
rx(3.2780951) q0[21];
rx(2.1483107) q0[10];
rx(2.2125048) q0[22];
rx(-1.552265) q0[11];
rx(-2.1338861) q0[23];
rx(3.5437778) q0[12];
rx(2.9294436) q0[24];
h q0[0];
cswap q0[0],q0[1],q0[13];
cswap q0[0],q0[2],q0[14];
cswap q0[0],q0[3],q0[15];
cswap q0[0],q0[4],q0[16];
cswap q0[0],q0[5],q0[17];
cswap q0[0],q0[6],q0[18];
cswap q0[0],q0[7],q0[19];
cswap q0[0],q0[8],q0[20];
cswap q0[0],q0[9],q0[21];
cswap q0[0],q0[10],q0[22];
cswap q0[0],q0[11],q0[23];
cswap q0[0],q0[12],q0[24];
h q0[0];
measure q0[0] -> c0[0];
