OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[16],q[15];
cx q[9],q[10];
cx q[11],q[12];
cx q[0],q[26];
cx q[20],q[19];
cx q[3],q[4];
cx q[6],q[5];
cx q[14],q[13];
cx q[17],q[18];
cx q[24],q[25];
cx q[21],q[22];
cx q[8],q[7];
cx q[1],q[2];
swap q[0],q[1];
