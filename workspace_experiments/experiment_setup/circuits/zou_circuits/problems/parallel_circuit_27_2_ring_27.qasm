OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[2],q[25];
cx q[4],q[3];
cx q[21],q[18];
cx q[23],q[22];
cx q[16],q[17];
cx q[19],q[20];
cx q[12],q[11];
cx q[6],q[9];
cx q[10],q[7];
cx q[21],q[20];
cx q[0],q[2];
cx q[25],q[1];
cx q[18],q[19];
cx q[16],q[15];
cx q[11],q[14];
cx q[10],q[8];
