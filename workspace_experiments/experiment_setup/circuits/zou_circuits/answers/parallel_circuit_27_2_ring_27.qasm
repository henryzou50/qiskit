OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[1],q[0];
cx q[4],q[3];
cx q[19],q[18];
cx q[23],q[22];
cx q[17],q[16];
cx q[20],q[21];
cx q[12],q[11];
cx q[6],q[7];
cx q[9],q[8];
swap q[19],q[20];
swap q[25],q[24];
swap q[12],q[11];
swap q[1],q[2];
swap q[10],q[9];
swap q[4],q[3];
swap q[17],q[16];
swap q[0],q[26];
swap q[23],q[22];
swap q[7],q[8];
cx q[20],q[21];
cx q[1],q[2];
cx q[26],q[0];
cx q[18],q[19];
cx q[16],q[15];
cx q[12],q[13];
cx q[10],q[9];
swap q[8],q[9];
swap q[26],q[25];
swap q[0],q[1];
swap q[11],q[12];
swap q[23],q[22];
swap q[4],q[3];
swap q[14],q[13];
swap q[20],q[21];
