OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
rx(0) q[6];
rx(0) q[7];
rx(0) q[8];
rx(0) q[9];
rx(0) q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[6],q[7];
rzz(pi/6) q[8],q[9];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[5],q[6];
rzz(pi/6) q[7],q[8];
rzz(pi/6) q[9],q[10];
rzz(pi/6) q[0],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
