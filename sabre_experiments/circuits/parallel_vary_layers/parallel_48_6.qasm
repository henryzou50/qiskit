OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rx(0) q[0];
rx(0) q[1];
rx(0) q[2];
rx(0) q[3];
rx(0) q[4];
rx(0) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
rzz(pi/6) q[0],q[1];
rzz(pi/6) q[2],q[3];
rzz(pi/6) q[4],q[5];
rzz(pi/6) q[1],q[2];
rzz(pi/6) q[3],q[4];
rzz(pi/6) q[0],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
