OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
u(0,0,-3*pi/4) q[0];
u(0,0,-3*pi/4) q[1];
u(0,0,3*pi/4) q[2];
cx q[1],q[2];
u(pi/2,0,pi) q[1];
cx q[2],q[1];
u(0,0,-pi/2) q[1];
cx q[2],q[1];
u(pi/2,0,-pi/2) q[1];
cx q[1],q[2];
u(0,0,-pi/2) q[1];
cx q[0],q[1];
u(pi/2,0,pi) q[0];
cx q[1],q[0];
u(0,0,0.12771539579680882) q[0];
cx q[1],q[0];
u(pi/2,0,3.013877257792984) q[0];
cx q[0],q[1];
u(0,0,0.3528565821394194) q[0];
u(0,0,pi/2) q[1];
u(0,0,pi/2) q[2];
u(0,0,3*pi/4) q[3];
cx q[2],q[3];
u(pi/2,0,pi) q[2];
cx q[3],q[2];
u(0,0,-0.12771539579680882) q[2];
cx q[3],q[2];
u(pi/2,0,-3.0138772577929847) q[2];
cx q[2],q[3];
u(0,0,-pi/2) q[2];
cx q[1],q[2];
u(pi/2,0,pi) q[1];
cx q[2],q[1];
u(0,0,-pi/2) q[1];
cx q[2],q[1];
u(pi/2,0,-pi/2) q[1];
cx q[1],q[2];
u(0,0,0.3528565821394194) q[1];
u(0,0,0.1774716964567742) q[2];
cx q[1],q[2];
u(pi/2,0,pi) q[1];
cx q[2],q[1];
u(0,0,-pi/2) q[1];
cx q[2],q[1];
u(pi/2,0,-pi/2) q[1];
cx q[1],q[2];
u(0,0,-pi/2) q[1];
cx q[0],q[1];
u(pi/2,0,pi) q[0];
cx q[1],q[0];
u(0,0,-0.08015696403871005) q[0];
cx q[1],q[0];
u(pi/2,0,-3.0614356895510833) q[0];
cx q[0],q[1];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
u(0,0,pi/2) q[2];
u(0,0,0.1774716964567742) q[3];
cx q[2],q[3];
u(pi/2,0,pi) q[2];
cx q[3],q[2];
u(0,0,0.08015696403871005) q[2];
cx q[3],q[2];
u(pi/2,0,3.0614356895510833) q[2];
cx q[2],q[3];
u(0,0,-pi/4) q[2];
u(0,0,pi/4) q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
