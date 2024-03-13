OPENQASM 2.0;
include "qelib1.inc";
qreg q0[2];
creg c0[4];
reset q0[0];
u(pi/2,pi/2,-pi) q0[0];
reset q0[1];
u(pi,0,pi) q0[1];
cx q0[0],q0[1];
u(0,0,-pi/2) q0[1];
cx q0[0],q0[1];
u(pi/2,0,pi) q0[0];
u(0,0,pi/2) q0[1];
measure q0[0] -> c0[0];
barrier q0[0],q0[1];
reset q0[0];
u(pi/2,pi/4,-pi) q0[0];
reset q0[1];
u(pi,0,pi) q0[1];
cx q0[0],q0[1];
u(0,0,-pi/4) q0[1];
cx q0[0],q0[1];
u(0,0,pi/4) q0[1];
if(c0==1) u(0,0,-pi/2) q0[0];
u(pi/2,0,pi) q0[0];
measure q0[0] -> c0[1];
barrier q0[0],q0[1];
reset q0[0];
u(pi/2,pi/8,-pi) q0[0];
reset q0[1];
u(pi,0,pi) q0[1];
cx q0[0],q0[1];
u(0,0,-pi/8) q0[1];
cx q0[0],q0[1];
if(c0==1) u(0,0,-pi/4) q0[0];
if(c0==2) u(0,0,-pi/2) q0[0];
if(c0==3) u(0,0,-3*pi/4) q0[0];
u(pi/2,0,pi) q0[0];
measure q0[0] -> c0[2];
u(0,0,pi/8) q0[1];
barrier q0[0],q0[1];
reset q0[0];
u(pi/2,pi/16,-pi) q0[0];
reset q0[1];
u(pi,0,pi) q0[1];
cx q0[0],q0[1];
u(0,0,-pi/16) q0[1];
cx q0[0],q0[1];
if(c0==1) u(0,0,-pi/8) q0[0];
if(c0==2) u(0,0,-pi/4) q0[0];
if(c0==3) u(0,0,-3*pi/8) q0[0];
if(c0==4) u(0,0,-pi/2) q0[0];
if(c0==5) u(0,0,-5*pi/8) q0[0];
if(c0==6) u(0,0,-3*pi/4) q0[0];
if(c0==7) u(0,0,-7*pi/8) q0[0];
u(pi/2,0,pi) q0[0];
measure q0[0] -> c0[3];
u(0,0,pi/16) q0[1];
barrier q0[0],q0[1];
