OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
u(pi/2,3*pi/2,3*pi/2) q[0];
u(pi/2,3*pi/2,0.9253564645174989) q[1];
u(0,1.4065829705916304,1.9121976308064648) q[2];
cx q[1],q[2];
u(0.39048419698856807,pi/2,3*pi/2) q[1];
u(0.3904841969885679,-pi,-pi) q[2];
cx q[1],q[2];
u(pi/2,2.2851472667738992,-pi/2) q[1];
cx q[0],q[1];
u(1.0312029416845523,pi/2,3*pi/2) q[0];
u(1.0312029416845523,-pi,-pi) q[1];
cx q[0],q[1];
u(3.0681103418073223,0,0) q[0];
u(pi/2,-pi/2,2.7097885059151636) q[1];
u(0,1.6569307818908463,-1.5632134394682002) q[2];
cx q[1],q[2];
u(0.7309297473898185,pi/2,3*pi/2) q[1];
u(0.7309297473898186,-pi,-pi) q[2];
cx q[1],q[2];
u(pi/2,-1.9657101219869773,-pi/2) q[1];
cx q[0],q[1];
u(1.0312029416845523,pi/2,3*pi/2) q[0];
u(1.0312029416845523,-pi,-pi) q[1];
cx q[0],q[1];
u(pi/2,1.1288315380582659,3*pi/2) q[0];
u(pi/2,pi/2,-0.9998502167671814) q[1];
u(pi,-1.026729889000492,-2.3524336830584245) q[2];
cx q[1],q[2];
u(0.39048419698856807,pi/2,3*pi/2) q[1];
u(0.3904841969885679,-pi,-pi) q[2];
cx q[1],q[2];
u(pi/2,3.6305670036045803,pi/2) q[1];
u(pi,-1.9780585059374756,3.0679839342039674) q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
