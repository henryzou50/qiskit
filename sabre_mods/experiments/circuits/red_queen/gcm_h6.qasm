OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[1];
u(pi/2,pi/4,-pi) q[0];
u(pi,-pi/4,-pi) q[1];
cx q[0],q[1];
u(pi/4,-pi,3*pi/4) q[1];
cx q[0],q[1];
u(0,0,-3*pi/4) q[0];
u(pi/4,pi/2,0) q[1];
u(pi,-pi/4,-pi) q[2];
cx q[0],q[2];
u(0.3648573517862774,-pi,-pi/4) q[2];
cx q[0],q[2];
u(pi/2,0.3648573517862772,pi/2) q[2];
cx q[1],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[2];
cx q[1],q[2];
u(0,0,pi/4) q[1];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
cx q[0],q[1];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[1];
cx q[0],q[1];
u(pi/4,-pi,-3*pi/4) q[1];
cx q[0],q[1];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[1];
u(pi/2,1.4922506383856682,-3*pi/4) q[2];
cx q[0],q[2];
u(0.5253852471287274,-pi,2.18700743192748) q[2];
cx q[0],q[2];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[2];
cx q[1],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[2];
cx q[1],q[2];
u(0,0,pi/4) q[1];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
cx q[0],q[1];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[1];
cx q[0],q[1];
u(pi/4,-pi,-3*pi/4) q[1];
cx q[0],q[1];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[1];
cx q[0],q[1];
u(0.3648573517862774,-pi,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[1];
u(pi/2,0,-3*pi/4) q[2];
cx q[0],q[2];
u(0.4205408116111712,-pi,pi/2) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[2];
cx q[0],q[2];
u(pi/4,-pi,3*pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[2];
u(pi,pi/2,-pi) q[3];
cx q[0],q[3];
u(pi/4,-pi,pi/2) q[3];
cx q[0],q[3];
u(0,0,pi/2) q[0];
u(pi/4,0,0) q[3];
u(0,0,pi/2) q[4];
cx q[0],q[4];
u(pi/4,-pi,pi/2) q[4];
cx q[0],q[4];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[4];
cx q[0],q[4];
u(pi/4,-pi,pi/2) q[4];
cx q[0],q[4];
u(pi/4,0,-pi) q[4];
cx q[3],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[3],q[4];
u(0,0,pi/4) q[3];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0.9980348646301893,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/2) q[0];
u(0.9980348646301903,-pi,-pi) q[3];
u(pi/2,pi/2,-3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,pi/2) q[4];
cx q[0],q[4];
u(pi/4,0,0) q[4];
cx q[0],q[4];
u(0.9980348646301903,-pi,-pi) q[4];
cx q[0],q[4];
u(0,0,pi/2) q[0];
u(0.9980348646301894,pi/2,0) q[4];
cx q[0],q[4];
u(pi/4,-pi,pi/2) q[4];
cx q[0],q[4];
u(pi/4,0,-pi) q[4];
cx q[3],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[3],q[4];
u(0,0,pi/4) q[3];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/2) q[0];
u(0,0,pi/2) q[3];
cx q[0],q[3];
u(pi/4,-pi,pi/2) q[3];
cx q[0],q[3];
u(0,0,-3*pi/4) q[0];
u(pi/4,-pi/4,0) q[3];
cx q[0],q[3];
u(0.3648573517862774,-pi,-pi/4) q[3];
cx q[0],q[3];
u(pi/2,0.3648573517862772,pi/2) q[3];
cx q[2],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[3];
cx q[2],q[3];
u(0,0,pi/4) q[2];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,-3*pi/4) q[2];
cx q[0],q[2];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[2];
u(pi/2,1.4922506383856682,-3*pi/4) q[3];
cx q[0],q[3];
u(0.5253852471287274,-pi,2.18700743192748) q[3];
cx q[0],q[3];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[3];
cx q[2],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[3];
cx q[2],q[3];
u(0,0,pi/4) q[2];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,-3*pi/4) q[2];
cx q[0],q[2];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[2];
cx q[0],q[2];
u(0.3648573517862774,-pi,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[2];
u(pi/2,0,-3*pi/4) q[3];
cx q[0],q[3];
u(0.4205408116111712,-pi,pi/2) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[3];
cx q[0],q[3];
u(pi/4,-pi,3*pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[3];
u(pi/2,pi/2,-3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,pi/2) q[4];
cx q[0],q[4];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[4];
cx q[0],q[4];
u(pi/4,-pi,pi/2) q[4];
cx q[0],q[4];
u(0,0,-3*pi/4) q[0];
u(pi/4,-pi/4,0) q[4];
cx q[0],q[4];
u(0.3648573517862774,-pi,-pi/4) q[4];
cx q[0],q[4];
u(pi/2,0.3648573517862772,pi/2) q[4];
cx q[3],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[3],q[4];
u(0,0,pi/4) q[3];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,-3*pi/4) q[3];
cx q[0],q[3];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[3];
u(pi/2,1.4922506383856682,-3*pi/4) q[4];
cx q[0],q[4];
u(0.5253852471287274,-pi,2.18700743192748) q[4];
cx q[0],q[4];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[4];
cx q[3],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[3],q[4];
u(0,0,pi/4) q[3];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,-3*pi/4) q[3];
cx q[0],q[3];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[3];
cx q[0],q[3];
u(0.3648573517862774,-pi,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[3];
u(pi/2,0,-3*pi/4) q[4];
cx q[0],q[4];
u(0.4205408116111712,-pi,pi/2) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[4];
cx q[0],q[4];
u(pi/4,-pi,3*pi/4) q[4];
cx q[0],q[4];
u(0,0,-3*pi/4) q[0];
u(pi/4,pi/2,0) q[4];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0.3648573517862774,-pi,-pi/4) q[5];
cx q[0],q[5];
u(pi/2,0.3648573517862772,pi/2) q[5];
cx q[4],q[5];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[5];
cx q[4],q[5];
u(0,0,pi/4) q[4];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,-3*pi/4) q[4];
cx q[0],q[4];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[4];
u(pi/2,1.4922506383856682,-3*pi/4) q[5];
cx q[0],q[5];
u(0.5253852471287274,-pi,2.18700743192748) q[5];
cx q[0],q[5];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[5];
cx q[4],q[5];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[5];
cx q[4],q[5];
u(0,0,pi/4) q[4];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,-3*pi/4) q[4];
cx q[0],q[4];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[4];
cx q[0],q[4];
u(0.3648573517862774,-pi,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[4];
u(pi/2,0,-3*pi/4) q[5];
cx q[0],q[5];
u(0.4205408116111712,-pi,pi/2) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
u(0.4205408116111712,-pi,0) q[5];
cx q[0],q[5];
u(pi/4,-pi,pi/2) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
u(pi/4,0,0) q[5];
u(0,0,pi/2) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(pi/4,0,-pi) q[6];
cx q[5],q[6];
u(0,0,-pi/4) q[6];
cx q[0],q[6];
u(0,0,pi/4) q[6];
cx q[5],q[6];
u(0,0,pi/4) q[5];
u(0,0,-pi/4) q[6];
cx q[0],q[6];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(2.813468447840606,-pi,3*pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
u(2.813468447840605,0,0) q[5];
u(pi/2,pi/2,-3*pi/4) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(pi/4,0,0) q[6];
cx q[0],q[6];
u(2.813468447840605,0,0) q[6];
cx q[0],q[6];
u(0,0,pi/2) q[0];
u(2.8134684478406053,-pi/2,-pi) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(pi/4,0,-pi) q[6];
cx q[5],q[6];
u(0,0,-pi/4) q[6];
cx q[0],q[6];
u(0,0,pi/4) q[6];
cx q[5],q[6];
u(0,0,pi/4) q[5];
u(0,0,-pi/4) q[6];
cx q[0],q[6];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
u(0,0,pi/2) q[5];
cx q[0],q[5];
u(pi/4,-pi,pi/2) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(pi/4,-pi/4,0) q[5];
cx q[0],q[5];
u(pi/4,-pi,3*pi/4) q[5];
cx q[0],q[5];
u(pi/4,pi/2,0) q[5];
cx q[5],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[5],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(pi/2,1.4922506383856682,-3*pi/4) q[4];
u(0,0,pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[4];
u(0.5253852471287274,-pi,2.18700743192748) q[4];
cx q[0],q[4];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[4];
u(0,0,3*pi/4) q[5];
cx q[0],q[5];
u(pi/4,-pi,-3*pi/4) q[5];
cx q[0],q[5];
u(pi/4,-pi,0) q[5];
cx q[5],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[5],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(pi/2,0,-3*pi/4) q[4];
u(0,0,pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
cx q[0],q[4];
u(0.4205408116111712,-pi,pi/2) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[4];
cx q[0],q[4];
u(pi/4,-pi,3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,pi/2,0) q[4];
cx q[4],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[3];
cx q[4],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(pi/2,1.4922506383856682,-3*pi/4) q[3];
u(0,0,pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[3];
u(0.5253852471287274,-pi,2.18700743192748) q[3];
cx q[0],q[3];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[3];
u(0,0,3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,-3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,0) q[4];
cx q[4],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[3];
cx q[4],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(pi/2,0,-3*pi/4) q[3];
u(0,0,pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/2) q[0];
cx q[0],q[3];
u(0.4205408116111712,-pi,pi/2) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[3];
cx q[0],q[3];
u(pi/4,-pi,3*pi/4) q[3];
cx q[0],q[3];
u(pi/4,pi/2,0) q[3];
cx q[3],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[2];
cx q[3],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(pi/2,1.4922506383856682,-3*pi/4) q[2];
u(0,0,pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[2];
u(0.5253852471287274,-pi,2.18700743192748) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[2];
u(0,0,3*pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,-3*pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,0) q[3];
cx q[3],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[2];
cx q[3],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(pi/2,0,-3*pi/4) q[2];
u(0,0,pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/2) q[0];
cx q[0],q[2];
u(0.4205408116111712,-pi,pi/2) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[2];
cx q[0],q[2];
u(pi/4,-pi,3*pi/4) q[2];
cx q[0],q[2];
u(pi/4,pi/2,0) q[2];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/4) q[1];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(pi/2,1.4922506383856682,-3*pi/4) q[1];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[1];
u(0.5253852471287274,-pi,2.18700743192748) q[1];
cx q[0],q[1];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[1];
u(0,0,3*pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,-3*pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,0) q[2];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/4) q[1];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(pi/2,0,-3*pi/4) q[1];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/2) q[0];
cx q[0],q[1];
u(0.4205408116111712,-pi,pi/2) q[1];
cx q[0],q[1];
u(0.4205408116111712,pi/2,0) q[1];
cx q[0],q[1];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(pi/4,-pi,3*pi/4) q[1];
cx q[0],q[1];
u(0,0,-pi/4) q[0];
u(pi/4,pi/2,0) q[1];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,-3*pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi/2,0) q[2];
cx q[0],q[2];
u(0,0,-3*pi/4) q[0];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0.3648573517862774,-pi,-pi/4) q[2];
cx q[0],q[2];
u(pi/2,0.3648573517862772,pi/2) q[2];
cx q[1],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[2];
cx q[1],q[2];
u(0,0,pi/4) q[1];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
cx q[0],q[1];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[1];
cx q[0],q[1];
u(pi/4,-pi,-3*pi/4) q[1];
cx q[0],q[1];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[1];
u(pi/2,1.4922506383856682,-3*pi/4) q[2];
cx q[0],q[2];
u(0.5253852471287274,-pi,2.18700743192748) q[2];
cx q[0],q[2];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[2];
cx q[1],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[2];
cx q[1],q[2];
u(0,0,pi/4) q[1];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
cx q[0],q[1];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[1];
cx q[0],q[1];
u(pi/4,-pi,-3*pi/4) q[1];
cx q[0],q[1];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[1];
cx q[0],q[1];
u(0.3648573517862774,-pi,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[1];
u(pi/2,0,-3*pi/4) q[2];
cx q[0],q[2];
u(0.4205408116111712,-pi,pi/2) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[2];
cx q[0],q[2];
u(pi/4,-pi,3*pi/4) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[0];
u(pi/4,pi/2,0) q[2];
u(0,0,pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,-3*pi/4) q[3];
cx q[0],q[3];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[3];
cx q[0],q[3];
u(0.3648573517862774,-pi,-pi/4) q[3];
cx q[0],q[3];
u(pi/2,0.3648573517862772,pi/2) q[3];
cx q[2],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[3];
cx q[2],q[3];
u(0,0,pi/4) q[2];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,-3*pi/4) q[2];
cx q[0],q[2];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[2];
u(pi/2,1.4922506383856682,-3*pi/4) q[3];
cx q[0],q[3];
u(0.5253852471287274,-pi,2.18700743192748) q[3];
cx q[0],q[3];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[3];
cx q[2],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[3];
cx q[2],q[3];
u(0,0,pi/4) q[2];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,-3*pi/4) q[2];
cx q[0],q[2];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[2];
cx q[0],q[2];
u(0.3648573517862774,-pi,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[2];
u(pi/2,0,-3*pi/4) q[3];
cx q[0],q[3];
u(0.4205408116111712,-pi,pi/2) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[3];
cx q[0],q[3];
u(pi/4,-pi,3*pi/4) q[3];
cx q[0],q[3];
u(0,0,-pi/4) q[0];
u(pi/4,pi/2,0) q[3];
u(0,0,pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,-3*pi/4) q[4];
cx q[0],q[4];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[4];
cx q[0],q[4];
u(0.3648573517862774,-pi,-pi/4) q[4];
cx q[0],q[4];
u(pi/2,0.3648573517862772,pi/2) q[4];
cx q[3],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[3],q[4];
u(0,0,pi/4) q[3];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,-3*pi/4) q[3];
cx q[0],q[3];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[3];
u(pi/2,1.4922506383856682,-3*pi/4) q[4];
cx q[0],q[4];
u(0.5253852471287274,-pi,2.18700743192748) q[4];
cx q[0],q[4];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[4];
cx q[3],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[3],q[4];
u(0,0,pi/4) q[3];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,-3*pi/4) q[3];
cx q[0],q[3];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[3];
cx q[0],q[3];
u(0.3648573517862774,-pi,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[3];
u(pi/2,0,-3*pi/4) q[4];
cx q[0],q[4];
u(0.4205408116111712,-pi,pi/2) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[4];
cx q[0],q[4];
u(pi/4,-pi,3*pi/4) q[4];
cx q[0],q[4];
u(0,0,-pi/4) q[0];
u(pi/4,pi/2,0) q[4];
u(0,0,pi/4) q[5];
cx q[0],q[5];
u(pi/4,-pi,-3*pi/4) q[5];
cx q[0],q[5];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[5];
cx q[0],q[5];
u(0.3648573517862774,-pi,-pi/4) q[5];
cx q[0],q[5];
u(pi/2,0.3648573517862772,pi/2) q[5];
cx q[4],q[5];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[5];
cx q[4],q[5];
u(0,0,pi/4) q[4];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,-3*pi/4) q[4];
cx q[0],q[4];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[4];
u(pi/2,1.4922506383856682,-3*pi/4) q[5];
cx q[0],q[5];
u(0.5253852471287274,-pi,2.18700743192748) q[5];
cx q[0],q[5];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[5];
cx q[4],q[5];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[5];
cx q[4],q[5];
u(0,0,pi/4) q[4];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,-3*pi/4) q[4];
cx q[0],q[4];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[4];
cx q[0],q[4];
u(0.3648573517862774,-pi,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[4];
u(pi/2,0,-3*pi/4) q[5];
cx q[0],q[5];
u(0.4205408116111712,-pi,pi/2) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
u(0.4205408116111712,-pi,0) q[5];
cx q[0],q[5];
u(pi/4,-pi,pi/2) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
u(pi/4,0,0) q[5];
u(pi/2,pi/2,-3*pi/4) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(pi/4,0,-pi) q[6];
cx q[5],q[6];
u(0,0,-pi/4) q[6];
cx q[0],q[6];
u(0,0,pi/4) q[6];
cx q[5],q[6];
u(0,0,pi/4) q[5];
u(0,0,-pi/4) q[6];
cx q[0],q[6];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(1.97981562699413,-pi,3*pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
u(1.9798156269941372,0,0) q[5];
u(pi/2,pi/2,-3*pi/4) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(pi/4,0,0) q[6];
cx q[0],q[6];
u(1.9798156269941372,0,0) q[6];
cx q[0],q[6];
u(0,0,pi/2) q[0];
u(1.9798156269941298,-pi/2,-pi) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(pi/4,0,-pi) q[6];
cx q[5],q[6];
u(0,0,-pi/4) q[6];
cx q[0],q[6];
u(0,0,pi/4) q[6];
cx q[5],q[6];
u(0,0,pi/4) q[5];
u(0,0,-pi/4) q[6];
cx q[0],q[6];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
u(0,0,pi/2) q[5];
cx q[0],q[5];
u(pi/4,-pi,pi/2) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(pi/4,-pi/4,0) q[5];
cx q[0],q[5];
u(pi/4,-pi,3*pi/4) q[5];
cx q[0],q[5];
u(pi/4,pi/2,0) q[5];
cx q[5],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[5],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(pi/2,1.4922506383856682,-3*pi/4) q[4];
u(0,0,pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[4];
u(0.5253852471287274,-pi,2.18700743192748) q[4];
cx q[0],q[4];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[4];
u(0,0,3*pi/4) q[5];
cx q[0],q[5];
u(pi/4,-pi,-3*pi/4) q[5];
cx q[0],q[5];
u(pi/4,-pi,0) q[5];
cx q[5],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[4];
cx q[5],q[4];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(pi/2,0,-3*pi/4) q[4];
u(0,0,pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
cx q[0],q[4];
u(0.4205408116111712,-pi,pi/2) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[4];
cx q[0],q[4];
u(pi/4,-pi,3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,pi/2,0) q[4];
cx q[4],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[3];
cx q[4],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(pi/2,1.4922506383856682,-3*pi/4) q[3];
u(0,0,pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[3];
u(0.5253852471287274,-pi,2.18700743192748) q[3];
cx q[0],q[3];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[3];
u(0,0,3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,-3*pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,0) q[4];
cx q[4],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[3];
cx q[4],q[3];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(pi/2,0,-3*pi/4) q[3];
u(0,0,pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[4];
cx q[0],q[4];
u(0,0,pi/2) q[0];
cx q[0],q[3];
u(0.4205408116111712,-pi,pi/2) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[3];
cx q[0],q[3];
u(pi/4,-pi,3*pi/4) q[3];
cx q[0],q[3];
u(pi/4,pi/2,0) q[3];
cx q[3],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[2];
cx q[3],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(pi/2,1.4922506383856682,-3*pi/4) q[2];
u(0,0,pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[2];
u(0.5253852471287274,-pi,2.18700743192748) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[2];
u(0,0,3*pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,-3*pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,0) q[3];
cx q[3],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[2];
cx q[3],q[2];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(pi/2,0,-3*pi/4) q[2];
u(0,0,pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[3];
cx q[0],q[3];
u(0,0,pi/2) q[0];
cx q[0],q[2];
u(0.4205408116111712,-pi,pi/2) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[2];
cx q[0],q[2];
u(pi/4,-pi,3*pi/4) q[2];
cx q[0],q[2];
u(pi/4,pi/2,0) q[2];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/4) q[1];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(pi/2,1.4922506383856682,-3*pi/4) q[1];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[1];
u(0.5253852471287274,-pi,2.18700743192748) q[1];
cx q[0],q[1];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[1];
u(0,0,3*pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,-3*pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,0) q[2];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/4) q[1];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(pi/2,0,-3*pi/4) q[1];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[2];
cx q[0],q[2];
u(0,0,pi/2) q[0];
cx q[0],q[1];
u(0.4205408116111712,-pi,pi/2) q[1];
cx q[0],q[1];
u(0,0,-pi/4) q[0];
u(0.4205408116111712,pi/2,0) q[1];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/4,-pi,-3*pi/4) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[0];
u(pi/4,-pi/2,0) q[2];
u(0,0,pi/4) q[3];
cx q[0],q[3];
u(pi/4,-pi,-3*pi/4) q[3];
cx q[0],q[3];
u(0,0,-pi/4) q[0];
u(pi/4,-pi/2,0) q[3];
u(0,0,pi/4) q[4];
cx q[0],q[4];
u(pi/4,-pi,-3*pi/4) q[4];
cx q[0],q[4];
u(0,0,-pi/4) q[0];
u(pi/4,-pi/2,0) q[4];
u(0,0,pi/4) q[5];
cx q[0],q[5];
u(pi/4,-pi,-3*pi/4) q[5];
cx q[0],q[5];
u(0,0,pi/2) q[0];
u(pi/4,-pi/2,0) q[5];
u(pi/2,pi/2,-3*pi/4) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[6];
cx q[0],q[6];
u(pi/4,-pi,pi/2) q[6];
cx q[0],q[6];
u(0,0,pi/4) q[0];
u(pi/4,0,0) q[6];
u(pi,-pi/4,-pi) q[7];
cx q[0],q[7];
u(pi/4,-pi,3*pi/4) q[7];
cx q[0],q[7];
u(0,0,-3*pi/4) q[0];
u(pi/4,pi/2,0) q[7];
u(pi,-pi/4,-pi) q[8];
cx q[0],q[8];
u(0.3648573517862774,-pi,-pi/4) q[8];
cx q[0],q[8];
u(pi/2,0.3648573517862772,pi/2) q[8];
cx q[7],q[8];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(0,0,pi/4) q[8];
cx q[7],q[8];
u(0,0,pi/4) q[7];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
cx q[0],q[7];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[7];
cx q[0],q[7];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[7];
cx q[0],q[7];
u(pi/4,-pi,-3*pi/4) q[7];
cx q[0],q[7];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[7];
u(pi/2,1.4922506383856682,-3*pi/4) q[8];
cx q[0],q[8];
u(0.5253852471287274,-pi,2.18700743192748) q[8];
cx q[0],q[8];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[8];
cx q[7],q[8];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(0,0,pi/4) q[8];
cx q[7],q[8];
u(0,0,pi/4) q[7];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
cx q[0],q[7];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[7];
cx q[0],q[7];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[7];
cx q[0],q[7];
u(pi/4,-pi,-3*pi/4) q[7];
cx q[0],q[7];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[7];
cx q[0],q[7];
u(0.3648573517862774,-pi,-pi/4) q[7];
cx q[0],q[7];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[7];
u(pi/2,0,-3*pi/4) q[8];
cx q[0],q[8];
u(0.4205408116111712,-pi,pi/2) q[8];
cx q[0],q[8];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[8];
cx q[0],q[8];
u(pi/4,-pi,3*pi/4) q[8];
cx q[0],q[8];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[8];
u(pi,pi/2,-pi) q[9];
cx q[0],q[9];
u(pi/4,-pi,pi/2) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
u(pi/4,0,0) q[9];
u(0,0,pi/2) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(pi/4,0,-pi) q[10];
cx q[9],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[10];
cx q[9],q[10];
u(0,0,pi/4) q[9];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(0.9980348646301893,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
u(0.9980348646301903,-pi,-pi) q[9];
u(pi/2,pi/2,-3*pi/4) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(pi/4,0,0) q[10];
cx q[0],q[10];
u(0.9980348646301903,-pi,-pi) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
u(0.9980348646301894,pi/2,0) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(pi/4,0,-pi) q[10];
cx q[9],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[10];
cx q[9],q[10];
u(0,0,pi/4) q[9];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
u(0,0,pi/2) q[9];
cx q[0],q[9];
u(pi/4,-pi,pi/2) q[9];
cx q[0],q[9];
u(0,0,-3*pi/4) q[0];
u(pi/4,-pi/4,0) q[9];
cx q[0],q[9];
u(0.3648573517862774,-pi,-pi/4) q[9];
cx q[0],q[9];
u(pi/2,0.3648573517862772,pi/2) q[9];
cx q[8],q[9];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/4) q[9];
cx q[8],q[9];
u(0,0,pi/4) q[8];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
cx q[0],q[8];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[8];
cx q[0],q[8];
u(pi/4,-pi,-3*pi/4) q[8];
cx q[0],q[8];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[8];
u(pi/2,1.4922506383856682,-3*pi/4) q[9];
cx q[0],q[9];
u(0.5253852471287274,-pi,2.18700743192748) q[9];
cx q[0],q[9];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[9];
cx q[8],q[9];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/4) q[9];
cx q[8],q[9];
u(0,0,pi/4) q[8];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
cx q[0],q[8];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[8];
cx q[0],q[8];
u(pi/4,-pi,-3*pi/4) q[8];
cx q[0],q[8];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[8];
cx q[0],q[8];
u(0.3648573517862774,-pi,-pi/4) q[8];
cx q[0],q[8];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[8];
u(pi/2,0,-3*pi/4) q[9];
cx q[0],q[9];
u(0.4205408116111712,-pi,pi/2) q[9];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[9];
cx q[0],q[9];
u(pi/4,-pi,3*pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[9];
u(pi/2,pi/2,-3*pi/4) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(0,0,-3*pi/4) q[0];
u(pi/4,-pi/4,0) q[10];
cx q[0],q[10];
u(0.3648573517862774,-pi,-pi/4) q[10];
cx q[0],q[10];
u(pi/2,0.3648573517862772,pi/2) q[10];
cx q[9],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[10];
cx q[9],q[10];
u(0,0,pi/4) q[9];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[9];
cx q[0],q[9];
u(pi/4,-pi,-3*pi/4) q[9];
cx q[0],q[9];
u(0,0,0.9545852216623132) q[0];
u(pi/4,-pi,0) q[9];
u(pi/2,1.4922506383856682,-3*pi/4) q[10];
cx q[0],q[10];
u(0.5253852471287274,-pi,2.18700743192748) q[10];
cx q[0],q[10];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[10];
cx q[9],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[10];
cx q[9],q[10];
u(0,0,pi/4) q[9];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[9];
cx q[0],q[9];
u(pi/4,-pi,-3*pi/4) q[9];
cx q[0],q[9];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[9];
cx q[0],q[9];
u(0.3648573517862774,-pi,-pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[9];
u(pi/2,0,-3*pi/4) q[10];
cx q[0],q[10];
u(0.4205408116111712,-pi,pi/2) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[10];
cx q[0],q[10];
u(pi/4,-pi,3*pi/4) q[10];
cx q[0],q[10];
u(0,0,-3*pi/4) q[0];
u(pi/4,pi/2,0) q[10];
u(0,0,-pi/4) q[11];
cx q[0],q[11];
u(0.3648573517862774,-pi,-pi/4) q[11];
cx q[0],q[11];
u(pi/2,0.3648573517862772,pi/2) q[11];
cx q[10],q[11];
u(0,0,-pi/4) q[11];
cx q[0],q[11];
u(0,0,pi/4) q[11];
cx q[10],q[11];
u(0,0,pi/4) q[10];
u(0,0,-pi/4) q[11];
cx q[0],q[11];
cx q[0],q[10];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,0.9545852216623132) q[0];
u(0,0,3*pi/4) q[10];
u(pi/2,1.4922506383856682,-3*pi/4) q[11];
cx q[0],q[11];
u(0.5253852471287274,-pi,2.18700743192748) q[11];
cx q[0],q[11];
u(0,0,-pi/4) q[0];
cx q[0],q[10];
u(pi/4,-pi,-3*pi/4) q[10];
cx q[0],q[10];
u(pi/4,-pi,0) q[10];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[11];
cx q[10],q[11];
u(0,0,-pi/4) q[11];
cx q[0],q[11];
u(0,0,pi/4) q[11];
cx q[10],q[11];
u(0,0,pi/4) q[10];
u(0,0,-pi/4) q[11];
cx q[0],q[11];
cx q[0],q[10];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
u(0,0,pi/4) q[10];
u(pi/2,0,-3*pi/4) q[11];
cx q[0],q[11];
u(0.4205408116111712,-pi,pi/2) q[11];
cx q[0],q[11];
u(0,0,pi/2) q[0];
u(0.4205408116111712,-pi,0) q[11];
cx q[0],q[11];
u(pi/4,-pi,pi/2) q[11];
cx q[0],q[11];
u(0,0,-pi/4) q[0];
cx q[0],q[10];
u(pi/4,-pi,-3*pi/4) q[10];
cx q[0],q[10];
u(0,0,-3*pi/4) q[0];
u(pi/4,-3*pi/4,0) q[10];
cx q[0],q[10];
u(0.3648573517862774,-pi,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
u(pi/2,0.3648573517862772,pi/2) q[10];
u(pi/4,0,0) q[11];
u(0,0,pi/2) q[12];
cx q[0],q[12];
u(pi/4,-pi,pi/2) q[12];
cx q[0],q[12];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[12];
cx q[0],q[12];
u(pi/4,-pi,pi/2) q[12];
cx q[0],q[12];
u(pi/4,0,-pi) q[12];
cx q[11],q[12];
u(0,0,-pi/4) q[12];
cx q[0],q[12];
u(0,0,pi/4) q[12];
cx q[11],q[12];
u(0,0,pi/4) q[11];
u(0,0,-pi/4) q[12];
cx q[0],q[12];
cx q[0],q[11];
u(0,0,pi/4) q[0];
u(2.813468447840606,-pi,3*pi/4) q[11];
cx q[0],q[11];
u(0,0,pi/2) q[0];
u(2.813468447840605,0,0) q[11];
u(pi/2,pi/2,-3*pi/4) q[12];
cx q[0],q[12];
u(pi/4,-pi,pi/2) q[12];
cx q[0],q[12];
u(pi/4,0,0) q[12];
cx q[0],q[12];
u(2.813468447840605,0,0) q[12];
cx q[0],q[12];
u(0,0,pi/2) q[0];
u(2.8134684478406053,-pi/2,-pi) q[12];
cx q[0],q[12];
u(pi/4,-pi,pi/2) q[12];
cx q[0],q[12];
u(pi/4,0,-pi) q[12];
cx q[11],q[12];
u(0,0,-pi/4) q[12];
cx q[0],q[12];
u(0,0,pi/4) q[12];
cx q[11],q[12];
u(0,0,pi/4) q[11];
u(0,0,-pi/4) q[12];
cx q[0],q[12];
cx q[0],q[11];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[11];
cx q[0],q[11];
u(0,0,pi/2) q[0];
u(0,0,pi/2) q[11];
cx q[0],q[11];
u(pi/4,-pi,pi/2) q[11];
cx q[0],q[11];
u(0,0,pi/4) q[0];
u(pi/4,-pi/4,0) q[11];
cx q[0],q[11];
u(pi/4,-pi,3*pi/4) q[11];
cx q[0],q[11];
u(pi/4,pi/2,0) q[11];
cx q[11],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[10];
cx q[11],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(pi/2,1.4922506383856682,-3*pi/4) q[10];
u(0,0,pi/4) q[11];
cx q[0],q[11];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[11];
cx q[0],q[11];
u(0,0,-pi/4) q[0];
u(0,0,3*pi/4) q[11];
cx q[0],q[11];
u(pi/4,-pi,-3*pi/4) q[11];
cx q[0],q[11];
u(0,0,pi/2) q[0];
u(pi/4,-pi,0) q[11];
u(pi/2,pi/2,-3*pi/4) q[12];
cx q[0],q[12];
u(pi/4,-pi,pi/2) q[12];
cx q[0],q[12];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[12];
cx q[0],q[12];
u(pi/4,-pi,pi/2) q[12];
cx q[0],q[12];
u(0,0,pi/2) q[0];
u(pi/4,0,0) q[12];
cx q[0],q[12];
u(pi/2,-pi,pi/2) q[12];
cx q[0],q[12];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[10];
u(0.5253852471287274,-pi,2.18700743192748) q[10];
cx q[0],q[10];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[10];
cx q[11],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[10];
cx q[11],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(pi/2,0,-3*pi/4) q[10];
u(0,0,pi/4) q[11];
cx q[0],q[11];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[11];
cx q[0],q[11];
u(0,0,-pi/4) q[0];
u(0,0,pi/4) q[11];
cx q[0],q[11];
u(pi/4,-pi,-3*pi/4) q[11];
cx q[0],q[11];
u(0,0,pi/2) q[0];
u(pi/4,-pi/2,0) q[11];
cx q[0],q[11];
u(pi/2,-pi,pi/2) q[11];
cx q[0],q[11];
u(0,0,pi/2) q[0];
cx q[0],q[10];
u(0.4205408116111712,-pi,pi/2) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[10];
cx q[0],q[10];
u(pi/4,-pi,3*pi/4) q[10];
cx q[0],q[10];
u(pi/4,pi/2,0) q[10];
cx q[10],q[9];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/4) q[9];
cx q[10],q[9];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(pi/2,1.4922506383856682,-3*pi/4) q[9];
u(0,0,pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[9];
u(0.5253852471287274,-pi,2.18700743192748) q[9];
cx q[0],q[9];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[9];
u(0,0,3*pi/4) q[10];
cx q[0],q[10];
u(pi/4,-pi,-3*pi/4) q[10];
cx q[0],q[10];
u(pi/4,-pi,0) q[10];
cx q[10],q[9];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/4) q[9];
cx q[10],q[9];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(pi/2,0,-3*pi/4) q[9];
u(0,0,pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
cx q[0],q[9];
u(0.4205408116111712,-pi,pi/2) q[9];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[9];
cx q[0],q[9];
u(pi/4,-pi,3*pi/4) q[9];
cx q[0],q[9];
u(pi/4,pi/2,0) q[9];
cx q[9],q[8];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(0,0,pi/4) q[8];
cx q[9],q[8];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(pi/2,1.4922506383856682,-3*pi/4) q[8];
u(0,0,pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[8];
u(0.5253852471287274,-pi,2.18700743192748) q[8];
cx q[0],q[8];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[8];
u(0,0,3*pi/4) q[9];
cx q[0],q[9];
u(pi/4,-pi,-3*pi/4) q[9];
cx q[0],q[9];
u(pi/4,-pi,0) q[9];
cx q[9],q[8];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(0,0,pi/4) q[8];
cx q[9],q[8];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(pi/2,0,-3*pi/4) q[8];
u(0,0,pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
cx q[0],q[8];
u(0.4205408116111712,-pi,pi/2) q[8];
cx q[0],q[8];
u(0,0,pi/4) q[0];
u(0.4205408116111712,pi/4,0) q[8];
cx q[0],q[8];
u(pi/4,-pi,3*pi/4) q[8];
cx q[0],q[8];
u(pi/4,pi/2,0) q[8];
cx q[8],q[7];
u(0,0,-pi/4) q[7];
cx q[0],q[7];
u(0,0,pi/4) q[7];
cx q[8],q[7];
u(0,0,-pi/4) q[7];
cx q[0],q[7];
u(pi/2,1.4922506383856682,-3*pi/4) q[7];
u(0,0,pi/4) q[8];
cx q[0],q[8];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(0,0,0.9545852216623132) q[0];
cx q[0],q[7];
u(0.5253852471287274,-pi,2.18700743192748) q[7];
cx q[0],q[7];
u(0,0,-pi/4) q[0];
u(1.1254377896453873,0.2886110755355906,2.5381420683163585) q[7];
u(0,0,3*pi/4) q[8];
cx q[0],q[8];
u(pi/4,-pi,-3*pi/4) q[8];
cx q[0],q[8];
u(pi/4,-pi,0) q[8];
cx q[8],q[7];
u(0,0,-pi/4) q[7];
cx q[0],q[7];
u(0,0,pi/4) q[7];
cx q[8],q[7];
u(0,0,-pi/4) q[7];
cx q[0],q[7];
u(pi/2,0,-3*pi/4) q[7];
u(0,0,pi/4) q[8];
cx q[0],q[8];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[8];
cx q[0],q[8];
u(0,0,pi/2) q[0];
cx q[0],q[7];
u(0.4205408116111712,-pi,pi/2) q[7];
cx q[0],q[7];
u(0,0,-pi/4) q[0];
u(0.4205408116111712,pi/2,0) q[7];
u(0,0,pi/4) q[8];
cx q[0],q[8];
u(pi/4,-pi,-3*pi/4) q[8];
cx q[0],q[8];
u(0,0,-pi/4) q[0];
u(pi/4,-pi/2,0) q[8];
u(0,0,pi/4) q[9];
cx q[0],q[9];
u(pi/4,-pi,-3*pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
u(pi/4,0,0) q[9];
cx q[0],q[9];
u(pi/4,-pi,pi/2) q[9];
cx q[0],q[9];
u(0,0,-pi/4) q[0];
u(pi/4,0,0) q[9];
u(0,0,pi/4) q[10];
cx q[0],q[10];
u(pi/4,-pi,-3*pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
u(pi/4,0,0) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(pi/4,0,-pi) q[10];
cx q[9],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[10];
cx q[9],q[10];
u(0,0,pi/4) q[9];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(1.97981562699413,-pi,3*pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
u(1.9798156269941372,0,0) q[9];
u(pi/2,pi/2,-3*pi/4) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(pi/4,0,0) q[10];
cx q[0],q[10];
u(1.9798156269941372,0,0) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
u(1.9798156269941298,-pi/2,-pi) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(pi/4,0,-pi) q[10];
cx q[9],q[10];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
u(0,0,pi/4) q[10];
cx q[9],q[10];
u(0,0,pi/4) q[9];
u(0,0,-pi/4) q[10];
cx q[0],q[10];
cx q[0],q[9];
u(0,0,pi/4) q[0];
u(0,0,-pi/4) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
u(0,0,pi/2) q[9];
cx q[0],q[9];
u(pi/4,-pi,pi/2) q[9];
cx q[0],q[9];
u(0,0,pi/2) q[0];
u(pi/4,0,0) q[9];
u(pi/2,pi/2,-3*pi/4) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(0,0,pi/2) q[0];
u(pi/4,pi/2,0) q[10];
cx q[0],q[10];
u(pi/4,-pi,pi/2) q[10];
cx q[0],q[10];
u(pi/2,0,-0.918896744766986) q[0];
u(pi/4,0,0) q[10];
u(pi/2,pi/2,0) q[11];
u(pi/2,pi/2,0) q[12];
measure q[0] -> c[0];
