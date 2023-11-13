OPENQASM 2.0;
include "qelib1.inc";
gate unitary q0,q1 { u(1.1146432749428323,-1.2372331854511032,2.614548866095893) q0; u(1.7278194331801149,1.9393420936317858,2.8140050655607087) q1; cx q0,q1; u(0.6924392015919727,-pi/2,-pi/2) q0; u(1.180935693808513,0.7561895602274635,0.38303504202921124) q1; cx q0,q1; u(0.009704567507648018,-pi,-pi/2) q0; u(1.0497790530438142,-2.4587918261666215,0.5492146463295438) q1; cx q0,q1; u(2.5967186703755103,1.978117992369409,-0.04078645185241481) q0; u(1.6480001796253858,-0.13751858861240152,2.7331827445086603) q1; }
gate unitary_5659863504 q0,q1 { u(1.579418713609123,0.7680040384297819,0.38488655031165475) q0; u(1.0331683818404602,1.8280184201020502,2.7796421379151033) q1; cx q0,q1; u(0.9190720841820982,-pi/2,-pi/2) q0; u(1.3269860774432936,0.8046333814535735,0.22824153537827918) q1; cx q0,q1; u(0.5750428643924093,0,pi/2) q0; u(1.0497790530438142,-2.4587918261666215,0.5492146463295438) q1; cx q0,q1; u(1.0434620796950314,0.045657736600237975,-1.4756066516699464) q0; u(1.4207531179245934,2.259433171247916,0.6379190010481484) q1; }
gate quantum_volume__5_1_42_ q0,q1,q2,q3,q4 { unitary q3,q2; unitary_5659863504 q0,q4; }
qreg q[5];
quantum_volume__5_1_42_ q[0],q[1],q[2],q[3],q[4];
