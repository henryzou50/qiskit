OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[0],q[10];
cx q[11],q[4];
cx q[8],q[22];
cx q[2],q[26];
cx q[20],q[7];
cx q[25],q[5];
cx q[13],q[22];
cx q[25],q[5];
cx q[15],q[0];
cx q[21],q[9];
cx q[1],q[7];
cx q[8],q[16];
cx q[10],q[19];
cx q[3],q[18];
cx q[11],q[23];
cx q[25],q[20];
cx q[12],q[19];
cx q[24],q[15];
cx q[9],q[21];
cx q[8],q[16];
cx q[18],q[6];
cx q[2],q[26];
cx q[14],q[10];
cx q[23],q[11];
cx q[21],q[9];
cx q[13],q[1];
cx q[6],q[3];
cx q[8],q[24];
cx q[2],q[17];
cx q[3],q[6];
cx q[1],q[18];
cx q[14],q[19];
cx q[9],q[11];
cx q[12],q[20];
cx q[7],q[21];
cx q[13],q[25];
cx q[15],q[24];
cx q[17],q[4];
cx q[21],q[7];
cx q[13],q[1];
cx q[14],q[19];
cx q[11],q[4];
cx q[25],q[20];
cx q[14],q[19];
cx q[0],q[18];
cx q[16],q[8];
cx q[4],q[23];
cx q[15],q[24];
cx q[12],q[10];
cx q[9],q[17];
cx q[18],q[1];
cx q[20],q[13];
cx q[4],q[17];
cx q[2],q[7];
cx q[24],q[22];
cx q[16],q[0];
cx q[9],q[21];
cx q[7],q[26];
cx q[4],q[23];
cx q[3],q[6];
cx q[14],q[5];
cx q[10],q[15];
cx q[20],q[25];
cx q[22],q[24];
cx q[8],q[16];
cx q[0],q[13];
cx q[18],q[13];
cx q[2],q[26];
cx q[15],q[19];
cx q[22],q[24];
cx q[8],q[16];
cx q[12],q[5];
cx q[9],q[21];
cx q[7],q[2];
cx q[5],q[12];
cx q[14],q[15];
cx q[22],q[24];
cx q[8],q[16];
cx q[20],q[1];
cx q[5],q[12];
cx q[16],q[8];
cx q[15],q[22];
cx q[11],q[21];
cx q[18],q[4];
cx q[25],q[13];
cx q[25],q[0];
cx q[15],q[19];
cx q[7],q[26];
cx q[14],q[12];
cx q[22],q[8];
cx q[24],q[10];
cx q[23],q[17];
cx q[18],q[4];
cx q[11],q[9];
cx q[20],q[1];
cx q[23],q[6];
cx q[15],q[19];
cx q[4],q[18];
cx q[11],q[21];
cx q[16],q[22];
cx q[24],q[10];
cx q[1],q[20];
cx q[24],q[19];
cx q[8],q[22];
cx q[21],q[17];
cx q[7],q[2];
cx q[0],q[25];
cx q[1],q[26];
cx q[11],q[9];
cx q[17],q[4];
cx q[21],q[9];
cx q[18],q[1];
cx q[24],q[22];
cx q[5],q[14];
cx q[8],q[13];
cx q[25],q[20];
cx q[11],q[23];
cx q[15],q[19];
cx q[7],q[26];
cx q[23],q[17];
cx q[8],q[16];
cx q[13],q[1];
cx q[18],q[7];
cx q[3],q[4];
cx q[25],q[2];
cx q[24],q[22];
cx q[19],q[5];
cx q[20],q[12];
cx q[21],q[11];
cx q[2],q[26];
cx q[15],q[12];
cx q[18],q[7];
cx q[24],q[13];
cx q[5],q[10];
cx q[4],q[23];
cx q[1],q[0];
cx q[7],q[25];
cx q[22],q[10];
cx q[9],q[17];
cx q[21],q[2];
cx q[5],q[15];
cx q[18],q[0];
cx q[23],q[4];
cx q[12],q[14];
cx q[18],q[0];
cx q[14],q[12];
cx q[23],q[1];
cx q[22],q[10];
cx q[21],q[6];
cx q[7],q[25];
cx q[8],q[13];
cx q[3],q[11];
cx q[26],q[2];
cx q[24],q[10];
cx q[20],q[15];
cx q[25],q[6];
cx q[23],q[9];
cx q[14],q[5];
cx q[16],q[8];
cx q[17],q[23];
cx q[10],q[24];
cx q[5],q[14];
cx q[21],q[6];
cx q[9],q[1];
cx q[4],q[0];
cx q[16],q[0];
cx q[7],q[5];
cx q[3],q[23];
cx q[10],q[22];
cx q[11],q[6];
cx q[18],q[12];
cx q[2],q[25];
cx q[13],q[8];
cx q[8],q[16];
cx q[15],q[7];
cx q[2],q[26];
cx q[11],q[23];
cx q[25],q[18];
cx q[10],q[24];
cx q[9],q[12];
cx q[20],q[14];
cx q[6],q[21];
cx q[1],q[4];
cx q[18],q[2];
cx q[21],q[6];
cx q[8],q[10];
cx q[9],q[4];
cx q[25],q[5];
cx q[5],q[25];
cx q[6],q[2];
cx q[4],q[9];
cx q[19],q[20];
cx q[24],q[22];
cx q[25],q[0];
cx q[18],q[21];
cx q[11],q[23];
cx q[2],q[3];
cx q[16],q[7];
cx q[25],q[18];
cx q[16],q[12];
cx q[20],q[0];
cx q[24],q[19];
cx q[9],q[17];
cx q[6],q[3];
cx q[11],q[2];
cx q[13],q[10];
cx q[11],q[23];
cx q[22],q[7];
cx q[25],q[0];
cx q[2],q[3];
cx q[15],q[5];
cx q[14],q[13];
cx q[24],q[19];
cx q[18],q[6];
cx q[9],q[4];
cx q[19],q[24];
cx q[8],q[13];
cx q[20],q[15];
cx q[21],q[26];
cx q[2],q[6];
cx q[12],q[0];
cx q[16],q[25];
cx q[4],q[17];
cx q[19],q[24];
cx q[22],q[14];
cx q[8],q[10];
cx q[15],q[5];
cx q[12],q[0];
cx q[23],q[6];
cx q[3],q[11];
cx q[21],q[2];
