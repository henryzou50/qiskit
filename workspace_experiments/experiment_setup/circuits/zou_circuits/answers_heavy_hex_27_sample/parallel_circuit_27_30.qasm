OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[7],q[4];
cx q[26],q[25];
cx q[3],q[5];
cx q[18],q[17];
cx q[20],q[19];
cx q[12],q[10];
cx q[13],q[14];
cx q[21],q[23];
cx q[9],q[8];
cx q[0],q[1];
swap q[5],q[3];
cx q[2],q[3];
cx q[8],q[9];
cx q[18],q[15];
cx q[20],q[19];
cx q[11],q[14];
cx q[21],q[23];
cx q[24],q[25];
swap q[10],q[12];
cx q[25],q[24];
cx q[20],q[19];
cx q[6],q[7];
cx q[23],q[21];
cx q[11],q[8];
cx q[1],q[4];
cx q[3],q[5];
cx q[15],q[12];
swap q[1],q[2];
cx q[21],q[23];
cx q[22],q[25];
cx q[18],q[17];
cx q[3],q[5];
cx q[8],q[9];
cx q[16],q[19];
cx q[4],q[1];
cx q[12],q[10];
swap q[16],q[14];
cx q[2],q[1];
cx q[21],q[18];
cx q[4],q[7];
cx q[8],q[11];
cx q[5],q[3];
swap q[22],q[25];
cx q[19],q[22];
cx q[3],q[2];
cx q[8],q[5];
cx q[23],q[21];
cx q[0],q[1];
swap q[12],q[15];
cx q[3],q[5];
cx q[9],q[8];
cx q[18],q[21];
cx q[24],q[23];
cx q[25],q[26];
cx q[14],q[11];
cx q[19],q[16];
cx q[7],q[10];
cx q[15],q[12];
swap q[10],q[7];
cx q[8],q[5];
cx q[25],q[26];
cx q[0],q[1];
cx q[2],q[3];
cx q[21],q[23];
cx q[20],q[19];
cx q[15],q[12];
cx q[7],q[10];
cx q[11],q[14];
cx q[17],q[18];
swap q[25],q[26];
cx q[21],q[18];
cx q[10],q[12];
cx q[4],q[7];
cx q[8],q[9];
cx q[3],q[5];
swap q[13],q[12];
cx q[12],q[10];
cx q[11],q[8];
cx q[25],q[22];
cx q[23],q[24];
cx q[6],q[7];
cx q[19],q[16];
cx q[14],q[13];
swap q[10],q[12];
cx q[25],q[26];
cx q[4],q[7];
cx q[1],q[2];
cx q[22],q[19];
cx q[23],q[21];
cx q[15],q[18];
cx q[14],q[16];
swap q[8],q[9];
cx q[24],q[23];
cx q[13],q[14];
cx q[4],q[1];
cx q[10],q[12];
cx q[3],q[5];
cx q[18],q[17];
swap q[24],q[23];
cx q[23],q[21];
cx q[13],q[12];
cx q[6],q[7];
cx q[19],q[16];
cx q[8],q[11];
cx q[25],q[24];
cx q[1],q[2];
cx q[15],q[18];
swap q[15],q[18];
cx q[3],q[5];
cx q[26],q[25];
cx q[21],q[23];
cx q[18],q[17];
cx q[14],q[11];
cx q[0],q[1];
cx q[9],q[8];
cx q[7],q[10];
cx q[22],q[19];
swap q[15],q[12];
cx q[23],q[21];
cx q[8],q[9];
cx q[11],q[14];
cx q[3],q[5];
cx q[18],q[17];
cx q[15],q[12];
cx q[25],q[26];
cx q[20],q[19];
cx q[7],q[4];
swap q[22],q[19];
cx q[8],q[9];
cx q[17],q[18];
cx q[25],q[22];
cx q[6],q[7];
cx q[3],q[5];
cx q[14],q[11];
cx q[1],q[0];
cx q[19],q[20];
cx q[12],q[10];
cx q[23],q[24];
swap q[21],q[23];
cx q[21],q[23];
cx q[11],q[14];
cx q[9],q[8];
cx q[24],q[25];
cx q[19],q[22];
cx q[13],q[12];
cx q[7],q[10];
cx q[5],q[3];
cx q[4],q[1];
swap q[15],q[18];
cx q[5],q[8];
cx q[12],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[19],q[20];
cx q[14],q[13];
cx q[24],q[23];
cx q[25],q[22];
swap q[18],q[17];
cx q[25],q[26];
cx q[11],q[8];
cx q[13],q[14];
cx q[19],q[20];
cx q[17],q[18];
cx q[6],q[7];
cx q[10],q[12];
cx q[3],q[5];
swap q[22],q[25];
cx q[2],q[1];
cx q[22],q[25];
cx q[12],q[10];
cx q[21],q[23];
cx q[8],q[9];
cx q[5],q[3];
cx q[20],q[19];
cx q[6],q[7];
swap q[3],q[5];
cx q[6],q[7];
cx q[25],q[22];
cx q[2],q[3];
cx q[13],q[12];
cx q[23],q[21];
cx q[16],q[14];
swap q[13],q[12];
cx q[14],q[11];
cx q[12],q[10];
cx q[4],q[7];
cx q[2],q[1];
cx q[25],q[26];
cx q[5],q[3];
cx q[9],q[8];
swap q[7],q[6];
cx q[1],q[2];
cx q[16],q[19];
cx q[6],q[7];
cx q[15],q[18];
cx q[25],q[26];
cx q[9],q[8];
cx q[11],q[14];
cx q[3],q[5];
swap q[18],q[15];
cx q[5],q[3];
cx q[8],q[11];
cx q[1],q[4];
cx q[24],q[25];
cx q[13],q[12];
cx q[21],q[18];
cx q[7],q[6];
swap q[18],q[17];
cx q[12],q[13];
cx q[25],q[26];
cx q[22],q[19];
cx q[5],q[3];
cx q[18],q[17];
cx q[23],q[21];
cx q[2],q[1];
cx q[8],q[9];
cx q[11],q[14];
cx q[7],q[4];
swap q[4],q[1];
cx q[12],q[13];
cx q[7],q[6];
cx q[19],q[16];
cx q[5],q[8];
cx q[3],q[2];
cx q[4],q[1];
swap q[1],q[2];
cx q[3],q[5];
cx q[24],q[23];
cx q[0],q[1];
cx q[12],q[15];
cx q[9],q[8];
cx q[7],q[10];
cx q[19],q[16];
cx q[25],q[22];
cx q[21],q[18];
swap q[18],q[15];
cx q[8],q[5];
cx q[22],q[25];
cx q[7],q[4];
cx q[14],q[11];
cx q[23],q[24];
cx q[20],q[19];
cx q[13],q[12];
cx q[15],q[18];
cx q[1],q[2];
swap q[19],q[20];
cx q[8],q[11];
cx q[7],q[10];
cx q[25],q[24];
cx q[13],q[14];
cx q[2],q[3];
swap q[13],q[12];
cx q[1],q[4];
cx q[8],q[5];
cx q[11],q[14];
cx q[7],q[10];
cx q[25],q[26];
cx q[23],q[21];
cx q[3],q[2];
cx q[17],q[18];
cx q[16],q[19];
cx q[13],q[12];
swap q[3],q[5];
