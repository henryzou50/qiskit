OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[9],q[12];
cx q[21],q[1];
cx q[15],q[8];
cx q[0],q[25];
cx q[24],q[20];
cx q[18],q[7];
cx q[4],q[14];
cx q[2],q[13];
cx q[26],q[22];
cx q[11],q[6];
cx q[1],q[11];
cx q[10],q[9];
cx q[12],q[5];
cx q[16],q[3];
cx q[17],q[22];
cx q[0],q[23];
cx q[2],q[14];
cx q[26],q[20];
cx q[19],q[18];
cx q[8],q[9];
cx q[6],q[1];
cx q[15],q[10];
cx q[22],q[2];
cx q[24],q[20];
cx q[3],q[11];
cx q[21],q[1];
cx q[5],q[24];
cx q[15],q[16];
cx q[0],q[25];
cx q[2],q[13];
cx q[0],q[8];
cx q[4],q[22];
cx q[20],q[5];
cx q[7],q[18];
cx q[17],q[12];
cx q[21],q[6];
cx q[10],q[26];
cx q[23],q[2];
cx q[4],q[23];
cx q[2],q[9];
cx q[11],q[3];
cx q[21],q[7];
cx q[15],q[8];
cx q[0],q[10];
cx q[5],q[24];
cx q[19],q[18];
cx q[8],q[15];
cx q[13],q[2];
cx q[3],q[1];
cx q[25],q[9];
cx q[11],q[1];
cx q[2],q[13];
cx q[21],q[18];
cx q[10],q[25];
cx q[17],q[19];
cx q[15],q[26];
cx q[6],q[3];
cx q[12],q[22];
cx q[5],q[4];
cx q[21],q[24];
cx q[23],q[13];
cx q[9],q[8];
cx q[26],q[15];
cx q[19],q[20];
cx q[12],q[4];
cx q[0],q[5];
cx q[11],q[16];
cx q[20],q[21];
cx q[25],q[2];
cx q[0],q[26];
cx q[1],q[7];
cx q[10],q[9];
cx q[23],q[14];
cx q[11],q[3];
cx q[16],q[18];
cx q[0],q[26];
cx q[15],q[17];
cx q[9],q[10];
cx q[22],q[4];
cx q[9],q[26];
cx q[7],q[1];
cx q[15],q[12];
cx q[23],q[14];
cx q[5],q[19];
cx q[22],q[4];
cx q[17],q[0];
cx q[3],q[5];
cx q[4],q[22];
cx q[13],q[23];
cx q[1],q[7];
cx q[21],q[24];
cx q[16],q[0];
cx q[9],q[26];
cx q[9],q[8];
cx q[13],q[4];
cx q[17],q[15];
cx q[14],q[23];
cx q[18],q[6];
cx q[4],q[12];
cx q[17],q[20];
cx q[14],q[13];
cx q[0],q[6];
cx q[15],q[16];
cx q[24],q[18];
cx q[25],q[8];
cx q[9],q[26];
cx q[21],q[3];
cx q[26],q[10];
cx q[1],q[16];
cx q[12],q[15];
cx q[19],q[5];
cx q[21],q[24];
cx q[8],q[2];
cx q[7],q[6];
cx q[13],q[25];
cx q[18],q[3];
cx q[12],q[19];
cx q[23],q[13];
cx q[26],q[2];
cx q[22],q[17];
cx q[0],q[6];
cx q[16],q[10];
cx q[19],q[5];
cx q[25],q[8];
cx q[22],q[14];
cx q[1],q[0];
cx q[2],q[9];
cx q[23],q[13];
cx q[12],q[20];
cx q[24],q[18];
cx q[15],q[2];
cx q[25],q[26];
cx q[19],q[5];
cx q[18],q[0];
cx q[1],q[7];
cx q[21],q[3];
cx q[20],q[16];
cx q[23],q[14];
cx q[4],q[17];
cx q[25],q[26];
cx q[21],q[0];
cx q[10],q[7];
cx q[5],q[18];
cx q[20],q[3];
cx q[16],q[12];
cx q[9],q[2];
cx q[6],q[15];
cx q[8],q[26];
cx q[13],q[25];
cx q[18],q[19];
cx q[15],q[12];
cx q[0],q[11];
cx q[14],q[4];
cx q[5],q[21];
cx q[6],q[2];
cx q[16],q[22];
cx q[20],q[15];
cx q[8],q[25];
cx q[24],q[1];
cx q[7],q[12];
cx q[26],q[23];
cx q[11],q[0];
cx q[5],q[21];
cx q[4],q[14];
cx q[11],q[0];
cx q[23],q[14];
cx q[24],q[1];
cx q[8],q[9];
cx q[16],q[20];
cx q[7],q[12];
cx q[21],q[5];
cx q[12],q[7];
cx q[13],q[23];
cx q[24],q[0];
cx q[22],q[4];
cx q[11],q[10];
cx q[15],q[17];
cx q[25],q[26];
cx q[16],q[3];
cx q[5],q[19];
cx q[9],q[8];
cx q[14],q[4];
cx q[26],q[13];
cx q[0],q[11];
cx q[1],q[7];
cx q[18],q[3];
cx q[25],q[8];
cx q[14],q[23];
cx q[18],q[15];
cx q[19],q[16];
cx q[11],q[21];
cx q[5],q[3];
cx q[24],q[7];
cx q[9],q[6];
cx q[1],q[12];
cx q[13],q[4];
cx q[14],q[23];
cx q[10],q[7];
cx q[6],q[2];
cx q[16],q[20];
cx q[17],q[18];
cx q[3],q[5];
cx q[9],q[15];
cx q[23],q[8];
cx q[3],q[10];
cx q[5],q[19];
cx q[14],q[26];
cx q[17],q[20];
cx q[25],q[6];
cx q[24],q[21];
cx q[4],q[13];
cx q[5],q[16];
cx q[4],q[18];
cx q[9],q[25];
cx q[22],q[23];
cx q[8],q[26];
cx q[0],q[11];
cx q[15],q[20];
cx q[6],q[1];
cx q[24],q[7];
cx q[6],q[26];
cx q[25],q[12];
cx q[23],q[22];
cx q[16],q[20];
cx q[19],q[3];
cx q[15],q[17];
cx q[4],q[18];
cx q[25],q[15];
cx q[2],q[9];
cx q[24],q[12];
cx q[16],q[5];
cx q[20],q[4];
cx q[16],q[3];
cx q[7],q[24];
cx q[1],q[2];
cx q[0],q[11];
cx q[8],q[13];
cx q[26],q[23];
cx q[6],q[1];
cx q[18],q[17];
cx q[15],q[25];
cx q[24],q[12];
cx q[19],q[20];
cx q[16],q[11];
cx q[16],q[5];
cx q[15],q[7];
cx q[4],q[17];
cx q[13],q[26];
cx q[12],q[0];
cx q[2],q[6];
cx q[19],q[11];
cx q[25],q[17];
cx q[15],q[1];
cx q[10],q[21];
cx q[8],q[26];
cx q[13],q[23];
cx q[12],q[0];
cx q[11],q[21];
cx q[8],q[14];
cx q[25],q[7];
cx q[1],q[9];
cx q[19],q[5];
cx q[6],q[13];
cx q[15],q[24];
cx q[17],q[18];
cx q[16],q[20];
cx q[6],q[26];
cx q[8],q[23];
cx q[7],q[1];
cx q[19],q[16];
cx q[3],q[17];
cx q[15],q[12];
cx q[9],q[2];
cx q[25],q[1];
cx q[0],q[11];
cx q[5],q[19];
cx q[14],q[6];
cx q[26],q[2];
cx q[20],q[3];
cx q[18],q[8];
cx q[4],q[3];
cx q[17],q[20];
cx q[26],q[13];
cx q[5],q[19];
cx q[6],q[23];
cx q[3],q[19];
cx q[5],q[21];
cx q[18],q[6];
cx q[26],q[14];
cx q[22],q[4];
cx q[10],q[0];
cx q[17],q[3];
cx q[2],q[25];
cx q[12],q[15];
cx q[14],q[13];
cx q[4],q[22];
cx q[7],q[26];
cx q[5],q[19];
cx q[6],q[18];
cx q[16],q[19];
cx q[25],q[15];
cx q[1],q[9];
cx q[14],q[26];
cx q[20],q[21];
cx q[5],q[0];
cx q[6],q[22];
cx q[24],q[5];
cx q[10],q[12];
cx q[1],q[2];
cx q[21],q[17];
cx q[23],q[13];
cx q[22],q[3];
cx q[4],q[8];
cx q[20],q[0];
cx q[26],q[9];
cx q[9],q[13];
cx q[26],q[14];
cx q[1],q[25];
cx q[23],q[18];
cx q[17],q[21];
cx q[10],q[12];
cx q[20],q[0];
cx q[2],q[22];
cx q[25],q[2];
cx q[22],q[19];
cx q[17],q[0];
cx q[9],q[14];
cx q[6],q[3];
cx q[13],q[11];
cx q[21],q[16];
cx q[23],q[26];
cx q[20],q[0];
cx q[5],q[15];
cx q[18],q[8];
cx q[10],q[12];
cx q[7],q[9];
cx q[13],q[26];
cx q[25],q[19];
cx q[21],q[16];
cx q[2],q[11];
cx q[5],q[1];
cx q[26],q[13];
cx q[24],q[15];
cx q[14],q[23];
cx q[2],q[9];
cx q[25],q[4];
cx q[7],q[11];
