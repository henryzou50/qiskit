OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[18],q[11];
cx q[25],q[10];
cx q[1],q[0];
cx q[15],q[17];
cx q[20],q[9];
cx q[5],q[22];
cx q[23],q[24];
cx q[6],q[3];
cx q[19],q[16];
cx q[26],q[7];
cx q[12],q[13];
cx q[2],q[21];
cx q[10],q[12];
cx q[3],q[8];
cx q[20],q[17];
cx q[26],q[4];
cx q[5],q[13];
cx q[24],q[2];
cx q[15],q[22];
cx q[14],q[16];
cx q[1],q[23];
cx q[18],q[11];
cx q[21],q[0];
cx q[14],q[16];
cx q[19],q[11];
cx q[22],q[15];
cx q[10],q[12];
cx q[17],q[20];
cx q[23],q[24];
cx q[8],q[3];
cx q[15],q[13];
cx q[14],q[17];
cx q[7],q[4];
cx q[9],q[12];
cx q[1],q[0];
cx q[18],q[21];
cx q[25],q[5];
cx q[3],q[26];
cx q[8],q[2];
cx q[16],q[11];
cx q[25],q[5];
cx q[13],q[12];
cx q[26],q[3];
cx q[9],q[15];
cx q[18],q[21];
cx q[7],q[10];
cx q[19],q[1];
cx q[16],q[11];
cx q[2],q[6];
cx q[23],q[8];
cx q[14],q[17];
cx q[0],q[16];
cx q[26],q[10];
cx q[2],q[23];
cx q[14],q[22];
cx q[3],q[6];
cx q[25],q[13];
cx q[20],q[9];
cx q[12],q[15];
cx q[24],q[21];
cx q[19],q[18];
cx q[0],q[16];
cx q[2],q[6];
cx q[20],q[15];
cx q[18],q[19];
cx q[13],q[12];
cx q[3],q[4];
cx q[11],q[17];
cx q[8],q[26];
cx q[5],q[25];
cx q[19],q[18];
cx q[15],q[20];
cx q[17],q[11];
cx q[0],q[16];
cx q[13],q[9];
cx q[25],q[12];
cx q[7],q[10];
cx q[22],q[14];
cx q[4],q[26];
cx q[1],q[24];
cx q[15],q[22];
cx q[18],q[16];
cx q[0],q[17];
cx q[2],q[21];
cx q[20],q[9];
cx q[26],q[8];
cx q[24],q[19];
cx q[11],q[14];
cx q[4],q[10];
cx q[12],q[13];
cx q[24],q[19];
cx q[25],q[7];
cx q[17],q[0];
cx q[8],q[26];
cx q[21],q[2];
cx q[4],q[5];
cx q[1],q[2];
cx q[7],q[12];
cx q[3],q[6];
cx q[23],q[19];
cx q[14],q[22];
cx q[20],q[9];
cx q[25],q[10];
cx q[16],q[24];
cx q[15],q[9];
cx q[18],q[17];
cx q[24],q[16];
cx q[13],q[12];
cx q[4],q[7];
cx q[23],q[21];
cx q[11],q[0];
cx q[22],q[14];
cx q[25],q[8];
cx q[3],q[1];
cx q[8],q[26];
cx q[4],q[7];
cx q[5],q[12];
cx q[11],q[0];
cx q[23],q[2];
cx q[15],q[22];
cx q[18],q[16];
cx q[10],q[6];
cx q[9],q[20];
cx q[21],q[19];
cx q[17],q[24];
cx q[12],q[13];
cx q[1],q[2];
cx q[23],q[19];
cx q[26],q[3];
cx q[21],q[17];
cx q[0],q[16];
cx q[7],q[4];
cx q[22],q[14];
cx q[20],q[9];
cx q[8],q[25];
cx q[13],q[7];
cx q[1],q[19];
cx q[6],q[3];
cx q[8],q[10];
cx q[11],q[15];
cx q[14],q[5];
cx q[0],q[16];
cx q[19],q[1];
cx q[17],q[21];
cx q[6],q[8];
cx q[9],q[5];
cx q[15],q[11];
cx q[25],q[4];
cx q[24],q[17];
cx q[13],q[7];
cx q[6],q[25];
cx q[4],q[10];
cx q[8],q[26];
cx q[22],q[15];
cx q[11],q[20];
cx q[4],q[25];
cx q[3],q[2];
cx q[26],q[23];
cx q[13],q[10];
cx q[18],q[15];
cx q[1],q[21];
cx q[6],q[25];
cx q[19],q[2];
cx q[17],q[18];
cx q[0],q[16];
cx q[10],q[4];
cx q[22],q[20];
cx q[23],q[3];
cx q[21],q[24];
cx q[9],q[14];
cx q[9],q[11];
cx q[10],q[7];
cx q[3],q[26];
cx q[25],q[4];
cx q[12],q[14];
cx q[23],q[2];
cx q[20],q[22];
cx q[5],q[13];
cx q[24],q[1];
cx q[18],q[0];
cx q[13],q[10];
cx q[8],q[6];
cx q[9],q[14];
cx q[4],q[25];
cx q[16],q[22];
cx q[23],q[2];
cx q[12],q[7];
cx q[17],q[21];
cx q[6],q[10];
cx q[18],q[17];
cx q[8],q[26];
cx q[15],q[14];
cx q[9],q[5];
cx q[24],q[19];
cx q[22],q[17];
cx q[14],q[5];
cx q[1],q[2];
cx q[26],q[23];
cx q[16],q[18];
cx q[3],q[6];
cx q[25],q[4];
cx q[15],q[20];
cx q[10],q[13];
cx q[17],q[22];
cx q[8],q[23];
cx q[7],q[9];
cx q[18],q[16];
cx q[5],q[14];
cx q[6],q[8];
cx q[7],q[12];
cx q[0],q[21];
cx q[20],q[16];
cx q[9],q[13];
cx q[18],q[17];
cx q[2],q[1];
cx q[5],q[14];
cx q[19],q[24];
cx q[21],q[2];
cx q[18],q[16];
cx q[15],q[20];
cx q[12],q[5];
cx q[26],q[23];
cx q[7],q[9];
cx q[13],q[10];
cx q[4],q[25];
cx q[16],q[17];
cx q[5],q[7];
cx q[3],q[8];
cx q[1],q[23];
cx q[24],q[19];
cx q[2],q[26];
cx q[10],q[9];
cx q[1],q[3];
cx q[0],q[22];
cx q[18],q[14];
cx q[15],q[11];
cx q[5],q[12];
cx q[26],q[19];
cx q[0],q[22];
cx q[23],q[26];
cx q[15],q[12];
cx q[8],q[4];
cx q[7],q[10];
cx q[1],q[3];
cx q[20],q[16];
cx q[24],q[19];
cx q[11],q[14];
cx q[22],q[16];
cx q[1],q[2];
cx q[14],q[11];
cx q[13],q[9];
cx q[19],q[21];
cx q[7],q[10];
cx q[8],q[6];
cx q[25],q[23];
cx q[24],q[26];
cx q[15],q[5];
cx q[4],q[3];
cx q[4],q[25];
cx q[0],q[19];
cx q[18],q[16];
cx q[12],q[11];
cx q[5],q[7];
cx q[6],q[3];
cx q[23],q[1];
cx q[9],q[10];
cx q[22],q[21];
cx q[8],q[10];
cx q[17],q[16];
cx q[18],q[20];
cx q[14],q[11];
cx q[3],q[6];
cx q[26],q[0];
cx q[5],q[13];
cx q[1],q[25];
cx q[15],q[12];
cx q[2],q[23];
cx q[9],q[7];
cx q[23],q[1];
cx q[26],q[2];
cx q[21],q[22];
cx q[8],q[10];
cx q[16],q[17];
cx q[14],q[15];
cx q[6],q[3];
cx q[18],q[20];
cx q[0],q[19];
cx q[11],q[12];
cx q[8],q[7];
cx q[5],q[10];
cx q[26],q[0];
cx q[19],q[22];
cx q[2],q[24];
cx q[25],q[1];
cx q[3],q[6];
cx q[13],q[9];
cx q[12],q[14];
cx q[9],q[8];
cx q[24],q[23];
cx q[12],q[11];
cx q[16],q[15];
cx q[0],q[26];
cx q[3],q[4];
cx q[7],q[10];
cx q[14],q[18];
cx q[22],q[19];
cx q[21],q[17];
cx q[5],q[6];
cx q[25],q[1];
cx q[4],q[3];
cx q[9],q[12];
cx q[17],q[16];
cx q[1],q[25];
cx q[8],q[7];
cx q[5],q[6];
cx q[20],q[21];
cx q[0],q[26];
cx q[19],q[22];
cx q[2],q[23];
cx q[3],q[6];
cx q[5],q[8];
cx q[18],q[20];
cx q[1],q[23];
cx q[9],q[12];
cx q[17],q[19];
cx q[4],q[25];
cx q[23],q[2];
cx q[21],q[22];
cx q[13],q[11];
cx q[24],q[26];
cx q[7],q[10];
cx q[1],q[4];
cx q[17],q[18];
cx q[3],q[5];
cx q[6],q[8];
cx q[20],q[19];
cx q[19],q[22];
cx q[20],q[18];
cx q[25],q[2];
cx q[24],q[21];
cx q[11],q[14];
cx q[16],q[15];
cx q[8],q[5];
cx q[3],q[1];
cx q[9],q[10];
cx q[0],q[23];
cx q[8],q[5];
cx q[17],q[18];
cx q[21],q[24];
cx q[9],q[10];
cx q[19],q[20];
cx q[13],q[11];
cx q[15],q[16];
cx q[23],q[26];
cx q[3],q[6];
cx q[0],q[2];
cx q[1],q[4];
