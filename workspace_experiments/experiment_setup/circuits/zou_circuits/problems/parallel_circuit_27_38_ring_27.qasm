OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[7],q[14];
cx q[17],q[20];
cx q[11],q[6];
cx q[23],q[1];
cx q[19],q[24];
cx q[0],q[21];
cx q[9],q[16];
cx q[23],q[19];
cx q[18],q[10];
cx q[4],q[12];
cx q[11],q[6];
cx q[15],q[14];
cx q[7],q[3];
cx q[8],q[26];
cx q[22],q[21];
cx q[25],q[1];
cx q[5],q[13];
cx q[16],q[9];
cx q[6],q[12];
cx q[24],q[19];
cx q[16],q[5];
cx q[9],q[11];
cx q[13],q[7];
cx q[2],q[4];
cx q[17],q[20];
cx q[25],q[23];
cx q[22],q[17];
cx q[6],q[2];
cx q[18],q[20];
cx q[3],q[16];
cx q[23],q[4];
cx q[12],q[11];
cx q[14],q[15];
cx q[1],q[25];
cx q[7],q[10];
cx q[9],q[26];
cx q[24],q[19];
cx q[5],q[8];
cx q[0],q[21];
cx q[3],q[7];
cx q[0],q[21];
cx q[25],q[4];
cx q[5],q[16];
cx q[14],q[15];
cx q[18],q[22];
cx q[2],q[12];
cx q[10],q[13];
cx q[4],q[25];
cx q[11],q[9];
cx q[0],q[19];
cx q[26],q[12];
cx q[14],q[20];
cx q[7],q[15];
cx q[2],q[6];
cx q[16],q[13];
cx q[23],q[24];
cx q[8],q[5];
cx q[22],q[21];
cx q[12],q[26];
cx q[16],q[7];
cx q[25],q[2];
cx q[5],q[3];
cx q[22],q[17];
cx q[19],q[21];
cx q[14],q[10];
cx q[18],q[20];
cx q[1],q[23];
cx q[8],q[9];
cx q[23],q[25];
cx q[16],q[5];
cx q[3],q[13];
cx q[26],q[12];
cx q[19],q[21];
cx q[9],q[11];
cx q[14],q[10];
cx q[18],q[20];
cx q[22],q[17];
cx q[2],q[4];
cx q[12],q[26];
cx q[25],q[23];
cx q[2],q[4];
cx q[24],q[21];
cx q[1],q[22];
cx q[7],q[14];
cx q[19],q[17];
cx q[9],q[11];
cx q[0],q[24];
cx q[6],q[8];
cx q[3],q[5];
cx q[25],q[23];
cx q[15],q[16];
cx q[10],q[14];
cx q[10],q[18];
cx q[4],q[12];
cx q[23],q[0];
cx q[14],q[16];
cx q[6],q[8];
cx q[9],q[26];
cx q[2],q[25];
cx q[3],q[11];
cx q[17],q[20];
cx q[21],q[24];
cx q[15],q[13];
cx q[1],q[19];
cx q[4],q[6];
cx q[5],q[15];
cx q[18],q[10];
cx q[25],q[0];
cx q[13],q[3];
cx q[17],q[22];
cx q[7],q[16];
cx q[21],q[1];
cx q[11],q[9];
cx q[24],q[19];
cx q[12],q[23];
cx q[22],q[19];
cx q[5],q[15];
cx q[1],q[24];
cx q[3],q[9];
cx q[17],q[10];
cx q[12],q[23];
cx q[11],q[26];
cx q[4],q[6];
cx q[16],q[14];
cx q[25],q[1];
cx q[13],q[11];
cx q[0],q[23];
cx q[15],q[5];
cx q[21],q[24];
cx q[2],q[12];
cx q[13],q[3];
cx q[14],q[15];
cx q[24],q[18];
cx q[1],q[23];
cx q[2],q[0];
cx q[5],q[16];
cx q[20],q[17];
cx q[8],q[4];
cx q[10],q[7];
cx q[21],q[25];
cx q[12],q[6];
cx q[15],q[3];
cx q[7],q[10];
cx q[8],q[26];
cx q[22],q[24];
cx q[21],q[25];
cx q[19],q[18];
cx q[23],q[1];
cx q[21],q[22];
cx q[3],q[11];
cx q[7],q[10];
cx q[19],q[18];
cx q[6],q[1];
cx q[25],q[23];
cx q[8],q[26];
cx q[17],q[20];
cx q[12],q[2];
cx q[15],q[13];
cx q[16],q[5];
cx q[9],q[3];
cx q[4],q[8];
cx q[1],q[23];
cx q[10],q[20];
cx q[16],q[7];
cx q[19],q[24];
cx q[7],q[16];
cx q[21],q[22];
cx q[10],q[20];
cx q[11],q[13];
cx q[17],q[18];
cx q[26],q[2];
cx q[15],q[14];
cx q[3],q[9];
cx q[0],q[23];
cx q[12],q[6];
cx q[24],q[19];
cx q[14],q[15];
cx q[1],q[22];
cx q[23],q[6];
cx q[8],q[2];
cx q[17],q[16];
cx q[3],q[11];
cx q[13],q[9];
cx q[5],q[7];
cx q[12],q[4];
cx q[25],q[0];
cx q[19],q[20];
cx q[21],q[18];
cx q[17],q[20];
cx q[16],q[10];
cx q[13],q[3];
cx q[6],q[26];
cx q[2],q[8];
cx q[22],q[23];
cx q[21],q[18];
cx q[15],q[5];
cx q[25],q[12];
cx q[11],q[7];
cx q[18],q[1];
cx q[10],q[15];
cx q[8],q[9];
cx q[24],q[0];
cx q[5],q[11];
cx q[6],q[25];
cx q[2],q[4];
cx q[22],q[23];
cx q[17],q[20];
cx q[12],q[26];
cx q[6],q[12];
cx q[16],q[20];
cx q[25],q[22];
cx q[7],q[14];
cx q[15],q[10];
cx q[4],q[26];
cx q[3],q[13];
cx q[24],q[23];
cx q[11],q[5];
cx q[2],q[8];
cx q[23],q[22];
cx q[10],q[16];
cx q[4],q[6];
cx q[21],q[18];
cx q[0],q[1];
cx q[9],q[13];
cx q[12],q[26];
cx q[17],q[20];
cx q[11],q[7];
cx q[5],q[15];
cx q[25],q[24];
cx q[4],q[6];
cx q[20],q[10];
cx q[5],q[11];
cx q[15],q[16];
cx q[19],q[17];
cx q[14],q[13];
cx q[1],q[0];
cx q[7],q[15];
cx q[23],q[24];
cx q[21],q[18];
cx q[5],q[16];
cx q[17],q[19];
cx q[10],q[20];
cx q[13],q[3];
cx q[17],q[18];
cx q[20],q[19];
cx q[1],q[24];
cx q[7],q[14];
cx q[4],q[23];
cx q[16],q[10];
cx q[11],q[5];
cx q[1],q[4];
cx q[0],q[24];
cx q[10],q[15];
cx q[14],q[7];
cx q[5],q[11];
cx q[26],q[23];
cx q[13],q[3];
cx q[8],q[2];
cx q[12],q[6];
cx q[20],q[17];
cx q[21],q[18];
cx q[10],q[16];
cx q[3],q[2];
cx q[20],q[19];
cx q[18],q[17];
cx q[11],q[14];
cx q[12],q[6];
cx q[5],q[13];
cx q[23],q[22];
cx q[25],q[4];
cx q[9],q[7];
cx q[24],q[21];
cx q[1],q[0];
cx q[13],q[5];
cx q[4],q[1];
cx q[9],q[7];
cx q[24],q[0];
cx q[19],q[20];
cx q[10],q[14];
cx q[6],q[26];
cx q[2],q[3];
cx q[8],q[12];
cx q[25],q[23];
cx q[20],q[19];
cx q[26],q[25];
cx q[11],q[5];
cx q[3],q[8];
cx q[12],q[6];
cx q[1],q[23];
cx q[22],q[0];
cx q[21],q[24];
cx q[13],q[14];
cx q[3],q[2];
cx q[11],q[15];
cx q[14],q[10];
cx q[17],q[21];
cx q[1],q[25];
cx q[20],q[16];
cx q[16],q[10];
cx q[8],q[2];
cx q[25],q[6];
cx q[1],q[4];
cx q[19],q[21];
cx q[22],q[17];
cx q[18],q[19];
cx q[20],q[16];
cx q[21],q[17];
cx q[25],q[1];
cx q[14],q[10];
cx q[8],q[3];
cx q[15],q[10];
cx q[0],q[23];
cx q[26],q[1];
cx q[18],q[14];
cx q[6],q[2];
cx q[8],q[3];
cx q[20],q[17];
cx q[9],q[12];
cx q[16],q[11];
cx q[14],q[11];
cx q[1],q[23];
cx q[0],q[22];
cx q[3],q[9];
cx q[24],q[21];
cx q[7],q[12];
cx q[20],q[18];
cx q[10],q[15];
cx q[2],q[6];
cx q[21],q[19];
cx q[26],q[1];
cx q[4],q[2];
cx q[7],q[12];
cx q[16],q[14];
cx q[22],q[24];
cx q[6],q[3];
cx q[16],q[14];
cx q[8],q[5];
cx q[17],q[18];
cx q[1],q[2];
cx q[22],q[23];
cx q[26],q[0];
cx q[10],q[9];
cx q[6],q[3];
