OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[23],q[21];
cx q[19],q[12];
cx q[2],q[1];
cx q[26],q[22];
cx q[24],q[7];
cx q[15],q[17];
cx q[8],q[9];
cx q[3],q[5];
cx q[10],q[4];
cx q[13],q[16];
cx q[16],q[9];
cx q[3],q[14];
cx q[25],q[26];
cx q[2],q[1];
cx q[21],q[18];
cx q[23],q[7];
cx q[4],q[10];
cx q[5],q[3];
cx q[23],q[21];
cx q[10],q[4];
cx q[20],q[12];
cx q[6],q[15];
cx q[9],q[8];
cx q[16],q[13];
cx q[7],q[24];
cx q[1],q[2];
cx q[17],q[21];
cx q[13],q[16];
cx q[20],q[12];
cx q[23],q[7];
cx q[9],q[8];
cx q[6],q[15];
cx q[26],q[22];
cx q[4],q[10];
cx q[3],q[5];
cx q[25],q[20];
cx q[16],q[9];
cx q[11],q[6];
cx q[12],q[13];
cx q[0],q[1];
cx q[24],q[7];
cx q[3],q[5];
cx q[12],q[13];
cx q[0],q[5];
cx q[19],q[20];
cx q[7],q[23];
cx q[3],q[14];
cx q[4],q[10];
cx q[11],q[6];
cx q[24],q[26];
cx q[11],q[6];
cx q[16],q[9];
cx q[12],q[13];
cx q[21],q[17];
cx q[19],q[20];
cx q[7],q[23];
cx q[15],q[4];
cx q[5],q[14];
cx q[9],q[3];
cx q[2],q[0];
cx q[12],q[11];
cx q[18],q[21];
cx q[4],q[15];
cx q[6],q[13];
cx q[20],q[25];
cx q[23],q[7];
cx q[26],q[24];
cx q[14],q[5];
cx q[5],q[0];
cx q[21],q[18];
cx q[20],q[19];
cx q[23],q[7];
cx q[12],q[16];
cx q[9],q[11];
cx q[1],q[0];
cx q[5],q[14];
cx q[21],q[23];
cx q[2],q[4];
cx q[6],q[17];
cx q[23],q[24];
cx q[25],q[26];
cx q[4],q[2];
cx q[21],q[17];
cx q[13],q[6];
cx q[5],q[0];
cx q[3],q[14];
cx q[7],q[6];
cx q[12],q[20];
cx q[2],q[4];
cx q[3],q[5];
cx q[8],q[9];
cx q[16],q[13];
cx q[22],q[26];
cx q[24],q[23];
cx q[18],q[17];
cx q[0],q[1];
cx q[4],q[15];
cx q[9],q[11];
cx q[25],q[20];
cx q[3],q[5];
cx q[17],q[18];
cx q[6],q[13];
cx q[23],q[24];
cx q[12],q[16];
cx q[22],q[26];
cx q[0],q[1];
cx q[20],q[19];
cx q[8],q[9];
cx q[15],q[6];
cx q[2],q[4];
cx q[14],q[0];
cx q[16],q[11];
cx q[2],q[4];
cx q[15],q[6];
cx q[16],q[13];
cx q[18],q[17];
cx q[19],q[20];
cx q[11],q[8];
cx q[22],q[25];
cx q[0],q[1];
cx q[24],q[23];
cx q[3],q[5];
cx q[23],q[22];
cx q[16],q[12];
cx q[7],q[17];
cx q[5],q[3];
cx q[4],q[15];
cx q[6],q[13];
cx q[0],q[1];
cx q[24],q[21];
cx q[23],q[24];
cx q[1],q[0];
cx q[3],q[8];
cx q[21],q[17];
cx q[6],q[15];
cx q[14],q[5];
cx q[4],q[10];
cx q[16],q[11];
cx q[19],q[20];
cx q[22],q[26];
cx q[24],q[23];
cx q[8],q[11];
cx q[1],q[14];
cx q[3],q[5];
cx q[22],q[25];
cx q[21],q[17];
cx q[10],q[4];
cx q[13],q[16];
cx q[19],q[20];
cx q[6],q[15];
cx q[14],q[1];
cx q[20],q[19];
cx q[12],q[16];
cx q[13],q[6];
cx q[18],q[21];
cx q[22],q[25];
cx q[26],q[22];
cx q[24],q[23];
cx q[0],q[2];
cx q[18],q[21];
cx q[11],q[8];
cx q[4],q[10];
cx q[12],q[16];
cx q[3],q[5];
cx q[19],q[20];
cx q[6],q[13];
cx q[20],q[25];
cx q[18],q[17];
cx q[3],q[5];
cx q[24],q[21];
cx q[6],q[13];
cx q[9],q[11];
cx q[14],q[2];
cx q[23],q[22];
cx q[20],q[19];
cx q[11],q[9];
cx q[22],q[23];
cx q[2],q[1];
cx q[18],q[7];
cx q[4],q[15];
cx q[5],q[14];
cx q[16],q[12];
cx q[22],q[23];
cx q[3],q[8];
cx q[25],q[20];
cx q[24],q[21];
cx q[4],q[15];
cx q[0],q[2];
cx q[18],q[7];
cx q[2],q[1];
cx q[16],q[20];
cx q[12],q[11];
cx q[22],q[25];
cx q[14],q[5];
cx q[10],q[4];
cx q[19],q[20];
cx q[3],q[9];
cx q[7],q[6];
cx q[21],q[18];
cx q[25],q[23];
cx q[14],q[2];
cx q[19],q[20];
cx q[18],q[17];
cx q[5],q[3];
cx q[25],q[22];
cx q[21],q[24];
cx q[15],q[6];
cx q[10],q[4];
cx q[11],q[9];
cx q[18],q[7];
cx q[3],q[2];
cx q[12],q[11];
cx q[16],q[20];
cx q[24],q[21];
cx q[5],q[9];
cx q[16],q[12];
cx q[23],q[24];
cx q[22],q[25];
cx q[7],q[18];
cx q[0],q[14];
cx q[9],q[5];
cx q[2],q[3];
cx q[11],q[12];
cx q[5],q[9];
cx q[15],q[4];
cx q[2],q[14];
cx q[16],q[20];
cx q[6],q[7];
cx q[22],q[25];
cx q[14],q[0];
cx q[21],q[17];
cx q[24],q[23];
cx q[12],q[16];
cx q[5],q[9];
cx q[7],q[15];
cx q[20],q[19];
cx q[4],q[6];
cx q[26],q[25];
cx q[3],q[5];
cx q[9],q[11];
cx q[20],q[19];
cx q[1],q[0];
cx q[25],q[22];
cx q[12],q[16];
cx q[24],q[23];
cx q[7],q[15];
cx q[23],q[21];
cx q[20],q[22];
cx q[25],q[24];
cx q[11],q[9];
cx q[2],q[3];
cx q[16],q[12];
cx q[1],q[0];
cx q[4],q[6];
cx q[17],q[7];
cx q[25],q[24];
cx q[15],q[13];
cx q[21],q[23];
cx q[15],q[7];
cx q[14],q[6];
cx q[8],q[9];
cx q[17],q[18];
cx q[1],q[2];
cx q[26],q[25];
cx q[20],q[22];
cx q[6],q[14];
cx q[2],q[4];
cx q[7],q[17];
cx q[9],q[8];
cx q[21],q[23];
cx q[1],q[3];
cx q[1],q[3];
cx q[10],q[6];
cx q[17],q[18];
cx q[9],q[8];
cx q[16],q[20];
cx q[4],q[2];
cx q[12],q[11];
cx q[13],q[14];
cx q[23],q[21];
cx q[22],q[25];
cx q[22],q[25];
cx q[1],q[3];
cx q[11],q[12];
cx q[17],q[18];
cx q[0],q[2];
cx q[20],q[19];
cx q[21],q[23];
cx q[15],q[12];
cx q[2],q[0];
cx q[13],q[7];
cx q[22],q[20];
cx q[3],q[1];
cx q[14],q[6];
cx q[16],q[20];
cx q[7],q[14];
cx q[26],q[25];
cx q[4],q[1];
cx q[11],q[12];
cx q[13],q[17];
cx q[23],q[24];
cx q[8],q[9];
cx q[6],q[10];
cx q[15],q[12];
cx q[25],q[26];
cx q[1],q[0];
cx q[6],q[4];
cx q[11],q[9];
cx q[21],q[23];
cx q[20],q[22];
cx q[13],q[17];
cx q[22],q[20];
cx q[7],q[6];
cx q[2],q[3];
cx q[12],q[16];
cx q[14],q[15];
cx q[23],q[21];
cx q[10],q[6];
cx q[4],q[1];
cx q[20],q[22];
cx q[26],q[25];
cx q[18],q[17];
cx q[8],q[11];
cx q[6],q[10];
cx q[0],q[1];
cx q[3],q[2];
cx q[22],q[25];
cx q[15],q[14];
cx q[20],q[19];
cx q[8],q[5];
cx q[17],q[23];
cx q[11],q[12];
cx q[22],q[20];
cx q[13],q[15];
cx q[3],q[2];
cx q[21],q[24];
cx q[4],q[10];
cx q[25],q[26];
cx q[18],q[17];
cx q[0],q[1];
cx q[8],q[9];
cx q[24],q[23];
cx q[8],q[9];
cx q[11],q[12];
cx q[15],q[7];
cx q[22],q[25];
cx q[10],q[6];
cx q[1],q[2];
cx q[3],q[5];
cx q[13],q[17];
cx q[23],q[24];
cx q[17],q[13];
cx q[25],q[26];
cx q[5],q[3];
cx q[11],q[8];
cx q[20],q[19];
cx q[14],q[11];
cx q[23],q[21];
cx q[18],q[17];
cx q[6],q[10];
cx q[9],q[8];
cx q[4],q[1];
cx q[15],q[13];
cx q[3],q[2];
cx q[7],q[10];
cx q[14],q[12];
cx q[20],q[16];
cx q[13],q[15];
cx q[24],q[23];
cx q[16],q[19];
cx q[0],q[1];
cx q[13],q[7];
cx q[21],q[18];
cx q[5],q[8];
cx q[10],q[4];
cx q[3],q[2];
cx q[24],q[23];
cx q[12],q[14];
cx q[3],q[5];
cx q[1],q[0];
cx q[10],q[13];
cx q[18],q[21];
cx q[11],q[8];
cx q[24],q[25];
