OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[18],q[1];
cx q[20],q[23];
cx q[4],q[17];
cx q[10],q[7];
cx q[26],q[5];
cx q[16],q[9];
cx q[12],q[3];
cx q[7],q[10];
cx q[13],q[25];
cx q[11],q[18];
cx q[16],q[8];
cx q[0],q[17];
cx q[24],q[14];
cx q[7],q[13];
cx q[9],q[20];
cx q[3],q[15];
cx q[11],q[6];
cx q[21],q[12];
cx q[26],q[5];
cx q[4],q[0];
cx q[2],q[8];
cx q[1],q[14];
cx q[22],q[17];
cx q[18],q[19];
cx q[14],q[1];
cx q[20],q[8];
cx q[12],q[7];
cx q[15],q[24];
cx q[10],q[21];
cx q[16],q[4];
cx q[25],q[13];
cx q[1],q[6];
cx q[23],q[2];
cx q[25],q[12];
cx q[22],q[4];
cx q[16],q[8];
cx q[25],q[13];
cx q[2],q[26];
cx q[8],q[23];
cx q[5],q[9];
cx q[7],q[12];
cx q[19],q[18];
cx q[16],q[0];
cx q[4],q[17];
cx q[1],q[11];
cx q[3],q[14];
cx q[6],q[24];
cx q[10],q[21];
cx q[19],q[1];
cx q[0],q[4];
cx q[15],q[14];
cx q[10],q[3];
cx q[25],q[13];
cx q[12],q[21];
cx q[11],q[1];
cx q[6],q[24];
cx q[16],q[4];
cx q[0],q[22];
cx q[23],q[20];
cx q[3],q[21];
cx q[12],q[14];
cx q[15],q[10];
cx q[10],q[6];
cx q[5],q[7];
cx q[12],q[14];
cx q[20],q[16];
cx q[11],q[17];
cx q[18],q[19];
cx q[4],q[22];
cx q[24],q[1];
cx q[20],q[4];
cx q[26],q[23];
cx q[3],q[5];
cx q[12],q[21];
cx q[7],q[9];
cx q[11],q[19];
cx q[18],q[22];
cx q[13],q[5];
cx q[4],q[0];
cx q[8],q[16];
cx q[23],q[2];
cx q[18],q[22];
cx q[4],q[0];
cx q[11],q[19];
cx q[9],q[26];
cx q[6],q[15];
cx q[5],q[3];
cx q[21],q[10];
cx q[8],q[25];
cx q[14],q[12];
cx q[2],q[7];
cx q[16],q[20];
cx q[17],q[24];
cx q[7],q[5];
cx q[17],q[1];
cx q[19],q[6];
cx q[18],q[0];
cx q[3],q[13];
cx q[4],q[20];
cx q[2],q[23];
cx q[21],q[10];
cx q[11],q[22];
cx q[12],q[3];
cx q[23],q[8];
cx q[0],q[22];
cx q[14],q[15];
cx q[18],q[20];
cx q[16],q[4];
cx q[13],q[7];
cx q[23],q[16];
cx q[11],q[22];
cx q[0],q[18];
cx q[26],q[25];
cx q[15],q[3];
cx q[12],q[9];
cx q[2],q[8];
cx q[14],q[21];
cx q[20],q[4];
cx q[6],q[24];
cx q[1],q[19];
cx q[5],q[10];
cx q[7],q[13];
cx q[18],q[11];
cx q[6],q[17];
cx q[9],q[5];
cx q[8],q[23];
cx q[4],q[16];
cx q[26],q[25];
cx q[3],q[12];
cx q[19],q[1];
cx q[15],q[21];
cx q[17],q[22];
cx q[0],q[11];
cx q[5],q[10];
cx q[12],q[9];
cx q[26],q[2];
cx q[16],q[20];
cx q[8],q[23];
cx q[3],q[15];
cx q[26],q[25];
cx q[2],q[7];
cx q[4],q[23];
cx q[15],q[21];
cx q[19],q[6];
cx q[13],q[5];
cx q[22],q[17];
cx q[11],q[16];
cx q[3],q[9];
cx q[6],q[17];
cx q[11],q[20];
cx q[25],q[2];
cx q[14],q[21];
cx q[1],q[19];
cx q[0],q[16];
cx q[10],q[7];
cx q[8],q[26];
cx q[9],q[3];
cx q[12],q[5];
cx q[14],q[1];
cx q[3],q[21];
cx q[17],q[19];
cx q[11],q[4];
cx q[26],q[2];
cx q[16],q[0];
cx q[23],q[25];
cx q[11],q[20];
cx q[26],q[10];
cx q[15],q[3];
cx q[8],q[25];
cx q[14],q[21];
cx q[24],q[18];
cx q[19],q[6];
cx q[23],q[4];
cx q[0],q[16];
cx q[8],q[11];
cx q[13],q[9];
cx q[6],q[14];
cx q[20],q[0];
cx q[21],q[3];
cx q[10],q[26];
cx q[22],q[24];
cx q[12],q[15];
cx q[17],q[19];
cx q[25],q[23];
cx q[5],q[2];
cx q[16],q[18];
cx q[0],q[11];
cx q[9],q[2];
cx q[23],q[8];
cx q[19],q[22];
cx q[21],q[1];
cx q[3],q[13];
cx q[24],q[17];
cx q[14],q[6];
cx q[18],q[16];
cx q[26],q[25];
cx q[6],q[1];
cx q[16],q[20];
cx q[22],q[17];
cx q[18],q[24];
cx q[21],q[12];
cx q[2],q[10];
cx q[11],q[4];
cx q[13],q[3];
cx q[5],q[7];
cx q[25],q[8];
cx q[14],q[19];
cx q[2],q[3];
cx q[26],q[5];
cx q[12],q[9];
cx q[15],q[7];
cx q[8],q[11];
cx q[0],q[20];
cx q[21],q[13];
cx q[14],q[19];
cx q[24],q[18];
cx q[17],q[22];
cx q[1],q[6];
cx q[3],q[15];
cx q[22],q[17];
cx q[23],q[0];
cx q[26],q[10];
cx q[24],q[18];
cx q[11],q[8];
cx q[19],q[1];
cx q[20],q[4];
cx q[6],q[13];
cx q[9],q[12];
cx q[2],q[7];
cx q[21],q[14];
cx q[4],q[16];
cx q[26],q[5];
cx q[24],q[20];
cx q[9],q[15];
cx q[23],q[0];
cx q[2],q[10];
cx q[11],q[8];
cx q[6],q[12];
cx q[4],q[20];
cx q[25],q[2];
cx q[8],q[11];
cx q[1],q[21];
cx q[9],q[7];
cx q[22],q[24];
cx q[5],q[26];
cx q[13],q[14];
cx q[18],q[17];
cx q[0],q[16];
cx q[19],q[14];
cx q[5],q[8];
cx q[9],q[15];
cx q[25],q[2];
cx q[11],q[0];
cx q[22],q[24];
cx q[17],q[18];
cx q[6],q[14];
cx q[9],q[3];
cx q[5],q[26];
cx q[25],q[10];
cx q[8],q[2];
cx q[0],q[11];
cx q[12],q[15];
cx q[6],q[15];
cx q[16],q[20];
cx q[13],q[14];
cx q[3],q[9];
cx q[22],q[19];
cx q[10],q[25];
cx q[12],q[7];
cx q[21],q[1];
cx q[5],q[26];
cx q[23],q[11];
cx q[2],q[8];
cx q[1],q[17];
cx q[11],q[20];
cx q[13],q[14];
cx q[0],q[23];
cx q[7],q[6];
cx q[19],q[22];
cx q[25],q[10];
cx q[1],q[13];
cx q[20],q[23];
cx q[6],q[7];
cx q[16],q[18];
cx q[24],q[19];
cx q[2],q[10];
cx q[12],q[9];
cx q[8],q[26];
cx q[14],q[15];
cx q[11],q[0];
cx q[4],q[5];
cx q[22],q[17];
cx q[18],q[16];
cx q[4],q[2];
cx q[11],q[5];
cx q[24],q[22];
cx q[14],q[7];
cx q[1],q[19];
cx q[9],q[10];
cx q[23],q[20];
cx q[17],q[21];
cx q[3],q[25];
cx q[13],q[15];
cx q[6],q[12];
cx q[6],q[25];
cx q[16],q[18];
cx q[4],q[11];
cx q[12],q[7];
cx q[3],q[9];
cx q[8],q[26];
cx q[0],q[23];
cx q[17],q[21];
cx q[13],q[14];
cx q[6],q[12];
cx q[15],q[17];
cx q[20],q[23];
cx q[5],q[0];
cx q[9],q[10];
cx q[25],q[3];
cx q[19],q[21];
cx q[2],q[8];
cx q[16],q[18];
cx q[16],q[23];
cx q[15],q[7];
cx q[9],q[8];
cx q[26],q[11];
cx q[14],q[21];
cx q[1],q[22];
cx q[25],q[2];
cx q[12],q[13];
cx q[17],q[14];
cx q[8],q[26];
cx q[16],q[20];
cx q[23],q[5];
cx q[3],q[10];
cx q[4],q[11];
cx q[6],q[25];
cx q[24],q[1];
cx q[11],q[26];
cx q[19],q[22];
cx q[2],q[8];
cx q[1],q[20];
cx q[7],q[14];
cx q[12],q[13];
cx q[3],q[10];
cx q[17],q[21];
cx q[26],q[0];
cx q[23],q[16];
cx q[8],q[9];
cx q[3],q[10];
cx q[4],q[5];
cx q[7],q[15];
cx q[21],q[14];
cx q[22],q[19];
cx q[25],q[11];
cx q[24],q[1];
cx q[20],q[18];
cx q[20],q[1];
cx q[15],q[21];
cx q[23],q[16];
cx q[3],q[7];
cx q[6],q[10];
cx q[0],q[2];
cx q[8],q[9];
cx q[19],q[22];
cx q[4],q[16];
cx q[2],q[0];
cx q[17],q[14];
cx q[21],q[7];
cx q[13],q[10];
cx q[22],q[18];
cx q[5],q[2];
cx q[16],q[26];
cx q[18],q[20];
cx q[7],q[12];
cx q[11],q[0];
cx q[21],q[15];
cx q[22],q[14];
cx q[8],q[6];
cx q[3],q[10];
cx q[7],q[13];
cx q[10],q[3];
cx q[2],q[26];
cx q[8],q[25];
cx q[15],q[12];
cx q[17],q[21];
cx q[20],q[19];
cx q[6],q[9];
cx q[16],q[4];
cx q[0],q[5];
cx q[24],q[23];
cx q[18],q[22];
cx q[14],q[18];
cx q[23],q[19];
cx q[13],q[7];
cx q[26],q[16];
cx q[22],q[21];
cx q[17],q[15];
cx q[11],q[9];
cx q[5],q[25];
cx q[26],q[0];
cx q[7],q[12];
cx q[10],q[3];
cx q[21],q[22];
cx q[18],q[14];
cx q[23],q[24];
cx q[4],q[1];
cx q[2],q[8];
cx q[21],q[22];
cx q[23],q[1];
cx q[5],q[8];
cx q[24],q[20];
cx q[26],q[2];
cx q[14],q[19];
cx q[25],q[5];
cx q[19],q[20];
cx q[4],q[16];
cx q[22],q[18];
cx q[7],q[12];
cx q[0],q[26];
cx q[23],q[1];
cx q[15],q[17];
cx q[10],q[13];
cx q[14],q[21];
cx q[3],q[6];
cx q[11],q[9];
cx q[3],q[7];
cx q[25],q[2];
cx q[14],q[22];
cx q[6],q[10];
cx q[4],q[16];
cx q[19],q[24];
cx q[21],q[20];
cx q[23],q[1];
cx q[17],q[13];
cx q[7],q[13];
cx q[11],q[6];
cx q[18],q[22];
cx q[5],q[25];
cx q[14],q[21];
cx q[3],q[9];
cx q[16],q[23];
cx q[15],q[22];
cx q[19],q[21];
cx q[9],q[3];
cx q[8],q[2];
cx q[23],q[24];
cx q[5],q[6];
cx q[25],q[8];
cx q[10],q[11];
cx q[3],q[7];
cx q[20],q[21];
cx q[1],q[16];
cx q[15],q[22];
cx q[19],q[14];
cx q[2],q[0];
cx q[21],q[19];
cx q[14],q[22];
cx q[5],q[9];
cx q[1],q[16];
cx q[11],q[10];
cx q[22],q[21];
cx q[5],q[25];
cx q[17],q[10];
cx q[3],q[11];
cx q[4],q[26];
cx q[18],q[15];
cx q[20],q[24];
cx q[19],q[23];
cx q[14],q[18];
cx q[3],q[17];
cx q[12],q[15];
cx q[1],q[26];
cx q[5],q[11];
cx q[9],q[10];
cx q[24],q[19];
cx q[11],q[10];
cx q[4],q[26];
cx q[3],q[17];
cx q[6],q[25];
cx q[7],q[15];
cx q[12],q[14];
cx q[23],q[16];
cx q[22],q[20];
cx q[3],q[10];
cx q[22],q[21];
cx q[0],q[26];
cx q[18],q[14];
cx q[19],q[20];
cx q[17],q[7];
cx q[16],q[24];
cx q[15],q[12];
cx q[5],q[6];
cx q[4],q[25];
cx q[1],q[23];
cx q[9],q[11];
cx q[13],q[12];
cx q[10],q[3];
cx q[5],q[8];
cx q[21],q[14];
cx q[22],q[16];
cx q[15],q[18];
cx q[4],q[25];
cx q[9],q[11];
cx q[17],q[7];
cx q[1],q[0];
cx q[8],q[4];
cx q[15],q[13];
cx q[0],q[26];
cx q[12],q[17];
cx q[22],q[20];
cx q[9],q[11];
cx q[5],q[6];
cx q[2],q[25];
cx q[21],q[16];
cx q[20],q[24];
cx q[23],q[26];
cx q[15],q[13];
cx q[2],q[25];
cx q[5],q[11];
cx q[7],q[10];
cx q[16],q[22];
cx q[3],q[9];
cx q[1],q[0];
cx q[17],q[12];
cx q[12],q[13];
cx q[22],q[18];
cx q[7],q[9];
cx q[23],q[26];
cx q[1],q[0];
cx q[17],q[14];
cx q[8],q[25];
cx q[19],q[15];
cx q[5],q[6];
cx q[19],q[18];
cx q[10],q[9];
cx q[25],q[1];
cx q[26],q[23];
cx q[6],q[5];
cx q[7],q[3];
cx q[8],q[4];
cx q[16],q[21];
cx q[12],q[17];
cx q[2],q[0];
cx q[15],q[13];
cx q[22],q[20];
cx q[3],q[6];
cx q[2],q[26];
cx q[24],q[0];
cx q[20],q[22];
cx q[4],q[8];
cx q[10],q[7];
cx q[25],q[1];
cx q[7],q[5];
cx q[10],q[11];
cx q[6],q[3];
cx q[25],q[4];
cx q[24],q[23];
cx q[13],q[12];
cx q[14],q[15];
cx q[16],q[21];
cx q[26],q[0];
cx q[20],q[22];
cx q[11],q[12];
cx q[7],q[9];
cx q[4],q[3];
cx q[0],q[24];
cx q[16],q[21];
cx q[17],q[13];
cx q[15],q[14];
cx q[20],q[21];
cx q[3],q[4];
cx q[25],q[2];
cx q[1],q[26];
cx q[17],q[14];
cx q[12],q[15];
cx q[9],q[11];
cx q[7],q[6];
cx q[13],q[11];
cx q[6],q[8];
cx q[21],q[20];
cx q[17],q[19];
cx q[3],q[4];
cx q[12],q[15];
cx q[7],q[10];
cx q[6],q[8];
cx q[5],q[4];
cx q[26],q[24];
cx q[16],q[17];
cx q[25],q[1];
cx q[2],q[3];
cx q[12],q[15];
cx q[18],q[21];
cx q[11],q[13];
cx q[23],q[20];
