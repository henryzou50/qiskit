OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[20],q[26];
cx q[25],q[5];
cx q[23],q[19];
cx q[9],q[17];
cx q[15],q[10];
cx q[24],q[11];
cx q[16],q[12];
cx q[21],q[1];
cx q[4],q[2];
cx q[14],q[6];
cx q[3],q[13];
cx q[26],q[0];
cx q[23],q[19];
cx q[14],q[6];
cx q[12],q[17];
cx q[10],q[13];
cx q[25],q[5];
cx q[11],q[9];
cx q[2],q[4];
cx q[21],q[1];
cx q[24],q[22];
cx q[7],q[20];
cx q[18],q[3];
cx q[21],q[4];
cx q[25],q[7];
cx q[18],q[15];
cx q[9],q[11];
cx q[13],q[10];
cx q[17],q[8];
cx q[26],q[20];
cx q[5],q[1];
cx q[16],q[6];
cx q[0],q[19];
cx q[20],q[26];
cx q[10],q[13];
cx q[25],q[5];
cx q[3],q[15];
cx q[16],q[12];
cx q[2],q[24];
cx q[8],q[11];
cx q[14],q[6];
cx q[23],q[18];
cx q[19],q[0];
cx q[21],q[1];
cx q[17],q[12];
cx q[8],q[11];
cx q[24],q[1];
cx q[25],q[4];
cx q[18],q[23];
cx q[22],q[2];
cx q[6],q[16];
cx q[20],q[26];
cx q[16],q[13];
cx q[17],q[12];
cx q[19],q[0];
cx q[15],q[14];
cx q[22],q[2];
cx q[9],q[11];
cx q[5],q[21];
cx q[5],q[7];
cx q[21],q[26];
cx q[16],q[10];
cx q[13],q[12];
cx q[25],q[4];
cx q[22],q[9];
cx q[19],q[0];
cx q[11],q[2];
cx q[18],q[3];
cx q[24],q[1];
cx q[8],q[17];
cx q[2],q[11];
cx q[10],q[16];
cx q[25],q[24];
cx q[17],q[8];
cx q[18],q[3];
cx q[5],q[4];
cx q[1],q[9];
cx q[23],q[0];
cx q[13],q[6];
cx q[7],q[21];
cx q[15],q[12];
cx q[20],q[26];
cx q[20],q[5];
cx q[22],q[9];
cx q[0],q[3];
cx q[12],q[13];
cx q[23],q[26];
cx q[4],q[7];
cx q[17],q[8];
cx q[2],q[11];
cx q[18],q[19];
cx q[21],q[26];
cx q[9],q[24];
cx q[23],q[0];
cx q[16],q[10];
cx q[13],q[17];
cx q[3],q[18];
cx q[4],q[7];
cx q[20],q[5];
cx q[19],q[18];
cx q[9],q[1];
cx q[26],q[5];
cx q[25],q[7];
cx q[15],q[6];
cx q[10],q[13];
cx q[3],q[23];
cx q[2],q[22];
cx q[17],q[8];
cx q[14],q[16];
cx q[21],q[0];
cx q[15],q[17];
cx q[2],q[8];
cx q[19],q[14];
cx q[12],q[10];
cx q[3],q[20];
cx q[4],q[1];
cx q[6],q[13];
cx q[15],q[17];
cx q[20],q[0];
cx q[25],q[9];
cx q[21],q[23];
cx q[10],q[19];
cx q[11],q[8];
cx q[2],q[24];
cx q[16],q[12];
cx q[18],q[14];
cx q[1],q[4];
cx q[5],q[26];
cx q[12],q[6];
cx q[3],q[5];
cx q[8],q[11];
cx q[1],q[25];
cx q[18],q[14];
cx q[9],q[22];
cx q[24],q[2];
cx q[2],q[9];
cx q[26],q[7];
cx q[22],q[25];
cx q[0],q[14];
cx q[12],q[13];
cx q[1],q[4];
cx q[15],q[8];
cx q[10],q[16];
cx q[18],q[23];
cx q[14],q[5];
cx q[16],q[18];
cx q[17],q[8];
cx q[19],q[23];
cx q[7],q[3];
cx q[10],q[13];
cx q[1],q[22];
cx q[25],q[9];
cx q[4],q[26];
cx q[12],q[15];
cx q[11],q[17];
cx q[13],q[15];
cx q[23],q[18];
cx q[4],q[1];
cx q[8],q[6];
cx q[25],q[22];
cx q[10],q[19];
cx q[2],q[24];
cx q[5],q[21];
cx q[5],q[0];
cx q[3],q[20];
cx q[13],q[16];
cx q[7],q[26];
cx q[10],q[19];
cx q[14],q[21];
cx q[6],q[8];
cx q[23],q[18];
cx q[1],q[4];
cx q[22],q[9];
cx q[14],q[0];
cx q[23],q[18];
cx q[21],q[5];
cx q[26],q[7];
cx q[13],q[12];
cx q[24],q[2];
cx q[25],q[1];
cx q[3],q[20];
cx q[22],q[9];
cx q[16],q[10];
cx q[14],q[18];
cx q[2],q[11];
cx q[24],q[22];
cx q[7],q[1];
cx q[4],q[9];
cx q[21],q[20];
cx q[26],q[3];
cx q[23],q[19];
cx q[6],q[17];
cx q[16],q[12];
cx q[3],q[1];
cx q[21],q[5];
cx q[24],q[9];
cx q[6],q[11];
cx q[19],q[18];
cx q[22],q[2];
cx q[15],q[8];
cx q[20],q[26];
cx q[0],q[14];
cx q[18],q[19];
cx q[1],q[3];
cx q[0],q[14];
cx q[9],q[24];
cx q[20],q[26];
cx q[17],q[8];
cx q[16],q[23];
cx q[5],q[21];
cx q[14],q[0];
cx q[1],q[25];
cx q[4],q[7];
cx q[2],q[24];
cx q[17],q[12];
cx q[15],q[13];
cx q[8],q[11];
cx q[26],q[3];
cx q[10],q[16];
cx q[19],q[18];
cx q[22],q[6];
cx q[21],q[20];
cx q[14],q[19];
cx q[21],q[5];
cx q[26],q[3];
cx q[13],q[10];
cx q[12],q[17];
cx q[4],q[1];
cx q[9],q[24];
cx q[2],q[22];
cx q[25],q[7];
cx q[6],q[11];
cx q[16],q[18];
cx q[0],q[20];
cx q[11],q[6];
cx q[9],q[22];
cx q[18],q[23];
cx q[8],q[12];
cx q[1],q[3];
cx q[0],q[20];
cx q[15],q[16];
cx q[3],q[5];
cx q[6],q[2];
cx q[14],q[18];
cx q[25],q[26];
cx q[11],q[12];
cx q[17],q[10];
cx q[8],q[13];
cx q[24],q[22];
cx q[23],q[19];
cx q[4],q[7];
cx q[22],q[9];
cx q[4],q[1];
cx q[10],q[17];
cx q[24],q[7];
cx q[21],q[14];
cx q[16],q[23];
cx q[6],q[2];
cx q[25],q[4];
cx q[14],q[20];
cx q[1],q[7];
cx q[16],q[23];
cx q[21],q[19];
cx q[22],q[9];
cx q[8],q[13];
cx q[0],q[5];
cx q[24],q[26];
cx q[2],q[11];
cx q[25],q[3];
cx q[10],q[23];
cx q[15],q[18];
cx q[2],q[6];
cx q[0],q[5];
cx q[9],q[22];
cx q[6],q[2];
cx q[14],q[21];
cx q[15],q[18];
cx q[13],q[8];
cx q[0],q[25];
cx q[12],q[11];
cx q[17],q[16];
cx q[5],q[20];
cx q[22],q[7];
cx q[26],q[3];
cx q[14],q[18];
cx q[24],q[3];
cx q[26],q[1];
cx q[23],q[15];
cx q[17],q[16];
cx q[25],q[5];
cx q[0],q[5];
cx q[23],q[18];
cx q[8],q[17];
cx q[16],q[15];
cx q[9],q[7];
cx q[3],q[24];
cx q[13],q[11];
cx q[4],q[22];
cx q[25],q[26];
cx q[4],q[22];
cx q[16],q[15];
cx q[14],q[21];
cx q[9],q[6];
cx q[24],q[26];
cx q[11],q[2];
cx q[8],q[17];
cx q[18],q[23];
cx q[24],q[26];
cx q[22],q[3];
cx q[10],q[13];
cx q[9],q[12];
cx q[23],q[18];
cx q[21],q[14];
cx q[5],q[20];
cx q[25],q[1];
cx q[19],q[18];
cx q[13],q[17];
cx q[24],q[26];
cx q[16],q[8];
cx q[3],q[6];
cx q[9],q[11];
cx q[12],q[10];
cx q[1],q[0];
cx q[23],q[14];
cx q[21],q[5];
cx q[22],q[4];
cx q[7],q[2];
cx q[11],q[9];
cx q[12],q[17];
cx q[20],q[25];
cx q[10],q[15];
cx q[21],q[5];
cx q[14],q[23];
cx q[26],q[22];
cx q[3],q[4];
cx q[22],q[26];
cx q[13],q[8];
cx q[24],q[0];
cx q[10],q[15];
cx q[18],q[14];
cx q[21],q[5];
cx q[2],q[6];
cx q[4],q[3];
cx q[9],q[12];
cx q[7],q[11];
cx q[9],q[11];
cx q[8],q[13];
cx q[0],q[24];
cx q[22],q[3];
cx q[19],q[16];
cx q[20],q[5];
cx q[14],q[23];
cx q[10],q[15];
cx q[25],q[1];
cx q[6],q[4];
cx q[12],q[17];
cx q[2],q[9];
cx q[13],q[12];
cx q[5],q[21];
cx q[3],q[22];
cx q[0],q[25];
cx q[17],q[7];
cx q[18],q[14];
cx q[1],q[20];
cx q[16],q[15];
cx q[19],q[8];
cx q[20],q[5];
cx q[15],q[8];
cx q[6],q[4];
cx q[2],q[9];
cx q[21],q[23];
cx q[11],q[10];
cx q[19],q[14];
cx q[17],q[7];
cx q[3],q[24];
cx q[23],q[20];
cx q[2],q[9];
cx q[1],q[25];
cx q[3],q[4];
cx q[21],q[14];
cx q[8],q[13];
cx q[17],q[11];
cx q[15],q[8];
cx q[4],q[22];
cx q[20],q[1];
cx q[2],q[3];
cx q[0],q[25];
cx q[19],q[18];
cx q[19],q[16];
cx q[0],q[5];
cx q[10],q[9];
cx q[14],q[21];
cx q[3],q[4];
cx q[15],q[8];
cx q[13],q[11];
cx q[6],q[2];
cx q[1],q[23];
cx q[26],q[24];
cx q[17],q[12];
cx q[20],q[23];
cx q[3],q[26];
cx q[2],q[22];
cx q[13],q[11];
cx q[10],q[9];
cx q[19],q[15];
cx q[22],q[26];
cx q[1],q[20];
cx q[5],q[0];
cx q[19],q[16];
cx q[23],q[21];
cx q[17],q[9];
cx q[17],q[13];
cx q[6],q[10];
cx q[22],q[26];
cx q[3],q[25];
cx q[7],q[2];
cx q[5],q[24];
cx q[23],q[21];
cx q[18],q[15];
cx q[18],q[8];
cx q[20],q[0];
cx q[10],q[6];
cx q[11],q[17];
cx q[5],q[25];
cx q[4],q[7];
cx q[2],q[26];
cx q[6],q[13];
cx q[24],q[1];
cx q[10],q[4];
cx q[16],q[21];
cx q[3],q[22];
cx q[19],q[14];
cx q[17],q[8];
cx q[26],q[2];
cx q[11],q[9];
cx q[5],q[25];
cx q[18],q[12];
cx q[23],q[20];
cx q[1],q[5];
cx q[17],q[11];
cx q[6],q[10];
cx q[25],q[3];
cx q[0],q[24];
cx q[4],q[7];
cx q[19],q[14];
cx q[22],q[2];
cx q[1],q[25];
cx q[5],q[0];
cx q[24],q[23];
cx q[4],q[10];
cx q[15],q[20];
cx q[21],q[14];
cx q[2],q[26];
cx q[12],q[18];
cx q[3],q[22];
cx q[4],q[7];
cx q[21],q[15];
cx q[8],q[9];
cx q[0],q[5];
cx q[12],q[17];
cx q[24],q[23];
cx q[20],q[16];
cx q[10],q[6];
cx q[26],q[1];
cx q[8],q[11];
cx q[4],q[26];
cx q[25],q[22];
cx q[10],q[7];
cx q[13],q[17];
cx q[24],q[5];
cx q[12],q[18];
cx q[9],q[6];
cx q[23],q[20];
cx q[15],q[14];
cx q[14],q[15];
cx q[10],q[7];
cx q[26],q[2];
cx q[20],q[23];
cx q[16],q[21];
cx q[11],q[12];
cx q[5],q[22];
cx q[8],q[17];
cx q[13],q[9];
cx q[18],q[19];
cx q[24],q[0];
cx q[1],q[4];
cx q[25],q[3];
cx q[2],q[26];
cx q[12],q[11];
cx q[14],q[18];
cx q[13],q[17];
cx q[19],q[15];
cx q[7],q[10];
cx q[9],q[6];
cx q[23],q[21];
cx q[22],q[0];
cx q[13],q[9];
cx q[15],q[12];
cx q[11],q[17];
cx q[6],q[10];
cx q[26],q[4];
cx q[19],q[14];
cx q[24],q[0];
cx q[25],q[3];
cx q[18],q[16];
cx q[17],q[9];
cx q[20],q[23];
cx q[12],q[11];
cx q[5],q[22];
cx q[24],q[0];
cx q[18],q[21];
cx q[14],q[15];
cx q[25],q[5];
cx q[3],q[26];
cx q[8],q[9];
cx q[13],q[10];
cx q[6],q[2];
cx q[7],q[4];
cx q[23],q[20];
cx q[16],q[19];
cx q[12],q[11];
cx q[8],q[9];
cx q[25],q[0];
cx q[3],q[5];
cx q[24],q[23];
cx q[17],q[12];
cx q[18],q[20];
cx q[1],q[26];
cx q[7],q[2];
cx q[13],q[10];
cx q[9],q[13];
cx q[19],q[16];
cx q[2],q[6];
cx q[26],q[1];
cx q[5],q[3];
cx q[23],q[24];
cx q[7],q[4];
cx q[14],q[15];
cx q[20],q[21];
cx q[25],q[0];
cx q[3],q[0];
cx q[12],q[8];
cx q[20],q[22];
cx q[16],q[19];
cx q[6],q[10];
cx q[7],q[4];
cx q[25],q[24];
cx q[11],q[17];
cx q[1],q[0];
cx q[4],q[2];
cx q[18],q[19];
cx q[10],q[9];
cx q[24],q[3];
cx q[5],q[7];
cx q[17],q[11];
cx q[25],q[22];
cx q[7],q[26];
cx q[15],q[12];
cx q[23],q[24];
cx q[2],q[4];
cx q[9],q[10];
cx q[3],q[0];
cx q[19],q[18];
cx q[14],q[16];
cx q[13],q[11];
cx q[6],q[7];
cx q[5],q[2];
cx q[18],q[21];
cx q[10],q[8];
cx q[9],q[4];
cx q[26],q[3];
cx q[0],q[24];
cx q[19],q[16];
cx q[14],q[13];
cx q[2],q[5];
cx q[7],q[4];
cx q[6],q[9];
cx q[1],q[26];
cx q[18],q[21];
cx q[10],q[11];
cx q[15],q[17];
cx q[1],q[26];
cx q[7],q[4];
cx q[8],q[11];
cx q[21],q[20];
cx q[24],q[23];
cx q[16],q[17];
cx q[18],q[19];
cx q[5],q[2];
cx q[12],q[15];
cx q[9],q[6];
cx q[25],q[0];
cx q[10],q[13];
