OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[21],q[24];
cx q[6],q[1];
cx q[25],q[23];
cx q[7],q[13];
cx q[3],q[2];
cx q[10],q[18];
cx q[12],q[11];
cx q[9],q[16];
cx q[20],q[21];
cx q[1],q[22];
cx q[15],q[26];
cx q[12],q[19];
cx q[25],q[23];
cx q[11],q[10];
cx q[0],q[13];
cx q[9],q[3];
cx q[14],q[7];
cx q[5],q[16];
cx q[8],q[18];
cx q[20],q[15];
cx q[16],q[9];
cx q[26],q[22];
cx q[1],q[25];
cx q[17],q[18];
cx q[8],q[3];
cx q[19],q[14];
cx q[10],q[12];
cx q[20],q[24];
cx q[26],q[15];
cx q[3],q[8];
cx q[16],q[9];
cx q[4],q[17];
cx q[10],q[12];
cx q[7],q[14];
cx q[18],q[17];
cx q[11],q[10];
cx q[25],q[23];
cx q[2],q[9];
cx q[13],q[0];
cx q[22],q[15];
cx q[25],q[23];
cx q[11],q[18];
cx q[10],q[7];
cx q[17],q[4];
cx q[6],q[22];
cx q[0],q[21];
cx q[26],q[20];
cx q[5],q[16];
cx q[9],q[16];
cx q[22],q[20];
cx q[24],q[13];
cx q[2],q[8];
cx q[12],q[18];
cx q[21],q[15];
cx q[5],q[25];
cx q[7],q[11];
cx q[23],q[6];
cx q[0],q[14];
cx q[26],q[1];
cx q[17],q[3];
cx q[5],q[23];
cx q[17],q[3];
cx q[9],q[8];
cx q[22],q[26];
cx q[20],q[21];
cx q[7],q[10];
cx q[14],q[0];
cx q[1],q[6];
cx q[11],q[18];
cx q[2],q[16];
cx q[24],q[13];
cx q[1],q[26];
cx q[0],q[14];
cx q[10],q[11];
cx q[5],q[23];
cx q[2],q[3];
cx q[18],q[4];
cx q[12],q[17];
cx q[25],q[9];
cx q[8],q[16];
cx q[18],q[11];
cx q[5],q[23];
cx q[3],q[2];
cx q[19],q[14];
cx q[21],q[22];
cx q[6],q[26];
cx q[0],q[13];
cx q[8],q[16];
cx q[11],q[10];
cx q[9],q[5];
cx q[15],q[21];
cx q[1],q[6];
cx q[20],q[0];
cx q[0],q[15];
cx q[12],q[17];
cx q[1],q[6];
cx q[3],q[8];
cx q[14],q[19];
cx q[21],q[22];
cx q[11],q[18];
cx q[13],q[24];
cx q[23],q[9];
cx q[16],q[9];
cx q[21],q[26];
cx q[1],q[6];
cx q[8],q[2];
cx q[13],q[24];
cx q[10],q[14];
cx q[11],q[18];
cx q[3],q[16];
cx q[15],q[26];
cx q[18],q[4];
cx q[23],q[1];
cx q[9],q[5];
cx q[20],q[0];
cx q[11],q[10];
cx q[16],q[2];
cx q[5],q[6];
cx q[1],q[22];
cx q[26],q[21];
cx q[7],q[10];
cx q[8],q[4];
cx q[25],q[17];
cx q[19],q[13];
cx q[17],q[25];
cx q[13],q[11];
cx q[20],q[26];
cx q[4],q[8];
cx q[18],q[12];
cx q[1],q[5];
cx q[3],q[23];
cx q[13],q[11];
cx q[4],q[17];
cx q[5],q[1];
cx q[12],q[8];
cx q[26],q[21];
cx q[19],q[24];
cx q[14],q[7];
cx q[3],q[16];
cx q[2],q[23];
cx q[15],q[0];
cx q[4],q[25];
cx q[5],q[3];
cx q[20],q[19];
cx q[21],q[22];
cx q[10],q[8];
cx q[13],q[24];
cx q[1],q[6];
cx q[7],q[18];
cx q[17],q[12];
cx q[11],q[14];
cx q[23],q[16];
cx q[15],q[20];
cx q[11],q[7];
cx q[24],q[19];
cx q[10],q[14];
cx q[25],q[4];
cx q[3],q[6];
cx q[21],q[1];
cx q[2],q[9];
cx q[17],q[8];
cx q[9],q[2];
cx q[5],q[3];
cx q[16],q[23];
cx q[18],q[7];
cx q[14],q[13];
cx q[19],q[24];
cx q[1],q[21];
cx q[15],q[20];
cx q[4],q[12];
cx q[22],q[26];
cx q[3],q[16];
cx q[21],q[6];
cx q[2],q[23];
cx q[15],q[20];
cx q[13],q[14];
cx q[16],q[3];
cx q[8],q[10];
cx q[2],q[23];
cx q[9],q[25];
cx q[19],q[24];
cx q[11],q[13];
cx q[21],q[1];
cx q[11],q[19];
cx q[21],q[26];
cx q[15],q[0];
cx q[9],q[25];
cx q[23],q[3];
cx q[5],q[6];
cx q[4],q[12];
cx q[14],q[7];
cx q[17],q[8];
cx q[15],q[0];
cx q[11],q[19];
cx q[24],q[20];
cx q[17],q[8];
cx q[5],q[2];
cx q[16],q[6];
cx q[10],q[18];
cx q[22],q[26];
cx q[10],q[17];
cx q[4],q[8];
cx q[16],q[5];
cx q[1],q[21];
cx q[7],q[11];
cx q[0],q[15];
cx q[19],q[20];
cx q[14],q[18];
cx q[23],q[2];
cx q[3],q[9];
cx q[26],q[0];
cx q[21],q[6];
cx q[5],q[2];
cx q[15],q[24];
cx q[18],q[10];
cx q[1],q[16];
cx q[3],q[23];
cx q[14],q[13];
cx q[8],q[17];
cx q[4],q[25];
cx q[11],q[19];
cx q[2],q[1];
cx q[17],q[25];
cx q[13],q[14];
cx q[21],q[16];
cx q[22],q[26];
cx q[3],q[9];
cx q[19],q[20];
cx q[8],q[12];
cx q[23],q[9];
cx q[6],q[21];
cx q[1],q[5];
cx q[0],q[22];
cx q[2],q[16];
cx q[3],q[4];
cx q[6],q[26];
cx q[21],q[1];
cx q[19],q[20];
cx q[15],q[24];
cx q[18],q[12];
cx q[8],q[10];
cx q[13],q[14];
cx q[4],q[25];
cx q[8],q[10];
cx q[16],q[22];
cx q[13],q[11];
cx q[0],q[24];
cx q[21],q[26];
cx q[3],q[4];
cx q[1],q[2];
cx q[3],q[4];
cx q[18],q[7];
cx q[15],q[11];
cx q[13],q[14];
cx q[20],q[0];
cx q[21],q[26];
cx q[17],q[25];
cx q[6],q[16];
cx q[8],q[12];
cx q[6],q[16];
cx q[19],q[15];
cx q[2],q[1];
cx q[9],q[5];
cx q[20],q[24];
cx q[14],q[11];
cx q[0],q[22];
cx q[3],q[4];
cx q[8],q[7];
cx q[23],q[5];
cx q[12],q[10];
cx q[17],q[25];
cx q[24],q[22];
cx q[21],q[6];
cx q[23],q[2];
cx q[7],q[13];
cx q[1],q[21];
cx q[5],q[4];
cx q[20],q[24];
cx q[0],q[6];
cx q[22],q[16];
cx q[8],q[7];
cx q[3],q[25];
cx q[10],q[12];
cx q[22],q[0];
cx q[23],q[2];
cx q[21],q[16];
cx q[13],q[18];
cx q[4],q[9];
cx q[20],q[24];
cx q[19],q[15];
cx q[26],q[1];
cx q[2],q[23];
cx q[13],q[8];
cx q[0],q[22];
cx q[21],q[16];
cx q[12],q[14];
cx q[24],q[19];
cx q[18],q[15];
cx q[7],q[17];
cx q[5],q[9];
cx q[25],q[10];
cx q[18],q[11];
cx q[26],q[6];
cx q[7],q[17];
cx q[22],q[20];
cx q[8],q[13];
cx q[24],q[15];
cx q[4],q[3];
cx q[21],q[16];
cx q[12],q[14];
cx q[0],q[19];
cx q[2],q[1];
cx q[18],q[24];
cx q[10],q[17];
cx q[9],q[3];
cx q[15],q[19];
cx q[12],q[7];
cx q[21],q[26];
cx q[14],q[8];
cx q[1],q[23];
cx q[16],q[22];
cx q[21],q[0];
cx q[24],q[18];
cx q[12],q[7];
cx q[9],q[4];
cx q[2],q[1];
cx q[8],q[13];
cx q[25],q[17];
cx q[6],q[23];
cx q[20],q[16];
cx q[26],q[22];
cx q[8],q[11];
cx q[5],q[25];
cx q[6],q[1];
cx q[0],q[20];
cx q[21],q[16];
cx q[9],q[2];
cx q[19],q[15];
cx q[3],q[4];
cx q[3],q[25];
cx q[23],q[26];
cx q[14],q[8];
cx q[24],q[19];
cx q[4],q[1];
cx q[21],q[16];
cx q[5],q[9];
cx q[17],q[7];
cx q[22],q[21];
cx q[15],q[20];
cx q[17],q[14];
cx q[18],q[13];
cx q[16],q[0];
cx q[26],q[23];
cx q[12],q[10];
cx q[4],q[5];
cx q[19],q[24];
cx q[9],q[25];
cx q[2],q[1];
cx q[11],q[18];
cx q[20],q[19];
cx q[10],q[12];
cx q[9],q[3];
cx q[4],q[25];
cx q[16],q[0];
cx q[14],q[7];
cx q[15],q[24];
cx q[17],q[13];
cx q[0],q[16];
cx q[3],q[10];
cx q[26],q[21];
cx q[6],q[1];
cx q[15],q[24];
cx q[14],q[7];
cx q[13],q[11];
cx q[8],q[17];
cx q[10],q[12];
cx q[23],q[6];
cx q[9],q[3];
cx q[15],q[20];
cx q[21],q[26];
cx q[8],q[7];
cx q[18],q[17];
cx q[13],q[11];
cx q[22],q[16];
cx q[21],q[26];
cx q[10],q[14];
cx q[7],q[8];
cx q[20],q[0];
cx q[15],q[24];
cx q[2],q[1];
cx q[9],q[12];
cx q[18],q[17];
cx q[23],q[6];
cx q[25],q[3];
cx q[20],q[0];
cx q[4],q[3];
cx q[11],q[17];
cx q[25],q[12];
cx q[26],q[23];
cx q[8],q[14];
cx q[5],q[1];
cx q[10],q[9];
cx q[19],q[24];
cx q[22],q[21];
cx q[4],q[5];
cx q[13],q[18];
cx q[19],q[20];
cx q[21],q[0];
cx q[17],q[14];
cx q[1],q[2];
cx q[12],q[9];
cx q[7],q[8];
cx q[23],q[16];
cx q[18],q[14];
cx q[3],q[25];
cx q[7],q[8];
cx q[22],q[20];
cx q[12],q[9];
cx q[19],q[15];
cx q[6],q[26];
cx q[5],q[4];
cx q[22],q[20];
cx q[25],q[4];
cx q[6],q[26];
cx q[14],q[18];
cx q[15],q[11];
cx q[21],q[16];
cx q[1],q[2];
cx q[9],q[3];
cx q[3],q[4];
cx q[2],q[6];
cx q[17],q[10];
cx q[9],q[8];
cx q[26],q[23];
cx q[13],q[11];
cx q[20],q[22];
cx q[12],q[7];
cx q[21],q[16];
cx q[23],q[26];
cx q[25],q[2];
cx q[1],q[6];
cx q[10],q[7];
cx q[20],q[22];
cx q[24],q[19];
cx q[16],q[21];
cx q[20],q[22];
cx q[14],q[12];
cx q[25],q[2];
cx q[9],q[10];
cx q[0],q[13];
cx q[7],q[11];
cx q[3],q[5];
cx q[1],q[26];
cx q[8],q[4];
cx q[18],q[15];
cx q[26],q[1];
cx q[16],q[21];
cx q[9],q[10];
cx q[6],q[5];
cx q[15],q[18];
cx q[11],q[7];
cx q[12],q[14];
cx q[16],q[23];
cx q[1],q[21];
cx q[17],q[18];
cx q[25],q[5];
cx q[9],q[10];
cx q[12],q[14];
cx q[6],q[26];
cx q[7],q[11];
cx q[22],q[20];
cx q[24],q[19];
cx q[15],q[13];
cx q[4],q[9];
cx q[11],q[7];
cx q[13],q[19];
cx q[25],q[5];
cx q[26],q[21];
cx q[14],q[12];
cx q[10],q[8];
cx q[15],q[18];
cx q[6],q[2];
cx q[22],q[16];
cx q[23],q[21];
cx q[3],q[5];
cx q[8],q[4];
cx q[18],q[19];
cx q[9],q[14];
cx q[0],q[20];
cx q[1],q[26];
cx q[7],q[12];
cx q[6],q[2];
cx q[11],q[17];
cx q[24],q[13];
cx q[18],q[15];
cx q[2],q[26];
cx q[20],q[16];
cx q[6],q[25];
cx q[9],q[14];
cx q[12],q[7];
cx q[26],q[22];
cx q[9],q[8];
cx q[25],q[5];
cx q[18],q[17];
cx q[23],q[2];
cx q[0],q[16];
cx q[7],q[14];
cx q[4],q[3];
cx q[20],q[19];
cx q[11],q[12];
cx q[24],q[19];
cx q[21],q[22];
cx q[8],q[14];
cx q[26],q[2];
cx q[20],q[15];
cx q[25],q[1];
cx q[13],q[17];
cx q[9],q[10];
cx q[2],q[23];
cx q[14],q[10];
cx q[4],q[25];
cx q[21],q[22];
cx q[11],q[12];
cx q[24],q[16];
cx q[1],q[6];
cx q[9],q[3];
cx q[15],q[19];
cx q[11],q[14];
cx q[13],q[18];
cx q[10],q[5];
cx q[21],q[22];
cx q[17],q[12];
cx q[24],q[20];
cx q[8],q[9];
cx q[16],q[0];
cx q[1],q[6];
cx q[19],q[15];
cx q[16],q[0];
cx q[7],q[11];
cx q[20],q[24];
cx q[23],q[2];
cx q[6],q[1];
cx q[26],q[21];
cx q[13],q[18];
cx q[5],q[10];
cx q[3],q[4];
cx q[9],q[14];
cx q[20],q[16];
cx q[17],q[11];
cx q[1],q[2];
cx q[8],q[5];
cx q[10],q[4];
cx q[7],q[14];
cx q[22],q[26];
cx q[25],q[6];
cx q[19],q[15];
cx q[0],q[21];
cx q[18],q[13];
cx q[5],q[9];
cx q[11],q[14];
cx q[10],q[3];
cx q[2],q[23];
cx q[13],q[17];
cx q[18],q[15];
cx q[25],q[4];
cx q[8],q[7];
cx q[20],q[19];
cx q[9],q[14];
cx q[21],q[24];
cx q[23],q[2];
cx q[1],q[6];
cx q[19],q[15];
cx q[26],q[0];
cx q[12],q[17];
cx q[8],q[10];
cx q[7],q[11];
cx q[16],q[20];
cx q[5],q[3];
cx q[13],q[18];
cx q[0],q[25];
cx q[5],q[8];
cx q[21],q[20];
cx q[14],q[17];
cx q[15],q[18];
cx q[4],q[3];
cx q[15],q[13];
cx q[16],q[19];
cx q[21],q[20];
cx q[25],q[2];
cx q[12],q[17];
cx q[7],q[14];
cx q[23],q[0];
cx q[8],q[10];
cx q[0],q[23];
cx q[1],q[3];
cx q[24],q[21];
cx q[4],q[8];
cx q[2],q[25];
cx q[9],q[7];
cx q[26],q[0];
cx q[12],q[9];
cx q[22],q[24];
cx q[1],q[3];
cx q[18],q[15];
cx q[23],q[2];
cx q[16],q[19];
cx q[17],q[11];
cx q[7],q[10];
cx q[23],q[26];
cx q[3],q[1];
cx q[4],q[6];
cx q[7],q[10];
cx q[18],q[16];
cx q[21],q[22];
cx q[8],q[5];
cx q[17],q[13];
cx q[12],q[9];
cx q[20],q[19];
cx q[2],q[1];
cx q[4],q[3];
cx q[9],q[12];
cx q[19],q[18];
cx q[14],q[11];
cx q[8],q[7];
cx q[24],q[23];
