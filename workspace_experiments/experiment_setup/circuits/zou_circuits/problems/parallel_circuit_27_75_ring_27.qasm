OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[23],q[8];
cx q[0],q[12];
cx q[26],q[25];
cx q[24],q[15];
cx q[5],q[16];
cx q[1],q[9];
cx q[2],q[20];
cx q[10],q[17];
cx q[22],q[0];
cx q[6],q[16];
cx q[5],q[1];
cx q[14],q[12];
cx q[21],q[18];
cx q[2],q[20];
cx q[15],q[24];
cx q[7],q[26];
cx q[9],q[19];
cx q[9],q[16];
cx q[7],q[6];
cx q[24],q[14];
cx q[12],q[22];
cx q[18],q[17];
cx q[0],q[21];
cx q[3],q[13];
cx q[13],q[11];
cx q[17],q[18];
cx q[3],q[23];
cx q[4],q[20];
cx q[25],q[10];
cx q[6],q[9];
cx q[26],q[7];
cx q[12],q[21];
cx q[24],q[2];
cx q[5],q[1];
cx q[4],q[2];
cx q[6],q[7];
cx q[25],q[26];
cx q[0],q[21];
cx q[1],q[23];
cx q[9],q[19];
cx q[24],q[14];
cx q[18],q[10];
cx q[13],q[20];
cx q[5],q[8];
cx q[22],q[12];
cx q[20],q[13];
cx q[7],q[26];
cx q[3],q[8];
cx q[23],q[11];
cx q[10],q[25];
cx q[15],q[22];
cx q[1],q[19];
cx q[15],q[24];
cx q[9],q[19];
cx q[14],q[4];
cx q[17],q[0];
cx q[18],q[25];
cx q[13],q[20];
cx q[1],q[9];
cx q[5],q[8];
cx q[2],q[4];
cx q[7],q[16];
cx q[26],q[6];
cx q[0],q[10];
cx q[23],q[11];
cx q[18],q[25];
cx q[13],q[14];
cx q[14],q[4];
cx q[7],q[9];
cx q[20],q[13];
cx q[16],q[6];
cx q[10],q[22];
cx q[26],q[19];
cx q[15],q[24];
cx q[1],q[8];
cx q[23],q[13];
cx q[25],q[10];
cx q[17],q[12];
cx q[2],q[21];
cx q[9],q[8];
cx q[26],q[18];
cx q[11],q[5];
cx q[7],q[16];
cx q[3],q[13];
cx q[5],q[11];
cx q[20],q[14];
cx q[2],q[21];
cx q[10],q[0];
cx q[17],q[12];
cx q[16],q[7];
cx q[9],q[8];
cx q[22],q[15];
cx q[6],q[18];
cx q[4],q[24];
cx q[12],q[10];
cx q[9],q[16];
cx q[13],q[3];
cx q[18],q[0];
cx q[19],q[1];
cx q[14],q[23];
cx q[20],q[4];
cx q[2],q[21];
cx q[22],q[15];
cx q[8],q[5];
cx q[7],q[6];
cx q[21],q[12];
cx q[3],q[5];
cx q[17],q[15];
cx q[20],q[24];
cx q[11],q[1];
cx q[0],q[10];
cx q[22],q[4];
cx q[23],q[14];
cx q[16],q[19];
cx q[9],q[7];
cx q[9],q[7];
cx q[22],q[21];
cx q[6],q[26];
cx q[8],q[11];
cx q[16],q[19];
cx q[12],q[15];
cx q[2],q[4];
cx q[23],q[5];
cx q[0],q[10];
cx q[12],q[22];
cx q[5],q[1];
cx q[26],q[18];
cx q[7],q[6];
cx q[2],q[21];
cx q[14],q[23];
cx q[25],q[0];
cx q[8],q[11];
cx q[3],q[13];
cx q[20],q[24];
cx q[16],q[19];
cx q[8],q[11];
cx q[7],q[9];
cx q[26],q[6];
cx q[18],q[25];
cx q[17],q[22];
cx q[5],q[13];
cx q[24],q[23];
cx q[21],q[12];
cx q[15],q[0];
cx q[19],q[16];
cx q[8],q[11];
cx q[5],q[13];
cx q[17],q[12];
cx q[6],q[7];
cx q[2],q[21];
cx q[9],q[19];
cx q[4],q[21];
cx q[12],q[17];
cx q[20],q[3];
cx q[14],q[5];
cx q[1],q[23];
cx q[25],q[0];
cx q[19],q[7];
cx q[9],q[16];
cx q[17],q[4];
cx q[21],q[24];
cx q[3],q[2];
cx q[26],q[6];
cx q[9],q[16];
cx q[14],q[5];
cx q[2],q[21];
cx q[16],q[19];
cx q[13],q[8];
cx q[17],q[10];
cx q[1],q[23];
cx q[25],q[26];
cx q[9],q[11];
cx q[18],q[0];
cx q[7],q[6];
cx q[10],q[17];
cx q[22],q[18];
cx q[13],q[8];
cx q[6],q[26];
cx q[16],q[9];
cx q[4],q[2];
cx q[7],q[11];
cx q[5],q[14];
cx q[12],q[15];
cx q[16],q[11];
cx q[21],q[20];
cx q[23],q[5];
cx q[6],q[19];
cx q[8],q[1];
cx q[24],q[10];
cx q[18],q[22];
cx q[15],q[12];
cx q[4],q[2];
cx q[9],q[13];
cx q[25],q[0];
cx q[24],q[10];
cx q[3],q[21];
cx q[19],q[26];
cx q[1],q[8];
cx q[16],q[11];
cx q[15],q[12];
cx q[25],q[18];
cx q[9],q[13];
cx q[4],q[20];
cx q[26],q[6];
cx q[0],q[25];
cx q[14],q[3];
cx q[18],q[12];
cx q[8],q[5];
cx q[1],q[5];
cx q[13],q[8];
cx q[24],q[20];
cx q[7],q[11];
cx q[6],q[19];
cx q[17],q[21];
cx q[26],q[25];
cx q[6],q[25];
cx q[17],q[4];
cx q[7],q[11];
cx q[8],q[13];
cx q[23],q[3];
cx q[12],q[0];
cx q[18],q[15];
cx q[21],q[14];
cx q[1],q[5];
cx q[16],q[9];
cx q[24],q[20];
cx q[21],q[4];
cx q[13],q[9];
cx q[17],q[24];
cx q[2],q[20];
cx q[22],q[15];
cx q[1],q[5];
cx q[0],q[18];
cx q[25],q[19];
cx q[9],q[13];
cx q[22],q[15];
cx q[23],q[4];
cx q[10],q[20];
cx q[6],q[25];
cx q[17],q[21];
cx q[8],q[1];
cx q[7],q[5];
cx q[2],q[10];
cx q[11],q[26];
cx q[19],q[16];
cx q[18],q[25];
cx q[24],q[4];
cx q[21],q[20];
cx q[23],q[14];
cx q[1],q[9];
cx q[8],q[1];
cx q[21],q[17];
cx q[16],q[19];
cx q[13],q[7];
cx q[5],q[9];
cx q[11],q[25];
cx q[15],q[22];
cx q[1],q[24];
cx q[22],q[0];
cx q[18],q[12];
cx q[26],q[25];
cx q[23],q[3];
cx q[20],q[4];
cx q[2],q[20];
cx q[13],q[7];
cx q[3],q[17];
cx q[0],q[15];
cx q[11],q[25];
cx q[23],q[4];
cx q[14],q[9];
cx q[19],q[26];
cx q[16],q[6];
cx q[1],q[24];
cx q[12],q[18];
cx q[11],q[6];
cx q[26],q[25];
cx q[21],q[10];
cx q[16],q[13];
cx q[9],q[14];
cx q[15],q[0];
cx q[18],q[12];
cx q[23],q[1];
cx q[8],q[24];
cx q[20],q[17];
cx q[7],q[5];
cx q[19],q[6];
cx q[11],q[26];
cx q[18],q[15];
cx q[2],q[10];
cx q[1],q[3];
cx q[17],q[4];
cx q[12],q[25];
cx q[19],q[16];
cx q[12],q[26];
cx q[4],q[1];
cx q[5],q[14];
cx q[13],q[7];
cx q[10],q[17];
cx q[20],q[2];
cx q[7],q[13];
cx q[2],q[20];
cx q[1],q[3];
cx q[19],q[16];
cx q[26],q[6];
cx q[18],q[0];
cx q[14],q[8];
cx q[24],q[9];
cx q[12],q[15];
cx q[22],q[0];
cx q[1],q[23];
cx q[3],q[4];
cx q[26],q[16];
cx q[6],q[11];
cx q[19],q[13];
cx q[2],q[10];
cx q[18],q[25];
cx q[21],q[17];
cx q[3],q[24];
cx q[4],q[1];
cx q[22],q[18];
cx q[20],q[10];
cx q[17],q[21];
cx q[9],q[8];
cx q[12],q[26];
cx q[14],q[13];
cx q[5],q[7];
cx q[25],q[15];
cx q[12],q[19];
cx q[24],q[3];
cx q[14],q[16];
cx q[2],q[10];
cx q[4],q[1];
cx q[17],q[20];
cx q[13],q[6];
cx q[9],q[5];
cx q[11],q[22];
cx q[20],q[10];
cx q[5],q[8];
cx q[1],q[4];
cx q[6],q[13];
cx q[7],q[16];
cx q[15],q[18];
cx q[1],q[4];
cx q[14],q[8];
cx q[20],q[17];
cx q[21],q[0];
cx q[12],q[19];
cx q[11],q[22];
cx q[23],q[3];
cx q[2],q[10];
cx q[5],q[24];
cx q[26],q[25];
cx q[13],q[6];
cx q[15],q[18];
cx q[16],q[7];
cx q[15],q[22];
cx q[9],q[8];
cx q[16],q[6];
cx q[24],q[3];
cx q[0],q[17];
cx q[2],q[10];
cx q[4],q[23];
cx q[13],q[25];
cx q[18],q[20];
cx q[26],q[11];
cx q[5],q[7];
cx q[0],q[21];
cx q[10],q[17];
cx q[2],q[23];
cx q[11],q[25];
cx q[24],q[3];
cx q[22],q[15];
cx q[9],q[8];
cx q[5],q[7];
cx q[13],q[12];
cx q[1],q[4];
cx q[8],q[9];
cx q[11],q[12];
cx q[3],q[4];
cx q[0],q[22];
cx q[20],q[10];
cx q[26],q[18];
cx q[1],q[2];
cx q[6],q[13];
cx q[14],q[7];
cx q[25],q[15];
cx q[17],q[23];
cx q[1],q[3];
cx q[16],q[14];
cx q[5],q[7];
cx q[15],q[25];
cx q[10],q[17];
cx q[13],q[12];
cx q[8],q[9];
cx q[26],q[0];
cx q[11],q[19];
cx q[22],q[20];
cx q[21],q[10];
cx q[26],q[0];
cx q[19],q[25];
cx q[11],q[13];
cx q[16],q[6];
cx q[7],q[8];
cx q[18],q[15];
cx q[23],q[1];
cx q[3],q[4];
cx q[9],q[3];
cx q[15],q[25];
cx q[10],q[22];
cx q[12],q[13];
cx q[18],q[26];
cx q[6],q[14];
cx q[1],q[24];
cx q[16],q[5];
cx q[23],q[17];
cx q[16],q[5];
cx q[22],q[10];
cx q[2],q[23];
cx q[14],q[6];
cx q[9],q[4];
cx q[8],q[7];
cx q[19],q[11];
cx q[22],q[17];
cx q[8],q[5];
cx q[6],q[14];
cx q[19],q[11];
cx q[15],q[26];
cx q[3],q[24];
cx q[4],q[7];
cx q[12],q[13];
cx q[23],q[2];
cx q[24],q[3];
cx q[9],q[4];
cx q[23],q[2];
cx q[12],q[14];
cx q[10],q[21];
cx q[11],q[13];
cx q[8],q[6];
cx q[0],q[15];
cx q[25],q[18];
cx q[1],q[4];
cx q[13],q[19];
cx q[24],q[9];
cx q[2],q[23];
cx q[22],q[17];
cx q[18],q[25];
cx q[10],q[21];
cx q[3],q[24];
cx q[16],q[14];
cx q[19],q[13];
cx q[15],q[25];
cx q[0],q[21];
cx q[6],q[7];
cx q[8],q[12];
cx q[2],q[23];
cx q[22],q[17];
cx q[4],q[1];
cx q[20],q[10];
cx q[24],q[9];
cx q[8],q[7];
cx q[5],q[6];
cx q[17],q[2];
cx q[4],q[1];
cx q[25],q[26];
cx q[0],q[22];
cx q[18],q[19];
cx q[10],q[20];
cx q[14],q[12];
cx q[11],q[16];
cx q[3],q[23];
cx q[15],q[26];
cx q[24],q[9];
cx q[19],q[13];
cx q[0],q[17];
cx q[6],q[7];
cx q[12],q[11];
cx q[14],q[8];
cx q[10],q[25];
cx q[21],q[23];
cx q[20],q[22];
cx q[4],q[5];
cx q[16],q[11];
cx q[24],q[1];
cx q[6],q[5];
cx q[8],q[7];
cx q[10],q[25];
cx q[19],q[15];
cx q[0],q[20];
cx q[3],q[2];
cx q[13],q[18];
cx q[9],q[4];
cx q[23],q[17];
cx q[5],q[6];
cx q[4],q[9];
cx q[18],q[13];
cx q[24],q[2];
cx q[12],q[16];
cx q[25],q[20];
cx q[22],q[26];
cx q[19],q[10];
cx q[3],q[24];
cx q[1],q[9];
cx q[21],q[0];
cx q[7],q[6];
cx q[5],q[4];
cx q[13],q[16];
cx q[22],q[26];
cx q[14],q[12];
cx q[19],q[22];
cx q[26],q[20];
cx q[5],q[4];
cx q[17],q[21];
cx q[23],q[24];
cx q[10],q[25];
cx q[13],q[15];
cx q[18],q[16];
cx q[2],q[0];
cx q[8],q[6];
cx q[1],q[3];
cx q[18],q[14];
cx q[20],q[21];
cx q[8],q[11];
cx q[19],q[13];
cx q[5],q[4];
cx q[26],q[25];
cx q[7],q[12];
cx q[9],q[3];
cx q[0],q[17];
cx q[22],q[10];
cx q[24],q[23];
cx q[2],q[23];
cx q[0],q[24];
cx q[8],q[11];
cx q[16],q[19];
cx q[4],q[5];
cx q[15],q[10];
cx q[7],q[12];
cx q[21],q[17];
cx q[13],q[14];
cx q[26],q[22];
cx q[20],q[25];
cx q[17],q[21];
cx q[13],q[11];
cx q[24],q[0];
cx q[2],q[23];
cx q[16],q[19];
cx q[10],q[15];
cx q[12],q[18];
cx q[6],q[9];
cx q[20],q[25];
cx q[22],q[26];
cx q[4],q[3];
cx q[1],q[4];
cx q[13],q[18];
cx q[7],q[6];
cx q[9],q[3];
cx q[19],q[10];
cx q[8],q[12];
cx q[16],q[14];
cx q[23],q[2];
cx q[22],q[15];
cx q[0],q[21];
cx q[24],q[17];
cx q[20],q[26];
cx q[13],q[14];
cx q[18],q[12];
cx q[16],q[10];
cx q[22],q[19];
cx q[15],q[26];
cx q[0],q[2];
cx q[23],q[21];
cx q[16],q[14];
cx q[2],q[0];
cx q[5],q[8];
cx q[13],q[18];
cx q[3],q[1];
cx q[4],q[9];
cx q[2],q[0];
cx q[21],q[23];
cx q[15],q[19];
cx q[13],q[10];
cx q[5],q[11];
cx q[22],q[20];
cx q[25],q[17];
cx q[18],q[12];
cx q[1],q[3];
cx q[6],q[8];
cx q[4],q[3];
cx q[5],q[11];
cx q[22],q[26];
cx q[16],q[10];
cx q[15],q[19];
cx q[25],q[21];
cx q[23],q[1];
cx q[12],q[11];
cx q[24],q[21];
cx q[14],q[18];
cx q[13],q[10];
cx q[26],q[17];
cx q[19],q[16];
cx q[6],q[4];
cx q[1],q[2];
cx q[25],q[20];
cx q[23],q[0];
cx q[24],q[26];
cx q[20],q[22];
cx q[15],q[16];
cx q[0],q[23];
cx q[6],q[8];
cx q[19],q[13];
cx q[2],q[1];
cx q[11],q[10];
cx q[8],q[5];
cx q[25],q[21];
cx q[18],q[11];
cx q[12],q[7];
cx q[4],q[9];
cx q[23],q[0];
cx q[17],q[22];
cx q[13],q[11];
cx q[18],q[19];
cx q[1],q[0];
cx q[6],q[4];
cx q[23],q[25];
cx q[12],q[7];
cx q[16],q[15];
cx q[22],q[20];
cx q[10],q[14];
cx q[26],q[21];
cx q[8],q[5];
cx q[2],q[3];
cx q[23],q[25];
cx q[5],q[9];
cx q[1],q[0];
cx q[18],q[13];
cx q[10],q[11];
cx q[21],q[26];
cx q[24],q[20];
cx q[0],q[1];
cx q[15],q[19];
cx q[12],q[8];
cx q[20],q[21];
cx q[4],q[2];
cx q[13],q[16];
cx q[25],q[26];
cx q[11],q[14];
cx q[9],q[5];
cx q[17],q[22];
cx q[26],q[21];
cx q[22],q[20];
cx q[4],q[6];
cx q[17],q[19];
cx q[14],q[13];
cx q[12],q[11];
cx q[23],q[24];
cx q[1],q[25];
cx q[22],q[21];
cx q[0],q[3];
cx q[14],q[16];
cx q[19],q[17];
cx q[18],q[15];
cx q[13],q[10];
cx q[12],q[11];
cx q[12],q[14];
cx q[13],q[16];
cx q[4],q[6];
cx q[25],q[26];
cx q[8],q[9];
cx q[10],q[11];
cx q[24],q[23];
cx q[7],q[5];
cx q[3],q[0];
cx q[19],q[22];
cx q[17],q[18];
