OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[9],q[7];
cx q[12],q[14];
cx q[21],q[5];
cx q[6],q[3];
cx q[13],q[18];
cx q[15],q[4];
cx q[9],q[5];
cx q[18],q[10];
cx q[19],q[7];
cx q[4],q[15];
cx q[21],q[1];
cx q[2],q[23];
cx q[20],q[8];
cx q[17],q[25];
cx q[11],q[14];
cx q[3],q[0];
cx q[20],q[24];
cx q[1],q[25];
cx q[9],q[21];
cx q[17],q[15];
cx q[10],q[13];
cx q[22],q[19];
cx q[6],q[23];
cx q[12],q[18];
cx q[0],q[2];
cx q[8],q[20];
cx q[22],q[5];
cx q[6],q[15];
cx q[26],q[17];
cx q[14],q[11];
cx q[13],q[10];
cx q[5],q[22];
cx q[26],q[19];
cx q[4],q[25];
cx q[13],q[18];
cx q[6],q[17];
cx q[21],q[9];
cx q[3],q[23];
cx q[20],q[24];
cx q[12],q[14];
cx q[19],q[17];
cx q[20],q[1];
cx q[25],q[11];
cx q[4],q[7];
cx q[2],q[23];
cx q[5],q[9];
cx q[21],q[8];
cx q[6],q[3];
cx q[12],q[18];
cx q[13],q[10];
cx q[19],q[22];
cx q[10],q[13];
cx q[17],q[4];
cx q[8],q[5];
cx q[9],q[21];
cx q[12],q[25];
cx q[0],q[2];
cx q[16],q[20];
cx q[1],q[11];
cx q[3],q[15];
cx q[8],q[19];
cx q[26],q[17];
cx q[13],q[10];
cx q[16],q[24];
cx q[3],q[23];
cx q[6],q[4];
cx q[8],q[5];
cx q[0],q[12];
cx q[17],q[7];
cx q[25],q[11];
cx q[21],q[20];
cx q[17],q[19];
cx q[6],q[15];
cx q[26],q[8];
cx q[24],q[16];
cx q[14],q[18];
cx q[13],q[12];
cx q[3],q[2];
cx q[11],q[1];
cx q[7],q[4];
cx q[14],q[18];
cx q[19],q[8];
cx q[13],q[0];
cx q[10],q[2];
cx q[17],q[7];
cx q[22],q[26];
cx q[16],q[20];
cx q[23],q[15];
cx q[1],q[24];
cx q[12],q[25];
cx q[7],q[26];
cx q[0],q[10];
cx q[19],q[8];
cx q[16],q[20];
cx q[3],q[15];
cx q[7],q[22];
cx q[6],q[17];
cx q[0],q[10];
cx q[14],q[12];
cx q[21],q[24];
cx q[9],q[21];
cx q[15],q[23];
cx q[11],q[16];
cx q[13],q[0];
cx q[19],q[8];
cx q[1],q[25];
cx q[12],q[0];
cx q[19],q[8];
cx q[3],q[23];
cx q[16],q[11];
cx q[5],q[7];
cx q[15],q[10];
cx q[2],q[10];
cx q[15],q[23];
cx q[4],q[1];
cx q[22],q[19];
cx q[11],q[21];
cx q[25],q[18];
cx q[6],q[17];
cx q[12],q[13];
cx q[13],q[0];
cx q[21],q[11];
cx q[8],q[9];
cx q[15],q[23];
cx q[10],q[12];
cx q[4],q[6];
cx q[9],q[24];
cx q[23],q[15];
cx q[16],q[4];
cx q[5],q[22];
cx q[2],q[12];
cx q[8],q[19];
cx q[17],q[6];
cx q[25],q[0];
cx q[11],q[1];
cx q[15],q[3];
cx q[5],q[26];
cx q[16],q[4];
cx q[0],q[12];
cx q[23],q[2];
cx q[22],q[9];
cx q[1],q[11];
cx q[6],q[17];
cx q[19],q[26];
cx q[4],q[16];
cx q[5],q[22];
cx q[10],q[2];
cx q[8],q[5];
cx q[11],q[20];
cx q[7],q[26];
cx q[21],q[9];
cx q[16],q[23];
cx q[3],q[12];
cx q[1],q[4];
cx q[18],q[13];
cx q[5],q[26];
cx q[1],q[14];
cx q[17],q[2];
cx q[23],q[6];
cx q[15],q[3];
cx q[9],q[11];
cx q[24],q[8];
cx q[0],q[25];
cx q[16],q[3];
cx q[7],q[19];
cx q[18],q[25];
cx q[4],q[14];
cx q[17],q[2];
cx q[8],q[11];
cx q[5],q[21];
cx q[13],q[10];
cx q[7],q[22];
cx q[21],q[5];
cx q[16],q[12];
cx q[10],q[13];
cx q[17],q[6];
cx q[19],q[3];
cx q[15],q[6];
cx q[12],q[16];
cx q[4],q[23];
cx q[17],q[3];
cx q[21],q[8];
cx q[0],q[14];
cx q[1],q[20];
cx q[26],q[5];
cx q[10],q[25];
cx q[6],q[12];
cx q[5],q[26];
cx q[20],q[4];
cx q[9],q[1];
cx q[19],q[3];
cx q[7],q[22];
cx q[13],q[2];
cx q[0],q[25];
cx q[14],q[25];
cx q[3],q[17];
cx q[5],q[26];
cx q[13],q[0];
cx q[9],q[20];
cx q[2],q[6];
cx q[18],q[23];
cx q[19],q[15];
cx q[7],q[22];
cx q[15],q[12];
cx q[4],q[17];
cx q[11],q[20];
cx q[7],q[3];
cx q[14],q[25];
cx q[5],q[26];
cx q[0],q[13];
cx q[1],q[23];
cx q[24],q[21];
cx q[10],q[6];
cx q[22],q[3];
cx q[8],q[11];
cx q[23],q[18];
cx q[21],q[26];
cx q[6],q[16];
cx q[1],q[9];
cx q[4],q[19];
cx q[10],q[0];
cx q[17],q[15];
cx q[22],q[21];
cx q[15],q[17];
cx q[8],q[11];
cx q[10],q[6];
cx q[4],q[9];
cx q[24],q[1];
cx q[7],q[5];
cx q[4],q[9];
cx q[7],q[5];
cx q[6],q[16];
cx q[13],q[18];
cx q[15],q[19];
cx q[8],q[24];
cx q[12],q[6];
cx q[26],q[22];
cx q[4],q[1];
cx q[14],q[10];
cx q[3],q[19];
cx q[23],q[9];
cx q[19],q[3];
cx q[4],q[8];
cx q[11],q[21];
cx q[16],q[2];
cx q[26],q[22];
cx q[13],q[18];
cx q[9],q[15];
cx q[14],q[10];
cx q[17],q[6];
cx q[7],q[20];
cx q[20],q[18];
cx q[23],q[1];
cx q[4],q[21];
cx q[12],q[15];
cx q[7],q[17];
cx q[5],q[3];
cx q[13],q[9];
cx q[0],q[16];
cx q[22],q[26];
cx q[10],q[14];
cx q[12],q[15];
cx q[4],q[1];
cx q[25],q[13];
cx q[0],q[10];
cx q[26],q[11];
cx q[2],q[6];
cx q[24],q[11];
cx q[22],q[26];
cx q[9],q[13];
cx q[4],q[23];
cx q[14],q[6];
cx q[17],q[7];
cx q[12],q[15];
cx q[10],q[18];
cx q[7],q[15];
cx q[20],q[9];
cx q[13],q[17];
cx q[21],q[24];
cx q[6],q[18];
cx q[8],q[23];
cx q[5],q[11];
cx q[1],q[8];
cx q[18],q[25];
cx q[10],q[6];
cx q[19],q[13];
cx q[26],q[24];
cx q[20],q[9];
cx q[15],q[7];
cx q[14],q[2];
cx q[17],q[9];
cx q[16],q[14];
cx q[4],q[21];
cx q[12],q[7];
cx q[11],q[3];
cx q[10],q[6];
cx q[18],q[25];
cx q[13],q[19];
cx q[23],q[8];
cx q[26],q[22];
cx q[21],q[4];
cx q[0],q[10];
cx q[14],q[2];
cx q[3],q[22];
cx q[15],q[16];
cx q[20],q[6];
cx q[13],q[19];
cx q[1],q[8];
cx q[4],q[21];
cx q[18],q[9];
cx q[20],q[25];
cx q[14],q[2];
cx q[5],q[26];
cx q[16],q[15];
cx q[22],q[3];
cx q[7],q[15];
cx q[1],q[4];
cx q[6],q[2];
cx q[18],q[13];
cx q[17],q[23];
cx q[9],q[20];
cx q[26],q[24];
cx q[5],q[21];
cx q[22],q[11];
cx q[16],q[10];
cx q[14],q[2];
cx q[7],q[12];
cx q[23],q[8];
cx q[18],q[19];
cx q[9],q[20];
cx q[0],q[6];
cx q[11],q[22];
cx q[1],q[25];
cx q[4],q[26];
cx q[7],q[18];
cx q[10],q[6];
cx q[4],q[1];
cx q[20],q[0];
cx q[2],q[16];
cx q[12],q[15];
cx q[11],q[3];
cx q[17],q[0];
cx q[4],q[8];
cx q[12],q[2];
cx q[6],q[20];
cx q[18],q[25];
cx q[13],q[1];
cx q[16],q[10];
cx q[24],q[26];
cx q[9],q[0];
cx q[19],q[7];
cx q[11],q[3];
cx q[5],q[21];
cx q[15],q[18];
cx q[1],q[13];
cx q[23],q[8];
cx q[6],q[20];
cx q[7],q[19];
cx q[20],q[16];
cx q[23],q[17];
cx q[11],q[22];
cx q[21],q[3];
cx q[14],q[16];
cx q[26],q[21];
cx q[4],q[24];
cx q[8],q[23];
cx q[0],q[6];
cx q[25],q[13];
cx q[7],q[2];
cx q[25],q[4];
cx q[17],q[18];
cx q[10],q[15];
cx q[0],q[6];
cx q[5],q[26];
cx q[1],q[17];
cx q[25],q[8];
cx q[21],q[3];
cx q[19],q[22];
cx q[20],q[15];
cx q[12],q[2];
cx q[26],q[21];
cx q[8],q[4];
cx q[23],q[5];
cx q[3],q[26];
cx q[24],q[5];
cx q[6],q[9];
cx q[4],q[23];
cx q[1],q[2];
cx q[17],q[13];
cx q[22],q[19];
cx q[7],q[13];
cx q[15],q[12];
cx q[20],q[10];
cx q[21],q[5];
cx q[9],q[6];
cx q[5],q[21];
cx q[10],q[20];
cx q[1],q[19];
cx q[9],q[6];
cx q[4],q[8];
cx q[22],q[11];
cx q[25],q[13];
cx q[10],q[18];
cx q[6],q[9];
cx q[21],q[24];
cx q[11],q[22];
cx q[3],q[26];
cx q[19],q[1];
cx q[7],q[23];
cx q[20],q[15];
cx q[4],q[5];
cx q[23],q[8];
cx q[9],q[14];
cx q[22],q[26];
cx q[16],q[0];
cx q[7],q[17];
cx q[18],q[2];
cx q[3],q[21];
cx q[1],q[19];
cx q[5],q[21];
cx q[14],q[16];
cx q[0],q[20];
cx q[9],q[6];
cx q[3],q[22];
cx q[19],q[11];
cx q[2],q[1];
cx q[1],q[12];
cx q[11],q[21];
cx q[16],q[6];
cx q[5],q[24];
cx q[17],q[8];
cx q[13],q[25];
cx q[10],q[0];
cx q[2],q[19];
cx q[24],q[4];
cx q[25],q[19];
cx q[11],q[22];
cx q[8],q[17];
cx q[9],q[7];
cx q[20],q[12];
cx q[21],q[3];
cx q[6],q[13];
cx q[9],q[6];
cx q[18],q[12];
cx q[24],q[11];
cx q[22],q[5];
cx q[10],q[14];
cx q[3],q[26];
cx q[1],q[25];
cx q[18],q[12];
cx q[21],q[24];
cx q[19],q[1];
cx q[7],q[17];
cx q[15],q[0];
cx q[5],q[22];
cx q[16],q[10];
cx q[18],q[1];
cx q[15],q[0];
cx q[25],q[2];
cx q[6],q[14];
cx q[17],q[7];
cx q[22],q[5];
cx q[8],q[23];
cx q[20],q[12];
cx q[24],q[11];
cx q[0],q[15];
cx q[25],q[19];
cx q[16],q[10];
cx q[9],q[14];
cx q[10],q[16];
cx q[8],q[13];
cx q[15],q[20];
cx q[21],q[3];
cx q[6],q[17];
cx q[19],q[14];
cx q[6],q[16];
cx q[18],q[12];
cx q[17],q[14];
cx q[8],q[13];
cx q[0],q[15];
cx q[11],q[24];
cx q[7],q[19];
cx q[1],q[20];
cx q[24],q[11];
cx q[0],q[12];
cx q[10],q[17];
cx q[4],q[5];
cx q[25],q[2];
cx q[7],q[19];
cx q[12],q[1];
cx q[24],q[3];
cx q[11],q[5];
cx q[18],q[26];
cx q[15],q[0];
cx q[17],q[6];
cx q[14],q[9];
cx q[3],q[4];
cx q[23],q[8];
cx q[16],q[9];
cx q[7],q[14];
cx q[10],q[15];
cx q[25],q[2];
cx q[26],q[1];
cx q[10],q[15];
cx q[3],q[5];
cx q[18],q[20];
cx q[14],q[26];
cx q[6],q[17];
cx q[24],q[13];
cx q[11],q[24];
cx q[18],q[20];
cx q[7],q[9];
cx q[5],q[25];
cx q[17],q[6];
cx q[14],q[26];
cx q[13],q[22];
cx q[10],q[15];
cx q[21],q[4];
cx q[16],q[19];
cx q[18],q[12];
cx q[19],q[23];
cx q[13],q[8];
cx q[11],q[24];
cx q[1],q[10];
cx q[17],q[9];
cx q[24],q[11];
cx q[4],q[25];
cx q[10],q[0];
cx q[18],q[20];
cx q[23],q[22];
cx q[0],q[10];
cx q[5],q[3];
cx q[19],q[14];
cx q[18],q[2];
cx q[11],q[8];
cx q[6],q[15];
cx q[26],q[1];
cx q[7],q[0];
cx q[16],q[17];
cx q[13],q[19];
cx q[1],q[20];
cx q[21],q[4];
cx q[10],q[15];
cx q[22],q[8];
cx q[18],q[14];
cx q[1],q[7];
cx q[25],q[5];
cx q[23],q[11];
cx q[3],q[2];
cx q[8],q[22];
cx q[17],q[6];
cx q[15],q[20];
cx q[25],q[24];
cx q[8],q[22];
cx q[26],q[14];
cx q[10],q[7];
cx q[1],q[20];
cx q[15],q[17];
cx q[18],q[13];
cx q[12],q[1];
cx q[4],q[2];
cx q[19],q[16];
cx q[7],q[20];
cx q[8],q[13];
cx q[17],q[15];
cx q[5],q[3];
cx q[9],q[6];
cx q[12],q[1];
cx q[25],q[8];
cx q[23],q[3];
cx q[15],q[17];
cx q[21],q[5];
cx q[19],q[14];
cx q[24],q[2];
cx q[18],q[6];
cx q[26],q[22];
cx q[24],q[2];
cx q[25],q[11];
cx q[5],q[4];
cx q[8],q[13];
cx q[18],q[6];
cx q[17],q[10];
