OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[3],q[15];
cx q[13],q[17];
cx q[7],q[16];
cx q[21],q[4];
cx q[12],q[5];
cx q[0],q[24];
cx q[2],q[6];
cx q[10],q[22];
cx q[7],q[0];
cx q[1],q[20];
cx q[14],q[11];
cx q[10],q[22];
cx q[6],q[2];
cx q[4],q[21];
cx q[18],q[19];
cx q[23],q[9];
cx q[5],q[12];
cx q[24],q[22];
cx q[9],q[21];
cx q[15],q[13];
cx q[10],q[20];
cx q[11],q[8];
cx q[3],q[19];
cx q[16],q[2];
cx q[19],q[3];
cx q[21],q[1];
cx q[4],q[6];
cx q[24],q[22];
cx q[13],q[8];
cx q[18],q[2];
cx q[15],q[5];
cx q[12],q[23];
cx q[20],q[10];
cx q[14],q[11];
cx q[7],q[16];
cx q[19],q[3];
cx q[18],q[14];
cx q[8],q[11];
cx q[20],q[23];
cx q[21],q[5];
cx q[13],q[15];
cx q[1],q[4];
cx q[7],q[0];
cx q[12],q[17];
cx q[1],q[5];
cx q[14],q[18];
cx q[22],q[24];
cx q[3],q[6];
cx q[4],q[16];
cx q[8],q[11];
cx q[20],q[10];
cx q[5],q[20];
cx q[6],q[2];
cx q[13],q[12];
cx q[1],q[24];
cx q[21],q[15];
cx q[11],q[3];
cx q[7],q[16];
cx q[19],q[8];
cx q[14],q[18];
cx q[4],q[0];
cx q[17],q[23];
cx q[22],q[10];
cx q[17],q[15];
cx q[1],q[5];
cx q[21],q[9];
cx q[8],q[19];
cx q[16],q[18];
cx q[7],q[0];
cx q[3],q[2];
cx q[20],q[10];
cx q[12],q[13];
cx q[8],q[13];
cx q[21],q[17];
cx q[2],q[18];
cx q[11],q[14];
cx q[24],q[0];
cx q[16],q[4];
cx q[20],q[5];
cx q[3],q[19];
cx q[12],q[23];
cx q[10],q[22];
cx q[9],q[6];
cx q[24],q[0];
cx q[3],q[14];
cx q[19],q[15];
cx q[11],q[2];
cx q[21],q[9];
cx q[16],q[7];
cx q[12],q[13];
cx q[17],q[20];
cx q[1],q[4];
cx q[10],q[22];
cx q[23],q[17];
cx q[5],q[20];
cx q[16],q[6];
cx q[14],q[8];
cx q[19],q[9];
cx q[11],q[2];
cx q[10],q[22];
cx q[24],q[1];
cx q[21],q[15];
cx q[4],q[0];
cx q[19],q[8];
cx q[1],q[4];
cx q[24],q[7];
cx q[10],q[22];
cx q[0],q[18];
cx q[12],q[13];
cx q[3],q[14];
cx q[17],q[20];
cx q[15],q[21];
cx q[15],q[21];
cx q[13],q[8];
cx q[4],q[6];
cx q[3],q[14];
cx q[0],q[18];
cx q[23],q[12];
cx q[17],q[20];
cx q[2],q[16];
cx q[9],q[19];
cx q[15],q[5];
cx q[2],q[11];
cx q[20],q[10];
cx q[3],q[14];
cx q[19],q[8];
cx q[12],q[17];
cx q[6],q[9];
cx q[18],q[16];
cx q[12],q[13];
cx q[20],q[10];
cx q[7],q[24];
cx q[11],q[14];
cx q[4],q[18];
cx q[23],q[17];
cx q[22],q[5];
cx q[6],q[3];
cx q[15],q[9];
cx q[16],q[2];
cx q[19],q[8];
cx q[12],q[13];
cx q[15],q[5];
cx q[19],q[3];
cx q[4],q[7];
cx q[9],q[6];
cx q[11],q[16];
cx q[22],q[24];
cx q[23],q[21];
cx q[7],q[4];
cx q[19],q[14];
cx q[3],q[6];
cx q[17],q[15];
cx q[9],q[1];
cx q[11],q[2];
cx q[18],q[16];
cx q[22],q[5];
cx q[23],q[12];
cx q[10],q[20];
cx q[21],q[8];
cx q[4],q[7];
cx q[8],q[1];
cx q[9],q[24];
cx q[5],q[20];
cx q[12],q[13];
cx q[22],q[10];
cx q[21],q[23];
cx q[3],q[6];
cx q[15],q[17];
cx q[2],q[14];
cx q[16],q[11];
cx q[0],q[18];
cx q[10],q[22];
cx q[9],q[5];
cx q[21],q[8];
cx q[0],q[18];
cx q[11],q[2];
cx q[17],q[15];
cx q[22],q[10];
cx q[19],q[8];
cx q[5],q[20];
cx q[23],q[17];
cx q[7],q[9];
cx q[24],q[4];
cx q[1],q[6];
cx q[16],q[18];
cx q[12],q[23];
cx q[18],q[0];
cx q[1],q[15];
cx q[21],q[13];
cx q[10],q[9];
cx q[2],q[14];
cx q[16],q[11];
cx q[8],q[6];
cx q[24],q[9];
cx q[2],q[11];
cx q[0],q[7];
cx q[3],q[8];
cx q[6],q[1];
cx q[13],q[19];
cx q[10],q[5];
cx q[12],q[23];
cx q[20],q[22];
cx q[15],q[21];
cx q[11],q[6];
cx q[10],q[22];
cx q[20],q[17];
cx q[1],q[24];
cx q[15],q[5];
cx q[13],q[12];
cx q[19],q[8];
cx q[6],q[16];
cx q[13],q[12];
cx q[24],q[1];
cx q[14],q[11];
cx q[3],q[2];
cx q[10],q[5];
cx q[20],q[17];
cx q[0],q[7];
cx q[4],q[18];
cx q[8],q[1];
cx q[20],q[24];
cx q[5],q[4];
cx q[11],q[2];
cx q[12],q[13];
cx q[7],q[9];
cx q[17],q[15];
cx q[6],q[16];
cx q[21],q[23];
cx q[24],q[10];
cx q[0],q[7];
cx q[23],q[21];
cx q[4],q[18];
cx q[22],q[20];
cx q[9],q[5];
cx q[2],q[3];
cx q[16],q[6];
cx q[15],q[17];
cx q[12],q[19];
cx q[13],q[8];
cx q[21],q[15];
cx q[24],q[10];
cx q[3],q[14];
cx q[5],q[9];
cx q[2],q[11];
cx q[0],q[16];
cx q[8],q[13];
cx q[6],q[1];
cx q[17],q[20];
cx q[4],q[7];
cx q[12],q[23];
cx q[2],q[11];
cx q[15],q[21];
cx q[8],q[14];
cx q[16],q[4];
cx q[10],q[9];
cx q[0],q[7];
cx q[12],q[19];
cx q[6],q[5];
cx q[18],q[3];
cx q[17],q[23];
cx q[20],q[22];
cx q[15],q[17];
cx q[13],q[11];
cx q[14],q[3];
cx q[10],q[22];
cx q[8],q[19];
cx q[6],q[4];
cx q[5],q[21];
cx q[5],q[6];
cx q[13],q[11];
cx q[16],q[18];
cx q[2],q[3];
cx q[15],q[17];
cx q[22],q[10];
cx q[23],q[12];
cx q[1],q[4];
cx q[24],q[20];
cx q[0],q[7];
cx q[14],q[21];
cx q[1],q[9];
cx q[24],q[5];
cx q[7],q[0];
cx q[4],q[6];
cx q[2],q[11];
cx q[13],q[21];
cx q[19],q[23];
cx q[10],q[22];
cx q[8],q[12];
cx q[3],q[14];
cx q[24],q[19];
cx q[7],q[6];
cx q[16],q[18];
cx q[11],q[2];
cx q[22],q[20];
cx q[23],q[17];
cx q[13],q[21];
cx q[0],q[4];
cx q[12],q[8];
cx q[1],q[15];
cx q[9],q[10];
cx q[0],q[4];
cx q[17],q[15];
cx q[9],q[1];
cx q[2],q[3];
cx q[21],q[13];
cx q[16],q[7];
cx q[5],q[24];
cx q[18],q[6];
cx q[14],q[11];
cx q[15],q[1];
cx q[23],q[20];
cx q[0],q[4];
cx q[8],q[19];
cx q[16],q[7];
cx q[21],q[14];
cx q[5],q[18];
cx q[2],q[11];
cx q[10],q[22];
cx q[3],q[6];
cx q[4],q[16];
cx q[13],q[21];
cx q[15],q[10];
cx q[18],q[14];
cx q[5],q[24];
cx q[6],q[7];
cx q[9],q[1];
cx q[23],q[20];
cx q[2],q[3];
cx q[8],q[19];
cx q[22],q[17];
cx q[8],q[13];
cx q[1],q[5];
cx q[10],q[22];
cx q[18],q[16];
cx q[4],q[9];
cx q[2],q[3];
cx q[15],q[17];
cx q[24],q[20];
cx q[21],q[19];
cx q[0],q[7];
cx q[23],q[12];
cx q[3],q[14];
cx q[21],q[5];
cx q[0],q[7];
cx q[10],q[9];
cx q[12],q[23];
cx q[2],q[11];
cx q[19],q[24];
cx q[8],q[13];
cx q[16],q[1];
cx q[6],q[18];
cx q[17],q[20];
cx q[15],q[10];
cx q[1],q[9];
cx q[19],q[13];
cx q[18],q[7];
cx q[5],q[21];
cx q[22],q[17];
cx q[16],q[0];
cx q[12],q[23];
cx q[24],q[20];
cx q[2],q[11];
cx q[6],q[3];
cx q[9],q[1];
cx q[15],q[17];
cx q[21],q[19];
cx q[7],q[4];
cx q[12],q[23];
cx q[8],q[11];
cx q[16],q[0];
cx q[6],q[18];
cx q[2],q[14];
cx q[16],q[0];
cx q[19],q[13];
cx q[23],q[12];
cx q[1],q[7];
cx q[18],q[6];
cx q[22],q[10];
cx q[4],q[9];
cx q[21],q[8];
cx q[15],q[9];
cx q[23],q[13];
cx q[10],q[22];
cx q[1],q[4];
cx q[21],q[14];
cx q[5],q[18];
cx q[12],q[19];
cx q[2],q[6];
cx q[0],q[7];
cx q[20],q[24];
cx q[11],q[8];
cx q[23],q[12];
cx q[14],q[18];
cx q[11],q[8];
cx q[6],q[2];
cx q[22],q[15];
cx q[16],q[4];
cx q[9],q[5];
cx q[0],q[7];
cx q[20],q[24];
cx q[6],q[2];
cx q[3],q[11];
cx q[16],q[0];
cx q[4],q[7];
cx q[9],q[1];
cx q[18],q[14];
cx q[7],q[0];
cx q[2],q[6];
cx q[23],q[13];
cx q[1],q[5];
cx q[12],q[19];
cx q[8],q[11];
cx q[15],q[10];
cx q[20],q[17];
cx q[17],q[20];
cx q[12],q[24];
cx q[1],q[4];
cx q[8],q[11];
cx q[7],q[18];
cx q[22],q[10];
cx q[3],q[2];
cx q[15],q[5];
cx q[15],q[17];
cx q[7],q[5];
cx q[11],q[3];
cx q[6],q[0];
cx q[14],q[2];
cx q[10],q[22];
cx q[13],q[23];
cx q[17],q[22];
cx q[7],q[5];
cx q[12],q[13];
cx q[16],q[1];
cx q[10],q[15];
cx q[20],q[19];
cx q[3],q[2];
cx q[9],q[24];
cx q[6],q[0];
cx q[21],q[14];
cx q[8],q[11];
cx q[21],q[14];
cx q[15],q[4];
cx q[1],q[16];
cx q[10],q[24];
cx q[22],q[17];
cx q[13],q[12];
cx q[19],q[20];
cx q[3],q[2];
cx q[4],q[15];
cx q[24],q[10];
cx q[19],q[12];
cx q[16],q[7];
cx q[18],q[21];
cx q[6],q[0];
cx q[13],q[23];
cx q[22],q[17];
cx q[11],q[14];
cx q[3],q[2];
cx q[18],q[21];
cx q[9],q[14];
cx q[8],q[11];
cx q[19],q[20];
cx q[5],q[7];
cx q[6],q[0];
cx q[4],q[16];
cx q[22],q[10];
cx q[5],q[4];
cx q[0],q[6];
cx q[8],q[11];
cx q[15],q[22];
cx q[24],q[18];
cx q[12],q[21];
cx q[10],q[17];
cx q[19],q[20];
cx q[13],q[23];
cx q[22],q[20];
cx q[12],q[24];
cx q[7],q[16];
cx q[2],q[6];
cx q[14],q[21];
cx q[17],q[19];
cx q[0],q[1];
cx q[18],q[9];
cx q[3],q[11];
cx q[24],q[18];
cx q[13],q[23];
cx q[22],q[20];
cx q[2],q[14];
cx q[15],q[4];
cx q[5],q[10];
cx q[7],q[16];
cx q[19],q[17];
cx q[13],q[21];
cx q[10],q[20];
cx q[7],q[9];
cx q[1],q[4];
cx q[11],q[3];
cx q[2],q[14];
cx q[12],q[8];
cx q[13],q[23];
cx q[3],q[11];
cx q[7],q[1];
cx q[24],q[17];
cx q[12],q[8];
cx q[19],q[21];
cx q[16],q[15];
cx q[18],q[9];
cx q[10],q[5];
cx q[2],q[14];
cx q[22],q[20];
cx q[23],q[21];
cx q[11],q[14];
cx q[7],q[16];
cx q[12],q[18];
cx q[1],q[0];
cx q[9],q[4];
cx q[19],q[13];
cx q[17],q[24];
cx q[20],q[10];
cx q[3],q[2];
cx q[2],q[3];
cx q[22],q[17];
cx q[9],q[4];
cx q[18],q[5];
cx q[12],q[8];
cx q[21],q[23];
cx q[11],q[14];
cx q[7],q[1];
cx q[20],q[15];
cx q[0],q[6];
cx q[24],q[10];
cx q[13],q[19];
cx q[21],q[12];
cx q[24],q[17];
cx q[13],q[19];
cx q[15],q[20];
cx q[10],q[22];
cx q[4],q[7];
cx q[18],q[5];
cx q[14],q[11];
cx q[8],q[23];
cx q[3],q[6];
cx q[0],q[1];
cx q[6],q[3];
cx q[21],q[24];
cx q[16],q[7];
cx q[1],q[0];
cx q[15],q[10];
cx q[4],q[5];
cx q[14],q[12];
cx q[22],q[20];
cx q[17],q[19];
cx q[9],q[18];
cx q[8],q[11];
cx q[22],q[19];
cx q[17],q[23];
cx q[3],q[2];
cx q[1],q[7];
cx q[4],q[9];
cx q[21],q[24];
cx q[12],q[18];
cx q[15],q[10];
cx q[11],q[8];
cx q[5],q[16];
cx q[6],q[0];
cx q[22],q[15];
cx q[6],q[9];
cx q[0],q[4];
cx q[2],q[11];
cx q[17],q[23];
cx q[16],q[20];
cx q[12],q[14];
cx q[1],q[7];
cx q[21],q[13];
cx q[5],q[18];
cx q[19],q[24];
cx q[13],q[17];
cx q[11],q[14];
cx q[5],q[4];
cx q[18],q[24];
cx q[22],q[15];
cx q[8],q[12];
cx q[1],q[7];
cx q[20],q[10];
cx q[3],q[6];
cx q[19],q[17];
cx q[7],q[16];
cx q[14],q[12];
cx q[23],q[13];
cx q[9],q[18];
cx q[24],q[10];
cx q[15],q[22];
cx q[6],q[0];
cx q[21],q[24];
cx q[0],q[4];
cx q[18],q[5];
cx q[3],q[2];
cx q[6],q[9];
cx q[23],q[8];
cx q[16],q[7];
cx q[20],q[10];
cx q[19],q[22];
cx q[14],q[11];
cx q[5],q[16];
cx q[10],q[22];
cx q[9],q[6];
cx q[12],q[18];
cx q[3],q[2];
cx q[7],q[4];
cx q[0],q[1];
cx q[17],q[19];
cx q[21],q[24];
cx q[23],q[13];
cx q[18],q[9];
cx q[12],q[8];
cx q[24],q[19];
cx q[14],q[11];
cx q[7],q[4];
cx q[13],q[23];
cx q[21],q[17];
cx q[10],q[22];
cx q[0],q[16];
cx q[20],q[15];
cx q[2],q[6];
cx q[22],q[15];
cx q[9],q[7];
cx q[21],q[18];
cx q[0],q[1];
cx q[16],q[5];
cx q[12],q[24];
cx q[2],q[6];
cx q[14],q[11];
cx q[20],q[10];
cx q[13],q[23];
cx q[17],q[19];
cx q[11],q[6];
cx q[15],q[22];
cx q[17],q[19];
cx q[24],q[23];
cx q[18],q[21];
cx q[8],q[12];
cx q[0],q[16];
cx q[4],q[1];
cx q[14],q[3];
cx q[5],q[10];
cx q[2],q[9];
cx q[17],q[13];
cx q[22],q[19];
cx q[7],q[11];
cx q[23],q[8];
cx q[21],q[18];
cx q[5],q[10];
cx q[15],q[20];
cx q[13],q[23];
cx q[5],q[16];
cx q[2],q[3];
cx q[19],q[21];
cx q[18],q[9];
cx q[10],q[20];
cx q[12],q[7];
cx q[15],q[22];
cx q[11],q[4];
cx q[16],q[5];
cx q[4],q[2];
cx q[9],q[14];
cx q[7],q[12];
cx q[15],q[22];
cx q[11],q[0];
cx q[13],q[23];
cx q[8],q[18];
cx q[20],q[15];
cx q[11],q[6];
cx q[21],q[10];
cx q[9],q[14];
cx q[5],q[16];
cx q[22],q[19];
cx q[4],q[3];
cx q[0],q[1];
cx q[17],q[13];
cx q[8],q[3];
cx q[6],q[1];
cx q[12],q[18];
cx q[2],q[4];
cx q[19],q[22];
cx q[10],q[20];
cx q[23],q[14];
cx q[0],q[11];
cx q[9],q[7];
cx q[8],q[9];
cx q[14],q[23];
cx q[15],q[22];
cx q[19],q[17];
cx q[12],q[18];
cx q[4],q[3];
cx q[1],q[6];
cx q[16],q[0];
cx q[21],q[24];
cx q[2],q[7];
cx q[0],q[11];
cx q[4],q[3];
cx q[20],q[15];
cx q[21],q[22];
cx q[6],q[2];
cx q[18],q[14];
cx q[12],q[5];
cx q[13],q[19];
cx q[17],q[24];
cx q[7],q[9];
cx q[6],q[0];
cx q[10],q[17];
cx q[2],q[4];
cx q[13],q[19];
cx q[22],q[21];
cx q[7],q[12];
cx q[11],q[1];
cx q[22],q[15];
cx q[10],q[17];
cx q[1],q[16];
cx q[21],q[19];
cx q[12],q[2];
cx q[11],q[6];
cx q[14],q[23];
cx q[18],q[9];
cx q[4],q[3];
cx q[24],q[23];
cx q[9],q[8];
cx q[13],q[19];
cx q[2],q[12];
cx q[11],q[0];
cx q[22],q[21];
cx q[10],q[17];
cx q[18],q[7];
cx q[4],q[3];
cx q[1],q[16];
cx q[11],q[5];
cx q[2],q[0];
cx q[3],q[4];
cx q[24],q[17];
cx q[14],q[8];
cx q[12],q[9];
cx q[7],q[17];
cx q[14],q[23];
cx q[6],q[11];
cx q[21],q[19];
cx q[16],q[1];
cx q[24],q[18];
cx q[10],q[5];
cx q[4],q[3];
cx q[9],q[8];
cx q[22],q[20];
cx q[12],q[2];
cx q[12],q[11];
cx q[15],q[22];
cx q[1],q[16];
cx q[13],q[19];
cx q[4],q[2];
cx q[7],q[5];
cx q[2],q[6];
cx q[11],q[12];
cx q[7],q[5];
cx q[1],q[0];
cx q[21],q[22];
cx q[24],q[23];
cx q[9],q[8];
cx q[10],q[20];
cx q[18],q[14];
cx q[20],q[15];
cx q[9],q[12];
cx q[23],q[19];
cx q[21],q[17];
cx q[4],q[2];
cx q[18],q[24];
cx q[6],q[11];
cx q[16],q[5];
cx q[3],q[8];
cx q[3],q[4];
cx q[2],q[11];
cx q[7],q[6];
cx q[15],q[20];
cx q[21],q[17];
cx q[24],q[13];
cx q[12],q[5];
cx q[8],q[9];
cx q[14],q[23];
cx q[9],q[12];
cx q[15],q[20];
cx q[2],q[11];
cx q[14],q[23];
cx q[5],q[6];
cx q[19],q[13];
cx q[7],q[18];
cx q[16],q[1];
cx q[3],q[4];
cx q[22],q[10];
cx q[24],q[17];
cx q[8],q[3];
cx q[15],q[22];
cx q[12],q[9];
cx q[6],q[5];
cx q[0],q[1];
cx q[24],q[13];
cx q[21],q[17];
cx q[23],q[19];
cx q[7],q[18];
cx q[4],q[3];
cx q[6],q[16];
cx q[17],q[7];
cx q[20],q[10];
cx q[2],q[12];
cx q[11],q[0];
cx q[23],q[19];
cx q[9],q[18];
cx q[1],q[5];
cx q[22],q[21];
cx q[13],q[24];
cx q[14],q[8];
cx q[8],q[3];
cx q[18],q[14];
cx q[4],q[9];
cx q[20],q[10];
cx q[6],q[15];
cx q[13],q[24];
cx q[23],q[19];
cx q[11],q[12];
cx q[20],q[10];
cx q[18],q[24];
cx q[22],q[21];
cx q[6],q[16];
cx q[5],q[11];
cx q[23],q[14];
cx q[13],q[19];
cx q[15],q[17];
cx q[9],q[8];
cx q[18],q[23];
cx q[4],q[8];
cx q[22],q[21];
cx q[1],q[0];
cx q[3],q[2];
cx q[16],q[10];
cx q[15],q[17];
cx q[19],q[13];
cx q[9],q[18];
cx q[14],q[8];
cx q[22],q[21];
cx q[7],q[2];
cx q[15],q[17];
cx q[24],q[23];
cx q[20],q[10];
cx q[16],q[6];
cx q[3],q[4];
cx q[1],q[5];
cx q[17],q[15];
cx q[9],q[3];
cx q[16],q[6];
cx q[12],q[18];
cx q[21],q[13];
cx q[8],q[4];
cx q[22],q[20];
cx q[21],q[22];
cx q[0],q[5];
cx q[13],q[23];
cx q[11],q[7];
cx q[15],q[10];
cx q[3],q[2];
cx q[17],q[12];
cx q[14],q[18];
cx q[24],q[19];
cx q[8],q[9];
cx q[1],q[2];
cx q[14],q[24];
cx q[13],q[19];
cx q[17],q[23];
cx q[15],q[10];
cx q[11],q[7];
cx q[4],q[3];
cx q[3],q[4];
cx q[23],q[24];
cx q[7],q[2];
cx q[17],q[12];
cx q[0],q[5];
cx q[6],q[15];
cx q[20],q[21];
cx q[9],q[8];
cx q[13],q[19];
cx q[5],q[0];
cx q[22],q[13];
cx q[17],q[12];
cx q[7],q[2];
cx q[8],q[18];
cx q[6],q[15];
cx q[21],q[20];
cx q[1],q[11];
cx q[24],q[23];
cx q[10],q[16];
cx q[9],q[4];
cx q[13],q[24];
cx q[5],q[0];
cx q[22],q[23];
cx q[14],q[18];
cx q[9],q[4];
cx q[6],q[12];
cx q[2],q[3];
cx q[18],q[8];
cx q[9],q[4];
cx q[12],q[6];
cx q[24],q[23];
cx q[16],q[20];
cx q[1],q[11];
cx q[10],q[5];
cx q[19],q[13];
cx q[2],q[3];
cx q[22],q[21];
cx q[0],q[5];
cx q[7],q[6];
cx q[4],q[9];
cx q[11],q[10];
cx q[12],q[17];
cx q[23],q[22];
cx q[1],q[2];
cx q[13],q[19];
cx q[8],q[18];
cx q[15],q[21];
cx q[15],q[21];
cx q[14],q[9];
cx q[11],q[6];
cx q[18],q[17];
cx q[22],q[23];
cx q[2],q[1];
cx q[3],q[4];
cx q[16],q[20];
cx q[19],q[24];
cx q[0],q[5];
cx q[13],q[8];
