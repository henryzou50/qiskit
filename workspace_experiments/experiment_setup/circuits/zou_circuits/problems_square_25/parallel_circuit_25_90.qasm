OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[0],q[18];
cx q[16],q[2];
cx q[6],q[11];
cx q[17],q[22];
cx q[19],q[13];
cx q[15],q[4];
cx q[12],q[23];
cx q[9],q[17];
cx q[19],q[21];
cx q[20],q[23];
cx q[5],q[0];
cx q[18],q[3];
cx q[15],q[16];
cx q[8],q[24];
cx q[6],q[11];
cx q[13],q[7];
cx q[16],q[10];
cx q[6],q[5];
cx q[1],q[0];
cx q[4],q[8];
cx q[9],q[19];
cx q[20],q[22];
cx q[14],q[13];
cx q[23],q[21];
cx q[3],q[15];
cx q[4],q[24];
cx q[13],q[9];
cx q[5],q[1];
cx q[11],q[21];
cx q[15],q[0];
cx q[10],q[2];
cx q[20],q[6];
cx q[8],q[10];
cx q[7],q[2];
cx q[15],q[0];
cx q[19],q[11];
cx q[1],q[21];
cx q[4],q[12];
cx q[17],q[19];
cx q[6],q[3];
cx q[15],q[18];
cx q[7],q[14];
cx q[1],q[20];
cx q[9],q[12];
cx q[0],q[22];
cx q[11],q[21];
cx q[2],q[8];
cx q[23],q[10];
cx q[24],q[16];
cx q[23],q[10];
cx q[5],q[13];
cx q[22],q[3];
cx q[16],q[8];
cx q[7],q[6];
cx q[11],q[1];
cx q[16],q[8];
cx q[0],q[15];
cx q[13],q[11];
cx q[12],q[14];
cx q[17],q[22];
cx q[23],q[9];
cx q[19],q[21];
cx q[2],q[7];
cx q[6],q[12];
cx q[15],q[22];
cx q[7],q[18];
cx q[14],q[4];
cx q[20],q[5];
cx q[1],q[21];
cx q[12],q[19];
cx q[6],q[14];
cx q[0],q[3];
cx q[2],q[8];
cx q[15],q[20];
cx q[10],q[22];
cx q[17],q[5];
cx q[18],q[7];
cx q[23],q[9];
cx q[13],q[11];
cx q[16],q[8];
cx q[15],q[22];
cx q[6],q[2];
cx q[11],q[17];
cx q[7],q[20];
cx q[21],q[1];
cx q[3],q[0];
cx q[10],q[18];
cx q[13],q[5];
cx q[19],q[12];
cx q[4],q[24];
cx q[23],q[14];
cx q[22],q[0];
cx q[11],q[12];
cx q[5],q[17];
cx q[21],q[15];
cx q[19],q[14];
cx q[8],q[6];
cx q[13],q[18];
cx q[2],q[19];
cx q[1],q[23];
cx q[20],q[3];
cx q[11],q[22];
cx q[21],q[15];
cx q[16],q[6];
cx q[4],q[24];
cx q[5],q[12];
cx q[0],q[7];
cx q[7],q[17];
cx q[12],q[11];
cx q[21],q[22];
cx q[18],q[13];
cx q[8],q[19];
cx q[15],q[20];
cx q[9],q[1];
cx q[10],q[16];
cx q[0],q[3];
cx q[6],q[14];
cx q[2],q[23];
cx q[20],q[11];
cx q[13],q[2];
cx q[21],q[22];
cx q[18],q[19];
cx q[3],q[10];
cx q[1],q[9];
cx q[10],q[16];
cx q[0],q[7];
cx q[6],q[14];
cx q[20],q[11];
cx q[12],q[21];
cx q[13],q[2];
cx q[15],q[17];
cx q[23],q[1];
cx q[9],q[19];
cx q[24],q[22];
cx q[8],q[14];
cx q[7],q[22];
cx q[0],q[10];
cx q[5],q[24];
cx q[21],q[2];
cx q[4],q[16];
cx q[18],q[6];
cx q[13],q[23];
cx q[8],q[24];
cx q[10],q[7];
cx q[9],q[1];
cx q[14],q[18];
cx q[11],q[21];
cx q[4],q[16];
cx q[5],q[12];
cx q[6],q[0];
cx q[3],q[22];
cx q[19],q[17];
cx q[21],q[20];
cx q[17],q[3];
cx q[5],q[6];
cx q[8],q[4];
cx q[19],q[11];
cx q[2],q[1];
cx q[23],q[13];
cx q[16],q[24];
cx q[14],q[18];
cx q[15],q[22];
cx q[10],q[7];
cx q[14],q[1];
cx q[18],q[16];
cx q[9],q[13];
cx q[20],q[19];
cx q[10],q[7];
cx q[17],q[15];
cx q[23],q[2];
cx q[5],q[6];
cx q[11],q[21];
cx q[6],q[16];
cx q[21],q[2];
cx q[17],q[11];
cx q[20],q[23];
cx q[22],q[0];
cx q[7],q[3];
cx q[8],q[24];
cx q[14],q[22];
cx q[3],q[17];
cx q[2],q[21];
cx q[12],q[20];
cx q[11],q[7];
cx q[0],q[15];
cx q[13],q[1];
cx q[24],q[8];
cx q[23],q[3];
cx q[17],q[0];
cx q[20],q[2];
cx q[7],q[10];
cx q[6],q[5];
cx q[22],q[14];
cx q[15],q[11];
cx q[9],q[12];
cx q[1],q[21];
cx q[4],q[8];
cx q[24],q[18];
cx q[4],q[24];
cx q[21],q[5];
cx q[11],q[17];
cx q[2],q[19];
cx q[16],q[3];
cx q[22],q[18];
cx q[14],q[15];
cx q[0],q[8];
cx q[23],q[7];
cx q[12],q[20];
cx q[13],q[9];
cx q[11],q[15];
cx q[17],q[10];
cx q[24],q[22];
cx q[13],q[5];
cx q[21],q[9];
cx q[2],q[16];
cx q[14],q[18];
cx q[4],q[3];
cx q[7],q[1];
cx q[6],q[12];
cx q[21],q[9];
cx q[14],q[6];
cx q[7],q[19];
cx q[16],q[4];
cx q[5],q[22];
cx q[20],q[23];
cx q[13],q[18];
cx q[21],q[18];
cx q[11],q[20];
cx q[10],q[15];
cx q[24],q[17];
cx q[8],q[22];
cx q[5],q[13];
cx q[9],q[23];
cx q[3],q[4];
cx q[17],q[0];
cx q[20],q[12];
cx q[6],q[8];
cx q[4],q[11];
cx q[9],q[1];
cx q[21],q[2];
cx q[22],q[18];
cx q[3],q[16];
cx q[5],q[24];
cx q[15],q[10];
cx q[13],q[23];
cx q[7],q[19];
cx q[23],q[21];
cx q[3],q[18];
cx q[11],q[0];
cx q[20],q[1];
cx q[5],q[8];
cx q[9],q[2];
cx q[19],q[7];
cx q[4],q[16];
cx q[22],q[6];
cx q[15],q[10];
cx q[9],q[1];
cx q[18],q[13];
cx q[5],q[3];
cx q[16],q[22];
cx q[2],q[20];
cx q[14],q[7];
cx q[4],q[6];
cx q[15],q[10];
cx q[8],q[17];
cx q[0],q[11];
cx q[12],q[20];
cx q[4],q[16];
cx q[15],q[0];
cx q[24],q[8];
cx q[19],q[1];
cx q[21],q[9];
cx q[22],q[18];
cx q[7],q[11];
cx q[5],q[3];
cx q[17],q[14];
cx q[2],q[23];
cx q[10],q[6];
cx q[3],q[8];
cx q[5],q[4];
cx q[11],q[12];
cx q[10],q[18];
cx q[17],q[16];
cx q[9],q[19];
cx q[15],q[17];
cx q[23],q[20];
cx q[2],q[13];
cx q[0],q[6];
cx q[11],q[19];
cx q[3],q[24];
cx q[8],q[16];
cx q[22],q[4];
cx q[21],q[24];
cx q[10],q[11];
cx q[14],q[17];
cx q[20],q[13];
cx q[9],q[23];
cx q[22],q[5];
cx q[13],q[20];
cx q[11],q[12];
cx q[24],q[21];
cx q[7],q[15];
cx q[14],q[8];
cx q[4],q[16];
cx q[3],q[18];
cx q[23],q[0];
cx q[19],q[9];
cx q[8],q[0];
cx q[19],q[10];
cx q[1],q[21];
cx q[11],q[13];
cx q[2],q[20];
cx q[12],q[7];
cx q[15],q[3];
cx q[17],q[5];
cx q[18],q[22];
cx q[2],q[17];
cx q[7],q[18];
cx q[0],q[10];
cx q[23],q[9];
cx q[3],q[5];
cx q[16],q[14];
cx q[4],q[15];
cx q[13],q[21];
cx q[5],q[18];
cx q[9],q[10];
cx q[23],q[19];
cx q[1],q[11];
cx q[21],q[22];
cx q[0],q[6];
cx q[16],q[12];
cx q[11],q[9];
cx q[14],q[18];
cx q[4],q[7];
cx q[21],q[23];
cx q[17],q[24];
cx q[10],q[6];
cx q[20],q[13];
cx q[0],q[3];
cx q[19],q[8];
cx q[15],q[5];
cx q[1],q[22];
cx q[23],q[22];
cx q[2],q[7];
cx q[1],q[11];
cx q[13],q[9];
cx q[17],q[14];
cx q[0],q[12];
cx q[5],q[24];
cx q[5],q[2];
cx q[0],q[3];
cx q[6],q[13];
cx q[16],q[8];
cx q[1],q[21];
cx q[11],q[22];
cx q[9],q[18];
cx q[12],q[10];
cx q[17],q[15];
cx q[23],q[20];
cx q[23],q[19];
cx q[11],q[22];
cx q[3],q[24];
cx q[18],q[16];
cx q[6],q[8];
cx q[1],q[20];
cx q[12],q[7];
cx q[0],q[10];
cx q[2],q[9];
cx q[21],q[13];
cx q[15],q[17];
cx q[4],q[14];
cx q[17],q[2];
cx q[3],q[24];
cx q[9],q[15];
cx q[6],q[18];
cx q[8],q[16];
cx q[1],q[11];
cx q[20],q[23];
cx q[17],q[14];
cx q[19],q[24];
cx q[5],q[2];
cx q[7],q[11];
cx q[3],q[0];
cx q[21],q[22];
cx q[10],q[6];
cx q[0],q[3];
cx q[6],q[10];
cx q[20],q[16];
cx q[15],q[4];
cx q[7],q[17];
cx q[24],q[18];
cx q[14],q[5];
cx q[9],q[12];
cx q[23],q[11];
cx q[19],q[8];
cx q[22],q[1];
cx q[19],q[20];
cx q[11],q[23];
cx q[6],q[0];
cx q[10],q[3];
cx q[5],q[4];
cx q[16],q[18];
cx q[9],q[24];
cx q[8],q[17];
cx q[21],q[13];
cx q[12],q[14];
cx q[12],q[10];
cx q[21],q[11];
cx q[0],q[4];
cx q[22],q[2];
cx q[17],q[15];
cx q[5],q[14];
cx q[14],q[22];
cx q[21],q[4];
cx q[24],q[0];
cx q[1],q[11];
cx q[16],q[9];
cx q[17],q[6];
cx q[13],q[7];
cx q[23],q[2];
cx q[24],q[8];
cx q[14],q[22];
cx q[9],q[11];
cx q[7],q[19];
cx q[10],q[3];
cx q[16],q[1];
cx q[5],q[7];
cx q[4],q[22];
cx q[18],q[6];
cx q[11],q[9];
cx q[15],q[19];
cx q[16],q[8];
cx q[2],q[14];
cx q[17],q[24];
cx q[23],q[1];
cx q[2],q[13];
cx q[19],q[1];
cx q[0],q[20];
cx q[23],q[5];
cx q[7],q[21];
cx q[4],q[10];
cx q[16],q[8];
cx q[14],q[12];
cx q[6],q[18];
cx q[15],q[3];
cx q[5],q[16];
cx q[24],q[4];
cx q[18],q[1];
cx q[3],q[20];
cx q[11],q[9];
cx q[10],q[14];
cx q[2],q[16];
cx q[20],q[3];
cx q[24],q[15];
cx q[22],q[14];
cx q[0],q[4];
cx q[11],q[8];
cx q[13],q[12];
cx q[10],q[21];
cx q[1],q[9];
cx q[18],q[6];
cx q[9],q[18];
cx q[1],q[17];
cx q[22],q[14];
cx q[3],q[6];
cx q[7],q[13];
cx q[23],q[2];
cx q[10],q[15];
cx q[19],q[5];
cx q[8],q[11];
cx q[21],q[24];
cx q[14],q[24];
cx q[4],q[21];
cx q[3],q[10];
cx q[11],q[0];
cx q[15],q[8];
cx q[5],q[18];
cx q[6],q[20];
cx q[2],q[7];
cx q[7],q[14];
cx q[12],q[5];
cx q[0],q[8];
cx q[19],q[6];
cx q[15],q[13];
cx q[22],q[2];
cx q[0],q[11];
cx q[23],q[12];
cx q[10],q[15];
cx q[18],q[1];
cx q[2],q[9];
cx q[3],q[13];
cx q[14],q[16];
cx q[7],q[21];
cx q[1],q[3];
cx q[15],q[7];
cx q[23],q[2];
cx q[21],q[16];
cx q[11],q[20];
cx q[13],q[22];
cx q[4],q[24];
cx q[12],q[8];
cx q[22],q[3];
cx q[10],q[1];
cx q[19],q[18];
cx q[23],q[12];
cx q[15],q[7];
cx q[11],q[24];
cx q[14],q[21];
cx q[6],q[17];
cx q[13],q[4];
cx q[9],q[2];
cx q[16],q[8];
cx q[7],q[16];
cx q[10],q[23];
cx q[0],q[18];
cx q[4],q[8];
cx q[20],q[22];
cx q[24],q[11];
cx q[12],q[2];
cx q[4],q[8];
cx q[16],q[7];
cx q[20],q[13];
cx q[5],q[17];
cx q[10],q[1];
cx q[22],q[18];
cx q[9],q[21];
cx q[23],q[12];
cx q[0],q[6];
cx q[3],q[24];
cx q[14],q[2];
cx q[5],q[2];
cx q[10],q[1];
cx q[24],q[22];
cx q[11],q[13];
cx q[0],q[19];
cx q[20],q[21];
cx q[6],q[17];
cx q[6],q[3];
cx q[0],q[2];
cx q[13],q[24];
cx q[17],q[23];
cx q[14],q[12];
cx q[20],q[8];
cx q[22],q[11];
cx q[13],q[11];
cx q[22],q[18];
cx q[21],q[2];
cx q[1],q[20];
cx q[16],q[8];
cx q[6],q[19];
cx q[24],q[3];
cx q[15],q[10];
cx q[5],q[9];
cx q[14],q[17];
cx q[7],q[0];
cx q[9],q[5];
cx q[21],q[10];
cx q[17],q[16];
cx q[15],q[8];
cx q[23],q[6];
cx q[24],q[0];
cx q[4],q[19];
cx q[15],q[20];
cx q[1],q[18];
cx q[23],q[24];
cx q[6],q[5];
cx q[3],q[7];
cx q[0],q[22];
cx q[10],q[8];
cx q[14],q[2];
cx q[9],q[4];
cx q[11],q[13];
cx q[23],q[9];
cx q[4],q[16];
cx q[18],q[1];
cx q[17],q[8];
cx q[24],q[7];
cx q[19],q[0];
cx q[10],q[3];
cx q[2],q[21];
cx q[20],q[15];
cx q[6],q[22];
cx q[11],q[10];
cx q[1],q[13];
cx q[6],q[23];
cx q[19],q[4];
cx q[2],q[0];
cx q[17],q[16];
cx q[15],q[3];
cx q[14],q[21];
cx q[12],q[20];
cx q[18],q[7];
cx q[15],q[11];
cx q[8],q[21];
cx q[0],q[16];
cx q[6],q[23];
cx q[18],q[10];
cx q[2],q[9];
cx q[24],q[13];
cx q[7],q[18];
cx q[13],q[10];
cx q[11],q[8];
cx q[17],q[21];
cx q[19],q[14];
cx q[5],q[22];
cx q[4],q[16];
cx q[8],q[20];
cx q[16],q[4];
cx q[17],q[21];
cx q[15],q[18];
cx q[6],q[23];
cx q[24],q[13];
cx q[3],q[12];
cx q[22],q[19];
cx q[21],q[16];
cx q[6],q[14];
cx q[1],q[18];
cx q[12],q[3];
cx q[11],q[0];
cx q[4],q[17];
cx q[23],q[19];
cx q[10],q[15];
cx q[13],q[7];
cx q[3],q[20];
cx q[21],q[11];
cx q[4],q[12];
cx q[18],q[2];
cx q[14],q[5];
cx q[10],q[24];
cx q[22],q[15];
cx q[8],q[0];
cx q[1],q[7];
cx q[19],q[9];
cx q[22],q[1];
cx q[0],q[18];
cx q[3],q[20];
cx q[24],q[10];
cx q[19],q[17];
cx q[6],q[15];
cx q[2],q[21];
cx q[11],q[16];
cx q[9],q[14];
cx q[18],q[2];
cx q[17],q[16];
cx q[0],q[3];
cx q[13],q[14];
cx q[9],q[19];
cx q[6],q[15];
cx q[20],q[11];
cx q[22],q[7];
cx q[23],q[14];
cx q[3],q[8];
cx q[9],q[21];
cx q[15],q[22];
cx q[10],q[1];
cx q[11],q[18];
cx q[7],q[2];
cx q[16],q[12];
cx q[18],q[6];
cx q[1],q[7];
cx q[22],q[20];
cx q[14],q[10];
cx q[5],q[13];
cx q[2],q[11];
cx q[15],q[23];
cx q[3],q[8];
cx q[18],q[10];
cx q[21],q[4];
cx q[22],q[1];
cx q[15],q[17];
cx q[2],q[0];
cx q[20],q[7];
cx q[16],q[8];
cx q[9],q[11];
cx q[13],q[17];
cx q[18],q[24];
cx q[15],q[11];
cx q[2],q[1];
cx q[8],q[10];
cx q[4],q[12];
cx q[5],q[22];
cx q[7],q[13];
cx q[23],q[24];
cx q[18],q[21];
cx q[17],q[15];
cx q[1],q[3];
cx q[20],q[10];
cx q[2],q[11];
cx q[9],q[8];
cx q[19],q[14];
cx q[4],q[12];
cx q[5],q[11];
cx q[24],q[19];
cx q[15],q[13];
cx q[18],q[23];
cx q[16],q[0];
cx q[1],q[3];
cx q[24],q[21];
cx q[11],q[5];
cx q[12],q[9];
cx q[4],q[23];
cx q[17],q[19];
cx q[10],q[16];
cx q[20],q[0];
cx q[2],q[6];
cx q[7],q[13];
cx q[13],q[24];
cx q[18],q[14];
cx q[21],q[19];
cx q[12],q[8];
cx q[23],q[3];
cx q[6],q[2];
cx q[0],q[16];
cx q[9],q[4];
cx q[7],q[1];
cx q[0],q[2];
cx q[12],q[8];
cx q[6],q[16];
cx q[10],q[20];
cx q[15],q[11];
cx q[21],q[23];
cx q[7],q[24];
cx q[13],q[18];
cx q[3],q[9];
cx q[13],q[19];
cx q[2],q[16];
cx q[8],q[17];
cx q[12],q[4];
cx q[20],q[10];
cx q[23],q[21];
cx q[1],q[3];
cx q[7],q[0];
cx q[24],q[15];
cx q[21],q[7];
cx q[24],q[18];
cx q[11],q[10];
cx q[6],q[16];
cx q[12],q[19];
cx q[20],q[22];
cx q[9],q[4];
cx q[18],q[14];
cx q[5],q[15];
cx q[17],q[13];
cx q[10],q[20];
cx q[6],q[16];
cx q[22],q[11];
cx q[12],q[9];
cx q[22],q[7];
cx q[2],q[1];
cx q[8],q[17];
cx q[15],q[11];
cx q[19],q[21];
cx q[23],q[13];
cx q[14],q[24];
cx q[5],q[10];
cx q[3],q[4];
cx q[0],q[20];
cx q[24],q[14];
cx q[22],q[15];
cx q[4],q[13];
cx q[10],q[20];
cx q[11],q[7];
cx q[3],q[2];
cx q[6],q[0];
cx q[17],q[12];
cx q[8],q[23];
cx q[14],q[7];
cx q[15],q[22];
cx q[17],q[11];
cx q[5],q[16];
cx q[12],q[3];
cx q[8],q[9];
cx q[13],q[4];
