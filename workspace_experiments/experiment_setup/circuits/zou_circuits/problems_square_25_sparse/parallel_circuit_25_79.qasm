OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[18],q[10];
cx q[3],q[14];
cx q[13],q[20];
cx q[24],q[6];
cx q[4],q[12];
cx q[0],q[2];
cx q[16],q[15];
cx q[21],q[5];
cx q[9],q[1];
cx q[17],q[23];
cx q[5],q[20];
cx q[3],q[18];
cx q[16],q[11];
cx q[22],q[14];
cx q[12],q[8];
cx q[21],q[13];
cx q[23],q[17];
cx q[7],q[9];
cx q[24],q[4];
cx q[1],q[8];
cx q[3],q[6];
cx q[10],q[18];
cx q[4],q[12];
cx q[22],q[21];
cx q[0],q[2];
cx q[11],q[20];
cx q[17],q[13];
cx q[19],q[24];
cx q[20],q[13];
cx q[17],q[23];
cx q[22],q[19];
cx q[14],q[24];
cx q[10],q[8];
cx q[5],q[21];
cx q[0],q[18];
cx q[7],q[9];
cx q[1],q[2];
cx q[11],q[16];
cx q[12],q[4];
cx q[2],q[10];
cx q[15],q[16];
cx q[20],q[11];
cx q[6],q[8];
cx q[0],q[18];
cx q[21],q[5];
cx q[17],q[22];
cx q[1],q[7];
cx q[9],q[4];
cx q[14],q[12];
cx q[19],q[24];
cx q[20],q[13];
cx q[21],q[5];
cx q[10],q[18];
cx q[16],q[11];
cx q[7],q[1];
cx q[8],q[6];
cx q[15],q[3];
cx q[19],q[23];
cx q[0],q[2];
cx q[22],q[17];
cx q[11],q[21];
cx q[8],q[2];
cx q[9],q[7];
cx q[10],q[3];
cx q[20],q[16];
cx q[22],q[17];
cx q[24],q[4];
cx q[1],q[0];
cx q[5],q[13];
cx q[18],q[15];
cx q[14],q[6];
cx q[20],q[21];
cx q[24],q[14];
cx q[11],q[18];
cx q[5],q[22];
cx q[2],q[7];
cx q[17],q[23];
cx q[4],q[9];
cx q[1],q[10];
cx q[6],q[3];
cx q[7],q[12];
cx q[4],q[9];
cx q[11],q[18];
cx q[8],q[2];
cx q[15],q[0];
cx q[22],q[5];
cx q[14],q[6];
cx q[7],q[9];
cx q[20],q[16];
cx q[5],q[22];
cx q[3],q[1];
cx q[15],q[0];
cx q[17],q[13];
cx q[21],q[11];
cx q[2],q[8];
cx q[23],q[19];
cx q[7],q[4];
cx q[12],q[9];
cx q[0],q[15];
cx q[6],q[3];
cx q[1],q[10];
cx q[14],q[24];
cx q[3],q[1];
cx q[4],q[9];
cx q[5],q[19];
cx q[2],q[10];
cx q[15],q[18];
cx q[24],q[12];
cx q[17],q[23];
cx q[16],q[11];
cx q[0],q[1];
cx q[4],q[9];
cx q[5],q[17];
cx q[12],q[14];
cx q[18],q[3];
cx q[7],q[2];
cx q[10],q[15];
cx q[2],q[7];
cx q[20],q[21];
cx q[6],q[8];
cx q[1],q[18];
cx q[9],q[24];
cx q[23],q[17];
cx q[5],q[19];
cx q[3],q[15];
cx q[24],q[9];
cx q[18],q[11];
cx q[23],q[5];
cx q[7],q[4];
cx q[22],q[6];
cx q[19],q[14];
cx q[20],q[21];
cx q[0],q[10];
cx q[10],q[2];
cx q[20],q[21];
cx q[9],q[24];
cx q[16],q[11];
cx q[23],q[5];
cx q[19],q[6];
cx q[13],q[22];
cx q[4],q[7];
cx q[12],q[17];
cx q[14],q[8];
cx q[18],q[3];
cx q[6],q[3];
cx q[1],q[10];
cx q[7],q[2];
cx q[11],q[18];
cx q[20],q[16];
cx q[8],q[15];
cx q[17],q[13];
cx q[19],q[12];
cx q[11],q[18];
cx q[13],q[22];
cx q[24],q[5];
cx q[1],q[0];
cx q[16],q[20];
cx q[15],q[3];
cx q[2],q[10];
cx q[19],q[17];
cx q[7],q[12];
cx q[14],q[6];
cx q[4],q[9];
cx q[5],q[23];
cx q[8],q[2];
cx q[0],q[18];
cx q[16],q[11];
cx q[20],q[21];
cx q[4],q[24];
cx q[13],q[22];
cx q[15],q[3];
cx q[19],q[17];
cx q[9],q[7];
cx q[14],q[12];
cx q[18],q[0];
cx q[5],q[23];
cx q[21],q[3];
cx q[9],q[7];
cx q[19],q[13];
cx q[16],q[11];
cx q[12],q[4];
cx q[5],q[23];
cx q[0],q[1];
cx q[16],q[20];
cx q[22],q[21];
cx q[17],q[19];
cx q[3],q[8];
cx q[11],q[18];
cx q[7],q[9];
cx q[10],q[2];
cx q[15],q[6];
cx q[24],q[4];
cx q[8],q[15];
cx q[21],q[11];
cx q[22],q[17];
cx q[10],q[7];
cx q[2],q[12];
cx q[4],q[9];
cx q[19],q[23];
cx q[5],q[24];
cx q[1],q[0];
cx q[21],q[22];
cx q[13],q[20];
cx q[3],q[14];
cx q[4],q[24];
cx q[11],q[16];
cx q[2],q[7];
cx q[10],q[15];
cx q[23],q[5];
cx q[12],q[8];
cx q[22],q[21];
cx q[19],q[13];
cx q[7],q[2];
cx q[20],q[16];
cx q[17],q[23];
cx q[14],q[5];
cx q[12],q[4];
cx q[10],q[15];
cx q[12],q[14];
cx q[3],q[8];
cx q[18],q[0];
cx q[5],q[22];
cx q[17],q[4];
cx q[24],q[9];
cx q[16],q[11];
cx q[0],q[1];
cx q[3],q[6];
cx q[8],q[10];
cx q[13],q[22];
cx q[4],q[14];
cx q[19],q[23];
cx q[22],q[20];
cx q[15],q[10];
cx q[0],q[1];
cx q[12],q[8];
cx q[11],q[18];
cx q[2],q[9];
cx q[14],q[4];
cx q[17],q[19];
cx q[0],q[10];
cx q[20],q[16];
cx q[11],q[18];
cx q[8],q[3];
cx q[13],q[21];
cx q[4],q[9];
cx q[2],q[7];
cx q[19],q[17];
cx q[15],q[1];
cx q[14],q[24];
cx q[5],q[17];
cx q[3],q[14];
cx q[4],q[12];
cx q[18],q[6];
cx q[8],q[10];
cx q[19],q[22];
cx q[1],q[0];
cx q[19],q[23];
cx q[6],q[10];
cx q[7],q[12];
cx q[14],q[5];
cx q[8],q[2];
cx q[22],q[20];
cx q[3],q[13];
cx q[24],q[17];
cx q[18],q[11];
cx q[9],q[4];
cx q[9],q[4];
cx q[22],q[20];
cx q[0],q[1];
cx q[6],q[18];
cx q[16],q[11];
cx q[14],q[12];
cx q[3],q[13];
cx q[23],q[5];
cx q[1],q[0];
cx q[19],q[23];
cx q[10],q[15];
cx q[17],q[4];
cx q[20],q[16];
cx q[11],q[18];
cx q[2],q[8];
cx q[14],q[12];
cx q[13],q[3];
cx q[18],q[6];
cx q[8],q[7];
cx q[3],q[13];
cx q[1],q[0];
cx q[4],q[17];
cx q[2],q[12];
cx q[15],q[10];
cx q[19],q[23];
cx q[5],q[13];
cx q[1],q[0];
cx q[21],q[18];
cx q[20],q[16];
cx q[9],q[4];
cx q[12],q[14];
cx q[7],q[8];
cx q[12],q[4];
cx q[14],q[17];
cx q[16],q[21];
cx q[22],q[20];
cx q[18],q[11];
cx q[23],q[19];
cx q[4],q[12];
cx q[5],q[24];
cx q[21],q[11];
cx q[13],q[3];
cx q[17],q[14];
cx q[22],q[20];
cx q[7],q[10];
cx q[1],q[0];
cx q[8],q[2];
cx q[18],q[6];
cx q[14],q[17];
cx q[12],q[7];
cx q[20],q[18];
cx q[2],q[8];
cx q[22],q[19];
cx q[6],q[0];
cx q[10],q[15];
cx q[20],q[21];
cx q[4],q[9];
cx q[22],q[13];
cx q[24],q[5];
cx q[1],q[15];
cx q[17],q[14];
cx q[7],q[8];
cx q[11],q[16];
cx q[12],q[2];
cx q[3],q[10];
cx q[0],q[6];
cx q[23],q[19];
cx q[4],q[14];
cx q[11],q[20];
cx q[24],q[23];
cx q[18],q[13];
cx q[2],q[8];
cx q[22],q[19];
cx q[5],q[17];
cx q[6],q[0];
cx q[15],q[1];
cx q[16],q[21];
cx q[12],q[7];
cx q[3],q[0];
cx q[17],q[18];
cx q[5],q[13];
cx q[22],q[11];
cx q[8],q[1];
cx q[6],q[15];
cx q[12],q[7];
cx q[10],q[3];
cx q[22],q[11];
cx q[0],q[6];
cx q[13],q[5];
cx q[18],q[7];
cx q[9],q[2];
cx q[17],q[12];
cx q[23],q[19];
cx q[6],q[15];
cx q[22],q[11];
cx q[2],q[12];
cx q[5],q[17];
cx q[20],q[13];
cx q[24],q[14];
cx q[23],q[19];
cx q[16],q[0];
cx q[1],q[8];
cx q[7],q[10];
cx q[18],q[3];
cx q[4],q[9];
cx q[15],q[6];
cx q[5],q[13];
cx q[7],q[1];
cx q[20],q[11];
cx q[0],q[16];
cx q[9],q[4];
cx q[9],q[4];
cx q[11],q[21];
cx q[3],q[10];
cx q[24],q[23];
cx q[15],q[6];
cx q[2],q[12];
cx q[15],q[6];
cx q[21],q[11];
cx q[23],q[19];
cx q[18],q[17];
cx q[7],q[3];
cx q[2],q[1];
cx q[16],q[0];
cx q[5],q[24];
cx q[10],q[20];
cx q[19],q[5];
cx q[8],q[6];
cx q[18],q[14];
cx q[12],q[4];
cx q[23],q[24];
cx q[15],q[3];
cx q[21],q[22];
cx q[13],q[17];
cx q[9],q[2];
cx q[0],q[16];
cx q[1],q[7];
cx q[23],q[19];
cx q[17],q[10];
cx q[20],q[16];
cx q[5],q[13];
cx q[6],q[0];
cx q[7],q[3];
cx q[1],q[8];
cx q[21],q[11];
cx q[4],q[9];
cx q[6],q[0];
cx q[5],q[18];
cx q[10],q[15];
cx q[20],q[11];
cx q[13],q[16];
cx q[4],q[14];
cx q[21],q[22];
cx q[24],q[14];
cx q[17],q[13];
cx q[12],q[18];
cx q[7],q[1];
cx q[15],q[0];
cx q[9],q[4];
cx q[11],q[20];
cx q[3],q[6];
cx q[3],q[8];
cx q[7],q[18];
cx q[21],q[22];
cx q[4],q[9];
cx q[5],q[19];
cx q[13],q[17];
cx q[23],q[24];
cx q[20],q[11];
cx q[15],q[10];
cx q[12],q[1];
cx q[6],q[0];
cx q[21],q[22];
cx q[15],q[10];
cx q[23],q[24];
cx q[13],q[17];
cx q[6],q[3];
cx q[0],q[8];
cx q[9],q[1];
cx q[20],q[15];
cx q[13],q[16];
cx q[4],q[2];
cx q[5],q[24];
cx q[19],q[22];
cx q[9],q[7];
cx q[6],q[0];
cx q[18],q[14];
cx q[10],q[17];
cx q[12],q[8];
cx q[11],q[21];
cx q[1],q[3];
cx q[19],q[22];
cx q[2],q[9];
cx q[11],q[21];
cx q[0],q[6];
cx q[10],q[17];
cx q[7],q[12];
cx q[20],q[15];
cx q[5],q[13];
cx q[14],q[24];
cx q[20],q[11];
cx q[7],q[9];
cx q[4],q[2];
cx q[13],q[17];
cx q[6],q[15];
cx q[14],q[18];
cx q[10],q[16];
cx q[16],q[21];
cx q[2],q[9];
cx q[4],q[7];
cx q[10],q[17];
cx q[12],q[8];
cx q[19],q[23];
cx q[1],q[3];
cx q[13],q[22];
cx q[24],q[18];
cx q[11],q[15];
cx q[16],q[11];
cx q[9],q[2];
cx q[17],q[14];
cx q[13],q[24];
cx q[5],q[22];
cx q[15],q[10];
cx q[0],q[1];
cx q[9],q[2];
cx q[15],q[11];
cx q[7],q[4];
cx q[19],q[23];
cx q[17],q[10];
cx q[1],q[8];
cx q[14],q[18];
cx q[6],q[0];
cx q[12],q[3];
cx q[22],q[5];
cx q[16],q[21];
cx q[24],q[13];
cx q[9],q[2];
cx q[12],q[3];
cx q[11],q[10];
cx q[14],q[17];
cx q[24],q[23];
cx q[15],q[16];
cx q[19],q[22];
cx q[0],q[8];
cx q[4],q[7];
cx q[1],q[6];
cx q[5],q[13];
cx q[21],q[20];
cx q[23],q[19];
cx q[21],q[20];
cx q[13],q[14];
cx q[8],q[12];
cx q[11],q[10];
cx q[24],q[18];
cx q[1],q[3];
cx q[6],q[0];
cx q[7],q[4];
cx q[13],q[14];
cx q[8],q[3];
cx q[16],q[10];
cx q[11],q[15];
cx q[9],q[2];
cx q[22],q[19];
cx q[20],q[21];
cx q[17],q[5];
cx q[12],q[1];
cx q[0],q[6];
cx q[24],q[18];
cx q[3],q[8];
cx q[5],q[22];
cx q[12],q[1];
cx q[19],q[23];
cx q[10],q[11];
cx q[15],q[20];
cx q[14],q[13];
cx q[2],q[4];
cx q[16],q[21];
cx q[6],q[0];
cx q[7],q[9];
cx q[20],q[15];
cx q[21],q[17];
cx q[1],q[12];
cx q[0],q[10];
cx q[14],q[18];
cx q[5],q[22];
cx q[23],q[19];
cx q[16],q[11];
cx q[9],q[2];
cx q[8],q[7];
cx q[24],q[13];
cx q[17],q[21];
cx q[23],q[19];
cx q[6],q[10];
cx q[18],q[14];
cx q[2],q[9];
cx q[22],q[5];
cx q[11],q[5];
cx q[21],q[17];
cx q[23],q[13];
cx q[24],q[19];
cx q[8],q[12];
cx q[9],q[3];
cx q[5],q[17];
cx q[6],q[12];
cx q[8],q[7];
cx q[20],q[15];
cx q[22],q[19];
cx q[23],q[13];
cx q[11],q[16];
cx q[1],q[3];
cx q[14],q[18];
cx q[2],q[4];
cx q[1],q[0];
cx q[12],q[8];
cx q[22],q[13];
cx q[17],q[19];
cx q[16],q[15];
cx q[6],q[10];
cx q[7],q[9];
cx q[15],q[20];
cx q[14],q[23];
cx q[7],q[18];
cx q[10],q[6];
cx q[11],q[5];
cx q[16],q[17];
cx q[8],q[11];
cx q[24],q[23];
cx q[4],q[7];
cx q[18],q[14];
cx q[17],q[19];
cx q[12],q[1];
cx q[10],q[5];
cx q[16],q[15];
cx q[22],q[13];
cx q[3],q[9];
cx q[0],q[6];
cx q[1],q[3];
cx q[23],q[14];
cx q[17],q[16];
cx q[8],q[12];
cx q[6],q[0];
cx q[22],q[17];
cx q[4],q[14];
cx q[23],q[19];
cx q[21],q[20];
cx q[16],q[5];
cx q[0],q[6];
cx q[24],q[13];
cx q[1],q[3];
cx q[10],q[15];
cx q[9],q[7];
cx q[14],q[24];
cx q[12],q[7];
cx q[15],q[10];
cx q[5],q[11];
cx q[17],q[16];
cx q[8],q[18];
cx q[19],q[23];
cx q[6],q[0];
cx q[3],q[1];
cx q[22],q[21];
cx q[4],q[14];
cx q[20],q[21];
cx q[10],q[15];
cx q[18],q[11];
cx q[19],q[24];
cx q[1],q[3];
cx q[6],q[0];
cx q[5],q[16];
cx q[13],q[17];
cx q[15],q[16];
cx q[6],q[5];
cx q[12],q[10];
cx q[4],q[2];
cx q[24],q[13];
cx q[18],q[8];
cx q[11],q[17];
cx q[23],q[19];
cx q[24],q[19];
cx q[21],q[22];
cx q[18],q[8];
cx q[23],q[13];
cx q[15],q[20];
cx q[3],q[9];
cx q[14],q[4];
cx q[12],q[10];
cx q[17],q[11];
cx q[5],q[10];
cx q[23],q[13];
cx q[21],q[22];
cx q[3],q[9];
cx q[15],q[20];
cx q[0],q[1];
cx q[24],q[19];
cx q[16],q[12];
cx q[10],q[15];
cx q[4],q[8];
cx q[23],q[13];
cx q[22],q[17];
cx q[18],q[11];
cx q[3],q[7];
cx q[6],q[1];
cx q[17],q[16];
cx q[11],q[12];
cx q[13],q[18];
cx q[23],q[24];
cx q[20],q[15];
cx q[9],q[2];
cx q[3],q[7];
cx q[14],q[19];
cx q[21],q[22];
cx q[4],q[8];
cx q[13],q[18];
cx q[2],q[8];
cx q[12],q[17];
cx q[9],q[4];
cx q[19],q[14];
cx q[23],q[24];
cx q[6],q[7];
cx q[13],q[12];
cx q[1],q[0];
cx q[6],q[7];
cx q[20],q[21];
cx q[18],q[19];
cx q[4],q[9];
cx q[2],q[8];
cx q[16],q[17];
cx q[10],q[15];
