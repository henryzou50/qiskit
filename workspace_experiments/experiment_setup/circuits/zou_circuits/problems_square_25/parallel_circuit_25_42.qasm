OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[6],q[9];
cx q[10],q[13];
cx q[0],q[8];
cx q[1],q[20];
cx q[12],q[17];
cx q[3],q[24];
cx q[23],q[16];
cx q[15],q[4];
cx q[19],q[11];
cx q[9],q[8];
cx q[24],q[12];
cx q[1],q[22];
cx q[17],q[19];
cx q[11],q[21];
cx q[23],q[16];
cx q[7],q[10];
cx q[2],q[3];
cx q[1],q[22];
cx q[12],q[24];
cx q[17],q[9];
cx q[8],q[6];
cx q[10],q[11];
cx q[16],q[23];
cx q[0],q[5];
cx q[21],q[19];
cx q[3],q[13];
cx q[7],q[15];
cx q[1],q[14];
cx q[8],q[17];
cx q[15],q[7];
cx q[13],q[3];
cx q[24],q[23];
cx q[11],q[19];
cx q[6],q[12];
cx q[10],q[9];
cx q[20],q[0];
cx q[18],q[5];
cx q[21],q[2];
cx q[4],q[9];
cx q[13],q[15];
cx q[12],q[0];
cx q[21],q[6];
cx q[7],q[3];
cx q[23],q[24];
cx q[19],q[22];
cx q[2],q[5];
cx q[6],q[8];
cx q[17],q[12];
cx q[7],q[13];
cx q[10],q[9];
cx q[16],q[20];
cx q[21],q[4];
cx q[3],q[18];
cx q[1],q[14];
cx q[3],q[18];
cx q[23],q[16];
cx q[0],q[19];
cx q[6],q[17];
cx q[11],q[15];
cx q[7],q[13];
cx q[4],q[10];
cx q[1],q[14];
cx q[2],q[12];
cx q[20],q[24];
cx q[21],q[8];
cx q[23],q[20];
cx q[12],q[16];
cx q[6],q[17];
cx q[10],q[9];
cx q[22],q[11];
cx q[19],q[0];
cx q[24],q[5];
cx q[7],q[1];
cx q[21],q[4];
cx q[15],q[14];
cx q[2],q[8];
cx q[13],q[18];
cx q[19],q[3];
cx q[11],q[0];
cx q[12],q[24];
cx q[21],q[9];
cx q[14],q[7];
cx q[20],q[5];
cx q[1],q[15];
cx q[16],q[23];
cx q[6],q[8];
cx q[2],q[17];
cx q[10],q[18];
cx q[11],q[0];
cx q[2],q[8];
cx q[19],q[10];
cx q[3],q[18];
cx q[22],q[17];
cx q[21],q[6];
cx q[4],q[9];
cx q[7],q[1];
cx q[9],q[3];
cx q[6],q[10];
cx q[23],q[12];
cx q[0],q[17];
cx q[5],q[2];
cx q[4],q[8];
cx q[1],q[13];
cx q[15],q[7];
cx q[11],q[19];
cx q[24],q[22];
cx q[18],q[14];
cx q[20],q[16];
cx q[14],q[18];
cx q[9],q[10];
cx q[22],q[5];
cx q[7],q[15];
cx q[12],q[20];
cx q[16],q[23];
cx q[19],q[6];
cx q[6],q[9];
cx q[2],q[21];
cx q[22],q[0];
cx q[13],q[18];
cx q[10],q[19];
cx q[15],q[16];
cx q[6],q[19];
cx q[24],q[20];
cx q[18],q[1];
cx q[0],q[17];
cx q[7],q[11];
cx q[4],q[9];
cx q[16],q[5];
cx q[23],q[12];
cx q[2],q[21];
cx q[8],q[22];
cx q[3],q[13];
cx q[24],q[20];
cx q[23],q[3];
cx q[11],q[5];
cx q[4],q[10];
cx q[22],q[2];
cx q[18],q[1];
cx q[9],q[20];
cx q[19],q[18];
cx q[16],q[12];
cx q[7],q[14];
cx q[17],q[0];
cx q[11],q[15];
cx q[2],q[24];
cx q[3],q[13];
cx q[8],q[22];
cx q[9],q[16];
cx q[24],q[22];
cx q[17],q[6];
cx q[19],q[0];
cx q[11],q[5];
cx q[23],q[21];
cx q[5],q[23];
cx q[4],q[3];
cx q[22],q[17];
cx q[12],q[9];
cx q[6],q[1];
cx q[8],q[2];
cx q[0],q[24];
cx q[18],q[19];
cx q[20],q[16];
cx q[11],q[13];
cx q[7],q[15];
cx q[23],q[11];
cx q[12],q[14];
cx q[2],q[8];
cx q[16],q[21];
cx q[10],q[13];
cx q[7],q[18];
cx q[19],q[1];
cx q[0],q[22];
cx q[4],q[6];
cx q[7],q[10];
cx q[14],q[23];
cx q[1],q[19];
cx q[16],q[18];
cx q[11],q[13];
cx q[15],q[20];
cx q[12],q[2];
cx q[18],q[16];
cx q[21],q[11];
cx q[15],q[8];
cx q[1],q[6];
cx q[0],q[22];
cx q[3],q[13];
cx q[7],q[17];
cx q[10],q[4];
cx q[6],q[1];
cx q[3],q[11];
cx q[14],q[23];
cx q[5],q[9];
cx q[7],q[21];
cx q[10],q[4];
cx q[24],q[0];
cx q[13],q[18];
cx q[17],q[22];
cx q[13],q[3];
cx q[8],q[16];
cx q[6],q[4];
cx q[15],q[7];
cx q[19],q[11];
cx q[2],q[9];
cx q[21],q[17];
cx q[15],q[13];
cx q[14],q[16];
cx q[21],q[20];
cx q[10],q[19];
cx q[9],q[12];
cx q[8],q[2];
cx q[23],q[18];
cx q[3],q[7];
cx q[4],q[1];
cx q[3],q[15];
cx q[17],q[21];
cx q[13],q[14];
cx q[16],q[12];
cx q[18],q[7];
cx q[4],q[6];
cx q[1],q[10];
cx q[12],q[24];
cx q[10],q[1];
cx q[11],q[0];
cx q[8],q[5];
cx q[17],q[4];
cx q[7],q[19];
cx q[13],q[3];
cx q[22],q[20];
cx q[6],q[21];
cx q[19],q[7];
cx q[9],q[18];
cx q[10],q[0];
cx q[2],q[23];
cx q[3],q[6];
cx q[14],q[15];
cx q[7],q[19];
cx q[3],q[14];
cx q[5],q[8];
cx q[4],q[20];
cx q[18],q[12];
cx q[10],q[11];
cx q[23],q[14];
cx q[21],q[5];
cx q[4],q[0];
cx q[20],q[2];
cx q[24],q[15];
cx q[18],q[16];
cx q[3],q[13];
cx q[2],q[20];
cx q[22],q[12];
cx q[23],q[24];
cx q[17],q[5];
cx q[11],q[19];
cx q[18],q[7];
cx q[16],q[21];
cx q[4],q[10];
cx q[9],q[3];
cx q[1],q[6];
cx q[2],q[12];
cx q[22],q[20];
cx q[4],q[1];
cx q[16],q[24];
cx q[8],q[5];
cx q[21],q[23];
cx q[13],q[18];
cx q[10],q[11];
cx q[24],q[18];
cx q[12],q[17];
cx q[11],q[6];
cx q[5],q[21];
cx q[14],q[13];
cx q[20],q[2];
cx q[3],q[15];
cx q[23],q[16];
cx q[16],q[23];
cx q[5],q[20];
cx q[10],q[6];
cx q[9],q[15];
cx q[11],q[4];
cx q[17],q[8];
cx q[18],q[19];
cx q[11],q[6];
cx q[10],q[14];
cx q[2],q[0];
cx q[17],q[21];
cx q[20],q[13];
cx q[16],q[5];
cx q[24],q[9];
cx q[1],q[7];
cx q[15],q[12];
cx q[15],q[20];
cx q[2],q[4];
cx q[3],q[23];
cx q[22],q[8];
cx q[11],q[10];
cx q[21],q[17];
cx q[5],q[12];
cx q[14],q[18];
cx q[24],q[16];
cx q[19],q[1];
cx q[13],q[7];
cx q[21],q[8];
cx q[9],q[23];
cx q[15],q[22];
cx q[7],q[1];
cx q[6],q[11];
cx q[12],q[5];
cx q[20],q[17];
cx q[18],q[24];
cx q[3],q[0];
cx q[10],q[16];
cx q[4],q[2];
cx q[3],q[12];
cx q[23],q[24];
cx q[16],q[5];
cx q[2],q[11];
cx q[1],q[18];
cx q[8],q[0];
cx q[10],q[6];
cx q[19],q[14];
cx q[17],q[21];
cx q[22],q[15];
cx q[19],q[1];
cx q[24],q[3];
cx q[20],q[16];
cx q[18],q[13];
cx q[6],q[10];
cx q[21],q[17];
cx q[7],q[12];
cx q[23],q[24];
cx q[20],q[11];
cx q[13],q[14];
cx q[16],q[19];
cx q[4],q[6];
cx q[10],q[2];
cx q[9],q[22];
cx q[15],q[17];
cx q[22],q[23];
cx q[12],q[19];
cx q[14],q[7];
cx q[11],q[20];
cx q[2],q[1];
cx q[17],q[15];
cx q[10],q[0];
cx q[5],q[4];
cx q[21],q[6];
cx q[18],q[13];
cx q[7],q[1];
cx q[13],q[4];
cx q[2],q[10];
cx q[19],q[18];
cx q[16],q[20];
cx q[14],q[3];
cx q[24],q[22];
cx q[23],q[17];
cx q[5],q[11];
cx q[12],q[6];
cx q[8],q[9];
cx q[1],q[12];
cx q[24],q[18];
cx q[22],q[23];
cx q[8],q[9];
cx q[19],q[17];
cx q[5],q[11];
cx q[2],q[0];
cx q[10],q[16];
cx q[20],q[15];
