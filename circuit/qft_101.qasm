OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
// Circuit: qft_101
x q[0];
x q[2];
h q[0];
cp(1.5707963267948966) q[1], q[0];
cp(0.7853981633974483) q[2], q[0];
h q[1];
cp(1.5707963267948966) q[2], q[1];
h q[2];
swap q[0], q[2];
measure q -> c;
