OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
// Two Toffolis that share controls, with a phase gate on a control in between.
ccx q[0], q[1], q[2];
t q[0];
ccx q[0], q[1], q[3];
tdg q[0];
ccx q[0], q[1], q[3];
cx q[2], q[3];
