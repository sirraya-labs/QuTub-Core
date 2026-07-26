# QuTub ↔ Qiskit QASM Interop

## What this proves
`qasm_export.rs` builds three circuits (Bell state, GHZ state, 3-qubit QFT
applied to |101⟩) with QuTub's `QuantumCircuit` builder and writes each one
as a real OPENQASM 2.0 `.qasm` file — the exact same format IBM Qiskit and
most other quantum SDKs consume.

`validate_with_qiskit.py` then loads each `.qasm` file into **actual,
independently-installed Qiskit** (not a mock, not a hand reimplementation),
simulates it there, and compares Qiskit's answer against QuTub's own
computed probabilities *and* full complex statevector (phase included) for
the same circuit. Result on this machine (Qiskit 2.5.1):

```
bell_state  -> PASS (fidelity 1.0000000000)
ghz_state   -> PASS (fidelity 1.0000000000)
qft_101     -> PASS (fidelity 1.0000000000)
```

The probability check alone is weak for the QFT case (its output
probabilities are uniform regardless of correctness — only the phases
carry information), so the statevector fidelity check is the one that
actually exercises the `cp`/rotation-gate export path and the QFT
bit-reversal convention documented in `qft.rs`.

## One real interop gotcha found along the way
Qiskit's `qasm2.load()` only wires up the *original 1996 OpenQASM paper's*
gate subset by default when it sees `include "qelib1.inc";` — notably
**missing `swap` and `cp`**, both of which QuTub's QFT export uses. Fixed
by passing `custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS` and
`custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL`, which is what every other
tool actually means by "the standard qelib1.inc gate set." Worth keeping
this note somewhere visible (e.g. a comment in the QASM header, or the
project README) since anyone loading QuTub's `.qasm` output straight into
`qasm2.load()` without those two kwargs will hit the same `'cp' is not
defined in this scope` error.

## Reproducing
```bash
cargo run --example qasm_export        # writes *.qasm, *.probs, *.state
pip install qiskit                      # if not already installed
python3 examples/validate_with_qiskit.py .
```