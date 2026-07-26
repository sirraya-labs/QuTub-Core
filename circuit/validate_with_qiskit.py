#!/usr/bin/env python3
"""
Cross-validate QuTub's OPENQASM 2.0 export against Qiskit.

For each <name>.qasm file produced by `cargo run --example qasm_export`,
this script:
  1. Loads the exact .qasm file into a Qiskit QuantumCircuit
     (qiskit.qasm2.load) -- no re-typing of the circuit by hand.
  2. Strips the trailing measurement so Qiskit can compute an exact
     statevector rather than a shot-noisy sample.
  3. Computes Qiskit's own probability distribution via
     qiskit.quantum_info.Statevector.
  4. Compares it, state by state, against the <name>.probs file that
     QuTub wrote from its *own* internal simulation of the same circuit.

If every probability matches to numerical precision, two independently
written quantum simulators -- QuTub's custom Rust state-vector engine
and IBM's Qiskit -- agree on the physics. That is a much stronger
correctness claim than either engine agreeing with itself.

Usage:
    python3 validate_with_qiskit.py [directory]

(directory defaults to the current directory, where qasm_export.rs
writes its .qasm/.probs files)
"""

import sys
import glob
import os
import numpy as np

from qiskit import qasm2
from qiskit.quantum_info import Statevector

TOLERANCE = 1e-6


def load_qutub_probs(path):
    """Parse a QuTub .probs file: lines of '<bitstring> <probability>'."""
    probs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            state, p = line.split()
            probs[state] = float(p)
    return probs


def load_qutub_state(path):
    """Parse a QuTub .state file: lines of '<index> <real> <imag>',
    returned as a complex numpy array indexed 0..dimension-1."""
    entries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, re, im = line.split()
            entries[int(idx)] = complex(float(re), float(im))
    dim = len(entries)
    return np.array([entries[i] for i in range(dim)], dtype=complex)


def statevector_fidelity(a, b):
    """Global-phase-invariant fidelity |<a|b>|^2 between two equal-length
    state vectors. 1.0 means identical physical states."""
    overlap = np.vdot(a, b)
    return abs(overlap) ** 2


def qiskit_state_from_qasm(qasm_path):
    """Load a .qasm file into Qiskit, drop measurements, return the exact
    complex statevector as a numpy array (same qubit-index convention as
    QuTub's own .state export -- qubit 0 is the least-significant bit of
    the index in both)."""
    circuit = qasm2.load(
        qasm_path,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
    )
    circuit.remove_final_measurements(inplace=True)
    return Statevector.from_instruction(circuit).data


def qiskit_probs_from_qasm(qasm_path):
    """Load a .qasm file into Qiskit, drop measurements, return its exact
    probability distribution keyed by bitstring in the same MSB-first
    convention QuTub uses (qubit n-1 leftmost, qubit 0 rightmost).

    Qiskit's magic `include "qelib1.inc";` handling only wires up the
    original 1996 OpenQASM paper's gate subset by default (u1/u2/u3, cx,
    the Paulis, h, s/sdg, t/tdg, rx/ry/rz, cz/cy/ch, ccx, crz, cu1, cu3
    -- notably missing `swap` and `cp`, both of which QuTub's own QFT
    export uses). Passing `qasm2.LEGACY_CUSTOM_INSTRUCTIONS` opts into
    Qiskit's full legacy qelib1.inc gate set instead, which is what
    every other simulator vendor actually treats "qelib1.inc" as
    meaning in practice.
    """
    circuit = qasm2.load(
        qasm_path,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
    )

    # Drop measure/barrier instructions so Statevector can simulate the
    # unitary part exactly instead of needing a sampled backend.
    circuit.remove_final_measurements(inplace=True)

    state = Statevector.from_instruction(circuit)

    # Qiskit's probabilities_dict() already keys by bitstring with
    # qubit 0 as the rightmost character -- the same convention QuTub's
    # index_to_bitstring uses -- so no reordering should be needed.
    return state.probabilities_dict()


def compare(name, qutub_probs, qiskit_probs, qutub_state=None, qiskit_state=None):
    print(f"\n{name}")
    print("-" * 64)
    all_states = sorted(set(qutub_probs) | set(qiskit_probs))
    ok = True
    print(f"  {'state':<10} {'QuTub p':>12} {'Qiskit p':>12} {'match':>8}")
    for state in all_states:
        p_qutub = qutub_probs.get(state, 0.0)
        p_qiskit = qiskit_probs.get(state, 0.0)
        matches = abs(p_qutub - p_qiskit) < TOLERANCE
        ok &= matches
        print(f"  |{state}>{'':<3} {p_qutub:>12.6f} {p_qiskit:>12.6f} {'OK' if matches else 'MISMATCH':>8}")

    if qutub_state is not None and qiskit_state is not None:
        # Phase-sensitive check: probabilities alone can't catch a wrong
        # relative phase (e.g. QFT's output probabilities are uniform
        # for *any* input, so this is the check that actually exercises
        # the rz/cp rotation-gate export).
        fidelity = statevector_fidelity(qutub_state, qiskit_state)
        phase_ok = abs(fidelity - 1.0) < TOLERANCE
        ok &= phase_ok
        print(f"  Full-statevector fidelity |<QuTub|Qiskit>|^2 = {fidelity:.10f}  "
              f"{'OK' if phase_ok else 'MISMATCH'}  (phase-sensitive check)")

    print(f"  Result: {'PASS -- QuTub and Qiskit agree' if ok else 'FAIL -- disagreement found'}")
    return ok


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    qasm_files = sorted(glob.glob(os.path.join(directory, "*.qasm")))

    if not qasm_files:
        print(f"No .qasm files found in {directory!r}. Run "
              f"`cargo run --example qasm_export` first.")
        sys.exit(1)

    print("=" * 64)
    print("  QuTub <-> Qiskit Cross-Validation")
    print("=" * 64)

    all_pass = True
    for qasm_path in qasm_files:
        name = os.path.splitext(os.path.basename(qasm_path))[0]
        probs_path = os.path.join(directory, f"{name}.probs")
        if not os.path.exists(probs_path):
            print(f"\n{name}: skipping -- no matching {name}.probs file")
            continue

        qutub_probs = load_qutub_probs(probs_path)
        qiskit_probs = qiskit_probs_from_qasm(qasm_path)

        qutub_state = None
        qiskit_state = None
        state_path = os.path.join(directory, f"{name}.state")
        if os.path.exists(state_path):
            qutub_state = load_qutub_state(state_path)
            qiskit_state = qiskit_state_from_qasm(qasm_path)

        all_pass &= compare(name, qutub_probs, qiskit_probs, qutub_state, qiskit_state)

    print("\n" + "=" * 64)
    print("SUMMARY:", "ALL CIRCUITS MATCH QISKIT" if all_pass else "MISMATCHES FOUND")
    print("=" * 64)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()