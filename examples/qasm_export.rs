//! Export QuTub circuits as real OPENQASM 2.0 files, byte for byte the
//! same format IBM Qiskit (and most other quantum SDKs) reads natively.
//! This is the interop test: build a circuit once in QuTub, write it to
//! `.qasm`, and load that *exact file* into an independent, third-party
//! quantum computing framework to check that its simulation agrees with
//! ours. If Qiskit and QuTub compute the same probability distribution
//! for the same circuit, that's a real, external correctness signal --
//! not just an internal self-consistency check.
//!
//! Alongside each `.qasm` file this writes a matching `.probs` file
//! containing QuTub's own computed probability distribution for that
//! circuit, so the validation script has something concrete, numeric,
//! and independently produced to diff Qiskit's answer against (rather
//! than a hand-typed "expected" value that could just as easily be
//! wrong in the same way on both sides).

use sirraya_qutub::core::QuantumCircuit;
use std::f64::consts::PI;
use std::fs;

fn write_outputs(name: &str, circuit: &QuantumCircuit, qasm: String) -> Result<(), String> {
    let qasm_path = format!("{}.qasm", name);
    let probs_path = format!("{}.probs", name);
    let state_path = format!("{}.state", name);

    fs::write(&qasm_path, &qasm).map_err(|e| e.to_string())?;

    let distribution = circuit.get_register().get_probability_distribution();
    let mut lines: Vec<(String, f64)> = distribution.into_iter().collect();
    lines.sort_by(|a, b| a.0.cmp(&b.0));
    let probs_text: String = lines
        .iter()
        .map(|(state, p)| format!("{} {:.12}\n", state, p))
        .collect();
    fs::write(&probs_path, probs_text).map_err(|e| e.to_string())?;

    // Full complex amplitudes, indexed 0..dimension-1 -- unlike the
    // probability distribution, this is sensitive to phase errors, which
    // matters for circuits like the QFT where every output probability
    // is uniform regardless of whether the phases are actually correct.
    let state_text: String = circuit
        .get_register()
        .get_state_vector()
        .iter()
        .enumerate()
        .map(|(i, amp)| format!("{} {:.12} {:.12}\n", i, amp.real(), amp.imag()))
        .collect();
    fs::write(&state_path, state_text).map_err(|e| e.to_string())?;

    println!("  wrote {}, {} and {}", qasm_path, probs_path, state_path);
    for (state, p) in &lines {
        println!("    |{}>  p = {:.6}", state, p);
    }
    Ok(())
}

fn main() -> Result<(), String> {
    println!("══════════════════════════════════════════════════════════════");
    println!("        QuTub • OPENQASM 2.0 Export (Qiskit Interop)");
    println!("══════════════════════════════════════════════════════════════");

    // ── Circuit 1: Bell state ────────────────────────────────────
    println!("\nCircuit 1 — Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2");
    println!("──────────────────────────────────────────────────────────────");
    let mut bell = QuantumCircuit::new(2)?;
    bell.hadamard(0).cnot(0, 1);
    bell.build().map_err(|errs| errs.join("; "))?;
    let bell_qasm = bell.to_qasm("bell_state");
    print!("{}", bell_qasm);
    write_outputs("bell_state", &bell, bell_qasm)?;

    // ── Circuit 2: GHZ state (3 qubits) ──────────────────────────
    println!("\nCircuit 2 — GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2");
    println!("──────────────────────────────────────────────────────────────");
    let mut ghz = QuantumCircuit::new(3)?;
    ghz.hadamard(0).cnot(0, 1).cnot(0, 2);
    ghz.build().map_err(|errs| errs.join("; "))?;
    let ghz_qasm = ghz.to_qasm("ghz_state");
    print!("{}", ghz_qasm);
    write_outputs("ghz_state", &ghz, ghz_qasm)?;

    // ── Circuit 3: 3-qubit QFT on |101⟩ ──────────────────────────
    // Same gate sequence as `quantum_fourier_transform` in core.rs,
    // but built through the QASM-emitting `QuantumCircuit` API instead
    // of calling that free function directly -- this exercises rz/cp
    // rotation-gate export (not just the pure-Clifford Bell/GHZ cases)
    // and cross-checks the QFT bit-reversal convention documented in
    // `qft.rs` against an independent framework.
    println!("\nCircuit 3 — 3-qubit QFT applied to |101⟩ (index 5)");
    println!("──────────────────────────────────────────────────────────────");
    let n = 3usize;
    let mut qft = QuantumCircuit::new(n)?;
    qft.x(0).x(2); // prepare |101>, i.e. basis index 5 (bit0=1, bit2=1)
    for i in 0..n {
        qft.hadamard(i);
        for j in (i + 1)..n {
            let angle = 2.0 * PI / (1u32 << (j - i + 1)) as f64;
            qft.controlled_phase(j, i, angle);
        }
    }
    for i in 0..n / 2 {
        qft.swap(i, n - 1 - i);
    }
    qft.build().map_err(|errs| errs.join("; "))?;
    let qft_qasm = qft.to_qasm("qft_101");
    print!("{}", qft_qasm);
    write_outputs("qft_101", &qft, qft_qasm)?;

    println!("\n══════════════════════════════════════════════════════════════");
    println!("Three .qasm files + matching .probs files written to disk.");
    println!("Next: load each .qasm into Qiskit and compare its computed");
    println!("probability distribution against the .probs file -- see");
    println!("validate_with_qiskit.py.");
    println!("══════════════════════════════════════════════════════════════");

    Ok(())
}