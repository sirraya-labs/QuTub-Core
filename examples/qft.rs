use sirraya_qutub::core::{quantum_fourier_transform, inverse_quantum_fourier_transform, QuantumRegister};
use std::f64::consts::PI;

const NUM_QUBITS: usize = 3;
const DIM: usize = 1 << NUM_QUBITS;

/// Prepare the computational basis state |value⟩ by flipping the qubits
/// whose bit is set in `value` (qubit q corresponds to bit 2^q).
fn prepare_basis_state(value: usize) -> Result<QuantumRegister, String> {
    let mut register = QuantumRegister::new(NUM_QUBITS)?;
    for q in 0..NUM_QUBITS {
        if (value >> q) & 1 == 1 {
            register.apply_pauli_x(q)?;
        }
    }
    Ok(register)
}

/// Reverse the low `bits` bits of `value`.
fn bit_reverse(mut value: usize, bits: usize) -> usize {
    let mut out = 0;
    for _ in 0..bits {
        out = (out << 1) | (value & 1);
        value >>= 1;
    }
    out
}

fn main() -> Result<(), String> {
    println!("══════════════════════════════════════════════════════════════");
    println!("           QuTub • Quantum Fourier Transform");
    println!("══════════════════════════════════════════════════════════════");
    println!("\n{} qubits, dimension {} ({} computational basis states)", NUM_QUBITS, DIM, DIM);

    // ── Part 1: QFT of |0...0⟩ is a uniform superposition ───────
    println!("\nPart 1 — QFT|0⟩ = Uniform Superposition");
    println!("──────────────────────────────────────────────────────────────");
    println!("QFT of the all-zero state should equal H^⊗n applied to |0⟩:");
    println!("every amplitude collapses to 1/√{} with zero relative phase.\n", DIM);

    let mut reg = prepare_basis_state(0)?;
    quantum_fourier_transform(&mut reg)?;
    reg.print_state();

    let expected_amp = 1.0 / (DIM as f64).sqrt();
    let mut part1_ok = true;
    for amp in reg.get_state_vector() {
        if (amp.real() - expected_amp).abs() > 1e-9 || amp.imag().abs() > 1e-9 {
            part1_ok = false;
        }
    }
    println!(
        "\nAll {} amplitudes equal 1/√{} = {:.6}: {}",
        DIM,
        DIM,
        expected_amp,
        if part1_ok { "✓" } else { "✗" }
    );

    // ── Part 2: QFT of a basis state |k⟩ produces the textbook ───
    // ── phase ramp e^{2πi j k / N} / √N ──────────────────────────
    //
    // NOTE ON INDEXING CONVENTION: QuTub represents qubit q as bit 2^q
    // of the state-vector index. Combined with the mid-circuit swaps in
    // `quantum_fourier_transform`, the amplitude landing at output index
    // j corresponds to the *bit-reversed* index in the textbook DFT
    // formula: amp[j] = (1/√N) e^(2πi·bitrev(j)·k/N). Every QFT circuit
    // built from an MSB-first gate ladder has to fix an endianness
    // convention somewhere -- worth knowing before wiring QuTub's QFT
    // into a phase-estimation or Shor's algorithm routine that expects
    // a particular bit order.
    let k = 5usize;
    println!("\nPart 2 — QFT|{}⟩ Produces the Phase Ramp", k);
    println!("──────────────────────────────────────────────────────────────");
    println!("QFT|k⟩ = (1/√N) Σⱼ e^(2πi·j·k/N) |j⟩  (textbook form)");
    println!("QuTub emits this bit-reversed: amp[j] = (1/√N) e^(2πi·bitrev(j)·k/N)");
    println!("Checking amplitude j against that closed form for k = {}:\n", k);

    let mut reg_k = prepare_basis_state(k)?;
    quantum_fourier_transform(&mut reg_k)?;
    let amps = reg_k.get_state_vector();

    println!("  j  bitrev(j)  Re(amp)   Im(amp)   Expected Re  Expected Im  Match");
    println!("──────────────────────────────────────────────────────────────");
    let mut part2_ok = true;
    for j in 0..DIM {
        let j_rev = bit_reverse(j, NUM_QUBITS);
        let angle = 2.0 * PI * (j_rev as f64) * (k as f64) / (DIM as f64);
        let expected_re = expected_amp * angle.cos();
        let expected_im = expected_amp * angle.sin();
        let got = amps[j];
        let matches = (got.real() - expected_re).abs() < 1e-9 && (got.imag() - expected_im).abs() < 1e-9;
        part2_ok &= matches;
        println!(
            "  {}     {}       {:>7.4}   {:>7.4}    {:>7.4}      {:>7.4}     {}",
            j,
            j_rev,
            got.real(),
            got.imag(),
            expected_re,
            expected_im,
            if matches { "✓" } else { "✗" }
        );
    }
    println!("\nBit-reversed phase ramp matches simulator output: {}", if part2_ok { "✓" } else { "✗" });

    // ── Part 3: Round trip — QFT⁻¹(QFT|ψ⟩) = |ψ⟩ ─────────────────
    println!("\nPart 3 — Round Trip: QFT⁻¹ ∘ QFT = Identity");
    println!("──────────────────────────────────────────────────────────────");
    let original = prepare_basis_state(k)?;
    let mut roundtrip = prepare_basis_state(k)?;
    quantum_fourier_transform(&mut roundtrip)?;
    inverse_quantum_fourier_transform(&mut roundtrip)?;

    let fidelity = original.fidelity(&roundtrip)?;
    println!("Starting state:  |{}⟩", k);
    println!("After QFT → QFT⁻¹:");
    roundtrip.print_state();
    println!(
        "\nFidelity ⟨ψ|QFT⁻¹ QFT|ψ⟩ = {:.6}  {}",
        fidelity,
        if (fidelity - 1.0).abs() < 1e-9 { "✓" } else { "✗" }
    );

    // ── Summary ───────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════");
    let all_ok = part1_ok && part2_ok && (fidelity - 1.0).abs() < 1e-9;
    if all_ok {
        println!("✓ QFT produces the correct uniform superposition on |0⟩");
        println!("✓ QFT phase ramp matches the closed-form DFT formula");
        println!("✓ QFT⁻¹ ∘ QFT recovers the original state exactly");
    } else {
        println!("✗ QFT verification failed");
    }
    println!("══════════════════════════════════════════════════════════════");
    println!("\nThe QFT is the quantum analogue of the discrete Fourier");
    println!("transform, and the engine room behind Shor's algorithm and");
    println!("quantum phase estimation: it maps computational-basis states");
    println!("to superpositions carrying frequency information in their");
    println!("relative phases, all in O(n²) gates instead of O(n·2ⁿ).");
    println!("══════════════════════════════════════════════════════════════");

    Ok(())
}