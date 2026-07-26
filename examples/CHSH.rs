//! CHSH inequality violation: the sharpest, least-ambiguous demonstration
//! that a quantum simulator is doing something a classical local-hidden-
//! variable model cannot (Clauser, Horne, Shimony & Holt, "Proposed
//! Experiment to Test Local Hidden-Variable Theories", PRL 1969).
//!
//! Any theory where each qubit carries a predetermined answer to every
//! possible measurement, and the two qubits don't communicate, is
//! mathematically bounded by |S| <= 2 (the classical/local bound).
//! Quantum mechanics predicts |S| = 2*sqrt(2) ~ 2.828 (the Tsirelson
//! bound) for a maximally entangled pair measured at the right angles.
//! Reproducing that number here is a direct correctness check on the
//! simulator's entangled-state and multi-qubit-observable machinery --
//! not just "does it look like a superposition", but "does it violate
//! an inequality that a fundamentally different (classical) model
//! cannot".

use sirraya_qutub::core::{create_bell_state, PauliOp, QuantumRegister};
use std::env;
use std::f64::consts::PI;

/// Correlator E(a, b) = <(cos a * Z + sin a * X) (x) (cos b * Z + sin b * X)>
/// for a state, computed exactly from the four Pauli-string expectation
/// values already exposed by `expectation_value_pauli_string` -- i.e.
/// "what would you get, on average, if Alice measured her qubit along
/// the axis at angle `a` in the XZ-plane and Bob measured his along `b`,
/// and you multiplied their +-1 outcomes together". No rotation gates
/// needed: this is just expanding the rotated-basis measurement operator
/// in the Pauli basis and using linearity of expectation.
fn correlator(reg: &QuantumRegister, a: f64, b: f64) -> Result<f64, String> {
    let zz = reg.expectation_value_pauli_string(&[(0, PauliOp::Z), (1, PauliOp::Z)])?;
    let zx = reg.expectation_value_pauli_string(&[(0, PauliOp::Z), (1, PauliOp::X)])?;
    let xz = reg.expectation_value_pauli_string(&[(0, PauliOp::X), (1, PauliOp::Z)])?;
    let xx = reg.expectation_value_pauli_string(&[(0, PauliOp::X), (1, PauliOp::X)])?;

    Ok(a.cos() * b.cos() * zz + a.cos() * b.sin() * zx + a.sin() * b.cos() * xz + a.sin() * b.sin() * xx)
}

/// Empirical correlator: actually measure `shots` fresh copies of the
/// state, each rotated so that "measure Z" becomes "measure along the
/// chosen angle", and average the product of +-1 outcomes. This is the
/// same quantity `correlator` computes in closed form -- run both and
/// they should agree within shot noise.
fn correlator_empirical(
    fresh_bell_state: impl Fn() -> Result<QuantumRegister, String>,
    a: f64,
    b: f64,
    shots: usize,
) -> Result<f64, String> {
    let mut sum = 0.0;
    for _ in 0..shots {
        let mut reg = fresh_bell_state()?;
        // Rotate the measurement axis from Z to (cos a, sin a) in the
        // XZ-plane by rotating the *state* by -a about Y before
        // measuring in the computational (Z) basis.
        reg.apply_ry(0, -a)?;
        reg.apply_ry(1, -b)?;
        let m0 = reg.measure_single_qubit(0)?;
        let m1 = reg.measure_single_qubit(1)?;
        let s0 = if m0 == 0 { 1.0 } else { -1.0 };
        let s1 = if m1 == 0 { 1.0 } else { -1.0 };
        sum += s0 * s1;
    }
    Ok(sum / shots as f64)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let shots: usize = args.get(1).map(|s| s.parse().unwrap_or(4000)).unwrap_or(4000);

    println!("══════════════════════════════════════════════════════════════");
    println!("           QuTub • CHSH Inequality Violation");
    println!("══════════════════════════════════════════════════════════════");
    println!("\nShared state: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 (qubits 0-1)");
    println!("Measurement settings (angles in the XZ-plane of the Bloch sphere):");

    // Standard CHSH-optimal settings for |Φ+>: Alice at 0, 90°; Bob at
    // 45°, -45°. These are the angles that saturate the Tsirelson bound.
    let a0 = 0.0;
    let a1 = PI / 2.0;
    let b0 = PI / 4.0;
    let b1 = -PI / 4.0;

    println!("  Alice: a0 = 0°,  a1 = 90°");
    println!("  Bob:   b0 = 45°, b1 = -45°");

    // ── Part 1: exact correlators from the state vector ──────────
    println!("\nPart 1 — Exact Correlators (closed-form, from the state vector)");
    println!("──────────────────────────────────────────────────────────────");

    let bell = create_bell_state()?;
    let e_a0b0 = correlator(&bell, a0, b0)?;
    let e_a0b1 = correlator(&bell, a0, b1)?;
    let e_a1b0 = correlator(&bell, a1, b0)?;
    let e_a1b1 = correlator(&bell, a1, b1)?;

    println!("  E(a0,b0) = {:>8.5}   (theory: cos(a0-b0) = {:.5})", e_a0b0, (a0 - b0).cos());
    println!("  E(a0,b1) = {:>8.5}   (theory: cos(a0-b1) = {:.5})", e_a0b1, (a0 - b1).cos());
    println!("  E(a1,b0) = {:>8.5}   (theory: cos(a1-b0) = {:.5})", e_a1b0, (a1 - b0).cos());
    println!("  E(a1,b1) = {:>8.5}   (theory: cos(a1-b1) = {:.5})", e_a1b1, (a1 - b1).cos());

    let s_exact = e_a0b0 + e_a0b1 + e_a1b0 - e_a1b1;
    println!("\n  S = E(a0,b0) + E(a0,b1) + E(a1,b0) - E(a1,b1) = {:.6}", s_exact);
    println!("  Tsirelson bound (quantum max):  2√2 = {:.6}", 2.0 * 2.0f64.sqrt());
    println!(
        "  Matches Tsirelson bound: {}",
        if (s_exact - 2.0 * 2.0f64.sqrt()).abs() < 1e-9 { "✓" } else { "✗" }
    );

    // ── Part 2: empirical correlators from simulated measurements ─
    println!("\nPart 2 — Empirical Correlators ({} simulated measurement shots per setting)", shots);
    println!("──────────────────────────────────────────────────────────────");

    let fresh = create_bell_state;
    let e_a0b0_emp = correlator_empirical(fresh, a0, b0, shots)?;
    let e_a0b1_emp = correlator_empirical(fresh, a0, b1, shots)?;
    let e_a1b0_emp = correlator_empirical(fresh, a1, b0, shots)?;
    let e_a1b1_emp = correlator_empirical(fresh, a1, b1, shots)?;

    println!("  E(a0,b0) = {:>8.5}   (exact: {:.5})", e_a0b0_emp, e_a0b0);
    println!("  E(a0,b1) = {:>8.5}   (exact: {:.5})", e_a0b1_emp, e_a0b1);
    println!("  E(a1,b0) = {:>8.5}   (exact: {:.5})", e_a1b0_emp, e_a1b0);
    println!("  E(a1,b1) = {:>8.5}   (exact: {:.5})", e_a1b1_emp, e_a1b1);

    let s_empirical = e_a0b0_emp + e_a0b1_emp + e_a1b0_emp - e_a1b1_emp;
    println!("\n  S (empirical, {} shots/setting) = {:.4}", shots, s_empirical);

    // ── Verdict ────────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════");
    println!("                        Verdict");
    println!("══════════════════════════════════════════════════════════════");
    println!("  Classical / local-hidden-variable bound:  |S| ≤ 2");
    println!("  Quantum mechanical (Tsirelson) bound:      |S| ≤ 2√2 ≈ 2.8284");
    println!("  Exact simulator result:                    S = {:.4}", s_exact);
    println!("  Empirical (measurement-based) result:      S = {:.4}", s_empirical);

    let violates_classical = s_exact > 2.0 && s_empirical > 2.0;
    if violates_classical {
        println!("\n✓ S exceeds 2 in both the exact and empirical calculations.");
        println!("✓ No local hidden-variable model can reproduce this correlation.");
        println!("✓ The simulator's entangled state + multi-qubit observables are");
        println!("  behaving as genuinely quantum, not as a classical-correlation");
        println!("  simulation dressed up in complex numbers.");
    } else {
        println!("\n✗ CHSH violation not observed -- something is wrong.");
    }
    println!("══════════════════════════════════════════════════════════════");

    Ok(())
}