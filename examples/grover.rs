//! Grover's search algorithm, toy instance: find a marked item among 8
//! unstructured entries (3 qubits) with quadratic speedup over classical
//! brute-force search (Grover, "A fast quantum mechanical algorithm for
//! database search", STOC 1996).
//!
//! This demo exercises the actual `QuantumAlgorithm::grover_iteration`
//! building block already shipped in `core.rs` (oracle phase-flip +
//! diffusion operator, applied as a single unit per call) -- it isn't a
//! separate reimplementation, it's the library's own primitive wired up
//! to a concrete oracle and run end-to-end with measurement and success
//! statistics, the same way `shor_n15.rs` wires up the library's QFT /
//! phase-estimation primitives to a concrete factoring instance.
//!
//! Search space: 3 qubits (N = 8 basis states), one marked item.
//! Marked state: |101> (basis index 5) -- an arbitrary but fixed choice,
//! analogous to fixing a = 7, N = 15 in the Shor's algorithm demo.
//!
//! Oracle construction: the standard X-conjugated multi-controlled-Z
//! phase-flip trick. `apply_multi_controlled_z` flips the phase of any
//! basis state where every qubit it touches is |1>. To mark an arbitrary
//! target bitstring, first apply X to every qubit where the target bit
//! is 0 (mapping the target state onto |11...1>), apply the
//! multi-controlled Z, then undo the X's. This is the same
//! X-conjugation technique the library's own diffusion operator
//! (`QuantumAlgorithm::grover_iteration`) already uses internally to
//! implement reflection about |00...0>.
//!
//! Iteration count: for N basis states and M marked items, the success
//! probability after k iterations is sin^2((2k+1)*theta), where
//! theta = asin(sqrt(M/N)). This is maximized near
//! k ~ (pi/4) * sqrt(N/M) - 1/2; overshooting this optimum *reduces*
//! success probability (Grover's algorithm is periodic, not
//! monotonically improving), which this demo also prints out explicitly
//! as a sanity check / teaching point.

use sirraya_qutub::core::{QuantumAlgorithm, QuantumRegister};
use std::collections::HashMap;
use std::env;
use std::f64::consts::PI;

const NUM_QUBITS: usize = 3;
const MARKED_INDEX: usize = 5; // |101>, arbitrary fixed target (like a=7,N=15 in Shor's demo)

/// Oracle marking the single basis state |101> (index 5) via the
/// standard X-conjugated multi-controlled-Z phase-flip: flip qubits
/// whose target bit is 0, apply multi-controlled Z across all qubits
/// (phase-flips iff every qubit is currently |1>), then undo the flips.
fn oracle_mark_101(reg: &mut QuantumRegister) -> Result<(), String> {
    let n = reg.num_qubits();
    for q in 0..n {
        if (MARKED_INDEX >> q) & 1 == 0 {
            reg.apply_pauli_x(q)?;
        }
    }
    let controls: Vec<usize> = (0..n - 1).collect();
    reg.apply_multi_controlled_z(&controls, n - 1)?;
    for q in 0..n {
        if (MARKED_INDEX >> q) & 1 == 0 {
            reg.apply_pauli_x(q)?;
        }
    }
    Ok(())
}

/// Optimal number of Grover iterations for `n_items` basis states and
/// `n_marked` marked items: round((pi / (4*theta)) - 1/2), theta =
/// asin(sqrt(n_marked / n_items)).
fn optimal_iterations(n_items: usize, n_marked: usize) -> usize {
    let theta = ((n_marked as f64 / n_items as f64).sqrt()).asin();
    (((PI / (4.0 * theta)) - 0.5).round().max(1.0)) as usize
}

/// Theoretical success probability after `k` Grover iterations.
fn theoretical_success_probability(n_items: usize, n_marked: usize, k: usize) -> f64 {
    let theta = ((n_marked as f64 / n_items as f64).sqrt()).asin();
    ((2.0 * k as f64 + 1.0) * theta).sin().powi(2)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let shots: usize = args.get(1).map(|s| s.parse().unwrap_or(200)).unwrap_or(200);
    let iterations_override: Option<usize> = args.get(2).and_then(|s| s.parse().ok());

    let dim = 1usize << NUM_QUBITS;
    let k = iterations_override.unwrap_or_else(|| optimal_iterations(dim, 1));

    println!("GROVER'S SEARCH -- TOY INSTANCE: {} QUBITS ({} STATES)", NUM_QUBITS, dim);
    println!("=========================================================\n");
    println!(
        "Marked state: |{:0width$b}>  (basis index {})",
        MARKED_INDEX,
        MARKED_INDEX,
        width = NUM_QUBITS
    );
    println!("Grover iterations: {} (theory-optimal for N={}, M=1)", k, dim);
    println!(
        "Theoretical success probability at k={}: {:.4}",
        k,
        theoretical_success_probability(dim, 1, k)
    );
    println!("\nFor comparison, classical unstructured search needs O(N) = up to {} queries on average.\n", dim);

    // Show the overshoot effect: success probability is periodic in k,
    // not monotonically increasing -- print a small table around the
    // optimum so this is visible rather than just asserted in a comment.
    println!("Success probability vs. iteration count (illustrates the periodicity / overshoot effect):");
    for kk in 0..=(k + 2) {
        println!(
            "  k={:<2} -> theoretical P(success) = {:.4}{}",
            kk,
            theoretical_success_probability(dim, 1, kk),
            if kk == k { "   <- used below" } else { "" }
        );
    }
    println!();

    let mut histogram: HashMap<usize, usize> = HashMap::new();
    let mut successes = 0usize;

    for _ in 0..shots {
        let mut reg = QuantumRegister::new(NUM_QUBITS)?;

        // Uniform superposition over all N basis states.
        for q in 0..NUM_QUBITS {
            reg.apply_hadamard(q)?;
        }

        // k rounds of oracle + diffusion, using the library's own
        // combined Grover-iteration primitive.
        for _ in 0..k {
            QuantumAlgorithm::grover_iteration(&mut reg, oracle_mark_101)?;
        }

        let bits = reg.measure_all_qubits()?;
        // `measure_all_qubits` returns bits MSB-first (qubit NUM_QUBITS-1
        // first, qubit 0 last), so reconstructing left-to-right gives
        // back the basis index directly -- no bit-reversal needed here
        // (unlike the QFT-based Shor's demo, there's no QFT in Grover's
        // algorithm to introduce that wrinkle).
        let mut measured = 0usize;
        for &b in bits.iter() {
            measured = (measured << 1) | (b as usize);
        }

        *histogram.entry(measured).or_insert(0) += 1;
        if measured == MARKED_INDEX {
            successes += 1;
        }
    }

    println!("Measured histogram over {} shots:", shots);
    let mut keys: Vec<&usize> = histogram.keys().collect();
    keys.sort();
    for k_idx in keys {
        let count = histogram[k_idx];
        let marker = if *k_idx == MARKED_INDEX { "  <-- marked item" } else { "" };
        println!(
            "  |{:0width$b}>  (index {:>2})  -> seen {:>3} times{}",
            k_idx,
            k_idx,
            count,
            marker,
            width = NUM_QUBITS
        );
    }

    println!(
        "\nMeasured |{:0width$b}> (the marked item) on {}/{} shots ({:.1}%).",
        MARKED_INDEX,
        successes,
        shots,
        100.0 * successes as f64 / shots as f64,
        width = NUM_QUBITS
    );
    println!(
        "Theoretical prediction: {:.1}%.",
        100.0 * theoretical_success_probability(dim, 1, k)
    );

    Ok(())
}