//! Sirraya QuTub — Experimental Visualization Dataset Generator
//!
//! Generates CSV datasets for:
//!
//! 1. State-vector scaling
//! 2. QFT scaling
//! 3. Ideal-vs-perturbed circuit fidelity as depth increases
//! 4. GHZ entanglement structure
//! 5. W-state structure
//! 6. Density-matrix noise / purity / entropy
//! 7. Bell-state probability distribution
//!
//! These experiments are separate from the correctness-validation suite.
//!
//! The validation suite asks:
//!     "Does QuTub implement the expected mathematics?"
//!
//! This example asks:
//!     "How does QuTub behave as the problem size, circuit depth,
//!      or noise strength changes?"

use sirraya_qutub::{
    create_bell_state,
    create_ghz_state,
    create_w_state,
    quantum_fourier_transform,
    DensityMatrix,
    QuantumRegister,
};

use std::fs;
use std::time::Instant;

/// QuTub currently rejects registers larger than 16 qubits.
///
/// Keep this synchronized with the current public implementation.
const MAX_SUPPORTED_QUBITS: usize = 16;

/// Maximum number of qubits used for the QFT experiment.
///
/// QFT becomes increasingly expensive because it contains O(n²) controlled
/// phase operations, each operating over a 2^n state vector.
const MAX_QFT_QUBITS: usize = 16;

/// Directory containing generated CSV datasets.
const OUTPUT_DIR: &str = "validation_output";

// ============================================================================
// Utility
// ============================================================================

fn ensure_output_directory() -> Result<(), String> {
    fs::create_dir_all(OUTPUT_DIR)
        .map_err(|e| format!("failed to create {OUTPUT_DIR}: {e}"))
}

// ============================================================================
// 1. STATE-VECTOR SCALING
// ============================================================================

/// Measures the cost of constructing a state vector and applying one
/// Hadamard layer across all qubits.
///
/// State-vector dimension is:
///
///     2^n
///
/// where n is the number of qubits.
///
/// This benchmark intentionally stops at QuTub's current maximum of
/// 16 qubits.
fn benchmark_state_vector_scaling() -> Result<(), String> {
    let path = format!("{OUTPUT_DIR}/state_vector_scaling.csv");

    let mut csv = String::from(
        "qubits,state_dimension,construction_ms,hadamard_chain_ms\n",
    );

    for qubits in 1usize..=MAX_SUPPORTED_QUBITS {
        let dimension = 1usize
            .checked_shl(qubits as u32)
            .ok_or_else(|| {
                format!("state dimension overflow for {qubits} qubits")
            })?;

        // Construction benchmark.
        let start = Instant::now();

        let _register = QuantumRegister::new(qubits)?;

        let construction_ms =
            start.elapsed().as_secs_f64() * 1000.0;

        // Gate-layer benchmark.
        let mut register = QuantumRegister::new(qubits)?;

        let start = Instant::now();

        for q in 0..qubits {
            register.apply_hadamard(q)?;
        }

        let hadamard_chain_ms =
            start.elapsed().as_secs_f64() * 1000.0;

        csv.push_str(&format!(
            "{qubits},{dimension},{construction_ms:.9},{hadamard_chain_ms:.9}\n"
        ));
    }

    fs::write(&path, csv)
        .map_err(|e| format!("failed to write {path}: {e}"))?;

    println!("wrote {path}");

    Ok(())
}

// ============================================================================
// 2. QFT SCALING
// ============================================================================

/// Measures QFT execution time as the number of qubits increases.
///
/// The input is deliberately non-trivial: every qubit starts in |+>.
fn benchmark_qft_scaling() -> Result<(), String> {
    let path = format!("{OUTPUT_DIR}/qft_scaling.csv");

    let mut csv = String::from(
        "qubits,state_dimension,qft_ms\n",
    );

    for qubits in 1usize..=MAX_QFT_QUBITS {
        let dimension = 1usize
            .checked_shl(qubits as u32)
            .ok_or_else(|| {
                format!("state dimension overflow for {qubits} qubits")
            })?;

        let mut register = QuantumRegister::new(qubits)?;

        // Prepare a non-trivial input state.
        for q in 0..qubits {
            register.apply_hadamard(q)?;
        }

        let start = Instant::now();

        quantum_fourier_transform(&mut register)?;

        let qft_ms =
            start.elapsed().as_secs_f64() * 1000.0;

        csv.push_str(&format!(
            "{qubits},{dimension},{qft_ms:.9}\n"
        ));
    }

    fs::write(&path, csv)
        .map_err(|e| format!("failed to write {path}: {e}"))?;

    println!("wrote {path}");

    Ok(())
}

// ============================================================================
// 3. FIDELITY VS CIRCUIT DEPTH
// ============================================================================

/// Measures the fidelity between:
///
///     ideal circuit
///
/// and
///
///     perturbed circuit
///
/// as circuit depth increases.
///
/// IMPORTANT:
///
/// This does NOT compare a state to itself.
///
/// The perturbed circuit receives a small deterministic Rz rotation after
/// every layer. Therefore:
///
///     F(ideal, perturbed)
///
/// measures the accumulated effect of coherent gate perturbations.
///
/// This is a numerical sensitivity experiment, NOT a physical hardware-noise
/// model.
fn fidelity_vs_depth() -> Result<(), String> {
    let path = format!("{OUTPUT_DIR}/fidelity_vs_depth.csv");

    let mut csv = String::from(
        "depth,ideal_vs_perturbed_fidelity,trace_distance\n",
    );

    const QUBITS: usize = 3;
    const MAX_DEPTH: usize = 50;

    // Small deterministic coherent perturbation.
    const PERTURBATION: f64 = 0.01;

    for depth in 1usize..=MAX_DEPTH {
        let mut ideal = QuantumRegister::new(QUBITS)?;
        let mut perturbed = QuantumRegister::new(QUBITS)?;

        for _layer in 0..depth {
            // ------------------------------------------------------------
            // Ideal circuit
            // ------------------------------------------------------------

            for q in 0..QUBITS {
                ideal.apply_hadamard(q)?;
            }

            ideal.apply_cnot(0, 1)?;
            ideal.apply_cnot(1, 2)?;

            // ------------------------------------------------------------
            // Perturbed circuit
            // ------------------------------------------------------------

            for q in 0..QUBITS {
                perturbed.apply_hadamard(q)?;
            }

            perturbed.apply_cnot(0, 1)?;
            perturbed.apply_cnot(1, 2)?;

            // Coherent perturbation.
            for q in 0..QUBITS {
                perturbed.apply_rz(q, PERTURBATION)?;
            }
        }

        // THIS is the meaningful comparison.
        let fidelity = ideal.fidelity(&perturbed)?;

        // For the pure-state fidelity convention used by QuTub:
        //
        //     D = sqrt(1 - F)
        //
        // for pure states.
        let trace_distance =
            (1.0 - fidelity).max(0.0).sqrt();

        csv.push_str(&format!(
            "{depth},{fidelity:.12},{trace_distance:.12}\n"
        ));
    }

    fs::write(&path, csv)
        .map_err(|e| format!("failed to write {path}: {e}"))?;

    println!("wrote {path}");

    Ok(())
}

// ============================================================================
// 4. GHZ ENTANGLEMENT STRUCTURE
// ============================================================================

/// Generates the probability of:
///
///     |00...0>
///
///     |11...1>
///
/// and all forbidden basis states.
///
/// For an ideal n-qubit GHZ state:
///
///     |GHZ_n> = (|00...0> + |11...1>) / sqrt(2)
///
/// therefore:
///
///     P(00...0) = 1/2
///     P(11...1) = 1/2
///     P(forbidden) = 0
fn ghz_entanglement_data() -> Result<(), String> {
    let path = format!("{OUTPUT_DIR}/ghz_entanglement.csv");

    let mut csv = String::from(
        "qubits,p_all_zero,p_all_ones,forbidden_probability\n",
    );

    for qubits in 2usize..=MAX_SUPPORTED_QUBITS {
        let ghz = create_ghz_state(qubits)?;

        let distribution =
            ghz.get_probability_distribution();

        let zero_state = "0".repeat(qubits);
        let one_state = "1".repeat(qubits);

        let p_zero = distribution
            .get(&zero_state)
            .copied()
            .unwrap_or(0.0);

        let p_one = distribution
            .get(&one_state)
            .copied()
            .unwrap_or(0.0);

        let forbidden_probability =
            (1.0 - p_zero - p_one).max(0.0);

        csv.push_str(&format!(
            "{qubits},{p_zero:.12},{p_one:.12},{forbidden_probability:.12}\n"
        ));
    }

    fs::write(&path, csv)
        .map_err(|e| format!("failed to write {path}: {e}"))?;

    println!("wrote {path}");

    Ok(())
}

// ============================================================================
// 5. W-STATE STRUCTURE
// ============================================================================

/// Checks that an n-qubit W state contains equal probability in every
/// single-excitation basis state.
///
/// For:
///
///     |W_n> = 1/sqrt(n) * sum_i |0...010...0>
///
/// every single-excitation state should have probability:
///
///     1/n
fn w_state_data() -> Result<(), String> {
    let path = format!("{OUTPUT_DIR}/w_state.csv");

    let mut csv = String::from(
        "qubits,expected_single_excitation_probability,max_error,total_probability\n",
    );

    for qubits in 2usize..=MAX_SUPPORTED_QUBITS {
        let w = create_w_state(qubits)?;

        let distribution =
            w.get_probability_distribution();

        let expected =
            1.0 / qubits as f64;

        let mut max_error: f64 = 0.0;

        for q in 0..qubits {
            let index = 1usize << q;

            // Convert basis index to a fixed-width binary string because
            // QuTub's probability-distribution API uses String keys.
            let state =
                format!("{index:0width$b}", width = qubits);

            let probability = distribution
                .get(&state)
                .copied()
                .unwrap_or(0.0);

            let error =
                (probability - expected).abs();

            max_error =
                max_error.max(error);
        }

        let total_probability: f64 =
            distribution.values().sum();

        csv.push_str(&format!(
            "{qubits},{expected:.12},{max_error:.12},{total_probability:.12}\n"
        ));
    }

    fs::write(&path, csv)
        .map_err(|e| format!("failed to write {path}: {e}"))?;

    println!("wrote {path}");

    Ok(())
}

// ============================================================================
// 6. DENSITY MATRIX / DEPOLARIZING NOISE
// ============================================================================

/// Applies increasing depolarizing noise to a Bell state and records:
///
///     purity
///     von Neumann entropy
///
/// This is a genuine mixed-state experiment.
///
/// At zero noise the Bell-state density matrix is pure.
///
/// Increasing noise should reduce purity and increase entropy.
fn noise_vs_purity() -> Result<(), String> {
    let path = format!("{OUTPUT_DIR}/noise_vs_purity.csv");

    let mut csv = String::from(
        "noise_probability,purity,von_neumann_entropy\n",
    );

    for step in 0usize..=20 {
        let probability =
            step as f64 / 20.0;

        let bell = create_bell_state()?;

        let mut density =
            DensityMatrix::from_state_vector(
                bell.get_state_vector(),
            )?;

        // Apply the same noise strength to both qubits.
        density.apply_depolarizing_channel(
            probability,
            0,
        )?;

        density.apply_depolarizing_channel(
            probability,
            1,
        )?;

        let purity =
            density.purity();

        let entropy =
            density.von_neumann_entropy();

        csv.push_str(&format!(
            "{probability:.6},{purity:.12},{entropy:.12}\n"
        ));
    }

    fs::write(&path, csv)
        .map_err(|e| format!("failed to write {path}: {e}"))?;

    println!("wrote {path}");

    Ok(())
}

// ============================================================================
// 7. BELL STATE DISTRIBUTION
// ============================================================================

/// Writes the exact probability distribution of the Bell state.
///
/// Expected:
///
///     00 -> 0.5
///     11 -> 0.5
///
/// and:
///
///     01 -> 0
///     10 -> 0
fn bell_distribution_data() -> Result<(), String> {
    let path =
        format!("{OUTPUT_DIR}/bell_distribution.csv");

    let bell =
        create_bell_state()?;

    let distribution =
        bell.get_probability_distribution();

    let mut csv =
        String::from("basis_state,probability\n");

    // Sort the basis states so the CSV is deterministic.
    let mut entries: Vec<_> =
        distribution.iter().collect();

    entries.sort_by(|a, b| a.0.cmp(b.0));

    for (state, probability) in entries {
        csv.push_str(&format!(
            "{state},{probability:.12}\n"
        ));
    }

    fs::write(&path, csv)
        .map_err(|e| format!("failed to write {path}: {e}"))?;

    println!("wrote {path}");

    Ok(())
}

// ============================================================================
// 8. SUMMARY
// ============================================================================

fn print_summary() {
    println!();
    println!("=== Dataset generation complete ===");
    println!();
    println!("Output directory:");
    println!("  {OUTPUT_DIR}");
    println!();
    println!("Generated datasets:");
    println!("  1. state_vector_scaling.csv");
    println!("  2. qft_scaling.csv");
    println!("  3. fidelity_vs_depth.csv");
    println!("  4. ghz_entanglement.csv");
    println!("  5. w_state.csv");
    println!("  6. noise_vs_purity.csv");
    println!("  7. bell_distribution.csv");
    println!();
    println!(
        "Maximum supported state-vector size: {MAX_SUPPORTED_QUBITS} qubits"
    );
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), String> {
    println!(
        "=== Sirraya QuTub experimental dataset generator ==="
    );
    println!();
    println!(
        "Configured maximum: {MAX_SUPPORTED_QUBITS} qubits"
    );
    println!();

    ensure_output_directory()?;

    println!("Generating state-vector scaling...");
    benchmark_state_vector_scaling()?;

    println!("Generating QFT scaling...");
    benchmark_qft_scaling()?;

    println!("Generating fidelity-vs-depth experiment...");
    fidelity_vs_depth()?;

    println!("Generating GHZ entanglement data...");
    ghz_entanglement_data()?;

    println!("Generating W-state data...");
    w_state_data()?;

    println!("Generating density-matrix noise data...");
    noise_vs_purity()?;

    println!("Generating Bell-state distribution...");
    bell_distribution_data()?;

    print_summary();

    Ok(())
}