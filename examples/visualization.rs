//! Generates visualization data for the sirraya-qutub documentation.
//!
//! Run:
//!     cargo run --example visualization
//!
//! Outputs CSV files in `data/` ready for plotting with the companion
//! `scripts/plot.py` script. Top labs use Python for final figures —
//! we follow that convention: compute in Rust, plot in Python.

use sirraya_qutub::{
    create_bell_state, create_ghz_state, create_w_state, QuantumBenchmark, QuantumRegister,
};
use std::fs::{self, File};
use std::io::Write;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const QUBIT_COUNTS: &[usize] = &[2, 4, 6, 8, 10, 12, 14, 16];
const BENCHMARK_ITERATIONS: usize = 10;
const DEPOLARIZING_PROBABILITIES: &[f64] = &[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0];
const GATE_DEPTHS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    println!("sirraya-qutub visualization data generation\n");

    benchmark_scaling().expect("benchmark_scaling failed");
    fidelity_vs_depth().expect("fidelity_vs_depth failed");
    entanglement_scaling().expect("entanglement_scaling failed");
    noise_channel_curves().expect("noise_channel_curves failed");

    println!("Done. Data files saved to data/");
    println!("Run: python scripts/plot.py");
}

// ---------------------------------------------------------------------------
// 1. Runtime scaling benchmark
// ---------------------------------------------------------------------------

/// Measures wall-clock time for Hadamard chain, CNOT chain, QFT, and
/// Bell-state preparation across qubit counts from 2 up to 16.
///
/// Output: `data/scaling.csv`
fn benchmark_scaling() -> std::io::Result<()> {
    fs::create_dir_all("data")?;
    let mut f = File::create("data/scaling.csv")?;
    writeln!(f, "qubits,dimension,hadamard_us,cnot_us,qft_us,bell_us")?;

    for &num_qubits in QUBIT_COUNTS {
        let dimension = 1usize << num_qubits;

        let h_time = QuantumBenchmark::benchmark_hadamard_chain(num_qubits, BENCHMARK_ITERATIONS);
        let cnot_time = QuantumBenchmark::benchmark_cnot_chain(num_qubits, BENCHMARK_ITERATIONS);
        let qft_time = QuantumBenchmark::benchmark_qft(num_qubits, BENCHMARK_ITERATIONS);

        let bell_time = {
            let mut total = Duration::new(0, 0);
            let num_pairs = num_qubits / 2;
            for _ in 0..BENCHMARK_ITERATIONS {
                let start = Instant::now();
                for _ in 0..num_pairs {
                    let _ = create_bell_state();
                }
                total += start.elapsed();
            }
            total / BENCHMARK_ITERATIONS as u32
        };

        let hadamard_us = h_time.as_micros() as f64;
        let cnot_us = cnot_time.as_micros() as f64;
        let qft_us = qft_time.as_micros() as f64;
        let bell_us = bell_time.as_micros() as f64;

        writeln!(
            f,
            "{num_qubits},{dimension},{hadamard_us},{cnot_us},{qft_us},{bell_us}"
        )?;

        println!(
            "  {num_qubits:>2} qubits (dim={dimension:>6}): \
             H={hadamard_us:>10.1} µs, CNOT={cnot_us:>10.1} µs, \
             QFT={qft_us:>10.1} µs, Bell={bell_us:>10.1} µs"
        );
    }

    println!("  -> data/scaling.csv\n");
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Fidelity vs gate depth
// ---------------------------------------------------------------------------

/// Applies a random circuit of increasing depth to a multi-qubit register
/// and measures self-fidelity at each depth.
///
/// Output: `data/fidelity_vs_depth.csv`
fn fidelity_vs_depth() -> std::io::Result<()> {
    let mut f = File::create("data/fidelity_vs_depth.csv")?;
    writeln!(f, "qubits,depth,fidelity")?;

    for &num_qubits in &[4, 6, 8] {
        for &depth in GATE_DEPTHS {
            let mut register =
                QuantumRegister::new_with_seed(num_qubits, 0xDEAD_BEEF)
                    .expect("register construction failed");

            for layer in 0..depth {
                for qubit in 0..num_qubits {
                    let angle = (layer as f64 * 1.7 + qubit as f64 * 2.3)
                        % (2.0 * std::f64::consts::PI);
                    register.apply_rx(qubit, angle).expect("RX gate failed");
                    register.apply_rz(qubit, angle * 0.5).expect("RZ gate failed");
                }
                for i in 0..num_qubits - 1 {
                    register.apply_cnot(i, i + 1).expect("CNOT gate failed");
                }
            }

            let original = register.clone();
            let fidelity = register.fidelity(&original).expect("fidelity failed");

            writeln!(f, "{num_qubits},{depth},{fidelity:.12}")?;
            println!("  {num_qubits} qubits, depth {depth:>3}: fidelity = {fidelity:.12}");
        }
    }

    println!("  -> data/fidelity_vs_depth.csv\n");
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Entanglement metrics vs qubit count
// ---------------------------------------------------------------------------

/// Computes entanglement measures (concurrence, negativity, von Neumann
/// entropy of reduced state) for GHZ and W states across qubit counts.
///
/// Output: `data/entanglement.csv`
fn entanglement_scaling() -> std::io::Result<()> {
    let mut f = File::create("data/entanglement.csv")?;
    writeln!(
        f,
        "qubits,state,concurrence,negativity,von_neumann_entropy"
    )?;

    for &num_qubits in &[2, 3, 4, 5, 6] {
        // GHZ state
        let ghz = create_ghz_state(num_qubits).expect("GHZ construction failed");
        let ghz_density = ghz.to_density_matrix().expect("density conversion failed");

        // W state
        let w = create_w_state(num_qubits).expect("W construction failed");
        let w_density = w.to_density_matrix().expect("density conversion failed");

        // Concurrence (only defined for 2-qubit systems)
        let concurrence = if num_qubits == 2 {
            ghz_density.concurrence().unwrap_or(-1.0)
        } else {
            -1.0
        };

        // Negativity across the bipartition (qubit 0 vs rest)
        let negativity = ghz_density.negativity(&[0]).unwrap_or(-1.0);

        // Von Neumann entropy of reduced single-qubit state
        let entropy_ghz = ghz_density
            .partial_trace(&[0])
            .map(|r| r.von_neumann_entropy())
            .unwrap_or(-1.0);

        let entropy_w = w_density
            .partial_trace(&[0])
            .map(|r| r.von_neumann_entropy())
            .unwrap_or(-1.0);

        writeln!(
            f,
            "{num_qubits},GHZ,{concurrence:.12},{negativity:.12},{entropy_ghz:.12}"
        )?;
        writeln!(
            f,
            "{num_qubits},W,,,{entropy_w:.12}"
        )?;

        println!(
            "  {num_qubits} qubits: GHZ negativity={negativity:.6}, \
             GHZ S_vn={entropy_ghz:.6}, W S_vn={entropy_w:.6}"
        );
    }

    println!("  -> data/entanglement.csv\n");
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. Noise channel action
// ---------------------------------------------------------------------------

/// Applies depolarizing and amplitude-damping channels at increasing
/// strength to a Bell state and tracks purity, fidelity, and trace.
///
/// Output: `data/noise.csv`
fn noise_channel_curves() -> std::io::Result<()> {
    let mut f = File::create("data/noise.csv")?;
    writeln!(f, "probability,channel,purity,fidelity,trace")?;

    // Depolarizing channel on a Bell state
    for &probability in DEPOLARIZING_PROBABILITIES {
        let bell = create_bell_state().expect("Bell construction failed");
        let bell_ref = bell.to_density_matrix().expect("density conversion failed");
        let mut density = bell.to_density_matrix().expect("density conversion failed");

        density
            .apply_depolarizing_channel(probability, 0)
            .expect("depolarizing channel failed");

        let purity = density.purity();
        let fidelity = density.fidelity(&bell_ref).unwrap_or(0.0);
        let trace = density.trace();

        writeln!(
            f,
            "{probability},depolarizing,{purity:.12},{fidelity:.12},{trace:.12}"
        )?;

        println!(
            "  depolarizing(p={probability:.2}): purity={purity:.6}, \
             fidelity={fidelity:.6}, trace={trace:.6}"
        );
    }

    // Amplitude damping channel on |1>
    for &gamma in &[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0] {
        let mut register = QuantumRegister::new(1).expect("register construction failed");
        register.apply_pauli_x(0).expect("X gate failed");

        let reference = register.to_density_matrix().expect("density conversion failed");
        let mut density = reference.clone();

        density
            .apply_amplitude_damping(gamma, 0)
            .expect("amplitude damping failed");

        let purity = density.purity();
        let fidelity = density.fidelity(&reference).unwrap_or(0.0);
        let trace = density.trace();

        writeln!(
            f,
            "{gamma},amplitude_damping,{purity:.12},{fidelity:.12},{trace:.12}"
        )?;

        println!(
            "  amplitude damping(γ={gamma:.2}): purity={purity:.6}, \
             fidelity={fidelity:.6}, trace={trace:.6}"
        );
    }

    println!("  -> data/noise.csv\n");
    Ok(())
}