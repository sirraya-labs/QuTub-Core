//! USP demo: Grover's search (3-qubit toy instance) run twice -- once as
//! an ideal, noiseless state-vector simulation, and once under a
//! per-gate depolarizing noise model calibrated to a real, currently-
//! published hardware fidelity figure (Quantinuum's Helios trapped-ion
//! system, as benchmarked by Sandia National Laboratories and reported
//! in Nature, June 2026 -- see `HardwareCalibration::quantinuum_helios_2026()`
//! in the `xeb` module).
//!
//! Same pairing as `shor_n15_noisy.rs`: a real algorithm (here, Grover's
//! search rather than Shor's factoring) run through a Monte Carlo
//! (quantum-trajectory) unraveling of the depolarizing channel, injected
//! directly into the fast state-vector simulator after every gate, so it
//! stays cheap even at hundreds of shots. Single-qubit gates (H, X) use
//! the calibration's single-qubit error rate; the multi-controlled-Z
//! gate used in both the oracle and the diffusion operator touches all
//! `NUM_QUBITS` qubits at once, so -- consistent with how
//! `shor_n15_noisy.rs` treats CSWAP (a genuinely multi-qubit gate) --
//! it is charged the two-qubit error rate independently on each qubit
//! it acts on.
//!
//! Grover's algorithm is a good complementary noise demo to Shor's: it
//! has an *exact*, closed-form ideal success probability (see
//! `grover_search.rs`) and a shallow, easily-scaled circuit, so the gap
//! between the theoretical ideal rate and the noisy Monte Carlo rate is
//! a clean, interpretable number -- rather than something that has to be
//! inferred indirectly from a factor-recovery histogram.

use sirraya_qutub::core::QuantumRegister;
use sirraya_qutub::xeb::HardwareCalibration;
use rand::Rng;
use std::env;
use std::f64::consts::PI;

const NUM_QUBITS: usize = 3;
const MARKED_INDEX: usize = 5; // |101>, same fixed target as grover_search.rs

#[derive(Clone, Copy)]
struct NoiseModel {
    p1: f64, // single-qubit gate error probability
    p2: f64, // two-qubit (and, here, multi-qubit-per-qubit) gate error probability
}

impl NoiseModel {
    fn ideal() -> Self {
        Self { p1: 0.0, p2: 0.0 }
    }
    fn from_calibration(c: &HardwareCalibration) -> Self {
        Self {
            p1: c.single_qubit_error_probability(),
            p2: c.two_qubit_error_probability(),
        }
    }
}

fn maybe_pauli_error(
    reg: &mut QuantumRegister,
    qubits: &[usize],
    p: f64,
    rng: &mut impl Rng,
) -> Result<(), String> {
    if p <= 0.0 {
        return Ok(());
    }
    for &q in qubits {
        if rng.gen::<f64>() < p {
            match rng.gen_range(0..3) {
                0 => reg.apply_pauli_x(q)?,
                1 => reg.apply_pauli_y(q)?,
                _ => reg.apply_pauli_z(q)?,
            }
        }
    }
    Ok(())
}

fn h(reg: &mut QuantumRegister, q: usize, nm: &NoiseModel, rng: &mut impl Rng) -> Result<(), String> {
    reg.apply_hadamard(q)?;
    maybe_pauli_error(reg, &[q], nm.p1, rng)
}

fn x(reg: &mut QuantumRegister, q: usize, nm: &NoiseModel, rng: &mut impl Rng) -> Result<(), String> {
    reg.apply_pauli_x(q)?;
    maybe_pauli_error(reg, &[q], nm.p1, rng)
}

/// Multi-controlled Z across `controls` + `target`, with the two-qubit
/// error rate applied independently to every qubit it touches (same
/// convention `shor_n15_noisy.rs` uses for CSWAP).
fn mcz(
    reg: &mut QuantumRegister,
    controls: &[usize],
    target: usize,
    nm: &NoiseModel,
    rng: &mut impl Rng,
) -> Result<(), String> {
    reg.apply_multi_controlled_z(controls, target)?;
    let mut touched: Vec<usize> = controls.to_vec();
    touched.push(target);
    maybe_pauli_error(reg, &touched, nm.p2, rng)
}

fn oracle_mark_101_noisy(
    reg: &mut QuantumRegister,
    nm: &NoiseModel,
    rng: &mut impl Rng,
) -> Result<(), String> {
    let n = reg.num_qubits();
    for q in 0..n {
        if (MARKED_INDEX >> q) & 1 == 0 {
            x(reg, q, nm, rng)?;
        }
    }
    let controls: Vec<usize> = (0..n - 1).collect();
    mcz(reg, &controls, n - 1, nm, rng)?;
    for q in 0..n {
        if (MARKED_INDEX >> q) & 1 == 0 {
            x(reg, q, nm, rng)?;
        }
    }
    Ok(())
}

/// Diffusion operator: H^{⊗n} (2|0><0| - I) H^{⊗n}, built the same way
/// `QuantumAlgorithm::grover_iteration` builds it in `core.rs`, but with
/// noise injected after every gate.
fn diffusion_noisy(reg: &mut QuantumRegister, nm: &NoiseModel, rng: &mut impl Rng) -> Result<(), String> {
    let n = reg.num_qubits();
    for q in 0..n {
        h(reg, q, nm, rng)?;
        x(reg, q, nm, rng)?;
    }
    let controls: Vec<usize> = (0..n - 1).collect();
    mcz(reg, &controls, n - 1, nm, rng)?;
    for q in 0..n {
        x(reg, q, nm, rng)?;
        h(reg, q, nm, rng)?;
    }
    Ok(())
}

fn grover_iteration_noisy(reg: &mut QuantumRegister, nm: &NoiseModel, rng: &mut impl Rng) -> Result<(), String> {
    oracle_mark_101_noisy(reg, nm, rng)?;
    diffusion_noisy(reg, nm, rng)
}

fn optimal_iterations(n_items: usize, n_marked: usize) -> usize {
    let theta = ((n_marked as f64 / n_items as f64).sqrt()).asin();
    (((PI / (4.0 * theta)) - 0.5).round().max(1.0)) as usize
}

fn theoretical_success_probability(n_items: usize, n_marked: usize, k: usize) -> f64 {
    let theta = ((n_marked as f64 / n_items as f64).sqrt()).asin();
    ((2.0 * k as f64 + 1.0) * theta).sin().powi(2)
}

fn run_once(k: usize, nm: &NoiseModel, rng: &mut impl Rng) -> Result<usize, String> {
    let mut reg = QuantumRegister::new(NUM_QUBITS)?;
    for q in 0..NUM_QUBITS {
        h(&mut reg, q, nm, rng)?;
    }
    for _ in 0..k {
        grover_iteration_noisy(&mut reg, nm, rng)?;
    }
    let bits = reg.measure_all_qubits()?;
    let mut measured = 0usize;
    for &b in bits.iter() {
        measured = (measured << 1) | (b as usize);
    }
    Ok(measured)
}

fn success_rate(k: usize, nm: &NoiseModel, shots: usize, rng: &mut impl Rng) -> f64 {
    let mut successes = 0;
    for _ in 0..shots {
        if let Ok(measured) = run_once(k, nm, rng) {
            if measured == MARKED_INDEX {
                successes += 1;
            }
        }
    }
    successes as f64 / shots as f64
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let shots: usize = args.get(1).map(|s| s.parse().unwrap_or(500)).unwrap_or(500);
    let iterations_override: Option<usize> = args.get(2).and_then(|s| s.parse().ok());

    let dim = 1usize << NUM_QUBITS;
    let k = iterations_override.unwrap_or_else(|| optimal_iterations(dim, 1));

    println!("GROVER'S SEARCH UNDER REALISTIC HARDWARE-CALIBRATED NOISE");
    println!("===========================================================\n");
    println!(
        "Searching {} qubits ({} states) for |{:0width$b}>, {} Grover iterations, {} shots per condition.\n",
        NUM_QUBITS, dim, MARKED_INDEX, k, shots, width = NUM_QUBITS
    );

    let calibration = HardwareCalibration::quantinuum_helios_2026();
    println!("Noise calibration: {}", calibration.name);
    println!("  Single-qubit gate fidelity: {:.6}%", calibration.single_qubit_fidelity * 100.0);
    println!("  Two-qubit gate fidelity:    {:.6}%", calibration.two_qubit_fidelity * 100.0);
    println!("  -> single-qubit depolarizing p = {:.8}", calibration.single_qubit_error_probability());
    println!("  -> two-qubit depolarizing p    = {:.8}\n", calibration.two_qubit_error_probability());

    let mut rng = rand::thread_rng();

    let ideal_nm = NoiseModel::ideal();
    let ideal_rate = success_rate(k, &ideal_nm, shots, &mut rng);

    let noisy_nm = NoiseModel::from_calibration(&calibration);
    let noisy_rate = success_rate(k, &noisy_nm, shots, &mut rng);

    let theory = theoretical_success_probability(dim, 1, k);

    println!("Results:");
    println!("  Theoretical (closed-form) ideal success probability: {:>6.2}%", theory * 100.0);
    println!("  Simulated ideal (noiseless) success rate:             {:>6.2}%", ideal_rate * 100.0);
    println!("  Quantinuum-Helios-calibrated noisy success rate:      {:>6.2}%", noisy_rate * 100.0);
    println!(
        "  Absolute degradation from realistic gate noise: {:.2} percentage points",
        (ideal_rate - noisy_rate) * 100.0
    );

    println!(
        "\nCircuit depth here is small ({} qubits, {} Grover iterations, each with one {}-qubit",
        NUM_QUBITS, k, NUM_QUBITS
    );
    println!(
        "multi-controlled-Z in the oracle and one more in the diffusion operator), so even"
    );
    println!(
        "Helios-class fidelities leave this toy instance largely intact. As with the Shor's"
    );
    println!(
        "algorithm demo, note that at these very low error rates ({:.1e}-{:.1e} per gate),",
        calibration.single_qubit_error_probability(),
        calibration.two_qubit_error_probability()
    );
    println!(
        "the true degradation can be smaller than shot noise at a few hundred shots -- increase"
    );
    println!(
        "`shots` for a tighter estimate, or search larger qubit counts / deeper circuits to see"
    );
    println!("noise-induced degradation more clearly.");

    Ok(())
}