//! USP demo: Shor's algorithm (N=15) run twice -- once as an ideal,
//! noiseless state-vector simulation, and once under a per-gate
//! depolarizing noise model calibrated to a real, currently-published
//! hardware fidelity figure (Quantinuum's Helios trapped-ion system, as
//! benchmarked by Sandia National Laboratories and reported in Nature,
//! June 2026 -- see `HardwareCalibration::quantinuum_helios_2026()` in
//! the `xeb` module).
//!
//! Most toy quantum-algorithm demos stop at "ideal state-vector
//! simulation." What differentiates sirraya-qutub is that the same
//! codebase that runs the algorithm also carries a noise model grounded
//! in a real, cited hardware fidelity number (not an arbitrary slider),
//! and validated methodology (linear XEB, see `xeb.rs`) to check that
//! noise model against a classical reference distribution. This demo
//! puts those two things to work together: it shows *quantitatively*
//! how much a realistic device's error rates degrade Shor's algorithm's
//! success probability relative to the ideal case, using a Monte Carlo
//! (quantum-trajectory) unraveling of the depolarizing channel injected
//! directly into the fast state-vector simulator, so it stays cheap even
//! at hundreds of shots.
//!
//! Every gate in the circuit is followed by an independent, calibrated
//! chance of a random Pauli error on the qubit(s) it touched -- single-
//! qubit gates use the calibration's single-qubit error rate, multi-
//! qubit gates (CNOT, CSWAP, SWAP) use its two-qubit error rate on each
//! qubit involved. Averaged over many trajectories this reproduces the
//! same depolarizing statistics as `DensityMatrix::apply_depolarizing_channel`,
//! without paying for a full density-matrix simulation.

use sirraya_qutub::core::QuantumRegister;
use sirraya_qutub::xeb::HardwareCalibration;
use rand::Rng;
use std::env;
use std::f64::consts::PI;

const N: u64 = 15;
const A: u64 = 7;

#[derive(Clone, Copy)]
struct NoiseModel {
    p1: f64, // single-qubit gate error probability
    p2: f64, // two-qubit gate error probability
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

fn cswap(
    reg: &mut QuantumRegister,
    control: usize,
    t1: usize,
    t2: usize,
    nm: &NoiseModel,
    rng: &mut impl Rng,
) -> Result<(), String> {
    reg.apply_cswap(control, t1, t2)?;
    maybe_pauli_error(reg, &[control, t1, t2], nm.p2, rng)
}

fn cnot(
    reg: &mut QuantumRegister,
    control: usize,
    target: usize,
    nm: &NoiseModel,
    rng: &mut impl Rng,
) -> Result<(), String> {
    reg.apply_cnot(control, target)?;
    maybe_pauli_error(reg, &[control, target], nm.p2, rng)
}

fn cphase(
    reg: &mut QuantumRegister,
    control: usize,
    target: usize,
    angle: f64,
    nm: &NoiseModel,
    rng: &mut impl Rng,
) -> Result<(), String> {
    reg.apply_controlled_phase(control, target, angle)?;
    maybe_pauli_error(reg, &[control, target], nm.p2, rng)
}

fn swap(
    reg: &mut QuantumRegister,
    q1: usize,
    q2: usize,
    nm: &NoiseModel,
    rng: &mut impl Rng,
) -> Result<(), String> {
    reg.apply_swap(q1, q2)?;
    maybe_pauli_error(reg, &[q1, q2], nm.p2, rng)
}

fn controlled_mult_7_mod_15(
    reg: &mut QuantumRegister,
    control: usize,
    t0: usize,
    power: u64,
    nm: &NoiseModel,
    rng: &mut impl Rng,
) -> Result<(), String> {
    for _ in 0..power {
        cswap(reg, control, t0, t0 + 1, nm, rng)?;
        cswap(reg, control, t0 + 1, t0 + 2, nm, rng)?;
        cswap(reg, control, t0 + 2, t0 + 3, nm, rng)?;
        cnot(reg, control, t0, nm, rng)?;
        cnot(reg, control, t0 + 1, nm, rng)?;
        cnot(reg, control, t0 + 2, nm, rng)?;
        cnot(reg, control, t0 + 3, nm, rng)?;
    }
    Ok(())
}

fn apply_inverse_qft_on_range(
    reg: &mut QuantumRegister,
    offset: usize,
    len: usize,
    nm: &NoiseModel,
    rng: &mut impl Rng,
) -> Result<(), String> {
    for i in 0..len / 2 {
        swap(reg, offset + i, offset + len - 1 - i, nm, rng)?;
    }
    for i in (0..len).rev() {
        for j in (i + 1..len).rev() {
            let angle = -2.0 * PI / (1u64 << (j - i + 1)) as f64;
            cphase(reg, offset + j, offset + i, angle, nm, rng)?;
        }
        h(reg, offset + i, nm, rng)?;
    }
    Ok(())
}

fn bit_reverse(mut v: u64, n: usize) -> u64 {
    let mut r = 0u64;
    for _ in 0..n {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

fn run_order_finding_circuit(
    n_count: usize,
    nm: &NoiseModel,
    rng: &mut impl Rng,
) -> Result<u64, String> {
    let total_qubits = n_count + 4;
    let mut reg = QuantumRegister::new(total_qubits)?;

    x(&mut reg, 0, nm, rng)?; // target register starts at |1>

    for c in 0..n_count {
        h(&mut reg, 4 + c, nm, rng)?;
    }

    for j in 0..n_count {
        let power = 1u64 << (n_count - 1 - j);
        controlled_mult_7_mod_15(&mut reg, 4 + j, 0, power, nm, rng)?;
    }

    apply_inverse_qft_on_range(&mut reg, 4, n_count, nm, rng)?;

    let bits = reg.measure_all_qubits()?;
    let mut raw: u64 = 0;
    for b in bits.iter().take(n_count) {
        raw = (raw << 1) | (*b as u64);
    }
    Ok(bit_reverse(raw, n_count))
}

fn continued_fraction_convergents(mut num: u64, mut den: u64, max_den: u64) -> Vec<(u64, u64)> {
    let mut convergents = Vec::new();
    let (mut h_prev, mut h_curr) = (0u64, 1u64);
    let (mut k_prev, mut k_curr) = (1u64, 0u64);

    while den != 0 {
        let a = num / den;
        let (h_next, k_next) = (a * h_curr + h_prev, a * k_curr + k_prev);
        if k_next > max_den {
            break;
        }
        h_prev = h_curr;
        h_curr = h_next;
        k_prev = k_curr;
        k_curr = k_next;
        convergents.push((h_curr, k_curr));

        let new_den = num % den;
        num = den;
        den = new_den;
    }
    let _ = (h_prev, k_prev);
    convergents
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result
}

fn recovers_valid_factor(n_count: usize, measured: u64) -> bool {
    if measured == 0 {
        return false;
    }
    let denom = 1u64 << n_count;
    let convergents = continued_fraction_convergents(measured, denom, N);
    for (_, r) in convergents {
        if r != 0 && r % 2 == 0 && mod_pow(A, r, N) == 1 {
            let x = mod_pow(A, r / 2, N);
            if x != N - 1 {
                let f1 = gcd(x + 1, N);
                let f2 = gcd((x + N - 1) % N, N);
                if (f1 != 1 && f1 != N) || (f2 != 1 && f2 != N) {
                    return true;
                }
            }
        }
    }
    false
}

fn success_rate(n_count: usize, nm: &NoiseModel, shots: usize, rng: &mut impl Rng) -> f64 {
    let mut successes = 0;
    for _ in 0..shots {
        if let Ok(measured) = run_order_finding_circuit(n_count, nm, rng) {
            if recovers_valid_factor(n_count, measured) {
                successes += 1;
            }
        }
    }
    successes as f64 / shots as f64
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let n_count: usize = args.get(1).map(|s| s.parse().unwrap_or(3)).unwrap_or(3);
    let shots: usize = args.get(2).map(|s| s.parse().unwrap_or(500)).unwrap_or(500);

    println!("SHOR'S ALGORITHM UNDER REALISTIC HARDWARE-CALIBRATED NOISE");
    println!("===========================================================\n");
    println!("Factoring N = {} (a = {}), {} counting qubits, {} shots per condition.\n", N, A, n_count, shots);

    let calibration = HardwareCalibration::quantinuum_helios_2026();
    println!("Noise calibration: {}", calibration.name);
    println!("  Single-qubit gate fidelity: {:.6}%", calibration.single_qubit_fidelity * 100.0);
    println!("  Two-qubit gate fidelity:    {:.6}%", calibration.two_qubit_fidelity * 100.0);
    println!("  -> single-qubit depolarizing p = {:.8}", calibration.single_qubit_error_probability());
    println!("  -> two-qubit depolarizing p    = {:.8}\n", calibration.two_qubit_error_probability());

    let mut rng = rand::thread_rng();

    let ideal_nm = NoiseModel::ideal();
    let ideal_rate = success_rate(n_count, &ideal_nm, shots, &mut rng);

    let noisy_nm = NoiseModel::from_calibration(&calibration);
    let noisy_rate = success_rate(n_count, &noisy_nm, shots, &mut rng);

    println!("Results:");
    println!("  Ideal (noiseless) factor-recovery rate:   {:>6.2}%", ideal_rate * 100.0);
    println!("  Quantinuum-Helios-calibrated noisy rate:  {:>6.2}%", noisy_rate * 100.0);
    println!(
        "  Absolute degradation from realistic gate noise: {:.2} percentage points",
        (ideal_rate - noisy_rate) * 100.0
    );
    println!(
        "\nCircuit depth here is small ({} qubits, ~{} two-qubit gates), so even Helios-class",
        n_count + 4,
        // rough count for context, not used elsewhere
        (n_count * 7 * 3) + (n_count * (n_count - 1))
    );
    println!("fidelities leave the algorithm largely intact -- this becomes far more pronounced");
    println!("for larger N / deeper modular-exponentiation circuits, which is exactly the regime");
    println!("where hardware-grounded noise modeling (rather than an ideal-only simulator) matters.");

    Ok(())
}
