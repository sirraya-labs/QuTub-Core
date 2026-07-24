//! Shor's algorithm, toy instance: factor N = 15 using base a = 7.
//!
//! N = 15 is the standard "hello world" instance for Shor's algorithm
//! (as used in the original Qiskit textbook demonstration) because its
//! order-finding subproblem is small enough to lay out explicitly by
//! hand, while still exercising every structural piece of the real
//! algorithm: a counting register, quantum phase estimation via QFT,
//! a genuinely *quantum* controlled modular-exponentiation circuit, and
//! classical post-processing (continued fractions) to recover the
//! period and then the factors.
//!
//! Register layout (qubits 0..num_qubits-1):
//!   - qubits 0..3   : the 4-qubit "target" register, holds values mod 15
//!   - qubits 4..    : the "counting" register (n_count qubits), which
//!                     after inverse-QFT is measured to estimate the
//!                     phase s/r, where r is the order of a mod N.
//!
//! Modular exponentiation for a = 7, N = 15 is implemented directly as a
//! permutation circuit (SWAPs + X gates), controlled by the relevant
//! counting qubit -- this is the well-known decomposition for this
//! specific (a, N) pair (multiplication by 7 mod 15 factors into three
//! 4-cycles on {1,7,4,13}, {2,14,8,11}, {3,6,12,9} and three fixed
//! points {0,5,10}), rather than a generic modular-exponentiation
//! circuit, which is only worth building for N this small as a teaching
//! example.

use sirraya_qutub::core::QuantumRegister;
use std::collections::HashMap;
use std::env;

const N: u64 = 15;
const A: u64 = 7; // coprime to 15; order of 7 mod 15 is 4

/// Controlled multiplication-by-7^power (mod 15) on the 4-qubit target
/// register (qubits `t0..t0+3`), controlled by qubit `control`.
///
/// The base circuit for a single multiplication by 7 mod 15 is:
///   CSWAP(control, t0,   t0+1)
///   CSWAP(control, t0+1, t0+2)
///   CSWAP(control, t0+2, t0+3)
///   CNOT (control, t0) ; CNOT(control, t0+1) ; CNOT(control, t0+2) ; CNOT(control, t0+3)
///
/// Repeating this block `power` times implements multiplication by
/// 7^power mod 15 (7 has order 4 mod 15, so power is effectively taken
/// mod 4 by the group structure itself -- e.g. power=4 reduces to the
/// identity automatically).
fn controlled_mult_7_mod_15(
    reg: &mut QuantumRegister,
    control: usize,
    t0: usize,
    power: u64,
) -> Result<(), String> {
    for _ in 0..power {
        reg.apply_cswap(control, t0, t0 + 1)?;
        reg.apply_cswap(control, t0 + 1, t0 + 2)?;
        reg.apply_cswap(control, t0 + 2, t0 + 3)?;
        reg.apply_cnot(control, t0)?;
        reg.apply_cnot(control, t0 + 1)?;
        reg.apply_cnot(control, t0 + 2)?;
        reg.apply_cnot(control, t0 + 3)?;
    }
    Ok(())
}

/// Continued-fraction expansion of x/y, used to recover the most likely
/// true fraction s/r (with r <= N) from the noisy phase estimate we
/// measure off the counting register.
fn continued_fraction_convergents(mut num: u64, mut den: u64, max_den: u64) -> Vec<(u64, u64)> {
    let mut convergents = Vec::new();
    // p_{-2}=0, p_{-1}=1, q_{-2}=1, q_{-1}=0 (standard continued-fraction
    // convergent recurrence: p_i = a_i*p_{i-1}+p_{i-2}, likewise for q).
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
    convergents
}

fn bit_reverse(mut v: u64, n: usize) -> u64 {
    let mut r = 0u64;
    for _ in 0..n {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
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

/// Run one shot of the quantum order-finding subroutine and return the
/// measured counting-register value (as an integer) together with the
/// number of counting qubits used.
fn run_order_finding_circuit(n_count: usize) -> Result<u64, String> {
    let total_qubits = n_count + 4;
    let mut reg = QuantumRegister::new(total_qubits)?;

    // Target register (qubits 0..3) starts in |0001> = |1>, the standard
    // starting point for order finding.
    reg.apply_pauli_x(0)?;

    // Counting register (qubits 4..4+n_count) starts in uniform
    // superposition.
    for c in 0..n_count {
        reg.apply_hadamard(4 + c)?;
    }

    // Controlled-U^(2^j) for each counting qubit j, U = multiply-by-7 mod 15.
    //
    // Power assignment note: this crate's `quantum_fourier_transform` /
    // `inverse_quantum_fourier_transform` bakes a qubit-order SWAP into
    // the QFT itself (see core.rs), which flips the usual "qubit j
    // controls U^(2^j)" convention for the *raw* measured bit-string.
    // Empirically (verified against known synthetic phases before wiring
    // this up): the counting qubit *closest* to the target register
    // (control = 4 + 0) must carry the *largest* power, and the result
    // read directly off `measure_all_qubits` comes out bit-reversed
    // relative to the true phase numerator -- so we un-reverse it below.
    for j in 0..n_count {
        let power = 1u64 << (n_count - 1 - j);
        controlled_mult_7_mod_15(&mut reg, 4 + j, 0, power)?;
    }

    // Inverse QFT on the counting register to convert phase kickback into
    // a directly measurable estimate of s/r.
    //
    // `inverse_quantum_fourier_transform` operates on a register's qubits
    // 0..n as a whole; we apply the identical gate sequence restricted to
    // our counting-qubit sub-range instead (no sub-register API exists).
    apply_inverse_qft_on_range(&mut reg, 4, n_count)?;

    let bits = reg.measure_all_qubits()?;
    // bits is MSB-first over the *whole* register (qubit total_qubits-1
    // first). The counting register occupies qubits 4..4+n_count, i.e.
    // the leading `n_count` bits of `bits`.
    let mut raw: u64 = 0;
    for b in bits.iter().take(n_count) {
        raw = (raw << 1) | (*b as u64);
    }
    Ok(bit_reverse(raw, n_count))
}

/// Same inverse-QFT construction as `inverse_quantum_fourier_transform`
/// in the core module, but restricted to a qubit sub-range
/// [offset, offset+len), so it can be applied to just the counting
/// register of a larger circuit.
fn apply_inverse_qft_on_range(reg: &mut QuantumRegister, offset: usize, len: usize) -> Result<(), String> {
    use std::f64::consts::PI;

    for i in 0..len / 2 {
        reg.apply_swap(offset + i, offset + len - 1 - i)?;
    }
    for i in (0..len).rev() {
        for j in (i + 1..len).rev() {
            let angle = -2.0 * PI / (1u64 << (j - i + 1)) as f64;
            reg.apply_controlled_phase(offset + j, offset + i, angle)?;
        }
        reg.apply_hadamard(offset + i)?;
    }
    Ok(())
}

fn try_recover_factors(n_count: usize, measured: u64) -> Option<(u64, u64, u64, u64)> {
    let denom = 1u64 << n_count;
    if measured == 0 {
        return None;
    }
    let convergents = continued_fraction_convergents(measured, denom, N);
    for (_, r) in convergents {
        if r == 0 {
            continue;
        }
        if mod_pow(A, r, N) == 1 {
            // Candidate order found; try to extract factors from it.
            if r % 2 == 0 {
                let x = mod_pow(A, r / 2, N);
                if x != N - 1 {
                    let f1 = gcd(x + 1, N);
                    let f2 = gcd((x + N - 1) % N, N);
                    if f1 != 1 && f1 != N {
                        return Some((f1, N / f1, r, measured));
                    }
                    if f2 != 1 && f2 != N {
                        return Some((f2, N / f2, r, measured));
                    }
                }
            }
        }
    }
    None
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let n_count: usize = args.get(1).map(|s| s.parse().unwrap_or(3)).unwrap_or(3);
    let shots: usize = args.get(2).map(|s| s.parse().unwrap_or(20)).unwrap_or(20);

    println!("SHOR'S ALGORITHM -- TOY INSTANCE: FACTOR N = {}", N);
    println!("=========================================================\n");
    println!("Base a = {} (coprime to N; true order of 7 mod 15 is 4)", A);
    println!("Counting qubits: {}  |  Target qubits: 4  |  Total: {}", n_count, n_count + 4);
    println!("(sirraya-qutub MAX_QUBITS = 16, so up to {} counting qubits fit)\n", 16 - 4);

    let mut histogram: HashMap<u64, usize> = HashMap::new();
    let mut successes = 0usize;
    let mut example_factors: Option<(u64, u64)> = None;

    for _ in 0..shots {
        let measured = run_order_finding_circuit(n_count)?;
        *histogram.entry(measured).or_insert(0) += 1;
        if let Some((f1, f2, r, _)) = try_recover_factors(n_count, measured) {
            successes += 1;
            example_factors.get_or_insert((f1, f2));
            let _ = r;
        }
    }

    println!("Measured counting-register histogram over {} shots:", shots);
    let denom = 1u64 << n_count;
    let mut keys: Vec<&u64> = histogram.keys().collect();
    keys.sort();
    for k in keys {
        let count = histogram[k];
        println!(
            "  {:>3} / {:<3}  (phase ~ {:.4})  -> seen {:>2} times{}",
            k,
            denom,
            *k as f64 / denom as f64,
            count,
            match try_recover_factors(n_count, *k) {
                Some((f1, f2, r, _)) => format!("   [order r={r} -> factors {f1} x {f2} = {}]", f1 * f2),
                None => String::new(),
            }
        );
    }

    println!(
        "\nFactor recovery succeeded on {}/{} shots ({:.1}%).",
        successes,
        shots,
        100.0 * successes as f64 / shots as f64
    );
    if let Some((f1, f2)) = example_factors {
        println!("Recovered factorization: {} = {} x {}", N, f1, f2);
    } else {
        println!(
            "No successful factor recovery this run -- try more shots or n_count, \
             or note that even ideally, order finding succeeds only on a subset of phases."
        );
    }

    Ok(())
}
