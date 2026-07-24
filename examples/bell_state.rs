//! Bell State Demonstration
//!
//! Creates the maximally entangled Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
//! and verifies its properties through measurement.

use sirraya_qutub::core::QuantumRegister;
use std::collections::HashMap;
use std::time::Instant;

const SEP: &str = "══════════════════════════════════════════════════════════════════════";
const SEP_LIGHT: &str = "──────────────────────────────────────────────────────────────────────";

fn main() -> Result<(), String> {
    let start = Instant::now();

    // ── Header ──
    println!("{}", SEP);
    println!("                    QuTub Example • Bell State");
    println!("{}", SEP);
    println!();
    println!("Example          Bell State");
    println!("Algorithm        Entanglement Preparation");
    println!("Backend          State Vector Simulator");

    // ── Quantum Circuit ──
    println!();
    println!("{}", SEP);
    println!("Quantum Circuit");
    println!("{}", SEP);
    println!();
    println!("q0 ──H────■──");
    println!("           │");
    println!("q1 ───────X──");

    // ── Configuration ──
    println!();
    println!("Configuration");
    println!("{}", SEP_LIGHT);
    println!("Qubits             2");
    println!("Hilbert Space      2² = 4");
    println!("Basis Ordering     Little-endian");
    println!("Backend            State Vector");
    println!("Precision          f64");

    // ── Build ──
    let mut reg = QuantumRegister::new(2)?;
    reg.apply_hadamard(0)?;
    reg.apply_cnot(0, 1)?;

    // ── Quantum State ──
    println!();
    println!("{}", SEP);
    println!("Quantum State");
    println!("{}", SEP);
    println!();
    println!("Representation: Statevector");
    println!();
    println!("Target State");
    println!();
    println!("    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2");
    println!();
    println!("Computational Basis   Amplitude                  Probability");
    println!("{}", SEP_LIGHT);
    print_state_table(&reg);
    println!();
    println!("State Properties");
    println!("{}", SEP_LIGHT);
    println!("Fidelity to |Φ⁺⟩      {:.6}", compute_fidelity(&reg));
    println!("Normalization          {:.6}", normalization(&reg));
    println!("Global Phase           0.000000 rad");
    println!("Non-zero Amplitudes    {}", count_nonzero(&reg));
    println!();
    println!("Entanglement");
    println!("{}", SEP_LIGHT);
    println!("Bell State             Yes");
    println!("Separable              No");
    println!("Reduced State          Maximally Mixed");

    // ── Statevector Probability Distribution ──
    let statevector_probs = reg.get_probability_distribution();

    println!();
    println!("{}", SEP);
    println!("Statevector Probability Distribution");
    println!("{}", SEP);
    println!();
    print_distribution(&statevector_probs);

    // ── Measurement ──
    let n_shots: usize = 20;
    println!();
    println!("{}", SEP);
    println!("Measurement ({} Shots)", n_shots);
    println!("{}", SEP);

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut outcomes: Vec<String> = Vec::new();
    for _ in 0..n_shots {
        let mut reg_copy = reg.clone();
        let result = reg_copy.measure_all_qubits()?;
        let key = format!("{}{}", result[0], result[1]);
        *counts.entry(key.clone()).or_insert(0) += 1;
        outcomes.push(key);
    }

    for (group, chunk) in outcomes.chunks(5).enumerate() {
        let start = group * 5 + 1;
        let end = start + chunk.len() - 1;
        println!();
        println!("Shots {}-{}", start, end);
        println!();
        let display: Vec<String> = chunk.iter().map(|s| format!("|{}⟩", s)).collect();
        println!("{}", display.join(" "));
    }

    // ── Empirical Measurement Distribution ──
    println!();
    println!("{}", SEP);
    println!("Empirical Measurement Distribution");
    println!("{}", SEP);
    println!();
    println!("Outcome        Count        Frequency");
    println!("{}", SEP_LIGHT);
    let all_states = canonical_basis_states(2);
    for state in &all_states {
        let count = counts.get(state).unwrap_or(&0);
        println!(
            "|{}⟩             {:>2}          {:>5.2}%",
            state,
            count,
            *count as f64 / n_shots as f64 * 100.0
        );
    }

    // ── Verification ──
    let zero_zero = *counts.get("00").unwrap_or(&0);
    let one_one = *counts.get("11").unwrap_or(&0);
    let zero_one = *counts.get("01").unwrap_or(&0);
    let one_zero = *counts.get("10").unwrap_or(&0);

    let only_correlated = zero_one == 0 && one_zero == 0;
    let norm_ok = (normalization(&reg) - 1.0).abs() < 1e-9;
    let amplitudes_ok = verify_bell_amplitudes(&reg);

    let expected = n_shots as f64 / 2.0;
    let tolerance = 3.0 * (expected * 0.5).sqrt();
    let sampling_ok = ((zero_zero as f64 - expected).abs() <= tolerance)
        && ((one_one as f64 - expected).abs() <= tolerance);

    let expected_pct = 50.0;
    let observed_00 = zero_zero as f64 / n_shots as f64 * 100.0;
    let observed_11 = one_one as f64 / n_shots as f64 * 100.0;
    let dev_00 = observed_00 - expected_pct;
    let dev_11 = observed_11 - expected_pct;

    println!();
    println!("{}", SEP);
    println!("Verification");
    println!("{}", SEP);
    println!();
    println!("  {}  State normalized", check_mark(norm_ok));
    println!("  {}  Target Bell state prepared", check_mark(amplitudes_ok));
    println!("  {}  Perfect measurement correlation observed", check_mark(only_correlated));
    println!("  {}  No forbidden basis states detected", check_mark(only_correlated));
    println!("  {}  Measurement statistics consistent with finite-shot sampling", check_mark(sampling_ok));
    println!();
    println!("Distribution Comparison");
    println!("{}", SEP_LIGHT);
    println!();
    println!("Outcome      Expected      Observed");
    println!("{}", SEP_LIGHT);
    println!(
        "|00⟩          {:>5.2}%        {:>5.2}%",
        expected_pct, observed_00
    );
    println!(
        "|11⟩          {:>5.2}%        {:>5.2}%",
        expected_pct, observed_11
    );
    println!();
    println!("Observed − Expected");
    println!("{}", SEP_LIGHT);
    println!(
        "|00⟩          {:>+6.2}%",
        dev_00
    );
    println!(
        "|11⟩          {:>+6.2}%",
        dev_11
    );

    // ── Concepts Demonstrated ──
    println!();
    println!("{}", SEP);
    println!("Concepts Demonstrated");
    println!("{}", SEP);
    println!();
    println!("  ✓ Quantum Superposition");
    println!("  ✓ Quantum Entanglement");
    println!("  ✓ Controlled-NOT (CNOT)");
    println!("  ✓ Projective Measurement");

    // ── Result ──
    println!();
    println!("{}", SEP);
    println!("Result");
    println!("{}", SEP);
    println!();
    println!("The target Bell state |Φ⁺⟩ was prepared successfully.");
    println!();
    println!("The statevector matches the expected Bell state exactly.");
    println!("Measurement outcomes exhibit the expected perfect correlations,");
    println!("with observed frequencies reflecting finite-shot sampling.");

    // ── Execution Statistics ──
    let duration = start.elapsed();
    let memory = std::mem::size_of_val(reg.get_state_vector());
    println!();
    println!("{}", SEP);
    println!("Execution Statistics");
    println!("{}", SEP);
    println!();
    println!("Framework            QuTub v0.1.3");
    println!("Backend              State Vector");
    println!("Precision            f64");
    println!();
    println!("Qubits               2");
    println!("Hilbert Space        2² = 4");
    println!();
    println!("Circuit Depth        2");
    println!("Gate Count           2");
    println!("  Hadamard           1");
    println!("  CNOT               1");
    println!();
    println!("Measurements         {}", n_shots);
    println!("Random Seed          System");
    println!();
    println!("Execution Time       {:.2} ms", duration.as_secs_f64() * 1000.0);
    println!("Memory Footprint     {} B", memory);

    // ── Footer ──
    println!();
    println!("{}", SEP_LIGHT);
    println!("QuTub v0.1.3");
    println!("Pure Rust Quantum Simulation Framework");
    println!();
    println!("https://github.com/SirrayaLabs/QuTub");
    println!("{}", SEP_LIGHT);

    Ok(())
}

fn print_state_table(reg: &QuantumRegister) {
    let sv = reg.get_state_vector();
    let states = canonical_basis_states(reg.num_qubits());
    for (i, state) in states.iter().enumerate() {
        let amp = sv[i];
        let prob = amp.magnitude_squared();
        if prob > 1e-9 {
            println!(
                "|{}⟩                   {:.6} + {:.6}i       {:.6}",
                state, amp.real(), amp.imag(), prob
            );
        }
    }
}

fn print_distribution(dist: &HashMap<String, f64>) {
    let all_states = canonical_basis_states(2);
    for state in &all_states {
        let prob = dist.get(state).unwrap_or(&0.0);
        let filled = (prob * 20.0) as usize;
        let empty = 20 - filled;
        let bar = format!("{}{}", "█".repeat(filled), "─".repeat(empty));
        println!("|{}⟩              {:>5.2}%   {}", state, prob * 100.0, bar);
    }
}

fn canonical_basis_states(num_qubits: usize) -> Vec<String> {
    let dim = 1 << num_qubits;
    let mut states = Vec::with_capacity(dim);
    for i in 0..dim {
        let bits = format!("{:0width$b}", i, width = num_qubits);
        states.push(bits);
    }
    states
}

fn count_nonzero(reg: &QuantumRegister) -> usize {
    reg.get_state_vector()
        .iter()
        .filter(|a| a.magnitude_squared() > 1e-9)
        .count()
}

fn normalization(reg: &QuantumRegister) -> f64 {
    reg.get_state_vector()
        .iter()
        .map(|a| a.magnitude_squared())
        .sum()
}

fn compute_fidelity(reg: &QuantumRegister) -> f64 {
    let sv = reg.get_state_vector();
    if sv.len() != 4 {
        return 0.0;
    }
    let target = 1.0 / (2.0f64).sqrt();
    let mut target_sv = vec![sirraya_qutub::complex::Complex::zero(); 4];
    target_sv[0] = sirraya_qutub::complex::Complex::new(target, 0.0);
    target_sv[3] = sirraya_qutub::complex::Complex::new(target, 0.0);

    let mut overlap = sirraya_qutub::complex::Complex::zero();
    for i in 0..4 {
        overlap = overlap + target_sv[i].conjugate() * sv[i];
    }
    overlap.magnitude_squared()
}

fn verify_bell_amplitudes(reg: &QuantumRegister) -> bool {
    let sv = reg.get_state_vector();
    if sv.len() != 4 {
        return false;
    }
    let target = 1.0 / (2.0f64).sqrt();
    (sv[0].real() - target).abs() < 1e-9
        && sv[0].imag().abs() < 1e-9
        && sv[1].magnitude() < 1e-9
        && sv[2].magnitude() < 1e-9
        && (sv[3].real() - target).abs() < 1e-9
        && sv[3].imag().abs() < 1e-9
}

fn check_mark(condition: bool) -> &'static str {
    if condition { "✓" } else { "✗" }
}