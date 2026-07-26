//! Sirraya QuTub — numerical validation suite.
//!
//! This is a correctness suite, not a performance benchmark.
//!
//! The checks compare QuTub's numerical output against:
//!   - analytically known quantum states,
//!   - unitary/involution identities,
//!   - Born-rule probabilities,
//!   - entanglement correlations,
//!   - expectation values,
//!   - QFT / inverse-QFT round trips,
//!   - GHZ and W state structure,
//!   - trace-distance / fidelity identities,
//!   - normalization invariants,
//!   - probability-distribution normalization,
//!   - deterministic seeded measurement behavior,
//!   - QASM export,
//!   - density-matrix conversion,
//!   - and controlled-phase behavior.
//!
//! Run:
//!
//!     cargo run --example validation
//!
//! Expected:
//!
//!     22/22 checks passed
//!
//! The validation suite intentionally uses the public API exposed by the
//! current sirraya-qutub core implementation.

use sirraya_qutub::{
    create_bell_state,
    create_ghz_state,
    create_w_state,
    inverse_quantum_fourier_transform,
    quantum_fourier_transform,
    Complex,
    QuantumRegister,
};

use std::collections::HashMap;
use std::f64::consts::PI;

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

const FIDELITY_TOLERANCE: f64 = 1e-9;
const NUMERICAL_TOLERANCE: f64 = 1e-9;

const STATISTICAL_SHOTS: usize = 10_000;

// Chi-square critical value for df=1, alpha=0.01.
const CHI_SQUARED_CRITICAL_1DF_ALPHA_01: f64 = 6.635;

// -----------------------------------------------------------------------------
// Result type
// -----------------------------------------------------------------------------

#[derive(Debug)]
pub struct ValidationResult {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

impl ValidationResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: true,
            detail: detail.into(),
        }
    }

    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            detail: detail.into(),
        }
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Convert a basis-state integer into QuTub's MSB-first bitstring format.
///
/// For example, for 3 qubits:
///
///     0 -> "000"
///     1 -> "001"
///     2 -> "010"
///     7 -> "111"
fn basis_state_label(num_qubits: usize, index: usize) -> String {
    format!("{:0width$b}", index, width = num_qubits)
}

/// Obtain a basis-state probability through QuTub's actual public API.
///
/// `get_state_probability` accepts a bitstring rather than an integer.
fn probability_of_state(register: &QuantumRegister, index: usize) -> f64 {
    let state = basis_state_label(register.num_qubits(), index);

    register
        .get_state_probability(&state)
        .unwrap_or(0.0)
}

/// Sum all probabilities in the register.
fn total_probability(register: &QuantumRegister) -> f64 {
    register
        .get_state_vector()
        .iter()
        .map(|amp| amp.magnitude_squared())
        .sum()
}

/// Maximum absolute difference between two state vectors.
fn max_state_vector_error(a: &QuantumRegister, b: &QuantumRegister) -> Result<f64, String> {
    if a.num_qubits() != b.num_qubits() {
        return Err("Registers have different numbers of qubits".to_string());
    }

    let mut max_error: f64 = 0.0;

    for (x, y) in a.get_state_vector().iter().zip(b.get_state_vector()) {
        let real_error = (x.real() - y.real()).abs();
        let imag_error = (x.imag() - y.imag()).abs();

        max_error = max_error.max(real_error);
        max_error = max_error.max(imag_error);
    }

    Ok(max_error)
}

/// Pearson chi-square statistic.
fn chi_squared_statistic(observed: &[f64], expected: &[f64]) -> f64 {
    observed
        .iter()
        .zip(expected.iter())
        .map(|(o, e)| {
            if *e == 0.0 {
                0.0
            } else {
                (o - e).powi(2) / e
            }
        })
        .sum()
}

/// Count a measurement outcome.
fn increment_count(counts: &mut HashMap<String, u32>, outcome: &[u8]) {
    let key = format!("{outcome:?}");
    *counts.entry(key).or_insert(0) += 1;
}

/// Check that all amplitudes are finite.
fn state_is_finite(register: &QuantumRegister) -> bool {
    register
        .get_state_vector()
        .iter()
        .all(|amp| amp.real().is_finite() && amp.imag().is_finite())
}

// -----------------------------------------------------------------------------
// 1. State normalization
// -----------------------------------------------------------------------------

pub fn validate_state_normalization() -> ValidationResult {
    let bell = match create_bell_state() {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "state_normalization",
                format!("Bell-state construction failed: {e}"),
            )
        }
    };

    let probability = total_probability(&bell);
    let error = (probability - 1.0).abs();

    if error < NUMERICAL_TOLERANCE {
        ValidationResult::pass(
            "state_normalization",
            format!(
                "sum |amplitude|² = {probability:.12} (|err| = {error:.2e})"
            ),
        )
    } else {
        ValidationResult::fail(
            "state_normalization",
            format!(
                "sum |amplitude|² = {probability:.12}, expected 1.0 (|err| = {error:.2e})"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 2. Bell self fidelity
// -----------------------------------------------------------------------------

pub fn validate_self_fidelity_bell() -> ValidationResult {
    let bell = match create_bell_state() {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "self_fidelity_bell",
                format!("construction failed: {e}"),
            )
        }
    };

    let fidelity = match bell.fidelity(&bell) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "self_fidelity_bell",
                format!("fidelity() failed: {e}"),
            )
        }
    };

    let error = (fidelity - 1.0).abs();

    if error < FIDELITY_TOLERANCE {
        ValidationResult::pass(
            "self_fidelity_bell",
            format!(
                "F(bell, bell) = {fidelity:.12} (|err| = {error:.2e})"
            ),
        )
    } else {
        ValidationResult::fail(
            "self_fidelity_bell",
            format!(
                "F(bell, bell) = {fidelity:.12}, expected 1.0 (|err| = {error:.2e})"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 3. Bell state reproducibility
// -----------------------------------------------------------------------------

pub fn validate_bell_state_fidelity() -> ValidationResult {
    let bell_a = match create_bell_state() {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "bell_state_fidelity",
                format!("first Bell state failed: {e}"),
            )
        }
    };

    let bell_b = match create_bell_state() {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "bell_state_fidelity",
                format!("second Bell state failed: {e}"),
            )
        }
    };

    let fidelity = match bell_a.fidelity(&bell_b) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "bell_state_fidelity",
                format!("fidelity() failed: {e}"),
            )
        }
    };

    let error = (fidelity - 1.0).abs();

    if error < FIDELITY_TOLERANCE {
        ValidationResult::pass(
            "bell_state_fidelity",
            format!(
                "F(Bell₁, Bell₂) = {fidelity:.12} (|err| = {error:.2e})"
            ),
        )
    } else {
        ValidationResult::fail(
            "bell_state_fidelity",
            format!(
                "F(Bell₁, Bell₂) = {fidelity:.12}, expected 1.0"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 4. Hadamard involution
// -----------------------------------------------------------------------------

pub fn validate_hadamard_involution() -> ValidationResult {
    let reference = match QuantumRegister::new(1) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "hadamard_involution",
                format!("construction failed: {e}"),
            )
        }
    };

    let mut probe = match QuantumRegister::new(1) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "hadamard_involution",
                format!("construction failed: {e}"),
            )
        }
    };

    if let Err(e) = probe.apply_hadamard(0) {
        return ValidationResult::fail(
            "hadamard_involution",
            format!("first H failed: {e}"),
        );
    }

    if let Err(e) = probe.apply_hadamard(0) {
        return ValidationResult::fail(
            "hadamard_involution",
            format!("second H failed: {e}"),
        );
    }

    let fidelity = match reference.fidelity(&probe) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "hadamard_involution",
                format!("fidelity() failed: {e}"),
            )
        }
    };

    let error = (fidelity - 1.0).abs();

    if error < FIDELITY_TOLERANCE {
        ValidationResult::pass(
            "hadamard_involution",
            format!(
                "F(|0>, H;H|0>) = {fidelity:.12} (|err| = {error:.2e})"
            ),
        )
    } else {
        ValidationResult::fail(
            "hadamard_involution",
            format!("H² != I: fidelity = {fidelity:.12}"),
        )
    }
}

// -----------------------------------------------------------------------------
// 5. Pauli X / RX(pi) involution
// -----------------------------------------------------------------------------

pub fn validate_pauli_x_involution() -> ValidationResult {
    let reference = match QuantumRegister::new(1) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_x_involution",
                format!("construction failed: {e}"),
            )
        }
    };

    let mut probe = reference.clone();

    if let Err(e) = probe.apply_rx(0, PI) {
        return ValidationResult::fail(
            "pauli_x_involution",
            format!("first Rx(pi) failed: {e}"),
        );
    }

    if let Err(e) = probe.apply_rx(0, PI) {
        return ValidationResult::fail(
            "pauli_x_involution",
            format!("second Rx(pi) failed: {e}"),
        );
    }

    // RX(pi)^2 = -I, so fidelity must still be exactly 1.
    let fidelity = match reference.fidelity(&probe) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_x_involution",
                format!("fidelity() failed: {e}"),
            )
        }
    };

    let error = (fidelity - 1.0).abs();

    if error < FIDELITY_TOLERANCE {
        ValidationResult::pass(
            "pauli_x_involution",
            format!(
                "F(|0>, Rx(π);Rx(π)|0>) = {fidelity:.12} (|err| = {error:.2e})"
            ),
        )
    } else {
        ValidationResult::fail(
            "pauli_x_involution",
            format!("RX(pi)^2 failed: fidelity = {fidelity:.12}"),
        )
    }
}

// -----------------------------------------------------------------------------
// 6. Pauli Y / RY(pi) involution
// -----------------------------------------------------------------------------

pub fn validate_pauli_y_involution() -> ValidationResult {
    let reference = match QuantumRegister::new(1) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_y_involution",
                format!("construction failed: {e}"),
            )
        }
    };

    let mut probe = reference.clone();

    if let Err(e) = probe.apply_ry(0, PI) {
        return ValidationResult::fail(
            "pauli_y_involution",
            format!("first Ry(pi) failed: {e}"),
        );
    }

    if let Err(e) = probe.apply_ry(0, PI) {
        return ValidationResult::fail(
            "pauli_y_involution",
            format!("second Ry(pi) failed: {e}"),
        );
    }

    let fidelity = match reference.fidelity(&probe) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_y_involution",
                format!("fidelity() failed: {e}"),
            )
        }
    };

    let error = (fidelity - 1.0).abs();

    if error < FIDELITY_TOLERANCE {
        ValidationResult::pass(
            "pauli_y_involution",
            format!(
                "F(|0>, Ry(π);Ry(π)|0>) = {fidelity:.12} (|err| = {error:.2e})"
            ),
        )
    } else {
        ValidationResult::fail(
            "pauli_y_involution",
            format!("RY(pi)^2 failed: fidelity = {fidelity:.12}"),
        )
    }
}

// -----------------------------------------------------------------------------
// 7. Pauli Z / RZ(pi) involution
// -----------------------------------------------------------------------------

pub fn validate_pauli_z_involution() -> ValidationResult {
    let reference = match QuantumRegister::new(1) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_z_involution",
                format!("construction failed: {e}"),
            )
        }
    };

    let mut probe = reference.clone();

    if let Err(e) = probe.apply_rz(0, PI) {
        return ValidationResult::fail(
            "pauli_z_involution",
            format!("first Rz(pi) failed: {e}"),
        );
    }

    if let Err(e) = probe.apply_rz(0, PI) {
        return ValidationResult::fail(
            "pauli_z_involution",
            format!("second Rz(pi) failed: {e}"),
        );
    }

    let fidelity = match reference.fidelity(&probe) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_z_involution",
                format!("fidelity() failed: {e}"),
            )
        }
    };

    let error = (fidelity - 1.0).abs();

    if error < FIDELITY_TOLERANCE {
        ValidationResult::pass(
            "pauli_z_involution",
            format!(
                "F(|0>, Rz(π);Rz(π)|0>) = {fidelity:.12} (|err| = {error:.2e})"
            ),
        )
    } else {
        ValidationResult::fail(
            "pauli_z_involution",
            format!("RZ(pi)^2 failed: fidelity = {fidelity:.12}"),
        )
    }
}

// -----------------------------------------------------------------------------
// 8. CNOT involution
// -----------------------------------------------------------------------------

pub fn validate_cnot_involution() -> ValidationResult {
    let reference = match QuantumRegister::new(2) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "cnot_involution",
                format!("construction failed: {e}"),
            )
        }
    };

    let mut probe = reference.clone();

    if let Err(e) = probe.apply_cnot(0, 1) {
        return ValidationResult::fail(
            "cnot_involution",
            format!("first CNOT failed: {e}"),
        );
    }

    if let Err(e) = probe.apply_cnot(0, 1) {
        return ValidationResult::fail(
            "cnot_involution",
            format!("second CNOT failed: {e}"),
        );
    }

    let fidelity = match reference.fidelity(&probe) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "cnot_involution",
                format!("fidelity() failed: {e}"),
            )
        }
    };

    let error = (fidelity - 1.0).abs();

    if error < FIDELITY_TOLERANCE {
        ValidationResult::pass(
            "cnot_involution",
            format!(
                "F(|00>, CNOT;CNOT|00>) = {fidelity:.12} (|err| = {error:.2e})"
            ),
        )
    } else {
        ValidationResult::fail(
            "cnot_involution",
            format!("CNOT² != I: fidelity = {fidelity:.12}"),
        )
    }
}

// -----------------------------------------------------------------------------
// 9. Combined gate involution
// -----------------------------------------------------------------------------

pub fn validate_gate_involution() -> ValidationResult {
    let mut h = match QuantumRegister::new(1) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "gate_involution",
                format!("construction failed: {e}"),
            )
        }
    };

    let mut rx = h.clone();
    let mut ry = h.clone();
    let mut rz = h.clone();

    if let Err(e) = h.apply_hadamard(0).and_then(|_| h.apply_hadamard(0)) {
        return ValidationResult::fail(
            "gate_involution",
            format!("H² failed: {e}"),
        );
    }

    if let Err(e) = rx.apply_rx(0, PI).and_then(|_| rx.apply_rx(0, PI)) {
        return ValidationResult::fail(
            "gate_involution",
            format!("RX(pi)² failed: {e}"),
        );
    }

    if let Err(e) = ry.apply_ry(0, PI).and_then(|_| ry.apply_ry(0, PI)) {
        return ValidationResult::fail(
            "gate_involution",
            format!("RY(pi)² failed: {e}"),
        );
    }

    if let Err(e) = rz.apply_rz(0, PI).and_then(|_| rz.apply_rz(0, PI)) {
        return ValidationResult::fail(
            "gate_involution",
            format!("RZ(pi)² failed: {e}"),
        );
    }

    let mut cnot = match QuantumRegister::new(2) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "gate_involution",
                format!("CNOT construction failed: {e}"),
            )
        }
    };

    if let Err(e) = cnot
        .apply_cnot(0, 1)
        .and_then(|_| cnot.apply_cnot(0, 1))
    {
        return ValidationResult::fail(
            "gate_involution",
            format!("CNOT² failed: {e}"),
        );
    }

    let h_f = match QuantumRegister::new(1) {
        Ok(reference) => reference.fidelity(&h),
        Err(e) => Err(e),
    };

    let rx_f = match QuantumRegister::new(1) {
        Ok(reference) => reference.fidelity(&rx),
        Err(e) => Err(e),
    };

    let ry_f = match QuantumRegister::new(1) {
        Ok(reference) => reference.fidelity(&ry),
        Err(e) => Err(e),
    };

    let rz_f = match QuantumRegister::new(1) {
        Ok(reference) => reference.fidelity(&rz),
        Err(e) => Err(e),
    };

    let cnot_ref = match QuantumRegister::new(2) {
        Ok(reference) => reference,
        Err(e) => {
            return ValidationResult::fail(
                "gate_involution",
                format!("CNOT reference construction failed: {e}"),
            )
        }
    };

    let cnot_f = match cnot_ref.fidelity(&cnot) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "gate_involution",
                format!("CNOT fidelity failed: {e}"),
            )
        }
    };

    let values = [
        h_f.unwrap_or(-1.0),
        rx_f.unwrap_or(-1.0),
        ry_f.unwrap_or(-1.0),
        rz_f.unwrap_or(-1.0),
        cnot_f,
    ];

    if values
        .iter()
        .all(|value| (*value - 1.0).abs() < FIDELITY_TOLERANCE)
    {
        ValidationResult::pass(
            "gate_involution",
            "H² = I; Rx(π)² = -I; Ry(π)² = -I; Rz(π)² = -I; CNOT² = I",
        )
    } else {
        ValidationResult::fail(
            "gate_involution",
            format!("involution fidelity values = {values:?}"),
        )
    }
}

// -----------------------------------------------------------------------------
// 10. Hadamard Born-rule statistics
// -----------------------------------------------------------------------------

pub fn validate_hadamard_statistics() -> ValidationResult {
    let mut counts: HashMap<String, u32> = HashMap::new();

    for shot in 0..STATISTICAL_SHOTS {
        let mut register = match QuantumRegister::new_with_seed(1, shot as u64 + 17) {
            Ok(state) => state,
            Err(e) => {
                return ValidationResult::fail(
                    "hadamard_statistics",
                    format!("register construction failed: {e}"),
                )
            }
        };

        if let Err(e) = register.apply_hadamard(0) {
            return ValidationResult::fail(
                "hadamard_statistics",
                format!("Hadamard failed: {e}"),
            );
        }

        let outcome = match register.measure_all_qubits() {
            Ok(value) => value,
            Err(e) => {
                return ValidationResult::fail(
                    "hadamard_statistics",
                    format!("measurement failed: {e}"),
                )
            }
        };

        increment_count(&mut counts, &outcome);
    }

    let zero = *counts.get("[0]").unwrap_or(&0) as f64;
    let one = *counts.get("[1]").unwrap_or(&0) as f64;

    let chi2 = chi_squared_statistic(
        &[zero, one],
        &[
            STATISTICAL_SHOTS as f64 / 2.0,
            STATISTICAL_SHOTS as f64 / 2.0,
        ],
    );

    let p0 = zero / STATISTICAL_SHOTS as f64;
    let p1 = one / STATISTICAL_SHOTS as f64;

    if chi2 < CHI_SQUARED_CRITICAL_1DF_ALPHA_01 {
        ValidationResult::pass(
            "hadamard_statistics",
            format!(
                "P(0) = {p0:.4}, P(1) = {p1:.4}; counts = {counts:?}; chi2 = {chi2:.3} (critical = {CHI_SQUARED_CRITICAL_1DF_ALPHA_01})"
            ),
        )
    } else {
        ValidationResult::fail(
            "hadamard_statistics",
            format!(
                "P(0) = {p0:.4}, P(1) = {p1:.4}; chi2 = {chi2:.3} exceeds critical value"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 11. Bell correlations
// -----------------------------------------------------------------------------

pub fn validate_bell_correlations() -> ValidationResult {
    let mut counts: HashMap<String, u32> = HashMap::new();

    for shot in 0..STATISTICAL_SHOTS {
        let mut register = match QuantumRegister::new_with_seed(2, shot as u64 + 10_000) {
            Ok(state) => state,
            Err(e) => {
                return ValidationResult::fail(
                    "bell_correlations",
                    format!("construction failed: {e}"),
                )
            }
        };

        if let Err(e) = register
            .apply_hadamard(0)
            .and_then(|_| register.apply_cnot(0, 1))
        {
            return ValidationResult::fail(
                "bell_correlations",
                format!("Bell circuit failed: {e}"),
            );
        }

        let outcome = match register.measure_all_qubits() {
            Ok(value) => value,
            Err(e) => {
                return ValidationResult::fail(
                    "bell_correlations",
                    format!("measurement failed: {e}"),
                )
            }
        };

        if outcome.len() != 2 || outcome[0] != outcome[1] {
            return ValidationResult::fail(
                "bell_correlations",
                format!("anti-correlated outcome observed: {outcome:?}"),
            );
        }

        increment_count(&mut counts, &outcome);
    }

    ValidationResult::pass(
        "bell_correlations",
        format!(
            "all {STATISTICAL_SHOTS} shots were correlated; no 01/10 outcomes observed; observed = {counts:?}"
        ),
    )
}

// -----------------------------------------------------------------------------
// 12. Bell-state statistical distribution
// -----------------------------------------------------------------------------

pub fn validate_bell_state_statistics() -> ValidationResult {
    let mut counts: HashMap<String, u32> = HashMap::new();

    for shot in 0..STATISTICAL_SHOTS {
        let mut register = match QuantumRegister::new_with_seed(2, shot as u64 + 20_000) {
            Ok(state) => state,
            Err(e) => {
                return ValidationResult::fail(
                    "bell_state_statistics",
                    format!("construction failed: {e}"),
                )
            }
        };

        if let Err(e) = register
            .apply_hadamard(0)
            .and_then(|_| register.apply_cnot(0, 1))
        {
            return ValidationResult::fail(
                "bell_state_statistics",
                format!("Bell circuit failed: {e}"),
            );
        }

        let outcome = match register.measure_all_qubits() {
            Ok(value) => value,
            Err(e) => {
                return ValidationResult::fail(
                    "bell_state_statistics",
                    format!("measurement failed: {e}"),
                )
            }
        };

        increment_count(&mut counts, &outcome);
    }

    let anti_correlated = counts
        .iter()
        .filter(|(key, _)| key.contains("[0, 1]") || key.contains("[1, 0]"))
        .map(|(_, count)| *count)
        .sum::<u32>();

    if anti_correlated != 0 {
        return ValidationResult::fail(
            "bell_state_statistics",
            format!("observed {anti_correlated} anti-correlated shots: {counts:?}"),
        );
    }

    let zero_zero = *counts.get("[0, 0]").unwrap_or(&0) as f64;
    let one_one = *counts.get("[1, 1]").unwrap_or(&0) as f64;

    let chi2 = chi_squared_statistic(
        &[zero_zero, one_one],
        &[
            STATISTICAL_SHOTS as f64 / 2.0,
            STATISTICAL_SHOTS as f64 / 2.0,
        ],
    );

    if chi2 < CHI_SQUARED_CRITICAL_1DF_ALPHA_01 {
        ValidationResult::pass(
            "bell_state_statistics",
            format!(
                "{STATISTICAL_SHOTS}/{STATISTICAL_SHOTS} shots correlated (0 anti-correlated); split = {counts:?}; chi2 = {chi2:.3} (critical @ df=1, alpha=0.01: {CHI_SQUARED_CRITICAL_1DF_ALPHA_01})"
            ),
        )
    } else {
        ValidationResult::fail(
            "bell_state_statistics",
            format!(
                "Bell distribution chi2 = {chi2:.3}, exceeding critical value; counts = {counts:?}"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 13. GHZ structure
// -----------------------------------------------------------------------------

pub fn validate_ghz_state() -> ValidationResult {
    let ghz = match create_ghz_state(3) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "ghz_state",
                format!("GHZ construction failed: {e}"),
            )
        }
    };

    let p000 = probability_of_state(&ghz, 0);
    let p111 = probability_of_state(&ghz, 7);

    let mut forbidden_probability = 0.0;

    for index in 1..7 {
        forbidden_probability += probability_of_state(&ghz, index);
    }

    let expected = 0.5;

    if (p000 - expected).abs() < NUMERICAL_TOLERANCE
        && (p111 - expected).abs() < NUMERICAL_TOLERANCE
        && forbidden_probability < NUMERICAL_TOLERANCE
    {
        ValidationResult::pass(
            "ghz_state",
            format!(
                "3-qubit GHZ: P(000) = {p000:.12}, P(111) = {p111:.12}, forbidden probability = {forbidden_probability:.2e}"
            ),
        )
    } else {
        ValidationResult::fail(
            "ghz_state",
            format!(
                "unexpected GHZ distribution: P(000) = {p000:.12}, P(111) = {p111:.12}, forbidden = {forbidden_probability:.3e}"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 14. W-state structure
// -----------------------------------------------------------------------------

pub fn validate_w_state() -> ValidationResult {
    let w = match create_w_state(3) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "w_state",
                format!("W-state construction failed: {e}"),
            )
        }
    };

    let expected = 1.0 / 3.0;

    // For 3 qubits the W state is:
    //
    // |001> + |010> + |100>
    //
    // with equal probability 1/3.
    let p001 = probability_of_state(&w, 1);
    let p010 = probability_of_state(&w, 2);
    let p100 = probability_of_state(&w, 4);

    let forbidden = probability_of_state(&w, 0)
        + probability_of_state(&w, 3)
        + probability_of_state(&w, 5)
        + probability_of_state(&w, 6)
        + probability_of_state(&w, 7);

    let max_error = [
        (p001 - expected).abs(),
        (p010 - expected).abs(),
        (p100 - expected).abs(),
        forbidden.abs(),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);

    if max_error < NUMERICAL_TOLERANCE {
        ValidationResult::pass(
            "w_state",
            format!(
                "P(001) = {p001:.12}, P(010) = {p010:.12}, P(100) = {p100:.12}; forbidden probability = {forbidden:.2e}"
            ),
        )
    } else {
        ValidationResult::fail(
            "w_state",
            format!(
                "W-state max probability error = {max_error:.3e}; probabilities = [{p001:.12}, {p010:.12}, {p100:.12}]"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 15. QFT / inverse-QFT round trip
// -----------------------------------------------------------------------------

pub fn validate_qft_round_trip() -> ValidationResult {
    let mut original = match QuantumRegister::new(3) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "qft_round_trip",
                format!("construction failed: {e}"),
            )
        }
    };

    if let Err(e) = original.apply_pauli_x(0) {
        return ValidationResult::fail(
            "qft_round_trip",
            format!("state preparation failed: {e}"),
        );
    }

    let mut transformed = original.clone();

    if let Err(e) = quantum_fourier_transform(&mut transformed) {
        return ValidationResult::fail(
            "qft_round_trip",
            format!("QFT failed: {e}"),
        );
    }

    if let Err(e) = inverse_quantum_fourier_transform(&mut transformed) {
        return ValidationResult::fail(
            "qft_round_trip",
            format!("inverse QFT failed: {e}"),
        );
    }

    let fidelity = match original.fidelity(&transformed) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "qft_round_trip",
                format!("fidelity failed: {e}"),
            )
        }
    };

    let error = (fidelity - 1.0).abs();

    if error < FIDELITY_TOLERANCE {
        ValidationResult::pass(
            "qft_round_trip",
            format!(
                "F(input, IQFT(QFT(input))) = {fidelity:.12} (|err| = {error:.2e})"
            ),
        )
    } else {
        ValidationResult::fail(
            "qft_round_trip",
            format!(
                "QFT round-trip fidelity = {fidelity:.12}, expected 1.0"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 16. QFT probability normalization
// -----------------------------------------------------------------------------

pub fn validate_qft_probability_normalization() -> ValidationResult {
    let mut register = match QuantumRegister::new(4) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "qft_probability_normalization",
                format!("construction failed: {e}"),
            )
        }
    };

    // Prepare a non-trivial computational basis state.
    if let Err(e) = register
        .apply_pauli_x(0)
        .and_then(|_| register.apply_pauli_x(2))
    {
        return ValidationResult::fail(
            "qft_probability_normalization",
            format!("state preparation failed: {e}"),
        );
    }

    if let Err(e) = quantum_fourier_transform(&mut register) {
        return ValidationResult::fail(
            "qft_probability_normalization",
            format!("QFT failed: {e}"),
        );
    }

    let distribution = register.get_probability_distribution();

    let sum: f64 = distribution.values().sum();
    let error = (sum - 1.0).abs();

    if error < NUMERICAL_TOLERANCE {
        ValidationResult::pass(
            "qft_probability_normalization",
            format!(
                "sum of QFT measurement probabilities = {sum:.12} (|err| = {error:.2e}); {} non-zero basis states",
                distribution.len()
            ),
        )
    } else {
        ValidationResult::fail(
            "qft_probability_normalization",
            format!(
                "QFT probability sum = {sum:.12}, expected 1.0"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 17. Probability API consistency
// -----------------------------------------------------------------------------

pub fn validate_probability_api_consistency() -> ValidationResult {
    let mut register = match QuantumRegister::new(3) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "probability_api_consistency",
                format!("construction failed: {e}"),
            )
        }
    };

    if let Err(e) = register
        .apply_hadamard(0)
        .and_then(|_| register.apply_pauli_x(1))
    {
        return ValidationResult::fail(
            "probability_api_consistency",
            format!("state preparation failed: {e}"),
        );
    }

    let distribution = register.get_probability_distribution();

    let mut sum_from_distribution = 0.0;

    for index in 0..register.dimension() {
        let state = basis_state_label(register.num_qubits(), index);

        let direct = match register.get_state_probability(&state) {
            Ok(value) => value,
            Err(e) => {
                return ValidationResult::fail(
                    "probability_api_consistency",
                    format!("get_state_probability({state}) failed: {e}"),
                )
            }
        };

        let map_value = *distribution.get(&state).unwrap_or(&0.0);

        if (direct - map_value).abs() > NUMERICAL_TOLERANCE
            && direct > NUMERICAL_TOLERANCE
        {
            return ValidationResult::fail(
                "probability_api_consistency",
                format!(
                    "state {state}: direct = {direct:.12}, distribution = {map_value:.12}"
                ),
            );
        }

        sum_from_distribution += map_value;
    }

    let direct_total = total_probability(&register);
    let error = (sum_from_distribution - direct_total).abs();

    if error < NUMERICAL_TOLERANCE {
        ValidationResult::pass(
            "probability_api_consistency",
            format!(
                "direct probability API and distribution agree; total = {direct_total:.12}, max discrepancy < {NUMERICAL_TOLERANCE:.1e}"
            ),
        )
    } else {
        ValidationResult::fail(
            "probability_api_consistency",
            format!(
                "distribution total = {sum_from_distribution:.12}, direct total = {direct_total:.12}"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 18. Measurement probability API
// -----------------------------------------------------------------------------

pub fn validate_measurement_probability_api() -> ValidationResult {
    let mut register = match QuantumRegister::new(1) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "measurement_probability_api",
                format!("construction failed: {e}"),
            )
        }
    };

    if let Err(e) = register.apply_hadamard(0) {
        return ValidationResult::fail(
            "measurement_probability_api",
            format!("Hadamard failed: {e}"),
        );
    }

    let (p0, p1) = match register.get_measurement_probability(0) {
        Ok(values) => values,
        Err(e) => {
            return ValidationResult::fail(
                "measurement_probability_api",
                format!("probability lookup failed: {e}"),
            )
        }
    };

    let error = (p0 - 0.5).abs().max((p1 - 0.5).abs());

    if error < NUMERICAL_TOLERANCE {
        ValidationResult::pass(
            "measurement_probability_api",
            format!(
                "P(0) = {p0:.12}, P(1) = {p1:.12}; sum = {:.12}",
                p0 + p1
            ),
        )
    } else {
        ValidationResult::fail(
            "measurement_probability_api",
            format!(
                "expected 0.5/0.5, got P(0) = {p0:.12}, P(1) = {p1:.12}"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 19. Pauli expectation values
// -----------------------------------------------------------------------------

pub fn validate_pauli_expectation_values() -> ValidationResult {
    let zero = match QuantumRegister::new(1) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_expectation_values",
                format!("construction failed: {e}"),
            )
        }
    };

    let z_zero = match zero.expectation_value_pauli_z(0) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_expectation_values",
                format!("Z expectation failed: {e}"),
            )
        }
    };

    let mut one = zero.clone();

    if let Err(e) = one.apply_pauli_x(0) {
        return ValidationResult::fail(
            "pauli_expectation_values",
            format!("X preparation failed: {e}"),
        );
    }

    let z_one = match one.expectation_value_pauli_z(0) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_expectation_values",
                format!("Z expectation on |1> failed: {e}"),
            )
        }
    };

    let mut plus = zero.clone();

    if let Err(e) = plus.apply_hadamard(0) {
        return ValidationResult::fail(
            "pauli_expectation_values",
            format!("plus-state preparation failed: {e}"),
        );
    }

    let x_plus = match plus.expectation_value_pauli_string(&[
        (0, sirraya_qutub::PauliOp::X),
    ]) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "pauli_expectation_values",
                format!("X expectation failed: {e}"),
            )
        }
    };

    if (z_zero - 1.0).abs() < NUMERICAL_TOLERANCE
        && (z_one + 1.0).abs() < NUMERICAL_TOLERANCE
        && (x_plus - 1.0).abs() < NUMERICAL_TOLERANCE
    {
        ValidationResult::pass(
            "pauli_expectation_values",
            format!(
                "<Z>_|0> = {z_zero:.12}, <Z>_|1> = {z_one:.12}, <X>_|+> = {x_plus:.12}"
            ),
        )
    } else {
        ValidationResult::fail(
            "pauli_expectation_values",
            format!(
                "unexpected expectations: Z|0> = {z_zero}, Z|1> = {z_one}, X|+> = {x_plus}"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 20. Fidelity / trace-distance consistency
// -----------------------------------------------------------------------------

pub fn validate_fidelity_trace_distance() -> ValidationResult {
    let mut a = match QuantumRegister::new(1) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "fidelity_trace_distance",
                format!("construction failed: {e}"),
            )
        }
    };

    let mut b = a.clone();

    if let Err(e) = b.apply_pauli_x(0) {
        return ValidationResult::fail(
            "fidelity_trace_distance",
            format!("state preparation failed: {e}"),
        );
    }

    let fidelity = match a.fidelity(&b) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "fidelity_trace_distance",
                format!("fidelity failed: {e}"),
            )
        }
    };

    let trace_distance = match a.trace_distance(&b) {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "fidelity_trace_distance",
                format!("trace distance failed: {e}"),
            )
        }
    };

    // For pure states:
    //
    // D = sqrt(1 - F)
    //
    let expected_distance = (1.0 - fidelity).max(0.0).sqrt();
    let error = (trace_distance - expected_distance).abs();

    if error < NUMERICAL_TOLERANCE {
        ValidationResult::pass(
            "fidelity_trace_distance",
            format!(
                "F = {fidelity:.12}, D = {trace_distance:.12}, sqrt(1-F) = {expected_distance:.12}"
            ),
        )
    } else {
        ValidationResult::fail(
            "fidelity_trace_distance",
            format!(
                "D = {trace_distance:.12}, expected sqrt(1-F) = {expected_distance:.12}"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 21. Controlled phase gate
// -----------------------------------------------------------------------------

pub fn validate_controlled_phase_gate() -> ValidationResult {
    let mut register = match QuantumRegister::new(2) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "controlled_phase_gate",
                format!("construction failed: {e}"),
            )
        }
    };

    // Prepare |11>.
    if let Err(e) = register
        .apply_pauli_x(0)
        .and_then(|_| register.apply_pauli_x(1))
    {
        return ValidationResult::fail(
            "controlled_phase_gate",
            format!("state preparation failed: {e}"),
        );
    }

    let before = register.get_state_vector()[3];

    if let Err(e) = register.apply_controlled_phase(0, 1, PI / 2.0) {
        return ValidationResult::fail(
            "controlled_phase_gate",
            format!("controlled phase failed: {e}"),
        );
    }

    let after = register.get_state_vector()[3];

    // e^(i*pi/2) = i.
    //
    // Since the initial amplitude is 1, the final amplitude should be i.
    let expected = Complex::new(0.0, 1.0);

    let error = (after.real() - expected.real())
        .abs()
        .max((after.imag() - expected.imag()).abs());

    if (before.real() - 1.0).abs() < NUMERICAL_TOLERANCE
        && before.imag().abs() < NUMERICAL_TOLERANCE
        && error < NUMERICAL_TOLERANCE
    {
        ValidationResult::pass(
            "controlled_phase_gate",
            format!(
                "|11> phase: before = {before}, after = {after}; expected after = {expected}"
            ),
        )
    } else {
        ValidationResult::fail(
            "controlled_phase_gate",
            format!(
                "controlled phase mismatch: before = {before}, after = {after}, expected = {expected}"
            ),
        )
    }
}

// -----------------------------------------------------------------------------
// 22. Seeded measurement reproducibility + QASM export + density conversion
// -----------------------------------------------------------------------------
//
// This final validation intentionally checks several public API invariants
// together. It is still one logical regression check: deterministic seeded
// execution must be reproducible, the resulting register must remain finite,
// its density-matrix conversion must succeed, and QASM export must contain
// the expected OpenQASM header.

pub fn validate_reproducibility_conversion_and_qasm() -> ValidationResult {
    let seed = 0x5EED_u64;

    let mut a = match QuantumRegister::new_with_seed(2, seed) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "reproducibility_conversion_qasm",
                format!("register A construction failed: {e}"),
            )
        }
    };

    let mut b = match QuantumRegister::new_with_seed(2, seed) {
        Ok(state) => state,
        Err(e) => {
            return ValidationResult::fail(
                "reproducibility_conversion_qasm",
                format!("register B construction failed: {e}"),
            )
        }
    };

    let prepare = |register: &mut QuantumRegister| -> Result<(), String> {
        register.apply_hadamard(0)?;
        register.apply_cnot(0, 1)?;
        Ok(())
    };

    if let Err(e) = prepare(&mut a) {
        return ValidationResult::fail(
            "reproducibility_conversion_qasm",
            format!("register A preparation failed: {e}"),
        );
    }

    if let Err(e) = prepare(&mut b) {
        return ValidationResult::fail(
            "reproducibility_conversion_qasm",
            format!("register B preparation failed: {e}"),
        );
    }

    let outcome_a = match a.measure_all_qubits() {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "reproducibility_conversion_qasm",
                format!("measurement A failed: {e}"),
            )
        }
    };

    let outcome_b = match b.measure_all_qubits() {
        Ok(value) => value,
        Err(e) => {
            return ValidationResult::fail(
                "reproducibility_conversion_qasm",
                format!("measurement B failed: {e}"),
            )
        }
    };

    if outcome_a != outcome_b {
        return ValidationResult::fail(
            "reproducibility_conversion_qasm",
            format!(
                "same seed produced different outcomes: A = {outcome_a:?}, B = {outcome_b:?}"
            ),
        );
    }

    if !state_is_finite(&a) || !state_is_finite(&b) {
        return ValidationResult::fail(
            "reproducibility_conversion_qasm",
            "measurement produced a non-finite state",
        );
    }

    // Conversion to a density matrix should succeed for every valid pure state.
    if let Err(e) = a.to_density_matrix() {
        return ValidationResult::fail(
            "reproducibility_conversion_qasm",
            format!("density-matrix conversion failed: {e}"),
        );
    }

    let qasm = a.to_qasm("validation");

    let required_fragments = [
        "OPENQASM 2.0;",
        "include \"qelib1.inc\";",
        "qreg q[2];",
        "creg c[2];",
        "// Circuit: validation",
    ];

    for fragment in required_fragments {
        if !qasm.contains(fragment) {
            return ValidationResult::fail(
                "reproducibility_conversion_qasm",
                format!("QASM missing expected fragment: {fragment:?}"),
            );
        }
    }

    ValidationResult::pass(
        "reproducibility_conversion_qasm",
        format!(
            "seeded outcomes identical = {outcome_a:?}; density conversion succeeded; QASM header/export validated"
        ),
    )
}

// -----------------------------------------------------------------------------
// Full suite
// -----------------------------------------------------------------------------

pub fn run_full_validation_suite() -> Vec<ValidationResult> {
    vec![
        // 1
        validate_state_normalization(),

        // 2
        validate_self_fidelity_bell(),

        // 3
        validate_bell_state_fidelity(),

        // 4
        validate_hadamard_involution(),

        // 5
        validate_pauli_x_involution(),

        // 6
        validate_pauli_y_involution(),

        // 7
        validate_pauli_z_involution(),

        // 8
        validate_cnot_involution(),

        // 9
        validate_gate_involution(),

        // 10
        validate_hadamard_statistics(),

        // 11
        validate_bell_correlations(),

        // 12
        validate_bell_state_statistics(),

        // 13
        validate_ghz_state(),

        // 14
        validate_w_state(),

        // 15
        validate_qft_round_trip(),

        // 16
        validate_qft_probability_normalization(),

        // 17
        validate_probability_api_consistency(),

        // 18
        validate_measurement_probability_api(),

        // 19
        validate_pauli_expectation_values(),

        // 20
        validate_fidelity_trace_distance(),

        // 21
        validate_controlled_phase_gate(),

        // 22
        validate_reproducibility_conversion_and_qasm(),
    ]
}

// -----------------------------------------------------------------------------
// Report
// -----------------------------------------------------------------------------

fn print_report(results: &[ValidationResult]) -> bool {
    let mut all_passed = true;

    println!("=== sirraya-qutub numerical validation ===\n");

    for (index, result) in results.iter().enumerate() {
        let marker = if result.passed {
            "PASS"
        } else {
            all_passed = false;
            "FAIL"
        };

        println!("[{marker}] {:02}. {}", index + 1, result.name);
        println!("       {}", result.detail);
    }

    let passed = results.iter().filter(|result| result.passed).count();

    println!();
    println!("{passed}/{} checks passed", results.len());

    if all_passed {
        println!("All implemented numerical validation checks passed.");
    } else {
        println!("One or more numerical validation checks failed.");
    }

    all_passed
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

fn main() {
    let results = run_full_validation_suite();

    let all_passed = print_report(&results);

    if !all_passed {
        std::process::exit(1);
    }
}