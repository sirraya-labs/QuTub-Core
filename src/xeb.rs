//! Hardware-calibrated noise and cross-entropy benchmarking (XEB).
//!
//! This module ties together two pieces of established methodology:
//!
//! 1. Per-gate depolarizing noise calibrated against a published hardware
//!    fidelity figure -- Quantinuum's 98-qubit Helios trapped-ion system,
//!    benchmarked by Sandia National Laboratories and published in Nature
//!    in June 2026: single-qubit gate fidelity 99.9975%, two-qubit gate
//!    fidelity 99.921%. (Sandia National Laboratories / Quantinuum, Nature,
//!    June 2026; see
//!    <https://thequantuminsider.com/2026/06/18/researchers-publish-peer-reviewed-results-on-quantinuums-helios-quantum-computer/>)
//!
//! 2. Linear cross-entropy benchmarking (XEB), the fidelity estimator used
//!    to validate noisy quantum devices/simulators against an ideal
//!    (classically-simulated) distribution: Arute et al., "Quantum
//!    supremacy using a programmable superconducting processor", Nature
//!    574, 505-510 (2019). The same linear-XEB estimator remains the
//!    standard cross-check used in current NISQ-era hardware validation
//!    work, including in the 2026 Helios benchmarking above.
//!
//! Given a target gate fidelity `F` for a depolarizing channel, the
//! corresponding per-gate error probability is `p = (1 - F) * d / (d - 1)`
//! for a d-dimensional system (`d = 2` for single-qubit gates, `d = 4` for
//! two-qubit gates) -- the standard relationship between average gate
//! fidelity and the depolarizing parameter used in randomized-benchmarking
//! literature.

use crate::complex::Complex;
use crate::core::{DensityMatrix, QuantumRegister};
use std::collections::HashMap;
use std::f64::consts::PI;
use rand::Rng;

/// Hardware error-rate calibration points that this simulator's noise
/// model can be pinned to, so demonstrations are grounded in a real,
/// currently-reported device rather than an arbitrary noise_level slider.
#[derive(Debug, Clone, Copy)]
pub struct HardwareCalibration {
    pub name: &'static str,
    pub single_qubit_fidelity: f64,
    pub two_qubit_fidelity: f64,
}

impl HardwareCalibration {
    /// Quantinuum Helios (98-qubit trapped-ion), as benchmarked by Sandia
    /// National Laboratories and published in Nature, June 2026.
    pub fn quantinuum_helios_2026() -> Self {
        Self {
            name: "Quantinuum Helios (Sandia benchmark, Nature, June 2026)",
            single_qubit_fidelity: 0.999975,
            two_qubit_fidelity: 0.99921,
        }
    }

    /// Convert an average gate fidelity to the depolarizing-channel error
    /// probability that reproduces it, for a gate acting on `num_qubits`
    /// qubits (1 for single-qubit gates, 2 for two-qubit gates). This is
    /// the standard fidelity <-> depolarizing-parameter relation used in
    /// randomized benchmarking: p = (1 - F) * d / (d - 1).
    pub fn fidelity_to_depolarizing_probability(fidelity: f64, num_qubits: usize) -> f64 {
        let d = (1usize << num_qubits) as f64;
        ((1.0 - fidelity) * d / (d - 1.0)).clamp(0.0, 1.0)
    }

    pub fn single_qubit_error_probability(&self) -> f64 {
        Self::fidelity_to_depolarizing_probability(self.single_qubit_fidelity, 1)
    }

    pub fn two_qubit_error_probability(&self) -> f64 {
        Self::fidelity_to_depolarizing_probability(self.two_qubit_fidelity, 2)
    }
}

/// Linear cross-entropy benchmarking (XEB) fidelity estimator (Arute et
/// al. 2019; see module doc comment above). Given the ideal (noiseless)
/// output distribution of a circuit and a set of bitstrings actually
/// sampled from a noisy run, estimates how much the noisy device's output
/// distribution resembles the ideal one.
///
/// F_XEB = (D * mean_over_samples[p_ideal(sample)] - 1) / (D - 1)
///
/// F_XEB = 1.0 for a perfectly noiseless device; F_XEB = 0.0 for a fully
/// depolarized (uniformly random) output; can go slightly negative for
/// finite sample counts even at F_XEB = 0 in expectation.
pub fn cross_entropy_benchmark_fidelity(
    ideal_probabilities: &HashMap<String, f64>,
    num_qubits: usize,
    sampled_bitstrings: &[String],
) -> f64 {
    let dim = (1u64 << num_qubits) as f64;
    if dim <= 1.0 || sampled_bitstrings.is_empty() {
        return 0.0;
    }

    let mean_ideal_prob: f64 = sampled_bitstrings
        .iter()
        .map(|s| *ideal_probabilities.get(s).unwrap_or(&0.0))
        .sum::<f64>()
        / sampled_bitstrings.len() as f64;

    (dim * mean_ideal_prob - 1.0) / (dim - 1.0)
}

/// Run a fixed benchmark circuit twice -- once ideally (pure state vector,
/// no noise) and once through a density-matrix simulation with per-gate
/// depolarizing noise calibrated to real published hardware fidelities --
/// then estimate how faithfully the noisy run reproduces the ideal
/// distribution using linear XEB. This is the same style of cross-check
/// used to validate real quantum processors against classical simulation.
pub fn run_xeb_demo(num_qubits: usize, calibration: HardwareCalibration, num_samples: usize) -> Result<f64, String> {
    // Build the ideal (noiseless) distribution via ordinary state-vector
    // simulation of a representative circuit: alternating Hadamards and a
    // ring of CNOTs, repeated a few layers -- enough entangling structure
    // to make the output distribution nontrivial.
    let build_ideal = || -> Result<QuantumRegister, String> {
        let mut reg = QuantumRegister::new(num_qubits)?;
        for layer in 0..3 {
            for q in 0..num_qubits {
                reg.apply_hadamard(q)?;
                let angle = PI / (2.0 + layer as f64 + q as f64 * 0.1);
                reg.apply_rz(q, angle)?;
            }
            for q in 0..num_qubits {
                reg.apply_cnot(q, (q + 1) % num_qubits)?;
            }
        }
        Ok(reg)
    };

    let ideal_register = build_ideal()?;
    let ideal_probabilities = ideal_register.get_probability_distribution();

    // Now the same circuit, but as a density matrix with a depolarizing
    // channel injected after every gate, using the O(d^2) single-qubit
    // Kraus implementation so this stays practical at real qubit counts.
    let single_qubit_error = calibration.single_qubit_error_probability();
    let two_qubit_error = calibration.two_qubit_error_probability();

    let mut density = DensityMatrix::new(num_qubits)?;
    for layer in 0..3 {
        for q in 0..num_qubits {
            let mut reg = QuantumRegister::new(1)?;
            reg.apply_hadamard(0)?;
            let h_unitary: Vec<Complex> = {
                let f = 1.0 / 2.0_f64.sqrt();
                vec![
                    Complex::new(f, 0.0), Complex::new(f, 0.0),
                    Complex::new(f, 0.0), Complex::new(-f, 0.0),
                ]
            };
            let _ = reg; // constructed above only to keep intent explicit
            density.apply_unitary_embedded(&h_unitary, &[q])?;
            density.apply_depolarizing_channel(single_qubit_error, q)?;

            let angle = PI / (2.0 + layer as f64 + q as f64 * 0.1);
            let (c, s) = ((angle / 2.0).cos(), (angle / 2.0).sin());
            let rz_unitary = vec![
                Complex::new(c, -s), Complex::zero(),
                Complex::zero(), Complex::new(c, s),
            ];
            density.apply_unitary_embedded(&rz_unitary, &[q])?;
            density.apply_depolarizing_channel(single_qubit_error, q)?;
        }
        for q in 0..num_qubits {
            let target = (q + 1) % num_qubits;
            density.apply_cnot_embedded(q, target)?;
            density.apply_depolarizing_channel(two_qubit_error, q)?;
            density.apply_depolarizing_channel(two_qubit_error, target)?;
        }
    }

    // Sample bitstrings from the noisy density matrix's diagonal
    // (measurement) distribution.
    let dim = 1usize << num_qubits;
    let diag_probs: Vec<f64> = (0..dim).map(|i| density.get_matrix()[i][i].real().max(0.0)).collect();
    let total: f64 = diag_probs.iter().sum();

    let mut rng = rand::thread_rng();
    let mut sampled_bitstrings = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let r: f64 = rng.gen::<f64>() * total;
        let mut cumulative = 0.0;
        let mut chosen = dim - 1;
        for (i, &p) in diag_probs.iter().enumerate() {
            cumulative += p;
            if r <= cumulative {
                chosen = i;
                break;
            }
        }
        let bitstring: String = (0..num_qubits)
            .rev()
            .map(|b| if (chosen >> b) & 1 == 1 { '1' } else { '0' })
            .collect();
        sampled_bitstrings.push(bitstring);
    }

    Ok(cross_entropy_benchmark_fidelity(&ideal_probabilities, num_qubits, &sampled_bitstrings))
}
