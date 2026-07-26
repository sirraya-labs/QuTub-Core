//! Sirraya QuTub: a quantum circuit and noise simulator.
//!
//! # Quickstart
//!
//! ```
//! use sirraya_qutub::{QuantumRegister, create_bell_state};
//!
//! // Build a Bell state (|00> + |11>) / sqrt(2) and inspect it.
//! let bell = create_bell_state()?;
//! assert!((bell.fidelity(&bell)? - 1.0).abs() < 1e-9);
//!
//! // Or build a register by hand and apply gates directly.
//! let mut reg = QuantumRegister::new(2)?;
//! reg.apply_hadamard(0)?;
//! reg.apply_cnot(0, 1)?;
//! let outcome = reg.measure_all_qubits()?;
//! assert_eq!(outcome.len(), 2);
//! # Ok::<(), String>(())
//! ```
//!
//! # Modules
//!
//! - [`complex`]: complex-number arithmetic underlying every amplitude and
//!   matrix entry.
//! - [`core`]: pure-state (`QuantumRegister`) and mixed-state/noise
//!   (`DensityMatrix`) simulation, circuit building, and standard
//!   algorithm building blocks (Bell/GHZ/W states, QFT, Deutsch-Jozsa,
//!   Grover).
//! - [`xeb`]: noise calibrated to published hardware fidelities, validated
//!   via cross-entropy benchmarking.
//! - [`reservoir`]: quantum reservoir computing on top of the core
//!   simulator.

/// Complex-number arithmetic (`Complex`) underlying every amplitude and
/// matrix entry used throughout this crate.
pub mod complex;

/// Pure-state (`QuantumRegister`) and mixed-state/noise (`DensityMatrix`)
/// simulation, the chainable `QuantumCircuit` builder, and standard
/// algorithm building blocks.
pub mod core;

/// Quantum reservoir computing (`QuantumReservoir`, `QuantumReservoirComputer`)
/// built on top of the `core` simulator.
pub mod reservoir;

/// Hardware-calibrated noise models and cross-entropy benchmarking (XEB).
pub mod xeb;

pub use complex::Complex;
pub use core::{
    create_bell_state, create_ghz_state, create_w_state, inverse_quantum_fourier_transform,
    quantum_fourier_transform, DensityMatrix, PauliOp, QuantumAlgorithm, QuantumBenchmark,
    QuantumCircuit, QuantumRegister,
};
pub use reservoir::{
    demonstrate_quantum_reservoir_computing, QuantumReservoir, QuantumReservoirComputer,
};
pub use xeb::{cross_entropy_benchmark_fidelity, run_xeb_demo, HardwareCalibration};