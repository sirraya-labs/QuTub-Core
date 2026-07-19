//! Sirraya QuTub: a quantum circuit and noise simulator.
//!
//! - [`complex`]: complex-number arithmetic underlying every amplitude and
//!   matrix entry.
//! - [`core`]: pure-state (`QuantumRegister`) and mixed-state/noise
//!   (`DensityMatrix`) simulation, circuit building, and standard
//!   algorithm building blocks.
//! - [`xeb`]: noise calibrated to published hardware fidelities, validated
//!   via cross-entropy benchmarking.
//! - [`reservoir`]: quantum reservoir computing on top of the core
//!   simulator.

pub mod complex;
pub mod core;
pub mod reservoir;
pub mod xeb;

pub use complex::Complex;
pub use core::{
    create_bell_state, create_ghz_state, inverse_quantum_fourier_transform,
    quantum_fourier_transform, DensityMatrix, QuantumAlgorithm, QuantumBenchmark, QuantumCircuit,
    QuantumRegister,
};
pub use reservoir::{
    demonstrate_quantum_reservoir_computing, QuantumReservoir, QuantumReservoirComputer,
};
pub use xeb::{cross_entropy_benchmark_fidelity, run_xeb_demo, HardwareCalibration};
