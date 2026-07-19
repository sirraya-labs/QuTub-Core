# Sirraya QuTub

A quantum circuit and noise simulator written from scratch in Rust, covering pure-state simulation, density-matrix noise modeling, quantum reservoir computing, and validation against published hardware fidelity data.

## Overview

Sirraya QuTub implements a full quantum simulation stack without relying on an existing simulator framework: its own complex arithmetic, a pure-state vector simulator, a mixed-state density-matrix simulator with physical noise channels, and a quantum reservoir computer built on top of both.

The design goal is to make noisy, mid-sized simulations practical rather than a research-notebook exercise. Noise-channel and gate-embedding operations are implemented at the same asymptotic complexity as the underlying density matrix (`O(d^2)`), rather than the dense `O(d^4)` operator conjugation a naive implementation would use, so density-matrix simulation stays usable well past toy qubit counts.

## Features

**Simulation core**
- Full single- and multi-qubit gate set: Hadamard, Pauli, S/T (and inverses), arbitrary rotations, CNOT, controlled-Z, controlled-phase, SWAP, CSWAP, Toffoli, and generalized multi-controlled-X/Z.
- Density-matrix simulation with physical noise channels: depolarizing and amplitude damping.
- Quantum Fourier Transform and its inverse.
- Standard algorithm building blocks: Deutsch-Jozsa and Grover iteration.
- QASM 2.0 circuit export for interoperability with other tooling.

**Noise validation**
- Noise channels can be calibrated to published, cited hardware fidelity figures rather than an arbitrary noise parameter.
- Linear cross-entropy benchmarking (XEB), the same fidelity estimator used to validate real quantum processors against classical simulation.

**Quantum reservoir computing**
- A fixed, small-world-connected qubit network used as a high-dimensional nonlinear dynamical system, paired with a trained linear readout.
- Feature-importance reporting, so the trained readout's weights are inspectable rather than opaque.

**Tooling**
- A command-line interface for reservoir training, prediction, and benchmarking.
- A benchmark suite reporting gate-chain and QFT throughput across qubit counts.

## Installation

Add the dependency to `Cargo.toml`:

```toml
[dependencies]
sirraya-qutub = "0.1"
```

This crate currently ships as a binary with an internal library module (`quantum_simulator`). To use it as a library dependency in another project, add a `src/lib.rs` exposing the module:

```rust
pub mod quantum_simulator;
```

## Quick start

### Pure-state simulation

```rust
use sirraya_qutub::quantum_simulator::{create_bell_state, QuantumRegister};

fn main() -> Result<(), String> {
    let bell = create_bell_state()?;
    let distribution = bell.get_probability_distribution();
    // {"00": 0.5, "11": 0.5}

    let mut register = QuantumRegister::new(3)?;
    register.apply_hadamard(0)?;
    register.apply_cnot(0, 1)?;
    register.apply_cnot(1, 2)?;

    Ok(())
}
```

### Noisy simulation

```rust
use sirraya_qutub::quantum_simulator::{create_bell_state, DensityMatrix};

fn main() -> Result<(), String> {
    let bell = create_bell_state()?;
    let mut density = DensityMatrix::from_state_vector(bell.get_state_vector())?;

    // A pure Bell state starts at purity 1.0.
    assert!(density.is_pure());

    density.apply_depolarizing_channel(0.2, 0)?;
    // Purity drops to ~0.653 under 20% single-qubit depolarizing noise.

    Ok(())
}
```

### Validating noise against real hardware

```rust
use sirraya_qutub::quantum_simulator::{run_xeb_demo, HardwareCalibration};

fn main() -> Result<(), String> {
    let calibration = HardwareCalibration::quantinuum_helios_2026();
    let fidelity = run_xeb_demo(6, calibration, 2000)?;
    println!("Estimated XEB fidelity: {:.4}", fidelity);
    Ok(())
}
```

`run_xeb_demo` runs a fixed benchmark circuit twice: once ideally via state-vector simulation, and once through the density-matrix path with per-gate depolarizing noise derived from the given `HardwareCalibration`'s published fidelity figures. Linear XEB then estimates how faithfully the noisy run reproduces the ideal distribution.

### Quantum reservoir computing

```rust
use sirraya_qutub::quantum_simulator::QuantumReservoirComputer;

fn main() -> Result<(), String> {
    let mut qrc = QuantumReservoirComputer::new(4, "small_world", 0.1)?;

    let training_inputs: Vec<Vec<f64>> = /* input sequences */ vec![];
    let training_outputs: Vec<f64> = /* targets */ vec![];

    let training_rmse = qrc.train(&training_inputs, &training_outputs, 0.05)?;
    let prediction = qrc.predict(&[0.1, 0.2, 0.3, 0.4], 0.05)?;

    for (feature_index, weight) in qrc.get_feature_importance().iter().take(5) {
        println!("Feature {feature_index}: |weight| = {weight:.6}");
    }

    Ok(())
}
```

## Command-line interface

```text
sirraya-qutub reservoir demo                     Run a 4-qubit reservoir demonstration
sirraya-qutub reservoir train <qubits>           Train an n-qubit reservoir on synthetic data
sirraya-qutub reservoir predict <qubits> <values...>   Predict with an n-qubit reservoir
sirraya-qutub reservoir info <qubits>            Show reservoir configuration and capacity
sirraya-qutub benchmark <qubits>                 Run the performance benchmark suite
sirraya-qutub help                               Show usage information
```

Running the binary with no arguments executes the full demonstration suite: state preparation, noise channels, gate circuits, QFT round-trip, measurement, benchmarking, QASM export, and reservoir computing.

## Architecture

The simulator is organized into four modules under `quantum_simulator`:

| Module | Contents |
|---|---|
| `complex` | Complex-number arithmetic underlying every amplitude and matrix entry. |
| `core` | `QuantumRegister` (pure-state simulation), `DensityMatrix` (mixed-state/noise simulation), `QuantumCircuit`, algorithm primitives, and benchmarking. |
| `xeb` | Hardware-calibrated noise (`HardwareCalibration`) and cross-entropy benchmark validation (`run_xeb_demo`). |
| `reservoir` | Quantum reservoir computing (`QuantumReservoir`, `QuantumReservoirComputer`). |

`core`, `xeb`, and `reservoir` depend only on `complex` and on each other's public API; there is no dependency on internal representation across module boundaries.

## Noise model

Depolarizing and amplitude-damping channels are implemented as qubit-local Kraus operators rather than dense embedded-operator conjugation, giving `O(d^2)` cost per channel application instead of `O(d^4)`. This is what keeps density-matrix simulation with per-gate noise practical at 8+ qubits.

Noise levels are not required to be arbitrary. `HardwareCalibration` provides fidelity figures for real, published hardware — currently the Quantinuum Helios system as benchmarked by Sandia National Laboratories and published in Nature (June 2026) — converted to the corresponding depolarizing-channel error probability using the standard fidelity-to-depolarizing relationship from randomized benchmarking literature:

```
p = (1 - F) * d / (d - 1)
```

where `F` is the average gate fidelity and `d` is the dimension of the gate (`d = 2` for single-qubit gates, `d = 4` for two-qubit gates).

Simulated noise can be validated against the ideal (noiseless) circuit distribution using linear cross-entropy benchmarking (XEB), following Arute et al., *Quantum supremacy using a programmable superconducting processor*, Nature 574, 505–510 (2019) — the same estimator used to validate real quantum hardware against classical simulation.

## Performance

State-vector gate-chain and QFT throughput, measured on this crate's benchmark suite:

| Qubits | Hadamard chain | CNOT chain | QFT |
|---|---|---|---|
| 4 | 5.2 µs | 2.9 µs | 15.2 µs |
| 8 | 297.6 µs | 72.6 µs | 506.4 µs |
| 12 | 4.9 ms | 1.7 ms | 11.8 ms |

Run the benchmark suite directly with `sirraya-qutub benchmark <qubits>`, or via `QuantumBenchmark::run_comprehensive_benchmark()`.

## Testing

```sh
cargo test
```

The test suite includes correctness checks against known closed-form results: Bell-state and GHZ-state distributions, purity and trace preservation under noise channels, von Neumann entropy against analytically known values (including the maximally mixed state), and QFT round-trip identity.

## Limitations

- Maximum register size is 16 qubits (2^16-dimensional state vector), configurable via `MAX_QUBITS` in `complex`.
- Density-matrix simulation is dense; memory scales as `O(4^n)` for an `n`-qubit system regardless of circuit sparsity.
- `von_neumann_entropy` for systems larger than 2 qubits diagonalizes the full density matrix via an internal eigenvalue solver rather than a specialized sparse method.

## License

See `LICENSE` for details.
