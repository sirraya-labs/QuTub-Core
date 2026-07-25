# Sirraya QuTub

**A quantum circuit and noise simulator built from scratch in Rust — pure-state and density-matrix simulation, entanglement and information-theoretic measures, hardware-calibrated noise validation, and quantum reservoir computing.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust edition](https://img.shields.io/badge/edition-2021-orange.svg)
![rustc](https://img.shields.io/badge/rustc-1.75%2B-orange.svg)

Sirraya QuTub implements the full simulation stack itself — complex arithmetic, a pure-state vector simulator, a mixed-state density-matrix simulator with physical noise channels, and a quantum reservoir computer — without wrapping an existing simulator framework. Every non-trivial numerical routine (Jacobi diagonalization, matrix square roots via the complex-to-real block embedding, Kraus-operator channel application, entanglement measures) is implemented in-crate and checked against closed-form results in the test suite, not treated as a black box.

---

## Why this exists

Most from-scratch simulators stop at "it runs a circuit." Sirraya QuTub is built around three additional commitments:

1. **Noise you can trust.** Depolarizing and amplitude-damping channels aren't parameterized by an arbitrary knob — `HardwareCalibration` derives them from published, cited hardware fidelity figures using the standard randomized-benchmarking fidelity-to-depolarizing relation, and the result is validated against the ideal circuit via linear XEB, the same estimator used to validate real quantum processors.
2. **Entanglement you can quantify, not just prepare.** Beyond state preparation, the crate computes partial traces, mixed-state fidelity, concurrence, and negativity — the standard literature tools for actually characterizing entanglement, not just producing it.
3. **Correctness that's tested against math, not vibes.** Every gate and channel is checked against its closed-form matrix, not just "does the circuit run." See [Correctness & references](#correctness--references) below.

---

## Features

**Simulation core**
- Full single- and multi-qubit gate set: Hadamard, Pauli (X/Y/Z), S/T (and inverses), arbitrary single-qubit rotations (RX, RY, RZ), the two-qubit Ising-coupling rotations (RXX, RYY, RZZ) used in variational ansätze, CNOT, controlled-Z, controlled-phase, SWAP, CSWAP, Toffoli, and generalized multi-controlled-X/Z.
- Density-matrix simulation with physical noise channels: depolarizing and amplitude damping, implemented as qubit-local Kraus operators.
- Quantum Fourier Transform and its inverse.
- Standard algorithm building blocks: Deutsch–Jozsa and Grover iteration.
- QASM 2.0 circuit export for interoperability with other tooling.

**Entanglement & information-theoretic tools**
- Partial trace (`DensityMatrix::partial_trace`) for reduced density matrices of any subsystem.
- Von Neumann entropy, including the analytic closed form for 2×2 reduced states.
- Mixed-state (Uhlmann) fidelity and pure-state fidelity / trace distance.
- Concurrence (two-qubit entanglement) and negativity (any bipartition).
- Kraus-operator completeness validation and full density-matrix validity checks (trace-1, Hermitian, positive semi-definite).
- General Pauli-string expectation values for VQE-style observable estimation, alongside the single-qubit Pauli-Z case.

**Noise validation**
- Noise channels calibrated to published, cited hardware fidelity figures rather than an arbitrary noise parameter.
- Linear cross-entropy benchmarking (XEB), the same fidelity estimator used to validate real quantum processors against classical simulation.

**Quantum reservoir computing**
- A fixed, small-world-connected qubit network used as a high-dimensional nonlinear dynamical system, paired with a trained linear readout.
- Feature-importance reporting, so the trained readout's weights are inspectable rather than opaque.

**Tooling**
- A command-line interface for reservoir training, prediction, and benchmarking.
- A benchmark suite reporting gate-chain and QFT throughput across qubit counts.

---

## Correctness & references

Every non-trivial routine below is implemented against a specific, citable source, and has a corresponding unit test checking it against a closed-form or known result — not just "it doesn't panic."

| Component | Reference |
|---|---|
| Depolarizing / amplitude-damping channels, Kraus formalism | Nielsen & Chuang, *Quantum Computation and Quantum Information*, Ch. 8 |
| Kraus completeness relation (Σ Kₖ†Kₖ = I) | Nielsen & Chuang, Theorem 8.3 |
| Density matrix validity (trace-1, Hermitian, PSD) | Nielsen & Chuang, §2.4 |
| Partial trace / reduced density matrices | Nielsen & Chuang, §2.4.3 |
| Von Neumann entropy, Hermitian eigendecomposition via the real-block embedding | Nielsen & Chuang, §2.2; Golub & Van Loan, *Matrix Computations*, for the Jacobi rotation formulas |
| Mixed-state (Uhlmann) fidelity | Uhlmann (1976); Jozsa, *J. Mod. Opt.* 41(12), 2315–2323 (1994); Nielsen & Chuang eq. 9.53 |
| Concurrence | Wootters, *Phys. Rev. Lett.* 80, 2245 (1998) |
| Negativity / PPT criterion | Peres, *Phys. Rev. Lett.* 77, 1413 (1996); Vidal & Werner, *Phys. Rev. A* 65, 032314 (2002) |
| GHZ vs. W entanglement classes | Dür, Vidal & Cirac, *Phys. Rev. A* 62, 062314 (2000) |
| Linear XEB validation | Arute et al., *Quantum supremacy using a programmable superconducting processor*, Nature 574, 505–510 (2019) |
| Hardware noise calibration | Quantinuum Helios system, benchmarked by Sandia National Laboratories, published in Nature (June 2026) |

The test suite (`cargo test`) currently runs 45 tests across the library and binary targets, including: Bell/GHZ/W-state distributions and amplitudes; purity, trace, and Hermiticity preservation under every noise channel; von Neumann entropy against analytic values (including the maximally mixed state); RX vs. RY phase-convention regression tests; trace-distance invariance under global phase; concurrence and negativity on both entangled and product states; and QFT round-trip identity.

---

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

---

## Quick start

### Pure-state simulation

```rust
use sirraya_qutub::quantum_simulator::{create_bell_state, create_w_state, QuantumRegister};

fn main() -> Result<(), String> {
    let bell = create_bell_state()?;
    let distribution = bell.get_probability_distribution();
    // {"00": 0.5, "11": 0.5}

    // The W state — unlike GHZ, its entanglement survives losing any one qubit.
    let w = create_w_state(3)?;

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

    // A pure Bell state starts at purity 1.0, and passes full validity checks.
    assert!(density.is_pure());
    assert!(density.is_valid().is_ok());

    density.apply_depolarizing_channel(0.2, 0)?;
    // Purity drops to ~0.653 under 20% single-qubit depolarizing noise.

    Ok(())
}
```

### Quantifying entanglement

```rust
use sirraya_qutub::quantum_simulator::create_bell_state;

fn main() -> Result<(), String> {
    let bell = create_bell_state()?;
    let density = bell.to_density_matrix()?;

    // Concurrence of a Bell state is 1 (maximally entangled).
    let c = density.concurrence()?;

    // Negativity across the 0|1 cut is 0.5, certifying entanglement via the PPT criterion.
    let n = density.negativity(&[1])?;

    // Tracing out qubit 1 leaves qubit 0 maximally mixed (I/2) — the textbook
    // signature of a pure entangled state having a mixed reduced state.
    let reduced = density.partial_trace(&[0])?;
    assert!(!reduced.is_pure());

    // Mixed-state fidelity between two density matrices (reduces to the
    // pure-state overlap |<psi|phi>|^2 when both are pure).
    let f = density.fidelity(&density)?; // ~1.0

    println!("concurrence={c:.4} negativity={n:.4} fidelity={f:.4}");
    Ok(())
}
```

### Variational-circuit gates and observables

```rust
use sirraya_qutub::quantum_simulator::{create_bell_state, PauliOp, QuantumRegister};

fn main() -> Result<(), String> {
    let mut register = QuantumRegister::new(2)?;
    // Ising-coupling rotations used in QAOA mixers and VQE ansätze.
    register.apply_rxx(0, 1, std::f64::consts::PI / 4.0)?;
    register.apply_rzz(0, 1, 0.3)?;

    // Sparse Pauli-string expectation values, e.g. <psi| X_0 X_1 |psi>.
    let bell = create_bell_state()?;
    let xx = bell.expectation_value_pauli_string(&[(0, PauliOp::X), (1, PauliOp::X)])?;
    // xx == 1.0 for the Bell state

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

---

## Command-line interface

```text
sirraya-qutub reservoir demo                           Run a 4-qubit reservoir demonstration
sirraya-qutub reservoir train <qubits>                 Train an n-qubit reservoir on synthetic data
sirraya-qutub reservoir predict <qubits> <values...>   Predict with an n-qubit reservoir
sirraya-qutub reservoir info <qubits>                  Show reservoir configuration and capacity
sirraya-qutub benchmark <qubits>                       Run the performance benchmark suite
sirraya-qutub help                                     Show usage information
```

Running the binary with no arguments executes the full demonstration suite: state preparation, noise channels, gate circuits, QFT round-trip, measurement, benchmarking, QASM export, and reservoir computing.

---

## Architecture

The simulator is organized into four modules under `quantum_simulator`:

| Module | Contents |
|---|---|
| `complex` | Complex-number arithmetic underlying every amplitude and matrix entry. |
| `core` | `QuantumRegister` (pure-state simulation), `DensityMatrix` (mixed-state/noise simulation and entanglement measures), `QuantumCircuit`, algorithm primitives, and benchmarking. |
| `xeb` | Hardware-calibrated noise (`HardwareCalibration`) and cross-entropy benchmark validation (`run_xeb_demo`). |
| `reservoir` | Quantum reservoir computing (`QuantumReservoir`, `QuantumReservoirComputer`). |

`core`, `xeb`, and `reservoir` depend only on `complex` and on each other's public API; there is no dependency on internal representation across module boundaries.

---

## Noise model

Depolarizing and amplitude-damping channels are implemented as qubit-local Kraus operators rather than dense embedded-operator conjugation, giving `O(d²)` cost per channel application instead of `O(d⁴)`. This is what keeps density-matrix simulation with per-gate noise practical at 8+ qubits. Any set of Kraus operators — including custom channels — can be checked against the completeness relation via `DensityMatrix::validate_kraus_operators` before being trusted.

Noise levels are not required to be arbitrary. `HardwareCalibration` provides fidelity figures for real, published hardware — currently the Quantinuum Helios system as benchmarked by Sandia National Laboratories and published in Nature (June 2026) — converted to the corresponding depolarizing-channel error probability using the standard fidelity-to-depolarizing relationship from randomized benchmarking literature:

```
p = (1 - F) * d / (d - 1)
```

where `F` is the average gate fidelity and `d` is the dimension of the gate (`d = 2` for single-qubit gates, `d = 4` for two-qubit gates).

Simulated noise can be validated against the ideal (noiseless) circuit distribution using linear cross-entropy benchmarking (XEB), following Arute et al., *Quantum supremacy using a programmable superconducting processor*, Nature 574, 505–510 (2019) — the same estimator used to validate real quantum hardware against classical simulation.

---

## Performance

State-vector gate-chain and QFT throughput, measured on this crate's benchmark suite:

| Qubits | Hadamard chain | CNOT chain | QFT |
|---|---|---|---|
| 4 | 5.2 µs | 2.9 µs | 15.2 µs |
| 8 | 297.6 µs | 72.6 µs | 506.4 µs |
| 12 | 4.9 ms | 1.7 ms | 11.8 ms |

Run the benchmark suite directly with `sirraya-qutub benchmark <qubits>`, or via `QuantumBenchmark::run_comprehensive_benchmark()`.

Density-matrix operations that require a full eigendecomposition (`von_neumann_entropy` above 2 qubits, mixed-state `fidelity`, `concurrence`, and `negativity`) use a classical Jacobi eigensolver on the equivalent real symmetric matrix. This is numerically simple and robust at the qubit counts this crate targets, but scales as roughly `O(n⁴)` in the density-matrix dimension — see [Limitations](#limitations).

---

## Testing

```sh
cargo test
```

The suite includes correctness checks against known closed-form results:
- Bell-state, GHZ-state, and W-state distributions and amplitudes.
- Purity, trace, and Hermiticity preservation under every noise channel.
- Von Neumann entropy against analytically known values (including the maximally mixed state).
- Gate-matrix regression tests distinguishing RX from RY by their complex phase, and RXX/RYY/RZZ against their known action on basis states.
- Trace distance and mixed-state fidelity, including invariance under global phase.
- Concurrence and negativity on both maximally entangled and product states.
- Kraus-operator completeness validation, including the exact operators used by `apply_amplitude_damping`.
- QFT round-trip identity.

---

## Limitations

- Maximum register size is 16 qubits (2¹⁶-dimensional state vector), configurable via `MAX_QUBITS` in `complex`.
- Density-matrix simulation is dense; memory scales as `O(4ⁿ)` for an `n`-qubit system regardless of circuit sparsity.
- `von_neumann_entropy`, mixed-state `fidelity`, and `concurrence` for systems larger than a couple of qubits diagonalize the full density matrix via an internal Jacobi eigensolver rather than a specialized sparse or iterative method (e.g. Lanczos); this becomes computationally expensive past roughly 10 qubits.
- `partial_trace` and `negativity` are `O(d²)` in the density-matrix dimension (i.e. `O(4ⁿ)` in qubit count), which is fine at the qubit counts this crate targets but is not intended for large registers.
- `concurrence` is currently defined only for two-qubit density matrices, matching Wootters' original formulation; `negativity` supports arbitrary bipartitions of larger systems.
- Measurement outcomes use `rand::thread_rng()` and are not currently seedable, so individual runs are not reproducible bit-for-bit (though statistical properties are, of course, reproducible in aggregate).

## Roadmap

- Seeded RNG option for reproducible measurement outcomes.
- `QuantumCircuit` builder methods currently swallow per-gate errors (`&mut Self` chaining rather than `Result` propagation); a parallel fallible builder API is planned rather than a breaking change to the existing one.
- Sparse or iterative eigensolvers (e.g. Lanczos) to extend entropy/fidelity/concurrence computation past the current qubit-count ceiling.

---

## License

See `LICENSE` for details.