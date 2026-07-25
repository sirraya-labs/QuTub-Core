//! Core quantum simulation primitives: pure-state simulation
//! (`QuantumRegister`), mixed-state/noise simulation (`DensityMatrix`),
//! a convenience circuit builder (`QuantumCircuit`), standard algorithm
//! building blocks (Bell/GHZ state prep, QFT, Deutsch-Jozsa, Grover),
//! and throughput benchmarking (`QuantumBenchmark`).

use crate::complex::{Complex, EPSILON, MAX_QUBITS};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::{Duration, Instant};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A single-qubit Pauli operator, used to specify sparse Pauli-string
/// observables for `QuantumRegister::expectation_value_pauli_string`.
/// Qubits not mentioned in a Pauli string are implicitly `I`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliOp {
    I,
    X,
    Y,
    Z,
}

#[derive(Debug, Clone)]
pub struct DensityMatrix {
    matrix: Vec<Vec<Complex>>,
    num_qubits: usize,
    dimension: usize,
}

impl DensityMatrix {
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        if num_qubits == 0 {
            return Err("Number of qubits must be at least 1".to_string());
        }
        if num_qubits > MAX_QUBITS {
            return Err(format!("Number of qubits exceeds maximum of {}", MAX_QUBITS));
        }

        let dimension = 1 << num_qubits;
        let mut matrix = vec![vec![Complex::zero(); dimension]; dimension];
        matrix[0][0] = Complex::one();

        Ok(Self {
            matrix,
            num_qubits,
            dimension,
        })
    }

    pub fn from_state_vector(state_vector: &[Complex]) -> Result<Self, String> {
        let dimension = state_vector.len();
        if dimension.count_ones() != 1 {
            return Err("State vector dimension must be a power of 2".to_string());
        }

        let num_qubits = dimension.trailing_zeros() as usize;
        let mut matrix = vec![vec![Complex::zero(); dimension]; dimension];

        for i in 0..dimension {
            for j in 0..dimension {
                matrix[i][j] = state_vector[i] * state_vector[j].conjugate();
            }
        }

        Ok(Self {
            matrix,
            num_qubits,
            dimension,
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn get_matrix(&self) -> &[Vec<Complex>] {
        &self.matrix
    }

    pub fn apply_unitary(&mut self, unitary: &[Complex]) -> Result<(), String> {
        if unitary.len() != self.dimension * self.dimension {
            return Err("Unitary matrix dimension mismatch".to_string());
        }

        let mut new_matrix = vec![vec![Complex::zero(); self.dimension]; self.dimension];

        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    for l in 0..self.dimension {
                        new_matrix[i][j] = new_matrix[i][j] + 
                            unitary[i * self.dimension + k] * 
                            self.matrix[k][l] * 
                            unitary[j * self.dimension + l].conjugate();
                    }
                }
            }
        }

        self.matrix = new_matrix;
        Ok(())
    }

    fn validate_qubit_index(&self, qubit: usize) -> Result<(), String> {
        if qubit >= self.num_qubits {
            Err(format!("Qubit index {} out of bounds for {} qubits", qubit, self.num_qubits))
        } else {
            Ok(())
        }
    }

    pub fn apply_depolarizing_channel(&mut self, probability: f64, qubit: usize) -> Result<(), String> {
        if probability < 0.0 || probability > 1.0 {
            return Err("Probability must be between 0 and 1".to_string());
        }
        self.validate_qubit_index(qubit)?;

        if probability < EPSILON {
            return Ok(());
        }

        let identity_weight = 1.0 - probability;
        let pauli_weight = probability / 3.0;

        let mut new_matrix = vec![vec![Complex::zero(); self.dimension]; self.dimension];

        // Apply identity
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                new_matrix[i][j] = new_matrix[i][j] + self.matrix[i][j].scale(identity_weight);
            }
        }

        let x = [Complex::zero(), Complex::one(), Complex::one(), Complex::zero()];
        let y = [Complex::zero(), -Complex::i(), Complex::i(), Complex::zero()];
        let z = [Complex::one(), Complex::zero(), Complex::zero(), -Complex::one()];

        self.apply_single_qubit_kraus(&mut new_matrix, qubit, x, pauli_weight);
        self.apply_single_qubit_kraus(&mut new_matrix, qubit, y, pauli_weight);
        self.apply_single_qubit_kraus(&mut new_matrix, qubit, z, pauli_weight);

        self.matrix = new_matrix;
        Ok(())
    }

    pub fn apply_amplitude_damping(&mut self, gamma: f64, qubit: usize) -> Result<(), String> {
        if gamma < 0.0 || gamma > 1.0 {
            return Err("Gamma must be between 0 and 1".to_string());
        }
        self.validate_qubit_index(qubit)?;

        let k0 = [
            Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new((1.0 - gamma).sqrt(), 0.0),
        ];
        let k1 = [
            Complex::new(0.0, 0.0), Complex::new(gamma.sqrt(), 0.0),
            Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
        ];

        let mut new_matrix = vec![vec![Complex::zero(); self.dimension]; self.dimension];
        self.apply_single_qubit_kraus(&mut new_matrix, qubit, k0, 1.0);
        self.apply_single_qubit_kraus(&mut new_matrix, qubit, k1, 1.0);

        self.matrix = new_matrix;
        Ok(())
    }

    /// Accumulate `weight * K rho K^dagger` into `result`, for a 2x2 Kraus
    /// operator `k` (row-major: [k00, k01, k10, k11]) acting on a single
    /// `qubit` embedded in this larger system.
    ///
    /// Runs in O(d^2): each basis state only ever mixes with the single
    /// "partner" state that differs in the target qubit's bit, so each
    /// output entry is a sum of at most 4 terms (the qubit's local 2x2
    /// index combinations) rather than a full d x d embedded-operator
    /// multiply.
    fn apply_single_qubit_kraus(
        &self,
        result: &mut [Vec<Complex>],
        qubit: usize,
        k: [Complex; 4],
        weight: f64,
    ) {
        let dim = self.dimension;
        let mask = 1usize << qubit;

        for r in 0..dim {
            let r_base = r & !mask;
            let rb = ((r & mask) != 0) as usize;
            for c in 0..dim {
                let c_base = c & !mask;
                let cb = ((c & mask) != 0) as usize;

                let mut sum = Complex::zero();
                for rb2 in 0..2usize {
                    let k_r = k[rb * 2 + rb2];
                    if k_r.magnitude_squared() < EPSILON {
                        continue;
                    }
                    let r_src = r_base | (rb2 << qubit);
                    for cb2 in 0..2usize {
                        let k_c = k[cb * 2 + cb2];
                        if k_c.magnitude_squared() < EPSILON {
                            continue;
                        }
                        let c_src = c_base | (cb2 << qubit);
                        sum = sum + k_r * self.matrix[r_src][c_src] * k_c.conjugate();
                    }
                }
                result[r][c] = result[r][c] + sum.scale(weight);
            }
        }
    }

    /// Checks the completeness relation `Sum_k K_k^dagger K_k = I` for a
    /// set of 2x2 single-qubit Kraus operators (each row-major:
    /// `[k00, k01, k10, k11]`). This is the defining condition for a
    /// physically valid quantum channel (Nielsen & Chuang, *Quantum
    /// Computation and Quantum Information*, Theorem 8.3) -- a channel
    /// whose Kraus operators violate it does not preserve trace, and will
    /// silently turn a valid density matrix into one with trace != 1.
    pub fn validate_kraus_operators(operators: &[[Complex; 4]]) -> Result<(), String> {
        if operators.is_empty() {
            return Err("At least one Kraus operator is required".to_string());
        }

        // sigma = sum_k K_k^dagger K_k, a 2x2 matrix, row-major.
        let mut sigma = [Complex::zero(); 4];
        for k in operators {
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = Complex::zero();
                    for l in 0..2 {
                        let k_dag_il = k[l * 2 + i].conjugate();
                        let k_lj = k[l * 2 + j];
                        sum = sum + k_dag_il * k_lj;
                    }
                    sigma[i * 2 + j] = sigma[i * 2 + j] + sum;
                }
            }
        }

        let diag_ok = (sigma[0] - Complex::one()).magnitude() < 1e-9
            && (sigma[3] - Complex::one()).magnitude() < 1e-9;
        let off_diag_ok = sigma[1].magnitude() < 1e-9 && sigma[2].magnitude() < 1e-9;

        if diag_ok && off_diag_ok {
            Ok(())
        } else {
            Err(format!(
                "Kraus operators do not satisfy the completeness relation \
                 Sum K^dagger K = I (got diagonal [{}, {}], off-diagonal [{}, {}])",
                sigma[0], sigma[3], sigma[1], sigma[2]
            ))
        }
    }

    /// Apply a single-qubit unitary `u` (row-major 2x2: [u00,u01,u10,u11])
    /// embedded at `target`: rho -> U rho U^dagger. A unitary conjugation is
    /// just a single Kraus operator with weight 1.0, so this reuses the
    /// O(d^2) qubit-local machinery in `apply_single_qubit_kraus` rather than
    /// building a dense d x d embedded unitary and doing an O(d^4) multiply.
    fn apply_single_qubit_unitary(&mut self, target: usize, u: [Complex; 4]) -> Result<(), String> {
        self.validate_qubit_index(target)?;
        let mut new_matrix = vec![vec![Complex::zero(); self.dimension]; self.dimension];
        self.apply_single_qubit_kraus(&mut new_matrix, target, u, 1.0);
        self.matrix = new_matrix;
        Ok(())
    }

    /// Embed a k-qubit gate (row-major 2^k x 2^k) into a full dimension x
    /// dimension unitary acting on `targets` (sorted ascending; bit position
    /// `p` corresponds to the p-th qubit once sorted). Dense O(d^2) to build,
    /// and `apply_unitary` on top of it is O(d^4) -- only used as a fallback
    /// for multi-qubit embedded gates, which nothing in this codebase
    /// currently exercises (the single-qubit case below has a fast path).
    fn create_embedded_gate_matrix(&self, targets: &[usize], gate: &[Complex]) -> Result<Vec<Complex>, String> {
        let num_targets = targets.len();
        if num_targets == 0 {
            return Err("At least one target qubit is required".to_string());
        }

        let gate_dim = 1 << num_targets;
        if gate.len() != gate_dim * gate_dim {
            return Err(format!(
                "Gate matrix must have dimension {}x{} for {} target qubit(s)",
                gate_dim, gate_dim, num_targets
            ));
        }

        for &t in targets {
            self.validate_qubit_index(t)?;
        }

        let mut sorted_targets = targets.to_vec();
        sorted_targets.sort();
        sorted_targets.dedup();
        if sorted_targets.len() != targets.len() {
            return Err("Duplicate qubit indices in targets".to_string());
        }

        let target_mask: usize = sorted_targets.iter().map(|&t| 1 << t).sum();
        let mut full_unitary = vec![Complex::zero(); self.dimension * self.dimension];

        for base_index in 0..self.dimension {
            if base_index & target_mask != 0 {
                continue;
            }

            for out_bits in 0..gate_dim {
                let mut out_index = base_index;
                for (bit_pos, &t) in sorted_targets.iter().enumerate() {
                    if (out_bits >> bit_pos) & 1 != 0 {
                        out_index |= 1 << t;
                    }
                }

                for in_bits in 0..gate_dim {
                    let mut in_index = base_index;
                    for (bit_pos, &t) in sorted_targets.iter().enumerate() {
                        if (in_bits >> bit_pos) & 1 != 0 {
                            in_index |= 1 << t;
                        }
                    }

                    full_unitary[out_index * self.dimension + in_index] = gate[out_bits * gate_dim + in_bits];
                }
            }
        }

        Ok(full_unitary)
    }

    /// Apply an arbitrary unitary embedded on the given target qubits:
    /// rho -> U rho U^dagger. Single-qubit targets (the only case this
    /// codebase currently uses, e.g. in `run_xeb_demo`) take the fast O(d^2)
    /// path; anything else falls back to the dense O(d^4) embedding.
    pub fn apply_unitary_embedded(&mut self, gate: &[Complex], targets: &[usize]) -> Result<(), String> {
        match targets {
            [q] => {
                if gate.len() != 4 {
                    return Err("Single-qubit gate must be 2x2".to_string());
                }
                self.apply_single_qubit_unitary(*q, [gate[0], gate[1], gate[2], gate[3]])
            }
            _ => {
                let full_unitary = self.create_embedded_gate_matrix(targets, gate)?;
                self.apply_unitary(&full_unitary)
            }
        }
    }

    /// Apply CNOT(control, target) embedded in this system: rho -> U rho
    /// U^dagger. CNOT is a permutation unitary (each basis state maps to
    /// exactly one other), so conjugation by it is itself just a
    /// permutation of matrix entries -- O(d^2), not the O(d^4) a generic
    /// `apply_unitary` conjugation would cost.
    pub fn apply_cnot_embedded(&mut self, control: usize, target: usize) -> Result<(), String> {
        self.validate_qubit_index(control)?;
        self.validate_qubit_index(target)?;
        if control == target {
            return Err("Control and target qubits must be different".to_string());
        }

        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        let dim = self.dimension;

        let perm = |i: usize| -> usize {
            if (i & control_mask) != 0 { i ^ target_mask } else { i }
        };

        let mut new_matrix = vec![vec![Complex::zero(); dim]; dim];
        for i in 0..dim {
            let pi = perm(i);
            for j in 0..dim {
                new_matrix[pi][perm(j)] = self.matrix[i][j];
            }
        }

        self.matrix = new_matrix;
        Ok(())
    }

    pub fn trace(&self) -> f64 {
        let mut trace = 0.0;
        for i in 0..self.dimension {
            trace += self.matrix[i][i].real();
        }
        trace
    }

    pub fn purity(&self) -> f64 {
        let mut purity = 0.0;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                purity += (self.matrix[i][j] * self.matrix[j][i]).real();
            }
        }
        purity
    }

    pub fn is_pure(&self) -> bool {
        (self.purity() - 1.0).abs() < EPSILON
    }

    pub fn von_neumann_entropy(&self) -> f64 {
        // For 2x2 density matrices, use analytical formula
        if self.dimension == 2 {
            let a = self.matrix[0][0].real();
            let b = self.matrix[0][1];
            let c = self.matrix[1][0];
            let d = self.matrix[1][1].real();
            
            // Check if diagonal
            if b.magnitude() < EPSILON {
                let mut entropy = 0.0;
                if a > EPSILON { entropy -= a * a.log2(); }
                if d > EPSILON { entropy -= d * d.log2(); }
                return entropy;
            }
            
            // Compute eigenvalues of 2x2 matrix
            let trace = a + d;
            let det = a * d - (b * c).real();
            
            let discriminant = trace * trace - 4.0 * det;
            if discriminant < 0.0 {
                return 0.0; // Shouldn't happen for valid density matrices
            }
            
            let sqrt_disc = discriminant.sqrt();
            let lambda1 = (trace + sqrt_disc) / 2.0;
            let lambda2 = (trace - sqrt_disc) / 2.0;
            
            let mut entropy = 0.0;
            if lambda1 > EPSILON { entropy -= lambda1 * lambda1.log2(); }
            if lambda2 > EPSILON { entropy -= lambda2 * lambda2.log2(); }
            
            return entropy.max(0.0); // Ensure non-negative
        }
        
        // For larger matrices, diagonalize the full Hermitian density
        // matrix and sum the von Neumann entropy formula over its entire
        // eigenspectrum.
        self.hermitian_eigenvalues()
            .iter()
            .filter(|&&lambda| lambda > EPSILON)
            .map(|&lambda| -lambda * lambda.log2())
            .sum()
    }

    /// Full eigenvalue spectrum of this (Hermitian) density matrix.
    ///
    /// A complex Hermitian matrix H = A + iB (A real symmetric, B real
    /// skew-symmetric) has the same eigenvalues -- each with doubled
    /// multiplicity -- as the real symmetric 2n x 2n block matrix
    /// M = [[A, -B], [B, A]]. This lets us diagonalize via the classical
    /// cyclic Jacobi eigenvalue algorithm on a real symmetric matrix
    /// (numerically simple and robust for the small dimensions this
    /// simulator targets) and then recover the n eigenvalues of H by
    /// taking every duplicated pair once.
    fn hermitian_eigenvalues(&self) -> Vec<f64> {
        Self::hermitian_eigenvalues_of(&self.matrix)
    }

    /// Full eigenvalue spectrum of an arbitrary Hermitian matrix (not
    /// necessarily this DensityMatrix's own `matrix` -- used internally
    /// for computations that involve two different density matrices,
    /// such as mixed-state fidelity and concurrence).
    ///
    /// A complex Hermitian matrix H = A + iB (A real symmetric, B real
    /// skew-symmetric) has the same eigenvalues -- each with doubled
    /// multiplicity -- as the real symmetric 2n x 2n block matrix
    /// M = [[A, -B], [B, A]]. This lets us diagonalize via the classical
    /// cyclic Jacobi eigenvalue algorithm on a real symmetric matrix
    /// (numerically simple and robust for the small dimensions this
    /// simulator targets) and then recover the n eigenvalues of H by
    /// taking every duplicated pair once.
    fn hermitian_eigenvalues_of(matrix: &[Vec<Complex>]) -> Vec<f64> {
        let n = matrix.len();
        let m = 2 * n;
        let mut real_block = vec![vec![0.0f64; m]; m];

        for i in 0..n {
            for j in 0..n {
                let a = matrix[i][j].real(); // symmetric part
                let b = matrix[i][j].imag(); // skew-symmetric part
                real_block[i][j] = a;
                real_block[i][j + n] = -b;
                real_block[i + n][j] = b;
                real_block[i + n][j + n] = a;
            }
        }

        let (mut eigenvalues, _) = Self::jacobi_eigen(&mut real_block);
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Each true eigenvalue of H appears twice (adjacently, once sorted)
        // in the spectrum of the doubled real block; keep one of each pair.
        eigenvalues.into_iter().step_by(2).collect()
    }

    /// Matrix square root of an arbitrary Hermitian positive-semidefinite
    /// matrix. Reuses the same complex-to-real block embedding as
    /// `hermitian_eigenvalues_of`: because that embedding is an algebra
    /// homomorphism, it commutes with any analytic functional calculus
    /// (in particular the square root), so `block(sqrt(H)) = sqrt(block(H))`.
    /// That lets the square root be computed with only real-valued Jacobi
    /// diagonalization -- diagonalize the real block, take the elementwise
    /// square root of its (non-negative, since H is PSD) eigenvalues, and
    /// reconstruct via `V * diag(sqrt(lambda)) * V^T` -- then read the
    /// complex result straight off the block structure.
    fn hermitian_sqrt_of(matrix: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
        let n = matrix.len();
        let m = 2 * n;
        let mut real_block = vec![vec![0.0f64; m]; m];

        for i in 0..n {
            for j in 0..n {
                let a = matrix[i][j].real();
                let b = matrix[i][j].imag();
                real_block[i][j] = a;
                real_block[i][j + n] = -b;
                real_block[i + n][j] = b;
                real_block[i + n][j + n] = a;
            }
        }

        let (eigenvalues, eigenvectors) = Self::jacobi_eigen(&mut real_block);

        // Reconstruct sqrt(M) = V * diag(sqrt(max(lambda, 0))) * V^T.
        // Negative eigenvalues (which shouldn't occur for a true density
        // matrix, only from floating-point error) are clamped to zero
        // rather than propagated as NaN.
        let mut sqrt_block = vec![vec![0.0f64; m]; m];
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += eigenvectors[i][k] * eigenvalues[k].max(0.0).sqrt() * eigenvectors[j][k];
                }
                sqrt_block[i][j] = sum;
            }
        }

        // block(sqrt(H)) has the same [[A', -B'], [B', A']] structure as
        // any block(complex matrix), so sqrt(H) = A' + iB' is read
        // straight off the top-left and bottom-left blocks.
        let mut result = vec![vec![Complex::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                result[i][j] = Complex::new(sqrt_block[i][j], sqrt_block[i + n][j]);
            }
        }
        result
    }

    /// Dense n x n complex matrix multiply. Used internally by
    /// mixed-state fidelity and concurrence, both of which are only ever
    /// called on small (single- or two-qubit) matrices, so the O(n^3)
    /// cost here is not a concern.
    fn complex_matmul(a: &[Vec<Complex>], b: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
        let n = a.len();
        let mut result = vec![vec![Complex::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex::zero();
                for k in 0..n {
                    sum = sum + a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        result
    }

    /// Classical cyclic Jacobi eigenvalue algorithm for a real symmetric
    /// matrix. Repeatedly zeroes the largest off-diagonal element via a
    /// Givens rotation until the matrix is (numerically) diagonal, then
    /// returns the diagonal (the eigenvalues). O(n^3) per sweep; a handful
    /// of sweeps is enough to converge for the small matrices used here.
    /// Eigendecomposition of a real symmetric matrix via cyclic Jacobi
    /// rotations (Golub & Van Loan, *Matrix Computations*, Sec. 8.4).
    ///
    /// This sweeps through every off-diagonal pair `(p, q)` in fixed
    /// order and zeroes each one in turn, rather than searching for the
    /// single largest off-diagonal entry before every rotation (the
    /// "classical" Jacobi variant). That distinction matters
    /// asymptotically: classical Jacobi pays an O(n^2) search before each
    /// O(n) rotation update, and needs O(n^2) rotations to converge, for
    /// O(n^4) total. Cyclic Jacobi does a fixed O(n^2) rotations per
    /// sweep (no search) at O(n) each -- O(n^3) per sweep -- and, once
    /// past the first couple of sweeps, converges quadratically (each
    /// sweep roughly squares the off-diagonal residual), so a small
    /// constant number of sweeps suffices regardless of n. Net effect:
    /// O(n^3) instead of O(n^4), which is what actually determines
    /// whether `von_neumann_entropy`, `fidelity`, and `concurrence` stay
    /// usable as qubit count grows.
    fn jacobi_eigen(a: &mut [Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = a.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        // V accumulates the product of all rotations; its columns converge
        // to the orthonormal eigenvectors of the original matrix.
        let mut v = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            v[i][i] = 1.0;
        }

        // Scale-relative convergence threshold: EPSILON alone is far too
        // tight for larger matrices, since it doesn't account for the
        // accumulated floating-point error of O(n^2) rotations.
        let frobenius_norm: f64 = a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        let threshold = (frobenius_norm * 1e-12).max(1e-14);

        // Quadratic convergence means this is a generous cap, not a
        // typical iteration count -- most matrices at the sizes this
        // simulator targets converge in well under 20 sweeps.
        let max_sweeps = 100;

        for _sweep in 0..max_sweeps {
            // Off-diagonal Frobenius norm at the start of this sweep
            // (i.e. the end state of the previous one). Checked before
            // doing any rotations so a converged matrix exits in O(n^2)
            // rather than paying for another full sweep.
            let off_diag_sum: f64 = (0..n)
                .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
                .map(|(i, j)| a[i][j] * a[i][j])
                .sum();
            if off_diag_sum.sqrt() < threshold {
                break;
            }

            for p in 0..n {
                for q in (p + 1)..n {
                    // Already zero (or effectively so): rotating would be
                    // wasted work, and if a[p][p] == a[q][q] too this also
                    // sidesteps a 0/0 in the theta computation below.
                    if a[p][q].abs() < 1e-300 {
                        continue;
                    }

                    let (c, s) = if a[p][p] == a[q][q] {
                        // 45 degree rotation when diagonal entries are equal
                        (std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2 * a[p][q].signum())
                    } else {
                        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                        let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                        let c = 1.0 / (1.0 + t * t).sqrt();
                        (c, t * c)
                    };
                    Self::apply_jacobi_rotation(a, p, q, c, s);

                    // Accumulate the same rotation into V's columns p, q.
                    for i in 0..n {
                        let vip = v[i][p];
                        let viq = v[i][q];
                        v[i][p] = c * vip - s * viq;
                        v[i][q] = s * vip + c * viq;
                    }
                }
            }
        }

        let eigenvalues = (0..n).map(|i| a[i][i]).collect();
        (eigenvalues, v)
    }

    fn apply_jacobi_rotation(a: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64) {
        let n = a.len();
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = c * aip - s * aiq;
                a[p][i] = a[i][p];
                a[i][q] = s * aip + c * aiq;
                a[q][i] = a[i][q];
            }
        }
    }

    /// Reduced density matrix Tr_B(rho_AB): traces out every qubit not
    /// listed in `keep`, leaving the density matrix of the retained
    /// subsystem (Nielsen & Chuang, Sec. 2.4.3). This is the standard
    /// tool for entanglement entropy of a subsystem -- e.g. computing
    /// `register.to_density_matrix()?.partial_trace(&[0])?.von_neumann_entropy()`
    /// gives the real reduced-state entropy of qubit 0, as opposed to
    /// simulating decoherence via a depolarizing channel.
    ///
    /// Runtime is O(dimension^2), i.e. O(4^n) in the number of qubits --
    /// fine for the small registers this simulator targets, but not
    /// intended for anything past the ~10 qubit range where dense
    /// density-matrix operations already become the bottleneck elsewhere
    /// in this module.
    pub fn partial_trace(&self, keep: &[usize]) -> Result<DensityMatrix, String> {
        for &q in keep {
            if q >= self.num_qubits {
                return Err(format!("Qubit index {} out of bounds for {} qubits", q, self.num_qubits));
            }
        }

        let mut keep_sorted = keep.to_vec();
        keep_sorted.sort();
        keep_sorted.dedup();
        if keep_sorted.len() != keep.len() {
            return Err("Duplicate qubit indices in `keep`".to_string());
        }
        if keep_sorted.is_empty() {
            return Err("`keep` must retain at least one qubit".to_string());
        }

        let trace_out: Vec<usize> = (0..self.num_qubits)
            .filter(|q| !keep_sorted.contains(q))
            .collect();

        let extract = |idx: usize, qubits: &[usize]| -> usize {
            let mut out = 0usize;
            for (pos, &q) in qubits.iter().enumerate() {
                if (idx >> q) & 1 == 1 {
                    out |= 1 << pos;
                }
            }
            out
        };

        let k = keep_sorted.len();
        let reduced_dim = 1usize << k;
        let mut reduced = vec![vec![Complex::zero(); reduced_dim]; reduced_dim];

        for i in 0..self.dimension {
            for j in 0..self.dimension {
                // Partial trace sums over matching diagonal entries of the
                // traced-out subsystem: <i_keep, k| rho |j_keep, k> summed
                // over k, so only pairs that agree on the traced-out bits
                // contribute.
                if extract(i, &trace_out) != extract(j, &trace_out) {
                    continue;
                }
                let ri = extract(i, &keep_sorted);
                let rj = extract(j, &keep_sorted);
                reduced[ri][rj] = reduced[ri][rj] + self.matrix[i][j];
            }
        }

        Ok(DensityMatrix {
            matrix: reduced,
            num_qubits: k,
            dimension: reduced_dim,
        })
    }

    /// Validates that this is a mathematically well-formed density
    /// operator: trace 1, Hermitian, and positive semi-definite (Nielsen
    /// & Chuang, Sec. 2.4 -- the three defining properties of a density
    /// matrix). Intended as an opt-in sanity check -- e.g. after
    /// constructing a `DensityMatrix` from untrusted data, or in tests --
    /// rather than something called on every operation, since the PSD
    /// check requires a full eigenvalue decomposition.
    pub fn is_valid(&self) -> Result<(), String> {
        let trace = self.trace();
        if (trace - 1.0).abs() > 1e-6 {
            return Err(format!("Trace must be 1, got {}", trace));
        }

        for i in 0..self.dimension {
            for j in 0..self.dimension {
                let diff = self.matrix[i][j] - self.matrix[j][i].conjugate();
                if diff.magnitude() > 1e-6 {
                    return Err(format!(
                        "Matrix is not Hermitian: entry ({}, {}) = {} but conjugate of entry ({}, {}) = {}",
                        i, j, self.matrix[i][j], j, i, self.matrix[j][i].conjugate()
                    ));
                }
            }
        }

        let min_eigenvalue = self.hermitian_eigenvalues()
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        if min_eigenvalue < -1e-6 {
            return Err(format!(
                "Matrix is not positive semi-definite: smallest eigenvalue is {}",
                min_eigenvalue
            ));
        }

        Ok(())
    }

    /// Mixed-state (Uhlmann) fidelity: `F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2`
    /// (Uhlmann 1976; Jozsa, *J. Mod. Opt.* 1994; Nielsen & Chuang eq. 9.53).
    /// Reduces to the pure-state overlap `|<psi|phi>|^2` used by
    /// `QuantumRegister::fidelity` when both states happen to be pure.
    pub fn fidelity(&self, other: &Self) -> Result<f64, String> {
        if self.num_qubits != other.num_qubits {
            return Err("Density matrices must have the same number of qubits".to_string());
        }

        let sqrt_rho = Self::hermitian_sqrt_of(&self.matrix);
        let inner = Self::complex_matmul(&sqrt_rho, &other.matrix);
        let product = Self::complex_matmul(&inner, &sqrt_rho);
        let eigenvalues = Self::hermitian_eigenvalues_of(&product);

        let trace_sqrt: f64 = eigenvalues.iter().map(|&l| l.max(0.0).sqrt()).sum();
        Ok((trace_sqrt * trace_sqrt).min(1.0))
    }

    /// Wootters concurrence for a two-qubit density matrix (Wootters,
    /// *Phys. Rev. Lett.* 80, 2245 (1998)): an entanglement measure
    /// ranging from 0 (separable) to 1 (maximally entangled, e.g. a Bell
    /// state). Computed as `max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)`,
    /// where the lambdas are the square roots (sorted descending) of the
    /// eigenvalues of `rho * rho_tilde`, and `rho_tilde = (Y tensor Y) rho* (Y tensor Y)`
    /// is the "spin-flipped" density matrix.
    pub fn concurrence(&self) -> Result<f64, String> {
        if self.dimension != 4 {
            return Err("Concurrence is only defined for two-qubit (4x4) density matrices".to_string());
        }

        // Y tensor Y in the computational basis, matching this module's
        // qubit-to-bit-position convention (qubit q at bit position q).
        let y = [
            Complex::zero(), Complex::new(0.0, -1.0),
            Complex::new(0.0, 1.0), Complex::zero(),
        ];
        let mut yy = vec![vec![Complex::zero(); 4]; 4];
        for i in 0..4 {
            let i1 = (i >> 1) & 1;
            let i0 = i & 1;
            for j in 0..4 {
                let j1 = (j >> 1) & 1;
                let j0 = j & 1;
                yy[i][j] = y[i1 * 2 + j1] * y[i0 * 2 + j0];
            }
        }

        // rho* : entrywise complex conjugate (not conjugate-transpose).
        let mut rho_star = vec![vec![Complex::zero(); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                rho_star[i][j] = self.matrix[i][j].conjugate();
            }
        }

        let rho_tilde = Self::complex_matmul(&Self::complex_matmul(&yy, &rho_star), &yy);
        let sqrt_rho = Self::hermitian_sqrt_of(&self.matrix);
        let m = Self::complex_matmul(&Self::complex_matmul(&sqrt_rho, &rho_tilde), &sqrt_rho);

        let mut eigenvalues = Self::hermitian_eigenvalues_of(&m);
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let lambdas: Vec<f64> = eigenvalues.iter().map(|&l| l.max(0.0).sqrt()).collect();
        Ok((lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]).max(0.0))
    }

    /// Negativity of the bipartition (`subsystem_b` vs. its complement):
    /// `N(rho) = sum of |lambda_i|` over the negative eigenvalues lambda_i
    /// of the partial transpose `rho^(T_B)` (Peres, *PRL* 1996;
    /// Vidal & Werner, *Phys. Rev. A* 65, 032314 (2002)). A nonzero value
    /// certifies entanglement across the cut (the Peres-Horodecki / PPT
    /// criterion); unlike concurrence this generalizes to any bipartition
    /// of a multi-qubit density matrix, not just two qubits.
    pub fn negativity(&self, subsystem_b: &[usize]) -> Result<f64, String> {
        for &q in subsystem_b {
            if q >= self.num_qubits {
                return Err(format!("Qubit index {} out of bounds for {} qubits", q, self.num_qubits));
            }
        }

        let mut b_sorted = subsystem_b.to_vec();
        b_sorted.sort();
        b_sorted.dedup();
        if b_sorted.len() != subsystem_b.len() {
            return Err("Duplicate qubit indices in subsystem_b".to_string());
        }
        if b_sorted.is_empty() || b_sorted.len() >= self.num_qubits {
            return Err("subsystem_b must be a proper, non-empty subset of the qubits".to_string());
        }

        let b_mask: usize = b_sorted.iter().map(|&q| 1usize << q).sum();

        // Partial transpose over B: swap the B-bits between the row and
        // column index while leaving the A-bits fixed.
        let mut transposed = vec![vec![Complex::zero(); self.dimension]; self.dimension];
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                let i_src = (i & !b_mask) | (j & b_mask);
                let j_src = (j & !b_mask) | (i & b_mask);
                transposed[i][j] = self.matrix[i_src][j_src];
            }
        }

        let eigenvalues = Self::hermitian_eigenvalues_of(&transposed);
        Ok(eigenvalues.iter().filter(|&&l| l < 0.0).map(|l| l.abs()).sum())
    }

    pub fn print_density_matrix(&self) {
        println!("Density Matrix ({} qubits):", self.num_qubits);
        println!("Purity: {:.6}, Trace: {:.6}, Pure: {}", self.purity(), self.trace(), self.is_pure());
        
        let display_size = self.dimension.min(4);
        for i in 0..display_size {
            for j in 0..display_size {
                print!("{:12} ", self.matrix[i][j]);
            }
            if self.dimension > display_size {
                println!("...");
            } else {
                println!();
            }
        }
        if self.dimension > display_size {
            println!("...");
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantumRegister {
    state_vector: Vec<Complex>,
    num_qubits: usize,
    dimension: usize,
    /// If set, measurement outcomes are drawn from a deterministic
    /// sequence keyed on this seed instead of the system RNG.
    seed: Option<u64>,
    /// Number of random draws made so far under `seed`, used to advance
    /// the deterministic sequence without storing any RNG state on the
    /// struct itself (keeping `QuantumRegister` trivially `Clone`/`Debug`).
    rng_calls: u64,
}

impl QuantumRegister {
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        if num_qubits == 0 {
            return Err("Number of qubits must be at least 1".to_string());
        }
        if num_qubits > MAX_QUBITS {
            return Err(format!("Number of qubits exceeds maximum of {}", MAX_QUBITS));
        }

        let dimension = 1 << num_qubits;
        let mut state_vector = vec![Complex::zero(); dimension];
        state_vector[0] = Complex::one();

        Ok(Self {
            state_vector,
            num_qubits,
            dimension,
            seed: None,
            rng_calls: 0,
        })
    }

    /// Construct a register whose measurement outcomes are reproducible:
    /// two registers built with `new_with_seed(n, seed)` and driven
    /// through the same sequence of gates and measurements will always
    /// collapse identically. Without a seed (the plain `new()`
    /// constructor), measurement draws from the system RNG and is not
    /// reproducible run-to-run -- which matters for regression tests and
    /// reproducible benchmark figures, but not for physical realism.
    pub fn new_with_seed(num_qubits: usize, seed: u64) -> Result<Self, String> {
        let mut register = Self::new(num_qubits)?;
        register.seed = Some(seed);
        Ok(register)
    }

    /// Draw the next uniform random value in [0, 1). If this register was
    /// constructed with a seed, the value comes from a fresh `StdRng`
    /// re-seeded from `(seed, call index)` -- deterministic and
    /// reproducible, at the cost of not being a single continuously
    /// advanced RNG stream (a design tradeoff made so `QuantumRegister`
    /// itself never has to store RNG state). Otherwise, falls back to
    /// the system RNG as before.
    fn next_random_unit(&mut self) -> f64 {
        match self.seed {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(self.rng_calls));
                self.rng_calls = self.rng_calls.wrapping_add(1);
                rng.gen()
            }
            None => rand::thread_rng().gen(),
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    fn validate_qubit_index(&self, qubit: usize) -> Result<(), String> {
        if qubit >= self.num_qubits {
            Err(format!("Qubit index {} out of bounds for {} qubits", qubit, self.num_qubits))
        } else {
            Ok(())
        }
    }

    fn validate_qubit_indices(&self, qubits: &[usize]) -> Result<(), String> {
        for &qubit in qubits {
            self.validate_qubit_index(qubit)?;
        }
        
        let mut sorted = qubits.to_vec();
        sorted.sort();
        sorted.dedup();
        
        if sorted.len() != qubits.len() {
            return Err("Duplicate qubit indices".to_string());
        }
        
        Ok(())
    }

    /// Convert a basis index to a bitstring with qubit (num_qubits - 1) on
    /// the left (most significant) and qubit 0 on the right (least
    /// significant) -- i.e. standard binary notation for `index`.
    fn index_to_bitstring(&self, index: usize) -> String {
        let mut s = String::with_capacity(self.num_qubits);
        // Iterate from highest bit (MSB) to lowest bit (LSB)
        for i in (0..self.num_qubits).rev() {
            let bit_set = ((index >> i) & 1) == 1;
            s.push(if bit_set { '1' } else { '0' });
        }
        s
    }

    pub fn apply_single_qubit_gate<F>(&mut self, target: usize, gate_fn: F) -> Result<(), String>
    where 
        F: Fn(Complex, Complex) -> (Complex, Complex),
    {
        self.validate_qubit_index(target)?;

        let step = 1 << target;
        
        for base in (0..self.dimension).step_by(step * 2) {
            for j in base..(base + step) {
                let k = j + step;
                if k < self.dimension {
                    let (left, right) = gate_fn(self.state_vector[j], self.state_vector[k]);
                    if left.is_nan() || right.is_nan() || !left.is_finite() || !right.is_finite() {
                        return Err("Gate produced invalid complex numbers".to_string());
                    }
                    self.state_vector[j] = left;
                    self.state_vector[k] = right;
                }
            }
        }
        Ok(())
    }

    pub fn apply_hadamard(&mut self, target: usize) -> Result<(), String> {
        let factor = 1.0 / 2.0_f64.sqrt();
        self.apply_single_qubit_gate(target, |left, right| {
            (
                (left + right).scale(factor),
                (left - right).scale(factor)
            )
        })
    }

    pub fn apply_pauli_x(&mut self, target: usize) -> Result<(), String> {
        self.apply_single_qubit_gate(target, |left, right| (right, left))
    }

    pub fn apply_pauli_y(&mut self, target: usize) -> Result<(), String> {
        self.apply_single_qubit_gate(target, |left, right| {
            (
                Complex::new(0.0, -1.0) * right,
                Complex::new(0.0, 1.0) * left
            )
        })
    }

    pub fn apply_pauli_z(&mut self, target: usize) -> Result<(), String> {
        self.apply_single_qubit_gate(target, |left, right| (left, right.scale(-1.0)))
    }

    pub fn apply_s_gate(&mut self, target: usize) -> Result<(), String> {
        self.apply_phase(target, Complex::i())
    }

    pub fn apply_s_dag_gate(&mut self, target: usize) -> Result<(), String> {
        self.apply_phase(target, -Complex::i())
    }

    pub fn apply_t_gate(&mut self, target: usize) -> Result<(), String> {
        let phase = (Complex::i() * PI / 4.0).exp();
        self.apply_phase(target, phase)
    }

    pub fn apply_t_dag_gate(&mut self, target: usize) -> Result<(), String> {
        let phase = (-Complex::i() * PI / 4.0).exp();
        self.apply_phase(target, phase)
    }

    pub fn apply_phase(&mut self, target: usize, phase: Complex) -> Result<(), String> {
        self.apply_single_qubit_gate(target, |left, right| (left, right * phase))
    }

    pub fn apply_rx(&mut self, target: usize, angle: f64) -> Result<(), String> {
        let cos = (angle / 2.0).cos();
        let sin = (angle / 2.0).sin();
        // RX(theta) = [[cos, -i*sin], [-i*sin, cos]] — the off-diagonal terms
        // must carry the imaginary unit, unlike RY which is purely real.
        let neg_i_sin = Complex::new(0.0, -sin);
        self.apply_single_qubit_gate(target, |left, right| {
            (
                left.scale(cos) + right * neg_i_sin,
                left * neg_i_sin + right.scale(cos)
            )
        })
    }

    pub fn apply_ry(&mut self, target: usize, angle: f64) -> Result<(), String> {
        let cos = (angle / 2.0).cos();
        let sin = (angle / 2.0).sin();
        self.apply_single_qubit_gate(target, |left, right| {
            (
                left.scale(cos) - right.scale(sin),   // Note: real coefficients
                left.scale(sin) + right.scale(cos)    // Note: real coefficients
            )
        })
    }

    pub fn apply_rz(&mut self, target: usize, angle: f64) -> Result<(), String> {
        let phase1 = Complex::new(0.0, -angle / 2.0).exp();
        let phase2 = Complex::new(0.0, angle / 2.0).exp();
        self.apply_single_qubit_gate(target, |left, right| {
            (
                left * phase1,
                right * phase2
            )
        })
    }

    /// RXX(theta) = exp(-i*theta/2 * X tensor X), the Ising XX-coupling
    /// gate. Standard entangling rotation in variational ansatze (VQE,
    /// QAOA mixers) and gate-model chemistry circuits; matches the
    /// Qiskit `RXXGate` / Pennylane `IsingXX` convention.
    pub fn apply_rxx(&mut self, qubit_a: usize, qubit_b: usize, angle: f64) -> Result<(), String> {
        self.validate_qubit_index(qubit_a)?;
        self.validate_qubit_index(qubit_b)?;
        if qubit_a == qubit_b {
            return Err("RXX requires two distinct qubits".to_string());
        }

        let cos = (angle / 2.0).cos();
        let neg_i_sin = Complex::new(0.0, -(angle / 2.0).sin());
        let mask_a = 1usize << qubit_a;
        let mask_b = 1usize << qubit_b;
        let pair_mask = mask_a | mask_b;

        for base in 0..self.dimension {
            // Process each of the 4 combinations of (qubit_a, qubit_b)
            // exactly once, keyed by the state with both bits cleared.
            if base & pair_mask != 0 {
                continue;
            }
            let s00 = base;
            let s01 = base | mask_b;
            let s10 = base | mask_a;
            let s11 = base | pair_mask;

            let a00 = self.state_vector[s00];
            let a01 = self.state_vector[s01];
            let a10 = self.state_vector[s10];
            let a11 = self.state_vector[s11];

            self.state_vector[s00] = a00.scale(cos) + a11 * neg_i_sin;
            self.state_vector[s11] = a11.scale(cos) + a00 * neg_i_sin;
            self.state_vector[s01] = a01.scale(cos) + a10 * neg_i_sin;
            self.state_vector[s10] = a10.scale(cos) + a01 * neg_i_sin;
        }
        Ok(())
    }

    /// RYY(theta) = exp(-i*theta/2 * Y tensor Y), the Ising YY-coupling
    /// gate (Qiskit `RYYGate` / Pennylane `IsingYY` convention).
    pub fn apply_ryy(&mut self, qubit_a: usize, qubit_b: usize, angle: f64) -> Result<(), String> {
        self.validate_qubit_index(qubit_a)?;
        self.validate_qubit_index(qubit_b)?;
        if qubit_a == qubit_b {
            return Err("RYY requires two distinct qubits".to_string());
        }

        let cos = (angle / 2.0).cos();
        let sin = (angle / 2.0).sin();
        let i_sin = Complex::new(0.0, sin);
        let neg_i_sin = Complex::new(0.0, -sin);
        let mask_a = 1usize << qubit_a;
        let mask_b = 1usize << qubit_b;
        let pair_mask = mask_a | mask_b;

        for base in 0..self.dimension {
            if base & pair_mask != 0 {
                continue;
            }
            let s00 = base;
            let s01 = base | mask_b;
            let s10 = base | mask_a;
            let s11 = base | pair_mask;

            let a00 = self.state_vector[s00];
            let a01 = self.state_vector[s01];
            let a10 = self.state_vector[s10];
            let a11 = self.state_vector[s11];

            self.state_vector[s00] = a00.scale(cos) + a11 * i_sin;
            self.state_vector[s11] = a11.scale(cos) + a00 * i_sin;
            self.state_vector[s01] = a01.scale(cos) + a10 * neg_i_sin;
            self.state_vector[s10] = a10.scale(cos) + a01 * neg_i_sin;
        }
        Ok(())
    }

    /// RZZ(theta) = exp(-i*theta/2 * Z tensor Z), the Ising ZZ-coupling
    /// gate (Qiskit `RZZGate` / Pennylane `IsingZZ` convention). Diagonal
    /// in the computational basis: a pure phase gate, no amplitude mixing.
    pub fn apply_rzz(&mut self, qubit_a: usize, qubit_b: usize, angle: f64) -> Result<(), String> {
        self.validate_qubit_index(qubit_a)?;
        self.validate_qubit_index(qubit_b)?;
        if qubit_a == qubit_b {
            return Err("RZZ requires two distinct qubits".to_string());
        }

        let phase_same = Complex::new(0.0, -angle / 2.0).exp(); // parity +1 (00 or 11)
        let phase_diff = Complex::new(0.0, angle / 2.0).exp();  // parity -1 (01 or 10)
        let mask_a = 1usize << qubit_a;
        let mask_b = 1usize << qubit_b;

        for i in 0..self.dimension {
            let bit_a = (i & mask_a) != 0;
            let bit_b = (i & mask_b) != 0;
            let phase = if bit_a == bit_b { phase_same } else { phase_diff };
            self.state_vector[i] = self.state_vector[i] * phase;
        }
        Ok(())
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), String> {
        self.validate_qubit_index(control)?;
        self.validate_qubit_index(target)?;
        
        if control == target {
            return Err("Control and target qubits must be different".to_string());
        }
    
        // Use optimized in-place implementation
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
    
        for basis in 0..self.dimension {
            if (basis & control_mask) == 0 {
                continue;
            }
            if (basis & target_mask) != 0 {
                continue;
            }
            let partner = basis | target_mask;
            self.state_vector.swap(basis, partner);
        }
        Ok(())
    }

    pub fn apply_controlled_z(&mut self, control: usize, target: usize) -> Result<(), String> {
        self.validate_qubit_index(control)?;
        self.validate_qubit_index(target)?;
        
        if control == target {
            return Err("Control and target qubits must be different".to_string());
        }

        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..self.dimension {
            if (i & control_mask) != 0 && (i & target_mask) != 0 {
                self.state_vector[i] = self.state_vector[i].scale(-1.0);
            }
        }
        Ok(())
    }

    pub fn apply_controlled_phase(&mut self, control: usize, target: usize, angle: f64) -> Result<(), String> {
        self.validate_qubit_index(control)?;
        self.validate_qubit_index(target)?;
        
        if control == target {
            return Err("Control and target qubits must be different".to_string());
        }

        let phase = Complex::new(0.0, angle).exp();
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..self.dimension {
            if (i & control_mask) != 0 && (i & target_mask) != 0 {
                self.state_vector[i] = self.state_vector[i] * phase;
            }
        }
        Ok(())
    }

    pub fn apply_swap(&mut self, qubit1: usize, qubit2: usize) -> Result<(), String> {
        self.validate_qubit_index(qubit1)?;
        self.validate_qubit_index(qubit2)?;
        
        if qubit1 == qubit2 {
            return Ok(());
        }

        self.apply_cnot(qubit1, qubit2)?;
        self.apply_cnot(qubit2, qubit1)?;
        self.apply_cnot(qubit1, qubit2)?;
        Ok(())
    }

    pub fn apply_cswap(&mut self, control: usize, target1: usize, target2: usize) -> Result<(), String> {
        self.validate_qubit_index(control)?;
        self.validate_qubit_index(target1)?;
        self.validate_qubit_index(target2)?;
        
        if control == target1 || control == target2 || target1 == target2 {
            return Err("CSWAP gate requires three distinct qubits".to_string());
        }

        let control_mask = 1 << control;
        let target1_mask = 1 << target1;
        let target2_mask = 1 << target2;

        // Restrict to the canonical "target1=1, target2=0" side of each
        // (i, j) swap pair so each pair is swapped exactly once.
        for i in 0..self.dimension {
            if (i & control_mask) != 0
                && (i & target1_mask) != 0
                && (i & target2_mask) == 0
            {
                let j = i ^ target1_mask ^ target2_mask;
                self.state_vector.swap(i, j);
            }
        }
        Ok(())
    }

    pub fn apply_toffoli(&mut self, control1: usize, control2: usize, target: usize) -> Result<(), String> {
        self.validate_qubit_index(control1)?;
        self.validate_qubit_index(control2)?;
        self.validate_qubit_index(target)?;
        
        if control1 == control2 || control1 == target || control2 == target {
            return Err("Toffoli gate requires three distinct qubits".to_string());
        }

        let control1_mask = 1 << control1;
        let control2_mask = 1 << control2;
        let target_mask = 1 << target;

        // Restrict to the target_bit == 0 side of each (i, i | target_mask)
        // pair so each pair is swapped exactly once.
        for i in 0..self.dimension {
            if (i & control1_mask) != 0 && (i & control2_mask) != 0 && (i & target_mask) == 0 {
                let j = i | target_mask;
                self.state_vector.swap(i, j);
            }
        }
        Ok(())
    }

    pub fn apply_multi_controlled_x(&mut self, controls: &[usize], target: usize) -> Result<(), String> {
        self.validate_qubit_indices(controls)?;
        self.validate_qubit_index(target)?;
        
        for &control in controls {
            if control == target {
                return Err("Control and target qubits must be different".to_string());
            }
        }

        let control_mask: usize = controls.iter().map(|&c| 1 << c).sum();
        let target_mask = 1 << target;

        // Restrict to the target_bit == 0 side of each pair so each pair
        // is swapped exactly once.
        for i in 0..self.dimension {
            if (i & control_mask) == control_mask && (i & target_mask) == 0 {
                let j = i | target_mask;
                self.state_vector.swap(i, j);
            }
        }
        Ok(())
    }

    pub fn apply_multi_controlled_z(&mut self, controls: &[usize], target: usize) -> Result<(), String> {
        self.validate_qubit_indices(controls)?;
        self.validate_qubit_index(target)?;
        
        for &control in controls {
            if control == target {
                return Err("Control and target qubits must be different".to_string());
            }
        }

        let control_mask: usize = controls.iter().map(|&c| 1 << c).sum();
        let target_mask = 1 << target;

        for i in 0..self.dimension {
            if (i & control_mask) == control_mask && (i & target_mask) != 0 {
                self.state_vector[i] = self.state_vector[i].scale(-1.0);
            }
        }
        Ok(())
    }

    pub fn apply_unitary_matrix(&mut self, targets: &[usize], matrix: &[Complex]) -> Result<(), String> {
        self.validate_qubit_indices(targets)?;
        
        let num_targets = targets.len();
        let matrix_dim = 1 << num_targets;
        
        if matrix.len() != matrix_dim * matrix_dim {
            return Err(format!("Unitary matrix must have dimension {}x{}", matrix_dim, matrix_dim));
        }

        let mut sorted_targets = targets.to_vec();
        sorted_targets.sort();

        let target_mask: usize = sorted_targets.iter().map(|&t| 1 << t).sum();

        for base_index in 0..self.dimension {
            if base_index & target_mask == 0 {
                let mut indices = Vec::with_capacity(matrix_dim);
                let mut values = Vec::with_capacity(matrix_dim);
                
                for i in 0..matrix_dim {
                    let mut index = base_index;
                    for (bit_pos, &target) in sorted_targets.iter().enumerate() {
                        if (i >> bit_pos) & 1 != 0 {
                            index |= 1 << target;
                        }
                    }
                    indices.push(index);
                    values.push(self.state_vector[index]);
                }

                let mut new_values = vec![Complex::zero(); matrix_dim];
                for i in 0..matrix_dim {
                    for j in 0..matrix_dim {
                        new_values[i] = new_values[i] + values[j] * matrix[i * matrix_dim + j];
                    }
                    if new_values[i].is_nan() || !new_values[i].is_finite() {
                        return Err("Unitary matrix produced invalid state".to_string());
                    }
                }

                for (idx, &index) in indices.iter().enumerate() {
                    self.state_vector[index] = new_values[idx];
                }
            }
        }
        Ok(())
    }

    pub fn measure_single_qubit(&mut self, qubit: usize) -> Result<u8, String> {
        self.validate_qubit_index(qubit)?;

        let mut prob_zero = 0.0;
        let mask = 1 << qubit;

        for i in 0..self.dimension {
            if (i & mask) == 0 {
                prob_zero += self.state_vector[i].magnitude_squared();
            }
        }

        let random_val: f64 = self.next_random_unit();
        let result = if random_val < prob_zero { 0 } else { 1 };

        self.collapse_after_single_measurement(qubit, result);
        Ok(result)
    }

    pub fn measure_single_qubit_with_probability(&mut self, qubit: usize) -> Result<(u8, f64), String> {
        self.validate_qubit_index(qubit)?;

        let mut prob_zero = 0.0;
        let mask = 1 << qubit;

        for i in 0..self.dimension {
            if (i & mask) == 0 {
                prob_zero += self.state_vector[i].magnitude_squared();
            }
        }

        let random_val: f64 = self.next_random_unit();
        let result = if random_val < prob_zero { 0 } else { 1 };
        let probability = if result == 0 { prob_zero } else { 1.0 - prob_zero };

        self.collapse_after_single_measurement(qubit, result);
        Ok((result, probability))
    }

    fn collapse_after_single_measurement(&mut self, qubit: usize, result: u8) {
        let mask = 1 << qubit;
        let norm_factor = if result == 0 {
            let mut prob = 0.0;
            for i in 0..self.dimension {
                if (i & mask) == 0 {
                    prob += self.state_vector[i].magnitude_squared();
                }
            }
            if prob > 0.0 { 1.0 / prob.sqrt() } else { 0.0 }
        } else {
            let mut prob = 0.0;
            for i in 0..self.dimension {
                if (i & mask) != 0 {
                    prob += self.state_vector[i].magnitude_squared();
                }
            }
            if prob > 0.0 { 1.0 / prob.sqrt() } else { 0.0 }
        };

        for i in 0..self.dimension {
            let matches_result = if result == 0 {
                (i & mask) == 0
            } else {
                (i & mask) != 0
            };
            
            if matches_result {
                self.state_vector[i] = self.state_vector[i].scale(norm_factor);
            } else {
                self.state_vector[i] = Complex::zero();
            }
        }
    }

    pub fn measure_all_qubits(&mut self) -> Result<Vec<u8>, String> {
        let random_val: f64 = self.next_random_unit();

        let mut cumulative_prob = 0.0;
        let mut result_state = 0;

        for (i, &amplitude) in self.state_vector.iter().enumerate() {
            cumulative_prob += amplitude.magnitude_squared();
            if random_val <= cumulative_prob || i == self.dimension - 1 {
                result_state = i;
                break;
            }
        }

        // Convert to MSB-first bit array (qubit 0 is first element)
        let result_bits: Vec<u8> = (0..self.num_qubits)
            .rev()
            .map(|i| ((result_state >> i) & 1) as u8)
            .collect();

        // Collapse state
        for i in 0..self.dimension {
            self.state_vector[i] = if i == result_state {
                Complex::one()
            } else {
                Complex::zero()
            };
        }

        Ok(result_bits)
    }

    pub fn measure_all_qubits_with_probability(&mut self) -> Result<(Vec<u8>, f64), String> {
        let random_val: f64 = self.next_random_unit();

        let mut cumulative_prob = 0.0;
        let mut result_state = 0;
        let mut result_probability = 0.0;

        for (i, &amplitude) in self.state_vector.iter().enumerate() {
            cumulative_prob += amplitude.magnitude_squared();
            if random_val <= cumulative_prob || i == self.dimension - 1 {
                result_state = i;
                result_probability = amplitude.magnitude_squared();
                break;
            }
        }

        // Convert to MSB-first bit array (qubit 0 is first element)
        let result_bits: Vec<u8> = (0..self.num_qubits)
            .rev()
            .map(|i| ((result_state >> i) & 1) as u8)
            .collect();

        // Collapse state
        for i in 0..self.dimension {
            self.state_vector[i] = if i == result_state {
                Complex::one()
            } else {
                Complex::zero()
            };
        }

        Ok((result_bits, result_probability))
    }

    pub fn get_state_probability(&self, state: &str) -> Result<f64, String> {
        if state.len() != self.num_qubits {
            return Err(format!("State string length must match number of qubits {}", self.num_qubits));
        }

        let mut index = 0;
        for (i, ch) in state.chars().rev().enumerate() {
            match ch {
                '0' => {},
                '1' => index |= 1 << i,
                _ => return Err("State string must contain only '0' and '1'".to_string()),
            }
        }

        if index >= self.dimension {
            return Err("State index out of bounds".to_string());
        }

        Ok(self.state_vector[index].magnitude_squared())
    }

    pub fn normalize(&mut self) -> Result<(), String> {
        let total_probability: f64 = self.state_vector.iter()
            .map(|amp| amp.magnitude_squared())
            .sum();

        if total_probability < EPSILON {
            return Err("Cannot normalize zero state vector".to_string());
        }

        let scale_factor = 1.0 / total_probability.sqrt();
        for amplitude in &mut self.state_vector {
            *amplitude = amplitude.scale(scale_factor);
        }

        Ok(())
    }

    pub fn get_state_vector(&self) -> &[Complex] {
        &self.state_vector
    }

    pub fn fidelity(&self, other: &Self) -> Result<f64, String> {
        if self.num_qubits != other.num_qubits {
            return Err("Quantum registers must have same number of qubits".to_string());
        }

        let mut overlap = Complex::zero();
        for i in 0..self.dimension {
            overlap = overlap + self.state_vector[i] * other.state_vector[i].conjugate();
        }

        Ok(overlap.magnitude_squared())
    }

    pub fn trace_distance(&self, other: &Self) -> Result<f64, String> {
        // For pure states, trace distance D(psi, phi) = sqrt(1 - |<psi|phi>|^2).
        // This is gauge-invariant (unlike a raw amplitude-difference norm), so
        // states that differ only by a global phase correctly give D = 0.
        let fidelity = self.fidelity(other)?;
        Ok((1.0 - fidelity).max(0.0).sqrt())
    }

    pub fn expectation_value_pauli_z(&self, qubit: usize) -> Result<f64, String> {
        self.validate_qubit_index(qubit)?;

        let mut expectation = 0.0;
        let mask = 1 << qubit;

        for i in 0..self.dimension {
            let value = if (i & mask) == 0 { 1.0 } else { -1.0 };
            expectation += value * self.state_vector[i].magnitude_squared();
        }

        Ok(expectation)
    }

    /// Expectation value of an arbitrary Pauli-string observable, given as
    /// a sparse list of `(qubit, PauliOp)` terms -- qubits not listed are
    /// implicitly `I`. This is the standard interface for estimating
    /// observables in VQE-style algorithms (cf. Qiskit's
    /// `Statevector.expectation_value`, Cirq's `PauliString`). A sparse
    /// `(qubit, op)` list is used rather than a dense per-qubit string to
    /// avoid any ambiguity with this module's qubit-to-bit-position
    /// convention.
    ///
    /// Computed as `<psi|P|psi> = <psi | (P|psi>)>`: apply P to a scratch
    /// copy of the state and take the inner product with the original,
    /// reusing the existing single-qubit gate machinery rather than
    /// building a dense Pauli-string matrix.
    pub fn expectation_value_pauli_string(&self, terms: &[(usize, PauliOp)]) -> Result<f64, String> {
        let qubits: Vec<usize> = terms.iter().map(|&(q, _)| q).collect();
        self.validate_qubit_indices(&qubits)?;

        let mut scratch = self.clone();
        for &(qubit, op) in terms {
            match op {
                PauliOp::I => {}
                PauliOp::X => scratch.apply_pauli_x(qubit)?,
                PauliOp::Y => scratch.apply_pauli_y(qubit)?,
                PauliOp::Z => scratch.apply_pauli_z(qubit)?,
            }
        }

        let mut overlap = Complex::zero();
        for i in 0..self.dimension {
            overlap = overlap + self.state_vector[i].conjugate() * scratch.state_vector[i];
        }

        // <psi|P|psi> for a Hermitian P is guaranteed real; the imaginary
        // part is discarded (it is zero, up to floating-point error, for
        // any valid Pauli string).
        Ok(overlap.real())
    }

    pub fn to_density_matrix(&self) -> Result<DensityMatrix, String> {
        DensityMatrix::from_state_vector(&self.state_vector)
    }

    // FIXED: Use the index_to_bitstring method instead of reverse_bits
    pub fn print_state(&self) {
        println!("Quantum Register State ({} qubits, dimension {}):", 
                 self.num_qubits, self.dimension);
        
        let mut non_zero_count = 0;
        let total_probability: f64 = self.state_vector.iter()
            .map(|amp| amp.magnitude_squared())
            .sum();

        for (i, &amplitude) in self.state_vector.iter().enumerate() {
            let probability = amplitude.magnitude_squared();
            if probability > EPSILON {
                non_zero_count += 1;
                let state_string = self.index_to_bitstring(i);
                print!("  |{}⟩: {}", state_string, amplitude);
                println!(" (probability: {:.6})", probability);
            }
        }

        println!("  Non-zero states: {}", non_zero_count);
        println!("  Total probability: {:.6}", total_probability);
        
        if (total_probability - 1.0).abs() > EPSILON {
            println!("  WARNING: State vector is not normalized");
        }
    }

    // FIXED: Use the index_to_bitstring method
    pub fn get_probability_distribution(&self) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();
        for (i, &amplitude) in self.state_vector.iter().enumerate() {
            let probability = amplitude.magnitude_squared();
            if probability > EPSILON {
                let state_string = self.index_to_bitstring(i);
                distribution.insert(state_string, probability);
            }
        }
        distribution
    }

    // Add a non-destructive measurement probability function
    pub fn get_measurement_probability(&self, qubit: usize) -> Result<(f64, f64), String> {
        self.validate_qubit_index(qubit)?;

        let mut prob_zero = 0.0;
        let mask = 1 << qubit;

        for i in 0..self.dimension {
            if (i & mask) == 0 {
                prob_zero += self.state_vector[i].magnitude_squared();
            }
        }

        Ok((prob_zero, 1.0 - prob_zero))
    }

    pub fn to_qasm(&self, circuit_name: &str) -> String {
        let mut qasm = String::new();
        qasm.push_str("OPENQASM 2.0;\n");
        qasm.push_str("include \"qelib1.inc\";\n");
        qasm.push_str(&format!("qreg q[{}];\n", self.num_qubits));
        qasm.push_str(&format!("creg c[{}];\n", self.num_qubits));
        qasm.push_str(&format!("// Circuit: {}\n", circuit_name));
        qasm
    }
}
/// A chainable circuit builder over a `QuantumRegister`.
///
/// Every gate method keeps the original ergonomic `&mut Self` chaining
/// signature (`circuit.hadamard(0).cnot(0, 1).rz(1, 0.3)`), but instead
/// of silently dropping a failed call (e.g. an out-of-range qubit index),
/// each failure is recorded internally. Call [`QuantumCircuit::build`]
/// after the chain to check whether anything went wrong -- and if so,
/// you get every error from the whole chain at once, not just the
/// first one. This "fail-slow" shape is deliberately different from a
/// `?`-per-call chain: when a circuit is built programmatically (e.g.
/// from a loop over generated gate indices), stopping at the first bad
/// index means fixing and re-running one error at a time, whereas
/// accumulating them surfaces every problem in the batch in a single
/// pass.
pub struct QuantumCircuit {
    register: QuantumRegister,
    operations: Vec<String>,
    errors: Vec<String>,
}

impl QuantumCircuit {
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        Ok(Self {
            register: QuantumRegister::new(num_qubits)?,
            operations: Vec::new(),
            errors: Vec::new(),
        })
    }

    /// Records the outcome of a single builder call: on success, logs
    /// the operation's QASM-style string; on failure, accumulates the
    /// error (tagged with the attempted operation) instead of dropping
    /// it. Shared by every gate method below to keep them one-line.
    fn record(&mut self, result: Result<(), String>, op: String) -> &mut Self {
        match result {
            Ok(()) => self.operations.push(op),
            Err(e) => self.errors.push(format!("{} (attempted: {})", e, op)),
        }
        self
    }

    pub fn hadamard(&mut self, target: usize) -> &mut Self {
        let result = self.register.apply_hadamard(target);
        self.record(result, format!("h q[{}];", target))
    }

    pub fn x(&mut self, target: usize) -> &mut Self {
        let result = self.register.apply_pauli_x(target);
        self.record(result, format!("x q[{}];", target))
    }

    pub fn y(&mut self, target: usize) -> &mut Self {
        let result = self.register.apply_pauli_y(target);
        self.record(result, format!("y q[{}];", target))
    }

    pub fn z(&mut self, target: usize) -> &mut Self {
        let result = self.register.apply_pauli_z(target);
        self.record(result, format!("z q[{}];", target))
    }

    pub fn s(&mut self, target: usize) -> &mut Self {
        let result = self.register.apply_s_gate(target);
        self.record(result, format!("s q[{}];", target))
    }

    pub fn sdg(&mut self, target: usize) -> &mut Self {
        let result = self.register.apply_s_dag_gate(target);
        self.record(result, format!("sdg q[{}];", target))
    }

    pub fn t(&mut self, target: usize) -> &mut Self {
        let result = self.register.apply_t_gate(target);
        self.record(result, format!("t q[{}];", target))
    }

    pub fn tdg(&mut self, target: usize) -> &mut Self {
        let result = self.register.apply_t_dag_gate(target);
        self.record(result, format!("tdg q[{}];", target))
    }

    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        let result = self.register.apply_cnot(control, target);
        self.record(result, format!("cx q[{}], q[{}];", control, target))
    }

    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> &mut Self {
        let result = self.register.apply_swap(qubit1, qubit2);
        self.record(result, format!("swap q[{}], q[{}];", qubit1, qubit2))
    }

    pub fn cswap(&mut self, control: usize, target1: usize, target2: usize) -> &mut Self {
        let result = self.register.apply_cswap(control, target1, target2);
        self.record(result, format!("cswap q[{}], q[{}], q[{}];", control, target1, target2))
    }

    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        let result = self.register.apply_toffoli(control1, control2, target);
        self.record(result, format!("ccx q[{}], q[{}], q[{}];", control1, control2, target))
    }

    pub fn rx(&mut self, target: usize, angle: f64) -> &mut Self {
        let result = self.register.apply_rx(target, angle);
        self.record(result, format!("rx({}) q[{}];", angle, target))
    }

    pub fn ry(&mut self, target: usize, angle: f64) -> &mut Self {
        let result = self.register.apply_ry(target, angle);
        self.record(result, format!("ry({}) q[{}];", angle, target))
    }

    pub fn rz(&mut self, target: usize, angle: f64) -> &mut Self {
        let result = self.register.apply_rz(target, angle);
        self.record(result, format!("rz({}) q[{}];", angle, target))
    }

    pub fn rxx(&mut self, qubit_a: usize, qubit_b: usize, angle: f64) -> &mut Self {
        let result = self.register.apply_rxx(qubit_a, qubit_b, angle);
        self.record(result, format!("rxx({}) q[{}], q[{}];", angle, qubit_a, qubit_b))
    }

    pub fn ryy(&mut self, qubit_a: usize, qubit_b: usize, angle: f64) -> &mut Self {
        let result = self.register.apply_ryy(qubit_a, qubit_b, angle);
        self.record(result, format!("ryy({}) q[{}], q[{}];", angle, qubit_a, qubit_b))
    }

    pub fn rzz(&mut self, qubit_a: usize, qubit_b: usize, angle: f64) -> &mut Self {
        let result = self.register.apply_rzz(qubit_a, qubit_b, angle);
        self.record(result, format!("rzz({}) q[{}], q[{}];", angle, qubit_a, qubit_b))
    }

    pub fn controlled_phase(&mut self, control: usize, target: usize, angle: f64) -> &mut Self {
        let result = self.register.apply_controlled_phase(control, target, angle);
        self.record(result, format!("cp({}) q[{}], q[{}];", angle, control, target))
    }

    pub fn multi_controlled_x(&mut self, controls: &[usize], target: usize) -> &mut Self {
        let result = self.register.apply_multi_controlled_x(controls, target);
        let controls_str = controls.iter()
            .map(|c| format!("q[{}]", c))
            .collect::<Vec<_>>()
            .join(", ");
        self.record(result, format!("mcx {}, q[{}];", controls_str, target))
    }

    /// Checks whether every gate call made on this circuit so far
    /// succeeded. Returns `Ok(())` if so, or `Err` with every
    /// accumulated error message (one per failed call, in the order
    /// they occurred) if not. Call this after building up a circuit
    /// through the chainable gate methods to find out whether anything
    /// silently failed -- e.g. `circuit.h(0).cnot(0, 5).build()?` on a
    /// 2-qubit register reports the out-of-range CNOT rather than
    /// dropping it.
    pub fn build(&self) -> Result<(), Vec<String>> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }

    /// True if any gate call on this circuit has failed so far.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Every accumulated error message, in the order the failing calls
    /// occurred. Does not clear the error log.
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    pub fn measure_single(&mut self, qubit: usize) -> Result<u8, String> {
        self.register.measure_single_qubit(qubit)
    }

    pub fn measure_single_with_probability(&mut self, qubit: usize) -> Result<(u8, f64), String> {
        self.register.measure_single_qubit_with_probability(qubit)
    }

    pub fn measure_all(&mut self) -> Result<Vec<u8>, String> {
        self.register.measure_all_qubits()
    }

    pub fn measure_all_with_probability(&mut self) -> Result<(Vec<u8>, f64), String> {
        self.register.measure_all_qubits_with_probability()
    }

    pub fn get_register(&self) -> &QuantumRegister {
        &self.register
    }

    pub fn get_register_mut(&mut self) -> &mut QuantumRegister {
        &mut self.register
    }

    pub fn to_qasm(&self, circuit_name: &str) -> String {
        let mut qasm = self.register.to_qasm(circuit_name);
        for op in &self.operations {
            qasm.push_str(&op);
            qasm.push('\n');
        }
        qasm.push_str("measure q -> c;\n");
        qasm
    }

    pub fn print_circuit(&self) {
        println!("Quantum Circuit ({} qubits):", self.register.num_qubits());
        for (i, op) in self.operations.iter().enumerate() {
            println!("  {}: {}", i, op);
        }
    }

    pub fn get_operations(&self) -> &[String] {
        &self.operations
    }
}

pub fn create_bell_state() -> Result<QuantumRegister, String> {
    let mut register = QuantumRegister::new(2)?;
    register.apply_hadamard(0)?;
    register.apply_cnot(0, 1)?;
    Ok(register)
}

pub fn create_ghz_state(num_qubits: usize) -> Result<QuantumRegister, String> {
    let mut register = QuantumRegister::new(num_qubits)?;
    register.apply_hadamard(0)?;
    for i in 1..num_qubits {
        register.apply_cnot(0, i)?;
    }
    Ok(register)
}

/// W state: an equal superposition of every single-excitation basis
/// state, `(|10..0> + |01..0> + ... + |00..1>) / sqrt(n)`. Unlike the GHZ
/// state, the W state's entanglement survives losing any one qubit,
/// which is why it's the standard alternative benchmark representative
/// for genuine multipartite entanglement (Dur, Vidal & Cirac,
/// *Phys. Rev. A* 62, 062314 (2000), which showed GHZ and W states are
/// not interconvertible by local operations -- i.e. they represent two
/// inequivalent classes of tripartite-or-more entanglement).
pub fn create_w_state(num_qubits: usize) -> Result<QuantumRegister, String> {
    let mut register = QuantumRegister::new(num_qubits)?;
    let amplitude = Complex::new(1.0 / (num_qubits as f64).sqrt(), 0.0);

    for amp in register.state_vector.iter_mut() {
        *amp = Complex::zero();
    }
    for q in 0..num_qubits {
        register.state_vector[1usize << q] = amplitude;
    }

    Ok(register)
}

pub fn quantum_fourier_transform(register: &mut QuantumRegister) -> Result<(), String> {
    let num_qubits = register.num_qubits();
    
    for i in 0..num_qubits {
        register.apply_hadamard(i)?;
        for j in (i + 1)..num_qubits {
            let angle = 2.0 * PI / (1 << (j - i + 1)) as f64;
            register.apply_controlled_phase(j, i, angle)?;
        }
    }
    
    for i in 0..num_qubits / 2 {
        register.apply_swap(i, num_qubits - 1 - i)?;
    }
    
    Ok(())
}

pub fn inverse_quantum_fourier_transform(register: &mut QuantumRegister) -> Result<(), String> {
    let num_qubits = register.num_qubits();
    
    for i in 0..num_qubits / 2 {
        register.apply_swap(i, num_qubits - 1 - i)?;
    }
    
    for i in (0..num_qubits).rev() {
        for j in (i + 1..num_qubits).rev() {
            let angle = -2.0 * PI / (1 << (j - i + 1)) as f64;
            register.apply_controlled_phase(j, i, angle)?;
        }
        register.apply_hadamard(i)?;
    }
    
    Ok(())
}

pub struct QuantumAlgorithm {}

impl QuantumAlgorithm {
    pub fn deutsch_josza(oracle: fn(&mut QuantumRegister) -> Result<(), String>, n: usize) -> Result<bool, String> {
        let mut register = QuantumRegister::new(n + 1)?;
        
        register.apply_pauli_x(n)?;
        
        for i in 0..=n {
            register.apply_hadamard(i)?;
        }
        
        oracle(&mut register)?;
        
        for i in 0..n {
            register.apply_hadamard(i)?;
        }
        
        let measurement = register.measure_all_qubits()?;
        
        let mut all_zero = true;
        for i in 0..n {
            if measurement[i] != 0 {
                all_zero = false;
                break;
            }
        }
        
        Ok(all_zero)
    }
    
    pub fn grover_iteration(register: &mut QuantumRegister, oracle: fn(&mut QuantumRegister) -> Result<(), String>) -> Result<(), String> {
        oracle(register)?;
        
        let num_qubits = register.num_qubits();
        
        // Diffusion operator: H^{⊗n} (2|0⟩⟨0| - I) H^{⊗n}
        for i in 0..num_qubits {
            register.apply_hadamard(i)?;
            register.apply_pauli_x(i)?;
        }
        
        // Multi-controlled Z gate over all qubits (phase flip on |11...1⟩)
        if num_qubits >= 2 {
            let controls: Vec<usize> = (0..num_qubits - 1).collect();
            register.apply_multi_controlled_z(&controls, num_qubits - 1)?;
        } else {
            // Single qubit case: just apply Z gate
            register.apply_pauli_z(0)?;
        }
        
        for i in 0..num_qubits {
            register.apply_pauli_x(i)?;
            register.apply_hadamard(i)?;
        }
        
        Ok(())
    }
}

pub struct QuantumBenchmark;

impl QuantumBenchmark {
    pub fn benchmark_hadamard_chain(num_qubits: usize, iterations: usize) -> Duration {
        let mut total_duration = Duration::new(0, 0);
        
        for _ in 0..iterations {
            let mut register = QuantumRegister::new(num_qubits).unwrap();
            let start = Instant::now();
            
            for qubit in 0..num_qubits {
                register.apply_hadamard(qubit).unwrap();
            }
            
            total_duration += start.elapsed();
        }
        
        total_duration / iterations as u32
    }

    pub fn benchmark_cnot_chain(num_qubits: usize, iterations: usize) -> Duration {
        let mut total_duration = Duration::new(0, 0);
        
        for _ in 0..iterations {
            let mut register = QuantumRegister::new(num_qubits).unwrap();
            let start = Instant::now();
            
            for i in 0..num_qubits - 1 {
                register.apply_cnot(i, i + 1).unwrap();
            }
            
            total_duration += start.elapsed();
        }
        
        total_duration / iterations as u32
    }

    pub fn benchmark_qft(num_qubits: usize, iterations: usize) -> Duration {
        let mut total_duration = Duration::new(0, 0);
        
        for _ in 0..iterations {
            let mut register = QuantumRegister::new(num_qubits).unwrap();
            let start = Instant::now();
            
            quantum_fourier_transform(&mut register).unwrap();
            
            total_duration += start.elapsed();
        }
        
        total_duration / iterations as u32
    }

    pub fn run_comprehensive_benchmark() {
        println!("Quantum Simulator Benchmark Results");
        println!("===================================");
        
        let qubit_counts = [4, 8, 12];
        let iterations = 10;
        
        for &num_qubits in &qubit_counts {
            println!("\n{} Qubits:", num_qubits);
            
            let hadamard_time = Self::benchmark_hadamard_chain(num_qubits, iterations);
            let cnot_time = Self::benchmark_cnot_chain(num_qubits, iterations);
            let qft_time = Self::benchmark_qft(num_qubits, iterations);
            
            println!("  Hadamard chain: {:?}", hadamard_time);
            println!("  CNOT chain: {:?}", cnot_time);
            println!("  QFT: {:?}", qft_time);
            
            let dimension = 1 << num_qubits;
            println!("  Dimension: 2^{} = {}", num_qubits, dimension);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_state_creation() {
        let bell = create_bell_state().unwrap();
        let distribution = bell.get_probability_distribution();
        
        assert!((distribution.get("00").unwrap_or(&0.0) - 0.5).abs() < EPSILON);
        assert!((distribution.get("11").unwrap_or(&0.0) - 0.5).abs() < EPSILON);
        assert!(distribution.get("01").is_none());
        assert!(distribution.get("10").is_none());
    }

    #[test]
    fn test_von_neumann_entropy_maximally_mixed() {
        // I/4 on 2 qubits: eigenvalues [0.25, 0.25, 0.25, 0.25],
        // true entropy = -4 * 0.25*log2(0.25) = log2(4) = 2.0
        let mut density = DensityMatrix::new(2).unwrap();
        // Turn the pure |00> state into a maximally mixed state directly.
        for i in 0..4 {
            for j in 0..4 {
                density.matrix[i][j] = if i == j { Complex::new(0.25, 0.0) } else { Complex::zero() };
            }
        }
        let entropy = density.von_neumann_entropy();
        assert!((entropy - 2.0).abs() < 1e-6, "expected entropy 2.0, got {}", entropy);
    }

    #[test]
    fn test_von_neumann_entropy_pure_state_is_zero() {
        let bell = create_bell_state().unwrap();
        let density = DensityMatrix::from_state_vector(bell.get_state_vector()).unwrap();
        let entropy = density.von_neumann_entropy();
        assert!(entropy.abs() < 1e-6, "pure state should have zero entropy, got {}", entropy);
    }

    #[test]
    fn test_depolarize_preserves_trace_on_any_qubit() {
        // Trace preservation must hold no matter which qubit is targeted,
        // not just qubit 0.
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut d = DensityMatrix::new(num_qubits).unwrap();
                d.apply_depolarizing_channel(1.0, target).unwrap();
                assert!(
                    (d.trace() - 1.0).abs() < 1e-9,
                    "trace not preserved for {} qubits, target {}: trace={}",
                    num_qubits, target, d.trace()
                );
            }
        }
    }

    #[test]
    fn test_von_neumann_entropy_3qubit_ghz_reduced_via_depolarize() {
        // Sanity check on a larger (8x8) mixed density matrix: fully
        // depolarize all 3 qubits of a GHZ state one at a time -> should
        // approach the maximally mixed 3-qubit state, entropy -> log2(8)=3.0
        // p=0.75 makes identity_weight = pauli_weight = 0.25, i.e. an exact
        // equal-weight twirl over {I, X, Y, Z} -- this is what actually
        // drives a qubit's reduced state to the maximally mixed I/2
        // regardless of input (p=1.0 only averages X, Y, Z and leaves a
        // residual, physically correct, non-zero gap from the ideal case).
        let ghz = create_ghz_state(3).unwrap();
        let mut density = DensityMatrix::from_state_vector(ghz.get_state_vector()).unwrap();
        for q in 0..3 {
            density.apply_depolarizing_channel(0.75, q).unwrap();
        }
        let entropy = density.von_neumann_entropy();
        assert!((entropy - 3.0).abs() < 1e-6, "expected entropy ~3.0, got {}", entropy);
    }

    #[test]
    fn test_density_matrix_purity() {
        let bell = create_bell_state().unwrap();
        let density = DensityMatrix::from_state_vector(bell.get_state_vector()).unwrap();
        
        assert!((density.purity() - 1.0).abs() < EPSILON);
        assert!(density.is_pure());
    }

    #[test]
    fn test_depolarizing_channel() {
        let mut density = DensityMatrix::new(1).unwrap();
        density.apply_depolarizing_channel(0.5, 0).unwrap();
        
        assert!(density.purity() < 1.0);
        assert!(!density.is_pure());
    }

    #[test]
    fn test_multi_controlled_gates() {
        let mut register = QuantumRegister::new(3).unwrap();
        register.apply_hadamard(0).unwrap();
        register.apply_hadamard(1).unwrap();
        
        register.apply_multi_controlled_x(&[0, 1], 2).unwrap();
        
        let prob_dist = register.get_probability_distribution();
        // Only the q0=1,q1=1 branch (25% of the superposition) trips the
        // control condition and flips q2, landing on |111>; the other
        // three branches are untouched (q2 stays 0), and "110" is not a
        // reachable state here at all (q1=1,q0=0 never gets q2 flipped).
        assert!((prob_dist.get("111").unwrap_or(&0.0) - 0.25).abs() < EPSILON);
        assert!((prob_dist.get("000").unwrap_or(&0.0) - 0.25).abs() < EPSILON);
        assert!((prob_dist.get("001").unwrap_or(&0.0) - 0.25).abs() < EPSILON);
        assert!((prob_dist.get("010").unwrap_or(&0.0) - 0.25).abs() < EPSILON);
        assert!(prob_dist.get("110").is_none());
    }

    #[test]
    fn test_quantum_fourier_transform() {
        let mut register = QuantumRegister::new(3).unwrap();
        register.apply_pauli_x(0).unwrap();
        
        quantum_fourier_transform(&mut register).unwrap();
        
        let prob_dist = register.get_probability_distribution();
        assert!(prob_dist.values().sum::<f64>() - 1.0 < EPSILON);
        
        inverse_quantum_fourier_transform(&mut register).unwrap();
        
        let final_state = register.get_state_vector();
        assert!((final_state[1].real() - 1.0).abs() < EPSILON);
        assert!(final_state[1].imag().abs() < EPSILON);
    }

    #[test]
    fn test_controlled_phase_gate() {
        let mut register = QuantumRegister::new(2).unwrap();
        register.apply_hadamard(0).unwrap();
        register.apply_hadamard(1).unwrap();
        
        register.apply_controlled_phase(0, 1, PI).unwrap();
        
        let prob_dist = register.get_probability_distribution();
        assert!((prob_dist.get("11").unwrap() - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_cswap_gate() {
        let mut register = QuantumRegister::new(3).unwrap();
        register.apply_pauli_x(0).unwrap(); // |100⟩
        register.apply_pauli_x(1).unwrap(); // |110⟩
        register.apply_cswap(0, 1, 2).unwrap();
        
        // Control is 1, target1 and target2 should swap: |110⟩ -> |101⟩
        let prob_dist = register.get_probability_distribution();
        assert!((prob_dist.get("101").unwrap_or(&0.0) - 1.0).abs() < EPSILON);
    }

    // --- Regression tests for the RX/RY bug fix ---

    #[test]
    fn test_apply_rx_pi_on_zero_gives_minus_i_one() {
        // RX(pi)|0> = cos(pi/2)|0> - i*sin(pi/2)|1> = -i|1>
        let mut reg = QuantumRegister::new(1).unwrap();
        reg.apply_rx(0, PI).unwrap();
        let state = reg.get_state_vector();

        assert!(
            state[0].magnitude() < 1e-9,
            "amplitude of |0> should vanish, got {}", state[0]
        );

        let expected = Complex::new(0.0, -1.0);
        assert!(
            (state[1] - expected).magnitude() < 1e-9,
            "expected -i|1>, got {}", state[1]
        );
    }

    #[test]
    fn test_apply_rx_differs_from_ry_by_imaginary_unit() {
        // RX and RY must produce the same measurement probabilities but
        // different phases: RX puts the off-diagonal amplitude on the
        // imaginary axis, RY keeps it real. Before the fix, apply_rx was
        // byte-for-byte identical to apply_ry.
        let angle = std::f64::consts::FRAC_PI_3;

        let mut rx_reg = QuantumRegister::new(1).unwrap();
        rx_reg.apply_rx(0, angle).unwrap();
        let mut ry_reg = QuantumRegister::new(1).unwrap();
        ry_reg.apply_ry(0, angle).unwrap();

        let rx_state = rx_reg.get_state_vector();
        let ry_state = ry_reg.get_state_vector();

        // Same probabilities...
        assert!(
            (rx_state[0].magnitude_squared() - ry_state[0].magnitude_squared()).abs() < 1e-9
        );
        assert!(
            (rx_state[1].magnitude_squared() - ry_state[1].magnitude_squared()).abs() < 1e-9
        );

        // ...but RX's |1> amplitude must be purely imaginary...
        assert!(
            rx_state[1].imag().abs() > 1e-9,
            "RX amplitude on |1> should be imaginary, got {}", rx_state[1]
        );
        assert!(
            rx_state[1].real().abs() < 1e-9,
            "RX amplitude on |1> should have no real part, got {}", rx_state[1]
        );

        // ...while RY's |1> amplitude must be purely real.
        assert!(
            ry_state[1].real().abs() > 1e-9,
            "RY amplitude on |1> should be real, got {}", ry_state[1]
        );
        assert!(
            ry_state[1].imag().abs() < 1e-9,
            "RY amplitude on |1> should have no imaginary part, got {}", ry_state[1]
        );
    }

    // --- Regression tests for the trace_distance bug fix ---

    #[test]
    fn test_trace_distance_identical_states_is_zero() {
        let bell = create_bell_state().unwrap();
        let distance = bell.trace_distance(&bell).unwrap();
        assert!(
            distance.abs() < 1e-6,
            "identical states should have zero trace distance, got {}", distance
        );
    }

    #[test]
    fn test_trace_distance_orthogonal_states_is_one() {
        let zero = QuantumRegister::new(1).unwrap();
        let mut one = QuantumRegister::new(1).unwrap();
        one.apply_pauli_x(0).unwrap();

        let distance = zero.trace_distance(&one).unwrap();
        assert!(
            (distance - 1.0).abs() < 1e-9,
            "orthogonal states should have trace distance 1, got {}", distance
        );
    }

    #[test]
    fn test_trace_distance_invariant_under_global_phase() {
        // D(psi, e^{i*theta} * psi) must be 0 — this is exactly what the old
        // L1-amplitude-difference formula got wrong.
        let mut original = QuantumRegister::new(1).unwrap();
        original.apply_hadamard(0).unwrap();

        let mut phase_shifted = QuantumRegister::new(1).unwrap();
        phase_shifted.apply_hadamard(0).unwrap();
        let theta = 0.7_f64;
        let phase = Complex::new(theta.cos(), theta.sin());
        for amp in phase_shifted.state_vector.iter_mut() {
            *amp = *amp * phase;
        }

        let distance = original.trace_distance(&phase_shifted).unwrap();
        assert!(
            distance.abs() < 1e-6,
            "trace distance should be 0 for states differing only by global phase, got {}", distance
        );
    }

    // --- partial_trace ---

    #[test]
    fn test_partial_trace_bell_state_gives_maximally_mixed_qubit() {
        // Tracing out either qubit of a Bell state must give I/2 on the
        // remaining qubit -- the textbook example of a reduced state that
        // is mixed even though the full state is pure.
        let bell = create_bell_state().unwrap();
        let density = bell.to_density_matrix().unwrap();
        let reduced = density.partial_trace(&[0]).unwrap();

        assert_eq!(reduced.num_qubits(), 1);
        let matrix = reduced.get_matrix();
        assert!((matrix[0][0].real() - 0.5).abs() < 1e-9);
        assert!((matrix[1][1].real() - 0.5).abs() < 1e-9);
        assert!(matrix[0][1].magnitude() < 1e-9);
        assert!(matrix[1][0].magnitude() < 1e-9);
        assert!((reduced.trace() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_partial_trace_product_state_is_unaffected_by_spectator_qubit() {
        // Tracing out qubit 1 from a product state |0>|0> must give back
        // the pure state |0><0| on qubit 0, unlike the Bell state case.
        let register = QuantumRegister::new(2).unwrap();
        let density = register.to_density_matrix().unwrap();
        let reduced = density.partial_trace(&[0]).unwrap();

        assert!(reduced.is_pure());
        assert!((reduced.get_matrix()[0][0].real() - 1.0).abs() < 1e-9);
    }

    // --- is_valid ---

    #[test]
    fn test_is_valid_accepts_well_formed_density_matrix() {
        let bell = create_bell_state().unwrap();
        let density = bell.to_density_matrix().unwrap();
        assert!(density.is_valid().is_ok());

        let reduced = density.partial_trace(&[0]).unwrap();
        assert!(reduced.is_valid().is_ok());
    }

    #[test]
    fn test_is_valid_rejects_non_hermitian_matrix() {
        let mut density = DensityMatrix::new(1).unwrap();
        // Corrupt an off-diagonal entry so matrix[0][1] != conj(matrix[1][0]).
        density.matrix[0][1] = Complex::new(1.0, 0.0);
        assert!(density.is_valid().is_err());
    }

    // --- validate_kraus_operators ---

    #[test]
    fn test_validate_kraus_operators_accepts_amplitude_damping_kraus_set() {
        // The exact K0/K1 pair used by apply_amplitude_damping should
        // satisfy the completeness relation for any valid gamma.
        let gamma: f64 = 0.3;
        let k0 = [
            Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new((1.0 - gamma).sqrt(), 0.0),
        ];
        let k1 = [
            Complex::new(0.0, 0.0), Complex::new(gamma.sqrt(), 0.0),
            Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
        ];
        assert!(DensityMatrix::validate_kraus_operators(&[k0, k1]).is_ok());
    }

    #[test]
    fn test_validate_kraus_operators_rejects_non_unitary_single_operator() {
        // A lone Kraus operator that isn't unitary (here, 2*I) fails the
        // completeness relation -- Sum K^dagger K = 4*I != I.
        let two_i = [
            Complex::new(2.0, 0.0), Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(2.0, 0.0),
        ];
        assert!(DensityMatrix::validate_kraus_operators(&[two_i]).is_err());
    }

    // --- mixed-state (Uhlmann) fidelity ---

    #[test]
    fn test_mixed_fidelity_identical_states_is_one() {
        let bell = create_bell_state().unwrap();
        let density = bell.to_density_matrix().unwrap();
        let fidelity = density.fidelity(&density).unwrap();
        assert!((fidelity - 1.0).abs() < 1e-6, "expected fidelity 1.0, got {}", fidelity);
    }

    #[test]
    fn test_mixed_fidelity_orthogonal_states_is_zero() {
        let zero = QuantumRegister::new(1).unwrap().to_density_matrix().unwrap();
        let mut one_reg = QuantumRegister::new(1).unwrap();
        one_reg.apply_pauli_x(0).unwrap();
        let one = one_reg.to_density_matrix().unwrap();

        let fidelity = zero.fidelity(&one).unwrap();
        assert!(fidelity.abs() < 1e-6, "expected fidelity 0.0, got {}", fidelity);
    }

    // --- concurrence ---

    #[test]
    fn test_concurrence_bell_state_is_one() {
        let bell = create_bell_state().unwrap();
        let density = bell.to_density_matrix().unwrap();
        let concurrence = density.concurrence().unwrap();
        assert!((concurrence - 1.0).abs() < 1e-6, "expected concurrence 1.0, got {}", concurrence);
    }

    #[test]
    fn test_concurrence_product_state_is_zero() {
        let register = QuantumRegister::new(2).unwrap();
        let density = register.to_density_matrix().unwrap();
        let concurrence = density.concurrence().unwrap();
        assert!(concurrence.abs() < 1e-6, "expected concurrence 0.0, got {}", concurrence);
    }

    // --- negativity ---

    #[test]
    fn test_negativity_bell_state_is_half() {
        // The negativity of a Bell state is exactly 0.5 (its partial
        // transpose has eigenvalues {0.5, 0.5, 0.5, -0.5}).
        let bell = create_bell_state().unwrap();
        let density = bell.to_density_matrix().unwrap();
        let negativity = density.negativity(&[1]).unwrap();
        assert!((negativity - 0.5).abs() < 1e-6, "expected negativity 0.5, got {}", negativity);
    }

    #[test]
    fn test_negativity_product_state_is_zero() {
        let register = QuantumRegister::new(2).unwrap();
        let density = register.to_density_matrix().unwrap();
        let negativity = density.negativity(&[1]).unwrap();
        assert!(negativity.abs() < 1e-6, "expected negativity 0.0, got {}", negativity);
    }

    // --- RXX / RYY / RZZ ---

    #[test]
    fn test_apply_rxx_pi_on_zero_zero_gives_minus_i_eleven() {
        // RXX(pi)|00> = -i|11>
        let mut register = QuantumRegister::new(2).unwrap();
        register.apply_rxx(0, 1, PI).unwrap();
        let state = register.get_state_vector();

        assert!(state[0].magnitude() < 1e-9);
        let expected = Complex::new(0.0, -1.0);
        assert!((state[3] - expected).magnitude() < 1e-9, "expected -i|11>, got {}", state[3]);
    }

    #[test]
    fn test_apply_ryy_pi_on_zero_zero_gives_i_eleven() {
        // RYY(pi)|00> = i|11>
        let mut register = QuantumRegister::new(2).unwrap();
        register.apply_ryy(0, 1, PI).unwrap();
        let state = register.get_state_vector();

        assert!(state[0].magnitude() < 1e-9);
        let expected = Complex::new(0.0, 1.0);
        assert!((state[3] - expected).magnitude() < 1e-9, "expected i|11>, got {}", state[3]);
    }

    #[test]
    fn test_apply_rzz_is_diagonal_and_preserves_probabilities() {
        // RZZ is a pure phase gate: |00> picks up exp(-i*pi/2) = -i, and
        // the probability distribution (hence physical state) must be
        // completely unchanged.
        let mut register = QuantumRegister::new(2).unwrap();
        register.apply_rzz(0, 1, PI).unwrap();
        let state = register.get_state_vector();

        let expected = Complex::new(0.0, -1.0);
        assert!((state[0] - expected).magnitude() < 1e-9, "expected -i|00>, got {}", state[0]);
        for i in 1..4 {
            assert!(state[i].magnitude() < 1e-9);
        }
    }

    // --- general Pauli-string expectation value ---

    #[test]
    fn test_expectation_value_pauli_string_matches_pauli_z_single_term() {
        let mut register = QuantumRegister::new(2).unwrap();
        register.apply_pauli_x(0).unwrap(); // qubit 0 -> |1>

        let via_z = register.expectation_value_pauli_z(0).unwrap();
        let via_string = register
            .expectation_value_pauli_string(&[(0, PauliOp::Z)])
            .unwrap();
        assert!((via_z - via_string).abs() < 1e-9);
        assert!((via_string - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_expectation_value_pauli_string_xx_on_bell_state() {
        // <Bell| X tensor X |Bell> = 1 for the (|00>+|11>)/sqrt2 Bell state.
        let bell = create_bell_state().unwrap();
        let expectation = bell
            .expectation_value_pauli_string(&[(0, PauliOp::X), (1, PauliOp::X)])
            .unwrap();
        assert!((expectation - 1.0).abs() < 1e-9, "expected 1.0, got {}", expectation);
    }

    // --- W state ---

    #[test]
    fn test_create_w_state_has_equal_single_excitation_amplitudes() {
        let w = create_w_state(3).unwrap();
        let state = w.get_state_vector();
        let expected_amplitude = 1.0 / (3.0f64).sqrt();

        for q in 0..3 {
            let idx = 1usize << q;
            assert!(
                (state[idx].real() - expected_amplitude).abs() < 1e-9,
                "amplitude at single-excitation state {} should be {}, got {}",
                idx, expected_amplitude, state[idx]
            );
            assert!(state[idx].imag().abs() < 1e-9);
        }

        // |000> and |111> (and any other multi/zero-excitation state)
        // must have zero amplitude.
        assert!(state[0].magnitude() < 1e-9);
        assert!(state[7].magnitude() < 1e-9);

        let total_probability: f64 = state.iter().map(|c| c.magnitude_squared()).sum();
        assert!((total_probability - 1.0).abs() < 1e-9);
    }

    // --- seeded RNG reproducibility ---

    #[test]
    fn test_seeded_measurement_is_reproducible() {
        // Two registers built from the same seed, driven through the
        // same gates and measurements, must collapse identically.
        let mut a = QuantumRegister::new_with_seed(2, 42).unwrap();
        a.apply_hadamard(0).unwrap();
        a.apply_hadamard(1).unwrap();
        let results_a = a.measure_all_qubits().unwrap();

        let mut b = QuantumRegister::new_with_seed(2, 42).unwrap();
        b.apply_hadamard(0).unwrap();
        b.apply_hadamard(1).unwrap();
        let results_b = b.measure_all_qubits().unwrap();

        assert_eq!(results_a, results_b);
    }

    #[test]
    fn test_unseeded_measurement_still_works() {
        // The default constructor must still measure without panicking
        // and produce a valid, normalized post-measurement state.
        let mut register = QuantumRegister::new(2).unwrap();
        register.apply_hadamard(0).unwrap();
        let result = register.measure_single_qubit(0);
        assert!(result.is_ok());
    }

    // --- cyclic Jacobi correctness at a size too large for the old
    //     search-based classical Jacobi to keep the same guarantees ---

    #[test]
    fn test_von_neumann_entropy_ghz_4qubit_partial_trace_matches_maximally_mixed() {
        // Tracing any proper subset of qubits out of an n-qubit GHZ state
        // leaves that subset maximally mixed -- a real reduced-state
        // entropy computed via partial_trace, exercising the cyclic
        // Jacobi solver at a larger (16x16 doubled-block) matrix than the
        // 2-qubit tests above reach.
        let ghz = create_ghz_state(4).unwrap();
        let density = ghz.to_density_matrix().unwrap();
        let reduced = density.partial_trace(&[0]).unwrap();

        let entropy = reduced.von_neumann_entropy();
        assert!(
            (entropy - 1.0).abs() < 1e-6,
            "expected 1 bit of entropy (maximally mixed single qubit), got {}", entropy
        );
    }

    // --- fail-slow QuantumCircuit builder ---

    #[test]
    fn test_quantum_circuit_build_ok_when_every_gate_succeeds() {
        let mut circuit = QuantumCircuit::new(2).unwrap();
        circuit.hadamard(0).cnot(0, 1);
        assert!(!circuit.has_errors());
        assert!(circuit.build().is_ok());
    }

    #[test]
    fn test_quantum_circuit_accumulates_every_error_in_the_chain() {
        let mut circuit = QuantumCircuit::new(2).unwrap();
        // Three deliberately invalid calls chained together: an
        // out-of-range single-qubit gate, an out-of-range CNOT target,
        // and a second out-of-range gate. All three must be reported --
        // not just the first one a `?`-per-call chain would have stopped at.
        circuit
            .hadamard(5)
            .cnot(0, 7)
            .x(9);

        assert!(circuit.has_errors());
        let errors = circuit.errors();
        assert_eq!(errors.len(), 3, "expected all 3 failed calls to be recorded, got {:?}", errors);

        match circuit.build() {
            Err(errs) => assert_eq!(errs.len(), 3),
            Ok(()) => panic!("build() should report the accumulated errors"),
        }
    }

    #[test]
    fn test_quantum_circuit_valid_gates_still_apply_after_an_earlier_error() {
        // A failed call must not derail the rest of the chain -- the
        // register should still reflect every gate that *did* succeed.
        let mut circuit = QuantumCircuit::new(2).unwrap();
        circuit.hadamard(5).cnot(0, 1); // first call invalid, second valid
        assert_eq!(circuit.errors().len(), 1);
        assert_eq!(circuit.get_operations().len(), 1);

        let distribution = circuit.get_register().get_probability_distribution();
        // Only the CNOT applied (control=0 in |0>, so it's a no-op on
        // |00>), so the register should still read |00> with certainty.
        assert!((distribution.get("00").unwrap_or(&0.0) - 1.0).abs() < 1e-9);
    }
}