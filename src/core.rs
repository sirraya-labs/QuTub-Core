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
        let n = self.dimension;
        let m = 2 * n;
        let mut real_block = vec![vec![0.0f64; m]; m];

        for i in 0..n {
            for j in 0..n {
                let a = self.matrix[i][j].real(); // symmetric part
                let b = self.matrix[i][j].imag(); // skew-symmetric part
                real_block[i][j] = a;
                real_block[i][j + n] = -b;
                real_block[i + n][j] = b;
                real_block[i + n][j + n] = a;
            }
        }

        let mut eigenvalues = Self::jacobi_eigenvalues(&mut real_block);
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Each true eigenvalue of H appears twice (adjacently, once sorted)
        // in the spectrum of the doubled real block; keep one of each pair.
        eigenvalues.into_iter().step_by(2).collect()
    }

    /// Classical cyclic Jacobi eigenvalue algorithm for a real symmetric
    /// matrix. Repeatedly zeroes the largest off-diagonal element via a
    /// Givens rotation until the matrix is (numerically) diagonal, then
    /// returns the diagonal (the eigenvalues). O(n^3) per sweep; a handful
    /// of sweeps is enough to converge for the small matrices used here.
    fn jacobi_eigenvalues(a: &mut [Vec<f64>]) -> Vec<f64> {
        let n = a.len();
        if n == 0 {
            return Vec::new();
        }

        // Scale-relative convergence threshold: EPSILON alone is far too
        // tight for larger matrices (each of the ~n^2/2 off-diagonal
        // rotations carries its own floating-point error, and this
        // algorithm eliminates only the single largest off-diagonal entry
        // per iteration rather than a full sweep, so it needs many more
        // iterations to converge as n grows). Since Jacobi rotations are
        // orthogonal similarity transforms, the trace is exactly preserved
        // at every step regardless of convergence -- so an under-iterated
        // matrix doesn't show up as "slightly off", it shows up as some
        // diagonal entries wildly too large offset by others wildly too
        // negative (still summing to the right trace), which then blows
        // up the entropy sum. Both a generous iteration cap and a
        // frobenius-norm-relative threshold are needed to avoid that.
        let frobenius_norm: f64 = a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        let threshold = (frobenius_norm * 1e-12).max(1e-14);
        let max_iterations = 200 * n * n + 500;

        for _iter in 0..max_iterations {
            // Find largest off-diagonal magnitude
            let mut off_diag_sum = 0.0;
            let (mut p, mut q, mut max_val) = (0usize, 1usize, 0.0f64);
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = a[i][j].abs();
                    off_diag_sum += v * v;
                    if v > max_val {
                        max_val = v;
                        p = i;
                        q = j;
                    }
                }
            }

            if off_diag_sum.sqrt() < threshold {
                break;
            }

            if a[p][p] == a[q][q] {
                // 45 degree rotation when diagonal entries are equal
                let (c, s) = (std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2 * a[p][q].signum());
                Self::apply_jacobi_rotation(a, p, q, c, s);
            } else {
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                Self::apply_jacobi_rotation(a, p, q, c, s);
            }
        }

        (0..n).map(|i| a[i][i]).collect()
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
        })
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

    /// Convert a basis index to MSB-first bitstring (qubit 0 on the left)
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

        let mut rng = rand::thread_rng();
        let random_val: f64 = rng.gen();
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

        let mut rng = rand::thread_rng();
        let random_val: f64 = rng.gen();
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
        let mut rng = rand::thread_rng();
        let random_val: f64 = rng.gen();

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
        let mut rng = rand::thread_rng();
        let random_val: f64 = rng.gen();

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
pub struct QuantumCircuit {
    register: QuantumRegister,
    operations: Vec<String>,
}

impl QuantumCircuit {
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        Ok(Self {
            register: QuantumRegister::new(num_qubits)?,
            operations: Vec::new(),
        })
    }

    pub fn hadamard(&mut self, target: usize) -> &mut Self {
        if self.register.apply_hadamard(target).is_ok() {
            self.operations.push(format!("h q[{}];", target));
        }
        self
    }

    pub fn x(&mut self, target: usize) -> &mut Self {
        if self.register.apply_pauli_x(target).is_ok() {
            self.operations.push(format!("x q[{}];", target));
        }
        self
    }

    pub fn y(&mut self, target: usize) -> &mut Self {
        if self.register.apply_pauli_y(target).is_ok() {
            self.operations.push(format!("y q[{}];", target));
        }
        self
    }

    pub fn z(&mut self, target: usize) -> &mut Self {
        if self.register.apply_pauli_z(target).is_ok() {
            self.operations.push(format!("z q[{}];", target));
        }
        self
    }

    pub fn s(&mut self, target: usize) -> &mut Self {
        if self.register.apply_s_gate(target).is_ok() {
            self.operations.push(format!("s q[{}];", target));
        }
        self
    }

    pub fn sdg(&mut self, target: usize) -> &mut Self {
        if self.register.apply_s_dag_gate(target).is_ok() {
            self.operations.push(format!("sdg q[{}];", target));
        }
        self
    }

    pub fn t(&mut self, target: usize) -> &mut Self {
        if self.register.apply_t_gate(target).is_ok() {
            self.operations.push(format!("t q[{}];", target));
        }
        self
    }

    pub fn tdg(&mut self, target: usize) -> &mut Self {
        if self.register.apply_t_dag_gate(target).is_ok() {
            self.operations.push(format!("tdg q[{}];", target));
        }
        self
    }

    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        if self.register.apply_cnot(control, target).is_ok() {
            self.operations.push(format!("cx q[{}], q[{}];", control, target));
        }
        self
    }

    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> &mut Self {
        if self.register.apply_swap(qubit1, qubit2).is_ok() {
            self.operations.push(format!("swap q[{}], q[{}];", qubit1, qubit2));
        }
        self
    }

    pub fn cswap(&mut self, control: usize, target1: usize, target2: usize) -> &mut Self {
        if self.register.apply_cswap(control, target1, target2).is_ok() {
            self.operations.push(format!("cswap q[{}], q[{}], q[{}];", control, target1, target2));
        }
        self
    }

    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        if self.register.apply_toffoli(control1, control2, target).is_ok() {
            self.operations.push(format!("ccx q[{}], q[{}], q[{}];", control1, control2, target));
        }
        self
    }

    pub fn rx(&mut self, target: usize, angle: f64) -> &mut Self {
        if self.register.apply_rx(target, angle).is_ok() {
            self.operations.push(format!("rx({}) q[{}];", angle, target));
        }
        self
    }

    pub fn ry(&mut self, target: usize, angle: f64) -> &mut Self {
        if self.register.apply_ry(target, angle).is_ok() {
            self.operations.push(format!("ry({}) q[{}];", angle, target));
        }
        self
    }

    pub fn rz(&mut self, target: usize, angle: f64) -> &mut Self {
        if self.register.apply_rz(target, angle).is_ok() {
            self.operations.push(format!("rz({}) q[{}];", angle, target));
        }
        self
    }

    pub fn controlled_phase(&mut self, control: usize, target: usize, angle: f64) -> &mut Self {
        if self.register.apply_controlled_phase(control, target, angle).is_ok() {
            self.operations.push(format!("cp({}) q[{}], q[{}];", angle, control, target));
        }
        self
    }

    pub fn multi_controlled_x(&mut self, controls: &[usize], target: usize) -> &mut Self {
        if self.register.apply_multi_controlled_x(controls, target).is_ok() {
            let controls_str = controls.iter()
                .map(|c| format!("q[{}]", c))
                .collect::<Vec<_>>()
                .join(", ");
            self.operations.push(format!("mcx {}, q[{}];", controls_str, target));
        }
        self
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
}