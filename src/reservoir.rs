//! Quantum reservoir computing: a fixed, randomly-wired entangling qubit
//! network used as a high-dimensional nonlinear dynamical system
//! (`QuantumReservoir`), paired with a trained linear readout
//! (`QuantumReservoirComputer`) so only the readout -- not the reservoir
//! itself -- needs training.

use crate::complex::MAX_QUBITS;
use crate::core::QuantumRegister;
use std::collections::HashMap;
use std::f64::consts::PI;
use rand::Rng;

/// A single fixed entangling operation baked into the reservoir's dynamics
/// at construction time. `gate` selects which two-qubit interaction runs
/// on the (source, target) edge; the angle is only meaningful for
/// `EdgeGate::ControlledPhase`.
#[derive(Debug, Clone, Copy)]
enum EdgeGate {
    Cnot,
    ControlledZ,
    ControlledPhase(f64),
}

#[derive(Debug, Clone)]
pub struct QuantumReservoir {
    num_qubits: usize,
    reservoir_size: usize,
    connectivity: Vec<Vec<usize>>,
    time_steps: usize,
    measurement_operators: Vec<usize>,
    // Fixed (frozen-at-construction) internal dynamics. A reservoir
    // computer's substrate must be a *fixed* nonlinear system -- only the
    // linear readout is trained. Freezing the entangling-gate choices,
    // their angles, and the local-rotation angles at construction time
    // makes `encode -> fixed dynamics -> measure` a reproducible function
    // of the input, which is what makes the readout learnable.
    fixed_edge_gates: Vec<((usize, usize), EdgeGate)>,
    fixed_local_rx: Vec<f64>,
    fixed_local_rz: Vec<f64>,
}

impl QuantumReservoir {
    pub fn new(num_qubits: usize, connectivity_pattern: &str) -> Result<Self, String> {
        if num_qubits == 0 || num_qubits > MAX_QUBITS {
            return Err("Invalid number of qubits for reservoir".to_string());
        }

        let reservoir_size = 1 << num_qubits; // Hilbert space dimension
        let connectivity = Self::generate_connectivity(num_qubits, connectivity_pattern);
        let measurement_operators = (0..num_qubits).collect(); // Measure all qubits initially

        // Freeze the reservoir's entangling dynamics once, here, rather
        // than resampling them on every evolution -- see the field-level
        // doc comment on `fixed_edge_gates` for why this matters.
        let mut rng = rand::thread_rng();
        let mut fixed_edge_gates = Vec::new();
        for (source, targets) in connectivity.iter().enumerate() {
            for &target in targets {
                if source < target {
                    let gate = match rng.gen_range(0..3) {
                        0 => EdgeGate::Cnot,
                        1 => EdgeGate::ControlledZ,
                        _ => EdgeGate::ControlledPhase(rng.gen_range(0.1..0.5) * PI),
                    };
                    fixed_edge_gates.push(((source, target), gate));
                }
            }
        }
        let fixed_local_rx: Vec<f64> = (0..num_qubits)
            .map(|_| rng.gen_range(0.0..0.2) * PI)
            .collect();
        let fixed_local_rz: Vec<f64> = (0..num_qubits)
            .map(|_| rng.gen_range(0.0..0.2) * PI)
            .collect();

        Ok(Self {
            num_qubits,
            reservoir_size,
            connectivity,
            time_steps: 10, // Default
            measurement_operators,
            fixed_edge_gates,
            fixed_local_rx,
            fixed_local_rz,
        })
    }

    fn generate_connectivity(num_qubits: usize, pattern: &str) -> Vec<Vec<usize>> {
        let mut connectivity = vec![vec![]; num_qubits];
        
        match pattern {
            "all_to_all" => {
                for i in 0..num_qubits {
                    for j in 0..num_qubits {
                        if i != j {
                            connectivity[i].push(j);
                        }
                    }
                }
            }
            "nearest_neighbor" => {
                for i in 0..num_qubits {
                    if i > 0 {
                        connectivity[i].push(i - 1);
                    }
                    if i < num_qubits - 1 {
                        connectivity[i].push(i + 1);
                    }
                }
            }
            "small_world" => {
                // Small-world network with some random long-range connections
                let mut rng = rand::thread_rng();
                for i in 0..num_qubits {
                    // Nearest neighbors
                    if i > 0 {
                        connectivity[i].push(i - 1);
                    }
                    if i < num_qubits - 1 {
                        connectivity[i].push(i + 1);
                    }
                    // Random long-range connections
                    for _ in 0..num_qubits / 3 {
                        let target = rng.gen_range(0..num_qubits);
                        if target != i && !connectivity[i].contains(&target) {
                            connectivity[i].push(target);
                        }
                    }
                }
            }
            _ => {
                // Default to all-to-all
                for i in 0..num_qubits {
                    for j in 0..num_qubits {
                        if i != j {
                            connectivity[i].push(j);
                        }
                    }
                }
            }
        }
        
        connectivity
    }

    pub fn set_time_steps(&mut self, steps: usize) {
        self.time_steps = steps;
    }

    pub fn set_measurement_operators(&mut self, operators: Vec<usize>) {
        self.measurement_operators = operators;
    }

    /// Evolve the reservoir state through time with input encoding
    pub fn evolve_reservoir(
        &self, 
        input_sequence: &[f64],
        noise_level: f64,
    ) -> Result<Vec<Vec<f64>>, String> {
        let mut reservoir_states = Vec::new();
        let mut current_register = QuantumRegister::new(self.num_qubits)?;

        for &input in input_sequence {
            // Encode input into quantum state
            self.encode_input(&mut current_register, input)?;
            
            // Apply reservoir dynamics (entangling operations)
            self.apply_reservoir_dynamics(&mut current_register, noise_level)?;
            
            // Measure and record reservoir state
            let state_features = self.measure_reservoir_state(&current_register)?;
            reservoir_states.push(state_features);
        }

        Ok(reservoir_states)
    }

    /// Encode classical input into quantum state using angle encoding
    fn encode_input(&self, register: &mut QuantumRegister, input: f64) -> Result<(), String> {
        // Normalize input to [0, π] range
        let angle = input * PI;
        
        // Apply rotation to each qubit with different phases
        for qubit in 0..self.num_qubits {
            let phase_shift = 2.0 * PI * (qubit as f64) / (self.num_qubits as f64);
            let effective_angle = angle + phase_shift;
            
            register.apply_ry(qubit, effective_angle)?;
        }
        
        Ok(())
    }

    /// Apply reservoir dynamics (entangling operations)
    fn apply_reservoir_dynamics(
        &self, 
        register: &mut QuantumRegister, 
        noise_level: f64
    ) -> Result<(), String> {
        // Apply the reservoir's *fixed* entangling gates (chosen once at
        // construction -- see `fixed_edge_gates` doc comment). Using the
        // same dynamics on every call is what makes this a reservoir
        // computer rather than a fresh random circuit each time.
        for &((source, target), gate) in &self.fixed_edge_gates {
            match gate {
                EdgeGate::Cnot => register.apply_cnot(source, target)?,
                EdgeGate::ControlledZ => register.apply_controlled_z(source, target)?,
                EdgeGate::ControlledPhase(angle) => {
                    register.apply_controlled_phase(source, target, angle)?
                }
            }
        }

        // Apply the reservoir's fixed local rotations for additional
        // nonlinearity (also frozen at construction time).
        for qubit in 0..self.num_qubits {
            register.apply_rx(qubit, self.fixed_local_rx[qubit])?;
            register.apply_rz(qubit, self.fixed_local_rz[qubit])?;
        }

        // Physical noise, unlike the substrate dynamics above, genuinely
        // should be resampled on every call -- it's modeling a stochastic
        // decoherence process the reservoir is subject to, not part of
        // its fixed transfer function.
        if noise_level > 0.0 {
            let mut rng = rand::thread_rng();
            for qubit in 0..self.num_qubits {
                if rng.gen::<f64>() < noise_level {
                    register.apply_pauli_x(qubit)?;
                }
            }
        }

        Ok(())
    }

    /// Measure reservoir state to extract classical features
    fn measure_reservoir_state(&self, register: &QuantumRegister) -> Result<Vec<f64>, String> {
        let mut features = Vec::new();
        
        // Expectation values of Pauli Z operators
        for &qubit in &self.measurement_operators {
            let expectation = register.expectation_value_pauli_z(qubit)?;
            features.push(expectation);
        }
        
        // Cross-terms for additional features
        for i in 0..self.measurement_operators.len() {
            for j in (i + 1)..self.measurement_operators.len() {
                // Approximate correlation through product of expectations
                let exp_i = register.expectation_value_pauli_z(self.measurement_operators[i])?;
                let exp_j = register.expectation_value_pauli_z(self.measurement_operators[j])?;
                features.push(exp_i * exp_j);
            }
        }
        
        Ok(features)
    }

    /// Get the dimension of the feature vector
    pub fn feature_dimension(&self) -> usize {
        let n = self.measurement_operators.len();
        n + (n * (n - 1)) / 2  // Single expectations + correlations
    }

    pub fn get_reservoir_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("num_qubits".to_string(), self.num_qubits.to_string());
        info.insert("reservoir_size".to_string(), self.reservoir_size.to_string());
        info.insert("feature_dimension".to_string(), self.feature_dimension().to_string());
        info.insert("time_steps".to_string(), self.time_steps.to_string());
        info.insert("connectivity_pattern".to_string(), format!("{:?}", self.connectivity));
        
        info
    }
}

#[derive(Debug, Clone)]
pub struct QuantumReservoirComputer {
    reservoir: QuantumReservoir,
    readout_weights: Vec<f64>,
    regularization: f64,
}

impl QuantumReservoirComputer {
    pub fn new(
        num_qubits: usize, 
        connectivity: &str,
        regularization: f64,
    ) -> Result<Self, String> {
        let reservoir = QuantumReservoir::new(num_qubits, connectivity)?;
        
        Ok(Self {
            reservoir,
            readout_weights: Vec::new(),
            regularization,
        })
    }

    /// Train the reservoir computer on input-output pairs
    pub fn train(
        &mut self,
        training_inputs: &[Vec<f64>],
        training_outputs: &[f64],
        noise_level: f64,
    ) -> Result<f64, String> {
        if training_inputs.len() != training_outputs.len() {
            return Err("Training inputs and outputs must have same length".to_string());
        }

        let mut reservoir_states = Vec::new();
        let mut targets = Vec::new();

        // Collect reservoir states for all training sequences
        for (input_sequence, &target) in training_inputs.iter().zip(training_outputs) {
            let states = self.reservoir.evolve_reservoir(input_sequence, noise_level)?;
            
            // Use the final reservoir state for prediction
            if let Some(final_state) = states.last() {
                reservoir_states.push(final_state.clone());
                targets.push(target);
            }
        }

        // Train readout weights using ridge regression
        self.train_readout_weights(&reservoir_states, &targets)?;

        // Calculate training error
        let training_error = self.calculate_training_error(&reservoir_states, &targets)?;
        
        Ok(training_error)
    }

    fn train_readout_weights(
        &mut self,
        reservoir_states: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<(), String> {
        let feature_dim = self.reservoir.feature_dimension();
        let num_samples = reservoir_states.len();

        // Construct design matrix X and target vector y
        let mut x_matrix = vec![vec![0.0; feature_dim + 1]; num_samples]; // +1 for bias
        let mut y_vector = vec![0.0; num_samples];

        for (i, state) in reservoir_states.iter().enumerate() {
            // Add bias term
            x_matrix[i][0] = 1.0;
            // Copy reservoir features
            for (j, &feature) in state.iter().enumerate() {
                x_matrix[i][j + 1] = feature;
            }
            y_vector[i] = targets[i];
        }

        // Solve (X^T X + λI) w = X^T y using normal equations with regularization
        self.readout_weights = self.ridge_regression(&x_matrix, &y_vector, self.regularization)?;
        
        Ok(())
    }

    fn ridge_regression(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        lambda: f64,
    ) -> Result<Vec<f64>, String> {
        let n_features = x[0].len();
        let n_samples = x.len();

        // Compute X^T X
        let mut xtx = vec![vec![0.0; n_features]; n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..n_samples {
                    xtx[i][j] += x[k][i] * x[k][j];
                }
                // Add regularization to diagonal
                if i == j {
                    xtx[i][j] += lambda;
                }
            }
        }

        // Compute X^T y
        let mut xty = vec![0.0; n_features];
        for i in 0..n_features {
            for k in 0..n_samples {
                xty[i] += x[k][i] * y[k];
            }
        }

        // Solve linear system (simplified - in production use a proper linear algebra library)
        self.solve_linear_system(&xtx, &xty)
    }

    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, String> {
        // Simplified Gaussian elimination for small systems
        let n = b.len();
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        let mut x = vec![0.0; n];

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for j in (i + 1)..n {
                if a[j][i].abs() > a[max_row][i].abs() {
                    max_row = j;
                }
            }

            // Swap rows
            a.swap(i, max_row);
            b.swap(i, max_row);

            // Eliminate
            for j in (i + 1)..n {
                let factor = a[j][i] / a[i][i];
                for k in i..n {
                    a[j][k] -= factor * a[i][k];
                }
                b[j] -= factor * b[i];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            x[i] = b[i];
            for j in (i + 1)..n {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }

        Ok(x)
    }

    fn calculate_training_error(
        &self,
        reservoir_states: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<f64, String> {
        let mut total_error = 0.0;
        let mut count = 0;

        for (state, &target) in reservoir_states.iter().zip(targets) {
            let prediction = self.predict_single(state)?;
            total_error += (prediction - target).powi(2);
            count += 1;
        }

        Ok((total_error / count as f64).sqrt()) // RMSE
    }

    fn predict_single(&self, reservoir_state: &[f64]) -> Result<f64, String> {
        if self.readout_weights.is_empty() {
            return Err("Model not trained".to_string());
        }

        let mut prediction = self.readout_weights[0]; // bias term
        
        for (i, &feature) in reservoir_state.iter().enumerate() {
            prediction += self.readout_weights[i + 1] * feature;
        }

        Ok(prediction)
    }

    /// Make predictions on new input sequences
    pub fn predict(&self, input_sequence: &[f64], noise_level: f64) -> Result<f64, String> {
        let reservoir_states = self.reservoir.evolve_reservoir(input_sequence, noise_level)?;
        
        if let Some(final_state) = reservoir_states.last() {
            self.predict_single(final_state)
        } else {
            Err("No reservoir states generated".to_string())
        }
    }

    /// Get model information
    pub fn get_model_info(&self) -> HashMap<String, String> {
        let mut info = self.reservoir.get_reservoir_info();
        info.insert("trained".to_string(), (!self.readout_weights.is_empty()).to_string());
        info.insert("regularization".to_string(), self.regularization.to_string());
        info.insert("num_weights".to_string(), self.readout_weights.len().to_string());
        
        info
    }

    /// Get feature importance from readout weights
    pub fn get_feature_importance(&self) -> Vec<(usize, f64)> {
        let mut importance = Vec::new();
        
        // Skip bias term (index 0)
        for (i, &weight) in self.readout_weights.iter().enumerate().skip(1) {
            importance.push((i - 1, weight.abs()));
        }
        
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        importance
    }

    /// Public function to demonstrate reservoir computing
    pub fn demonstrate() -> Result<(), String> {
        println!("QUANTUM RESERVOIR COMPUTING DEMONSTRATION");
        println!("=========================================\n");

        // Create a quantum reservoir computer
        let mut qrc = QuantumReservoirComputer::new(4, "small_world", 0.1)?;
        
        println!("Reservoir Information:");
        let info = qrc.get_model_info();
        for (key, value) in info {
            println!("  {}: {}", key, value);
        }
        println!();

        // Generate synthetic training data (sine wave prediction)
        let sequence_length = 20;
        let mut training_inputs = Vec::new();
        let mut training_outputs = Vec::new();

        for i in 0..50 {
            let mut input_seq = Vec::new();
            for j in 0..sequence_length {
                let t = (i + j) as f64 * 0.1;
                input_seq.push((t).sin());
            }
            training_inputs.push(input_seq);
            
            // Predict next value
            let next_t = (i + sequence_length) as f64 * 0.1;
            training_outputs.push(next_t.sin());
        }

        println!("Training Quantum Reservoir Computer...");
        let training_error = qrc.train(&training_inputs, &training_outputs, 0.05)?;
        println!("Training RMSE: {:.6}", training_error);
        println!();

        // Test prediction
        println!("Testing Predictions:");
        let test_input: Vec<f64> = (0..sequence_length).map(|i| (i as f64 * 0.1).sin()).collect();
        let true_next = (sequence_length as f64 * 0.1).sin();
        
        let prediction = qrc.predict(&test_input, 0.05)?;
        println!("Predicted: {:.6}, True: {:.6}, Error: {:.6}", 
                 prediction, true_next, (prediction - true_next).abs());

        // Show feature importance
        println!("\nTop 5 Most Important Features:");
        let importance = qrc.get_feature_importance();
        for (i, (feature_idx, weight)) in importance.iter().take(5).enumerate() {
            println!("  {}. Feature {}: |weight| = {:.6}", i + 1, feature_idx, weight);
        }

        Ok(())
    }
}

/// Run a small end-to-end demonstration of the quantum reservoir computer.
pub fn demonstrate_quantum_reservoir_computing() -> Result<(), String> {
    QuantumReservoirComputer::demonstrate()
}