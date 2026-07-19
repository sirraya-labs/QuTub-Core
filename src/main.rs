// quantum_simulator is a general-purpose library module: it exposes a
// public API (e.g. von_neumann_entropy, the Deutsch-Jozsa/Grover helpers,
// DensityMatrix constructors) that this CLI doesn't happen to call yet but
// that downstream users of the module -- and its own #[cfg(test)] suite --
// do. #[allow(dead_code)] suppresses the resulting "never used" noise
// without hiding genuinely dead code inside the module itself.
#[allow(dead_code)]
mod quantum_simulator;

use quantum_simulator::*;
use std::f64::consts::PI;
use std::env;

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    
    // Check for CLI commands
    if args.len() > 1 {
        match args[1].as_str() {
            "reservoir" => {
                return handle_reservoir_command(&args);
            }
            "benchmark" => {
                return handle_benchmark_command(&args);
            }
            "help" | "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                println!("Unknown command: {}", args[1]);
                print_help();
                return Ok(());
            }
        }
    }

    // Default: run all demonstrations
    run_all_demonstrations()
}

fn handle_reservoir_command(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        println!("Usage: {} reservoir <subcommand>", args[0]);
        println!("Subcommands:");
        println!("  demo          - Run reservoir computing demonstration");
        println!("  train         - Train on custom data");
        println!("  predict       - Make prediction on input sequence");
        println!("  info          - Show reservoir information");
        return Ok(());
    }

    match args[2].as_str() {
        "demo" => {
            println!("QUANTUM RESERVOIR COMPUTING DEMONSTRATION");
            println!("=========================================\n");
            demonstrate_quantum_reservoir_computing()
        }
        "train" => {
            if args.len() < 4 {
                println!("Usage: {} reservoir train <qubits>", args[0]);
                return Ok(());
            }
            let qubits: usize = args[3].parse().map_err(|_| "Invalid number of qubits".to_string())?;
            train_custom_reservoir(qubits)
        }
        "predict" => {
            if args.len() < 5 {
                println!("Usage: {} reservoir predict <qubits> <input_values...>", args[0]);
                return Ok(());
            }
            let qubits: usize = args[3].parse().map_err(|_| "Invalid number of qubits".to_string())?;
            let input_values: Vec<f64> = args[4..].iter()
                .map(|s| s.parse().map_err(|_| format!("Invalid input value: {}", s)))
                .collect::<Result<Vec<f64>, String>>()?;
            predict_with_reservoir(qubits, &input_values)
        }
        "info" => {
            if args.len() < 4 {
                println!("Usage: {} reservoir info <qubits>", args[0]);
                return Ok(());
            }
            let qubits: usize = args[3].parse().map_err(|_| "Invalid number of qubits".to_string())?;
            show_reservoir_info(qubits)
        }
        _ => {
            println!("Unknown reservoir subcommand: {}", args[2]);
            Ok(())
        }
    }
}

fn handle_benchmark_command(args: &[String]) -> Result<(), String> {
    if args.len() > 2 && args[2] == "xeb" {
        let qubits: usize = if args.len() > 3 {
            args[3].parse().map_err(|_| "Invalid number of qubits".to_string())?
        } else {
            6
        };
        let samples: usize = if args.len() > 4 {
            args[4].parse().map_err(|_| "Invalid sample count".to_string())?
        } else {
            8000
        };
        return run_xeb_benchmark_command(qubits, samples);
    }

    let qubits = if args.len() > 2 {
        args[2].parse().map_err(|_| "Invalid number of qubits".to_string())?
    } else {
        4
    };

    println!("QUANTUM RESERVOIR COMPUTING BENCHMARK");
    println!("=====================================\n");
    
    benchmark_reservoir_performance(qubits)
}

fn run_xeb_benchmark_command(qubits: usize, samples: usize) -> Result<(), String> {
    println!("CROSS-ENTROPY BENCHMARKING (XEB)");
    println!("=================================\n");
    println!("Methodology: linear XEB fidelity estimator, as used to validate");
    println!("real quantum hardware against classical simulation (Arute et al.,");
    println!("\"Quantum supremacy using a programmable superconducting processor\",");
    println!("Nature 574, 505-510, 2019).\n");

    let calibration = HardwareCalibration::quantinuum_helios_2026();
    println!("Noise calibration: {}", calibration.name);
    println!("  Single-qubit gate fidelity: {:.6}%", calibration.single_qubit_fidelity * 100.0);
    println!("  Two-qubit gate fidelity:    {:.6}%", calibration.two_qubit_fidelity * 100.0);
    println!("  -> single-qubit depolarizing p = {:.8}", calibration.single_qubit_error_probability());
    println!("  -> two-qubit depolarizing p    = {:.8}\n", calibration.two_qubit_error_probability());

    println!("Running {}-qubit benchmark circuit with {} samples...", qubits, samples);
    let fidelity = run_xeb_demo(qubits, calibration, samples)?;

    println!("\nResults:");
    println!("  F_XEB = {:.6}", fidelity);
    println!("  (1.0 = perfect match to ideal/noiseless simulation,");
    println!("   0.0 = fully depolarized/random output)");

    Ok(())
}

fn print_help() {
    println!("Quantum Simulator CLI");
    println!("====================");
    println!();
    println!("Commands:");
    println!("  reservoir    - Quantum Reservoir Computing operations");
    println!("  benchmark    - Performance benchmarking");
    println!("  help         - Show this help message");
    println!();
    println!("Reservoir Subcommands:");
    println!("  demo         - Run demonstration with 4-qubit reservoir");
    println!("  train <n>    - Train n-qubit reservoir on synthetic data");
    println!("  predict <n> <values...> - Make prediction using n-qubit reservoir");
    println!("  info <n>     - Show information about n-qubit reservoir");
    println!();
    println!("Examples:");
    println!("  {} reservoir demo", env::args().next().unwrap());
    println!("  {} reservoir train 6", env::args().next().unwrap());
    println!("  {} reservoir predict 4 0.1 0.2 0.3 0.4", env::args().next().unwrap());
    println!("  {} benchmark 8", env::args().next().unwrap());
}

fn run_all_demonstrations() -> Result<(), String> {
    println!("Complete Quantum Simulator with All Features");
    println!("===========================================\n");

    // Test 1: Basic quantum states
    println!("Test 1: Basic Quantum States");
    println!("---------------------------");
    
    let bell_state = create_bell_state()?;
    println!("Bell State:");
    bell_state.print_state();
    
    let ghz_state = create_ghz_state(3)?;
    println!("\nGHZ State (3 qubits):");
    ghz_state.print_state();
    println!();

    // Test 2: Density matrix and noise
    println!("Test 2: Density Matrix and Noise Channels");
    println!("----------------------------------------");
    
    let mut density = DensityMatrix::from_state_vector(bell_state.get_state_vector())?;
    println!("Initial pure Bell state density matrix:");
    density.print_density_matrix();
    
    density.apply_depolarizing_channel(0.2, 0)?;
    println!("\nAfter 20% depolarizing noise on qubit 0:");
    density.print_density_matrix();
    
    let mut density2 = DensityMatrix::from_state_vector(bell_state.get_state_vector())?;
    density2.apply_amplitude_damping(0.3, 0)?;
    println!("\nAfter 30% amplitude damping on qubit 0:");
    density2.print_density_matrix();
    println!();

    // Test 3: Advanced gates and circuits
    println!("Test 3: Advanced Quantum Gates");
    println!("-----------------------------");
    
    let mut circuit = QuantumCircuit::new(4)?;
    circuit
        .hadamard(0)
        .s(1)
        .t(2)
        .cnot(0, 1)
        .cswap(0, 2, 3)
        .toffoli(1, 2, 3)
        .rx(0, PI/4.0)
        .multi_controlled_x(&[0, 1], 2);
    
    circuit.print_circuit();
    circuit.get_register().print_state();
    println!();

    // Test 4: Quantum Fourier Transform
    println!("Test 4: Quantum Fourier Transform");
    println!("---------------------------------");
    
    let mut qft_reg = QuantumRegister::new(3)?;
    qft_reg.apply_pauli_x(0)?;
    qft_reg.apply_pauli_x(2)?;
    
    println!("Before QFT:");
    qft_reg.print_state();
    
    quantum_fourier_transform(&mut qft_reg)?;
    println!("\nAfter QFT:");
    qft_reg.print_state();
    
    inverse_quantum_fourier_transform(&mut qft_reg)?;
    println!("\nAfter inverse QFT:");
    qft_reg.print_state();
    println!();

    // Test 5: Measurements and probabilities
    println!("Test 5: Measurements and Probabilities");
    println!("-------------------------------------");
    
    let mut meas_reg = QuantumRegister::new(2)?;
    meas_reg.apply_hadamard(0)?;
    meas_reg.apply_cnot(0, 1)?;
    
    let (result, prob) = meas_reg.measure_single_qubit_with_probability(0)?;
    println!("Single qubit measurement: result={}, probability={:.6}", result, prob);
    
    let dist = meas_reg.get_probability_distribution();
    println!("Probability distribution: {:?}", dist);
    println!();

    // Test 6: Performance benchmarks
    println!("Test 6: Performance Benchmarks");
    println!("------------------------------");
    QuantumBenchmark::run_comprehensive_benchmark();
    println!();

    // Test 7: QASM export
    println!("Test 7: QASM Circuit Export");
    println!("---------------------------");
    let qasm_code = circuit.to_qasm("advanced_circuit");
    println!("{}", qasm_code);

    // Test 8: Quantum Reservoir Computing
    println!("\nTest 8: Quantum Reservoir Computing");
    println!("-----------------------------------");
    demonstrate_quantum_reservoir_computing()?;

    Ok(())
}

// Reservoir-specific functions
fn train_custom_reservoir(qubits: usize) -> Result<(), String> {
    println!("TRAINING {}-QUBIT QUANTUM RESERVOIR COMPUTER", qubits);
    println!("============================================\n");

    let mut qrc = QuantumReservoirComputer::new(qubits, "small_world", 0.1)?;
    
    // Generate training data (sine wave prediction)
    let sequence_length = 15;
    let mut training_inputs = Vec::new();
    let mut training_outputs = Vec::new();

    println!("Generating training data (sine wave prediction task)...");
    for i in 0..100 {
        let mut input_seq = Vec::new();
        for j in 0..sequence_length {
            let t = (i + j) as f64 * 0.1;
            input_seq.push((t).sin());
        }
        training_inputs.push(input_seq);
        
        let next_t = (i + sequence_length) as f64 * 0.1;
        training_outputs.push(next_t.sin());
    }

    println!("Training reservoir with {} samples...", training_inputs.len());
    let training_error = qrc.train(&training_inputs, &training_outputs, 0.05)?;
    
    println!("\nTraining Results:");
    println!("  Qubits: {}", qubits);
    println!("  Reservoir Size: {}", 1 << qubits);
    println!("  Feature Dimension: {}", qrc.get_model_info().get("feature_dimension").unwrap());
    println!("  Training RMSE: {:.6}", training_error);
    println!("  Model Trained: ✓");

    // Show feature importance
    println!("\nTop 5 Most Important Features:");
    let importance = qrc.get_feature_importance();
    for (i, (feature_idx, weight)) in importance.iter().take(5).enumerate() {
        println!("  {}. Feature {}: |weight| = {:.6}", i + 1, feature_idx, weight);
    }

    Ok(())
}

fn predict_with_reservoir(qubits: usize, input_values: &[f64]) -> Result<(), String> {
    println!("PREDICTION WITH {}-QUBIT QUANTUM RESERVOIR", qubits);
    println!("==========================================\n");

    // Create and train a reservoir
    let mut qrc = QuantumReservoirComputer::new(qubits, "small_world", 0.1)?;
    
    // Quick training on simple pattern
    let sequence_length = input_values.len();
    let mut training_inputs = Vec::new();
    let mut training_outputs = Vec::new();

    for i in 0..50 {
        let mut input_seq = Vec::new();
        for j in 0..sequence_length {
            let t = (i + j) as f64 * 0.1;
            input_seq.push((t).sin());
        }
        training_inputs.push(input_seq);
        
        let next_t = (i + sequence_length) as f64 * 0.1;
        training_outputs.push(next_t.sin());
    }

    println!("Quick training reservoir...");
    let _ = qrc.train(&training_inputs, &training_outputs, 0.05)?;

    // Make prediction
    println!("Input sequence: {:?}", input_values);
    let prediction = qrc.predict(input_values, 0.05)?;
    
    println!("\nPrediction Result:");
    println!("  Input length: {}", input_values.len());
    println!("  Predicted value: {:.6}", prediction);
    
    // If we can compute the expected next value for demonstration
    if input_values.len() > 0 {
        let last_input = input_values[input_values.len() - 1];
        let expected_next = (last_input + 0.1).sin(); // Simple extrapolation
        println!("  Expected next (sin(x+0.1)): {:.6}", expected_next);
        println!("  Prediction error: {:.6}", (prediction - expected_next).abs());
    }

    Ok(())
}

fn show_reservoir_info(qubits: usize) -> Result<(), String> {
    println!("{}-QUBIT QUANTUM RESERVOIR INFORMATION", qubits);
    println!("=====================================\n");

    let qrc = QuantumReservoirComputer::new(qubits, "small_world", 0.1)?;
    let info = qrc.get_model_info();

    println!("Reservoir Properties:");
    for (key, value) in info {
        println!("  {:20}: {}", key, value);
    }

    println!("\nTheoretical Capacity:");
    println!("  Hilbert Space Dimension: 2^{} = {}", qubits, 1 << qubits);
    println!("  Maximum Feature Dimension: {}", qubits + (qubits * (qubits - 1)) / 2);
    println!("  Connectivity: Small-world network");
    println!("  Entanglement: Full quantum correlations");

    Ok(())
}

fn benchmark_reservoir_performance(qubits: usize) -> Result<(), String> {
    println!("BENCHMARKING {}-QUBIT QUANTUM RESERVOIR", qubits);
    println!("======================================\n");

    let mut qrc = QuantumReservoirComputer::new(qubits, "small_world", 0.1)?;
    
    // Generate benchmark data
    let sequence_length = 10;
    let mut training_inputs = Vec::new();
    let mut training_outputs = Vec::new();

    let num_samples = 100;
    println!("Generating {} training samples...", num_samples);
    
    for i in 0..num_samples {
        let mut input_seq = Vec::new();
        for j in 0..sequence_length {
            let t = (i + j) as f64 * 0.1;
            input_seq.push((t).sin());
        }
        training_inputs.push(input_seq);
        
        let next_t = (i + sequence_length) as f64 * 0.1;
        training_outputs.push(next_t.sin());
    }

    println!("Starting benchmark...");
    let start = std::time::Instant::now();
    
    let training_error = qrc.train(&training_inputs, &training_outputs, 0.05)?;
    
    let training_time = start.elapsed();
    
    // Test prediction speed
    let test_input: Vec<f64> = (0..sequence_length).map(|i| (i as f64 * 0.1).sin()).collect();
    
    let predict_start = std::time::Instant::now();
    let prediction = qrc.predict(&test_input, 0.05)?;
    let predict_time = predict_start.elapsed();

    println!("\nBenchmark Results:");
    println!("  Qubits: {}", qubits);
    println!("  Reservoir Size: {}", 1 << qubits);
    println!("  Training Samples: {}", num_samples);
    println!("  Training Time: {:?}", training_time);
    println!("  Prediction Time: {:?}", predict_time);
    println!("  Training RMSE: {:.6}", training_error);
    println!("  Final Prediction: {:.6}", prediction);
    println!("  Samples/second: {:.1}", num_samples as f64 / training_time.as_secs_f64());

    Ok(())
}