use sirraya_qutub::core::QuantumRegister;

fn main() -> Result<(), String> {
    println!("══════════════════════════════════════════════════════════════");
    println!("             QuTub • Quantum Teleportation");
    println!("══════════════════════════════════════════════════════════════");

    // ── Prepare the unknown state |ψ⟩ = (|0⟩ + i|1⟩)/√2 ────────
    let mut register = QuantumRegister::new(3)?;
    register.apply_hadamard(0)?;
    register.apply_s_gate(0)?;

    println!("\nInput State (qubit 0)");
    println!("──────────────────────────────────────────────────────────────");
    println!("|ψ⟩ = (|0⟩ + i|1⟩) / √2");
    register.print_state();

    // ── Create Bell pair between q1 and q2 ──────────────────────
    register.apply_hadamard(1)?;
    register.apply_cnot(1, 2)?;

    println!("\nBell Pair (qubits 1-2)");
    println!("──────────────────────────────────────────────────────────────");
    println!("q1-q2 in |Φ⁺⟩ = (|00⟩ + |11⟩) / √2");

    // ── Alice: CNOT q0→q1, then Hadamard q0 ─────────────────────
    register.apply_cnot(0, 1)?;
    register.apply_hadamard(0)?;

    // ── Alice measures ──────────────────────────────────────────
    let m0 = register.measure_single_qubit(0)?;
    let m1 = register.measure_single_qubit(1)?;

    println!("\nAlice's Measurements");
    println!("──────────────────────────────────────────────────────────────");
    println!("q0 = {}", m0);
    println!("q1 = {}", m1);

    // ── Bob's corrections ───────────────────────────────────────
    println!("\nClassical Corrections (applied to q2)");
    println!("──────────────────────────────────────────────────────────────");
    if m1 == 1 {
        register.apply_pauli_x(2)?;
        println!("X correction     Yes  (m1 = 1)");
    } else {
        println!("X correction     No   (m1 = 0)");
    }
    if m0 == 1 {
        register.apply_pauli_z(2)?;
        println!("Z correction     Yes  (m0 = 1)");
    } else {
        println!("Z correction     No   (m0 = 0)");
    }

    // ── Target reduced state (what Bob should receive) ──────────
    println!("\nTarget Reduced State");
    println!("──────────────────────────────────────────────────────────────");
    println!("ρ_target = |ψ⟩⟨ψ|");
    println!();
    println!("Expected density matrix:");
    println!("  [[0.5, -0.5i],");
    println!("   [0.5i,  0.5]]");
    println!("  Trace:  1.0");
    println!("  Purity: 1.0 (pure)");

    // ── Extract Bob's reduced state via partial trace ───────────
    let density = register.to_density_matrix()?;
    let bob = density.partial_trace(&[2])?;

    println!("\nBob's Reduced State");
    println!("──────────────────────────────────────────────────────────────");
    println!("ρ_B (partial trace over qubits 0 and 1)");
    bob.print_density_matrix();

    // ── Pauli expectations from the reduced density matrix ──────
    let bob_matrix = bob.get_matrix();
    let rho00 = bob_matrix[0][0];
    let rho01 = bob_matrix[0][1];
    let rho11 = bob_matrix[1][1];
    let bob_x = 2.0 * rho01.real();
    let bob_y = -2.0 * rho01.imag();
    let bob_z = rho00.real() - rho11.real();

    // Expected Bloch vector for |ψ⟩ = (|0⟩+i|1⟩)/√2 is (0, 1, 0)
    let expected_x = 0.0;
    let expected_y = 1.0;
    let expected_z = 0.0;

    // ── Fidelity: F = (1 + r_ψ · r_B) / 2 ───────────────────────
    let fidelity =
        (1.0 + bob_x * expected_x + bob_y * expected_y + bob_z * expected_z) / 2.0;

    println!("\n══════════════════════════════════════════════════════════════");
    println!("                    Verification");
    println!("══════════════════════════════════════════════════════════════");

    println!("\nPauli Expectations");
    println!("──────────────────────────────────────────────────────────────");
    println!("Observable       Target         ρ_B           Match");
    println!("──────────────────────────────────────────────────────────────");
    let match_x = (bob_x - expected_x).abs() < 1e-6;
    let match_y = (bob_y - expected_y).abs() < 1e-6;
    let match_z = (bob_z - expected_z).abs() < 1e-6;
    println!(
        "⟨X⟩              {:.6}        {:.6}     {}",
        expected_x,
        bob_x,
        if match_x { "✓" } else { "✗" }
    );
    println!(
        "⟨Y⟩              {:.6}        {:.6}     {}",
        expected_y,
        bob_y,
        if match_y { "✓" } else { "✗" }
    );
    println!(
        "⟨Z⟩              {:.6}        {:.6}     {}",
        expected_z,
        bob_z,
        if match_z { "✓" } else { "✗" }
    );

    println!("\nState Fidelity");
    println!(
        "    F(|ψ⟩, ρ_B) = {:.6}  {}",
        fidelity,
        if (fidelity - 1.0).abs() < 1e-6 {
            "✓  ρ_B matches ρ_target"
        } else {
            "✗"
        }
    );

    println!("\nClassical Communication: 2 bits");
    println!("Entanglement Resource:   1 Bell pair");

    println!("\n──────────────────────────────────────────────────────────────");
    if (fidelity - 1.0).abs() < 1e-6 {
        println!("✓ Quantum state successfully teleported");
        println!("✓ ρ_B matches ρ_target");
        println!("✓ All three Pauli expectations match");
        println!("✓ Fidelity = 1.0");
    } else {
        println!("✗ Teleportation verification failed");
    }

    println!("\n══════════════════════════════════════════════════════════════");
    println!("The Bell pair alone doesn't teleport anything.");
    println!("Entanglement creates the resource. Alice's measurement");
    println!("generates two classical bits. Those bits tell Bob which");
    println!("correction to perform. That's the interaction between");
    println!("entanglement + measurement + classical information that");
    println!("makes quantum teleportation work.");
    println!("══════════════════════════════════════════════════════════════");

    Ok(())
}