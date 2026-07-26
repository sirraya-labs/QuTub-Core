use sirraya_qutub::core::create_bell_state;

/// Alice's encoding: apply one of {I, X, Z, ZX} to her half of the Bell
/// pair depending on the 2-bit message (b1, b0).
fn encode(register: &mut sirraya_qutub::core::QuantumRegister, b1: u8, b0: u8) -> Result<&'static str, String> {
    match (b1, b0) {
        (0, 0) => Ok("I  (identity)"),
        (0, 1) => {
            register.apply_pauli_x(0)?;
            Ok("X  (bit flip)")
        }
        (1, 0) => {
            register.apply_pauli_z(0)?;
            Ok("Z  (phase flip)")
        }
        (1, 1) => {
            register.apply_pauli_z(0)?;
            register.apply_pauli_x(0)?;
            Ok("ZX (bit + phase flip)")
        }
        _ => unreachable!(),
    }
}

/// Bob's decoding: disentangle the pair and measure in the computational
/// basis to recover both classical bits from the single qubit he received.
fn decode(register: &mut sirraya_qutub::core::QuantumRegister) -> Result<(u8, u8), String> {
    register.apply_cnot(0, 1)?;
    register.apply_hadamard(0)?;
    let m0 = register.measure_single_qubit(0)?;
    let m1 = register.measure_single_qubit(1)?;
    Ok((m0, m1))
}

fn main() -> Result<(), String> {
    println!("══════════════════════════════════════════════════════════════");
    println!("             QuTub • Superdense Coding");
    println!("══════════════════════════════════════════════════════════════");
    println!("\nSuperdense coding sends 2 classical bits by physically");
    println!("transmitting only 1 qubit, using a pre-shared Bell pair as");
    println!("the resource. It's the mirror image of teleportation: there,");
    println!("2 classical bits + entanglement send 1 qubit; here, 1 qubit");
    println!("+ entanglement sends 2 classical bits.");

    let messages: [(u8, u8); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let mut all_ok = true;

    for (b1, b0) in messages {
        println!("\n──────────────────────────────────────────────────────────────");
        println!("Message to send: b1={} b0={}  (\"{}{}\")", b1, b0, b1, b0);
        println!("──────────────────────────────────────────────────────────────");

        // ── Shared resource: one Bell pair, qubit 0 = Alice, qubit 1 = Bob
        let mut register = create_bell_state()?;
        println!("Shared Bell pair |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 (qubits 0-1)");

        // ── Alice encodes her 2 bits onto her single qubit ───────
        let gate_applied = encode(&mut register, b1, b0)?;
        println!("Alice applies:   {}", gate_applied);
        println!("Alice sends her qubit (qubit 0) to Bob — 1 qubit, physically");

        // ── Bob decodes using his qubit + the one he received ────
        let (m0, m1) = decode(&mut register)?;
        println!("Bob decodes:     CNOT(0→1), H(0), measure both");
        println!("Bob recovers:    b1={} b0={}", m0, m1);

        let matches = m0 == b1 && m1 == b0;
        all_ok &= matches;
        println!("Match:           {}", if matches { "✓" } else { "✗" });
    }

    println!("\n══════════════════════════════════════════════════════════════");
    println!("                    Verification");
    println!("══════════════════════════════════════════════════════════════");
    if all_ok {
        println!("✓ All 4 two-bit messages decoded correctly");
        println!("✓ 1 physical qubit carried 2 classical bits of information");
        println!("✓ Bell pair consumed as the shared entanglement resource");
    } else {
        println!("✗ Superdense coding verification failed");
    }

    println!("\nClassical Communication: 0 bits (before qubit transmission)");
    println!("Quantum Communication:   1 qubit");
    println!("Entanglement Resource:   1 Bell pair (consumed)");
    println!("Classical Information Delivered: 2 bits");

    println!("\n══════════════════════════════════════════════════════════════");
    println!("Without the pre-shared Bell pair, a single qubit can only");
    println!("ever carry 1 bit of reliably distinguishable classical");
    println!("information (Holevo's bound). Entanglement lets the two");
    println!("parties trade a shared quantum resource, prepared in advance,");
    println!("for extra classical channel capacity later — the same");
    println!("resource that teleportation spends to move a qubit using only");
    println!("classical bits, run in reverse.");
    println!("══════════════════════════════════════════════════════════════");

    Ok(())
}