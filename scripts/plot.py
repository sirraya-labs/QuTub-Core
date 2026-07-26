#!/usr/bin/env python3
"""
Generate publication-quality plots from sirraya-qutub benchmark data.

Usage:
    python scripts/plot.py

Input:  data/scaling.csv, data/fidelity_vs_depth.csv,
        data/entanglement.csv, data/noise.csv

Output: docs/scaling.png, docs/fidelity_vs_depth.png,
        docs/entanglement.png, docs/noise.png

Requires: pandas, matplotlib, numpy
    pip install pandas matplotlib numpy
"""

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style: clean, readable, journal-ready
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

# Use DejaVu Sans if available (ships with matplotlib, supports Unicode subscripts).
# Fall back to sans-serif if not.
matplotlib.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
})

# ---------------------------------------------------------------------------
# 1. Runtime scaling (log scale — standard for exponential algorithms)
# ---------------------------------------------------------------------------

def plot_runtime_scaling():
    df = pd.read_csv(DATA_DIR / "scaling.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(df["qubits"], df["hadamard_us"], "o-", label="Hadamard chain", linewidth=1.5, markersize=6)
    ax.semilogy(df["qubits"], df["cnot_us"], "s-", label="CNOT chain", linewidth=1.5, markersize=6)
    ax.semilogy(df["qubits"], df["qft_us"], "^-", label="QFT", linewidth=1.5, markersize=6)

    ax.set_xlabel("Qubits")
    ax.set_ylabel("Time (µs)")
    ax.set_title("sirraya-qutub Runtime Scaling")
    ax.legend(frameon=True)
    ax.set_xticks(df["qubits"])

    ax.annotate(
        "O(2\u207f) regime",
        xy=(16, 19098),
        xytext=(12, 50000),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
        color="gray",
    )

    fig.savefig(DOCS_DIR / "scaling.png")
    plt.close(fig)
    print("  -> docs/scaling.png")


# ---------------------------------------------------------------------------
# 2. Fidelity vs gate depth
# ---------------------------------------------------------------------------

def plot_fidelity_vs_depth():
    df = pd.read_csv(DATA_DIR / "fidelity_vs_depth.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for qubits, color in zip(sorted(df["qubits"].unique()), colors):
        subset = df[df["qubits"] == qubits]
        ax.semilogx(
            subset["depth"],
            subset["fidelity"],
            "o-",
            label=f"{qubits} qubits",
            linewidth=1.5,
            markersize=5,
            color=color,
        )

    ax.set_xlabel("Circuit Depth (log scale)")
    ax.set_ylabel("Self-fidelity")
    ax.set_title("Numerical Fidelity vs Circuit Depth\n(no noise — exact unitary evolution)")
    ax.legend(frameon=True)
    ax.set_ylim(0.9999, 1.0001)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)

    fig.savefig(DOCS_DIR / "fidelity_vs_depth.png")
    plt.close(fig)
    print("  -> docs/fidelity_vs_depth.png")


# ---------------------------------------------------------------------------
# 3. Entanglement metrics
# ---------------------------------------------------------------------------

def plot_entanglement():
    df = pd.read_csv(DATA_DIR / "entanglement.csv")
    ghz = df[df["state"] == "GHZ"]
    w = df[df["state"] == "W"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel 1: Negativity vs qubit count ---
    axes[0].plot(
        ghz["qubits"], ghz["negativity"], "o-",
        color="#1f77b4", linewidth=2, markersize=7
    )
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Theory: 0.5")
    axes[0].set_xlabel("Qubits")
    axes[0].set_ylabel("Negativity")
    axes[0].set_title("GHZ Negativity\n(cut: q0 vs rest)")
    axes[0].legend()
    axes[0].set_xticks(ghz["qubits"])

    # --- Panel 2: Von Neumann entropy ---
    axes[1].plot(
        ghz["qubits"], ghz["von_neumann_entropy"], "o-",
        label="GHZ", linewidth=2, markersize=7, color="#1f77b4"
    )
    axes[1].plot(
        w["qubits"], w["von_neumann_entropy"], "s-",
        label="W", linewidth=2, markersize=7, color="#ff7f0e"
    )
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, label="I/2 (maximally mixed)")
    axes[1].set_xlabel("Qubits")
    axes[1].set_ylabel("von Neumann Entropy (bits)")
    axes[1].set_title("Reduced-State Entropy\n(trace out all but q0)")
    axes[1].legend()
    axes[1].set_xticks(ghz["qubits"])

    # --- Panel 3: Concurrence (2-qubit only) ---
    concur = ghz[ghz["concurrence"] >= 0]
    if len(concur) > 0:
        axes[2].bar(
            ["Bell state"], [concur["concurrence"].values[0]],
            color="#2ca02c", width=0.3
        )
        axes[2].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Maximally entangled")
        axes[2].set_ylabel("Concurrence")
        axes[2].set_title("Concurrence (2-qubit)")
        axes[2].legend()
        axes[2].set_ylim(0, 1.2)
        axes[2].text(
            0, concur["concurrence"].values[0] + 0.03,
            f'{concur["concurrence"].values[0]:.3f}',
            ha="center", fontsize=11
        )
    else:
        axes[2].text(0.5, 0.5, "No 2-qubit data", ha="center", va="center")
        axes[2].set_title("Concurrence (2-qubit)")

    fig.suptitle("sirraya-qutub Entanglement Metrics", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "entanglement.png")
    plt.close(fig)
    print("  -> docs/entanglement.png")


# ---------------------------------------------------------------------------
# 4. Noise channel behavior
# ---------------------------------------------------------------------------

def plot_noise_curves():
    df = pd.read_csv(DATA_DIR / "noise.csv")
    depolarizing = df[df["channel"] == "depolarizing"]
    damping = df[df["channel"] == "amplitude_damping"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel 1: Purity ---
    axes[0].plot(
        depolarizing["probability"], depolarizing["purity"],
        "o-", label="Depolarizing", linewidth=2, markersize=7, color="#1f77b4"
    )
    axes[0].plot(
        damping["probability"], damping["purity"],
        "s-", label="Amplitude Damping", linewidth=2, markersize=7, color="#ff7f0e"
    )
    axes[0].axhline(y=1.0/3.0, color="#1f77b4", linestyle=":", alpha=0.4, label="Depol. minimum: 1/3")
    axes[0].set_xlabel("Noise Strength (p or \u03b3)")
    axes[0].set_ylabel("Purity")
    axes[0].set_title("Purity Decay")
    axes[0].legend(fontsize=9)

    # --- Panel 2: Fidelity ---
    axes[1].plot(
        depolarizing["probability"], depolarizing["fidelity"],
        "o-", label="Depolarizing", linewidth=2, markersize=7, color="#1f77b4"
    )
    axes[1].plot(
        damping["probability"], damping["fidelity"],
        "s-", label="Amplitude Damping", linewidth=2, markersize=7, color="#ff7f0e"
    )
    axes[1].set_xlabel("Noise Strength (p or \u03b3)")
    axes[1].set_ylabel("Fidelity")
    axes[1].set_title("Fidelity Decay")
    axes[1].legend()

    # --- Panel 3: Trace preservation ---
    axes[2].plot(
        depolarizing["probability"], depolarizing["trace"],
        "o-", label="Depolarizing", linewidth=2, markersize=7, color="#1f77b4"
    )
    axes[2].plot(
        damping["probability"], damping["trace"],
        "s-", label="Amplitude Damping", linewidth=2, markersize=7, color="#ff7f0e"
    )
    axes[2].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    axes[2].set_xlabel("Noise Strength (p or \u03b3)")
    axes[2].set_ylabel("Trace")
    axes[2].set_title("Trace Preservation\n(both channels are CPTP)")
    axes[2].legend()
    axes[2].set_ylim(0.999, 1.001)

    fig.suptitle("sirraya-qutub Noise Channel Behavior", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "noise.png")
    plt.close(fig)
    print("  -> docs/noise.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating plots from sirraya-qutub benchmark data...\n")
    plot_runtime_scaling()
    plot_fidelity_vs_depth()
    plot_entanglement()
    plot_noise_curves()
    print("\nDone. Figures saved to docs/")