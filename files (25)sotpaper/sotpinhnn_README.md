# sotpinhnn

### Split-Operator Techniques, Physics-Informed Neural Networks, and Hamiltonian Neural Networks — diagnosing where machine learning breaks physics, and where it doesn't

A neural network trained to directly mimic a physics simulation can violate an exact conservation law by 260% while its training loss looks perfectly healthy. This repository diagnoses exactly why that happens, confirms the mechanism on an independent quantum system, and tests a structurally different architecture — a Hamiltonian Neural Network — that makes the same violation mathematically impossible rather than merely unlikely.

**[Read the full explainer, with industrial context and every animation →](docs/index.html)**   ·   **[Full paper (PDF) →](paper/manuscript.pdf)**   ·   **[Complete math derivations (PDF) →](supplementary/derivations.pdf)**

---

## The one-sentence concept

> Training a network to *directly output* a physics simulation's answer lets it find loopholes that satisfy the training loss while quietly breaking the physics; training it to learn the *underlying rules of motion* instead, and letting a real theorem enforce the physics, closes the loophole by construction rather than by hoping the optimizer behaves.

If it holds: a network trained on local dynamics data conserves probability/energy exactly, regardless of how well or badly it was trained — this is provable, not just observed (see [the proof](docs/hnn_explainer.html#proof)).

If it doesn't hold — a network trained by direct residual regression — it can settle into a low-amplitude background that trivially satisfies its own training objective almost everywhere, while integrating to a physically nonsensical total.

---

## What's actually in this repo

```
sotpinhnn/
├── README.md                    ← you are here
├── paper/
│   ├── manuscript.tex           ← full REVTeX4-2 source (Physical Review A style)
│   └── manuscript.pdf           ← 15 pages, 24 figures, 20 references
├── supplementary/
│   ├── derivations.tex          ← every equation, derived step by step
│   └── derivations.pdf
├── docs/
│   └── index.html              ← single-page explainer: industrial motivation, objectives, SOT phase space, PINN failure, HNN fix, full GIF gallery
├── src/
│   ├── classical/                ← double-well PINN, hard-constraint fix, HNN, chaos test (32 scripts)
│   ├── quantum/                  ← Lindblad qubit PINN + dissipation-rate robustness sweep
│   └── network/                  ← coupled two-oscillator extension
├── figures/                       ← all 27 figures used across the paper and docs
└── animations/                    ← 4 GIFs: training dynamics and phase-space trajectories
```

## The method, step by step

**`src/classical/reference_solution.py`** — builds the *exact* ground truth via the method of characteristics: integrate Hamilton's equations backward in time from every query point, evaluate the known initial condition there. Verified to conserve mass to 7×10⁻¹⁵ (machine precision) before trusting any comparison against it.

**`src/classical/pinn_liouville.py`** — the failure. A network `ρ_θ(q,p,t)` trained against the Liouville equation's residual plus an initial-condition loss. Runs in under a minute; produces a density that integrates to 3.6× the correct total.

**`src/classical/hard_constraint_plus_data_v2.py`** — repair #1. Reparametrizes the output so it's forced to sum to exactly 1 by construction, plus sparse supervision from the exact solver. An 8× accuracy improvement, but required real engineering (gradient clipping, LR scheduling, best-checkpoint selection) to get there reliably — the naive version of this fix is documented failing first.

**`src/classical/hnn.py` + `propagate_with_hnn.py`** — repair #2. Learns the *dynamics*, not the density, then propagates via the same characteristics method as the ground truth. Conserves mass to 1.0000042 automatically, for a provable reason — see the proof in the [HNN explainer](docs/hnn_explainer.html).

**`src/classical/step3_final_comparison.py`** — the real test: the actual periodically-kicked chaotic system, not just the smooth sub-flow.

**`src/quantum/lindblad_pinn.py`** — the same residual-only philosophy, applied to a driven-dissipative qubit (a Lindblad master equation — the same mathematical object [IBM's quantum error-mitigation stack uses in production](https://arxiv.org/abs/2402.07617)). Used as an independent check on the diagnosis, not just a second demo.

## Configurations tested — deliberately, not just the best case

| # | Configuration | What it tests |
|---|---|---|
| 1 | Standard PINN (density regression) | Does direct residual training conserve mass? (No.) |
| 2 | + sine activation | Does the fix for a *different* known PINN problem (spectral bias) help here? (No — makes it worse.) |
| 3 | + more collocation points | Is it just an undersampling problem? (No — non-monotonic, doesn't help.) |
| 4 | + reweighted IC loss | Is it a loss-balancing problem? (No — one setting goes to *negative* mass.) |
| 5 | Hard-constraint + data supervision | Structural fix, numerically enforced | **Works** (8× improvement) |
| 6 | Hamiltonian Neural Network | Structural fix, provably guaranteed | **Works better** (11× improvement, automatic) |
| 7 | HNN under actual chaos (4 kick periods) | Does the fix survive the hard case? | Qualitatively yes, quantitatively erodes ~15× — as chaos theory predicts |
| 8 | HNN across 5 random seeds | Is the result reproducible? | Up to 38× spread — a real, previously unreported caveat |
| 9 | HNN on a coupled 2-oscillator network | Does it scale to more dimensions? | Partially — linear part learned well, nonlinear coupling term not |

## Results from the last full run

| Configuration | Mass (should be 1) | Pointwise MSE |
|---|---|---|
| Standard PINN | 3.6 | 6.1×10⁻⁵ |
| Hard-constraint + data | 1.0000 | 7.8×10⁻⁶ |
| **Hamiltonian Neural Network** | **1.0000042** | **5.8×10⁻⁶** |
| HNN, quantum qubit analog (γ-swept, 7 values) | trace error ≤ 0.65% across whole range | — |

Read honestly: the standard PINN's failure is not subtle (260% off). Both repairs are not "improvements" so much as a change of several orders of magnitude. The HNN's advantage is real but is not unconditional — it degrades under chaos and under added dimensionality, and we tested both directly rather than stopping at the best case.

## Quickstart

```bash
git clone https://github.com/<your-username>/sotpinhnn.git
cd sotpinhnn
pip install torch numpy matplotlib scipy --break-system-packages

# Reproduce the failure
python src/classical/reference_solution.py   # ground truth
python src/classical/pinn_liouville.py       # watch it fail

# Reproduce both fixes
python src/classical/hard_constraint_plus_data_v2.py
python src/classical/hnn.py

# The real test: actual chaos
python src/classical/step3_final_comparison.py

# Independent check: quantum analog
python src/quantum/lindblad_pinn.py
```

Each script is self-contained and prints its numerical results directly; no shared config files or hidden state between them.

## Purpose

This repo is for anyone using — or considering using — a physics-informed neural network on a transport/kinetic equation with a known conservation law (plasma kinetics, semiconductor charge transport, quantum noise characterization) who wants to know, concretely, what can go wrong and what a validated fix actually looks like, rather than taking either "PINNs work" or "PINNs don't work" on faith.

## Why this matters industrially (short version — full version in the explainer page)

- **Quantum computing (IBM):** quantum hardware noise is characterized in production using a Pauli-Lindblad model — the exact quantum generalization of the classical equation this repo studies.
- **Chip design (Intel, Samsung, and the semiconductor industry generally):** transistor-level charge transport is governed by the Boltzmann transport equation, the same mathematical family; Samsung has published hybrid deep-learning+TCAD work specifically because full-physics simulation is too slow for production, and PINN-based device modeling already exists in the research literature.
- **AI-for-science broadly (Google and others):** the core question here — does a network trained to accelerate a physics simulation actually respect the physics, or just the training signal — applies to any large-scale effort using neural networks as physics surrogates, not only to the system tested in this repo.

See [`docs/index.html`](docs/index.html) §1 for the full framing and citations.

## Novelty, stated carefully

Neither core architecture here is new. PINNs go back to Lagaris, Likas & Fotiadis (1998) and Raissi et al. (2019); Hamiltonian Neural Networks are Greydanus, Dzamba & Yosinski (2019); the hard-constraint idea mirrors an independent 2024 fix for the analogous quantum problem (Ullah et al.). What this repository contributes is the specific combination: a mechanistic (not just empirical) diagnosis of *why* the failure happens, independent confirmation of that mechanism on a different physical system, a mathematical proof for why the deeper fix works regardless of training quality, and an honest test of where that fix's advantage holds and where it erodes — reported together, with the failures included, not filtered out.

## Limitations (please read before using any of this on something real)

- **This is a low-dimensional toy problem** (2D classical phase space, one qubit, one small 4D network). The higher-dimensional regime that motivates interest in PINNs over grid-based methods in the first place is *not* yet tested.
- **The HNN's training data was generated from the true equations of motion.** In any real application you don't already know exactly, you'd need real observational trajectory data instead — untested here.
- **Single-run point estimates are not reliable.** The 5-seed robustness test showed up to 38× spread. Report a distribution, not one run, if you deploy this.
- **The chaos test only goes to 4 kick periods.** We do not know whether the HNN's advantage plateaus, keeps degrading, or eventually gets overtaken entirely at longer horizons.
- **This is a preprint-stage draft, not peer-reviewed work.** Real authorship, independent human expert review, and an AI-assistance disclosure are still outstanding — see the paper's own text for the full checklist.

---

*This repository was built with AI assistance (Claude) for code, analysis, and drafting. Every bug caught during that process — including a swapped gradient index, a scrambled figure order, and a mislabeled subsection reference — is disclosed in the paper's own text rather than silently fixed, in the interest of the same honesty this repo asks of the neural networks it studies.*
