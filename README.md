# Entanglement in (1+2) QED in Double-Layer Honeycomb Lattices — reproduction code

Numerical code that reproduces the figures of:

> F. Arreyes, F. Escudero, A. Gorza, and J. S. Ardenghi,
> *Entanglement in (1+2) QED in Double-Layer Honeycomb Lattices*,
> submitted to SciPost Physics (2026).

The script computes the momentum-resolved von Neumann entropy of a two-body
massive Dirac quasiparticle state in a double-layer honeycomb lattice
embedded in a planar electromagnetic cavity. The Bethe–Salpeter equation is
solved at Born (ladder) level with a single-photon-exchange kernel; the
reduced density matrix is obtained by tracing out one layer's sublattice
(pseudospin) degree of freedom.

## Requirements

- Python ≥ 3.10
- A CUDA-capable GPU is **strongly recommended** for the high-resolution
  scans (`figure_momentum`, `figure_selfenergy`, `figure_position2d`,
  `figure_angle` at paper resolution). The script falls back to CPU
  automatically, but those scans will take many hours on CPU.
- Dependencies are pinned in [`requirements.txt`](requirements.txt):
  `numpy`, `scipy`, `matplotlib`, `torch`, `tqdm`.

## Installation

```bash
git clone https://github.com/<user>/double-layer-qed-entanglement.git
cd double-layer-qed-entanglement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you have a CUDA installation, install the matching `torch` build from the
official selector at <https://pytorch.org/get-started/locally/> before
running `pip install -r requirements.txt`.

## Usage

All figures are reproduced through a single CLI:

```bash
python reproduce_figures.py --figure FIG_NAME
```

| `FIG_NAME`     | Output                                                                | Paper          |
|----------------|-----------------------------------------------------------------------|----------------|
| `selfenergy`   | Entropy vs (Re Σ₁, Re Σ₂) — heatmap, isolines, 1D cuts                | **Fig. 4**     |
| `momentum`     | Entropy vs (p₁, p₂) — 2D map and 1D cuts                              | **Fig. 6**     |
| `coherence`    | Entropy vs τ_coh / t_light                                            | **Fig. 5**     |
| `position2d`   | Entropy vs cavity positions (d₁, d₂) for several N_max                | supplementary  |
| `position1d`   | Entropy along d₂ = L − d₁ for several N_max                           | supplementary  |
| `angle`        | Entropy vs (φ₁, φ₂)                                                   | supplementary  |
| `all`          | Run every figure in sequence                                          | —              |

For a quick smoke test on modest hardware:

```bash
python reproduce_figures.py --figure selfenergy --quick
```

The `--quick` flag reduces resolution; results are noisier but generated in
seconds rather than hours. Output PNGs are written to `./figures/`.

## File layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── reproduce_figures.py     # all physics + plotting, with CLI
└── figures/                 # PNG output (created on first run)
```

The script is a single file of ~900 lines organized as:

1. Physical constants and globally-mutable controls (`N_MAX`, `GLOBAL_SIGMA_*`).
2. The Bethe–Salpeter kernel integrand, vectorized across the four
   interaction channels (ee, eh, he, hh) and integrated by Gauss–Legendre
   quadrature.
3. The free state Ψ⁽⁰⁾ and the first-order correction Ψ⁽¹⁾ obtained from a
   batched linear solve of the 4×4 inverse propagator.
4. The reduced density matrix and the conditional pseudospin von Neumann
   entropy, including a Born-validity ratio ‖Ψ⁽¹⁾‖ / ‖Ψ⁽⁰⁾‖ that is plotted
   as a red contour wherever it exceeds unity.
5. Six figure-producing functions, each callable directly with custom
   parameters or via the CLI.

## Reproducing the published figures

The default arguments of each `figure_*` function match the parameters
reported in the paper figure captions. To regenerate a paper figure
exactly:

```bash
python reproduce_figures.py --figure selfenergy
```

Approximate single-GPU wall times (NVIDIA A100 / V100 class):
- `coherence` (1D scan, 1000 points): ~1 min
- `position1d` (4 × 1000 points): ~3 min
- `selfenergy` (100 × 100): ~5 min
- `momentum` (200 × 200): ~15 min
- `angle` (200 × 200): ~15 min
- `position2d` (4 × 240 × 240): ~45 min

## Method summary

For a chosen set of external momenta (p₁, p₂), angles (φ₁, φ₂), and cavity
positions (d₁, d₂):

1. Construct the on-shell q₀ from energy–momentum conservation across all
   four interaction channels (ee, eh, he, hh).
2. Evaluate the angular integral of the cavity-photon kernel
   K(q; d₁, d₂) by Gauss–Legendre quadrature.
3. Solve G₍A₎⁻¹ ⊗ G₍B₎⁻¹ · Ψ⁽¹⁾ = I for the four-component first-order
   correction. The phenomenological self-energy (Re Σ + i Im Σ = Re Σ + iΓ)
   enters as a mass shift in each layer's inverse propagator.
4. Form the normalized total state Ψ = Ψ⁽⁰⁾ + ½ Σ_channels Ψ⁽¹⁾, trace over
   layer B's pseudospin to obtain ρ_A, and compute S(ρ_A) = −Tr(ρ_A log ρ_A).
5. Report the Born-validity ratio ‖Ψ⁽¹⁾‖ / ‖Ψ⁽⁰⁾‖ alongside the entropy so
   the perturbative regime can be told apart unambiguously from the
   non-perturbative one (shown as a red contour in the published figures).

This is a strict first-iteration (Born-level) implementation. A full
non-perturbative solution of the homogeneous Bethe–Salpeter equation is
the subject of follow-up work.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Arreyes2026EntanglementQED,
  author  = {Arreyes, Facundo and Escudero, Federico and Gorza, Ari{\'a}n and
             Ardenghi, Juan Sebasti{\'a}n},
  title   = {{Entanglement in (1+2) QED in Double-Layer Honeycomb Lattices}},
  journal = {SciPost Physics},
  year    = {2026},
  note    = {submitted}
}
```

A machine-readable citation is also provided in [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
