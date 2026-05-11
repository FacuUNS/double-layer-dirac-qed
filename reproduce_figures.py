#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduction code for the figures of:

    F. Arreyes, F. Escudero, A. Gorza, and J. S. Ardenghi,
    "Entanglement in (1+2) QED in Double-Layer Honeycomb Lattices",
    submitted to SciPost Physics (2026).

The code evaluates the momentum-resolved von Neumann entropy of a two-body
massive Dirac quasiparticle state in a double-layer honeycomb lattice
embedded in a planar electromagnetic cavity, using the Bethe-Salpeter
equation at Born (ladder) level with a single-photon-exchange kernel.

USAGE
-----
    python reproduce_figures.py --figure FIG_NAME

Available figure names:
    selfenergy   Entropy vs (Re Sigma_1, Re Sigma_2)        [paper Fig. 4]
    momentum     Entropy vs (p_1, p_2) at fixed self-energy [paper Fig. 6]
    coherence    Entropy vs coherence time tau_coh/t_light  [paper Fig. 5]
    position2d   Entropy vs cavity positions (d_1, d_2)
                 for several N_max
    position1d   Entropy along d_2 = L - d_1 for several N_max
    angle        Entropy vs angular variables (phi_1, phi_2)
    all          Run all of the above

Output PNGs are written to ./figures/

A CUDA-capable GPU is strongly recommended for the higher-resolution scans;
the script falls back to CPU but the larger scans will be slow.

UNITS
-----
All quantities in natural units: hbar = 1, with Fermi velocity v_F playing
the role of the speed of light in the in-plane Dirac equation.
Lengths in eV^-1, energies in eV.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.interpolate import griddata
import torch
from tqdm.auto import tqdm

# Memory tuning for large parameter scans on CUDA
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ============================================================================
# Device, dtype, physical parameters
# ============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU (GPU strongly recommended for high-resolution scans).")

torch.set_default_dtype(torch.float64)

# Physical constants (natural units)
HBAR       = torch.tensor(1.0,     device=DEVICE)
V_FA       = torch.tensor(1.833e-3, device=DEVICE)  # Fermi velocity, layer A
V_FB       = torch.tensor(1.833e-3, device=DEVICE)  # Fermi velocity, layer B
L_CAVITY   = torch.tensor(2.0,     device=DEVICE)   # Cavity length, eV^-1
LAMBDA_SO  = torch.tensor(3.9e-3,  device=DEVICE)   # Spin-orbit gap, eV
EPSILON    = torch.tensor(1e-6,    device=DEVICE)   # Pole regulator i0+
II         = torch.tensor(1j,      device=DEVICE)
PI         = torch.tensor(math.pi, device=DEVICE)
E_CHARGE   = torch.tensor(0.30282, device=DEVICE)   # Electron charge (Gauss units)
PERM       = torch.tensor(1.0,     device=DEVICE)   # Cavity permittivity

# Derived
DELTA_A = LAMBDA_SO / V_FA                          # Mass scale, layer A
DELTA_B = LAMBDA_SO / V_FB                          # Mass scale, layer B
ZETA    = torch.sqrt(HBAR / L_CAVITY * PERM)        # Mode-amplitude prefactor

# Globally mutable controls (set by figure functions via `global`)
N_MAX = 1
GLOBAL_SIGMA_A = torch.tensor(1.3e-7 + 1e-8j, dtype=torch.complex128, device=DEVICE)
GLOBAL_SIGMA_B = torch.tensor(1.3e-7 + 1e-8j, dtype=torch.complex128, device=DEVICE)

# Sign tables for the four interaction channels: ee, eh, he, hh.
#   sign_q0: +1 in the q_0 numerator/denominator for identical-charge pairs
#            (ee, hh), -1 for opposite-charge (eh, he)
#   sign_a:  +1 if layer A holds an electron, -1 if a hole
#   sign_b:  +1 if layer B holds an electron, -1 if a hole
SIGN_Q0 = torch.tensor([ 1., -1., -1.,  1.], device=DEVICE)
SIGN_A  = torch.tensor([ 1.,  1., -1., -1.], device=DEVICE)
SIGN_B  = torch.tensor([ 1., -1.,  1., -1.], device=DEVICE)
CHANNEL_NAMES = ("ee", "eh", "he", "hh")


# ============================================================================
# Core physics: single-photon-exchange kernel integrand
# ============================================================================

def _integrand_all_channels(
    theta_nodes: torch.Tensor,
    theta_weights: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate and integrate the angular integrand of the Bethe-Salpeter
    kernel (paper Eq. 27) for all four interaction channels simultaneously.

    For each channel and each parameter set in the batch, the function:
      1. Computes the on-shell q_0 from energy-momentum conservation.
      2. Builds the shifted spinor amplitudes (chi_a, chi_b) and their phases.
      3. Sums over cavity-mode index n from 1 to N_MAX.
      4. Forms the four-component spinor integrand and integrates over theta
         by Gauss-Legendre quadrature on [0, 2*pi].

    Parameters
    ----------
    theta_nodes : (N_pts, 1) tensor
        Gauss-Legendre quadrature nodes mapped to [0, 2*pi].
    theta_weights : (N_pts,) tensor
        Gauss-Legendre quadrature weights (Jacobian-corrected).
    params : (B, 6) tensor
        Per-batch parameters: (p_1, p_2, phi_1, phi_2, d_1, d_2).

    Returns
    -------
    (B, 4_channels, 4_components) complex tensor
        The integrated kernel I = (I_1, I_2, I_3, I_4) for each channel.
    """
    N_pts = theta_nodes.shape[0]
    B = params.shape[0]
    eps = 1e-12

    # Kinematic parameters, broadcast to [1, B, 1] so the trailing dim is the
    # channel axis (4 channels) and middle dim is the batch.
    pa   = params[:, 0].view(1, B, 1)
    pb   = params[:, 1].view(1, B, 1)
    phi1 = params[:, 2].view(1, B, 1)
    phi2 = params[:, 3].view(1, B, 1)
    d1   = params[:, 4].view(1, B, 1)
    d2   = params[:, 5].view(1, B, 1)

    theta = theta_nodes.view(N_pts, 1, 1)

    # Free Dirac energies
    pa0 = torch.sqrt(pa**2 + DELTA_A**2)
    pb0 = torch.sqrt(pb**2 + DELTA_B**2)

    # Channel-dependent signs, shape [1, 1, 4]
    sgn_q0 = SIGN_Q0.view(1, 1, 4)
    sgn_a  = SIGN_A.view(1, 1, 4)
    sgn_b  = SIGN_B.view(1, 1, 4)

    cos_p1_t = torch.cos(phi1 - theta)
    cos_p2_t = torch.cos(phi2 - theta)

    # On-shell q_0 from the delta function of energy-momentum conservation
    num = 2 * sgn_q0 * (pa0 + sgn_q0 * pb0) * (
        pb0 * pa * cos_p1_t - sgn_q0 * pa0 * pb * cos_p2_t
    )
    den = (pa0 + pb0) ** 2 - (pa * cos_p1_t + pb * cos_p2_t) ** 2
    q0 = num / (den + eps)

    # Energies of the shifted quasiparticles
    pa_q_E = torch.sqrt(pa0**2 + q0**2 - 2 * pa * q0 * cos_p1_t)
    pb_q_E = torch.sqrt(pb0**2 + q0**2 + 2 * pb * q0 * cos_p2_t)

    # f'(q_0) for the change-of-variables Jacobian
    f_prime = -sgn_q0 * (
        (q0 - pa * cos_p1_t) / pa_q_E + (q0 + pb * cos_p2_t) / pb_q_E
    )

    # Shifted momenta and their phases
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    cos_p1, sin_p1 = torch.cos(phi1), torch.sin(phi1)
    cos_p2, sin_p2 = torch.cos(phi2), torch.sin(phi2)

    pa_q_x = pa * cos_p1 - q0 * cos_t
    pa_q_y = pa * sin_p1 - q0 * sin_t
    pb_q_x = pb * cos_p2 + q0 * cos_t
    pb_q_y = pb * sin_p2 + q0 * sin_t

    phi_pa_q = torch.atan2(pa_q_y, pa_q_x)
    phi_pb_q = torch.atan2(pb_q_y, pb_q_x)

    pa_q = torch.sqrt(pa_q_x**2 + pa_q_y**2 + eps)
    pb_q = torch.sqrt(pb_q_x**2 + pb_q_y**2 + eps)

    # Spinor amplitudes chi for electrons (+) or holes (-)
    chi_a = pa_q / (sgn_a * pa_q_E + DELTA_A + eps)
    chi_b = pb_q / (sgn_b * pb_q_E + DELTA_B + eps)

    # Sum over cavity modes n = 1, ..., N_MAX
    n_terms = torch.arange(1, N_MAX + 1, device=DEVICE).view(N_MAX, 1, 1, 1)
    d1_n = d1.view(1, 1, B, 1)
    d2_n = d2.view(1, 1, B, 1)
    sin1 = torch.sin(n_terms * PI * d1_n / L_CAVITY)
    sin2 = torch.sin(n_terms * PI * d2_n / L_CAVITY)
    zeta_n_sq = ZETA**2 * sin1 * sin2

    # Denominator: (E_term)^2 - q_0^2 - omega_n^2 + i0+
    energy_diff = pa0 - pa_q_E
    energy_q_sq = (energy_diff**2 - q0**2).unsqueeze(0).expand(N_MAX, N_pts, B, 4)
    omega_n_sq = (n_terms * PI / L_CAVITY) ** 2

    denom = energy_q_sq - omega_n_sq + II * EPSILON
    mode_sum = torch.sum(zeta_n_sq / denom, dim=0)  # [N_pts, B, 4]

    # Four spinor components stacked along the last axis
    norm_inv = 1.0 / (
        torch.sqrt(1 + chi_a**2 + eps) * torch.sqrt(1 + chi_b**2 + eps)
    )
    comp = torch.stack(
        [
            -II * chi_a * chi_b * torch.exp(II * (phi_pa_q + phi_pb_q)),
            -torch.exp(II * phi_pa_q) * chi_a,
            -torch.exp(II * phi_pb_q) * chi_b,
            II * torch.ones_like(chi_a),
        ],
        dim=-1,
    )
    comp = comp * norm_inv.unsqueeze(-1)

    prefactor = (E_CHARGE / (2 * PI)) ** 2
    integrand = (
        q0.unsqueeze(-1)
        * mode_sum.unsqueeze(-1)
        * comp
        * prefactor
        / (f_prime.unsqueeze(-1) + eps)
    )  # [N_pts, B, 4_channels, 4_components]

    # Gauss-Legendre quadrature
    w = theta_weights.view(N_pts, 1, 1, 1)
    return torch.sum(integrand * w, dim=0)  # [B, 4_channels, 4_components]


def compute_kernel_integrals(
    params: torch.Tensor, n_quadrature: int = 128
) -> torch.Tensor:
    """
    Wrap _integrand_all_channels with Gauss-Legendre quadrature setup.

    Returns
    -------
    (B, 4_channels, 4_components) complex tensor.
    """
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quadrature)
    # Map [-1, 1] -> [0, 2*pi]
    theta_nodes = PI * (torch.tensor(nodes_np, device=DEVICE) + 1)
    theta_weights = torch.tensor(weights_np, device=DEVICE) * PI
    theta_nodes = theta_nodes.view(-1, 1)
    return _integrand_all_channels(theta_nodes, theta_weights, params)


# ============================================================================
# Unperturbed two-body state Psi^(0)
# ============================================================================

def unperturbed_state_all_channels(
    pa: torch.Tensor, pb: torch.Tensor,
    phi1: torch.Tensor, phi2: torch.Tensor,
) -> torch.Tensor:
    """
    Free two-body Dirac state Psi^(0) for all four channels (ee, eh, he, hh).

    Returns
    -------
    (4_channels, 4_components, B) complex tensor.
    """
    eps = 1e-12
    B = pa.shape[0]

    sgn_a = SIGN_A.view(4, 1)
    sgn_b = SIGN_B.view(4, 1)

    pa_e   = pa.unsqueeze(0).expand(4, B)
    pb_e   = pb.unsqueeze(0).expand(4, B)
    phi1_e = phi1.unsqueeze(0).expand(4, B)
    phi2_e = phi2.unsqueeze(0).expand(4, B)

    chi_a = pa_e / (sgn_a * torch.sqrt(pa_e**2 + DELTA_A**2) + DELTA_A + eps)
    chi_b = pb_e / (sgn_b * torch.sqrt(pb_e**2 + DELTA_B**2) + DELTA_B + eps)

    norm_a = torch.sqrt(1 + chi_a**2 + eps)
    norm_b = torch.sqrt(1 + chi_b**2 + eps)

    u_a1 = 1.0 / norm_a
    u_a2 = chi_a * torch.exp(II * phi1_e) / norm_a
    u_b1 = 1.0 / norm_b
    u_b2 = chi_b * torch.exp(II * phi2_e) / norm_b

    # Kronecker product on the 2x2 spinor structure
    psi0 = torch.stack(
        [u_a1 * u_b1, u_a1 * u_b2, u_a2 * u_b1, u_a2 * u_b2],
        dim=1,
    )  # [4_channels, 4_components, B]
    return psi0


# ============================================================================
# First-order correction Psi^(1) via Bethe-Salpeter at Born level
# ============================================================================

def solve_first_order_correction(
    pa: torch.Tensor, pb: torch.Tensor,
    phi1: torch.Tensor, phi2: torch.Tensor,
    sigma_a: torch.Tensor, sigma_b: torch.Tensor,
    I_kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Solve G_total^{-1} . Psi^(1) = I  for all four channels in one batched solve.

    Builds the inverse propagator as a Kronecker product G_A^{-1} (x) G_B^{-1}
    of the two single-layer Dirac inverse propagators, including the
    phenomenological self-energies sigma_a, sigma_b (may be complex).

    Parameters
    ----------
    pa, pb, phi1, phi2 : (B,) real tensors
    sigma_a, sigma_b : complex scalar or (B,) tensor
    I_kernel : (4_channels, B, 4_components) complex tensor from
               compute_kernel_integrals(...)

    Returns
    -------
    (4_channels, B, 4_components) complex tensor.
    """
    B = pa.shape[0]

    sgn_a = SIGN_A.view(4, 1)
    sgn_b = SIGN_B.view(4, 1)

    pa_e   = pa.unsqueeze(0).expand(4, B)
    pb_e   = pb.unsqueeze(0).expand(4, B)
    phi1_e = phi1.unsqueeze(0).expand(4, B)
    phi2_e = phi2.unsqueeze(0).expand(4, B)

    # On-shell energies with channel-dependent sign (hole -> negative energy)
    p0_a = sgn_a * torch.sqrt(pa_e**2 + DELTA_A**2)
    p0_b = sgn_b * torch.sqrt(pb_e**2 + DELTA_B**2)

    m_eff_a = DELTA_A + sigma_a
    m_eff_b = DELTA_B + sigma_b

    exp_m1 = torch.exp(-II * phi1_e)
    exp_p1 = torch.exp( II * phi1_e)
    exp_m2 = torch.exp(-II * phi2_e)
    exp_p2 = torch.exp( II * phi2_e)

    G_A = torch.zeros((4, B, 2, 2), dtype=torch.complex128, device=DEVICE)
    G_A[:, :, 0, 0] =  p0_a - m_eff_a
    G_A[:, :, 0, 1] = -pa_e * exp_m1
    G_A[:, :, 1, 0] =  pa_e * exp_p1
    G_A[:, :, 1, 1] = -p0_a - m_eff_a

    G_B = torch.zeros((4, B, 2, 2), dtype=torch.complex128, device=DEVICE)
    G_B[:, :, 0, 0] =  p0_b - m_eff_b
    G_B[:, :, 0, 1] = -pb_e * exp_m2
    G_B[:, :, 1, 0] =  pb_e * exp_p2
    G_B[:, :, 1, 1] = -p0_b - m_eff_b

    # Batched Kronecker product -> [4_channels, B, 4, 4]
    G_total = torch.einsum("cbij,cbkl->cbikjl", G_A, G_B).reshape(4, B, 4, 4)

    # Stack channels and batch into one big solve
    G_flat = G_total.reshape(4 * B, 4, 4)
    I_flat = I_kernel.reshape(4 * B, 4)

    psi1_flat = torch.linalg.solve(G_flat, I_flat)
    return psi1_flat.reshape(4, B, 4)


# ============================================================================
# Entropy from reduced density matrix (Born level)
# ============================================================================

def entropy_partial_trace(
    params: torch.Tensor, n_quadrature: int = 96
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the conditional pseudospin von Neumann entropy at Born level.

    Workflow:
      1. Build I_kernel = compute_kernel_integrals(params) for all 4 channels.
      2. Build Psi^(0) = unperturbed_state_all_channels(...).
      3. Solve for Psi^(1) at first order in the cavity-mediated kernel.
      4. Total normalized state: Psi = Psi^(0) + 0.5 * sum_channels Psi^(1).
      5. Reduce over layer-B (pseudospin) to get rho_A (2x2 per batch entry).
      6. Compute S(rho_A) = -tr(rho_A log rho_A) via eigvalsh.

    Returns
    -------
    entropies : (B,) real tensor
    validity_ratio : (B,) real tensor
        Ratio ||Psi^(1)|| / ||Psi^(0)||. Values >= 1 indicate the Born
        approximation has broken down and the entropy in that region should
        not be interpreted as a physical prediction.
    """
    B = params.shape[0]
    p1, p2, phi1, phi2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

    # 1) Kernel integrals
    I_kernel = compute_kernel_integrals(params, n_quadrature=n_quadrature)

    # 2) Psi^(0) for all four channels, then weighted sum (peso = 1/2 per channel)
    psi0_all = unperturbed_state_all_channels(p1, p2, phi1, phi2)
    half = torch.tensor(0.5 + 0j, dtype=torch.complex128, device=DEVICE)
    psi0_total = half * psi0_all.permute(2, 1, 0).sum(dim=-1)   # [B, 4]

    # 3) Psi^(1) at Born level
    psi1_all = solve_first_order_correction(
        p1, p2, phi1, phi2, GLOBAL_SIGMA_A, GLOBAL_SIGMA_B, I_kernel
    )
    psi1_total = half * psi1_all.sum(dim=0)                     # [B, 4]

    # Born validity diagnostic
    norm_psi0 = torch.sqrt(torch.sum(torch.abs(psi0_total) ** 2, dim=1))
    norm_psi1 = torch.sqrt(torch.sum(torch.abs(psi1_total) ** 2, dim=1))
    validity_ratio = (norm_psi1 / (norm_psi0 + 1e-14)).real

    # 4) Total normalized state
    psi_tot = psi0_total + psi1_total
    norms = torch.clamp(
        torch.sqrt(torch.sum(torch.abs(psi_tot) ** 2, dim=1, keepdim=True)),
        min=1e-14,
    )
    psi_tot = psi_tot / norms

    c00, c01, c10, c11 = psi_tot[:, 0], psi_tot[:, 1], psi_tot[:, 2], psi_tot[:, 3]

    # 5) Reduced density matrix on layer A
    rho_A = torch.zeros((B, 2, 2), dtype=torch.complex128, device=DEVICE)
    rho_A[:, 0, 0] = c00 * torch.conj(c00) + c01 * torch.conj(c01)
    rho_A[:, 1, 1] = c10 * torch.conj(c10) + c11 * torch.conj(c11)
    rho_A[:, 0, 1] = c00 * torch.conj(c10) + c01 * torch.conj(c11)
    rho_A[:, 1, 0] = c10 * torch.conj(c00) + c11 * torch.conj(c01)

    # Hermitianize before diagonalizing
    rho_sym = 0.5 * (rho_A + torch.conj(rho_A.transpose(-1, -2)))

    try:
        eigvals = torch.linalg.eigvalsh(rho_sym).real
    except Exception:
        eigvals = _eigvalsh_fallback(rho_sym, B)

    eigvals = torch.clamp(eigvals, min=0.0)
    eig_sum = eigvals.sum(dim=1, keepdim=True)
    safe_sum = torch.clamp(eig_sum, min=1e-12)
    eigvals_n = eigvals / safe_sum

    log_ev = torch.log(torch.clamp(eigvals_n, min=1e-14))
    entropies = -torch.sum(eigvals_n * log_ev, dim=1).real
    entropies = torch.where(
        eig_sum.squeeze(1) > 1e-12, entropies, torch.zeros_like(entropies)
    )

    return entropies, validity_ratio


def _eigvalsh_fallback(rho_sym: torch.Tensor, B: int) -> torch.Tensor:
    """Per-sample eigvalsh fallback for batches containing NaN entries."""
    eigvals = torch.zeros((B, 2), dtype=torch.float64, device=DEVICE)
    for i in range(B):
        try:
            eigvals[i] = torch.linalg.eigvalsh(rho_sym[i]).real
        except Exception:
            eigvals[i] = float("nan")
    return eigvals


def batch_entropy(
    params: torch.Tensor, batch_size: int = 100, n_quadrature: int = 96
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run entropy_partial_trace over a long parameter array in chunks of size
    batch_size, with automatic halving on CUDA OOM.
    """
    ent_chunks, val_chunks = [], []
    total = params.shape[0]
    for i in tqdm(range(0, total, batch_size),
                  desc="Entropy scan", unit="batch"):
        chunk = params[i : i + batch_size]
        try:
            e, v = entropy_partial_trace(chunk, n_quadrature=n_quadrature)
            ent_chunks.append(e.cpu())
            val_chunks.append(v.cpu())
        except RuntimeError as exc:
            if "out of memory" in str(exc):
                torch.cuda.empty_cache()
                return batch_entropy(
                    params, batch_size=max(batch_size // 2, 1),
                    n_quadrature=n_quadrature,
                )
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(ent_chunks), torch.cat(val_chunks)


# ============================================================================
# Figure: entropy vs (Re Sigma_1, Re Sigma_2)            -- paper Fig. 4
# ============================================================================

def figure_selfenergy(
    p1: float = 0.13, p2: float = 0.12,
    phi1: float = 0.0, phi2: float = 0.0,
    d1: float = 0.9, d2: float = 1.1,
    sigma_range: tuple[float, float] = (3.5e-3, 7e-3),
    gamma: float = 1e-6,
    n_points: int = 100,
    n_quadrature: int = 128,
    log_scale: bool = True,
    outfile: str = "fig_selfenergy.png",
):
    """
    Entanglement entropy as a function of (Re Sigma_1, Re Sigma_2) at fixed
    quasiparticle momenta, angles and cavity positions. Reproduces paper
    Fig. 4 with the default parameters.
    """
    print("\n" + "=" * 70)
    print("Figure: entropy vs (Re Sigma_1, Re Sigma_2)")
    print("=" * 70)
    print(f"  p_1 = {p1}, p_2 = {p2}, d_1 = {d1}, d_2 = {d2}")
    print(f"  Im(Sigma) = {gamma:.2e},  range = {sigma_range}")

    global GLOBAL_SIGMA_A, GLOBAL_SIGMA_B
    orig_a, orig_b = GLOBAL_SIGMA_A.clone(), GLOBAL_SIGMA_B.clone()

    if log_scale:
        s_vals = torch.logspace(
            math.log10(sigma_range[0]), math.log10(sigma_range[1]),
            n_points, device=DEVICE,
        )
    else:
        s_vals = torch.linspace(sigma_range[0], sigma_range[1], n_points, device=DEVICE)

    SA, SB = torch.meshgrid(s_vals, s_vals, indexing="ij")

    params = torch.zeros((n_points * n_points, 6), device=DEVICE)
    params[:, 0], params[:, 1] = p1, p2
    params[:, 2], params[:, 3] = phi1, phi2
    params[:, 4], params[:, 5] = d1, d2

    sa_flat = SA.reshape(-1)
    sb_flat = SB.reshape(-1)

    ent = torch.zeros(n_points * n_points, dtype=torch.float64, device=DEVICE)
    val = torch.zeros(n_points * n_points, dtype=torch.float64, device=DEVICE)
    batch = 100
    for i in tqdm(range(0, n_points * n_points, batch), desc="Sigma scan"):
        j = min(i + batch, n_points * n_points)
        GLOBAL_SIGMA_A = sa_flat[i:j] + 1j * gamma
        GLOBAL_SIGMA_B = sb_flat[i:j] + 1j * gamma
        try:
            e, v = entropy_partial_trace(params[i:j], n_quadrature=n_quadrature)
            ent[i:j] = e
            val[i:j] = v
        except Exception as exc:
            print(f"Batch {i}-{j} failed: {exc}")
            ent[i:j] = float("nan")
            val[i:j] = float("nan")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    GLOBAL_SIGMA_A, GLOBAL_SIGMA_B = orig_a, orig_b

    ent_grid = ent.reshape(n_points, n_points).cpu().numpy()
    val_grid = val.reshape(n_points, n_points).cpu().numpy()
    SA_np, SB_np = SA.cpu().numpy(), SB.cpu().numpy()
    s_diag = np.linspace(sigma_range[0], sigma_range[1], 100)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    levels = 30

    # Panel 1: heatmap
    cf = axes[0].contourf(SA_np, SB_np, ent_grid, levels=levels, cmap="viridis")
    if log_scale:
        axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].contour(SA_np, SB_np, ent_grid, levels=levels,
                    colors="white", linewidths=0.5, alpha=0.3)
    if np.nanmax(val_grid) >= 1.0:
        axes[0].contour(SA_np, SB_np, val_grid, levels=[1.0],
                        colors="red", linewidths=3)
        axes[0].plot([], [], color="red", linewidth=3,
                     label=r"Born Breakdown ($||\Psi^{(1)}|| \geq ||\Psi^{(0)}||$)")
    axes[0].plot(s_diag, s_diag, "w--", linewidth=2.5,
                 label=r"$\Sigma_1 = \Sigma_2$", alpha=0.8)
    axes[0].set_xlabel(r"$\mathrm{Re}(\Sigma_1)$ [eV]", fontsize=18)
    axes[0].set_ylabel(r"$\mathrm{Re}(\Sigma_2)$ [eV]", fontsize=18)
    axes[0].set_title("Entanglement Entropy Map", fontsize=18)
    axes[0].tick_params(axis="both", labelsize=15)
    axes[0].legend(loc="upper left", fontsize=13)
    cb1 = plt.colorbar(cf, ax=axes[0])
    cb1.set_label("Entropy $S_1$", fontsize=18)
    cb1.ax.tick_params(labelsize=15)

    # Panel 2: isolines
    cs = axes[1].contour(SA_np, SB_np, ent_grid, levels=levels,
                         cmap="plasma", linewidths=1.5)
    if log_scale:
        axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].clabel(cs, inline=True, fontsize=11, fmt="%.3f")
    if np.nanmax(val_grid) >= 1.0:
        axes[1].contour(SA_np, SB_np, val_grid, levels=[1.0],
                        colors="red", linewidths=3)
    axes[1].set_xlabel(r"$\mathrm{Re}(\Sigma_1)$ [eV]", fontsize=18)
    axes[1].set_ylabel(r"$\mathrm{Re}(\Sigma_2)$ [eV]", fontsize=18)
    axes[1].set_title("Entropy Isolines", fontsize=18)
    axes[1].tick_params(axis="both", labelsize=15)

    # Panel 3: 1D cuts
    ent_diag = ent_grid[np.arange(n_points), np.arange(n_points)]
    mid = n_points // 2
    ent_sigB_fixed = ent_grid[:, mid]
    s_np = s_vals.cpu().numpy()
    plotter = axes[2].semilogx if log_scale else axes[2].plot
    plotter(s_np, ent_diag, "-", linewidth=2.5, color="#2E86AB",
            label=r"$\Sigma_1 = \Sigma_2$ (diagonal)")
    plotter(s_np, ent_sigB_fixed, "-", linewidth=2.5, color="#E63946",
            label=rf"$\Sigma_2 = {s_vals[mid].cpu().numpy():.1e}$")
    axes[2].axhline(y=np.log(2), color="gray", linestyle="--", linewidth=2,
                    label=rf"Max ($\ln 2 \approx {np.log(2):.3f}$)", alpha=0.8)
    axes[2].set_xlabel(r"$\mathrm{Re}(\Sigma)$ [eV]", fontsize=18)
    axes[2].set_ylabel("Entanglement Entropy $S_1$", fontsize=18)
    axes[2].set_title("1D Cuts", fontsize=18)
    axes[2].tick_params(axis="both", labelsize=15)
    axes[2].legend(fontsize=13)

    plt.tight_layout()
    path = FIG_DIR / outfile
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ============================================================================
# Figure: entropy vs (p_1, p_2)                            -- paper Fig. 6
# ============================================================================

def figure_momentum(
    d1: float = 0.9, d2: float = 1.1,
    phi1: float = 0.0, phi2: float = 0.0,
    sigma_real: float = 4.2e-3, gamma: float = 1e-6,
    p_min: float = 0.05, p_max: float = 0.15,
    n_points: int = 200,
    n_quadrature: int = 128,
    outfile_2d: str = "fig_momentum_2D.png",
    outfile_1d: str = "fig_momentum_1D.png",
):
    """
    Entanglement entropy as a function of (p_1, p_2) at fixed self-energy,
    angles and cavity positions. Reproduces paper Fig. 6 with the defaults.
    """
    print("\n" + "=" * 60)
    print("Figure: entropy vs (p_1, p_2)")
    print("=" * 60)
    print(f"  d_1 = {d1}, d_2 = {d2}")
    print(f"  Re(Sigma) = {sigma_real:.2e}, Im(Sigma) = {gamma:.2e}")

    global GLOBAL_SIGMA_A, GLOBAL_SIGMA_B
    orig_a, orig_b = GLOBAL_SIGMA_A.clone(), GLOBAL_SIGMA_B.clone()
    GLOBAL_SIGMA_A = torch.tensor(sigma_real + 1j * gamma, dtype=torch.complex128, device=DEVICE)
    GLOBAL_SIGMA_B = torch.tensor(sigma_real + 1j * gamma, dtype=torch.complex128, device=DEVICE)

    p1_vals = torch.linspace(p_min, p_max, n_points, device=DEVICE)
    p2_vals = torch.linspace(p_min, p_max, n_points, device=DEVICE)
    P1, P2 = torch.meshgrid(p1_vals, p2_vals, indexing="ij")

    params = torch.zeros((n_points * n_points, 6), device=DEVICE)
    params[:, 0], params[:, 1] = P1.reshape(-1), P2.reshape(-1)
    params[:, 2], params[:, 3] = phi1, phi2
    params[:, 4], params[:, 5] = d1, d2

    ent, val = batch_entropy(params, n_quadrature=n_quadrature)
    GLOBAL_SIGMA_A, GLOBAL_SIGMA_B = orig_a, orig_b

    ent_grid = ent.reshape(n_points, n_points).numpy()
    val_grid = val.reshape(n_points, n_points).numpy()
    P1_np, P2_np = P1.cpu().numpy(), P2.cpu().numpy()

    # Interpolate any NaN cells from neighbours
    valid = np.isfinite(ent_grid)
    if 4 < valid.sum() < ent_grid.size:
        pts = np.column_stack((P1_np[valid], P2_np[valid]))
        grid = np.column_stack((P1_np.ravel(), P2_np.ravel()))
        ent_grid = np.where(
            valid, ent_grid,
            griddata(pts, ent_grid[valid], grid, method="linear").reshape(n_points, n_points),
        )
        val_grid = np.where(
            valid, val_grid,
            griddata(pts, val_grid[valid], grid, method="linear").reshape(n_points, n_points),
        )

    # 2D heatmap
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    cf = ax1.contourf(P1_np, P2_np, ent_grid, levels=25, cmap="viridis")
    ax1.contour(P1_np, P2_np, ent_grid, levels=25,
                colors="white", linewidths=0.5, alpha=0.3)
    if val_grid.max() >= 1.0:
        ax1.contour(P1_np, P2_np, val_grid, levels=[1.0],
                    colors="red", linewidths=3.5)
        ax1.plot([], [], color="red", linewidth=3.5,
                 label=r"Born Breakdown ($||\Psi^{(1)}|| \geq ||\Psi^{(0)}||$)")
    mid_p = 0.5 * (p_min + p_max)
    ax1.axhline(y=mid_p, color="white", linestyle="--", linewidth=1.5,
                alpha=0.6, label=f"$p_2 = {mid_p:.3f}$")
    ax1.axvline(x=mid_p, color="orange", linestyle="--", linewidth=1.5,
                alpha=0.6, label=f"$p_1 = {mid_p:.3f}$")
    ax1.set_xlabel("Momentum $p_1$ [eV]", fontsize=18)
    ax1.set_ylabel("Momentum $p_2$ [eV]", fontsize=18)
    ax1.tick_params(axis="both", labelsize=15)
    ax1.legend(loc="upper left", fontsize=13, framealpha=0.9)
    cb = plt.colorbar(cf, ax=ax1)
    cb.set_label("Entropy $S_A$", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    ax1.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    path1 = FIG_DIR / outfile_2d
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    print(f"  Saved {path1}")
    plt.close(fig1)

    # 1D cuts
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    idx_mid = np.clip(int(round((mid_p - p_min) / max((p_max - p_min), 1e-12) * (n_points - 1))),
                      0, n_points - 1)
    p_np = p1_vals.cpu().numpy()
    ent_p2_fixed = ent_grid[:, idx_mid]
    val_p2_fixed = val_grid[:, idx_mid]
    ent_p1_fixed = ent_grid[idx_mid, :]
    val_p1_fixed = val_grid[idx_mid, :]
    ax2.plot(p_np, ent_p2_fixed, "-", linewidth=2.5, color="#2E86AB",
             label=f"$p_2 = {mid_p:.3f}$ fixed")
    ax2.plot(p_np, ent_p1_fixed, "--", linewidth=2.5, color="#E63946",
             label=f"$p_1 = {mid_p:.3f}$ fixed")
    red_label_done = False
    for cut_vals, cut_ent in ((val_p2_fixed, ent_p2_fixed),
                              (val_p1_fixed, ent_p1_fixed)):
        broken = cut_vals >= 1.0
        if broken.any():
            ent_b = cut_ent.copy()
            ent_b[~broken] = np.nan
            ax2.plot(p_np, ent_b, "-", color="red", linewidth=4.0, alpha=0.8,
                     label=None if red_label_done else "Born Breakdown")
            red_label_done = True
    ax2.set_xlabel("Momentum $p$ [eV]", fontsize=18)
    ax2.set_ylabel("Entanglement Entropy $S_1$", fontsize=18)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.legend(fontsize=15, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(p_min, p_max)
    plt.tight_layout()
    path2 = FIG_DIR / outfile_1d
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    print(f"  Saved {path2}")
    plt.close(fig2)


# ============================================================================
# Figure: entropy vs coherence time tau_coh/t_light       -- paper Fig. 5
# ============================================================================

def figure_coherence(
    p1: float = 0.13, p2: float = 0.125,
    phi1: float = 0.0, phi2: float = 0.0,
    d1: float = 0.99, d2: float = 1.01,
    sigma_real: float = 4.2e-3,
    gamma_min: float = 1e-7, gamma_max: float = 100.0,
    n_points: int = 1000,
    n_quadrature: int = 64,
    outfile: str = "fig_coherence.png",
):
    """
    Entanglement entropy as a function of the ratio tau_coh / t_light, where
    tau_coh = 1 / Im(Sigma) and t_light = |d_1 - d_2| (in natural units).
    Reproduces paper Fig. 5.
    """
    print("\n" + "=" * 70)
    print("Figure: entropy vs coherence time")
    print("=" * 70)

    HBAR_S = 6.582119569e-16   # eV.s

    t_light_eV = abs(d1 - d2)
    t_light_s = t_light_eV * HBAR_S
    print(f"  t_light = {t_light_eV} eV^-1 = {t_light_s:.2e} s")

    global GLOBAL_SIGMA_A, GLOBAL_SIGMA_B
    orig_a, orig_b = GLOBAL_SIGMA_A.clone(), GLOBAL_SIGMA_B.clone()

    gamma_vals = torch.logspace(math.log10(gamma_max), math.log10(gamma_min),
                                n_points, device=DEVICE)
    tau_over_tlight = (1.0 / gamma_vals * HBAR_S) / t_light_s

    base = torch.zeros((1, 6), device=DEVICE)
    base[0, 0], base[0, 1] = p1, p2
    base[0, 2], base[0, 3] = phi1, phi2
    base[0, 4], base[0, 5] = d1, d2
    params = base.expand(n_points, -1).contiguous()

    GLOBAL_SIGMA_A = (sigma_real + 1j * gamma_vals).to(dtype=torch.complex128)
    GLOBAL_SIGMA_B = (sigma_real + 1j * gamma_vals).to(dtype=torch.complex128)

    ent, val = entropy_partial_trace(params, n_quadrature=n_quadrature)
    GLOBAL_SIGMA_A, GLOBAL_SIGMA_B = orig_a, orig_b

    tau_np = tau_over_tlight.cpu().numpy()
    ent_np = ent.cpu().numpy()
    val_np = val.cpu().numpy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.semilogx(tau_np, ent_np, "-", color="#1f77b4", linewidth=2.5)
    ax.axvline(x=1.0, color="#2ca02c", linestyle="--", linewidth=2.5,
               label=r"$\tau_{coh} = t_{light}$")

    broken = val_np >= 1.0
    if broken.any():
        ent_b = ent_np.copy()
        ent_b[~broken] = np.nan
        ax.semilogx(tau_np, ent_b, "o-", color="red", linewidth=4.0,
                    markersize=8,
                    label=r"Born Breakdown ($||\Psi^{(1)}|| \geq ||\Psi^{(0)}||$)")
        first_broken = tau_np[broken][0]
        ax.axvspan(first_broken, tau_np[-1], color="red", alpha=0.1,
                   label="Non-perturbative regime")

    ax.axvspan(tau_np[0], 1.0, color="gray", alpha=0.1)
    ax.set_xlabel(r"$\tau_{coh} / t_{light}$", fontsize=18)
    ax.set_ylabel("Entanglement Entropy $S_1$", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(loc="upper left", fontsize=13, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = FIG_DIR / outfile
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ============================================================================
# Figure: entropy vs (d_1, d_2) for several N_max
# ============================================================================

def figure_position2d(
    p1: float = 0.13, p2: float = 0.12,
    phi1: float = 0.0, phi2: float = 0.0,
    L: float = 2.0,
    # NOTE: in the original __main__ block, sigma=1 was set, which appears to
    # be a leftover test value (the paper uses ~1e-3 eV scale).  The function
    # default below is taken from the function signature in facu_paper.py
    # (1.3e-7 eV).  Set explicitly to the value used for your published figure.
    sigma_real: float = 1.3e-7, gamma: float = 1e-3,
    nmax_values: tuple[int, ...] = (1, 5, 10, 50),
    n_points: int = 240,
    n_quadrature: int = 128,
    outfile: str = "fig_position2D.png",
):
    """
    Entanglement entropy as a function of cavity positions (d_1, d_2),
    for several choices of N_max (number of cavity modes in the sum).
    """
    print("\n" + "=" * 70)
    print("Figure: entropy vs (d_1, d_2) for several N_max")
    print("=" * 70)

    global N_MAX, GLOBAL_SIGMA_A, GLOBAL_SIGMA_B
    orig_n = N_MAX
    orig_a, orig_b = GLOBAL_SIGMA_A.clone(), GLOBAL_SIGMA_B.clone()
    GLOBAL_SIGMA_A = torch.tensor(sigma_real + 1j * gamma,
                                  dtype=torch.complex128, device=DEVICE)
    GLOBAL_SIGMA_B = torch.tensor(sigma_real + 1j * gamma,
                                  dtype=torch.complex128, device=DEVICE)

    d1_vals = torch.linspace(0.0, L, n_points, device=DEVICE)
    d2_vals = torch.linspace(0.0, L, n_points, device=DEVICE)
    D1, D2 = torch.meshgrid(d1_vals, d2_vals, indexing="ij")

    params = torch.zeros((n_points * n_points, 6), device=DEVICE)
    params[:, 0], params[:, 1] = p1, p2
    params[:, 2], params[:, 3] = phi1, phi2
    params[:, 4], params[:, 5] = D1.reshape(-1), D2.reshape(-1)

    ent_grids, val_grids, ent_pools = {}, {}, []
    for nmax in nmax_values:
        print(f"  N_max = {nmax}")
        N_MAX = nmax
        e, v = batch_entropy(params, n_quadrature=n_quadrature)
        ent_grids[nmax] = e.reshape(n_points, n_points).numpy()
        val_grids[nmax] = v.reshape(n_points, n_points).numpy()
        ent_pools.append(ent_grids[nmax][np.isfinite(ent_grids[nmax])])

    N_MAX = orig_n
    GLOBAL_SIGMA_A, GLOBAL_SIGMA_B = orig_a, orig_b

    vmin = np.nanmin([p.min() for p in ent_pools if p.size > 0])
    vmax = np.nanmax([p.max() for p in ent_pools if p.size > 0])

    fig = plt.figure(figsize=(16, 17))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.05],
                           wspace=0.1, hspace=0.3)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    cax = fig.add_subplot(gs[2, :])

    D1n = D1.cpu().numpy() / L
    D2n = D2.cpu().numpy() / L

    for idx, nmax in enumerate(nmax_values):
        ax = axes[idx]
        cf = ax.contourf(D1n, D2n, ent_grids[nmax], levels=25, cmap="hot",
                         extend="both", vmin=vmin, vmax=vmax)
        if np.nanmax(val_grids[nmax]) >= 1.0:
            ax.contour(D1n, D2n, val_grids[nmax], levels=[1.0],
                       colors="cyan", linewidths=2.5)
            if idx == 0:
                ax.plot([], [], color="cyan", linewidth=2.5, label="Born Breakdown")
        ax.plot(np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100),
                "white", linewidth=2.5, linestyle="--", alpha=0.8,
                label="$d_1 + d_2 = L$")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=13)
        ax.set_xlabel("$d_1 / L$", fontsize=24)
        ax.set_ylabel("$d_2 / L$", fontsize=24)
        ax.set_title(f"$N_{{max}}$ = {nmax}", fontsize=24)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_aspect("equal")

    cb = fig.colorbar(cf, cax=cax, orientation="horizontal",
                      format=ticker.FormatStrFormatter("%.1e"))
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Entanglement Entropy $S_1$", fontsize=24)
    path = FIG_DIR / outfile
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ============================================================================
# Figure: entropy along d_2 = L - d_1, for several N_max
# ============================================================================

def figure_position1d(
    p1: float = 0.13, p2: float = 0.12,
    phi1: float = 0.0, phi2: float = 0.0,
    L: float = 2.0,
    sigma_real: float = 4.2e-3, gamma: float = 1e-6,
    nmax_values: tuple[int, ...] = (1, 5, 10, 50),
    n_points: int = 1000,
    n_quadrature: int = 128,
    outfile: str = "fig_position1D.png",
):
    """
    Entanglement entropy along the slice d_2 = L - d_1, for several N_max.
    Useful to gauge how mode truncation affects the position dependence.
    """
    print("\n" + "=" * 70)
    print("Figure: entropy along d_2 = L - d_1 for several N_max")
    print("=" * 70)

    global N_MAX, GLOBAL_SIGMA_A, GLOBAL_SIGMA_B
    orig_n = N_MAX
    orig_a, orig_b = GLOBAL_SIGMA_A.clone(), GLOBAL_SIGMA_B.clone()
    GLOBAL_SIGMA_A = torch.tensor(sigma_real + 1j * gamma,
                                  dtype=torch.complex128, device=DEVICE)
    GLOBAL_SIGMA_B = torch.tensor(sigma_real + 1j * gamma,
                                  dtype=torch.complex128, device=DEVICE)

    d1_vals = torch.linspace(0.0, L, n_points, device=DEVICE)
    d2_vals = L - d1_vals

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    results = {}

    for idx, nmax in enumerate(nmax_values):
        print(f"  N_max = {nmax}")
        N_MAX = nmax
        params = torch.zeros((n_points, 6), device=DEVICE)
        params[:, 0], params[:, 1] = p1, p2
        params[:, 2], params[:, 3] = phi1, phi2
        params[:, 4], params[:, 5] = d1_vals, d2_vals
        e, v = batch_entropy(params, n_quadrature=n_quadrature)
        mask = ~torch.isnan(e)
        d1p = d1_vals[mask].cpu().numpy()
        ent = e[mask].numpy()
        val = v[mask].numpy()
        results[nmax] = {"d1": d1p, "entropy": ent, "validity": val}
        color = colors[idx % len(colors)]
        ax.plot(d1p, ent, "-", linewidth=2.5, color=color,
                label=f"$N_{{max}}$ = {nmax}", alpha=0.8)
        broken = val >= 1.0
        if broken.any():
            ax.plot(d1p[broken], ent[broken], "x", color="red",
                    markersize=8, markeredgewidth=2)
            if idx == 0:
                ax.plot([], [], "x", color="red", markersize=8,
                        markeredgewidth=2, label="Born Breakdown")

    N_MAX = orig_n
    GLOBAL_SIGMA_A, GLOBAL_SIGMA_B = orig_a, orig_b

    ax.axhline(y=np.log(2), color="gray", linestyle="--", linewidth=2,
               label=rf"Max ($\ln 2 \approx {np.log(2):.3f}$)", alpha=0.8)
    ax.axvline(x=L / 2, color="lightgray", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Position $d_1$ [1/eV]", fontsize=18)
    ax.set_ylabel("Entanglement Entropy $S_1$", fontsize=18)
    ax.set_title(rf"$\mathrm{{Re}}\,\Sigma = {sigma_real:.1e}$, "
                 rf"$\mathrm{{Im}}\,\Sigma = {gamma:.1e}$ eV", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=15, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, L)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = FIG_DIR / outfile
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ============================================================================
# Figure: entropy vs angular variables (phi_1, phi_2)
# ============================================================================

def figure_angle(
    p1: float = 0.13, p2: float = 0.13,
    d1: float = 0.5, d2: float = 1.5,
    sigma_real: float = 3.1e-3, gamma: float = 1e-4,
    n_points: int = 200,
    n_quadrature: int = 84,
    outfile_2d: str = "fig_angle_2D.png",
    outfile_1d: str = "fig_angle_1D.png",
):
    """
    Entanglement entropy as a function of the propagation angles
    (phi_1, phi_2) at fixed |p_1|, |p_2|. The diagonal phi_1 = phi_2
    contains the collinear configurations where the entropy drops.
    """
    print("\n" + "=" * 60)
    print("Figure: entropy vs (phi_1, phi_2)")
    print("=" * 60)

    global GLOBAL_SIGMA_A, GLOBAL_SIGMA_B
    orig_a, orig_b = GLOBAL_SIGMA_A.clone(), GLOBAL_SIGMA_B.clone()
    GLOBAL_SIGMA_A = torch.tensor(sigma_real + 1j * gamma,
                                  dtype=torch.complex128, device=DEVICE)
    GLOBAL_SIGMA_B = torch.tensor(sigma_real + 1j * gamma,
                                  dtype=torch.complex128, device=DEVICE)

    phi1_vals = torch.linspace(0.0, 2 * np.pi, n_points, device=DEVICE)
    phi2_vals = torch.linspace(0.0, 2 * np.pi, n_points, device=DEVICE)
    Phi1, Phi2 = torch.meshgrid(phi1_vals, phi2_vals, indexing="ij")

    params = torch.zeros((n_points * n_points, 6), device=DEVICE)
    params[:, 0], params[:, 1] = p1, p2
    params[:, 2], params[:, 3] = Phi1.reshape(-1), Phi2.reshape(-1)
    params[:, 4], params[:, 5] = d1, d2

    ent, val = batch_entropy(params, n_quadrature=n_quadrature)
    GLOBAL_SIGMA_A, GLOBAL_SIGMA_B = orig_a, orig_b

    ent_grid = ent.reshape(n_points, n_points).numpy()
    val_grid = val.reshape(n_points, n_points).numpy()
    Phi1_np, Phi2_np = Phi1.cpu().numpy(), Phi2.cpu().numpy()

    valid = np.isfinite(ent_grid)
    if 4 < valid.sum() < ent_grid.size:
        pts = np.column_stack((Phi1_np[valid], Phi2_np[valid]))
        grid = np.column_stack((Phi1_np.ravel(), Phi2_np.ravel()))
        ent_grid = np.where(
            valid, ent_grid,
            griddata(pts, ent_grid[valid], grid, method="linear").reshape(n_points, n_points),
        )
        val_grid = np.where(
            valid, val_grid,
            griddata(pts, val_grid[valid], grid, method="linear").reshape(n_points, n_points),
        )

    pi_ticks  = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    pi_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    # 2D heatmap
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    cf = ax1.contourf(Phi1_np, Phi2_np, ent_grid, levels=25, cmap="viridis")
    ax1.contour(Phi1_np, Phi2_np, ent_grid, levels=25,
                colors="white", linewidths=0.5, alpha=0.3)
    if val_grid.max() >= 1.0:
        ax1.contour(Phi1_np, Phi2_np, val_grid, levels=[1.0],
                    colors="red", linewidths=3.5)
        ax1.plot([], [], color="red", linewidth=3.5,
                 label=r"Born Breakdown ($||\Psi^{(1)}|| \geq ||\Psi^{(0)}||$)")
    diag = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(diag, diag, "w--", linewidth=2.5, alpha=0.8,
             label=r"$\phi_1 = \phi_2$ (collinear)")
    ax1.set_xticks(pi_ticks); ax1.set_xticklabels(pi_labels)
    ax1.set_yticks(pi_ticks); ax1.set_yticklabels(pi_labels)
    ax1.set_xlabel(r"Angle $\phi_1$ [rad]", fontsize=16, fontweight="bold")
    ax1.set_ylabel(r"Angle $\phi_2$ [rad]", fontsize=16, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)
    cb = plt.colorbar(cf, ax=ax1)
    cb.set_label("Entropy $S_A$", fontsize=16)
    plt.tight_layout()
    path1 = FIG_DIR / outfile_2d
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    print(f"  Saved {path1}")
    plt.close(fig1)

    # 1D cuts
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    idx = np.arange(n_points)
    ent_diag = ent_grid[idx, idx]
    val_diag = val_grid[idx, idx]
    mid = n_points // 2
    ent_phi1_fixed = ent_grid[mid, :]
    val_phi1_fixed = val_grid[mid, :]
    phi = phi1_vals.cpu().numpy()
    ax2.plot(phi, ent_diag, "o-", linewidth=2.5, markersize=5, color="#2E86AB",
             label=r"$\phi_1 = \phi_2$ (diagonal)")
    ax2.plot(phi, ent_phi1_fixed, "s-", linewidth=2.5, markersize=5,
             color="#E63946", label=r"$\phi_1 = \pi$ fixed")
    red_label_done = False
    for cut_val, cut_ent in ((val_diag, ent_diag),
                             (val_phi1_fixed, ent_phi1_fixed)):
        broken = cut_val >= 1.0
        if broken.any():
            ent_b = cut_ent.copy()
            ent_b[~broken] = np.nan
            ax2.plot(phi, ent_b, "-", color="red", linewidth=4.0, alpha=0.8,
                     label=None if red_label_done else "Born Breakdown")
            red_label_done = True
    ax2.set_xticks(pi_ticks); ax2.set_xticklabels(pi_labels)
    ax2.set_xlabel(r"Angle $\phi$ [rad]", fontsize=16, fontweight="bold")
    ax2.set_ylabel("Entanglement Entropy $S_A$", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2 * np.pi)
    plt.tight_layout()
    path2 = FIG_DIR / outfile_1d
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    print(f"  Saved {path2}")
    plt.close(fig2)


# ============================================================================
# CLI
# ============================================================================

FIGURE_FUNCTIONS = {
    "selfenergy":  figure_selfenergy,
    "momentum":    figure_momentum,
    "coherence":   figure_coherence,
    "position2d":  figure_position2d,
    "position1d":  figure_position1d,
    "angle":       figure_angle,
}


def main():
    p = argparse.ArgumentParser(
        description=("Reproduce figures from Arreyes et al., "
                     "'Entanglement in (1+2) QED in Double-Layer Honeycomb Lattices'."),
    )
    p.add_argument(
        "--figure", "-f",
        choices=list(FIGURE_FUNCTIONS.keys()) + ["all"],
        required=True,
        help="Which figure to reproduce. 'all' runs every figure in sequence.",
    )
    p.add_argument(
        "--quick", action="store_true",
        help=("Run with reduced resolution (n_points and quadrature order) "
              "for quick smoke testing.  Output figures will be noisier."),
    )
    args = p.parse_args()

    quick_overrides = dict(n_points=40, n_quadrature=64) if args.quick else {}

    if args.figure == "all":
        for name, fn in FIGURE_FUNCTIONS.items():
            print(f"\n>>> Running figure '{name}' <<<")
            fn(**{k: v for k, v in quick_overrides.items()
                  if k in fn.__code__.co_varnames})
    else:
        fn = FIGURE_FUNCTIONS[args.figure]
        fn(**{k: v for k, v in quick_overrides.items()
              if k in fn.__code__.co_varnames})


if __name__ == "__main__":
    main()
