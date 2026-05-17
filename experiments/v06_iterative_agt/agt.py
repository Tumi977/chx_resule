"""AGT (Adaptive Ground Truth) — fixed for paired data setting.

Background:
    The original IterativePFN (CVPR'23) trains T stages with progressively
    smaller noise σ_0 > σ_1 > ... > σ_{T-1}. The τ-th module's job is to
    denoise inputs at noise level σ_τ.

    In their setting, training data is unpaired (PointCleanNet style), so
    they use NN-projection: target_i = NN(x_i, Y + σ_τ·ξ) - x_i. This works
    because x_i drifts from its corresponding clean point during the cascade.

    In our setting, pc_noisy and pc_clean are point-wise paired. The NN
    projection degenerates to x_i -> y_τ_i (paired point), so the supervision
    target collapses to -σ_τ·ξ — pure noise direction with no learning signal
    once the network is initialized.

Fix:
    Replace NN-projection with explicit per-stage noise generation:

        x_τ_input  = pc_clean + σ_τ · ξ_τ           # generate σ_τ noisy input
        target_τ   = pc_clean - x_τ_input
                   = -σ_τ · ξ_τ                     # deterministic GT

    Each stage learns a denoiser for *its specific* σ_τ. At inference, we run
    stages serially: ItM_0 (large σ) -> ItM_1 (smaller σ) -> ... handling
    decreasing noise levels in cascade.

    This is the IterativePFN paper's actual training intent — they describe
    the NN trick because they don't have pairing, but the underlying principle
    is "each stage learns one noise level". With pairing we can do this
    directly.
"""
from __future__ import annotations
from dataclasses import dataclass

import jittor as jt
import numpy as np


@dataclass
class AGTConfig:
    n_steps: int = 4              # T
    sigma0_min: float = 0.005     # σ_0 ~ U(min, max), per-shape
    sigma0_max: float = 0.020
    delta: float = 2.0            # σ_τ = σ_0 / δ^τ ; default δ=2 -> σ_T = σ_0/16
    sigma_floor: float = 1e-4     # σ_T can't be smaller than this


def schedule_sigmas(sigma0: float, n_steps: int, delta: float, sigma_floor: float = 1e-4) -> np.ndarray:
    """Returns σ_0 .. σ_{T-1} as np.ndarray of shape (n_steps,)."""
    out = np.empty(n_steps, dtype=np.float32)
    for tau in range(n_steps):
        out[tau] = max(sigma_floor, sigma0 / (delta ** tau))
    return out


def make_stage_input(
    pc_clean: jt.Var,
    sigma_tau: jt.Var,
) -> tuple[jt.Var, jt.Var]:
    """Generate stage-τ training input + target (FIXED VERSION).

    Args:
        pc_clean:  (B, M, 3) — fully clean GT, point-wise paired
        sigma_tau: (B,)      — per-batch σ at this stage

    Returns:
        x_input:  (B, M, 3) = pc_clean + σ_τ · ξ
        target:   (B, M, 3) = -σ_τ · ξ = pc_clean - x_input
    """
    B = pc_clean.shape[0]
    xi = jt.randn_like(pc_clean)
    sigma = sigma_tau.reshape(B, 1, 1)
    x_input = pc_clean + sigma * xi
    target = -sigma * xi
    return x_input, target


def stage_loss(
    pred_disp: jt.Var,
    target: jt.Var,
    sigma_tau: jt.Var | None = None,
) -> jt.Var:
    """Per-stage L2 loss, optionally normalized by σ_τ² for scale-invariance.

    Args:
        pred_disp: (B, M, 3) network output
        target:    (B, M, 3)
        sigma_tau: (B,) optional; if given, divides each batch's loss by σ_τ²
                   so all stages contribute equally regardless of noise scale
    """
    B = pred_disp.shape[0]
    err = ((pred_disp - target) ** 2).sum(-1)         # (B, M)
    if sigma_tau is not None:
        # divide each batch by its σ² so stages contribute equally
        s2 = (sigma_tau.reshape(B, 1) ** 2 + 1e-8)
        err = err / s2
    return err.mean()


# --- Backwards compat shims (so old train_p2.py / train_iter_p1.py still
#     import without breaking, but route to the fixed paths) -------------

def make_agt_target_jt(x_prev, y_clean, sigma_tau, rng_state=None):
    """DEPRECATED legacy interface, returns (y_tau, target_disp, nn_idx).

    Now reroutes to the fixed make_stage_input formulation. The "y_tau"
    returned is actually pc_clean (placeholder). We keep this for compat with
    older training scripts but new code should call make_stage_input directly.
    """
    x_input, target = make_stage_input(y_clean, sigma_tau)
    # placeholder NN-idx (not used by fixed code)
    B, M, _ = x_input.shape
    nn_idx = jt.zeros((B, M)).int32()
    return y_clean, target, nn_idx


def agt_step_loss(pred_disp, x_input, y_tau, weights=None):
    """DEPRECATED. Use stage_loss with explicit target instead."""
    # Old behaviour: NN(x_input, y_tau) - x_input as target.
    # In paired setting this equals (y_tau[i] - x_input[i]) = pc_clean - x_input.
    # So we can compute target directly without NN.
    target = y_tau - x_input
    return stage_loss(pred_disp, target)


def rect_flow_mix(x_prev, y_tau, p=1.0):
    """DEPRECATED. The fixed AGT loop generates input directly, no mixing."""
    B = x_prev.shape[0]
    if p < 1.0:
        active = (jt.rand((B, 1, 1)) < p).float32()
    else:
        active = jt.ones((B, 1, 1))
    t = jt.rand((B, 1, 1)) * active
    x_input = (1.0 - t) * x_prev + t * y_tau
    return x_input, t
