"""
models.py
=========
Closed-form breakthrough-curve models for the parent compound and its
byproducts.

Each model returns the predicted effluent concentration of one species over
time, given its apparent rate constants and an ``ExperimentConfig``. The
expressions are the semi-analytical solutions used in the published study:
a first-order reaction term multiplied by the gamma breakthrough ramp during
the step phase (``transport.gamma_rtd``), followed by exponential washout
during the pulse phase (``transport.step_pulse_profile``).

The reaction network reproduced here (parent -> byproducts) is, in the
published case, glyphosate (GP) -> AMPA / glycine (Gly) -> orthophosphate (Pi)
/ ammonium (NH4), with manganese-oxide reductive dissolution coupled to GP
oxidation. To model a different system, replace these functions with the
closed-form (or numerical) solutions of your own rate laws; the transport and
fitting layers do not change.

Factory functions ``make_*_model(cfg)`` return ``curve_fit``-compatible
callables ``f(t, *params)`` with the configuration captured by closure.
"""

from __future__ import annotations
import numpy as np
from .transport import gamma_rtd, step_pulse_profile
from .config import ExperimentConfig


# --------------------------------------------------------------------------- #
# Parent compound and primary byproducts
# --------------------------------------------------------------------------- #
def gp_btc(t, k_gp_loss, cfg: ExperimentConfig):
    """Parent compound (glyphosate) breakthrough curve.

    First-order loss during the step phase, washout during the pulse phase.
    """
    def step_fn(tt):
        return cfg.C0 * gamma_rtd(tt, cfg.tau_step_eff, cfg.N) * np.exp(-k_gp_loss * tt)

    return step_pulse_profile(t, cfg.t_s, step_fn, 1.0 / cfg.tau_pulse + k_gp_loss)


def _form_loss_btc(t, k_form, k_loss, cfg: ExperimentConfig):
    """Generic byproduct formed from the parent at rate ``k_form`` and lost at
    ``k_loss`` (used for both AMPA and glycine)."""
    def step_fn(tt):
        return (
            cfg.C0
            * (k_form / k_loss)
            * gamma_rtd(tt, cfg.tau_step_eff, cfg.N)
            * (1.0 - np.exp(-k_loss * tt))
        )

    return step_pulse_profile(t, cfg.t_s, step_fn, 1.0 / cfg.tau_pulse + k_loss)


def ampa_btc(t, k_ampa_form, k_ampa_loss, cfg: ExperimentConfig):
    """AMPA breakthrough curve (C(2)-N cleavage pathway)."""
    return _form_loss_btc(t, k_ampa_form, k_ampa_loss, cfg)


def gly_btc(t, k_gly_form, k_gly_loss, cfg: ExperimentConfig):
    """Glycine breakthrough curve (glycine / sarcosine pathway)."""
    return _form_loss_btc(t, k_gly_form, k_gly_loss, cfg)


def pi_btc(t, k_gly_form, k_ampa_form, k_gp_loss, k_ampa_loss, cfg: ExperimentConfig):
    """Orthophosphate breakthrough curve.

    Phosphate is released both from the glycine/sarcosine branch (tracking GP
    loss) and from AMPA oxidation. Washout decays at ``1/tau_pulse`` (no extra
    first-order loss term for Pi).
    """
    def step_fn(tt):
        term_ampa = k_ampa_form * (1.0 - np.exp(-k_ampa_loss * tt))
        term_gly = k_gly_form * np.exp(-k_gp_loss * tt)
        return cfg.C0 * (term_ampa + term_gly) * gamma_rtd(tt, cfg.tau_step_eff, cfg.N)

    return step_pulse_profile(t, cfg.t_s, step_fn, 1.0 / cfg.tau_pulse)


def nh4_btc(
    t, k_gly_form, k_gly_loss, k_ampa_form, k_ampa_loss, k_nh4_loss,
    cfg: ExperimentConfig,
):
    """Ammonium breakthrough curve.

    NH4+ accumulates from both the glycine and AMPA branches and is itself lost
    at ``k_nh4_loss``.

    NOTE: this reproduces the original notebook expression faithfully. In the
    original, the pulse-phase anchor multiplied the time-varying generation
    terms rather than their value at ``t_s``; that behaviour is preserved here
    so released results match the paper. If you re-use this for a new system,
    consider whether you want the standard ``step_pulse_profile`` anchoring
    (value evaluated at ``t_s``) instead - see the commented alternative below.
    """
    t = np.asarray(t, dtype=float)
    term1 = k_gly_form * (1.0 - np.exp(-k_gly_loss * t))
    term2 = k_ampa_form * (1.0 - np.exp(-k_ampa_loss * t))
    step = cfg.C0 * (term1 + term2) * gamma_rtd(t, cfg.tau_step_eff, cfg.N)
    step_val_ts = cfg.C0 * (term1 + term2) * gamma_rtd(cfg.t_s, cfg.tau_step_eff, cfg.N)
    pulse = step_val_ts * np.exp(-(1.0 / cfg.tau_pulse + k_nh4_loss) * (t - cfg.t_s))
    return np.where(t <= cfg.t_s, step, pulse)

    # --- Standard-anchoring alternative (recommended for new systems) --------
    # def step_fn(tt):
    #     a = k_gly_form * (1 - np.exp(-k_gly_loss * tt))
    #     b = k_ampa_form * (1 - np.exp(-k_ampa_loss * tt))
    #     return cfg.C0 * (a + b) * gamma_rtd(tt, cfg.tau_step_eff, cfg.N)
    # return step_pulse_profile(t, cfg.t_s, step_fn, 1/cfg.tau_pulse + k_nh4_loss)


# --------------------------------------------------------------------------- #
# Manganese-oxide dissolution / regeneration
# --------------------------------------------------------------------------- #
def mnsoln_btc(t, k_diss, cfg: ExperimentConfig):
    """Soluble Mn breakthrough curve (simple first-order dissolution form)."""
    def step_fn(tt):
        return cfg.C0 * gamma_rtd(tt, cfg.tau_step_eff, cfg.N) * (1.0 - np.exp(-k_diss * tt))

    return step_pulse_profile(t, cfg.t_s, step_fn, 1.0 / cfg.tau_pulse)


def mnsoln_btc_fixed_gp(t, k_mn_ox, k_gp_loss, cfg: ExperimentConfig):
    """Soluble Mn breakthrough with reductive dissolution coupled to GP loss.

    Dissolution is stoichiometrically tied to GP oxidation (rate scales with a
    fixed ``k_gp_loss``), while soluble Mn is re-oxidized at ``k_mn_ox``.
    """
    def step_fn(tt):
        return (
            cfg.C0
            * (k_gp_loss / k_mn_ox)
            * gamma_rtd(tt, cfg.tau_step_eff, cfg.N)
            * (1.0 - np.exp(-k_mn_ox * tt))
        )

    return step_pulse_profile(t, cfg.t_s, step_fn, 1.0 / cfg.tau_pulse + k_mn_ox)


# --------------------------------------------------------------------------- #
# curve_fit-compatible factories (config captured by closure)
# --------------------------------------------------------------------------- #
def make_pi_model(cfg: ExperimentConfig):
    """Return f(t, k_gly_form, k_ampa_form, k_gp_loss, k_ampa_loss)."""
    return lambda t, a, b, c, d: pi_btc(t, a, b, c, d, cfg)


def make_nh4_model(cfg: ExperimentConfig):
    """Return f(t, k_gly_form, k_gly_loss, k_ampa_form, k_ampa_loss, k_nh4_loss)."""
    return lambda t, a, b, c, d, e: nh4_btc(t, a, b, c, d, e, cfg)


def make_mnsoln_model(cfg: ExperimentConfig):
    """Return f(t, k_diss)."""
    return lambda t, k: mnsoln_btc(t, k, cfg)


def make_mnsoln_fixed_gp_model(cfg: ExperimentConfig, k_gp_loss: float):
    """Return f(t, k_mn_ox) with k_gp_loss held fixed."""
    return lambda t, k_mn_ox: mnsoln_btc_fixed_gp(t, k_mn_ox, k_gp_loss, cfg)
