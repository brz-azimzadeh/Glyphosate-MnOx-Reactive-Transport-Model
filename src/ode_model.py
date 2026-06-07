"""
ode_model.py
============
Fully-coupled NUMERICAL reactive-transport model.

Instead of the closed-form per-species expressions (``models.py``), this module
integrates the coupled first-order kinetic system derived in the Supporting
Information (Text S4, eqs S21/S26/S31/S34/S39, plus the Mn balance eq 9) with
``scipy.integrate.solve_ivp``, and applies the gamma residence-time
distribution (RTD) for transport.

Why this matters: every species is driven by the SAME shared rate constants.
Pi follows from the actual [GP] and [AMPA] trajectories (eq S34), NH4 from [Gly]
and [AMPA] (eq S39), and soluble Mn from [GP] (eq 9). Because the system is
integrated over the full experiment, byproduct accumulation emerges naturally
from one shared parameter set - no separate effective amplitude constants are
needed. This is the fully-coupled realization of the framework.

Reaction network (state vector y = [GP, AMPA, Gly, Pi, NH4, Mn]):
    d[GP]/dt   = -kGP*[GP]
    d[AMPA]/dt =  kAf*[GP] - kAl*[AMPA]
    d[Gly]/dt  =  kGf*[GP] - kGl*[Gly]
    d[Pi]/dt   =  kGf*[GP] + kAl*[AMPA]
    d[NH4]/dt  =  kGl*[Gly] + kAl*[AMPA] - kNH4*[NH4]
    d[Mn]/dt   =  kGP*[GP] - kMn*[Mn]

Parameter vector (shared across all species):
    k = [kGP, kAf, kAl, kGf, kGl, kNH4, kMn]
      =  k_GP_loss, k_AMPA_form, k_AMPA_loss, k_Gly_form, k_Gly_loss,
         k_NH4_loss, k_Mn_ox

Transport: during the step phase (t <= t_s) the reacted concentration is scaled
by the cumulative gamma RTD (breakthrough ramp); during washout (t > t_s) the
effluent decays from its value at t_s at rate (1/tau_pulse + species loss),
consistent with SI eqs S46/S48/S50/S52/S54.
"""

from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammaincc

from .config import ExperimentConfig

SPECIES = ["GP", "AMPA", "Gly", "Pi", "NH4", "Mn"]
ODE_PARAM_NAMES = ["k_GP_loss", "k_AMPA_form", "k_AMPA_loss",
                   "k_Gly_form", "k_Gly_loss", "k_NH4_loss", "k_Mn_ox"]


def reaction_rhs(t, y, k):
    """Right-hand side of the coupled reaction ODE system (no transport)."""
    GP, AMPA, Gly, Pi, NH4, Mn = y
    kGP, kAf, kAl, kGf, kGl, kNH4, kMn = k
    dGP = -kGP * GP
    dAMPA = kAf * GP - kAl * AMPA
    dGly = kGf * GP - kGl * Gly
    dPi = kGf * GP + kAl * AMPA
    dNH4 = kGl * Gly + kAl * AMPA - kNH4 * NH4
    dMn = kGP * GP - kMn * Mn
    return [dGP, dAMPA, dGly, dPi, dNH4, dMn]


def build_effluent(k: Sequence[float], cfg: ExperimentConfig):
    """Integrate the coupled ODEs once and return an evaluator.

    Returns
    -------
    callable
        ``eval_species(name, t)`` giving the predicted effluent concentration of
        ``name`` (one of SPECIES) at times ``t`` (scalar or array), including the
        RTD breakthrough ramp and the washout phase.
    """
    k = np.asarray(k, dtype=float)
    y0 = [cfg.C0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(reaction_rhs, [0.0, cfg.t_s], y0, args=(k,),
                    dense_output=True, rtol=1e-7, atol=1e-9)

    def G(t):  # cumulative gamma RTD (breakthrough fraction)
        return 1.0 - gammaincc(cfg.N, np.asarray(t, float) / cfg.tau_step_eff)

    Creact_ts = sol.sol(cfg.t_s)
    Gts = float(G(cfg.t_s))
    # First-order loss that adds to hydraulic washout, per species (SI S46-S54)
    kloss = {"GP": k[0], "AMPA": k[2], "Gly": k[4],
             "Pi": 0.0, "NH4": k[5], "Mn": k[6]}
    eff_ts = {sp: Creact_ts[i] * Gts for i, sp in enumerate(SPECIES)}

    def eval_species(name: str, t):
        i = SPECIES.index(name)
        t = np.asarray(t, dtype=float)
        scalar = (t.ndim == 0)
        t = np.atleast_1d(t)
        out = np.empty_like(t, dtype=float)
        step = t <= cfg.t_s
        if step.any():
            Cr = sol.sol(t[step])[i]
            out[step] = Cr * G(t[step])
        if (~step).any():
            rate = 1.0 / cfg.tau_pulse + kloss[name]
            out[~step] = eff_ts[name] * np.exp(-rate * (t[~step] - cfg.t_s))
        return float(out[0]) if scalar else out

    return eval_species


def simulate(k: Sequence[float], cfg: ExperimentConfig, t,
             species: Sequence[str] = SPECIES) -> Dict[str, np.ndarray]:
    """Convenience wrapper: predict all requested species on a single grid ``t``."""
    ev = build_effluent(k, cfg)
    return {sp: ev(sp, t) for sp in species}
