"""
transport.py
============
Hydraulic transport for a nonideal plug-flow reactor.

The reactor is never an ideal PFR, so axial dispersion is represented with a
gamma-distributed residence-time distribution (RTD). Two helpers are provided:

* ``gamma_rtd``        - the cumulative RTD (fraction of a step signal that has
                         broken through by time t).
* ``step_pulse_profile`` - assembles a full breakthrough curve from a step-phase
                         generation function plus an exponential washout.

These are independent of the chemistry: any reaction model in ``models.py``
plugs its step-phase expression into ``step_pulse_profile`` and gets a
transport-aware effluent curve back.
"""

from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.special import gammaincc


def gamma_rtd(t, tau: float, N: int = 4):
    """Cumulative gamma residence-time distribution (breakthrough fraction).

    Returns the regularized *lower* incomplete gamma function, i.e. the gamma
    CDF, which gives the fraction of a conservative step input that has exited
    the reactor by time ``t``. Implemented as ``1 - Q(N, t/tau)`` where
    ``Q = gammaincc`` is the regularized *upper* incomplete gamma in SciPy.

    Parameters
    ----------
    t : array_like
        Time (min).
    tau : float
        Mean residence time (min).
    N : int
        Gamma shape parameter (larger -> sharper, more plug-flow-like).

    Returns
    -------
    ndarray
        Breakthrough fraction in [0, 1].
    """
    t = np.asarray(t, dtype=float)
    return 1.0 - gammaincc(N, t / tau)


def step_pulse_profile(
    t,
    t_s: float,
    step_fn: Callable[[np.ndarray], np.ndarray],
    pulse_decay_rate: float,
):
    """Assemble a step + washout breakthrough curve.

    Phase 1 (t <= t_s): the effluent follows ``step_fn(t)`` (reaction modulated
    by the breakthrough ramp). Phase 2 (t > t_s): the influent is replaced by
    background solution and the effluent decays exponentially from its value at
    the switch time, ``step_fn(t_s)``, at rate ``pulse_decay_rate``.

    Parameters
    ----------
    t : array_like
        Time points (min).
    t_s : float
        Step-to-pulse switch time (min).
    step_fn : callable
        Function of time returning the Phase-1 effluent concentration.
    pulse_decay_rate : float
        Combined washout rate for Phase 2, typically ``1/tau_pulse`` plus any
        first-order loss of the species (min^-1).

    Returns
    -------
    ndarray
        Effluent concentration over ``t``.
    """
    t = np.asarray(t, dtype=float)
    step_vals = step_fn(t)
    anchor = float(step_fn(np.array([t_s], dtype=float))[0])  # scalar at switch
    pulse_vals = anchor * np.exp(-pulse_decay_rate * (t - t_s))
    return np.where(t <= t_s, step_vals, pulse_vals)
