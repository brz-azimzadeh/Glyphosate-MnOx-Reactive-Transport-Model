"""
bromide.py
==========
Hydraulic characterization from a conservative (Br-) tracer test.

This implements the bromide transport model of SI Text S3 (eqs S1-S5): a
gamma residence-time distribution (RTD) describing a non-ideal plug-flow
reactor, fit to a two-phase tracer breakthrough (step input then washout).
The fitted hydraulics - the RTD shape ``N``, the step residence ``tau``, and
the washout time constant ``tau_pulse`` - are reactor properties and are then
held FIXED when fitting the reactive-transport model (glyphosate + byproducts).

Model (same gamma form used throughout the package):

    step  (0 <= t <= t_s):  C(t) = C0 * [1 - Q(N, t/tau)]
    pulse (t > t_s)      :  C(t) = C(t_s) * exp(-(t - t_s)/tau_pulse)

where Q(N, x) = Gamma(N, x)/Gamma(N) is the regularized upper incomplete gamma
function (``scipy.special.gammaincc``), so 1 - Q(N, t/tau) is the cumulative
gamma RTD (the breakthrough fraction).

Note on residence time: ``tau`` here is the gamma *scale*; the mean residence
time is ``N * tau``. The published study reports a mean residence ~1.58 min.

Note on reactive species: a conservative tracer is not retarded, whereas
adsorbing species (e.g. glyphosate) break through later. The tracer fixes the
dispersion shape ``N`` and the washout ``tau_pulse``; the reactive step
residence may be larger (retardation) - see ``ExperimentConfig.retardation``.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammaincc


def tracer_step(t, C0, tau, N):
    """Step-phase tracer breakthrough: C0 * cumulative gamma RTD."""
    return C0 * (1.0 - gammaincc(N, np.asarray(t, float) / tau))


def tracer_curve(t, C0, t_s, tau, N, tau_pulse):
    """Full two-phase tracer breakthrough (step then washout)."""
    t = np.asarray(t, float)
    out = np.empty_like(t)
    step = t <= t_s
    out[step] = tracer_step(t[step], C0, tau, N)
    c_ts = float(tracer_step(t_s, C0, tau, N))
    out[~step] = c_ts * np.exp(-(t[~step] - t_s) / tau_pulse)
    return out


@dataclass
class HydraulicResult:
    """Fitted hydraulic parameters from a tracer test."""
    tau: float            # gamma scale (min); mean residence = N*tau
    N: float              # RTD shape parameter
    tau_pulse: float      # washout time constant (min)
    C0: float             # tracer plateau concentration
    t_s: float            # step -> washout switch time (min)
    r2: float             # overall R^2 of the breakthrough fit
    predict: Callable     # predict(t) -> tracer concentration

    @property
    def mean_residence(self) -> float:
        return self.N * self.tau

    def summary(self) -> str:
        return (
            "Hydraulic (Br- tracer) fit:\n"
            f"  tau (gamma scale)   = {self.tau:.3f} min\n"
            f"  N (RTD shape)       = {self.N:.3f}\n"
            f"  mean residence N*tau= {self.mean_residence:.3f} min\n"
            f"  tau_pulse (washout) = {self.tau_pulse:.3f} min\n"
            f"  C0                  = {self.C0:.4f}\n"
            f"  t_s                 = {self.t_s:.1f} min\n"
            f"  R^2                 = {self.r2:.4f}"
        )


def _detect_ts(t, c):
    """Heuristic: washout begins where the signal first drops below 70% of the
    plateau after having reached it."""
    cmax = np.nanmax(c)
    plateau = c >= 0.7 * cmax
    if not plateau.any():
        return float(t[-1])
    first_plateau = np.argmax(plateau)
    for i in range(first_plateau + 1, len(c)):
        if c[i] < 0.5 * cmax:
            return float(t[i - 1])
    return float(t[-1])


def fit_tracer(
    t,
    c,
    C0: Optional[float] = None,
    t_s: Optional[float] = None,
    fix_N: Optional[float] = None,
    fit_C0: bool = False,
) -> HydraulicResult:
    """Fit the gamma-RTD tracer model to a Br- breakthrough curve.

    Parameters
    ----------
    t, c : array-like
        Tracer sampling times (min) and effluent concentrations.
    C0 : float, optional
        Known plateau concentration of the tracer feed. If None, taken as the
        observed plateau (max of the step phase).
    t_s : float, optional
        Step -> washout switch time (min). If None, auto-detected.
    fix_N : float, optional
        Fix the RTD shape N (e.g. 4 as in the paper) instead of fitting it.
    fit_C0 : bool
        If True, also fit C0 (otherwise held at the value above).

    Returns
    -------
    HydraulicResult
        Fitted tau, N, tau_pulse (+ C0, t_s, R^2, and a predict() callable).
    """
    t = np.asarray(t, float)
    c = np.asarray(c, float)
    m = np.isfinite(t) & np.isfinite(c)
    t, c = t[m], c[m]

    if t_s is None:
        t_s = _detect_ts(t, c)
    if C0 is None:
        C0 = float(np.nanmax(c[t <= t_s])) if (t <= t_s).any() else float(np.nanmax(c))

    step = t <= t_s
    wash = t > t_s

    # --- Fit the step phase for tau (and N, C0 if requested) ---
    if fix_N is not None and not fit_C0:
        f = lambda tt, tau: tracer_step(tt, C0, tau, fix_N)
        p, _ = curve_fit(f, t[step], c[step], p0=[1.0], bounds=([1e-3], [1e3]))
        tau, N = float(p[0]), float(fix_N)
    elif fix_N is not None and fit_C0:
        f = lambda tt, tau, c0: tracer_step(tt, c0, tau, fix_N)
        p, _ = curve_fit(f, t[step], c[step], p0=[1.0, C0],
                         bounds=([1e-3, 1e-6], [1e3, np.inf]))
        tau, C0, N = float(p[0]), float(p[1]), float(fix_N)
    elif fix_N is None and not fit_C0:
        f = lambda tt, tau, N: tracer_step(tt, C0, tau, N)
        p, _ = curve_fit(f, t[step], c[step], p0=[1.0, 4.0],
                         bounds=([1e-3, 1.0], [1e3, 50.0]))
        tau, N = float(p[0]), float(p[1])
    else:
        f = lambda tt, tau, N, c0: tracer_step(tt, c0, tau, N)
        p, _ = curve_fit(f, t[step], c[step], p0=[1.0, 4.0, C0],
                         bounds=([1e-3, 1.0, 1e-6], [1e3, 50.0, np.inf]))
        tau, N, C0 = float(p[0]), float(p[1]), float(p[2])

    # --- Fit the washout phase for tau_pulse ---
    c_ts = float(tracer_step(t_s, C0, tau, N))
    if wash.sum() >= 2 and c_ts > 0:
        fw = lambda tt, taup: c_ts * np.exp(-(tt - t_s) / taup)
        pw, _ = curve_fit(fw, t[wash], c[wash], p0=[max(tau, 1.0)],
                          bounds=([1e-3], [1e4]))
        tau_pulse = float(pw[0])
    else:
        tau_pulse = tau  # fall back to step residence if no washout data

    def predict(tt):
        return tracer_curve(tt, C0, t_s, tau, N, tau_pulse)

    ss_res = float(np.sum((c - predict(t)) ** 2))
    ss_tot = float(np.sum((c - np.mean(c)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return HydraulicResult(tau=tau, N=N, tau_pulse=tau_pulse, C0=C0,
                           t_s=float(t_s), r2=r2, predict=predict)
