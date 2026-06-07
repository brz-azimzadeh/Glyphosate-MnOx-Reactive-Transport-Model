"""
fit.py
======
Parameter-estimation routines that recover apparent rate constants from
measured effluent breakthrough curves.

Two strategies are provided, matching the published workflow:

* ``fit_gp_ampa_gly`` - a single constrained global optimization that fits the
  parent compound and its two primary byproducts simultaneously (5 constants),
  enforcing the mechanistic ordering reported in the paper.
* ``fit_curve`` - a thin wrapper around ``scipy.optimize.curve_fit`` for the
  remaining species (Pi, NH4, soluble Mn), with parameter uncertainties from
  the covariance matrix.

All routines return a ``FitResult`` carrying the fitted values, standard
deviations, goodness-of-fit, and a ``predict`` callable for plotting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit, least_squares, Bounds, NonlinearConstraint
from sklearn.metrics import mean_squared_error, r2_score

from .config import ExperimentConfig
from . import models


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    """Holds fitted parameters and diagnostics for one fit."""
    species: List[str]
    param_names: List[str]
    values: np.ndarray
    sd: np.ndarray
    rmse: float
    r2: float
    predict: Optional[Callable] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Parameter": self.param_names,
             "Value (min^-1)": self.values,
             "+/-SD": self.sd}
        )

    def summary(self) -> str:
        lines = [f"Fit for: {', '.join(self.species)}",
                 f"  RMSE = {self.rmse:.4g}   R2 = {self.r2:.4f}"]
        for n, v, s in zip(self.param_names, self.values, self.sd):
            lines.append(f"  {n:<14} = {v:.4e}  +/- {s:.2e}")
        for k, v in self.extra.items():
            lines.append(f"  {k:<14} = {v:.4g}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean(df: pd.DataFrame, cfg: ExperimentConfig, cols: Sequence[str]):
    """Return (time, {col: values}) keeping rows where all requested columns
    are positive and finite."""
    t = df[cfg.columns["time"]].to_numpy(float)
    series = {c: df[cfg.columns[c]].to_numpy(float) for c in cols}
    mask = np.isfinite(t)
    for v in series.values():
        mask &= np.isfinite(v) & (v > 0)
    return t[mask], {c: series[c][mask] for c in cols}


# --------------------------------------------------------------------------- #
# Global constrained fit: GP + AMPA + glycine
# --------------------------------------------------------------------------- #
GAG_PARAM_NAMES = ["k_GP_loss", "k_AMPA_form", "k_AMPA_loss",
                   "k_Gly_form", "k_Gly_loss"]


def fit_gp_ampa_gly(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    p0: Sequence[float] = (1e-5, 6e-4, 5e-4, 1.3e-3, 3e-4),
    use_constraints: bool = True,
    ordering: bool = True,
    fixed: Optional[Dict[str, float]] = None,
    maxiter: int = 10000,
) -> FitResult:
    """Global fit of the 5 apparent rate constants for GP, AMPA and glycine.

    Order of parameters / p0:
        [k_GP_loss, k_AMPA_form, k_AMPA_loss, k_Gly_form, k_Gly_loss]

    Parameters
    ----------
    use_constraints : bool
        Apply branching constraints:
        * k_GP_loss >= k_AMPA_form + k_Gly_form  (parent loss feeds both branches)
        * k_Gly_form >= k_AMPA_form              (glycine branch faster)
    ordering : bool
        If True (recommended, matches the paper's stated ordering), enforce
        k_GP_loss >= k_AMPA_loss >= k_Gly_loss. If False, fall back to the
        original heuristic k_AMPA_loss >= 1.3 * k_Gly_loss.
    fixed : dict, optional
        Hold specific constants fixed (by name) and fit only the rest. Use this
        to reproduce published values for weakly-identified constants, e.g.
        ``fixed={"k_AMPA_loss": 5.6e-4, "k_Gly_loss": 3.4e-4}``.

    Notes
    -----
    The byproduct *loss* constants (k_AMPA_loss, k_Gly_loss) are weakly
    identified by step+washout breakthrough data: washout is dominated by
    1/tau_pulse, so the optimum tends to push them to their upper bound. Prefer
    ``ordering=True`` and, when reproducing a paper, fix them via ``fixed``.

    Also computes the AMPA-pathway selectivity ``S_AMPA = [AMPA]/[Gly]`` at the
    end of the step phase and the parent half-life ``t_1/2 = ln2 / k_GP_loss``.
    """
    fixed = dict(fixed or {})
    names = GAG_PARAM_NAMES
    base = np.asarray(p0, dtype=float).copy()
    for nm, v in fixed.items():
        base[names.index(nm)] = v
    free_idx = [i for i, nm in enumerate(names) if nm not in fixed]

    def expand(free_vals):
        p = base.copy()
        p[free_idx] = free_vals
        return p

    t, s = _clean(df, cfg, ["GP", "AMPA", "Gly"])
    t_all = np.concatenate([t, t, t])
    y_all = np.concatenate([s["GP"], s["AMPA"], s["Gly"]])

    def predict_stacked(params, t_stacked):
        kgl, kaf, kal, kgf, kgls = params
        n = len(t_stacked) // 3
        tg, ta, ty = t_stacked[:n], t_stacked[n:2 * n], t_stacked[2 * n:]
        gp = models.gp_btc(tg, kgl, cfg)
        am = models.ampa_btc(ta, kaf, kal, cfg)
        gy = models.gly_btc(ty, kgf, kgls, cfg)
        return np.concatenate([gp, am, gy])

    def loss(free_vals):
        params = expand(free_vals)
        return float(np.sqrt(np.mean((y_all - predict_stacked(params, t_all)) ** 2)))

    constraints = []
    if use_constraints:
        constraints += [
            NonlinearConstraint(lambda f: expand(f)[0] - (expand(f)[1] + expand(f)[3]), 0, np.inf),
            NonlinearConstraint(lambda f: expand(f)[3] - expand(f)[1], 0, np.inf),
        ]
    if ordering:
        constraints += [
            NonlinearConstraint(lambda f: expand(f)[0] - expand(f)[2], 0, np.inf),
            NonlinearConstraint(lambda f: expand(f)[2] - expand(f)[4], 0, np.inf),
        ]
    else:
        constraints += [
            NonlinearConstraint(lambda f: expand(f)[2] - 1.3 * expand(f)[4], 0, np.inf),
        ]

    lb = [1e-12] * len(free_idx)
    ub = [1.0] * len(free_idx)
    res = minimize(
        loss, base[free_idx],
        method="trust-constr",
        bounds=Bounds(lb, ub),
        constraints=constraints,
        options={"maxiter": maxiter, "verbose": 0},
    )
    params = expand(res.x)

    # Approximate SD for free params from inverse-Hessian; fixed params get 0
    sd = np.zeros(5)
    try:
        H = res.hess_inv
        H = H.todense() if hasattr(H, "todense") else np.asarray(H)
        free_sd = np.sqrt(np.abs(np.diag(H)))
    except Exception:
        free_sd = np.full(len(free_idx), np.nan)
    for k, i in enumerate(free_idx):
        sd[i] = free_sd[k]

    # Goodness of fit on observed points
    y_hat = predict_stacked(params, t_all)
    rmse = float(np.sqrt(mean_squared_error(y_all, y_hat)))
    r2 = float(r2_score(y_all, y_hat))

    # Derived metrics at the end of the step phase
    ampa_ts = float(models.ampa_btc(np.array([cfg.t_s]), params[1], params[2], cfg)[0])
    gly_ts = float(models.gly_btc(np.array([cfg.t_s]), params[3], params[4], cfg)[0])
    s_ampa = ampa_ts / gly_ts if gly_ts > 0 else np.nan
    half_life_h = (np.log(2) / params[0]) / 60.0 if params[0] > 0 else np.nan

    def predict(t_pred):
        kgl, kaf, kal, kgf, kgls = params
        return {
            "GP": models.gp_btc(t_pred, kgl, cfg),
            "AMPA": models.ampa_btc(t_pred, kaf, kal, cfg),
            "Gly": models.gly_btc(t_pred, kgf, kgls, cfg),
        }

    return FitResult(
        species=["GP", "AMPA", "Gly"],
        param_names=["k_GP_loss", "k_AMPA_form", "k_AMPA_loss",
                     "k_Gly_form", "k_Gly_loss"],
        values=params, sd=sd, rmse=rmse, r2=r2, predict=predict,
        extra={"S_AMPA": s_ampa, "t_half_GP_h": half_life_h},
    )


# --------------------------------------------------------------------------- #
# Generic curve_fit wrapper (Pi, NH4, Mn)
# --------------------------------------------------------------------------- #
def fit_curve(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    species_key: str,
    model_fn: Callable,
    param_names: Sequence[str],
    p0: Sequence[float],
    bounds=(0.0, np.inf),
    maxfev: int = 20000,
) -> FitResult:
    """Fit a single-species model with ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    species_key : str
        Key into ``cfg.columns`` for the measured column (e.g. "Pi", "NH4", "Mn").
    model_fn : callable
        A ``curve_fit``-compatible f(t, *params), e.g. from ``models.make_*``.
    """
    t, s = _clean(df, cfg, [species_key])
    y = s[species_key]
    popt, pcov = curve_fit(model_fn, t, y, p0=list(p0), bounds=bounds, maxfev=maxfev)
    sd = np.sqrt(np.abs(np.diag(pcov))) if np.size(pcov) else np.full(len(popt), np.nan)
    y_hat = model_fn(t, *popt)
    rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
    r2 = float(r2_score(y, y_hat))
    return FitResult(
        species=[species_key],
        param_names=list(param_names),
        values=popt, sd=sd, rmse=rmse, r2=r2,
        predict=lambda t_pred: {species_key: model_fn(t_pred, *popt)},
    )


# --------------------------------------------------------------------------- #
# Single-run GLOBAL fit of all species (option A)
# --------------------------------------------------------------------------- #
# Free-parameter layout for fit_all_species. The "core" block is shared by GP,
# AMPA, Gly (and feeds the derived metrics + the Mn fit). Pi and NH4 keep their
# own effective amplitude constants so the closed-form results are preserved;
# k_Mn_ox uses the shared core k_GP_loss. This optimizes everything in ONE
# minimize() call rather than in separate per-species fits.
ALL_PARAM_NAMES = [
    "k_GP_loss", "k_AMPA_form", "k_AMPA_loss", "k_Gly_form", "k_Gly_loss",   # core
    "nh4_k_Gly_form", "nh4_k_Gly_loss", "nh4_k_AMPA_form",
    "nh4_k_AMPA_loss", "k_NH4_loss",                                          # NH4
    "pi_k_Gly_form", "pi_k_AMPA_form", "pi_k_GP_loss", "pi_k_AMPA_loss",      # Pi
    "k_Mn_ox",                                                                # Mn
]
_ALL_P0 = [1e-5, 6e-4, 5e-4, 1.3e-3, 3e-4,
           1e-2, 1e-2, 7e-3, 1e-2, 5e-3,
           1e-1, 1e-1, 2e-3, 1e-2,
           1e-2]


def fit_all_species(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    mndf: Optional[pd.DataFrame] = None,
    cfg_mn: Optional[ExperimentConfig] = None,
    p0: Sequence[float] = tuple(_ALL_P0),
    ordering: bool = True,
    use_constraints: bool = True,
    fixed: Optional[Dict[str, float]] = None,
    maxiter: int = 20000,
) -> FitResult:
    """Fit GP, AMPA, glycine, Pi, NH4 (and optionally soluble Mn) in ONE run.

    A single ``trust-constr`` optimization minimizes a per-species
    scale-normalized RMSE across all curves simultaneously. The GP/AMPA/Gly
    "core" constants are shared (and reused for the Mn model); Pi and NH4 keep
    their own effective amplitude constants, so results match the staged
    per-species fits while everything is solved in one call.

    Parameters
    ----------
    df : DataFrame
        Must contain GP, AMPA, Gly, Pi, NH4 columns (per ``cfg.columns``).
    mndf, cfg_mn : optional
        Soluble-Mn data and its config (often a different tau_pulse). If given,
        Mn is included in the joint objective via the fixed-GP dissolution model
        using the shared core ``k_GP_loss``.
    ordering, use_constraints, fixed :
        As in ``fit_gp_ampa_gly`` (constraints/ordering apply to the core block;
        ``fixed`` can pin any parameter by name from ``ALL_PARAM_NAMES``).

    Returns
    -------
    FitResult
        ``values`` follow ``ALL_PARAM_NAMES``; ``extra`` holds S_AMPA, the
        parent half-life, and per-species R2.
    """
    names = ALL_PARAM_NAMES
    fixed = dict(fixed or {})
    base = np.asarray(p0, float).copy()
    for nm, v in fixed.items():
        base[names.index(nm)] = v
    free_idx = [i for i, nm in enumerate(names) if nm not in fixed]

    def expand(free_vals):
        p = base.copy()
        p[free_idx] = free_vals
        return p

    # Observed data (per-species cleaning)
    t_g, sg = _clean(df, cfg, ["GP", "AMPA", "Gly"])
    t_pi, spi = _clean(df, cfg, ["Pi"])
    t_n, sn = _clean(df, cfg, ["NH4"])
    have_mn = mndf is not None
    if have_mn:
        cfg_mn = cfg_mn or cfg
        t_m, sm = _clean(mndf, cfg_mn, ["Mn"])

    def models_at(p, which):
        (kGP, kAf, kAl, kGf, kGl,
         nGf, nGl, nAf, nAl, kNH4,
         pGf, pAf, pGP, pAl, kMn) = p
        if which == "GP":
            return models.gp_btc(t_g, kGP, cfg)
        if which == "AMPA":
            return models.ampa_btc(t_g, kAf, kAl, cfg)
        if which == "Gly":
            return models.gly_btc(t_g, kGf, kGl, cfg)
        if which == "Pi":
            return models.pi_btc(t_pi, pGf, pAf, pGP, pAl, cfg)
        if which == "NH4":
            return models.nh4_btc(t_n, nGf, nGl, nAf, nAl, kNH4, cfg)
        if which == "Mn":
            return models.mnsoln_btc_fixed_gp(t_m, kMn, kGP, cfg_mn)

    # Per-species weights = mean of observed series (scale normalization)
    blocks = [("GP", sg["GP"]), ("AMPA", sg["AMPA"]), ("Gly", sg["Gly"]),
              ("Pi", spi["Pi"]), ("NH4", sn["NH4"])]
    if have_mn:
        blocks.append(("Mn", sm["Mn"]))
    weights = {k: max(np.mean(y), 1e-9) for k, y in blocks}

    PEN = 1e3  # soft-constraint penalty weight

    def residuals(free_vals):
        p = expand(free_vals)
        parts = [(y - models_at(p, k)) / weights[k] for k, y in blocks]
        # Soft ordering / branching penalties (active only when violated)
        kGP, kAf, kAl, kGf, kGl = p[0], p[1], p[2], p[3], p[4]
        pen = []
        if use_constraints:
            pen += [max(0.0, (kAf + kGf) - kGP), max(0.0, kAf - kGf)]
        if ordering:
            pen += [max(0.0, kAl - kGP), max(0.0, kGl - kAl)]
        if pen:
            parts.append(PEN * np.asarray(pen))
        return np.concatenate(parts)

    lb = np.array([1e-12] * len(free_idx))
    ub = np.array([2.0] * len(free_idx))
    x0 = np.clip(base[free_idx], lb, ub)
    sol = least_squares(residuals, x0, bounds=(lb, ub),
                        max_nfev=maxiter, method="trf")
    params = expand(sol.x)

    # Per-species R2 + overall metrics
    per_r2 = {}
    for k, y in blocks:
        yh = models_at(params, k)
        per_r2[f"R2_{k}"] = float(r2_score(y, yh))
    ampa_ts = float(models.ampa_btc(np.array([cfg.t_s]), params[1], params[2], cfg)[0])
    gly_ts = float(models.gly_btc(np.array([cfg.t_s]), params[3], params[4], cfg)[0])
    s_ampa = ampa_ts / gly_ts if gly_ts > 0 else np.nan
    half_life_h = (np.log(2) / params[0]) / 60.0 if params[0] > 0 else np.nan

    def predict(t_pred):
        (kGP, kAf, kAl, kGf, kGl, nGf, nGl, nAf, nAl, kNH4,
         pGf, pAf, pGP, pAl, kMn) = params
        out = {"GP": models.gp_btc(t_pred, kGP, cfg),
               "AMPA": models.ampa_btc(t_pred, kAf, kAl, cfg),
               "Gly": models.gly_btc(t_pred, kGf, kGl, cfg),
               "Pi": models.pi_btc(t_pred, pGf, pAf, pGP, pAl, cfg),
               "NH4": models.nh4_btc(t_pred, nGf, nGl, nAf, nAl, kNH4, cfg)}
        if have_mn:
            out["Mn"] = models.mnsoln_btc_fixed_gp(t_pred, kMn, kGP, cfg_mn)
        return out

    extra = {"S_AMPA": s_ampa, "t_half_GP_h": half_life_h}
    extra.update(per_r2)
    overall_rmse = float(np.sqrt(np.mean(np.concatenate(
        [(y - models_at(params, k)) for k, y in blocks]) ** 2)))
    return FitResult(
        species=[k for k, _ in blocks],
        param_names=names,
        values=params, sd=np.full(len(names), np.nan),
        rmse=overall_rmse, r2=float(np.mean(list(per_r2.values()))),
        predict=predict, extra=extra,
    )


# --------------------------------------------------------------------------- #
# Fully-coupled NUMERICAL ODE global fit (independent per-species time axes)
# --------------------------------------------------------------------------- #
def fit_global_ode(
    data: Dict[str, "tuple"],
    cfg: ExperimentConfig,
    p0: Sequence[float] = (1.9e-3, 6e-4, 6e-4, 1.3e-3, 4e-4, 3e-3, 1.3e-2),
    ordering: bool = False,
    use_constraints: bool = False,
    species_weights: Optional[Dict[str, float]] = None,
    maxiter: int = 6000,
) -> FitResult:
    """Fully-coupled fit of all species by numerical ODE integration.

    Every species shares ONE parameter set (``ode_model.ODE_PARAM_NAMES``); Pi,
    NH4 and Mn follow from the integrated trajectories of the parent system, so
    no separate effective constants are introduced. The coupled ODEs are solved
    once per optimizer evaluation and compared to each species' data.

    Parameters
    ----------
    data : dict
        Maps species name (subset of ``ode_model.SPECIES``: "GP", "AMPA", "Gly",
        "Pi", "NH4", "Mn") to a ``(t, y)`` pair of 1-D arrays. **Each species may
        have its own, independent time axis** - they need not share sampling
        times or lengths.
    cfg : ExperimentConfig
        C0, t_s, N, tau_step, tau_pulse for the (single, shared) reactor.
    ordering, use_constraints : bool
        Impose k_GP_loss >= k_AMPA_loss >= k_Gly_loss and the branching relations
        as soft penalties.
    species_weights : dict, optional
        Per-species multipliers on the (RMS-normalized, equal-block) residuals,
        to prioritize harder/under-fit species. Default weight 1.0 each.

    Returns
    -------
    FitResult
        ``values`` follow ``ode_model.ODE_PARAM_NAMES``; ``extra`` carries
        S_AMPA, the parent half-life, and per-species R2.

    Notes
    -----
    Full coupling shares ``k_Gly_loss`` (glycine loss) and ``k_Gly_form``
    between glycine, Pi and NH4. When the glycine and NH4 trajectories pull that
    constant in different directions (glycine still rising while NH4 has
    plateaued), no single shared value fits both perfectly - this is the
    intrinsic cost of full coupling versus per-species effective constants.
    """
    from . import ode_model

    sw = dict(species_weights or {})
    # Clean each species' (t, y): keep finite, non-negative points
    clean = {}
    for sp, (t, y) in data.items():
        t = np.asarray(t, float); y = np.asarray(y, float)
        m = np.isfinite(t) & np.isfinite(y) & (y >= 0)
        clean[sp] = (t[m], y[m])
    # RMS scale per species (so each curve is dimensionless ~O(1))
    scale = {sp: max(np.sqrt(np.mean(y ** 2)), 1e-9) for sp, (t, y) in clean.items()}

    names = ode_model.ODE_PARAM_NAMES
    PEN = 1e4

    def residuals(k):
        ev = ode_model.build_effluent(k, cfg)
        parts = []
        for sp, (t, y) in clean.items():
            r = (y - ev(sp, t)) / scale[sp]            # RMS-normalized, O(1)
            parts.append(r * sw.get(sp, 1.0))
        kGP, kAf, kAl, kGf, kGl, kNH4, kMn = k
        pen = []
        if use_constraints:
            pen += [max(0.0, (kAf + kGf) - kGP), max(0.0, kAf - kGf)]
        if ordering:
            pen += [max(0.0, kAl - kGP), max(0.0, kGl - kAl)]
        if pen:
            parts.append(PEN * np.asarray(pen))
        return np.concatenate(parts)

    lb = np.array([1e-9] * 7)
    ub = np.array([1.0] * 7)
    x0 = np.clip(np.asarray(p0, float), lb, ub)
    sol = least_squares(residuals, x0, bounds=(lb, ub),
                        max_nfev=maxiter, method="trf")
    params = sol.x

    ev = ode_model.build_effluent(params, cfg)
    per_r2 = {}
    for sp, (t, y) in clean.items():
        per_r2[f"R2_{sp}"] = float(r2_score(y, ev(sp, t)))
    # Derived metrics from the integrated solution at t_s
    ats = float(ev("AMPA", cfg.t_s)); gts = float(ev("Gly", cfg.t_s))
    s_ampa = ats / gts if gts > 0 else np.nan
    half_life_h = (np.log(2) / params[0]) / 60.0 if params[0] > 0 else np.nan

    def predict(t_pred):
        e = ode_model.build_effluent(params, cfg)
        return {sp: e(sp, t_pred) for sp in ode_model.SPECIES}

    overall = []
    for sp, (t, y) in clean.items():
        overall.append(y - ev(sp, t))
    rmse = float(np.sqrt(np.mean(np.concatenate(overall) ** 2)))

    extra = {"S_AMPA": s_ampa, "t_half_GP_h": half_life_h}
    extra.update(per_r2)
    return FitResult(
        species=list(clean.keys()),
        param_names=names,
        values=params, sd=np.full(len(names), np.nan),
        rmse=rmse, r2=float(np.mean(list(per_r2.values()))),
        predict=predict, extra=extra,
    )
