# Model Description

This document describes the mathematics, assumptions, and numerical strategy of the
reactive-transport model as actually implemented. It is self-contained so a user can
re-implement or extend it for their own system.

## 1. Overview

The model predicts the **effluent breakthrough curve** of each measured species from a
flow-through reactor and fits **apparent first-order rate constants** to data. Two
interchangeable engines are provided (Section 5):

- a **closed-form (semi-analytical)** engine, in which each species' effluent is written as a
  chemistry term multiplied by a residence-time-distribution (RTD) term — fast, avoids
  stiffness, and reproduces the paper's staged fits, but requires an analytic solution per
  rate law and fits Pi/NH₄ with their own effective constants;
- a **fully-coupled numerical ODE** engine, which integrates the coupled rate laws with
  `scipy.integrate.solve_ivp` under one shared parameter set (no per-species effective
  constants; species may use independent time axes).

Both share the same transport treatment and the same tracer-derived, fixed hydraulics.

The two operating phases are:

- **Phase 1 — step input** (`0 ≤ t ≤ t_s`): constant influent of the parent compound
  (`C0`, e.g. 300 µM at `t_s = 160 min` in the paper).
- **Phase 2 — washout / pulse** (`t > t_s`): influent replaced by background solution;
  the effluent decays according to the reactor RTD and any first-order loss.

## 2. Transport: gamma residence-time distribution

A real reactor is not an ideal plug-flow reactor, so axial dispersion is described with a
**gamma RTD**. The cumulative RTD (fraction of a conservative step signal that has broken
through by time `t`) is

```
gamma_rtd(t, tau, N) = 1 − Q(N, t/tau)
```

where `Q` is the regularized upper incomplete gamma function (`scipy.special.gammaincc`),
so `1 − Q` is the regularized lower incomplete gamma, i.e. the gamma CDF. `N` is the shape
parameter (larger → sharper, more plug-flow-like) and `tau` the gamma scale (mean hydraulic
residence = `N·tau`).

**Hydraulics are measured, not fit alongside chemistry.** `bromide.fit_tracer` fits this RTD
to a conservative Br⁻ step+pulse test (SI Text S3, eqs S1–S5): step phase
`C0·[1 − Q(N, t/tau)]`, washout `C(t_s)·exp(−(t−t_s)/tau_pulse)`. On the bundled tracer it
returns N=4, tau≈0.65 min (mean ≈ 2.6 min), tau_pulse≈4.1 min, R²≈0.99. These hydraulics are
then held fixed in every reactive fit.

`ExperimentConfig` carries all fixed/operating parameters and reads/writes a CSV
(`from_csv`/`to_csv`), so a different reactor or schedule is just a different file: `C0`, `t_s`
(experiment) and `N`, `tau_step`, `tau_pulse` (tracer), plus a `retardation` factor. The model
uses `tau_step_eff = tau_step · retardation`: a conservative tracer is not retarded, but
adsorbing glyphosate breaks through ~2.5× later, so `retardation ≈ 2.5` (→ `tau_step_eff ≈
1.6 min`) reproduces the parent breakthrough while N and tau_pulse stay exactly as measured.
The paper reports a mean residence τ ≈ 1.58 min.

## 3. Step + pulse assembly

For a species whose Phase-1 effluent is `f(t)`, the full curve is

```
C(t) = f(t)                                        for t ≤ t_s
C(t) = f(t_s) · exp(−r · (t − t_s))                for t > t_s
```

where `r` is the washout rate, typically `1/tau_pulse` plus any first-order loss of the
species. This is `transport.step_pulse_profile`.

## 4. Reaction expressions (the published glyphosate network)

Let `RTD(t) = gamma_rtd(t, tau_step, N)`. The Phase-1 expressions are:

| Species | Phase-1 effluent `f(t)` | Washout rate `r` |
|---|---|---|
| Parent (GP) | `C0 · RTD(t) · exp(−k_GP_loss·t)` | `1/tau_pulse + k_GP_loss` |
| AMPA | `C0 · (k_AMPA_form/k_AMPA_loss) · RTD(t) · (1 − exp(−k_AMPA_loss·t))` | `1/tau_pulse + k_AMPA_loss` |
| Glycine | `C0 · (k_Gly_form/k_Gly_loss) · RTD(t) · (1 − exp(−k_Gly_loss·t))` | `1/tau_pulse + k_Gly_loss` |
| Orthophosphate (Pi) | `C0 · [k_AMPA_form·(1−exp(−k_AMPA_loss·t)) + k_Gly_form·exp(−k_GP_loss·t)] · RTD(t)` | `1/tau_pulse` |
| Ammonium (NH₄⁺) | `C0 · [k_Gly_form·(1−exp(−k_Gly_loss·t)) + k_AMPA_form·(1−exp(−k_AMPA_loss·t))] · RTD(t)` | `1/tau_pulse + k_NH4_loss` |
| Soluble Mn | `C0 · RTD(t) · (1 − exp(−k_diss·t))` | `1/tau_pulse` |
| Soluble Mn (fixed GP) | `C0 · (k_GP_loss/k_Mn_ox) · RTD(t) · (1 − exp(−k_Mn_ox·t))` | `1/tau_pulse + k_Mn_ox` |

**Important about Pi and NH₄⁺:** in the original notebook these sub-models are fit
**independently**, with their own free amplitude constants. Although the parameter names
overlap with the global GP/AMPA/Gly fit, they are *not* the same quantities — they are
effective fitted amplitudes specific to the Pi or NH₄⁺ curve. Treat them accordingly.

## 5. Parameter estimation

### Global constrained fit (GP + AMPA + glycine)

Five constants are fit simultaneously by minimizing the pooled RMSE across the three
curves (`scipy.optimize.minimize`, method `trust-constr`), subject to constraints:

- `k_GP_loss ≥ k_AMPA_form + k_Gly_form` (parent loss feeds both branches)
- `k_Gly_form ≥ k_AMPA_form` (glycine branch faster — consistent with the paper)
- with `ordering=True` (default): `k_GP_loss ≥ k_AMPA_loss ≥ k_Gly_loss`
  (the loss ordering stated in the paper); otherwise the original heuristic
  `k_AMPA_loss ≥ 1.3 · k_Gly_loss` is used.

Any constant can be **held fixed** via `fixed={...}` and only the rest fit — the
recommended way to reproduce published values for weakly-identified constants.

**Identifiability.** The byproduct loss constants `k_AMPA_loss` and `k_Gly_loss` are only
weakly constrained by step+washout breakthrough data. During the step phase the byproduct
curves are near-linear when `k_loss · t_s` is small, and the washout decay is dominated by
`1/tau_pulse` (≈ 0.27 min⁻¹ here), both of which are insensitive to `k_loss`. On the
included real data, fixing these constants to the published Table 2 values versus letting
them float changes R² by < 0.001 — so report them with the published values fixed, and
interpret the floated values as upper-bound-limited rather than determined.

### Independent fits (Pi, NH₄⁺, soluble Mn)

Each is fit with `scipy.optimize.curve_fit` (non-negative bounds), with parameter SDs
from the covariance matrix.

### Single-run global fits

Two single-run options exist:

* `fit_all_species` (closed-form): minimizes one scale-normalized residual vector across all
  species using the closed-form models, where Pi/NH4 retain their own effective amplitude
  constants. Fast, reproduces the staged numbers.
* `fit_global_ode` (numerical, **fully coupled**): integrates the coupled reaction ODEs and
  shares one parameter set across every species — see the ODE section below.

### Fully-coupled numerical ODE engine (`ode_model.py`)

The reaction network is integrated directly with `scipy.integrate.solve_ivp`:

```
d[GP]/dt   = -kGP*[GP]
d[AMPA]/dt =  kAf*[GP] - kAl*[AMPA]
d[Gly]/dt  =  kGf*[GP] - kGl*[Gly]
d[Pi]/dt   =  kGf*[GP] + kAl*[AMPA]            (SI eq S34)
d[NH4]/dt  =  kGl*[Gly] + kAl*[AMPA] - kNH4*[NH4]   (SI eq S39)
d[Mn]/dt   =  kGP*[GP] - kMn*[Mn]              (eq 9)
```

with the shared parameter set k = [kGP, kAf, kAl, kGf, kGl, kNH4, kMn]. Transport is applied
the same way as the closed forms: during the step phase the integrated solution is scaled by
the cumulative gamma RTD; during washout each species decays from its t_s value at
(1/tau_pulse + its own first-order loss). Because the system is integrated over the full
experiment, byproduct accumulation (e.g. Pi ≈ ∫kGf·[GP] dt) emerges from the same constants
that fit GP/AMPA/Gly — no separate effective constants are needed. This is the most faithful
realization of the SI derivation, and each species may be supplied on its own time axis.

**Intrinsic limitations of full coupling** (observed on the real data, both physical):

- Pi has no sink in eq S34, so integrated phosphate grows ~linearly while measured Pi
  saturates (phosphate adsorbs on the Mn-oxide surface). A shared `k_Pi_loss` term would
  fix this if one chooses to extend the network.
- NH4 shares `k_Gly_loss` with glycine; the glycine trajectory (still rising) and the NH4
  trajectory (plateaued) pull that constant in opposite directions, so no single shared
  value fits both. Typical fit quality: GP/AMPA/Gly/Mn R² ≈ 0.93–0.99, Pi ≈ 0.6, NH4 ≈ 0.4.

## 6. Derived metrics

- **AMPA-pathway selectivity:** `S_AMPA = [AMPA]_{t_s} / [Glycine]_{t_s}` evaluated from
  the fitted curves at the end of the step phase. Higher `S_AMPA` → greater favorability
  for the AMPA (C(2)–N cleavage) pathway over the glycine/sarcosine pathways.
- **Parent half-life:** `t½ = ln2 / k_GP_loss` (reported in hours by the code).
- **Net reductive dissolution** of Mn oxide is approximately `k_diss ≈ k_Mn_ox / k_GP_loss`
  (dissolution is stoichiometrically coupled to oxidation).

## 7. Assumptions and limitations

- Apparent constants lump adsorption, desorption, and electron transfer; they are
  condition-specific, not intrinsic.
- Short-lived intermediates (sarcosine, MPA, methylamine, adsorbed GP/AMPA/Pi) are assumed
  not to accumulate.
- Constant volumetric flow (Qin = Qout); negligible molecular diffusion across boundaries.
- Some amplitude constants in the Pi / NH₄⁺ sub-models are weakly identified or correlated;
  the curve fit can be excellent (high R²) even when individual constants are uncertain.
- Global-fit parameter SDs are approximate; `trust-constr` may not expose a usable inverse
  Hessian (SDs can return `NaN`). Use bootstrap or MCMC for rigorous uncertainty.

## 8. Extending to a new system — checklist

1. Choose an engine. **Closed-form:** derive (or numerically obtain) the Phase-1 effluent
   expression `f(t)` for each species and add a function in `models.py` that plugs it into
   `transport.step_pulse_profile`. **Coupled ODE:** instead, edit `ode_model.reaction_rhs`
   (and `SPECIES` / `ODE_PARAM_NAMES`) with your coupled rate laws — no analytic solution
   needed.
2. Characterize your reactor RTD with a tracer test; set `tau_step`, `tau_pulse`, `N` in
   `ExperimentConfig`.
3. Define your influent regime (`C0`, `t_s`).
4. Provide effluent data (one CSV column per species + `time`); map headers in
   `ExperimentConfig(columns=...)`.
5. Fit with `fit_curve` (single species), a custom global objective modeled on
   `fit_gp_ampa_gly` / `fit_all_species` (closed-form), or `fit_global_ode` (coupled ODE).
6. Compute any derived metrics relevant to your problem.

## 9. Reproducibility

- Pin library versions (`pip freeze` → `requirements.txt`).
- Record the tracer-derived RTD parameters used for every fit.
- Report optimizer, bounds, initial guesses, constraints, and convergence settings.
