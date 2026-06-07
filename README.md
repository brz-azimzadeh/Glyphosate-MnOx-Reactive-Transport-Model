# Reactive-Transport Model for Coupled Adsorption–Oxidation at Mn-Oxide Interfaces

A Python toolkit (originally developed in Google Colab) that recovers **apparent
reaction-rate constants** and **pathway-selectivity metrics** from effluent breakthrough
curves of a flow-through (microfluidic) reactor. It couples first-order surface reaction
kinetics with nonideal solute transport (a gamma residence-time distribution), and offers
two interchangeable solvers: a **closed-form (semi-analytical)** engine and a
**fully-coupled numerical-ODE** engine.

Developed for, and validated against, the experiments in:

> Azimzadeh, B.; Martínez, C. E. *Chemical Choreography at the Mn Oxide Interface:
> Protein Association Modulates Glyphosate Bonding Configurations and Favors the AMPA
> Oxidation Pathway.* **Environ. Sci. Technol.** 2025, 59 (36), 19513–19525.
> DOI: [10.1021/acs.est.5c09711](https://doi.org/10.1021/acs.est.5c09711)

The published study tracks **glyphosate** oxidation on **Mn oxides** (K-birnessite,
hausmannite) with and without an adsorbed protein, but the code is modular: the reaction
network, reactor hydraulics, and input regime are user-defined, so the same engine applies
to **other contaminants, minerals, and flow reactors**.

---

## How the model works

For every species, the predicted reactor effluent combines a **chemistry** term with a
**transport** term. Transport is the same throughout: axial dispersion is captured with a
gamma residence-time distribution (RTD), whose shape `N` and time constants come from a
conservative Br⁻ tracer test and are held **fixed** (see *Hydraulic calibration* below).
During the step phase (`t ≤ t_s`) the reacted concentration is multiplied by the RTD ramp;
during washout (`t > t_s`) it decays exponentially.

Two fitting engines build the chemistry term — **keep whichever suits your goal**:

| | Closed-form engine | Fully-coupled ODE engine |
|---|---|---|
| Functions | `fit_gp_ampa_gly`, `fit_curve`, `fit_all_species` | `fit_global_ode` |
| Chemistry | analytic per-species breakthrough expressions (`models.py`) | numerically integrated coupled ODEs (`ode_model.py`, `solve_ivp`) |
| Parameters | GP/AMPA/Gly share 5 constants; **Pi, NH₄ get their own effective constants** | **one shared 7-constant set**; Pi, NH₄, Mn derived from the parent trajectories |
| Per-species time axes | must share a grid (merged) | independent grids allowed |
| Best for | reproducing the paper's Table 2; speed | a single, mechanistically coupled model |
| Typical R² | GP/AMPA/Gly ≈ 0.99; Pi ≈ 0.95; NH₄ ≈ 0.94 | GP 0.99, AMPA 0.97, Gly 0.93, Mn 0.97; Pi 0.60, NH₄ 0.44 |

They give slightly different numbers (k_GP_loss 1.92e-3 vs 2.17e-3; t½ 6.0 vs 5.4 h; S_AMPA
0.35 vs 0.30). The closed-form engine is the paper's own staged method, and the extra freedom
in its Pi/NH₄ sub-models is what lets those byproducts reach high R². The ODE engine is the
stricter, fully-coupled realization of the SI derivation: one parameter set drives everything,
which is more rigorous but cannot bend Pi/NH₄ to their own data (a physically meaningful
limitation — see the ODE section). Neither supersedes the other; they answer different
questions, and both read the same tracer-derived configuration.

**Outputs (both engines):** fitted constants (with uncertainties where available),
AMPA-pathway selectivity `S_AMPA = [AMPA]/[Glycine]` at the end of the step phase, parent
half-life `t½ = ln2 / k_GP_loss`, per-species R², and observed-vs-predicted figures.

Full equations and modeling notes are in [`docs/MODEL_DESCRIPTION.md`](docs/MODEL_DESCRIPTION.md).

## Validation against the paper

Running the **closed-form** engine on the included real data (Bir, pH 4.6, no BSA)
reproduces the well-identified quantities in Table 2 of the paper almost exactly:

| Quantity | This code | Paper (Table 2) |
|---|---|---|
| k_GP_loss | 1.92e-3 min⁻¹ | 1.93e-3 min⁻¹ |
| k_Mn_ox | 1.37e-2 min⁻¹ | 1.38e-2 min⁻¹ |
| S_AMPA (selectivity) | 0.35 | 0.33 |
| Parent half-life | 6.0 h | 6 h |
| Fit quality (GP/AMPA/Gly) | R² ≈ 0.996 | — |

The byproduct **loss** constants (k_AMPA_loss, k_Gly_loss) are *weakly identified*
by step+washout data: fixing them to the published values versus letting them float
changes R² by < 0.001. They are therefore best reported with the published values
held fixed (see `fixed=` below), and interpreted with that caveat.

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py        # ExperimentConfig: CSV-driven reactor/experiment parameters
│   ├── bromide.py       # Br- tracer fit -> fixed reactor hydraulics (N, tau, tau_pulse)
│   ├── transport.py     # gamma RTD + step/pulse profile assembly
│   ├── models.py        # closed-form breakthrough models per species
│   ├── ode_model.py     # fully-coupled numerical ODE reaction-transport engine
│   ├── fit.py           # staged closed-form fits + coupled ODE global fit
│   └── plotting.py      # observed-vs-predicted figures
├── examples/
│   ├── run_fit.py                 # staged closed-form fits (reproduces the paper)
│   ├── run_global_ode.py          # fully-coupled ODE fit of ALL species at once
│   └── run_tracer_calibration.py  # Br- tracer -> config CSV -> reactive fit
├── data/                # one independent CSV per species (own time axis)
│   ├── GP.csv  AMPA.csv  Glycine.csv  Pi.csv  NH4.csv  Mnsoln.csv
│   ├── Br_tracer.csv               # conservative Br- tracer breakthrough
│   └── experiment_config.csv       # fixed parameters (hydraulics + conditions)
├── notebooks/
│   └── reactive_transport.ipynb   # Colab-style notebook using the modules
└── docs/
    └── MODEL_DESCRIPTION.md
```

---

## Installation

```bash
git clone https://github.com/brz-azimzadeh/Glyphosate-MnOx-Reactive-Transport-Model.git
cd Glyphosate-MnOx-Reactive-Transport-Model
pip install -r requirements.txt
```

Or open `notebooks/reactive_transport.ipynb` in
[Google Colab](https://colab.research.google.com/) — no local install needed.

---

## Quick start

```python
import pandas as pd
from src import ExperimentConfig, fit_gp_ampa_gly, fit_curve, models, plotting

# 1. Load data. Species live in separate CSVs (each with its own time axis);
#    the staged closed-form fits below merge the co-gridded ones.
import functools
def _read(f, col):
    d = pd.read_csv(f, encoding="utf-8-sig")
    return d[["Time (min)", col]]
df = functools.reduce(lambda a, b: pd.merge(a, b, on="Time (min)", how="outer"),
                      [_read("data/GP.csv","GP (uM)"), _read("data/AMPA.csv","AMPA (uM)"),
                       _read("data/Glycine.csv","Glycine (uM)"), _read("data/Pi.csv","Pi (uM)"),
                       _read("data/NH4.csv","NH4+ (uM)")])

# 2. Describe the reactor / experiment (tau values from your tracer test)
cfg = ExperimentConfig(C0=300, t_s=160, N=4, tau_step=1.6, tau_pulse=3.7)

# 3. Constrained global fit: parent + AMPA + glycine
#    Default enforces k_GP_loss >= k_AMPA_loss >= k_Gly_loss (paper ordering).
result = fit_gp_ampa_gly(df, cfg)
print(result.summary())          # constants, RMSE/R2, S_AMPA, half-life

# 3b. To reproduce published values for weakly-identified loss constants,
#     fix them and fit the rest:
# result = fit_gp_ampa_gly(df, cfg,
#                          fixed={"k_AMPA_loss": 0.56e-3, "k_Gly_loss": 0.34e-3})

# 4. Orthophosphate, ammonium, soluble Mn (each fit independently)
pi  = fit_curve(df, cfg, "Pi",  models.make_pi_model(cfg),
                ["k_Gly_form","k_AMPA_form","k_GP_loss","k_AMPA_loss"],
                p0=[1e-1, 1e-1, 2e-3, 1e-2])
mn  = fit_curve(pd.read_csv("data/Mnsoln.csv", encoding="utf-8-sig"),
                ExperimentConfig(tau_pulse=4.6), "Mn",
                models.make_mnsoln_fixed_gp_model(ExperimentConfig(tau_pulse=4.6),
                                                  k_gp_loss=result.values[0]),
                ["k_Mn_ox"], p0=[1e-2])
```

The complete, runnable version is in [`examples/run_fit.py`](examples/run_fit.py):

```bash
python examples/run_fit.py
```

It writes fitted constants and figures to an `output/` folder.

---

## Adapting the model to your own system

Three things are user-defined and independent:

1. **Your data** (`data/*.csv`): a `time` column plus one column per measured species.
   Map your column headers in `ExperimentConfig(columns=...)`.

2. **Your reactor hydraulics** (`ExperimentConfig`): run a conservative tracer test, fit
   the gamma RTD, and pass `tau_step`, `tau_pulse`, and shape `N`. For near-ideal flow,
   use a large `N`.

3. **Your reaction network** — two ways, matching the two engines:
   - *Closed-form* (`src/models.py`): write a breakthrough expression for each species and
     rate law, following the existing functions as templates (each plugs a step-phase
     expression into `transport.step_pulse_profile`), then fit with `fit_curve` (single
     species), `fit_gp_ampa_gly`, or `fit_all_species`.
   - *Coupled ODE* (`src/ode_model.py`): edit `reaction_rhs` to your own coupled rate laws
     (and `SPECIES` / `ODE_PARAM_NAMES`), then fit everything at once with `fit_global_ode`.
     No analytic solution needed.

The transport, configuration, fitting, and plotting layers do not change.

## Hydraulic calibration (Br⁻ tracer) and fixed-parameter config

The reactor hydraulics are not fit alongside the chemistry — they are measured
independently with a conservative Br⁻ tracer test and then held **fixed**. `src/bromide.py`
implements the tracer transport model of SI Text S3 (gamma RTD, eqs S1–S5) and fits it:

```python
import pandas as pd
from src import fit_tracer, ExperimentConfig

br = pd.read_csv("data/Br_tracer.csv", encoding="utf-8-sig")
hyd = fit_tracer(br["Time (min)"], br["[Br] mM"], C0=1.0, t_s=74.0, fix_N=4)
print(hyd.summary())   # tau, N, tau_pulse, mean residence, R^2 (~0.99 here)
```

Those hydraulics plus the experiment conditions (influent `C0`, step duration `t_s`) become
an `ExperimentConfig`, which is saved to / loaded from a **CSV of fixed parameters**:

```python
cfg = ExperimentConfig.from_tracer(hyd, C0=300, t_s=160, retardation=2.5)
cfg.to_csv("data/experiment_config.csv")        # editable fixed-parameter file
cfg = ExperimentConfig.from_csv("data/experiment_config.csv")
```

`data/experiment_config.csv` (key/value with units + provenance):

| parameter | value | unit | source | meaning |
|-----------|-------|------|--------|---------|
| C0 | 300 | uM | experiment | parent influent during step |
| t_s | 160 | min | experiment | step → washout switch time |
| N | 4 | – | Br tracer | gamma RTD shape (tanks-in-series) |
| tau_step | 0.646 | min | Br tracer | gamma scale (mean hydraulic residence = N·tau_step) |
| tau_pulse | 4.13 | min | Br tracer | washout time constant |
| retardation | 2.5 | – | reactive | step-residence multiplier for adsorbing species |

**Retardation.** A conservative tracer is not retarded, but glyphosate adsorbs and breaks
through ~2.5× later, so the model uses `tau_step_eff = tau_step × retardation`. With the
tracer hydraulics fixed, `retardation = 2.5` (→ `tau_step_eff ≈ 1.6 min`) recovers GP R² ≈ 0.99;
set `retardation = 1.0` for a non-adsorbing solute. This is the only hydraulic-side knob the
reactive data informs; N and tau_pulse stay exactly as measured.

**Different reactor / schedule.** Supply your own tracer CSV and edit the config CSV
(`C0`, `t_s`, `retardation`, …). Nothing in the model code is hard-coded — the full workflow
is in `examples/run_tracer_calibration.py`.

## Fully-coupled ODE global fit (all species, independent time axes)

`fit_global_ode` (in `src/ode_model.py` + `src/fit.py`, example `run_global_ode.py`)
integrates the coupled kinetic system numerically (SI Text S4: eqs S21/S26/S31/S34/S39,
plus Mn eq 9) with `scipy.integrate.solve_ivp` and applies the gamma RTD for transport.
Every species shares **one** parameter set — Pi follows from the integrated `[GP]` and
`[AMPA]`, NH₄⁺ from `[Gly]` and `[AMPA]`, Mn from `[GP]` — so there are no per-species
effective constants and no fixed values.

Each species is supplied from its **own CSV with its own time axis** (they need not share
sampling times or lengths):

```python
import pandas as pd
from src import ExperimentConfig, fit_global_ode

def load(path, col):
    d = pd.read_csv(path, encoding="utf-8-sig")
    return d["Time (min)"].to_numpy(float), d[col].to_numpy(float)

data = {
    "GP":   load("data/GP.csv",      "GP (uM)"),
    "AMPA": load("data/AMPA.csv",    "AMPA (uM)"),
    "Gly":  load("data/Glycine.csv", "Glycine (uM)"),
    "Pi":   load("data/Pi.csv",      "Pi (uM)"),
    "NH4":  load("data/NH4.csv",     "NH4+ (uM)"),
    "Mn":   load("data/Mnsoln.csv",  "Mnsoln (uM)"),
}
cfg = ExperimentConfig.from_csv("data/experiment_config.csv")  # tracer-fixed hydraulics
result = fit_global_ode(data, cfg)     # one shared parameter set, fully coupled
print(result.summary())
```

On the included real data this gives k_GP_loss ≈ 2.2e-3 (t½ ≈ 5.4 h), k_Mn_ox ≈ 1.2e-2,
S_AMPA ≈ 0.30, with per-species R²: **GP 0.99, AMPA 0.97, Gly 0.93, Mn 0.97**, and
**Pi ≈ 0.60, NH₄ ≈ 0.44**.

Two honest limitations of *full* coupling (both physically meaningful):

- **Pi** has no sink in eq S34, so the integrated phosphate climbs ~linearly while the
  measured Pi saturates — consistent with phosphate adsorbing onto the Mn-oxide surface
  (noted in the paper). Adding a shared `k_Pi_loss` term would address this if desired.
- **NH₄⁺** shares `k_Gly_loss` with glycine. Glycine keeps rising (wanting small loss)
  while NH₄⁺ plateaus (wanting glycine's production to level off); one shared constant
  cannot satisfy both. Independent effective constants previously hid this tension.

`species_weights={...}` lets you prioritize Pi/NH₄ at the cost of the others, but it cannot
remove the structural conflict above. This is the trade-off you accepted in choosing a
fully-coupled model over per-species constants.

---

## Notes, assumptions, and limitations

These are stated plainly so others can use the results responsibly:

- **Apparent constants.** All rate constants lump the underlying adsorption, desorption,
  and electron-transfer steps; they are effective values for the given conditions (pH,
  ionic strength, surface loading), not intrinsic mechanistic constants.
- **Transient intermediates** (sarcosine, MPA, methylamine, adsorbed species) are assumed
  not to accumulate.
- **Pi and NH₄⁺ — closed-form engine.** Their sub-models fit their **own effective
  amplitude constants** independently of the global GP/AMPA/Gly fit (the shared parameter
  names are not the same quantities). With strong signal these reproduce the curves well
  (high R²), but individual amplitude constants can be weakly identified / correlated —
  interpret them as effective fitted values.
- **Pi and NH₄⁺ — ODE engine.** Under full coupling these are derived from the shared
  constants, so they fit less well (Pi ≈ 0.60, NH₄ ≈ 0.44): Pi lacks a sink in eq S34
  (real phosphate adsorbs), and NH₄ shares `k_Gly_loss` with glycine, which the two curves
  pull in opposite directions. This is an honest, physical consequence of one shared set.
- **Parameter uncertainties** for the global fit are approximate (the `trust-constr`
  optimizer does not always expose a usable inverse Hessian, so some SDs return `NaN`).
  For rigorous uncertainties, consider a bootstrap or an MCMC re-fit.
- **NH₄⁺ washout anchoring** in `models.nh4_btc` reproduces the original notebook exactly;
  a commented alternative shows the standard anchoring for new systems.
- **Parameter ordering and fixing.** `fit_gp_ampa_gly` defaults to `ordering=True`
  (k_GP_loss ≥ k_AMPA_loss ≥ k_Gly_loss, matching the paper). Pass `fixed={...}` to hold
  weakly-identified constants at chosen/published values and fit the rest.

---

## Citing this work

Please cite both the paper and the software (see [`CITATION.cff`](CITATION.cff)):

> Azimzadeh, B.; Martínez, C. E. Chemical Choreography at the Mn Oxide Interface:
> Protein Association Modulates Glyphosate Bonding Configurations and Favors the AMPA
> Oxidation Pathway. *Environ. Sci. Technol.* 2025, 59 (36), 19513–19525.
> DOI: 10.1021/acs.est.5c09711

---

## License

Released under the MIT License (see `LICENSE`). **Before publishing, confirm that MIT is
compatible with any Cornell University and NSF/USDA grant requirements for code produced
under your awards**, and update `LICENSE` and `CITATION.cff` if you choose differently.

## Data

The CSV files under `data/` are real breakthrough data for the Bir / pH 4.6 / no-BSA
condition, included so the example is fully reproducible.

## Contact

Behrooz Azimzadeh (ba324@cornell.edu) and Carmen Enid Martínez (corresponding author;
cem20@cornell.edu)
Soil and Crop Sciences, School of Integrative Plant Science, Cornell University

## Acknowledgments

This material is based upon work supported by the U.S. National Science Foundation under
Grant No. CHE-2003505.

This work is supported by the Agriculture and Food Research Initiative [grant no.
2016-67019-25265/project accession no. 1009565] and the Hatch program [project accession
no. 1020955] from the USDA National Institute of Food and Agriculture.

Partial research funding was also provided by a Graduate Research Grant from the
Schmittau-Novak Small Grant Program of the School of Integrative Plant Science, Cornell
University.

## Disclaimer

Any opinions, findings, conclusions, or recommendations expressed in this material are
those of the author(s) and do not necessarily reflect the views of the U.S. National
Science Foundation or the U.S. Department of Agriculture.

The software is provided "as is", without warranty of any kind, as stated in the
[`LICENSE`](LICENSE).
