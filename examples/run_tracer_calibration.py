"""
run_tracer_calibration.py
=========================
End-to-end hydraulic-then-reactive workflow:

  1. Fit the conservative Br- tracer test  -> reactor hydraulics (N, tau, tau_pulse).
  2. Write those + the experiment conditions (C0, t_s) to a config CSV.
  3. Reload the config CSV and run the fully-coupled reactive ODE global fit
     with the hydraulics held FIXED.

To model a DIFFERENT microfluidic reactor or a different experiment schedule,
just supply a different tracer CSV and/or edit the generated config CSV
(C0, t_s, retardation, ...) - no code changes needed.

Run from the repo root:
    python examples/run_tracer_calibration.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import fit_tracer, tracer_curve, ExperimentConfig, fit_global_ode, plotting

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "output")
os.makedirs(OUT, exist_ok=True)

# Experiment conditions (change these for a different schedule / influent).
C0_REACTIVE = 300.0     # uM glyphosate influent during the step
T_S_REACTIVE = 160.0    # min, step -> washout switch in the reactive run
RETARDATION = 2.5       # adsorbing-species step-residence multiplier (>=1)

FILES = {
    "GP":   ("GP.csv",      "GP (uM)"),
    "AMPA": ("AMPA.csv",    "AMPA (uM)"),
    "Gly":  ("Glycine.csv", "Glycine (uM)"),
    "Pi":   ("Pi.csv",      "Pi (uM)"),
    "NH4":  ("NH4.csv",     "NH4+ (uM)"),
    "Mn":   ("Mnsoln.csv",  "Mnsoln (uM)"),
}


def load(fname, col):
    d = pd.read_csv(os.path.join(DATA, fname), encoding="utf-8-sig")
    return d["Time (min)"].to_numpy(float), d[col].to_numpy(float)


def main():
    # 1. Hydraulic characterization from the Br- tracer test --------------- #
    br = pd.read_csv(os.path.join(DATA, "Br_tracer.csv"), encoding="utf-8-sig")
    t_br = br["Time (min)"].to_numpy(float)
    c_br = br["[Br] mM"].to_numpy(float)
    # C0=1 mM tracer feed; step ended at 74 min; fix N=4 (tanks-in-series).
    hyd = fit_tracer(t_br, c_br, C0=1.0, t_s=74.0, fix_N=4)
    print(hyd.summary(), "\n")

    # 2. Build a config from the tracer + experiment conditions, save to CSV  #
    cfg = ExperimentConfig.from_tracer(hyd, C0=C0_REACTIVE, t_s=T_S_REACTIVE,
                                       retardation=RETARDATION)
    cfg_path = os.path.join(DATA, "experiment_config.csv")
    cfg.to_csv(cfg_path)
    print(f"Wrote fixed-parameter config -> {os.path.abspath(cfg_path)}")

    # 3. Reload the config (round-trip) and run the reactive fit ----------- #
    cfg = ExperimentConfig.from_csv(cfg_path)
    print(f"Loaded config: N={cfg.N}, tau_step={cfg.tau_step:.3f}, "
          f"tau_pulse={cfg.tau_pulse:.3f}, retardation={cfg.retardation}, "
          f"tau_step_eff={cfg.tau_step_eff:.3f}, C0={cfg.C0}, t_s={cfg.t_s}\n")

    data = {sp: load(f, c) for sp, (f, c) in FILES.items()}
    res = fit_global_ode(data, cfg)   # hydraulics fixed; only chemistry is fit
    print(res.summary())
    print("\nPer-species R2:",
          {k.replace("R2_", ""): round(v, 3)
           for k, v in res.extra.items() if k.startswith("R2_")})

    # Plots: tracer fit + reactive fit ------------------------------------- #
    import matplotlib.pyplot as plt
    tt = np.linspace(0, t_br.max(), 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_br, c_br, "o", ms=3, alpha=0.5, label="Br- observed")
    ax.plot(tt, hyd.predict(tt), "-", color="navy", label="gamma-RTD fit")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("[Br-] (mM)")
    ax.set_title(f"Br- tracer (N={hyd.N:.0f}, mean res {hyd.mean_residence:.2f} min, "
                 f"R2={hyd.r2:.3f})")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tracer_fit.png"), dpi=150)

    t_pred = np.arange(0, 260, 1)
    pred = res.predict(t_pred)
    obs = {sp: data[sp][1] for sp in ["GP", "AMPA", "Gly", "Pi", "NH4"]}
    plotting.plot_global(data["GP"][0], obs, t_pred,
                         {k: v for k, v in pred.items() if k != "Mn"},
                         title="Reactive fit with tracer-fixed hydraulics",
                         savepath=os.path.join(OUT, "reactive_fixed_hydraulics.png"))
    print(f"\nFigures written to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
