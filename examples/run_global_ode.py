"""
run_global_ode.py
=================
Fully-coupled NUMERICAL ODE global fit of all species in a single run.

Each species is read from its OWN CSV file with its OWN time axis - the species
do not need to share sampling times or lengths. All species are then fit
simultaneously with one shared parameter set (no per-species effective
constants, no fixed values).

Run from the repo root:
    python examples/run_global_ode.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import ExperimentConfig, fit_global_ode, plotting

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "output")
os.makedirs(OUT, exist_ok=True)

# Each species: (csv filename, value column). Independent files / time axes.
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
    data = {sp: load(f, c) for sp, (f, c) in FILES.items()}
    print("Points per species (independent time axes):",
          {sp: len(t) for sp, (t, y) in data.items()})

    cfg = ExperimentConfig(C0=300, t_s=160, N=4, tau_step=1.6, tau_pulse=3.7)

    # Fully coupled ODE fit (one shared parameter set, no fixed values).
    res = fit_global_ode(data, cfg)
    print("\n" + res.summary())
    print("\nPer-species R2:",
          {k.replace("R2_", ""): round(v, 3)
           for k, v in res.extra.items() if k.startswith("R2_")})

    # Plot: 5 species share an axis; Mn has its own time grid.
    t_pred = np.arange(0, 260, 1)
    pred = res.predict(t_pred)
    obs = {sp: data[sp][1] for sp in ["GP", "AMPA", "Gly", "Pi", "NH4"]}
    t_obs = data["GP"][0]
    plotting.plot_global(t_obs, obs, t_pred,
                         {k: v for k, v in pred.items() if k != "Mn"},
                         title="Fully-coupled ODE global fit (all species)",
                         savepath=os.path.join(OUT, "ode_global_all_species.png"))

    import matplotlib.pyplot as plt
    t_mn, y_mn = data["Mn"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_mn, y_mn, "o", color="tab:brown", alpha=0.5, label="Mn observed")
    ax.plot(t_pred, pred["Mn"], "-", color="tab:brown", label="Mn fitted")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Soluble Mn (uM)")
    ax.set_title("Soluble Mn (coupled ODE)")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ode_global_Mn.png"), dpi=150)
    print(f"\nFigures written to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
