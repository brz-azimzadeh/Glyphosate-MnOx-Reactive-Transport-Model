"""
run_fit.py
==========
End-to-end example on real data (Bir, pH 4.6, no BSA) from
Azimzadeh & Martinez, Environ. Sci. Technol. 2025, 59, 19513-19525.

Loads breakthrough data, fits apparent rate constants, prints results, and
saves figures. To use your own system, change the CSV paths and the
ExperimentConfig (C0, t_s, tau_step, tau_pulse, N, column names).

Run from the repository root:
    python examples/run_fit.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import ExperimentConfig, fit_gp_ampa_gly, fit_curve, models, plotting

HERE = os.path.dirname(__file__)
DATA_GP = os.path.join(HERE, "..", "data", "GP.csv")

OUT = os.path.join(HERE, "..", "output")
os.makedirs(OUT, exist_ok=True)

# Published Table 2 values (min^-1) for this condition, for reference
PAPER = {"k_GP_loss": 1.93e-3, "k_AMPA_form": 0.65e-3, "k_AMPA_loss": 0.56e-3,
         "k_Gly_form": 1.28e-3, "k_Gly_loss": 0.34e-3, "k_Mn_ox": 13.81e-3}


def main():
    # Staged closed-form fits expect co-gridded species; merge the per-species
    # CSVs that share a time axis (GP/AMPA/Gly/Pi/NH4), and read Mn separately.
    import functools
    base = os.path.join(os.path.dirname(DATA_GP), "..", "data")
    def _read(f, col):
        d = pd.read_csv(os.path.join(base, f), encoding="utf-8-sig")
        return d.rename(columns={col: col})[["Time (min)", col]]
    parts = [_read("GP.csv", "GP (uM)"), _read("AMPA.csv", "AMPA (uM)"),
             _read("Glycine.csv", "Glycine (uM)"), _read("Pi.csv", "Pi (uM)"),
             _read("NH4.csv", "NH4+ (uM)")]
    df = functools.reduce(lambda a, b: pd.merge(a, b, on="Time (min)", how="outer"), parts)
    mndf = pd.read_csv(os.path.join(base, "Mnsoln.csv"), encoding="utf-8-sig")

    # tau values from the Br- tracer fits used for these curves
    cfg = ExperimentConfig(C0=300, t_s=160, N=4, tau_step=1.6, tau_pulse=3.7)
    t_pred = np.arange(0, 260, 1)

    # --- Global fit: GP + AMPA + glycine -----------------------------------
    # Default enforces k_GP_loss >= k_AMPA_loss >= k_Gly_loss (paper ordering).
    gag = fit_gp_ampa_gly(df, cfg)
    print(gag.summary(), "\n")

    # To reproduce the paper's Table 2 exactly, fix the weakly-identified loss
    # constants to the published values and fit the rest:
    #   gag = fit_gp_ampa_gly(df, cfg,
    #                         fixed={"k_AMPA_loss": 0.56e-3, "k_Gly_loss": 0.34e-3})

    gag.as_dataframe().to_csv(os.path.join(OUT, "Rates_GP_AMPA_Gly.csv"), index=False)

    # --- Pi and NH4 (independent effective-amplitude fits) ------------------
    pi = fit_curve(df, cfg, "Pi", models.make_pi_model(cfg),
                   ["k_Gly_form", "k_AMPA_form", "k_GP_loss", "k_AMPA_loss"],
                   p0=[1e-1, 1e-1, 2e-3, 1e-2])
    print(pi.summary(), "\n")

    nh4 = fit_curve(df, cfg, "NH4", models.make_nh4_model(cfg),
                    ["k_Gly_form", "k_Gly_loss", "k_AMPA_form",
                     "k_AMPA_loss", "k_NH4_loss"],
                    p0=[1e-2, 1e-2, 7e-3, 1e-2, 5e-3])
    print(nh4.summary(), "\n")

    # --- Soluble Mn: regeneration rate with k_GP_loss fixed from above -----
    cfg_mn = ExperimentConfig(C0=300, t_s=160, N=4, tau_step=1.6, tau_pulse=4.6)
    mn = fit_curve(mndf, cfg_mn, "Mn",
                   models.make_mnsoln_fixed_gp_model(cfg_mn, k_gp_loss=gag.values[0]),
                   ["k_Mn_ox"], p0=[1e-2])
    print(mn.summary(), "\n")

    # --- Quick comparison to the paper -------------------------------------
    print("Comparison to published Table 2 (min^-1):")
    for n, v in zip(gag.param_names, gag.values):
        print(f"  {n:<12} fit={v:.2e}  paper={PAPER[n]:.2e}")
    print(f"  {'k_Mn_ox':<12} fit={mn.values[0]:.2e}  paper={PAPER['k_Mn_ox']:.2e}")
    print(f"  S_AMPA = {gag.extra['S_AMPA']:.2f} (paper 0.33), "
          f"t_half = {gag.extra['t_half_GP_h']:.1f} h (paper 6 h)")

    # --- Figures -----------------------------------------------------------
    t_obs = df[cfg.columns["time"]].to_numpy(float)
    obs = {k: df[cfg.columns[k]].to_numpy(float)
           for k in ["GP", "AMPA", "Gly", "Pi", "NH4"]}
    obs["Mn"] = mndf[cfg.columns["Mn"]].to_numpy(float)
    t_mn = mndf[cfg.columns["time"]].to_numpy(float)

    pred = {}
    pred.update(gag.predict(t_pred))
    pred.update(pi.predict(t_pred))
    pred.update(nh4.predict(t_pred))

    plotting.plot_fit(t_obs, obs, t_pred, gag.predict(t_pred),
                      title="GP / AMPA / Glycine (Bir, pH 4.6, no BSA)",
                      savepath=os.path.join(OUT, "fit_GP_AMPA_Gly.png"))
    plotting.plot_global(t_obs, obs, t_pred, pred, title="GP and byproducts",
                         savepath=os.path.join(OUT, "fit_all_species.png"))

    # Soluble Mn (separate dataset / axis)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_mn, obs["Mn"], "o", color="tab:brown", alpha=0.5, label="Mn observed")
    ax.plot(t_pred, mn.predict(t_pred)["Mn"], "-", color="tab:brown", label="Mn fitted")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Soluble Mn (uM)")
    ax.set_title("Soluble Mn — regeneration fit")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fit_Mn.png"), dpi=150)

    print(f"\nResults and figures written to: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
