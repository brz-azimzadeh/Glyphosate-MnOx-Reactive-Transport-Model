"""
config.py
=========
Experiment / reactor configuration for the reactive-transport model.

Every fit depends on a small set of operating parameters: the influent
concentration, the step-to-pulse switch time, and the reactor's residence-time
distribution (RTD). Collecting them in one place keeps the model functions free
of hard-coded constants, so the same code applies to ANY flow-through reactor
and ANY contaminant - you only change the configuration.

Hydraulic vs. experiment parameters
-----------------------------------
* ``N``, ``tau_step``, ``tau_pulse`` describe the *reactor hydraulics* and are
  obtained from a conservative tracer (Br-) test - see ``bromide.fit_tracer``.
  These are held FIXED when fitting the reactive model.
* ``C0`` and ``t_s`` describe the *experiment* (influent strength and the step
  duration / timing) and are set per run. Change them freely for a different
  experiment schedule.
* ``retardation`` (>= 1) optionally lengthens the step residence for adsorbing
  species (a conservative tracer is not retarded; glyphosate is). The model
  uses ``tau_step_eff = tau_step * retardation``; leave at 1.0 for none.

Load/save these as a CSV (``from_csv`` / ``to_csv``) so the fixed parameters
live in a file you can edit, version, and reuse.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import csv


# Parameters that are written to / read from the config CSV (scalars only).
_CSV_FIELDS = ["C0", "t_s", "N", "tau_step", "tau_pulse", "retardation"]
_CSV_META = {
    "C0":         ("uM",  "experiment", "Influent (parent) concentration during step"),
    "t_s":        ("min", "experiment", "Step -> washout switch time"),
    "N":          ("-",   "Br tracer",  "Gamma RTD shape (tanks-in-series)"),
    "tau_step":   ("min", "Br tracer",  "Gamma scale; mean hydraulic residence = N*tau_step"),
    "tau_pulse":  ("min", "Br tracer",  "Washout time constant"),
    "retardation":("-",   "reactive",   "Step-residence multiplier for adsorbing species (>=1)"),
}


@dataclass
class ExperimentConfig:
    """Operating conditions for a single microfluidic / flow-through run."""

    C0: float = 300.0
    t_s: float = 160.0
    N: float = 4.0
    tau_step: float = 1.6
    tau_pulse: float = 4.6
    retardation: float = 1.0
    columns: Dict[str, str] = field(
        default_factory=lambda: {
            "time": "Time (min)",
            "GP": "GP (uM)",
            "AMPA": "AMPA (uM)",
            "Gly": "Glycine (uM)",
            "Pi": "Pi (uM)",
            "NH4": "NH4+ (uM)",
            "Mn": "Mnsoln (uM)",
        }
    )

    @property
    def tau_step_eff(self) -> float:
        """Effective step residence used by the model (tau_step * retardation)."""
        return self.tau_step * self.retardation

    # ----------------------------- CSV I/O ------------------------------ #
    def to_csv(self, path: str) -> None:
        """Write the fixed scalar parameters to a key/value CSV."""
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["parameter", "value", "unit", "source", "description"])
            for k in _CSV_FIELDS:
                unit, src, desc = _CSV_META[k]
                v = getattr(self, k)
                if isinstance(v, float):
                    v = float(f"{v:.6g}")
                w.writerow([k, v, unit, src, desc])

    @classmethod
    def from_csv(cls, path: str, columns: Optional[Dict[str, str]] = None) -> "ExperimentConfig":
        """Build a config from a key/value CSV (``parameter,value,...``).

        Only the ``parameter`` and ``value`` columns are required; any extra
        columns (unit, source, description) are ignored. Unknown parameters are
        skipped, so the same CSV format is forward-compatible.
        """
        vals: Dict[str, float] = {}
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "parameter" not in reader.fieldnames:
                raise ValueError("config CSV must have a 'parameter' and 'value' header")
            for row in reader:
                name = (row.get("parameter") or "").strip()
                raw = (row.get("value") or "").strip()
                if name in _CSV_FIELDS and raw != "":
                    vals[name] = float(raw)
        kwargs = {k: vals[k] for k in _CSV_FIELDS if k in vals}
        if columns is not None:
            kwargs["columns"] = columns
        return cls(**kwargs)

    # ------------------------- From a tracer fit ------------------------ #
    @classmethod
    def from_tracer(cls, hydraulic, C0: float, t_s: float,
                    retardation: float = 1.0,
                    columns: Optional[Dict[str, str]] = None) -> "ExperimentConfig":
        """Build a config from a ``bromide.HydraulicResult`` plus experiment
        conditions. The hydraulics (N, tau_step, tau_pulse) come from the tracer
        and are thereby held fixed; C0 and t_s describe the reactive run.
        """
        kwargs = dict(C0=C0, t_s=t_s, N=hydraulic.N,
                      tau_step=hydraulic.tau, tau_pulse=hydraulic.tau_pulse,
                      retardation=retardation)
        if columns is not None:
            kwargs["columns"] = columns
        return cls(**kwargs)
