"""
Reactive-transport model for coupled adsorption-oxidation at Mn-oxide
interfaces.

Two interchangeable fitting engines are provided (see docs/MODEL_DESCRIPTION.md):

  Closed-form (semi-analytical) engine
    fit_gp_ampa_gly   - constrained global fit of parent + 2 byproducts (5 const.)
    fit_curve         - single-species curve_fit wrapper (Pi, NH4, soluble Mn)
    fit_all_species   - one-run closed-form fit of every species together
    models.*          - per-species breakthrough expressions + make_* factories

  Numerical, fully-coupled ODE engine
    fit_global_ode    - one shared parameter set; all species integrated together,
                        each species may use its own independent time axis
    ode_model.*       - the coupled ODE system and its effluent evaluator

  Shared infrastructure
    ExperimentConfig  - reactor/experiment parameters (CSV-driven, from_csv/to_csv)
    fit_tracer        - Br- tracer fit -> fixed reactor hydraulics (bromide.py)
    transport.*       - gamma RTD + step/pulse assembly
    plotting.plot_fit / plot_global

See README.md and docs/MODEL_DESCRIPTION.md for the equations and usage.
"""

from .config import ExperimentConfig
from .fit import fit_gp_ampa_gly, fit_curve, fit_all_species, fit_global_ode, FitResult
from .bromide import fit_tracer, tracer_curve, HydraulicResult
from . import models, transport, plotting, ode_model, bromide

__all__ = [
    "ExperimentConfig", "fit_gp_ampa_gly", "fit_curve", "fit_all_species",
    "fit_global_ode", "FitResult",
    "fit_tracer", "tracer_curve", "HydraulicResult",
    "models", "transport", "plotting", "ode_model", "bromide",
]
__version__ = "1.0.0"
