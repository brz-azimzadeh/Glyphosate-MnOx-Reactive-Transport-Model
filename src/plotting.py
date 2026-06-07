"""
plotting.py
===========
Lightweight plotting helpers for observed-vs-predicted breakthrough curves.

``plot_fit`` makes a simple multi-panel comparison. ``plot_global`` reproduces
the broken-y-axis overlay from the notebook (useful when the parent compound
is ~10x higher than its byproducts); it degrades gracefully to a normal axis
if the optional ``brokenaxes`` package is not installed.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_fit(t_obs, obs: Dict[str, np.ndarray], t_pred, pred: Dict[str, np.ndarray],
             title: str = "", savepath: Optional[str] = None):
    """One panel per species: observed points + predicted line."""
    keys = list(pred.keys())
    fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, k in zip(axs[0], keys):
        if k in obs:
            ax.plot(t_obs, obs[k], "o", alpha=0.5, label=f"{k} observed")
        ax.plot(t_pred, pred[k], "-", label=f"{k} fitted")
        ax.set_title(k)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Concentration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


def plot_global(t_obs, obs: Dict[str, np.ndarray], t_pred, pred: Dict[str, np.ndarray],
                ylims=((-5, 55), (180, 310)), title: str = "Global fit",
                savepath: Optional[str] = None):
    """Overlay all species on a broken y-axis (parent vs. byproduct scales)."""
    colors = {"GP": "tab:blue", "AMPA": "tab:green", "Gly": "tab:orange",
              "Pi": "tab:purple", "NH4": "tab:red", "Mn": "tab:brown"}
    try:
        from brokenaxes import brokenaxes
        fig = plt.figure(figsize=(6, 4))
        bax = brokenaxes(ylims=ylims, hspace=0.05)
        bax.set_xlim(left=t_pred.min() - 5, right=t_pred.max() + 5)
        for k in pred:
            c = colors.get(k)
            bax.plot(t_pred, pred[k], "-", color=c, label=f"{k} (pred)")
            if k in obs:
                bax.plot(t_obs, obs[k], "o", color=c, alpha=0.4, label=f"{k} (obs)")
        bax.set_xlabel("Time (min)")
        bax.set_ylabel("Concentration (uM)")
        bax.set_title(title)
        bax.legend(loc="upper right", fontsize=8)
    except Exception:  # brokenaxes missing -> single axis
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in pred:
            c = colors.get(k)
            ax.plot(t_pred, pred[k], "-", color=c, label=f"{k} (pred)")
            if k in obs:
                ax.plot(t_obs, obs[k], "o", color=c, alpha=0.4, label=f"{k} (obs)")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Concentration (uM)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig
