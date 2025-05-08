"""
Reusable Matplotlib helpers for the Gaussian-threshold demos.
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def _pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def _gaussian_cdf(t: float, mu: float, sigma: float) -> float:
    z = (t - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


# ---------------------------------------------------------------------
#  1.  All Gaussians with first-pair highlighted
# ---------------------------------------------------------------------

COLOR_MAP = {
    "narrow_low":  "blue",
    "wide_low":    "green",
    "narrow_high": "red",
    "wide_high":   "yellow",
}


def plot_all_gaussians(
    categories: Dict[str, Tuple[float, float]],
    highlight_pair: Sequence[str],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    ax = ax or plt.gca()
    x = np.linspace(0.1, 0.45, 400)

    # Background
    for mu, sigma in categories.values():
        ax.plot(x, _pdf(x, mu, sigma), color="lightgrey", lw=1)
        
    for cid in highlight_pair:
        mu, sigma = categories[cid]
        ax.plot(x, _pdf(x, mu, sigma),
                color=COLOR_MAP[cid], lw=2.5,
                label=cid.replace("_", " ").title())
    
    ax.set(title="Two options presented", xlabel="x", ylabel="Probability density")
    ax.legend()
    return ax


# ---------------------------------------------------------------------
#  2.  Pair with shaded success region
# ---------------------------------------------------------------------
def plot_pair_with_threshold(
    pair: Dict[str, Tuple[float, float]],
    threshold: float,
    colors: Dict[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    ax = ax or plt.gca()
    colors = colors or {"wide_low": "yellow", "wide_high": "blue"}

    mus, sigs = zip(*pair.values())
    x = np.linspace(min(mus) - 4 * max(sigs), max(mus) + 4 * max(sigs), 500)

    for cid, (mu, sigma) in pair.items():
        colour = COLOR_MAP[cid]
        pdf = _pdf(x, mu, sigma)
        ax.plot(x, pdf, color=colour, lw=2, label=cid.replace("_", " ").title())
        mask = x <= threshold

        ax.fill_between(x[mask], pdf[mask], color=colour, alpha=0.3)

    ax.axvline(threshold, color="black", ls="--", lw=1.5, label="Threshold T")
    ax.set(title="Probability of Success is CDF up to Threshold", xlabel="x", ylabel="Probability density")
    ax.legend()
    return ax


# ---------------------------------------------------------------------
#  3.  Bernoulli from success-probabilities
# ---------------------------------------------------------------------
def plot_success_bernoulli(
    pair: Dict[str, Tuple[float, float]],
    threshold: float,
    colors: Dict[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    ax = ax or plt.gca()
    colors = colors or {"wide_low": "yellow", "wide_high": "blue"}

    p_succ = {cid: _gaussian_cdf(threshold, *params) for cid, params in pair.items()}
    total = sum(p_succ.values())
    probs = {cid: p / total for cid, p in p_succ.items()}

    ax.bar(probs.keys(), probs.values(),
       color=[COLOR_MAP[c] for c in probs], alpha=0.35)

    ax.set_ylim(0, 1)
    ax.set(ylabel="Probability", title="Posterior probability P(success|c)")
    for cid, p in probs.items():
        ax.text(cid, p + 0.02, f"{p:.2f}", ha="center") 
    return ax


def plot_bar_from_probs(
    probs: Dict[str, float],
    ax: Optional[plt.Axes] = None,
    color_map: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    alpha: float = 0.35,
    ) -> plt.Axes:
    """
    Plot a bar chart of a discrete distribution, with per-label colours.

    Parameters
    ----------
    probs : Dict[str, float]
        Mapping from category ID to probability. Values need not sum to 1.
    ax : plt.Axes, optional
        Axes on which to draw. If None, uses plt.gca().
    color_map : Dict[str,str], optional
        Mapping from category ID to a matplotlib-compatible color string.
        Defaults to the module-level COLOR_MAP.
    title : str, optional
        Title for the plot.
    alpha : float, default=1.0
        Opacity for the bars (0 transparent → 1 opaque).

    Returns
    -------
    ax : plt.Axes
        The Axes containing the bar chart.
    """
    ax = ax or plt.gca()
    color_map = color_map or COLOR_MAP

    labels = list(probs.keys())
    values = [probs[l] for l in labels]

    # look up each label’s colour (or grey if missing)
    colors = [color_map.get(l, "grey") for l in labels]

    bars = ax.bar(labels, values, color=colors, alpha=alpha)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(title or "Reward-weighted probabilities")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)

    # annotate each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )

    return ax
# ---------------------------------------------------------------------
#  4.  Bernoulli from expected reward
# ---------------------------------------------------------------------
def plot_reward_bernoulli(
    pair: Dict[str, Tuple[float, float, float]],
    threshold: float,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    ax = ax or plt.gca()

    exp_r = {}
    for cid, (mu, sigma, r) in pair.items():
        p = _gaussian_cdf(threshold, mu, sigma)
        exp_r[cid] = p * r

    total = sum(exp_r.values())
    probs = {cid: v / total for cid, v in exp_r.items()}

    ax.bar(probs.keys(), probs.values(),
       color=[COLOR_MAP.get(c, "grey") for c in probs])
    ax.set_ylim(0, 1)
    ax.set(
        ylabel="Probability",
        title="Posterior probability weighted by rewards P(success|c,r) ",
    )
    for cid, p in probs.items():
        ax.text(cid, p + 0.02, f"{p:.2f}", ha="center")
    return ax
