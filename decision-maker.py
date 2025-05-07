# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import random



# ------------------------------------------------------------------
#  Basic Gaussian helper
# ------------------------------------------------------------------


def gaussian_cdf(t: float, mu: float, sigma: float) -> float:
    """Cumulative distribution Φ((t‑μ)/σ) using math.erf.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (t - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


# ------------------------------------------------------------------
#  Category representation
# ------------------------------------------------------------------


@dataclass(slots=True)
class GaussianCategory:
    """A univariate Gaussian with fixed parameters."""

    mu: float
    sigma: float

    def success_prob(self, threshold: float) -> float:
        """Return P[X < threshold]."""
        return gaussian_cdf(threshold, self.mu, self.sigma)


# ------------------------------------------------------------------
#  Decision agent
# ------------------------------------------------------------------


class DecisionAgent:
    """Chooses between categories by maximising expected utility.

    Parameters
    ----------
    categories
        Mapping *id → GaussianCategory*.
    rewards
        Mapping *id → (reward_success, reward_failure)*.
        If a category id is missing, defaults to (1.0, 0.0).
    threshold
        Global decision threshold T.
    beta
        Inverse ‘temperature’ for the exponential utility.
        beta=1  -> risk‑sensitive expected exp‑utility
        beta=0  -> risk‑neutral expected reward
    soft
        If True, draw action from soft‑max distribution instead of arg‑max.
    rng
        Optional random.Random instance used only when *soft* is True.
    """

    def __init__(
        self,
        categories: Dict[int, GaussianCategory],
        rewards: Dict[int, Tuple[float, float]] | None = None,
        threshold: float = 0.0,
        beta: float = 1.0,
        soft: bool = False,
        rng: random.Random | None = None,
    ):
        self.categories = categories
        self.rewards = rewards or {}
        self.T = threshold
        self.beta = beta
        self.soft = soft
        self.rng = rng or random.Random()

    # --------------------------
    #  Public interface
    # --------------------------

    def choose(self, pair: Sequence[int]) -> int:
        """Return the chosen category id (arg‑max or soft‑max)."""
        if len(pair) != 2:
            raise ValueError("pair must contain exactly two category ids")
        utilities = {cid: self._expected_utility(cid) for cid in pair}

        if self.soft:
            return self._soft_sample(utilities)
        # deterministic
        return max(utilities, key=utilities.get)

    # --------------------------
    #  Internals
    # --------------------------

    def _expected_utility(self, cid: int) -> float:
        cat = self.categories[cid]  
        p_succ = cat.success_prob(self.T)
        w_succ, w_fail = self.rewards.get(cid, (1.0, 0.0))

        if self.beta == 0.0:
            # risk‑neutral: ordinary expected reward
            return p_succ * w_succ + (1.0 - p_succ) * w_fail

        # risk‑sensitive: expected exp‑utility
        return p_succ * math.exp(self.beta * w_succ) + (1.0 - p_succ) * math.exp(
            self.beta * w_fail
        )

    def _soft_sample(self, utilities: Dict[int, float]) -> int:
        """Draw action with probability ∝ utility (soft‑max)."""
        total = sum(utilities.values())
        if total == 0.0:
            # fallback to uniform
            return self.rng.choice(list(utilities.keys()))
        probs = {k: u / total for k, u in utilities.items()}
        r = self.rng.random()
        acc = 0.0
        for k, p in probs.items():
            acc += p
            if r <= acc:
                return k
        return k  # numerical edge case
    
    
# ------------------------------------------------------------------
#  Quick demo when executed directly
# ------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    # Define four Gaussian categories
    cats = {
        1: GaussianCategory(mu=0.22, sigma=0.06), # wide-low
        2: GaussianCategory(mu=0.22, sigma=0.02), # narrow-low
        3: GaussianCategory(mu=0.30, sigma=0.06), # wide-high
        4: GaussianCategory(mu=0.30, sigma=0.02), # narrow-high
    }

    # Rewards (success, failure) – asymmetric example
    rews = {
        1: (11.0, 0.0),
        2: (10.0, 0.0), 
        3: (18.0, 0.0), 
        2: (14.0, 0.0), 
    }

    agent = DecisionAgent(
        categories=cats,
        rewards=rews,
        threshold=0.31,
        beta=1.0,
        soft=False,
    )

    # Evaluate all pairs
    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    for pair in pairs:
        choice = agent.choose(pair)
        print(f"pair {pair}  →  chose {choice}")