"""
* `choose(pair)`:
    1. calls `posterior(pair)` to obtain a dict {id: prob}
    2. if `self.soft`: sample choice from that posterior
       else:           take MAP choice (arg‑max)
    3. computes confidence = –entropy(posterior)
    4. returns (choice, confidence)
* `posterior(pair)` centralises the expected‑utility ➜ normalised‑prob computation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import random


def gaussian_cdf(t: float, mu: float, sigma: float) -> float:
    """Cumulative distribution Φ((t‑μ)/σ) using math.erf.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (t - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

@dataclass(slots=True)
class GaussianCategory:
    """A univariate Gaussian with fixed parameters."""

    mu: float
    sigma: float

    def success_prob(self, threshold: float) -> float:
        """Return P[X < threshold], tells you how probable 
        a draw from that category will end below the threshold."""
        return gaussian_cdf(threshold, self.mu, self.sigma)


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


    def choose(self, pair: Sequence[str]) -> Tuple[str, float]:
        """Return (choice, confidence = –entropy(posterior))."""
        post = self.posterior(pair)  # full posterior over the two actions

        if self.soft:
            choice = self._sample_from(post)
        else:
            choice = max(post, key=post.get)  # MAP

        confidence = self._neg_entropy(post)
        return choice, confidence
    
    def posterior(self, pair: Sequence[str]) -> Dict[str, float]:
        """Compute normalised posterior p(c|β,T,rewards) for the pair."""
        if len(pair) != 2:
            raise ValueError("pair must contain exactly two category ids")

        util = {cid: self._expected_utility(cid) for cid in pair}
        total = sum(util.values())
        if total == 0.0:
            # in pathological case, fall back to uniform
            return {cid: 1.0 / len(pair) for cid in pair}

        return {cid: u / total for cid, u in util.items()}


    def _expected_utility(self, cid: int) -> float:
        cat = self.categories[cid]  
        p_succ = cat.success_prob(self.T) # compute prob of success via cdf
        w_succ, w_fail = self.rewards.get(cid, (1.0, 0.0)) # get the rewards

        if self.beta == 0.0:
            # risk‑neutral: ordinary expected reward
            return p_succ * w_succ + (1.0 - p_succ) * w_fail

        # risk‑sensitive: expected exp‑utility
        return p_succ * math.exp(self.beta * w_succ) + (1.0 - p_succ) * math.exp(
            self.beta * w_fail
        )
    
    @staticmethod
    def _neg_entropy(posterior: Dict[str, float]) -> float:
        """Return –∑ p log p  (≥ 0 when some probabilities >1, ≤ 0 otherwise)."""
        return sum(p * math.log(p) for p in posterior.values() if p > 0.0)

    def _sample_from(self, posterior: Dict[str, float]) -> str:
        """Draw key according to its probability weight."""
        r = self.rng.random()
        acc = 0.0
        for k, p in posterior.items():
            acc += p
            if r <= acc:
                return k
        return k  # numerical precision fallback
    
    
if __name__ == "__main__": 
    cats = {
        "narrow_low": GaussianCategory(mu=0.22, sigma=0.02),
        "wide_low": GaussianCategory(mu=0.22, sigma=0.06),
        "narrow_high": GaussianCategory(mu=0.30, sigma=0.02),
        "wide_high": GaussianCategory(mu=0.30, sigma=0.06),
    }
    rewards = {
        "narrow_low": (10.0, 0.0),
        "wide_low": (11.0, 0.0),
        "narrow_high": (14.0, 0.0),
        "wide_high": (18.0, 0.0),
    }

    agent_det = DecisionAgent(cats, rewards, threshold=0.31, beta=1.0, soft=False)
    agent_soft = DecisionAgent(cats, rewards, threshold=0.31, beta=1.0, soft=True)

    for pair in [("wide_low", "wide_high"), ("narrow_low", "narrow_high")]:
        ch_det, conf_det = agent_det.choose(pair)
        ch_soft, conf_soft = agent_soft.choose(pair)
        print(f"{pair}: MAP→{ch_det} (conf={conf_det:+.3f}), "
              f"sample→{ch_soft} (conf={conf_soft:+.3f})")