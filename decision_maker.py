
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from utils import *

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
        model: str="entropy", 
        threshold: float = 0.0,
        beta: float = 1.0, # reward inverse temperature
        alpha: float = 1.0,      # perceptual inverse‐temperature
        soft: bool = False,
        rng: random.Random | None = None,
        verbose: bool = False,
    ):
        
        allowed_models = {"map", "diff", "entropy"}
        if model not in allowed_models:
            raise ValueError(f"Invalid model '{model}'. Must be one of {allowed_models}.")
        self.categories = categories
        self.rewards = rewards or {}
        self.model=model
        self.T = threshold
        self.beta = beta
        self.alpha = alpha
        self.soft = soft
        self.rng = rng or random.Random()
        self.verbose = verbose


    def choose(self, pair: Sequence[str]) -> Tuple[str, float]:
        """Return (choice, confidence = –entropy(posterior))."""
        post = self.posterior(pair)  # full posterior over the two actions
        
        if self.verbose:
            self.visualize_pair(pair)


        if self.soft:
            choice = self._sample_from(post)
        else:
            choice = max(post, key=post.get)  # MAP

        confidence = self._confidence(post)

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
    
    def visualize_pair(self, pair: Sequence[str]) -> None:
        """
        1) Overplot all 4 gaussians in grey;
        2) Highlight the two in `pair` (yellow / blue).
        3) Shade their success‐regions under the curves.
        4) Bar‐plot the renormalised P(success) Bernoulli.
        """
        # 1. Grab the full categories dict:
        cat_params = {cid: (c.mu, c.sigma) for cid, c in self.categories.items()}
        # 2. Posterior of raw success-prob (no reward):
        post = self.posterior(pair)
        # 3. Call your existing plotting functions—but pass them only
        #    the raw CDF probabilities and colours:
        fig, axes = plt.subplots(1,1, figsize=(5,5))
    
        # # A: all four in grey, highlight pair
        # plot_all_gaussians(
        #     cat_params,
        #     highlight_pair=pair,
        #     ax=axes[0])
    
        # # B: plot pair with shaded success region:
        # pair_params = {cid: cat_params[cid] for cid in pair}
        
        # plot_pair_with_threshold(
        #     pair_params,
        #     threshold=self.T,
        #     ax=axes[1])
        
        # # PK: bar‐plot of raw success-prob
        # raw_prob = self._raw(pair)
        # plot_bar_from_probs(
        #     probs=raw_prob,
        #     raw=True,
        #     ax=axes[2])    
    
    
        # C: bar‐plot Bernoulli of raw success-prob: ??? PK: NOT RAW - reward weighted
        #    pass post directly—no recomputation!
        
        # calculate confidence and print in title of final plot
        confidence  = self._confidence(post)

        
        plot_bar_from_probs(
            probs=post,
            c=confidence,
            beta=self.beta)

        plt.tight_layout()
        plt.show()


    def _expected_utility(self, cid: int) -> float:
        cat = self.categories[cid]
        raw_p = cat.success_prob(self.T)
        
        # 1) temper the perceptual likelihood
        p_succ = self._tempered_prob(raw_p)
        p_fail = 1.0 - p_succ

        # 2) get reward weights
        w_succ, w_fail = self.rewards.get(cid, (1.0, 0.0))

        # 3) combine with reward-shaping β exactly as before
        return p_succ * (w_succ ** self.beta) + p_fail * (w_fail ** self.beta)
    
    def _raw(self, pair: Sequence[str]) -> Dict[str, float]:
        raw={}
        for cid in pair: 
            cat = self.categories[cid]  
            p_succ = cat.success_prob(self.T) # compute prob of success via cdf
            raw[cid]=p_succ
        return raw

        
    
    @staticmethod
    def _neg_entropy(posterior: Dict[str, float]) -> float:
        """Return –∑ p log p  (≥ 0 when some probabilities >1, ≤ 0 otherwise)."""
        return sum(p * math.log(p) for p in posterior.values() if p > 0.0)
    
    def _confidence(self, posterior: Dict[str, float]) -> float:
        """
        Normalised confidence in [0,1]:
        1 when the posterior is a delta (min entropy),
        0 when it's uniform (max entropy).
        """
        if self.model=="entropy":
            # 1) raw negative-entropy = sum p log p
            ne = sum(p * math.log(p) for p in posterior.values() if p > 0.0)
            # 2) maximum entropy (nats) for N equally-likely outcomes
            H_max = math.log(len(posterior))
            # 3) normalise: 1 + ne/H_max
            return 1.0 + ne / H_max
        elif self.model=="map":
            my_max = max(posterior.values())
            return (len(posterior)*my_max-1)/(len(posterior)-1)
        elif self.model=="diff":
            sorted_values=list(posterior.values())
            sorted_values.sort(reverse=True)
            diff = sorted_values[0] - sorted_values[1]
            return diff

    def _sample_from(self, posterior: Dict[str, float]) -> str:
        """Draw key according to its probability weight."""
        r = self.rng.random()
        acc = 0.0
        for k, p in posterior.items():
            acc += p
            if r <= acc:
                return k
        return k  # numerical precision fallback
    
    def _tempered_prob(self, p: float) -> float:
        """
        Raise-and‐renormalize p to the power alpha:
        p^(α) = p^α / [p^α + (1-p)^α].
        """
        # corner‐cases: if p is exactly 0 or 1, the formula still behaves:
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0
        num = p**self.alpha
        den = num + (1.0 - p)**self.alpha
        return num / den
    
    
if __name__ == "__main__": 
    
    # ------------------------------------------------------------------
    #  Model parameters
    # ------------------------------------------------------------------
    
    # cats = {
    #     "narrow_low": GaussianCategory(mu=0.22, sigma=0.02),
    #     "wide_low": GaussianCategory(mu=0.22, sigma=0.06),
    #     "narrow_high": GaussianCategory(mu=0.30, sigma=0.02),
    #     "wide_high": GaussianCategory(mu=0.30, sigma=0.06),
    # }
    # rewards = {
    #     "narrow_low": (10.0, 1e-6),
    #     "wide_low": (11.0, 1e-6),
    #     "narrow_high": (14.0, 1e-6),
    #     "wide_high": (18.0, 1e-6),
    # }
    
    cats = {
        "narrow_low": GaussianCategory(mu=0.22, sigma=0.02),
        #"wide_low": GaussianCategory(mu=0.22, sigma=0.06),
        # "narrow_high": GaussianCategory(mu=0.30, sigma=0.02),
        
        
        "wide_low": GaussianCategory(mu=-0.5, sigma=0.5),
        "narrow_high": GaussianCategory(mu=0.5, sigma=0.5),
        
        
        "wide_high": GaussianCategory(mu=0.30, sigma=0.06),
    }
    # rewards = {
    #     "narrow_low": (10.0, 1e-6),
    #     "wide_low": (11.0, 1e-6),
    #     "narrow_high": (14.0, 1e-6),
    #     "wide_high": (18.0, 1e-6),
    # }
    rewards = {
        "narrow_low": (4.0, 1e-6),
        "wide_low": (2.0, 1e-6),
        "narrow_high": (4.0, 1e-6),
        "wide_high": (40.0, 1e-6),
    }



    
    threshold = 0.31
    first_pair = ("wide_low", "narrow_high")

    # ------------------------------------------------------------------
    #  Agents
    # ------------------------------------------------------------------

    agent_det = DecisionAgent(cats, rewards, threshold=0., beta=10, alpha=10, soft=False, verbose=True)
    # agent_soft = DecisionAgent(cats, rewards, threshold=0.31, beta=10, soft=True, verbose=True)
    
    ch_det, conf_det = agent_det.choose(first_pair)
    
    

    
    # ------------------------------------------------------------------
    #  Produce plots
    # ------------------------------------------------------------------
    # fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    # plot_all_gaussians(
    #     {k: (v.mu, v.sigma) for k, v in cats.items()},
    #     highlight_pair=first_pair,
    #     ax=axs[0, 0],
    # )
    # plot_pair_with_threshold(
    #     {k: (cats[k].mu, cats[k].sigma) for k in first_pair},
    #     threshold,
    #     ax=axs[0, 1],
    # )
    # plot_success_bernoulli(
    #     {k: (cats[k].mu, cats[k].sigma) for k in first_pair},
    #     threshold,
    #     ax=axs[1, 0],
    # )
    # plot_reward_bernoulli(
    #     {
    #         "wide_low":  (0.22, 0.06, 11),
    #         "narrow_high":   (0.30, 0.06, 18),
    #     },
    #     threshold,
    #     {
    #         "wide_low":  (0.22, 0.06, 11),
    #         "narrow_high":   (0.30, 0.06, 18),
    #     },
    #     threshold,
    #     ax=axs[1, 1],
    # )
    # plt.tight_layout()
    # plt.show()
    #     ax=axs[1, 1],
    # )
    # plt.tight_layout()
    # plt.show()

    # ------------------------------------------------------------------
    #  Run two example trials
    # ------------------------------------------------------------------
    # for pair in [first_pair, ("narrow_low", "narrow_high")]:
    #     ch_det, conf_det = agent_det.choose(pair)
    #     ch_soft, conf_soft = agent_soft.choose(pair)
    #     # print(
        #     f"{pair}: MAP→{ch_det} (conf={conf_det:+.3f})"
        #     f"sample→{ch_soft} (conf={conf_soft:+.3f})"
        # )


    # for pair in [("wide_low", "wide_high"), ("narrow_low", "narrow_high")]:
    #     ch_det, conf_det = agent_det.choose(pair)
    #     ch_soft, conf_soft = agent_soft.choose(pair)
    #     print(f"{pair}: MAP→{ch_det} (conf={conf_det:+.3f}), "
    #           f"sample→{ch_soft} (conf={conf_soft:+.3f})")