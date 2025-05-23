#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:41:43 2025

@author: pzc
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import random

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 0,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
})


def plot_beta_alpha_variations(
    cats: Dict[str, GaussianCategory],
    rewards: Dict[str, Tuple[float, float]],
    priors: Dict[str, float], 
    x: float,
    pair=("A", "B"),
    model="entropy",
    soft=False
):
    """
    Panel 1: vary reward β    (α = 1)
    Panel 2: vary perceptual α (β = 1)
    Panel 3: vary both β = α

    Now requires you to pass in the datapoint x.
    """
    params = np.logspace(-2, 2.5, 50)
    ch_hist = [[], [], []]
    conf_hist = [[], [], []]

    for p in params:
        # 1) reward β sweep, α fixed = 1
        agent1 = DecisionAgent(
            cats,
            rewards,
            priors,
            beta=p, alpha=1.0,
            soft=soft, model=model
        )
        c1, f1 = agent1.choose(pair, x)
        ch_hist[0].append(c1)
        conf_hist[0].append(f1)

        # 2) perceptual α sweep, β fixed = 1
        agent2 = DecisionAgent(
            cats,
            rewards,
            priors,
            beta=1.0, alpha=p,
            soft=soft, model=model
        )
        c2, f2 = agent2.choose(pair, x)
        ch_hist[1].append(c2)
        conf_hist[1].append(f2)

        # 3) both β and α = p
        agent3 = DecisionAgent(
            cats, 
            rewards,
            priors,
            beta=p, alpha=p,
            soft=soft, model=model
        )
        c3, f3 = agent3.choose(pair, x)
        ch_hist[2].append(c3)
        conf_hist[2].append(f3)

    titles = ["Vary reward β (α=1)",
              "Vary perceptual α (β=1)",
              "Vary both β and α"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, ch_h, conf_h, title in zip(axes, ch_hist, conf_hist, titles):
        ax2 = ax.twinx()
        ln1 = ax.plot(params, ch_h, label="Choice", color='k')
        ln2 = ax2.plot(params, conf_h, label="Confidence", color='r')
        ax.set_xscale('log')
        ax.set_xlabel("Temperature value")
        ax.set_ylabel("Choice", color='k')
        ax2.set_ylabel("Confidence", color='r')
        ax2.set_yticks([0, 0.5, 1])

        ax.tick_params(axis='y', colors='k')
        ax2.tick_params(axis='y', colors='r')
        ax.set_title(title)
        # ax.legend(loc="upper left")
    
    plt.rcParams['font.size'] = 13
    plt.tight_layout()
    plt.show()


def gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """Univariate Gaussian PDF."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coef * math.exp(exponent)

@dataclass(slots=True)
class GaussianCategory:
    """A univariate Gaussian with fixed parameters."""
    mu: float
    sigma: float

    def likelihood(self, x: float) -> float:
        """Return p(x | this category)."""
        return gaussian_pdf(x, self.mu, self.sigma)
    
def sample_observation(
        true_dist: GaussianCategory,
        manual_x: Optional[float] = None,
        rng: Optional[random.Random] = None
        ) -> float:
    """
    Return a data point x.
    If manual_x is provided, return that;
    otherwise sample from true_dist ~ N(mu, sigma).
    """
    if manual_x is not None:
        return manual_x
    rng = rng or random.Random()
    return rng.gauss(true_dist.mu, true_dist.sigma)



class DecisionAgent:
    """Performs Bayesian classification over two Gaussians, then
    picks the action that maximizes expected (shaped) reward."""

    def __init__(
        self,
        categories: Dict[str, GaussianCategory],
        rewards: Dict[str, Tuple[float, float]] | None = None,
        priors: Dict[str, float] | None = None,
        beta: float = 1.0,   # reward inverse‐temperature
        alpha: float = 1.0,  # perceptual inverse‐temperature on likelihoods
        soft: bool = False,
        rng: random.Random | None = None,
        model: str = "entropy",  # how to compute confidence
        verbose: bool = False,
    ):
        self.categories = categories
        self.rewards = rewards or {}
        # default 50/50 prior if none given
        self.priors = priors or {cid: 1.0 / len(categories) for cid in categories}
        self.beta = beta
        self.alpha = alpha
        self.soft = soft
        self.rng = rng or random.Random()
        if model not in {"map", "diff", "entropy"}:
            raise ValueError(f"Invalid model '{model}'.")
        self.model = model
        self.verbose = verbose
        
    def posterior(self, pair: Sequence[str], x: float) -> Dict[str, float]:
        """Bayes with a sensory‐temperature α on the likelihood."""
        if len(pair) != 2:
            raise ValueError("pair must contain exactly two category ids")

        unnorm = {}
        for cid in pair:
            # 1) temper the likelihood by α
            like_t = self.categories[cid].likelihood(x) ** self.alpha
            prior_c = self.priors.get(cid, 0.0)
            unnorm[cid] = like_t * prior_c

        total = sum(unnorm.values())
        if total == 0.0:
            return {cid: 1.0 / len(pair) for cid in pair}
        return {cid: u / total for cid, u in unnorm.items()}

    def choose(self, pair: Sequence[str], x: float) -> Tuple[str, float]:
        # 1) standard Bayes
        post = self.posterior(pair, x)

        # 2) reward‐shaped utility
        util = {}
        for cid in pair:
            p_c = post[cid]
            w_succ, w_fail = self.rewards.get(cid, (1.0, 0.0))
            util[cid] = p_c * (w_succ  ** self.beta) \
                      + (1-p_c) * (w_fail ** self.beta)

        # 3) normalize
        total = sum(util.values())
        if total == 0:
            post_util = {cid: 1.0/len(pair) for cid in pair}
        else:
            post_util = {cid: u/total for cid,u in util.items()}

        # 4) decision
        if self.soft:
            choice = self._sample_from(post_util)
        else:
            choice = max(post_util, key=post_util.get)

        # 5) confidence
        conf = self._confidence(post_util)
        
        if self.verbose:
            print(f"x={x:.3f} → posterior={post} → post_util={post_util}")

        return choice, conf

    def _posterior_prior(self, pair: Sequence[str], x: float) -> Dict[str, float]:
        """Compute p(c|x) ∝ [p(x|c)]^α · prior(c)."""
        raw = {}
        for cid in pair:
            like = self.categories[cid].likelihood(x)
            # temper the likelihood
            like_t = like ** self.alpha
            raw[cid] = like_t * self.priors.get(cid, 0.0)
        total = sum(raw.values())
        if total == 0.0:
            # pathological fallback
            return {cid: 1.0 / len(pair) for cid in pair}
        return {cid: v / total for cid, v in raw.items()}

    def _expected_utility(self, cid: str, p_c: float) -> float:
        """
        Given P(c|x)=p_c, and rewards (r_success,r_fail), compute
        risk-sensitive expected reward:
          p_c * (r_success**β) + (1−p_c) * (r_fail**β)
        """
        r_succ, r_fail = self.rewards.get(cid, (1.0, 0.0))
        return p_c * (r_succ ** self.beta) + (1.0 - p_c) * (r_fail ** self.beta)

        # return p_c * (math.exp(r_succ * self.beta)) + (1.0 - p_c) * (math.exp(r_fail * self.beta))

    def _confidence(self, posterior: Dict[str, float]) -> float:
        """Normalized confidence [0,1] using entropy, MAP‐gap, or diff‐gap."""
        ps = [p for p in posterior.values() if p > 0.0]
        if self.model == "entropy":
            H = -sum(p * math.log(p) for p in ps)
            H_max = math.log(len(ps))
            return 1.0 - H / H_max
        elif self.model == "map":
            m = max(ps)
            return (len(ps) * m - 1) / (len(ps) - 1)
        else:  # "diff"
            a, b = sorted(ps, reverse=True)[:2]
            return a - b

    def _sample_from(self, dist: Dict[str, float]) -> str:
        r = self.rng.random()
        cum = 0.0
        for k, p in dist.items():
            cum += p
            if r <= cum:
                return k
        return k  # fallback

    def _viz_likelihoods(self, pair: Sequence[str], x: float, post: Dict[str, float]):
        """Simple plot: two PDFs and their posterior heights at x."""
        xs = np.linspace(
            min(self.categories[c].mu for c in pair) - 3,
            max(self.categories[c].mu for c in pair) + 3,
            300,
        )
        plt.figure(figsize=(5, 4))
        for cid in pair:
            y = [gaussian_pdf(xx, self.categories[cid].mu, self.categories[cid].sigma)
                 for xx in xs]
            plt.plot(xs, y, label=f"{cid} (post={post[cid]:.2f})")
            # mark the likelihood at x
            ly = gaussian_pdf(x, self.categories[cid].mu, self.categories[cid].sigma)
            plt.scatter([x], [ly], s=50)
        plt.title(f"Observation x={x:.2f}")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example usage:
    cats = {
        "A":  GaussianCategory(mu=0.5, sigma=0.5),
        "B": GaussianCategory(mu=-0.5, sigma=0.5),
    }
    rewards = {
        "A": (1.0, 0.0),
        "B": (1.0, 0.0),
    }
    
    priors = {
        "A": 0.75, 
        "B":0.25}
    
    agent = DecisionAgent(cats, 
                          rewards, 
                          priors, 
                          beta=1.0, 
                          alpha=1.0, 
                          soft=False, 
                          verbose=True)

    # define your true data‐generating distribution
    true_cat = GaussianCategory(mu=0.0, sigma=0.05)
    

    x = sample_observation(true_cat, manual_x=0.5)
    
    choice, conf = agent.choose(["A", "B"], x)
    print(f"Chose {choice} with confidence {conf:.2f}")
    
    # Suppose you want to test at x = 0.0
    x_observed = -0.5
    plot_beta_alpha_variations(cats, rewards, priors, x_observed)
    
    
    
