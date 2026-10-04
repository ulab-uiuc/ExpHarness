"""
Utility-grounded reward used to train the ExpHarness retrieval copilot.

    r = (s_with - s_without) + eta * s_with,   eta = 1

Utility = score_with_experience - score_without_experience
Generation = score_with_experience
Total = Utility + Generation

Range: [-1, 2]
  2 = zero-shot fails, with-experience succeeds (best)
  1 = both succeed
  0 = both fail
 -1 = zero-shot succeeds, with-experience fails (worst, penalizes bad retrieval)
"""

import json
import os
import threading


class ALFWorldZeroshotCache:
    """Cache for zero-shot (no experience) baseline results per gamefile."""

    def __init__(self, cache_path="data/alfworld_zeroshot_cache.json"):
        self.cache_path = cache_path
        self.lock = threading.Lock()
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.cache = json.load(f)
        else:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            self.cache = {}

    def get(self, gamefile: str):
        """Get cached zero-shot score. Returns None if not cached."""
        return self.cache.get(gamefile)

    def set(self, gamefile: str, score: float):
        """Cache a zero-shot score."""
        with self.lock:
            self.cache[gamefile] = score

    def save(self):
        """Persist cache to disk."""
        with self.lock:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2)


def compute_alfworld_reward(
    score_with_exp: float,
    score_without_exp: float,
    use_utility: bool = True,
    use_generation: bool = True,
) -> float:
    """
    Compute ALFWorld reward with utility scoring.

    Args:
        score_with_exp: success score when agent has retrieved experiences (0 or 1)
        score_without_exp: success score when agent has no experiences (0 or 1)
        use_utility: whether to include utility component
        use_generation: whether to include generation component

    Returns:
        Combined reward in range [-1, 2]
    """
    utility_score = 0.0
    generation_score = 0.0

    if use_generation:
        generation_score = score_with_exp

    if use_utility:
        utility_score = score_with_exp - score_without_exp

    if use_utility and use_generation:
        return utility_score + generation_score
    elif use_generation:
        return generation_score
    elif use_utility:
        return utility_score
    else:
        raise ValueError("At least one of use_utility/use_generation must be True")


def compute_reward_metrics(results_with_exp, results_without_exp):
    """
    Compute reward metrics for logging.

    Args:
        results_with_exp: list of episode result dicts (with experiences)
        results_without_exp: list of episode result dicts (without experiences)

    Returns:
        dict of metric_name -> value
    """
    metrics = {}
    n = len(results_with_exp)
    if n == 0:
        return metrics

    # Success rates
    sr_with = sum(1 for r in results_with_exp if r['won']) / n
    sr_without = sum(1 for r in results_without_exp if r['won']) / n
    metrics['reward/success_rate_with_exp'] = sr_with
    metrics['reward/success_rate_without_exp'] = sr_without
    metrics['reward/utility_gain'] = sr_with - sr_without

    # Reward distribution
    rewards = []
    for rw, rwo in zip(results_with_exp, results_without_exp):
        reward = compute_alfworld_reward(rw['score'], rwo['score'])
        rewards.append(reward)

    metrics['reward/mean'] = sum(rewards) / len(rewards)
    metrics['reward/max'] = max(rewards)
    metrics['reward/min'] = min(rewards)

    # Category breakdown
    n_both_success = sum(1 for rw, rwo in zip(results_with_exp, results_without_exp)
                         if rw['won'] and rwo['won'])
    n_only_with = sum(1 for rw, rwo in zip(results_with_exp, results_without_exp)
                      if rw['won'] and not rwo['won'])
    n_only_without = sum(1 for rw, rwo in zip(results_with_exp, results_without_exp)
                         if not rw['won'] and rwo['won'])
    n_both_fail = sum(1 for rw, rwo in zip(results_with_exp, results_without_exp)
                      if not rw['won'] and not rwo['won'])

    metrics['reward/both_success'] = n_both_success / n
    metrics['reward/only_with_exp_success'] = n_only_with / n
    metrics['reward/only_without_exp_success'] = n_only_without / n
    metrics['reward/both_fail'] = n_both_fail / n

    return metrics
