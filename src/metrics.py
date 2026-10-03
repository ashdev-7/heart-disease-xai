"""Agreement metrics between attribution rankings, with chance levels."""
from math import comb

import numpy as np
from scipy.stats import rankdata


def ranks_desc(a):
    """Rank 1 = largest |attribution| along the last axis (ties averaged)."""
    return rankdata(-np.abs(a), axis=-1)


def expected_jaccard(F, k):
    """Exact expectation of Jaccard between two independent random k-subsets of F items."""
    tot = comb(F, k)
    return sum(comb(k, i) * comb(F - k, k - i) / tot * (i / (2 * k - i)) for i in range(k + 1))


def corrected(obs, chance):
    return (obs - chance) / (1 - chance)


def jaccard_topk(Ra, Rb, k):
    """Elementwise top-k Jaccard between rank arrays of the same shape [..., F]."""
    A, B = Ra <= k, Rb <= k
    inter = (A & B).sum(-1)
    return inter / (A.sum(-1) + B.sum(-1) - inter)


def spearman(Ra, Rb):
    Za = Ra - Ra.mean(-1, keepdims=True)
    Zb = Rb - Rb.mean(-1, keepdims=True)
    den = np.linalg.norm(Za, axis=-1) * np.linalg.norm(Zb, axis=-1)
    return (Za * Zb).sum(-1) / np.where(den == 0, 1, den)


def boot_ci(x, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = [np.nanmean(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return np.percentile(m, [2.5, 97.5])
