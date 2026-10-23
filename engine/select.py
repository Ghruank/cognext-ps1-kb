"""Layer 4, Expected-Utility Set Selector.

Given calibrated relevance probabilities p_i, the utility-optimal set under independence
is a prefix of the units sorted by p (Lewis 1995; Ye et al., ICML 2012). For prefix k:
    E[precision] ~= S_k / k            (S_k = sum of the top-k probabilities)
    E[recall]    ~= min(1, S_k / G)    (G = expected gold count = sum of all p)
We pick the k that maximises the chosen utility. Capping recall at 1 means returning
more units than the expected gold count is never rewarded.
"""

import numpy as np


def select(p: np.ndarray, utility: str = "f1", min_k: int = 1, max_k: int = 400,
           g_scale: float = 1.0) -> tuple[np.ndarray, float]:
    """Returns (indices into p of the chosen set, G_hat)."""
    order = np.argsort(-p)
    ps = p[order]
    g = max(float(p.sum()) * g_scale, 1e-6)
    s = np.cumsum(ps)
    k = np.arange(1, len(ps) + 1)
    prec = s / k
    rec = np.minimum(1.0, s / g)
    if utility == "f1":
        u = 2 * prec * rec / (prec + rec + 1e-9)
    else:  # mean of precision and recall, as both are reported separately
        u = (prec + rec) / 2
    u[: min_k - 1] = -1
    u[max_k:] = -1
    best = int(np.argmax(u)) + 1
    return order[:best], g


def abstain(p: np.ndarray, threshold: float = 0.15) -> bool:
    """No unit is even moderately likely to be relevant: answer that nothing was found."""
    return len(p) == 0 or float(p.max()) < threshold
