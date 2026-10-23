"""Evidence-graph layer: message passing over the candidate pool's paragraph graph.

Relevance is strongly clustered along the document: on the training set, the paragraph
after a relevant one is relevant 63% of the time, against a 3.2% base rate (about 19x).
Two published graph-learning designs exploit that with very little training data:

  SIGN / SGC (Rossi et al. 2020; Wu et al. ICML 2019): precompute features diffused
  over 1..K hops of the normalised adjacency, then train any classifier on the stack.
  The graph does the message passing; the classifier needs no graph-specific training.

  Correct & Smooth (Huang et al. ICLR 2021): after a base model predicts, smooth those
  predictions over the graph so a confident node lifts its neighbours.
"""

import numpy as np
import scipy.sparse as sp


def norm_adj(edges: np.ndarray, n: int) -> sp.csr_matrix:
    """Symmetric-normalised adjacency with self-loops: D^-1/2 (A + I) D^-1/2."""
    a = sp.coo_matrix((np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float32) + sp.eye(n, dtype=np.float32, format="csr")
    d = np.asarray(a.sum(1)).ravel()
    dinv = sp.diags(1 / np.sqrt(d))
    return (dinv @ a @ dinv).tocsr()


def sign_features(x: np.ndarray, adj: sp.csr_matrix, hops: int = 2) -> np.ndarray:
    """[X, AX, A^2X, ...] — multi-hop diffused node features."""
    out, cur = [x], x
    for _ in range(hops):
        cur = adj @ cur
        out.append(cur)
    return np.hstack(out).astype(np.float32)


def smooth(p: np.ndarray, adj: sp.csr_matrix, alpha: float = 0.6, iters: int = 10) -> np.ndarray:
    """Label-propagation smoothing of predictions (the 'Smooth' step of Correct & Smooth)."""
    z = p.copy()
    for _ in range(iters):
        z = (1 - alpha) * p + alpha * (adj @ z)
    return np.clip(z, 0, 1)
