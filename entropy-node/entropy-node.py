import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    n = len(y)
    if n == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / n

    log_probs = np.where(probs > 0, np.log2(probs), 0.0)
    return float(-np.sum(probs * log_probs))
