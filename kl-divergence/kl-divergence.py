import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    result=0
    for i in range(len(p)):
        result+=p[i]*(math.log(p[i]+eps)-math.log(q[i])+eps)

    return result
        