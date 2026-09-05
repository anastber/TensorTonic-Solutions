import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    # Normalize into attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attn_weights, V)
    return output