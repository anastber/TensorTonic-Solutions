import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    # Column vector of positions: shape (seq_length, 1)
    positions = np.arange(seq_length).reshape(-1, 1)

    # Frequency divisor for each pair index i = 0, 1, ..., d_model/2 - 1
    # div_term[i] = 10000^(2i / d_model)
    i = np.arange(0, d_model, 2)  # 0, 2, 4, ... (length d_model/2)
    div_term = np.power(10000.0, i / d_model)  # shape (d_model/2,)

    # Angles matrix via broadcasting: shape (seq_length, d_model/2)
    angles = positions / div_term

    # Allocate output
    pe = np.zeros((seq_length, d_model))

    # Even columns get sine, odd columns get cosine
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)

    return pe