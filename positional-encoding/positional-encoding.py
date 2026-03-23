import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Positions (T, 1)
    pos = np.arange(seq_len)[:, np.newaxis]

    # Number of sin/cos pairs = ceil(d_model / 2)
    i = np.arange((d_model + 1) // 2)

    # Compute denominator: base^(2i/d_model)
    denom = np.power(base, (2 * i) / d_model)

    # Angles (T, ceil(d_model/2))
    angles = pos / denom

    # Initialize output
    pe = np.zeros((seq_len, d_model), dtype=float)

    # Even indices → sin
    pe[:, 0::2] = np.sin(angles[:, :pe[:, 0::2].shape[1]])

    # Odd indices → cos
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])

    return pe