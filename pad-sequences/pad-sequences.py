import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    
    if not seqs:
        return np.zeros((0, 0), dtype=int)

    # Determine max_len
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0

    N = len(seqs)

    # Initialize array with pad_value
    result = np.full((N, max_len), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]  # truncate if longer
        result[i, :len(trunc)] = trunc  # place values

    return result
