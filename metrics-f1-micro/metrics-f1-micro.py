import numpy as np

def f1_micro(y_true, y_pred):
    """
    Compute micro-averaged F1 score for single-label multi-class classification.
    
    Args:
        y_true: list/array of true labels
        y_pred: list/array of predicted labels
    
    Returns:
        float in [0, 1]
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # True positives = correct predictions
    tp = np.sum(y_true == y_pred)
    
    # Total samples
    n = len(y_true)
    
    # Micro F1 = accuracy
    return float(tp / n)