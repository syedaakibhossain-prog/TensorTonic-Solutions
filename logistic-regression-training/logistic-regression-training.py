import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    x = np.array(X ,dtype=float)
    y = np.array(y , dtype=float)
    n , d = x.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        z = x @ w + b
        pred = _sigmoid(z)
        grad_w = x.T @ (pred -y) /n
        grad_b = np.mean(pred -y)
        w -= lr*grad_w
        b -= lr*grad_b

    return w , b
        