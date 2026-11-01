import numpy as np

def train_bigram_nn(xs, ys, vocab_size, lr=10.0, num_iters=100, alpha=0.0):
    """
    Train a one-layer neural-net bigram language model with full-batch gradient descent.

    Args:
        xs: list[int] of length N - input character indices in [0, vocab_size)
        ys: list[int] of length N - target next-character indices in [0, vocab_size)
        vocab_size: int V - number of distinct characters
        lr: float - learning rate
        num_iters: int - number of gradient descent steps
        alpha: float - L2 regularization strength on mean(W**2)

    Returns:
        list[list[float]] - trained weight matrix W of shape (V, V)
    """
    V = vocab_size
    N = len(xs)

    W = np.zeros((V,V), dtype = float)

    X = np.eye(V)[xs]
    Y = np.eye(V)[ys]

    for _ in range(num_iters):
        Z = X @ W 

        Z = Z - np.max(Z, axis = 1, keepdims=True)
        expZ = np.exp(Z)
        P = expZ / np.sum(expZ, axis=1,keepdims=True)

        grad = X.T @ (P-Y) / N
        grad += (2 * alpha / (V**2)) * W

        W -= lr * grad
    
    return W.tolist()