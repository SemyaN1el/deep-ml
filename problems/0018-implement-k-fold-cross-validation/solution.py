import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    # Your code here
    n_samples = len(X)
    indices = np.arange(n_samples)
    if shuffle:
        if np.random.seed is not None:
            np.random.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype = int)
    fold_sizes[:n_samples % k] += 1
    curr = 0
    folds = []
    for fold_size in fold_sizes:
        folds.append(indices[curr:curr+fold_size])
        curr += fold_size
    ans = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        ans.append((train_idx,test_idx))
    return ans

