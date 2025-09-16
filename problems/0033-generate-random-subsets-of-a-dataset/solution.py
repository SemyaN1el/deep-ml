import numpy as np

def get_random_subsets(X, y, n_subsets, replacements, seed=None):
    """
    Generate random subsets of a given dataset.
    
    Parameters:
    X (numpy.ndarray): 2D array of shape (n_samples, n_features)
    y (numpy.ndarray): 1D array of shape (n_samples,)
    n_subsets (int): Number of random subsets to generate
    replacements (bool): If True, sample with replacement; otherwise, without replacement
    seed (int, optional): Random seed for reproducibility. Default is None.
    
    Returns:
    list: List of n_subsets tuples, each containing (X_subset, y_subset)
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    n_samples = len(X)
    
    subset_size = n_samples if replacements else n_samples // 2  
    
    subsets = []
    
    if seed is not None:
        np.random.seed(seed)
    
    for _ in range(n_subsets):
        if replacements:
            indices = np.r