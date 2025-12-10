import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.
    
    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    # Initialize
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()
    
    # Calculate initial residual vector
    r = b - A @ x  # Используем @ для матричного умножения
    p = r.copy()   # Initial search direction
    
    # Check if already converged
    if np.linalg.norm(r) < tol:
        return x
    
    # Iteration loop
    for k in range(n):
        # Step size
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        
        # Update solution
        x = x + alpha * p
        
    