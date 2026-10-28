import torch

def grad_of_quadratic(x: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    x = torch.tensor(x, dtype = torch.float32, requires_grad = True)
    y = x**2 + 3*x + 2

    y.backward()
    return x.grad.item()
