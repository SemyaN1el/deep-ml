from tinygrad import Tensor

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked tensor for x, compute f(x), run backprop, return df/dx as a float
    x = Tensor(float(x_value), requires_grad=True)

    y = x**2 + 3*x + 2

    y.backward()

    return float(x.grad.item())