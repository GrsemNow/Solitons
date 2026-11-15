import numpy as np

def rk4_step(rhs, x, y, h):
    k1 = rhs(x, y)
    k2 = rhs(x + h/2, y + k1 * h/2)
    k3 = rhs(x + h/2, y + k2 * h/2)
    k4 = rhs(x + h, y + k3 * h)
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

def integrate_step(rhs, xs, y0, step_fn=rk4_step):
    yield y0
    y = y0
    for i in range(len(xs) - 1):
        h = xs[i+1] - xs[i]
        y = step_fn(rhs, xs[i], y, h)
        yield y

def integrate(rhs, xs, y0, step_fn=rk4_step):
    return np.array(list(integrate_step(rhs, xs, y0, step_fn)))
