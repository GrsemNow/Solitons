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


def bisection(f, a0, b0):
    a = np.array([a0])
    b = np.array([b0])

    fa = f(a)
    fb = f(b)
    while True:
        c = (a+b) / 2.0
        fc = f(c)
        
        yield True, c, np.linalg.norm(fc), ((b-a) / 2.0)[0]

        if (fa * fc)[0] < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc


def root_search(f, params, method=bisection ,tol=1e-6, max_iter=100):
    gen = method(f, **params)
    for i in range(max_iter):
        flag, x, fx, err = next(gen) # выполнение значение невязка шаг
        
        if not flag:
            raise Exception(f'выроджение{method}')

        if fx < tol or err < tol:
            return x
    raise Exception(f"не сходится{method}")


def newton(f, x0, h=1e-7):
    x = np.asarray(x0)
    n = x.size

    while True:
        fx = np.asarray(f(x))
        J = np.zeros((n, n))

        for i in range(n):
            dx = np.zeros(n)
            dx[i] = h 

            J[:, i] = (np.asarray(f(x+dx)) - np.asarray(f(x-dx))) / (2 * h)
        det = np.linalg.det(J)

        if abs(det) < 1e-15:
            yield False, x, 0.0, 0.0
            return
        
        step = np.linalg.solve(J, -fx)

        xn = x + step
        yield True, xn, np.linalg.norm(fx), np.linalg.norm(step)

        x = xn
