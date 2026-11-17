from modules.bps_system import make_bps_rhs, make_bps_init
from modules.integrators import integrate, rk4_step

def make_bps_res(params, rho_grid):
    rho_min = rho_grid[0]
    v = params['v']
    
    def res(C):
        rhs = make_bps_rhs(params)
        y0 = make_bps_init(params, rho_min, C)

        sol = integrate(rhs, rho_grid, y0, step_fn=rk4_step)
        
        f_max = sol[-1, 0]

        return f_max - v

    return res


def root_search(f, param, method=bisection ,tol=1e-6, max_iter=100):
    gen = method(f, **params)
    for i in range(max_iter):
        flag, x, fx, err = next(gen) # выполнение значение невязка шаг
        
        if not flag:
            raise Exception('выроджение')

        if fx < tol or err < tol:
            return x
    raise Exception("не сходится")


def bisection(f, a, b):
    fa = f(a)
    fb = f(b)
    while True:
        c = (a+b) / 2.0
        fc = f(c)
        
        yield True, c, abs(fc), (b-a) / 2.0

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        
'''
def newton_1d(f, x0, h=1e-7):
    x = x0
    
    while True:
        fx = f(x)
        dfx = (f(x+h) - f(x-h)) / (2 * h)

        xn = x - fx/dfx

        yield xn, abd(fx), abs(xn-x)

        x = xn
'''

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
            yield False, (0, 0, 0)
        
        step = np.linalg.solve(J, -fx)

        xn = x + step
        yield True, xn, np.linalg.norm(fx), np.linalg.norm(step)

        x = xn

