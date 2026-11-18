"""
Уравнения вихря.
Система:
    | f'' + f'/r - f/r^2 (n-a)^2 - l/4 f(f^2 - v^2) = 0
    | a'' - a'/r + 2e^2 f^2 (n-a) = 0
Начальные условия:
    | f(r->0) = C_f * r^n
    | a(r->0) = C_a r^2
    | f(r->inf) = v
    | a(r->inf) = n
"""

import numpy as np

from modules.math_methods import integrate, rk4_step

def make_vortex_rhs(params):
    e = params['e']
    lam = params['lam']
    v = params['v']
    n = params['n']

    def rhs(rho, y):
       f, fp, a, ap = y
       fpp = - fp/rho + f/(rho**2) * (n-a)**2 + lam/4 * f * (f**2 - v**2)
       app = ap/rho - 2 * e**2 * f**2 * (n - a)
       return np.array([fp, fpp, ap, app])
    
    return rhs


def make_vortex_init(params, rho_min, Cf, Ca):
    n = params['n']

    f0 = Cf * rho_min**n 
    fp0 = n * Cf * rho_min ** (n-1)
    a0 = Ca * rho_min**2
    ap0 = 2 * Ca * rho_min

    return np.array([f0, fp0, a0, ap0])


def make_vortex_res(params, rho_grid):
    rho_min = rho_grid[0]
    v = params['v']
    n = params['n']
    rhs = make_vortex_rhs(params)
 
    def res(x):
        Cf, Ca = x[0], x[1]
        y0 = make_vortex_init(params, rho_min, Cf, Ca)
        sol = integrate(rhs, rho_grid, y0, step_fn=rk4_step)

        f_end = sol[-1, 0]
        a_end = sol[-1, 2]

        return np.array([f_end - v, a_end - n])

    return res

    
