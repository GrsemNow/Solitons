"""
Здесь описывается вихрь в основном состоянии,
описываемый уравнениями Богомольного.
Система имеет вид:
    | f' = f/p (n - a)
    | a' = e^2 p (v^2 - f^2)
Начальные условия:
    | f(r->0) = C * r^n
    | a(r->0) = 1/2 * (evr)^2
    | f(r->inf) = v
    | a(r->inf) = n
"""

import numpy as np

from modules.math_methods import integrate, rk4_step


def make_bps_rhs(params):
    e = params['e']  # P.S. lam = 4e^2
    v = params['v']
    n = params['n']
    
    def rhs(rho, y):
        f, a = y
        fp = f/rho * (n - a)
        ap = e**2 * rho * (v**2 - f**2) # a растет до вихря
        return np.array([fp, ap])

    return rhs


def make_bps_init(params, rho_min, C):
    e = params['e']
    v = params['v']
    n = params['n']

    f0 = C * (rho_min ** n)
    a0 = 0.5 * e**2 * rho_min**2 * v**2

    return np.array([f0, a0])


def make_bps_res(params, rho_grid):
    rho_min = rho_grid[0]
    v = params['v']
    rhs = make_bps_rhs(params)

    def res(x):
        C = x[0]
        y0 = make_bps_init(params, rho_min, C)
        sol = integrate(rhs, rho_grid, y0, step_fn=rk4_step)
        
        f_end = sol[-1, 0]
        
        return np.array([f_end - v])

    return res


