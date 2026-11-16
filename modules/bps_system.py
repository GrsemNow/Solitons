"""
Здесь описывается вихрь в основном состоянии,
описываемый уравнениями Богомольного.
Система имеет вид:
    | f' = f/p (a - n)
    | a' = e^2 p (v^2 - f^2)
Начальные условия:
    | f(r->0) = C * r^n
    | a(r->0) = 1/2 * (evr)^2
    | f(r->inf) = v
    | a(r->inf) = n
"""

import numpy as np 

def make_bps_rhs(params):
    e = params['e']  # P.S. lam = 4e^2
    v = params['v']
    n = params['n']
    
    def rhs(rho, y):
        f, a = y
        f_prime = f/rho * (n - a)
        a_prime = e**2 * rho * (v**2 - f**2) # a растет до вихря
        return np.array([f_prime, a_prime])

    return rhs


def make_bps_init(params, rho_min, C):
    e = params['e']
    v = params['v']
    n = params['n']

    f0 = C * (rho_min ** n)
    a0 = 0.5 * e**2 * rho_min**2 * (v**2 - C**2)

    return np.array([f0, a0])
