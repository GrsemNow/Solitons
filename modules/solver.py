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


def bisection(f, a, b, tol=1e-6, max_iter=100):
    fa = f(a)
    fb = f(b)
   
    if fa * fb > 0:
        raise ValueError("одного знака")
    
    for i in range(max_iter):
        c = (a+b) / 2.0
        fc = f(c)

        if abs(fc) < tol or (b-a)/2.0 < tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        
    return (a + b) / 2.0
    
