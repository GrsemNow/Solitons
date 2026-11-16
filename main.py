import numpy as np
import matplotlib.pyplot as plt

from modules.config import VORTEX_DEFAULT
from modules.solver import make_bps_res, bisection
from modules.bps_system import make_bps_rhs, make_bps_init
from modules.integrators import integrate, rk4_step

def main():
    # ингициализация
    params = VORTEX_DEFAULT
    rho_grid = np.linspace(1e-5, 10.0, 700)
    
    # создание функции невязки
    res = make_bps_res(params, rho_grid)
    
    # предположение интервала по константам
    C_min = 0.1
    C_max = 2.0 
    
    # прогон
    try:
        C = bisection(res, a=C_min, b=C_max, tol=1e-6)
        print(f"C = {C:.6f}")
    except ValueError as e:
        print(e)
        return

    # решение с константой
    rhs = make_bps_rhs(params)
    y0 = make_bps_init(params, rho_grid[0], C)
    sol = integrate(rhs, rho_grid, y0, step_fn=rk4_step)
    
    f_sol = sol[:, 0]
    a_sol = sol[:, 1]

    # графики полей
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(rho_grid, f_sol)
    ax1.set_title(r'f($\rho$)')
    ax2.plot(rho_grid, a_sol)
    ax2.set_title(r'a($\rho$)')
    plt.savefig('img/bps_graph.svg')


if __name__ == "__main__":
    main()
