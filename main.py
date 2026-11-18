import numpy as np
import matplotlib.pyplot as plt

from modules.config import VORTEX_DEFAULT, make_params
from modules.bps_system import make_bps_rhs, make_bps_init, make_bps_res
from modules.math_methods import integrate, rk4_step, root_search, bisection, newton
from modules.vortex_system import make_vortex_rhs, make_vortex_init, make_vortex_res

def main():
    # ингициализация
    params = VORTEX_DEFAULT
    rho_grid = np.linspace(1e-5, 6, 700)
    
    # создание функции невязки
    res_bps = make_bps_res(params, rho_grid)
    
    # предположение интервала по константам
    C_min = 0.1
    C_max = 2.0 
    
    # Бисекция
    C_bps_1 = root_search(
            f=res_bps, 
            params={"a0":C_min, "b0":C_max},
            method=bisection,
            tol=1e-8,
            max_iter=100)

    print(C_bps_1[0])

    # Ньютон
    C_bps_2 = root_search(
            f=res_bps,
            params={"x0":C_bps_1},
            method=newton,
            tol=1e-8,
            max_iter=100
            )
    print(C_bps_2[0])
    

    # решение с константой
    rhs = make_bps_rhs(params)
    y0 = make_bps_init(params, rho_grid[0], C_bps_2[0])
    sol = integrate(rhs, rho_grid, y0, step_fn=rk4_step)
    
    f_sol = sol[:, 0]
    a_sol = sol[:, 1]
    

    print(VORTEX_DEFAULT)
    # графики полей
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(rho_grid, f_sol)
    ax1.set_title(r'f($\rho$)')
    ax2.plot(rho_grid, a_sol)
    ax2.set_title(r'a($\rho$)')
    #plt.show()
    plt.savefig('img/bps_graph.svg')



    # ОБЩЕЕ
    params_v = make_params(e=1.0, lam=4.05, v=1.0, n=1)
    
    rho_grid = np.linspace(1e-5, 6.0, 500)
    # создание функции невязки
    res_v = make_vortex_res(params_v, rho_grid)
    
    # предположение интервала по константам
    Cf = C_bps_2[0]
    Ca = 0.5 * params_v['e']**2 * params_v['v']**2
    x0 = np.array([Cf, Ca])
    
    print(f"Cf:{Cf}, Ca:{Ca}")
    
    # Ньютон
    C_v = root_search(
            f=res_v,
            params={"x0":x0, "h":1e-4},
            method=newton,
            tol=1e-4,
            max_iter=50
            )
    Cf0, Ca0 = C_v
    print(f"Cf0: {Cf0}, Ca0:{Ca0}")

    # решение с константой
    rhs_v = make_vortex_rhs(params_v)
    y0_v = make_vortex_init(params_v, rho_grid[0], Cf0, Ca0)
    sol_v = integrate(rhs_v, rho_grid, y0_v, step_fn=rk4_step)
    
    f_sol_v = sol_v[:, 0]
    a_sol_v = sol_v[:, 2]
    

    print(params_v)
    # графики полей
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(rho_grid, f_sol_v)
    ax1.set_title(r'f($\rho$)')
    ax2.plot(rho_grid, a_sol_v)
    ax2.set_title(r'a($\rho$)')
    #plt.show()
    plt.savefig('img/lam4_05_graph.svg')





if __name__ == "__main__":
    main()
