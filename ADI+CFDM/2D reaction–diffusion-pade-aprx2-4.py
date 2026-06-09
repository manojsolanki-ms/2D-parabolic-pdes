import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt


# ============================================================
# PDE USED IN THIS CODE  (Manufactured Solution: u = (1+t) sin(pi x) sin(pi y))
# ------------------------------------------------------------
# General PDE:
#        u_t = a(x,y) u_xx + b(x,y) u_yy + f(x,y,t)
#a
# Domain:
#        (x, y) ∈ (0,1) × (0,1),     t ∈ (0, 1]
#
# Boundary Condition (Dirichlet):
#        u(x,y,t) = 0   on ∂Ω
#
# Initial Condition:
#        u(x,y,0) = sin(pi x) sin(pi y)
#
# Manufactured Exact Solution:
#        u_exact(x,y,t) = (1 + t) sin(pi x) sin(pi y)
#
# Automatic forcing term:
#        f(x,y,t) = u_t - a(x,y) u_xx - b(x,y) u_yy
#
# NOTE:
#   The source term f is NOT typed manually.
#   It is computed automatically using the exact solution.
#
# ------------------------------------------------------------
# CASE A (Constant coefficients):
#        a(x,y) = 1/pi^2
#        b(x,y) = 1/pi^2
#
# CASE B (Variable coefficients):
#        a(x,y) = 1 + 0.5 x y
#        b(x,y) = 1 + 0.3 x^2
# ============================================================





# ---------------------------
# Solver wrapper (variable-coeff ADI + CFDM)
# ---------------------------
def run_solver(h, a_func, b_func, T=1.0):
    # choose time step small so spatial error dominates
    k = 1/10000
    m = int(1.0 / h)
    N = int(T / k)

    # grid
    x = np.linspace(0, 1, m+1)
    y = np.linspace(0, 1, m+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # exact solution and forcing (manufactured)
    def u_exact_time(Xg, Yg, t):
        return (1.0 + t) * (np.sin(np.pi * Xg) * np.sin(np.pi * Yg))

    def forcing_f(Xg, Yg, t, a_grid, b_grid):
        base = np.sin(np.pi * Xg) * np.sin(np.pi * Yg)
        u_t  = base
        u_xx = - (np.pi**2) * (1.0 + t) * base
        u_yy = u_xx
        return u_t - a_grid * u_xx - b_grid * u_yy

    # prepare coefficient grids (allow scalar or array returns)
    a_grid = np.asarray(a_func(X, Y))
    b_grid = np.asarray(b_func(X, Y))
    if a_grid.ndim == 0:
        a_grid = np.full_like(X, float(a_grid))
    if b_grid.ndim == 0:
        b_grid = np.full_like(X, float(b_grid))
    if a_grid.shape != X.shape or b_grid.shape != X.shape:
        raise ValueError("a_func or b_func returned wrong shape.")

    # build compact 1D operator for interior points (size m-1)
    if m-1 < 1:
        raise ValueError("m too small for interior points. Choose smaller h.")
    delta2_1D = diags(
        [np.ones(m-2), -2*np.ones(m-1), np.ones(m-2)],
        offsets=[-1, 0, 1],
        shape=(m-1, m-1)
    ).toarray()
    D1D = (1/h**2) * delta2_1D
    L1D = D1D - (h**2 / 12.0) * (D1D @ D1D)
    I1D = np.eye(m-1)

    mu = k / 2.0

    # solution storage
    u = np.zeros((N+1, m+1, m+1))
    u[0] = u_exact_time(X, Y, 0.0)

    # enforce Dirichlet zero BC at t=0
    u[0, 0, :] = 0.0; u[0, m, :] = 0.0; u[0, :, 0] = 0.0; u[0, :, m] = 0.0

    # time-stepping
    for n in range(1, N+1):
        t_old = (n-1)*k
        t_new = n*k
        u_old = u[n-1]

        f_old = forcing_f(X, Y, t_old, a_grid, b_grid)
        f_new = forcing_f(X, Y, t_new, a_grid, b_grid)

        # Ly u_old (apply L1D along y for each fixed x)
        Ly_u_old = np.zeros_like(u_old)
        for i in range(1, m):
            row_int = u_old[i, 1:m]                 # length m-1
            Ly1 = L1D @ row_int
            Ly_u_old[i, 1:m] = b_grid[i, 1:m] * Ly1

        # RHS for X-sweep
        rhs1 = np.zeros_like(u_old)
        rhs1[1:m, 1:m] = u_old[1:m, 1:m] + mu * Ly_u_old[1:m, 1:m] + (k/2.0) * f_old[1:m, 1:m]

        # X-sweep: column solves
        u_half = np.zeros_like(u_old)
        for j in range(1, m):
            rhs_col = rhs1[1:m, j].copy()
            a_col = a_grid[1:m, j]
            A_col = np.diag(a_col) @ L1D
            Mx = I1D - mu * A_col
            u_half[1:m, j] = np.linalg.solve(Mx, rhs_col)

        # BC on intermediate solution
        u_half[0, :] = 0.0; u_half[m, :] = 0.0; u_half[:, 0] = 0.0; u_half[:, m] = 0.0

        # Lx u_half (apply L1D along x)
        Lx_u_half = np.zeros_like(u_half)
        for j in range(1, m):
            col_int = u_half[1:m, j]
            Lx1 = L1D @ col_int
            Lx_u_half[1:m, j] = a_grid[1:m, j] * Lx1

        # RHS for Y-sweep
        rhs2 = np.zeros_like(u_half)
        rhs2[1:m, 1:m] = u_half[1:m, 1:m] + mu * Lx_u_half[1:m, 1:m] + (k/2.0) * f_new[1:m, 1:m]

        # Y-sweep: row solves
        u_new = np.zeros_like(u_old)
        for i in range(1, m):
            rhs_row = rhs2[i, 1:m].copy()
            b_row = b_grid[i, 1:m]
            B_row = np.diag(b_row) @ L1D
            My = I1D - mu * B_row
            u_new[i, 1:m] = np.linalg.solve(My, rhs_row)

        # BCs and store
        u_new[0, :] = 0.0; u_new[m, :] = 0.0; u_new[:, 0] = 0.0; u_new[:, m] = 0.0
        u[n] = u_new

    # final-time error norms
    exact = u_exact_time(X, Y, T)
    num = u[N]

    # L-inf
    err_inf = np.max(np.abs(num - exact))
    # discrete L2 (trapezoidal-like; here uniform interior weighting)
    err_l2 = np.sqrt(h*h * np.sum((num - exact)**2))

    return err_inf, err_l2

# ---------------------------
# Experiment runner: several h values
# ---------------------------
def convergence_test(h_vals, a_func, b_func, case_name="case"):
    errs_inf = []
    errs_l2 = []
    for h in h_vals:
        print(f"Running {case_name}: h = {h}")
        e_inf, e_l2 = run_solver(h, a_func, b_func)
        errs_inf.append(e_inf)
        errs_l2.append(e_l2)
        print(f"                                                 L2 = {e_l2:.3e}")

    # compute observed orders
    # orders_inf = [np.log2(errs_inf[i] / errs_inf[i+1]) for i in range(len(errs_inf)-1)]
    orders_l2  = [np.log2(errs_l2[i]  / errs_l2[i+1])  for i in range(len(errs_l2)-1)]

    print("\nRESULTS for", case_name)
    print(" h_values:", h_vals)
    # print(" L_inf errors:", [f"{e:.3e}" for e in errs_inf])
    print(" L2    errors:", [f"{e:.3e}" for e in errs_l2])
    for i in range(len(orders_l2)):
        print(f" h={h_vals[i]:.5f} -> {h_vals[i+1]:.5f}  order_inf ≈ {orders_l2[i]:.4f}, order_l2 ≈ {orders_l2[i]:.4f}")
    print("--------------------------------------------------\n")

# ---------------------------
# Define test cases
# ---------------------------

h_values = [1/8, 1/16]

# Case A: constant coefficients (should match ~4th order)
a_const = lambda X, Y: 1.0 / (np.pi**2)
b_const = lambda X, Y: 1.0 / (np.pi**2)

# Case B: variable coefficients (smooth)
a_var = lambda X, Y: 1.0 + 0.5 * X * Y          # smooth var coeff
b_var = lambda X, Y: 1.0 + 0.3 * X**2          # smooth var coeff

# Run tests
convergence_test(h_values, a_const, b_const, case_name="constant a=b=1/pi^2")
convergence_test(h_values, a_var, b_var, case_name="variable a=1+0.5xy, b=1+0.3x^2")

# (Optional) plot errors vs h on log-log
# Collect last-run outputs for plotting convenience (re-run or adapt as needed)
