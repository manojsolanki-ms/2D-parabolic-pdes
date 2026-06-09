import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# QUESTION 1 (Constant–Coefficient PDE)
# ------------------------------------------------------------
# PDE:
#     u_t = (1/pi^2) * u_xx + (1/pi^2) * u_yy 
#           + e^{-t} sin(pi x) sin(pi y)
#
# Domain:
#     (x, y) ∈ (0, 1) × (0, 1),   0 < t ≤ 1
#
# Initial Condition (IC):
#     u(x, y, 0) = sin(pi x) sin(pi y)
#
# Boundary Condition (BC):
#     u(x, y, t) = 0   on ∂Ω
#
# Exact Solution:
#     u_exact(x, y, t) = e^{-t} sin(pi x) sin(pi y)
#
# NOTE:
# The forcing term is NOT written manually.
# It is computed automatically as:
#     f = u_t - a * u_xx - b * u_yy
# ============================================================


# ============================================================
# QUESTION 2 (Variable–Coefficient PDE)
# ------------------------------------------------------------
# PDE:
#     u_t = (1 + 0.5 xy) * u_xx  +  (1 + 0.3 x^2) * u_yy  + f(x, y, t)
#
# Domain:
#     (x, y) ∈ (0, 1) × (0, 1),   0 < t ≤ 1
#
# Initial Condition (IC):
#     u(x, y, 0) = 0
#
# Boundary Condition (BC):
#     u(x, y, t) = 0   on ∂Ω
#
# Exact Solution:
#     u_exact(x, y, t) = t^2 sin(2π x) sin(π y)
#
# NOTE:
# The forcing term f(x, y, t) is AUTOMATICALLY generated using:
#     f = u_t - a(x, y) * u_xx - b(x, y) * u_yy
#
# ============================================================



# ---------------------------
# General solver wrapper (ADI + CFDM), parameterized by exact/forcing
# ---------------------------
def run_solver(h, a_func, b_func, u_exact_time_func, forcing_func, T=1.0):
    # choose time step small so spatial error dominates (left as in original)
    k = 1/10000
    m = int(1.0 / h)
    N = int(T / k)

    # grid
    x = np.linspace(0, 1, m+1)
    y = np.linspace(0, 1, m+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

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
    u[0] = u_exact_time_func(X, Y, 0.0)

    # enforce Dirichlet zero BC at t=0
    u[0, 0, :] = 0.0; u[0, m, :] = 0.0; u[0, :, 0] = 0.0; u[0, :, m] = 0.0

    # time-stepping
    for n in range(1, N+1):
        t_old = (n-1)*k
        t_new = n*k
        u_old = u[n-1]

        f_old = forcing_func(X, Y, t_old, a_grid, b_grid)
        f_new = forcing_func(X, Y, t_new, a_grid, b_grid)

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
    exact = u_exact_time_func(X, Y, T)
    num = u[N]

    # L-inf
    err_inf = np.max(np.abs(num - exact))
    # discrete L2 (uniform interior weighting)
    err_l2 = np.sqrt(h*h * np.sum((num - exact)**2))

    return err_inf, err_l2, X, Y, num, exact

# ---------------------------
# Exact + forcing functions for both cases
# ---------------------------

# ---------- Constant coefficient case ----------
def u_exact_time_const(X, Y, t):
    return np.exp(-t) * np.sin(np.pi*X) * np.sin(np.pi*Y)

def forcing_const(X, Y, t, a_grid, b_grid):
    S = np.sin(np.pi*X)*np.sin(np.pi*Y)
    u_t = -np.exp(-t)*S
    u_xx = -np.pi**2 * np.exp(-t) * S
    u_yy = u_xx
    return u_t - a_grid*u_xx - b_grid*u_yy

# coefficients for constant case
a_const = lambda X, Y: 1.0 / (np.pi**2)
b_const = lambda X, Y: 1.0 / (np.pi**2)

# ---------- Variable coefficient case ----------
def u_exact_time_var(X, Y, t):
    return (t**2) * np.sin(2*np.pi*X) * np.sin(np.pi*Y)

def forcing_var(X, Y, t, a_grid, b_grid):
    S = np.sin(2*np.pi*X)*np.sin(np.pi*Y)
    u_t = 2*t * S
    u_xx = -(2*np.pi)**2 * t**2 * S
    u_yy = -np.pi**2 * t**2 * S
    return u_t - a_grid*u_xx - b_grid*u_yy

# coefficients for variable case
a_var = lambda X, Y: 1.0 + 0.5 * X * Y
b_var = lambda X, Y: 1.0 + 0.3 * X**2

# ---------------------------
# Convergence test runner
# ---------------------------
def convergence_test(h_vals, a_func, b_func, u_exact_func, forcing_func, case_name="case"):
    errs_inf = []
    errs_l2 = []
    last_data = None
    for h in h_vals:
        print(f"Running {case_name}: h = {h}")
        e_inf, e_l2, X, Y, num, exact = run_solver(h, a_func, b_func, u_exact_func, forcing_func)
        errs_inf.append(e_inf)
        errs_l2.append(e_l2)
        last_data = (h, X, Y, num, exact)
        print(f"                                                 L2 = {e_l2:.3e}")

    orders_l2  = [np.log2(errs_l2[i]  / errs_l2[i+1])  for i in range(len(errs_l2)-1)]

    print("\nRESULTS for", case_name)
    print(" h_values:", h_vals)
    print(" L2    errors:", [f"{e:.3e}" for e in errs_l2])
    for i in range(len(orders_l2)):
        print(f" h={h_vals[i]:.5f} -> {h_vals[i+1]:.5f}  order_l2 ≈ {orders_l2[i]:.4f}")
    print("--------------------------------------------------\n")
    return errs_l2, last_data

# ---------------------------
# Run both tests
# ---------------------------
if __name__ == "__main__":
    h_values = [1/8,1/16]

    # Constant coefficient test
    errs_l2_const, last_const = convergence_test(
        h_values, a_const, b_const, u_exact_time_const, forcing_const, case_name="constant a=b=1/pi^2 (u = e^{-t} sinπx sinπy)"
    )

    # Variable coefficient test
    errs_l2_var, last_var = convergence_test(
        h_values, a_var, b_var, u_exact_time_var, forcing_var, case_name="variable a=1+0.5xy, b=1+0.3x^2 (u = t^2 sin2πx sinπy)"
    )

    # ---------------------------
    # Plotting: errors vs h (log-log)
    # ---------------------------
    h_vals = np.array(h_values)

    plt.figure(figsize=(8,4))
    plt.loglog(h_vals, errs_l2_const, marker='o', label='constant case (L2)')
    plt.loglog(h_vals, errs_l2_var, marker='s', label='variable case (L2)')
    plt.gca().invert_xaxis()
    plt.xlabel('h')
    plt.ylabel('L2 error')
    plt.title('Convergence (L2) — constant vs variable coefficient')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()

    # 3D surface plots for the finest h for each case
    h_fin_c, Xc, Yc, num_c, exact_c = last_const
    h_fin_v, Xv, Yv, num_v, exact_v = last_var

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2,2,1, projection='3d')
    ax1.plot_surface(Xc, Yc, num_c, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax1.set_title(f'Numerical (constant) h={h_fin_c}')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')

    ax2 = fig.add_subplot(2,2,2, projection='3d')
    ax2.plot_surface(Xc, Yc, exact_c, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax2.set_title('Exact (constant)')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')

    ax3 = fig.add_subplot(2,2,3, projection='3d')
    ax3.plot_surface(Xv, Yv, num_v, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax3.set_title(f'Numerical (variable) h={h_fin_v}')
    ax3.set_xlabel('x'); ax3.set_ylabel('y')

    ax4 = fig.add_subplot(2,2,4, projection='3d')
    ax4.plot_surface(Xv, Yv, exact_v, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax4.set_title('Exact (variable)')
    ax4.set_xlabel('x'); ax4.set_ylabel('y')

    plt.tight_layout()
    plt.show()
