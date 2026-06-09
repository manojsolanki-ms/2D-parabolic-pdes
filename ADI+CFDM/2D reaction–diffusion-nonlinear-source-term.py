import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

# ============================================================
# PROBLEM PARAMETERS
# ============================================================

Lx = 1.0
Ly = 1.0
T  = 1.0

m = 8     # number of spatial intervals
h = 1.0 / m

k = 1/1000                # time step
N = int(T / k)
mu = k / 2.0            # Crank–Nicolson parameter

# ============================================================
# GRID
# ============================================================

x = np.linspace(0, Lx, m+1)
y = np.linspace(0, Ly, m+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# ============================================================
# VARIABLE COEFFICIENTS
# ============================================================

a_grid = 0.5 * (X + Y + 1.0)
b_grid = 2.0 / np.sqrt(X**2 + Y**2 + 1.0)

# ============================================================
# 4th-ORDER COMPACT FD OPERATOR (1D)
# ============================================================

interior = m - 1

off  = np.ones(interior-1)
diag = -2.0 * np.ones(interior)

D2 = (1.0 / h**2) * diags([off, diag, off], [-1, 0, 1]).toarray()

# Compact operator:  L = D2 - (h^2/12) D2^2
L1D = D2 - (h**2 / 12.0) * (D2 @ D2)
I1D = np.eye(interior)

# ============================================================
# INITIAL CONDITION (exact at t = 0)
# ============================================================

u_old   = np.sin(np.pi * X) * np.sin(np.pi * Y)
u_older = u_old.copy()           # needed for extrapolation

# Enforce Dirichlet BCs (exact = 0)
u_old[0,:] = u_old[-1,:] = 0.0
u_old[:,0] = u_old[:,-1] = 0.0

# ============================================================
# TIME LOOP
# ============================================================

for n in range(1, N+1):

    t_old = (n-1) * k
    t_new = n * k

    # --------------------------------------------------------
    # Exact spatial factor
    # --------------------------------------------------------
    S = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # --------------------------------------------------------
    # Manufactured source s(x,y,t)
    # --------------------------------------------------------
    s_old = (
        np.exp(-t_old) * S * (-1.0 + np.pi**2 * (a_grid + b_grid))
        - np.exp(-2*t_old) * S**2
    )

    s_new = (
        np.exp(-t_new) * S * (-1.0 + np.pi**2 * (a_grid + b_grid))
        - np.exp(-2*t_new) * S**2
    )

    # --------------------------------------------------------
    # NONLINEAR TERM: extrapolation for u^{n+1}
    # u^{n+1} ≈ 2u^n - u^{n-1}
    # --------------------------------------------------------
    u_ext = 2.0 * u_old - u_older
    NL = 0.5 * (u_old**2 + u_ext**2)

    # ========================================================
    # EXPLICIT Y-DIRECTION OPERATOR  b(x,y) u_yy^n
    # ========================================================
    Ly_u = np.zeros_like(u_old)

    for i in range(1, m):
        row = u_old[i, 1:m]
        Ly1 = L1D @ row
        Ly_u[i, 1:m] = b_grid[i, 1:m] * Ly1

    # ========================================================
    # RHS FOR X-SWEEP
    # ========================================================
    rhs1 = np.zeros_like(u_old)
    rhs1[1:m, 1:m] = (
        u_old[1:m, 1:m]
        + mu * Ly_u[1:m, 1:m]
        + k * NL[1:m, 1:m]
        + (k/2.0) * (s_old[1:m, 1:m] + s_new[1:m, 1:m])
    )

    # ========================================================
    # X-SWEEP: implicit in x
    # ========================================================
    u_half = np.zeros_like(u_old)

    for j in range(1, m):
        rhs_col = rhs1[1:m, j]
        a_col = a_grid[1:m, j]
        Mx = I1D - mu * (np.diag(a_col) @ L1D)
        u_half[1:m, j] = np.linalg.solve(Mx, rhs_col)

    # BCs on intermediate solution
    u_half[0,:] = u_half[-1,:] = 0.0
    u_half[:,0] = u_half[:,-1] = 0.0

    # ========================================================
    # EXPLICIT X-DIRECTION OPERATOR  a(x,y) u_xx^*
    # ========================================================
    Lx_u = np.zeros_like(u_half)

    for j in range(1, m):
        col = u_half[1:m, j]
        Lx1 = L1D @ col
        Lx_u[1:m, j] = a_grid[1:m, j] * Lx1

    # ========================================================
    # RHS FOR Y-SWEEP
    # ========================================================
    rhs2 = np.zeros_like(u_half)
    rhs2[1:m, 1:m] = (
        u_half[1:m, 1:m]
        + mu * Lx_u[1:m, 1:m]
    )

    # ========================================================
    # Y-SWEEP: implicit in y
    # ========================================================
    u_new = np.zeros_like(u_old)

    for i in range(1, m):
        rhs_row = rhs2[i, 1:m]
        b_row = b_grid[i, 1:m]
        My = I1D - mu * (np.diag(b_row) @ L1D)
        u_new[i, 1:m] = np.linalg.solve(My, rhs_row)

    # BCs at t_{n+1}
    u_new[0,:] = u_new[-1,:] = 0.0
    u_new[:,0] = u_new[:,-1] = 0.0

    # advance in time
    u_older = u_old
    u_old   = u_new

# ============================================================
# ERROR COMPUTATION
# ============================================================

u_exact = np.exp(-T) * np.sin(np.pi * X) * np.sin(np.pi * Y)
err_L2 = np.sqrt(h*h * np.sum((u_old - u_exact)**2))

print("L2 error =", err_L2)

# ============================================================
# MID-LINE PLOT (y = 0.5)
# ============================================================

j_mid = np.argmin(np.abs(y - 0.5))

plt.figure(figsize=(7,4))
plt.plot(x, u_exact[:, j_mid], 'k-', lw=2, label='Exact')
plt.plot(x, u_old[:, j_mid], 'ro', label='Numerical')
plt.xlabel('x')
plt.ylabel('u(x,0.5,T)')
plt.title('Mid-line solution at T = 1')
plt.legend()
plt.grid(ls='--', lw=0.4)
plt.show()
