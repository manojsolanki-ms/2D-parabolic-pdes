import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# -------------------------------------
# Parameters
# -------------------------------------
a = 1.0
b = 1.0
Lx = 1.0
Ly = 1.0

T_final = 0.01          # final time
dt      = 1.0e-5        # VERY SMALL to see spatial 4th order

N_list = [10, 20, 40, 80]   # refine N; keep dt fixed

errors = []
hs      = []

for N in N_list:
    Nx = N
    Ny = N
    h  = Lx / Nx        # assume Lx = Ly and hx = hy
    hs.append(h)

    alpha = a * dt / (2.0 * h**2)
    beta  = b * dt / (2.0 * h**2)

    Nt = int(round(T_final / dt))

    # -------------------------------------
    # Grid & initial condition
    # -------------------------------------
    x = np.linspace(0.0, Lx, Nx + 1)
    y = np.linspace(0.0, Ly, Ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Dirichlet boundary (homogeneous)
    u[0, :]  = 0.0
    u[-1, :] = 0.0
    u[:, 0]  = 0.0
    u[:, -1] = 0.0

    # -------------------------------------
    # Time stepping: CFDM–ADI (Case 1)
    # -------------------------------------
    for n in range(Nt):

        # ============================
        # Step 0: compute tilde u^n in y
        # (Eq. (12) and following tridiagonal system)
        # ============================
        tilde = np.zeros_like(u)

        u_over_beta = u / beta

        # δ_y^2(u/β)
        d2_y_u_over_beta = np.zeros_like(u)
        d2_y_u_over_beta[:, 1:Ny] = (
            u_over_beta[:, 2:Ny+1]
            - 2.0 * u_over_beta[:, 1:Ny]
            + u_over_beta[:, 0:Ny-1]
        )

        term1 = u_over_beta + (1.0 / 12.0) * d2_y_u_over_beta

        # δ_y^2 u
        d2_y_u = np.zeros_like(u)
        d2_y_u[:, 1:Ny] = (
            u[:, 2:Ny+1]
            - 2.0 * u[:, 1:Ny]
            + u[:, 0:Ny-1]
        )

        R = term1 + d2_y_u   # right-hand side inside Eq. (12) form

        # For each i, solve: tilde_{j-1} + 10 tilde_j + tilde_{j+1} = 12 β R_j
        for i in range(1, Nx):
            n_unknown = Ny - 1
            d = (12.0 * beta * R[i, 1:Ny]).copy()

            a_sub = np.ones(n_unknown)      # subdiagonal
            a_diag = np.ones(n_unknown)*10  # diagonal
            a_sup = np.ones(n_unknown)      # superdiagonal

            a_sub[0]  = 0.0
            a_sup[-1] = 0.0

            # Thomas forward elimination
            for j in range(1, n_unknown):
                w = a_sub[j] / a_diag[j-1]
                a_diag[j] -= w * a_sup[j-1]
                d[j]      -= w * d[j-1]

            # Thomas back substitution
            sol = np.zeros_like(d)
            sol[-1] = d[-1] / a_diag[-1]
            for j in range(n_unknown-2, -1, -1):
                sol[j] = (d[j] - a_sup[j]*sol[j+1]) / a_diag[j]

            tilde[i, 1:Ny] = sol

        # boundaries of tilde are 0 for homogeneous BC
        tilde[0, :]  = 0.0
        tilde[-1, :] = 0.0
        tilde[:, 0]  = 0.0
        tilde[:, -1] = 0.0

        # ============================
        # Step 1: X-sweep (solve for u_star)
        # ============================
        u_star = np.zeros_like(u)

        tilde_over_alpha = tilde / alpha

        # δ_x^2(tilde/α)
        d2_x_t_over_alpha = np.zeros_like(tilde)
        d2_x_t_over_alpha[1:Nx, :] = (
            tilde_over_alpha[2:Nx+1, :]
            - 2.0 * tilde_over_alpha[1:Nx, :]
            + tilde_over_alpha[0:Nx-1, :]
        )

        termR1 = tilde_over_alpha + (1.0 / 12.0) * d2_x_t_over_alpha

        # δ_x^2 tilde
        d2_x_t = np.zeros_like(tilde)
        d2_x_t[1:Nx, :] = (
            tilde[2:Nx+1, :]
            - 2.0 * tilde[1:Nx, :]
            + tilde[0:Nx-1, :]
        )

        RHS_x = termR1 + d2_x_t

        # Tridiagonal coefficients in x:
        # (1/(12α)-1) u_{i-1} + (10/(12α)+2) u_i + (1/(12α)-1) u_{i+1} = RHS
        a_sub_base = (1.0 / (12.0 * alpha)) - 1.0
        a_diag_base = (10.0 / (12.0 * alpha)) + 2.0
        a_sup_base = a_sub_base

        for j in range(1, Ny):
            n_unknown = Nx - 1
            d = RHS_x[1:Nx, j].copy()

            a_sub = np.zeros(n_unknown)
            a_diag = np.zeros(n_unknown)
            a_sup = np.zeros(n_unknown)

            a_diag[:] = a_diag_base
            a_sub[1:] = a_sub_base
            a_sup[:-1] = a_sup_base

            # Thomas forward
            for i in range(1, n_unknown):
                w = a_sub[i] / a_diag[i-1]
                a_diag[i] -= w * a_sup[i-1]
                d[i]      -= w * d[i-1]

            # Thomas back
            sol = np.zeros_like(d)
            sol[-1] = d[-1] / a_diag[-1]
            for i in range(n_unknown-2, -1, -1):
                sol[i] = (d[i] - a_sup[i]*sol[i+1]) / a_diag[i]

            u_star[1:Nx, j] = sol

        # boundaries of u_star from Dirichlet:
        u_star[0, :]  = 0.0
        u_star[-1, :] = 0.0
        u_star[:, 0]  = 0.0
        u_star[:, -1] = 0.0

        # ============================
        # Step 2: Y-sweep (solve for u^{n+1})
        # ============================
        u_new = np.zeros_like(u)

        ustar_over_beta = u_star / beta

        # δ_y^2(u_star/β)
        d2_y_ustar_over_beta = np.zeros_like(u_star)
        d2_y_ustar_over_beta[:, 1:Ny] = (
            ustar_over_beta[:, 2:Ny+1]
            - 2.0 * ustar_over_beta[:, 1:Ny]
            + ustar_over_beta[:, 0:Ny-1]
        )

        RHS_y_full = ustar_over_beta + (1.0 / 12.0) * d2_y_ustar_over_beta

        # Tridiagonal coefficients in y:
        base_y = (1.0 / (12.0 * beta)) - 1.0
        diag_y = (10.0 / (12.0 * beta)) + 2.0

        for i in range(1, Nx):
            n_unknown = Ny - 1
            d = RHS_y_full[i, 1:Ny].copy()

            a_sub = np.zeros(n_unknown)
            a_diag = np.zeros(n_unknown)
            a_sup = np.zeros(n_unknown)

            a_diag[:] = diag_y
            a_sub[1:] = base_y
            a_sup[:-1] = base_y

            # Thomas forward
            for j in range(1, n_unknown):
                w = a_sub[j] / a_diag[j-1]
                a_diag[j] -= w * a_sup[j-1]
                d[j]      -= w * d[j-1]

            # Thomas back
            sol = np.zeros_like(d)
            sol[-1] = d[-1] / a_diag[-1]
            for j in range(n_unknown-2, -1, -1):
                sol[j] = (d[j] - a_sup[j]*sol[j+1]) / a_diag[j]

            u_new[i, 1:Ny] = sol

        # apply BC
        u_new[0, :]  = 0.0
        u_new[-1, :] = 0.0
        u_new[:, 0]  = 0.0
        u_new[:, -1] = 0.0

        u = u_new

    # -------------------------------------
    # Exact solution & error at T_final
    # -------------------------------------
    u_exact = math.exp(-(a + b) * math.pi**2 * T_final) \
              * np.sin(math.pi * X) * np.sin(math.pi * Y)

    err = np.max(np.abs(u - u_exact))
    errors.append(err)
    print(f"N = {N:3d}, h = {h:.5e}, max error = {err:.5e}")

# -----------------------------------------
# Compute observed spatial order
# -----------------------------------------
print("\nObserved spatial orders (using max-norm errors):")
for k in range(1, len(N_list)):
    e1 = errors[k-1]
    e2 = errors[k]
    h1 = hs[k-1]
    h2 = hs[k]
    p  = math.log(e1 / e2) / math.log(h1 / h2)
    print(f"between N={N_list[k-1]} and N={N_list[k]}: p ≈ {p:.3f}")

# -----------------------------------------
# Plots: numerical vs exact (for finest grid), and error vs h
# -----------------------------------------

# 3D plots for finest grid (last N in the loop)
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, u, rstride=1, cstride=1)
ax1.set_title(f"Numerical solution (N = {N_list[-1]}, t = {T_final})")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, u_exact, rstride=1, cstride=1)
ax2.set_title(f"Exact solution (N = {N_list[-1]}, t = {T_final})")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
    
plt.tight_layout()
plt.show()

# # Log-log plot of error vs h to visualize order
# plt.figure()
# plt.loglog(hs, errors, 'o-', basex=10, basey=10)
# for i in range(len(hs)):
#     plt.text(hs[i], errors[i]*1.1, f"N={N_list[i]}", fontsize=8)

# plt.gca().invert_xaxis()  # smaller h to the right or left; optional
# plt.xlabel("h (grid size)")
# plt.ylabel("max-norm error")
# plt.title("Error vs h (log-log)")
# plt.grid(True, which="both", ls="--")
# plt.show()
