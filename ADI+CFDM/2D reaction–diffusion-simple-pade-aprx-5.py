import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------
# Simple flowing script (no def)
# Domain: [0, pi] x [0, pi], T = 1.0
# Exact: u = exp(-t) * sin(x + pi/4) * cos(2y)
# Coeffs: a = 0.5*(x+y+1), b = 2/sqrt(x^2+y^2+1)
# ---------------------------

Lx = 1
# Lx = np.pi
Ly = 1

# Ly = np.pi
T_final = 1.0

# choose a single simple h
# h = np.pi / 20

# # build grid counts and adjusted h to fit exactly
# m = int(round(Lx / h))
# if m < 2:
#     raise ValueError("h too large; choose smaller h")
# h = Lx / m   # ensure exact partitioning
# print("Using h =", h, "with m =", m)

# # time stepping
# k = 1e-4                # small time step so spatial error dominates
# N = int(np.ceil(T_final / k))
# k = T_final / N         # adjust k so final time matches exactly
# 
# print("Time step k =", k, "N steps =", N, "mu =", mu)

m = 20
h = 1/m
T = 1
k = 1/1000
N = int(T/k)
mu = k / 2.0
# coordinates and grids
x = np.linspace(0.0, Lx, m+1)
y = np.linspace(0.0, Ly, m+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# coefficient grids (variable)
a_grid = 0.5 * (X + Y + 1.0)                     # a(x,y)
b_grid = 2.0 / np.sqrt(X**2 + Y**2 + 1.0)        # b(x,y)

# exact solution (used for BCs and final error)
# u_exact_full = np.exp(-0.0) * np.sin(X + np.pi/4.0) * np.cos(2.0 * Y)  # at t=0 (prefill shape)
u_exact_full = np.exp(-0.0) * np.sin(X*np.pi) * np.sin(np.pi*Y)  # at t=0 (prefill shape)

# helper: build 1D compact L operator for interior points (dense)
interior = m - 1
if interior < 1:
    raise ValueError("Need at least one interior point")
off = np.ones(interior - 1)
diag = -2.0 * np.ones(interior)
D2 = (1.0 / h**2) * diags([off, diag, off], offsets=[-1, 0, 1]).toarray()
L1D = D2 - (h**2 / 12.0) * (D2 @ D2)   # 4th-order compact correction
I1D = np.eye(interior)

# initial condition (exact at t=0)
# u_old = np.exp(-0.0) * np.sin(X + np.pi/4) * np.cos(2.0 * Y)
u_old = np.exp(-0.0) * np.sin(X * np.pi) * np.sin (Y* np.pi)


# # enforce Dirichlet BCs (from exact)
# u_old[0, :] = np.exp(-0.0) * np.sin(x[0] + np.pi/4.0) * np.cos(2.0 * y)
# u_old[m, :] = np.exp(-0.0) * np.sin(x[-1] + np.pi/4.0) * np.cos(2.0 * y)
# u_old[:, 0] = np.exp(-0.0) * np.sin(x + np.pi/4.0) * np.cos(2.0 * y[0])
# u_old[:, m] = np.exp(-0.0) * np.sin(x + np.pi/4.0) * np.cos(2.0 * y[-1])

# enforce Dirichlet BCs (from exact)
u_old[0, :] = np.exp(-0.0) * np.sin(x[0]*np.pi) * np.sin(np.pi* y)
u_old[m, :] = np.exp(-0.0) * np.sin(x[-1]*np.pi) * np.sin(np.pi * y)
u_old[:, 0] = np.exp(-0.0) * np.sin(x*np.pi) * np.sin(np.pi* y[0])
u_old[:, m] = np.exp(-0.0) * np.sin(x *np.pi) * np.sin(np.pi* y[-1])





# time loop
for n in range(1, N+1):
    t_old = (n-1) * k
    t_new = n * k

    # # compute forcing at t_old and t_new using factored expression
    # S = np.sin(X + np.pi/4.0) * np.cos(2.0 * Y)         # spatial factor
    # pref_old = np.exp(-t_old) * S
    # pref_new = np.exp(-t_new) * S
    # f_old = pref_old * (-1.0 + a_grid + 4.0 * b_grid)   # f = exp(-t)*S*(-1 + a + 4b)
    # f_new = pref_new * (-1.0 + a_grid + 4.0 * b_grid)

    # compute forcing at t_old and t_new using factored expression
    S = np.sin(X*np.pi) * np.sin(np.pi*Y)         # spatial factor
    pref_old = np.exp(-t_old) * S
    pref_new = np.exp(-t_new) * S
    f_old = pref_old * (-1.0 + a_grid*np.pi**2 + np.pi**2 * b_grid)   # f = exp(-t)*S*(-1 + a + 4b)
    f_new = pref_new * (-1.0 + a_grid*np.pi**2  + np.pi**2* b_grid)

    # --- compute Ly(u_old) : apply L1D along y for each fixed x ---
    Ly_u = np.zeros_like(u_old)
    for i in range(1, m):                       # for each interior x-row
        row_int = u_old[i, 1:m]                 # interior points along y
        Ly1 = L1D @ row_int
        Ly_u[i, 1:m] = b_grid[i, 1:m] * Ly1

    # RHS for X-sweep (interior only)
    rhs1 = np.zeros_like(u_old)
    rhs1[1:m, 1:m] = u_old[1:m, 1:m] + mu * Ly_u[1:m, 1:m] + (k/2.0) * f_old[1:m, 1:m]

    # --- X-sweep: solve for u_half by columns ---
    u_half = np.zeros_like(u_old)
    for j in range(1, m):                       # for each interior column (fixed y)
        rhs_col = rhs1[1:m, j].copy() #(RHS)
        
        #matrix formation
        a_col = a_grid[1:m, j]
        A_col = np.diag(a_col) @ L1D
        Mx = I1D - mu * A_col
        # print(Mx)
        u_half[1:m, j] = np.linalg.solve(Mx, rhs_col)
       
    # set Dirichlet BCs on u_half from exact at t_new (keeps boundary error zero)
    u_half[0, :] = np.exp(-t_new) * np.sin(x[0]+np.pi/4) * np.cos( 2* y)
    u_half[m, :] = np.exp(-t_new) * np.sin(x[-1]+ np.pi/4) * np.cos( y*2)
    u_half[:, 0] = np.exp(-t_new) * np.sin(x + np.pi/4) * np.cos(y[0]*2)
    u_half[:, m] = np.exp(-t_new) * np.sin(x +np.pi/4) * np.cos(y[-1]* 2)
    
    
    # set Dirichlet BCs on u_half from exact at t_new (keeps boundary error zero)
    u_half[0, :] = np.exp(-t_new) * np.sin(x[0]* np.pi) * np.sin( np.pi* y)
    u_half[m, :] = np.exp(-t_new) * np.sin(x[-1]* np.pi) * np.sin( y* np.pi)
    u_half[:, 0] = np.exp(-t_new) * np.sin(x * np.pi) * np.sin(y[0]* np.pi)
    u_half[:, m] = np.exp(-t_new) * np.sin(x* np.pi) * np.sin(y[-1]* np.pi)

    # --- compute Lx(u_half) : apply L1D along x for each fixed y ---
    Lx_u = np.zeros_like(u_half)
    for j in range(1, m):                       # for interior columns
        col_int = u_half[1:m, j]
        Lx1 = L1D @ col_int
        Lx_u[1:m, j] = a_grid[1:m, j] * Lx1
        
   
    # RHS for Y-sweep
    rhs2 = np.zeros_like(u_half)
    rhs2[1:m, 1:m] = u_half[1:m, 1:m] + mu * Lx_u[1:m, 1:m] + (k/2.0) * f_new[1:m, 1:m]

    # --- Y-sweep: solve for u_new by rows ---
    u_new = np.zeros_like(u_old)
    for i in range(1, m):                       # for each interior row (fixed x)
        rhs_row = rhs2[i, 1:m].copy() #(RHS)
        #matrix formation
        b_row = b_grid[i, 1:m]
        B_row = np.diag(b_row) @ L1D
        My = I1D - mu * B_row
        u_new[i, 1:m] = np.linalg.solve(My, rhs_row)
        
        
    # # enforce Dirichlet BCs at t_new from exact
    # u_new[0, :] = np.exp(-t_new) * np.sin(x[0]+np.pi/4) * np.cos( 2* y)
    # u_new[m, :] = np.exp(-t_new) * np.sin(x[-1]+ np.pi/4) * np.cos( y*2)
    # u_new[:, 0] = np.exp(-t_new) * np.sin(x + np.pi/4) * np.cos(y[0]*2)
    # u_new[:, m] = np.exp(-t_new) * np.sin(x +np.pi/4) * np.cos(y[-1]* 2)

    # enforce Dirichlet BCs at t_new from exact
    u_new[0, :] = np.exp(-t_new) * np.sin(x[0]* np.pi) * np.sin( y* np.pi)
    u_new[m, :] = np.exp(-t_new) * np.sin(x[-1] * np.pi) * np.sin(y* np.pi)
    u_new[:, 0] = np.exp(-t_new) * np.sin(x * np.pi) * np.sin(y[0]* np.pi)
    u_new[:, m] = np.exp(-t_new) * np.sin(x * np.pi) * np.sin(y[-1]* np.pi)

    # advance
    u_old = u_new

# after time-stepping: compute errors and plot
u_num = u_old

# u_exact_final = np.exp(-T_final) * np.sin(X + np.pi/4) * np.cos(Y * 2)

u_exact_final = np.exp(-T_final) * np.sin(X * np.pi) * np.sin(Y * np.pi)

err_inf = np.max(np.abs(u_num - u_exact_final))
err_l2 = np.sqrt(h*h * np.sum((u_num - u_exact_final)**2))
print("Final errors: L2 =", err_l2)

# convergence-ish: (single h only, so just print errors)
# Plot numerical vs exact and error

# # 3D surface numerical and exact
# fig = plt.figure(figsize=(12,5))
# ax1 = fig.add_subplot(1,3,1, projection='3d')
# ax1.plot_surface(X, Y, u_num, rstride=1, cstride=1, linewidth=0, antialiased=True)
# ax1.set_title('Numerical')
# ax1.set_xlabel('x'); ax1.set_ylabel('y')

# ax2 = fig.add_subplot(1,3,2, projection='3d')
# ax2.plot_surface(X, Y, u_exact_final, rstride=1, cstride=1, linewidth=0, antialiased=True)
# ax2.set_title('Exact')
# ax2.set_xlabel('x'); ax2.set_ylabel('y')

# ax3 = fig.add_subplot(1,3,3)
# err2d = np.abs(u_num - u_exact_final)
# pcm = ax3.pcolormesh(X, Y, err2d, shading='auto')
# plt.colorbar(pcm, ax=ax3)
# ax3.set_title('Absolute error (heatmap)')
# ax3.set_xlabel('x'); ax3.set_ylabel('y')
# plt.tight_layout()
# plt.show()

# 1D mid-line plot at y = pi/2
y_mid = 1 * 0.5
y_grid = Y[0, :]
j_mid = np.argmin(np.abs(y_grid - y_mid))
x_grid = X[:, 0]
plt.figure(figsize=(7,4))
plt.plot(x_grid, u_exact_final[:, j_mid], '-k', lw=1.5, label='Exact')
plt.plot(x_grid, u_num[:, j_mid], 'o', label='Numerical ')
plt.xlabel('x'); plt.ylabel('u(x , T)')
plt.title(f'Line plot , T={T_final}, h={h:.3e}')
plt.legend(); plt.grid(ls='--', lw=0.4)
plt.show()
