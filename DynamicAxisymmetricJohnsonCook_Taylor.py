"""
Axisymmetric Taylor impact test, Johnson-Cook thermo-viscoplasticity (dolfinx 0.9).

Reproduces Fig. 10 / Table 3 of
  L. Ming & O. Pantale (2018), "An efficient and robust VUMAT implementation of
  elastoplastic constitutive laws in Abaqus/Explicit", Mech. & Industry 19, 308.

  * 42CrMo4 steel (Table 1), rod R = 3.2 mm, L = 32.4 mm, 5 x 50 elements,
    V0 = 287 m/s onto a rigid frictionless wall, t_end = 80 us, adiabatic.
  * Implicit Newmark (trapezoidal) in time, P2 displacement, 2x2 Gauss points.
  * Updated Lagrangian: mesh moved every step, Jaumann rotation of the stress.
  * Radial return with a safeguarded Newton (Newton + bisection) 
  * Hypoelastic pressure p1 = p0 + K tr(d_eps) (paper Eq. 13). A Mie-Gruneisen
    EOS is kept as an option (USE_EOS) for later use with real steel Hugoniot data.

Run:  python DynamicAxisymmetricJohnsonCook_Taylor.py [Nincr]
      mpirun -n 4 python DynamicAxisymmetricJohnsonCook_Taylor.py 4000
"""
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import basix
import ufl
from dolfinx import fem, geometry, io, mesh
import dolfinx.fem.petsc

comm = MPI.COMM_WORLD
rank = comm.rank

# ----------------------------------------------------------------------------
# Geometry / mesh  (x[0] = r, x[1] = z ; impact face at z = 0)
# ----------------------------------------------------------------------------
length, height = 3.2e-3, 32.4e-3
Nx, Ny = 5, 50
domain = mesh.create_rectangle(
    comm, [[0.0, 0.0], [length, height]], [Nx, Ny], cell_type=mesh.CellType.quadrilateral
)
tdim = domain.topology.dim
fdim = tdim - 1
gdim = 2

left = lambda x: np.isclose(x[0], 0.0)
bottom = lambda x: np.isclose(x[1], 0.0)

# ----------------------------------------------------------------------------
# Material: 42CrMo4 (Table 1 of the paper)
# ----------------------------------------------------------------------------
E_, nu_ = 206.9e9, 0.29
G_ = E_ / (2 * (1 + nu_))
K_ = E_ / (3 * (1 - 2 * nu_))
rho = 7830.0                       # kg/m3  (was 8930 = copper)
cp, chi = 460.0, 0.9               # J/kg/K, Taylor-Quinney
JC_A, JC_B, JC_n = 806e6, 614e6, 0.168
JC_C, JC_m = 0.0089, 1.1
eps0dot = 1.0
Tref, Tmelt = 20.0, 1540.0
v0 = 287.0                         # impact velocity, m/s

USE_EOS = False                    # True -> Mie-Gruneisen pressure (needs steel data!)
C0_eos, S1_eos, gamma0_eos = 4570.0, 1.49, 2.17   # placeholders for iron/steel

mu = fem.Constant(domain, G_)
Kc = fem.Constant(domain, K_)

# ----------------------------------------------------------------------------
# Function spaces
# ----------------------------------------------------------------------------
deg_u, deg_quad = 2, 2
V = fem.functionspace(domain, ("P", deg_u, (gdim,)))
We = basix.ufl.quadrature_element(domain.basix_cell(), value_shape=(4,), scheme="default", degree=deg_quad)
W0e = basix.ufl.quadrature_element(domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad)
W = fem.functionspace(domain, We)
W0 = fem.functionspace(domain, W0e)

left_facets = mesh.locate_entities_boundary(domain, fdim, left)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
left_dofs_x = fem.locate_dofs_topological(V.sub(0), fdim, left_facets)
bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), fdim, bottom_facets)
bcs = [
    fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs_x, V.sub(0)),    # symmetry axis
    fem.dirichletbc(PETSc.ScalarType(0.0), bottom_dofs_y, V.sub(1)),  # rigid wall
]

# state at Gauss points
sig = fem.Function(W, name="Stress")        # [rr, tt, zz, rz]
sig_old = fem.Function(W)
n_elas = fem.Function(W)
beta = fem.Function(W0)
Hp = fem.Function(W0)
p = fem.Function(W0, name="Equivalent_plastic_strain")
T = fem.Function(W0, name="Temperature")
theta = fem.Function(W0)                    # total log volumetric strain (EOS only)

u = fem.Function(V, name="Displacement")
Du = fem.Function(V, name="Increment")
du = fem.Function(V)
u_old, ve_old, a_old = fem.Function(V), fem.Function(V), fem.Function(V)
ve_new, a_new = fem.Function(V), fem.Function(V)
v_, u_ = ufl.TrialFunction(V), ufl.TestFunction(V)

x = ufl.SpatialCoordinate(domain)           # follows the moving mesh


def grad_axi(v):
    return ufl.as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                          [0, v[0] / x[0], 0],
                          [v[1].dx(0), 0, v[1].dx(1)]])


def eps(v):
    return ufl.sym(grad_axi(v))


def as_3D_tensor(X):
    return ufl.as_tensor([[X[0], 0, X[3]], [0, X[1], 0], [X[3], 0, X[2]]])


def to_vect(X):
    return ufl.as_vector([X[0, 0], X[1, 1], X[2, 2], X[0, 2]])


# ----------------------------------------------------------------------------
# Newmark (beta = 1/4, gamma = 1/2)
# ----------------------------------------------------------------------------
bN, gN = 0.25, 0.5
dt = fem.Constant(domain, 0.0)
a_Du = 1 / bN / dt**2 * Du - 1 / bN / dt * ve_old - (1 / (2 * bN) - 1) * a_old
a_expr = fem.Expression(1 / bN / dt**2 * (u - u_old - dt * ve_old) + a_old * (1 - 1 / 2 / bN),
                        V.element.interpolation_points())
ve_expr = fem.Expression(ve_old + dt * ((1 - gN) * a_old + gN * (1 / bN / dt**2 * (u - u_old - dt * ve_old)
                                                                 + a_old * (1 - 1 / 2 / bN))),
                         V.element.interpolation_points())

dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

dxm = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})


def sigma_tang(e):
    N = as_3D_tensor(n_elas)
    return (Kc * ufl.tr(e) * ufl.Identity(3) + 2 * mu * ufl.dev(e)
            - 3 * mu * (3 * mu / (3 * mu + Hp) - beta) * ufl.inner(N, e) * N
            - 2 * mu * beta * ufl.dev(e))


Residual = rho * ufl.dot(a_Du, u_) * x[0] * dxm + ufl.inner(eps(u_), as_3D_tensor(sig)) * x[0] * dx
tangent_form = (rho / bN / dt**2 * ufl.inner(v_, u_) * x[0] * dxm
                + ufl.inner(eps(v_), sigma_tang(eps(u_))) * x[0] * dx)

# ----------------------------------------------------------------------------
# Trial state: Jaumann-rotated deviator + elastic predictor
# ----------------------------------------------------------------------------
qp, _ = basix.make_quadrature(getattr(basix.CellType, domain.topology.cell_type.name), deg_quad)
nq = qp.shape[0]
imap = domain.topology.index_map(tdim)
cells = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)

Wspin = ufl.skew(grad_axi(Du))
sig_n = as_3D_tensor(sig_old)
sig_n = sig_n + ufl.dot(Wspin, sig_n) - ufl.dot(sig_n, Wspin)   # Jaumann rotation
s_trial_ufl = to_vect(ufl.dev(sig_n) + 2 * mu * ufl.dev(eps(Du)))
trial_expr = fem.Expression(ufl.as_vector([s_trial_ufl[0], s_trial_ufl[1], s_trial_ufl[2], s_trial_ufl[3],
                                           ufl.tr(sig_n) / 3.0, ufl.tr(eps(Du))]), qp)


def eval_trial():
    v = trial_expr.eval(domain, cells).reshape(-1, 6)    # one row per Gauss point
    return v[:, :4], v[:, 4], v[:, 5]


# ----------------------------------------------------------------------------
# Johnson-Cook law 
# ----------------------------------------------------------------------------
def jc_yield(pl, rate, Tk):
    f1 = JC_A + JC_B * np.maximum(pl, 0.0) ** JC_n
    f2 = np.where(rate >= eps0dot, 1.0 + JC_C * np.log(np.maximum(rate, eps0dot) / eps0dot), 1.0)
    Ts = np.clip((Tk - Tref) / (Tmelt - Tref), 0.0, 1.0)
    return f1 * f2 * (1.0 - Ts**JC_m)


heat = chi / (rho * cp)


def yield_of_dp(dp, p0, T0, sy0, h):
    """sigma_y at end of step as a function of dp (temperature: trapezoidal rule, Eq. 17)."""
    sy_T0 = jc_yield(p0 + dp, dp / h, T0)
    T1 = T0 + 0.5 * heat * (sy0 + sy_T0) * dp
    sy1 = jc_yield(p0 + dp, dp / h, T1)
    T1 = T0 + 0.5 * heat * (sy0 + sy1) * dp            
    return jc_yield(p0 + dp, dp / h, T1), T1


def radial_return(seq, p0, T0, sy0, h, tol=1e-12, itmax=60):
    """Safe Newton (Newton + bisection, paper Fig. 1) on dp for f(dp)=seq-3G dp-sy(dp)=0."""
    dp = np.zeros_like(seq)
    sy1 = jc_yield(p0, 0.0 * p0, T0)
    T1 = T0.copy()
    plastic = seq > sy1
    if not plastic.any():
        return dp, sy1, T1, np.zeros_like(seq), plastic
    s, P0, Tt, S0 = seq[plastic], p0[plastic], T0[plastic], sy0[plastic]
    lo, hi = np.zeros_like(s), s / (3 * G_)
    xk = 0.5 * (lo + hi)

    def f(z):
        return s - 3 * G_ * z - yield_of_dp(z, P0, Tt, S0, h)[0]

    for _ in range(itmax):
        fx = f(xk)
        lo = np.where(fx > 0, xk, lo)
        hi = np.where(fx < 0, xk, hi)
        hstep = np.maximum(1e-7 * xk, 1e-14)
        dfx = (f(xk + hstep) - fx) / hstep
        xn = xk - fx / np.where(dfx < 0, dfx, -3 * G_)
        bad = (xn <= lo) | (xn >= hi) | ~np.isfinite(xn)
        xn = np.where(bad, 0.5 * (lo + hi), xn)
        if np.all(np.abs(xn - xk) <= tol * np.maximum(hi, 1e-30)):
            xk = xn
            break
        xk = xn
    syk, Tk = yield_of_dp(xk, P0, Tt, S0, h)
    hstep = np.maximum(1e-7 * xk, 1e-14)
    Hk = (yield_of_dp(xk + hstep, P0, Tt, S0, h)[0] - syk) / hstep   # d sigma_y / d dp (consistent)
    dp[plastic], sy1[plastic], T1[plastic] = xk, syk, Tk
    H = np.zeros_like(seq)
    H[plastic] = np.maximum(Hk, 0.0)
    return dp, sy1, T1, H, plastic


def mie_gruneisen(theta_v, e):
    J = np.exp(theta_v)
    mc = 1.0 / J - 1.0
    den = 1.0 - (S1_eos - 1.0) * mc
    pc = np.where(mc >= 0, rho * C0_eos**2 * mc * (1 + (1 - gamma0_eos / 2) * mc) / den**2, rho * C0_eos**2 * mc)
    return pc + gamma0_eos * e



sy_old = None         
state = {}


def constitutive_update(store=False):
    """Compute sig, n_elas, beta, Hp from the current Du; optionally commit p, T."""
    s_tr, pm_old, trde = eval_trial()
    seq = np.sqrt(1.5 * (s_tr[:, 0]**2 + s_tr[:, 1]**2 + s_tr[:, 2]**2 + 2 * s_tr[:, 3]**2))
    p0, T0 = p.x.array.copy(), T.x.array.copy()
    dp, sy1, T1, H, plastic = radial_return(seq, p0, T0, sy_old, dt.value)

    seq_safe = np.maximum(seq, 1.0)
    bet = np.where(plastic, 3 * G_ * dp / seq_safe, 0.0)
    s_new = (1.0 - bet)[:, None] * s_tr

    if USE_EOS:
        th = theta.x.array + trde
        pm_new = -mie_gruneisen(th, rho * cp * (T1 - Tref))
    else:
        th = theta.x.array + trde
        pm_new = pm_old + K_ * trde                       # mean stress (tension +)

    sig_new = s_new.copy()
    sig_new[:, :3] += pm_new[:, None]
    sig.x.array[:] = sig_new.ravel()
    n_elas.x.array[:] = (np.where(plastic, 1.0, 0.0)[:, None] * s_tr / seq_safe[:, None]).ravel()
    beta.x.array[:] = bet
    Hp.x.array[:] = H
    if store:
        state.update(dp=dp, sy1=sy1, T1=T1, th=th)


# ----------------------------------------------------------------------------
# Linear solver wrapper
# ----------------------------------------------------------------------------
class CustomLinearProblem(fem.petsc.LinearProblem):
    def assemble_rhs(self, u=None):
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector(self._b, self._L)
        x0 = [] if u is None else [u.x.petsc_vec]
        fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs], x0=x0, alpha=-1.0)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self._b, self.bcs, None if u is None else u.x.petsc_vec)

    def assemble_lhs(self):
        self._A.zeroEntries()
        fem.petsc.assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

    def solve_system(self):
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()


tangent_problem = CustomLinearProblem(
    tangent_form, -Residual, u=du, bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
)

# ----------------------------------------------------------------------------
# Mesh update helpers (updated Lagrangian)
# ----------------------------------------------------------------------------
Vg = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
Du_g = fem.Function(Vg)
gdofs = domain.geometry.dofmap.reshape(-1)
vdofs = Vg.dofmap.list.reshape(-1)


def move_mesh(disp):
    Du_g.interpolate(disp)
    Du_g.x.scatter_forward()
    shift = np.zeros((domain.geometry.x.shape[0], gdim))
    shift[gdofs] = Du_g.x.array.reshape(-1, gdim)[vdofs]
    domain.geometry.x[:, :gdim] += shift


# centre element of the impact face (r ~ 0, z ~ 0); identity is kept (Lagrangian)
pt = np.array([[0.5 * length / Nx, 0.5 * height / Ny, 0.0]])
tree = geometry.bb_tree(domain, tdim)
coll = geometry.compute_colliding_cells(domain, geometry.compute_collisions_points(tree, pt), pt).array
centre = int(coll[0]) if len(coll) and coll[0] < imap.size_local else -1


def centre_value(arr):
    loc = arr[centre * nq:(centre + 1) * nq].mean() if centre >= 0 else -np.inf
    return comm.allreduce(loc, op=MPI.MAX)


# reaction force on the wall (2*pi for the full ring)
v_reac = fem.Function(V)
v_reac.x.array[bottom_dofs_y] = 1.0
reac_form = fem.form(ufl.replace(Residual, {u_: v_reac}))

# ----------------------------------------------------------------------------
# Time loop
# ----------------------------------------------------------------------------
Nincr = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
End_Time = 80e-6
dt.value = End_Time / Nincr
Nitermax, tol = 25, 1e-8

for f_ in (sig, sig_old, p, n_elas, beta, Hp, theta, u, u_old, a_old):
    f_.x.array[:] = 0.0
T.x.array[:] = Tref
sy_old = jc_yield(p.x.array, 0.0 * p.x.array, T.x.array)
ve_old.x.array[:] = 0.0
ve_old.x.array[1::2] = -v0                  
ve_old.x.array[bottom_dofs_y] = 0.0        
ve_old.x.scatter_forward()

outdir = Path("results")
if rank == 0:
    outdir.mkdir(exist_ok=True)
comm.Barrier()
P0 = fem.functionspace(domain, ("DG", 0))
p_avg = fem.Function(P0, name="Equivalent_plastic_strain")
T_avg = fem.Function(P0, name="Temperature")
vtk = io.VTKFile(comm, str(outdir / "taylor.pvd"), "w")

hist = [(0.0, 0.0, Tref, height, 2 * length, 0.0)]
for i in range(Nincr):
    t = (i + 1) * dt.value

    Du.x.array[:] = 0.0
    Du.x.scatter_forward()
    constitutive_update()
    tangent_problem.assemble_rhs(u=Du)
    nRes0 = tangent_problem._b.norm()
    nRes = nRes0
    for niter in range(Nitermax):
        tangent_problem.assemble_lhs()
        tangent_problem.solve_system()
        Du.x.petsc_vec.axpy(1.0, du.x.petsc_vec)
        Du.x.scatter_forward()
        constitutive_update()
        tangent_problem.assemble_rhs(u=Du)
        nRes = tangent_problem._b.norm()
        if nRes < tol * nRes0:
            break
    else:
        if rank == 0:
            print(f"  step {i+1}: Newton not converged, |R|/|R0| = {nRes/nRes0:.2e}", flush=True)

   
    constitutive_update(store=True)
    F_wall = 2 * np.pi * comm.allreduce(fem.assemble_scalar(reac_form), op=MPI.SUM)

    u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
    u.x.scatter_forward()
    ve_new.interpolate(ve_expr); ve_new.x.scatter_forward()
    a_new.interpolate(a_expr); a_new.x.scatter_forward()
    u_old.x.array[:] = u.x.array
    ve_old.x.array[:] = ve_new.x.array
    a_old.x.array[:] = a_new.x.array

    p.x.array[:] += state["dp"]
    T.x.array[:] = state["T1"]
    theta.x.array[:] = state["th"]
    sy_old = state["sy1"]
    sig_old.x.array[:] = sig.x.array

    move_mesh(Du)                                    # updated Lagrangian

    Lf = comm.allreduce(domain.geometry.x[:, 1].max(), op=MPI.MAX)
    on_wall = np.isclose(domain.geometry.x[:, 1], 0.0, atol=1e-9)
    Df = 2 * comm.allreduce(domain.geometry.x[on_wall, 0].max(initial=0.0), op=MPI.MAX)
    hist.append((t, centre_value(p.x.array), centre_value(T.x.array), Lf, Df, abs(F_wall)))

    if rank == 0 and (i + 1) % 100 == 0:
        print(f"step {i+1}/{Nincr}  t={t*1e6:6.2f} us  it={niter+1}  eps_p(centre)={hist[-1][1]:.4f}  "
              f"T={hist[-1][2]:7.2f} C  Lf={Lf*1e3:.3f} mm  Df={Df*1e3:.3f} mm", flush=True)
    if (i + 1) % (Nincr // 20) == 0:
        p_avg.x.array[:] = p.x.array.reshape(-1, nq).mean(axis=1)[: p_avg.x.array.size]
        T_avg.x.array[:] = T.x.array.reshape(-1, nq).mean(axis=1)[: T_avg.x.array.size]
        vtk.write_function([u, p_avg, T_avg], t)
vtk.close()

# ----------------------------------------------------------------------------
# Output + comparison with Fig. 10 / Table 3
# ----------------------------------------------------------------------------
if rank == 0:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = np.array(hist)
    np.savetxt(outdir / "taylor_history.csv", hist, delimiter=",",
               header="t_s,eps_p_centre,T_centre_C,Lf_m,Df_m,F_wall_N", comments="")
    # Fig. 10 of the paper, read off the plot (marker positions, approximate)
    fig10 = np.array([[0, 0], [5.5, 0.66], [11.0, 1.23], [16.5, 1.59], [22.0, 1.77],
                      [27.5, 1.80], [33, 1.80], [44, 1.80], [55, 1.80], [66, 1.80], [80, 1.81]])
    plt.figure(figsize=(6, 4))
    plt.plot(hist[:, 0] , hist[:, 1], "b-", label="Current Study")
    plt.plot(fig10[:, 0] * 1e-6, fig10[:, 1], "rs", ms=5, label="Ming & Pantalé, 2018")
    plt.xlabel(r"Time (s)")
    plt.ylabel(r"Equivalent plastic strain")
   
    plt.legend(loc="lower right")
    plt.savefig(outdir / "fig10_comparison.png", dpi=200, bbox_inches="tight")
    print("\nFinal (t = 80 us)       this run     paper (A-N-R / Built-in)")
    print(f"eps_p centre          {hist[-1,1]:8.3f}     1.810 / 1.797")
    print(f"T centre  [C]         {hist[-1,2]:8.1f}     561.1 / 560.2")
    print(f"Lf        [mm]        {hist[-1,3]*1e3:8.2f}     26.56 / 26.57")
    print(f"Df        [mm]        {hist[-1,4]*1e3:8.2f}     11.10 / 11.12")
