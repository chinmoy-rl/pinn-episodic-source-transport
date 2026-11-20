"""
Reference solutions.

  laplace(X, T, Pe)  -- semi-analytical, de Hoog inversion at 30 digits
  fdm(Pe, ...)       -- explicit FTCS with second-order ghost-node outlet
  grid_ladder(Pe)    -- convergence table, verified against laplace()
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")   # the reference must be float64

import numpy as np
from mpmath import mp, mpf, sinh, cosh, exp, sqrt, invertlaplace

import params as P

mp.dps = 30

# reference.py runs in float64 and pinn.py in float32; they are never imported
# into the same process (analyze.py uses reference, train.py uses pinn).


# ============================================================= Laplace
def _Ubar(X, s, Pe):
    Pe = mpf(Pe)
    alpha = Pe / 2
    mu = sqrt(alpha ** 2 + Pe * (P.R * s + P.DA))
    beta = mpf(P.BETA_ST) + alpha
    M = mpf(0) if P.UB == 0 else (-Pe * P.R * P.UB) / (beta ** 2 - mu ** 2)
    Ub0 = P.U0 / (s + P.H) * (1 - exp(-(s + P.H) * P.TON) + exp(-s * P.TOFF))
    D1 = (Ub0 - M) / sinh(mu)
    D2 = (mu * D1 + M * exp(-beta) * (beta - alpha)) / (alpha * sinh(mu) + mu * cosh(mu))
    return exp(alpha * X) * (D1 * sinh(mu * (1 - X)) + D2 * sinh(mu * X) + M * exp(-beta * X))


def laplace(Xs, T, Pe, degree=18):
    """Time-domain solution at time T for an array of X."""
    out = np.empty(len(Xs))
    for i, X in enumerate(Xs):
        f = lambda s, _X=mpf(float(X)): _Ubar(_X, s, Pe)
        out[i] = float(invertlaplace(f, mpf(float(T)), method="dehoog", degree=degree))
    return out


# ================================================================= FDM
def fdm(Pe, Nx=None, T_end=None, save_times=None, order=None, safety=None,
        backend="jax"):
    """
    Explicit FTCS march.
    order=1 : U_N = U_{N-1}          (first order)
    order=2 : ghost node U_{N+1}=U_{N-1}  (second order)  -- default
    """
    Nx = P.NX_REF if Nx is None else Nx
    T_end = P.T_END if T_end is None else T_end
    order = P.NEUMANN_ORDER if order is None else order

    dX = 1.0 / Nx
    X = np.linspace(0.0, 1.0, Nx + 1)
    if dX > 2.0 / Pe:
        raise ValueError("cell Peclet condition violated")

    dT_max = P.stable_dt(Nx, Pe, safety=P.DT_SAFETY if safety is None else safety)
    save_times = (np.linspace(0.0, T_end, P.NT_EVAL) if save_times is None
                  else np.sort(np.unique(np.asarray(save_times, float))))

    nz = save_times[save_times > 0]
    base = np.min(np.diff(np.concatenate(([0.0], nz)))) if len(nz) else T_end
    dT = base / int(np.ceil(base / dT_max))
    Nt = int(round(T_end / dT))
    dT = T_end / Nt

    U = P.UB * np.exp(-P.BETA_ST * X)
    U[0] = P.inlet(0.0)

    idx = np.round(save_times / dT).astype(int)
    out = np.empty((len(save_times), Nx + 1))
    ptr = 0
    if idx[0] == 0:
        out[0] = U
        ptr = 1

    cD = dT / (P.R * Pe * dX ** 2)
    cA = dT / (P.R * 2.0 * dX)
    cK = dT * P.DA / P.R

    if backend == "jax":
        # March between save times in compiled chunks; the Dirichlet inlet
        # value g(T) is re-imposed at every time step inside the loop.
        import jax, jax.numpy as jnp
        from functools import partial

        Tarr = jnp.asarray(np.arange(Nt + 1) * dT)

        @partial(jax.jit, static_argnums=(3,))
        def chunk(U0, n0, n_steps, order_):
            def body(k, Uc):
                Un = Uc
                inner = (Un[1:-1] + cD * (Un[2:] - 2 * Un[1:-1] + Un[:-2])
                         - cA * (Un[2:] - Un[:-2]) - cK * Un[1:-1])
                if order_ == 1:
                    last = inner[-1]
                else:
                    last = Un[-1] + cD * 2.0 * (Un[-2] - Un[-1]) - cK * Un[-1]
                Tn = (n0 + k + 1) * dT
                g = jnp.where(Tn < P.TON, P.U0 * jnp.exp(-P.H * Tn),
                              jnp.where(Tn < P.TOFF, 0.0,
                                        P.U0 * jnp.exp(-P.H * (Tn - P.TOFF))))
                if P.RAMP_TAU > 0:      # static at trace time
                    rr = jnp.clip(Tn / P.RAMP_TAU, 0.0, 1.0)
                    g = g * (3 * rr ** 2 - 2 * rr ** 3)
                return jnp.concatenate([jnp.array([g]), inner, jnp.array([last])])
            return jax.lax.fori_loop(0, n_steps, body, U0)

        Uj = jnp.asarray(U)
        prev = 0
        for target in idx[ptr:]:
            Uj = chunk(Uj, prev, int(target - prev), order)
            out[ptr] = np.asarray(Uj)
            ptr += 1
            prev = int(target)
        return X, save_times, out, dict(dX=dX, dT=dT, Nt=Nt, Nx=Nx, Pe=Pe, order=order)

    Unew = np.empty_like(U)
    for n in range(1, Nt + 1):
        Unew[1:-1] = (U[1:-1] + cD * (U[2:] - 2 * U[1:-1] + U[:-2])
                      - cA * (U[2:] - U[:-2]) - cK * U[1:-1])
        if order == 1:
            Unew[-1] = Unew[-2]
        else:
            Unew[-1] = U[-1] + cD * 2.0 * (U[-2] - U[-1]) - cK * U[-1]
        Unew[0] = P.inlet(n * dT)
        U, Unew = Unew, U
        if ptr < len(idx) and n == idx[ptr]:
            out[ptr] = U
            ptr += 1

    return X, save_times, out, dict(dX=dX, dT=dT, Nt=Nt, Nx=Nx, Pe=Pe, order=order)


def grid_ladder(Pe, T_check=0.5, nx_list=None, order=None):
    """Convergence of the FDM towards the semi-analytical solution."""
    nx_list = P.NX_LADDER if nx_list is None else nx_list
    rows, prev = [], None
    for Nx in nx_list:
        X, _, U, meta = fdm(Pe, Nx=Nx, T_end=T_check, save_times=[T_check], order=order)
        step = Nx // 100
        Xc, uf = X[::step], U[0][::step]
        ue = laplace(Xc, T_check, Pe)
        err = float(np.linalg.norm(uf - ue) / np.linalg.norm(ue))
        rows.append(dict(Nx=Nx, dX=meta["dX"], dT=meta["dT"], err=err,
                         rate=(np.nan if prev is None else float(np.log2(prev / err)))))
        prev = err
    return rows


def reference_field(Pe, cache_dir="cache"):
    """FDM ground truth on the evaluation grid, cached with a physics signature."""
    import os
    os.makedirs(cache_dir, exist_ok=True)
    tag = f"_tau{P.RAMP_TAU:g}" if P.RAMP_TAU else ""
    f = os.path.join(cache_dir, f"fdm_pe{Pe}{tag}.npz")
    if os.path.exists(f):
        d = np.load(f)
        if "signature" in d.files and str(d["signature"]) == P.signature():
            return d["U"], dict(dX=float(d["dX"]), dT=float(d["dT"]), Nt=int(d["Nt"]))
        print(f"  [cache] {f} was built with different parameters -- rebuilding")
    Xf, _, U, meta = fdm(Pe, save_times=P.T_EVAL)
    Ui = np.array([np.interp(P.X_EVAL, Xf, row) for row in U])
    np.savez_compressed(f, U=Ui, signature=P.signature(), **meta)
    return Ui, meta


if __name__ == "__main__":
    for Pe in P.PE_LIST:
        print(f"Pe={Pe}")
        for r in grid_ladder(Pe):
            print(f"  Nx={r['Nx']:>4} dX={r['dX']:.2e} dT={r['dT']:.2e} "
                  f"L2={r['err']:.3e} rate={r['rate']:.2f}")
