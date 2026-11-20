"""
PINN formulations for the intermittently forced ADRE (JAX + optax).

Seven variants, all trained with an identical budget (epochs, optimiser,
total collocation count), so the ablation isolates one factor at a time:

    base    uniform sampling, soft boundary conditions       (control)
    fs      focused sampling near Ton and Toff               (sampling)
    pw      proximity weighting of the PDE residual          (weighting)
    rd      residual-based adaptive sampling (RAD)           (sampling)
    causal  causal temporal weighting                        (weighting)
    dd      three temporal subnetworks split at the switches (architecture)
    hc      hard inlet constraint via a boundary lift        (architecture)

An additional combination "fspw" (focused sampling with proximity weighting)
is supported but not part of the reported study.

All seeds for a given (variant, Pe) are trained simultaneously with vmap.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

import params as P

jax.config.update("jax_enable_x64", False)


# ============================================================== network
def init_mlp(layers, key):
    ps = []
    keys = jax.random.split(key, len(layers) - 1)
    for k, (ni, no) in zip(keys, zip(layers[:-1], layers[1:])):
        wk, _ = jax.random.split(k)
        lim = jnp.sqrt(6.0 / (ni + no))
        ps.append((jax.random.uniform(wk, (ni, no), minval=-lim, maxval=lim),
                   jnp.zeros((no,))))
    return ps


def mlp(ps, xt):
    h = xt
    for (W, b) in ps[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = ps[-1]
    return (h @ W + b)[..., 0]


def _norm(X, T):
    return jnp.stack([2.0 * X - 1.0, 2.0 * T / P.T_END - 1.0], axis=-1)


# --------------------------------------------------------- inlet forcing
def g_inlet(T):
    on = T < P.TON
    reac = T >= P.TOFF
    v = jnp.where(on, P.U0 * jnp.exp(-P.H * T), 0.0)
    g = jnp.where(reac, P.U0 * jnp.exp(-P.H * (T - P.TOFF)), v)
    if P.RAMP_TAU > 0:          # static at trace time; see params.RAMP_TAU
        r = jnp.clip(T / P.RAMP_TAU, 0.0, 1.0)
        g = g * (3 * r ** 2 - 2 * r ** 3)
    return g


# ------------------------------------------------------------ solution maps
def u_soft(ps, X, T):
    """Soft-constraint network: the inlet condition is a penalty term."""
    return mlp(ps, _norm(X, T))


def _lift(X):
    """
    Boundary lift psi(X) for the hard inlet constraint, with psi(0) = 1.

    "exp":  psi = exp(-X/ell) (default); support confined near the inlet.
    "quad": psi = (1-X)^2; support spans the whole domain (lift study only).
    """
    if P.HC_LIFT == "quad":
        return (1.0 - X) ** 2
    return jnp.exp(-X / P.HC_LENGTH)


def u_hard(ps, X, T):
    """
    Hard inlet constraint:
        U(X,T) = psi(X) g(T) + (1 - psi(X)) N(X,T)

    U(0,T) = g(T) identically, for every parameter value, so the temporal
    discontinuity never enters the optimisation problem at all. Because
    psi is negligible at the outlet, the Neumann condition is imposed softly
    exactly as in the other variants. See _lift for the choice of psi.
    """
    psi = _lift(X)
    return psi * g_inlet(T) + (1.0 - psi) * mlp(ps, _norm(X, T))


# ---------------------------------------------------------- derivatives
def _make_ops(u_fn):
    def u_pt(ps, X, T):
        return u_fn(ps, X, T)

    def res_pt(ps, X, T, Pe):
        uT = jax.grad(u_pt, argnums=2)(ps, X, T)
        uX = jax.grad(u_pt, argnums=1)(ps, X, T)
        uXX = jax.grad(jax.grad(u_pt, argnums=1), argnums=1)(ps, X, T)
        return P.R * uT - (1.0 / Pe) * uXX + uX + P.DA * u_pt(ps, X, T)

    def dudx_pt(ps, X, T):
        return jax.grad(u_pt, argnums=1)(ps, X, T)

    return (jax.vmap(u_pt, (None, 0, 0)),
            jax.vmap(res_pt, (None, 0, 0, None)),
            jax.vmap(dudx_pt, (None, 0, 0)))


U_SOFT, RES_SOFT, DX_SOFT = _make_ops(u_soft)
U_HARD, RES_HARD, DX_HARD = _make_ops(u_hard)


def ops_for(variant):
    return (U_HARD, RES_HARD, DX_HARD) if variant == "hc" else (U_SOFT, RES_SOFT, DX_SOFT)


# ============================================================== sampling
def sample_uniform(n, key):
    kx, kt = jax.random.split(key)
    return (jax.random.uniform(kx, (n,)),
            jax.random.uniform(kt, (n,)) * P.T_END)


def sample_focused(n, key, centre, half):
    kx, kt = jax.random.split(key)
    X = jax.random.uniform(kx, (n,))
    T = centre + half * jax.random.uniform(kt, (n,), minval=-1.0, maxval=1.0)
    return X, jnp.clip(T, 0.0, P.T_END)


def make_batch(key, focused):
    """
    Total collocation count is P.N_F for EVERY variant. Focused sampling
    reallocates points rather than adding them, so the budget is identical.
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    if focused:
        n_fo = P.N_F_FOCUS
        Xu, Tu = sample_uniform(P.N_F - 2 * n_fo, k1)
        Xa, Ta = sample_focused(n_fo, k2, P.TON, P.DELTA_FOCUS)
        Xb, Tb = sample_focused(n_fo, k3, P.TOFF, P.DELTA_FOCUS)
        Xf = jnp.concatenate([Xu, Xa, Xb])
        Tf = jnp.concatenate([Tu, Ta, Tb])
    else:
        Xf, Tf = sample_uniform(P.N_F, k1)
    Xic, _ = sample_uniform(P.N_IC, k4)
    _, Tbc = sample_uniform(P.N_BC, k5)
    return Xf, Tf, Xic, Tbc


# ================================================================ losses
def prox_weight(T):
    return (1.0 + P.BETA_W * jnp.exp(-((T - P.TON) / P.SIGMA_T) ** 2)
                + P.BETA_W * jnp.exp(-((T - P.TOFF) / P.SIGMA_T) ** 2))


def _loss_parts(ps, batch, Pe, variant, bin_idx=None):
    Ubatch, Resbatch, DXbatch = ops_for(variant)
    Xf, Tf, Xic, Tbc = batch
    r = Resbatch(ps, Xf, Tf, Pe)

    if variant in ("pw", "fspw"):
        l_pde = jnp.mean(prox_weight(Tf) * r ** 2)
    elif variant == "causal":
        # Wang et al. (2022): bin-wise weights w_k = exp(-eps * sum_{j<k} L_j)
        nb = P.CAUSAL_BINS
        sq = r ** 2
        per_bin = jnp.zeros(nb).at[bin_idx].add(sq)
        cnt = jnp.zeros(nb).at[bin_idx].add(1.0)
        Lk = per_bin / jnp.maximum(cnt, 1.0)
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(Lk)[:-1]])
        w = jax.lax.stop_gradient(jnp.exp(-P.CAUSAL_EPS * cum))
        l_pde = jnp.sum(w * Lk) / nb
    else:
        l_pde = jnp.mean(r ** 2)

    l_ic = jnp.mean((Ubatch(ps, Xic, jnp.zeros_like(Xic))
                     - P.UB * jnp.exp(-P.BETA_ST * Xic)) ** 2)
    if variant == "hc":
        l_bc0 = jnp.array(0.0)          # satisfied identically by construction
    else:
        l_bc0 = jnp.mean((Ubatch(ps, jnp.zeros_like(Tbc), Tbc) - g_inlet(Tbc)) ** 2)
    l_bc1 = jnp.mean(DXbatch(ps, jnp.ones_like(Tbc), Tbc) ** 2)
    return l_pde, l_ic, l_bc0, l_bc1


def _loss(ps, batch, Pe, variant, bin_idx=None):
    parts = _loss_parts(ps, batch, Pe, variant, bin_idx)
    return sum(parts), parts


# ============================================ domain decomposition losses
DD_BOUNDS = ((0.0, P.TON), (P.TON, P.TOFF), (P.TOFF, P.T_END))


def _dd_loss(all_ps, batches, Xic, Xif, Pe):
    n0, n1, n2 = all_ps
    l_pde, l_bc0, l_bc1 = 0.0, 0.0, 0.0
    for net, (Xf, Tf, Tbc) in zip(all_ps, batches):
        l_pde = l_pde + jnp.mean(RES_SOFT(net, Xf, Tf, Pe) ** 2)
        l_bc0 = l_bc0 + jnp.mean((U_SOFT(net, jnp.zeros_like(Tbc), Tbc) - g_inlet(Tbc)) ** 2)
        # outlet Neumann condition on every subdomain
        l_bc1 = l_bc1 + jnp.mean(DX_SOFT(net, jnp.ones_like(Tbc), Tbc) ** 2)
    l_pde, l_bc0, l_bc1 = l_pde / 3, l_bc0 / 3, l_bc1 / 3

    l_ic = jnp.mean((U_SOFT(n0, Xic, jnp.zeros_like(Xic))
                     - P.UB * jnp.exp(-P.BETA_ST * Xic)) ** 2)

    # interface: match value AND flux, as in cPINN/XPINN
    def iface(a, b, Tc):
        Tarr = jnp.full_like(Xif, Tc)
        dv = U_SOFT(a, Xif, Tarr) - U_SOFT(b, Xif, Tarr)
        df = DX_SOFT(a, Xif, Tarr) - DX_SOFT(b, Xif, Tarr)
        return jnp.mean(dv ** 2) + jnp.mean(df ** 2)

    l_if = iface(n0, n1, P.TON) + iface(n1, n2, P.TOFF)
    return l_pde + l_ic + l_bc0 + l_bc1 + l_if, (l_pde, l_ic, l_bc0, l_bc1, l_if)


def dd_predict(all_ps, X, T):
    n0, n1, n2 = all_ps
    out = U_SOFT(n2, X, T)
    out = jnp.where(T < P.TOFF, U_SOFT(n1, X, T), out)
    out = jnp.where(T < P.TON, U_SOFT(n0, X, T), out)
    return out


# ============================================================== training
def _schedule():
    return optax.piecewise_constant_schedule(
        P.LR, boundaries_and_scales={P.LR_DECAY_AT: 0.1})



# ============================================================ checkpointing
def _to_np(tree):
    return jax.tree_util.tree_map(lambda a: np.asarray(a), tree)


def save_ckpt(path, **kw):
    """Atomic write: a disconnect mid-save must not corrupt the checkpoint."""
    import pickle, os
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({k: (_to_np(v) if k in ("ps", "state", "batch", "key") else v)
                     for k, v in kw.items()}, f, protocol=4)
    os.replace(tmp, path)


def load_ckpt(path):
    import pickle, os
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [warn] unreadable checkpoint {path} ({e}); starting fresh")
        return None


def train(variant, Pe, seeds=None, epochs=None, log_every=2000, verbose=True,
          ckpt_file=None, save_every=1000):
    """
    Train all seeds for one (variant, Pe) simultaneously. Returns (params, info).

    If ckpt_file is given, the parameters, optimiser state, sampling state and
    accumulated wall-clock time are written there every save_every epochs, and
    an existing checkpoint is resumed from.
    """
    seeds = P.SEEDS if seeds is None else seeds
    epochs = P.EPOCHS if epochs is None else epochs
    ns = len(seeds)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    opt = optax.adam(_schedule())

    if variant == "dd":
        return _train_dd(Pe, seeds, epochs, keys, opt, log_every, verbose,
                         ckpt_file, save_every)

    focused = variant in ("fs", "fspw")
    init_keys, batch_keys = jax.vmap(lambda k: jax.random.split(k))(keys).transpose(1, 0, 2)
    ps = jax.vmap(init_mlp, (None, 0))(P.LAYERS, init_keys)
    batch = jax.vmap(make_batch, (0, None))(batch_keys, focused)

    bin_idx = None
    if variant == "causal":
        edges = jnp.linspace(0.0, P.T_END, P.CAUSAL_BINS + 1)
        bin_idx = jnp.clip(jnp.digitize(batch[1], edges) - 1, 0, P.CAUSAL_BINS - 1)

    state = opt.init(ps)

    def total_loss(ps, batch, bin_idx):
        if variant == "causal":
            per = jax.vmap(lambda p, b, bi: _loss(p, b, Pe, variant, bi)[0])(ps, batch, bin_idx)
        else:
            per = jax.vmap(lambda p, b: _loss(p, b, Pe, variant)[0])(ps, batch)
        return jnp.sum(per), per

    @jax.jit
    def step(ps, state, batch, bin_idx):
        (tot, per), gr = jax.value_and_grad(total_loss, has_aux=True)(ps, batch, bin_idx)
        upd, state = opt.update(gr, state)
        return optax.apply_updates(ps, upd), state, per

    ep0, elapsed, hist = 0, 0.0, []
    key = jax.random.PRNGKey(sum(seeds))

    ck = load_ckpt(ckpt_file)
    if ck is not None:
        ps, state, key = ck["ps"], ck["state"], ck["key"]
        batch, ep0, elapsed, hist = ck["batch"], ck["epoch"], ck["elapsed"], ck["hist"]
        if variant == "causal":
            edges = jnp.linspace(0.0, P.T_END, P.CAUSAL_BINS + 1)
            bin_idx = jnp.clip(jnp.digitize(batch[1], edges) - 1, 0, P.CAUSAL_BINS - 1)
        print(f"  [{variant} Pe={Pe}] resuming from epoch {ep0} "
              f"({elapsed/60:.1f} min already spent)", flush=True)

    t0 = time.time()
    for ep in range(ep0, epochs):
        if variant == "rd" and ep > 0 and ep % P.RD_EVERY == 0:
            key, ck = jax.random.split(key)
            ckeys = jax.random.split(ck, ns)
            cand = jax.vmap(lambda k: sample_uniform(P.RD_POOL, k))(ckeys)
            resid = jax.vmap(lambda p, X, T: RES_SOFT(p, X, T, Pe))(ps, cand[0], cand[1])
            # RAD (Wu et al., 2023): sample with probability proportional to
            # |residual|^k / mean(|residual|^k) + c
            a = jnp.abs(resid) ** P.RD_K
            prob = a / jnp.mean(a, axis=1, keepdims=True) + P.RD_C
            prob = prob / jnp.sum(prob, axis=1, keepdims=True)
            skeys = jax.random.split(ck, ns)
            sel = jax.vmap(lambda k, p_: jax.random.choice(
                k, P.RD_POOL, shape=(P.N_F,), replace=False, p=p_))(skeys, prob)
            Xf = jnp.take_along_axis(cand[0], sel, axis=1)
            Tf = jnp.take_along_axis(cand[1], sel, axis=1)
            batch = (Xf, Tf, batch[2], batch[3])

        ps, state, per = step(ps, state, batch, bin_idx)
        if ep % log_every == 0 or ep == epochs - 1:
            hist.append((ep, np.asarray(per)))
            if verbose:
                print(f"  [{variant} Pe={Pe}] ep {ep:6d}  loss {float(jnp.mean(per)):.3e}",
                      flush=True)

        if ckpt_file and (ep + 1) % save_every == 0 and ep + 1 < epochs:
            save_ckpt(ckpt_file, ps=ps, state=state, key=key, batch=batch,
                      epoch=ep + 1, elapsed=elapsed + (time.time() - t0), hist=hist)

    return ps, dict(time=elapsed + (time.time() - t0), seeds=list(seeds),
                    history=hist, variant=variant, Pe=Pe)


def _train_dd(Pe, seeds, epochs, keys, opt, log_every, verbose,
              ckpt_file=None, save_every=1000):
    ns = len(seeds)
    sub_keys = jax.vmap(lambda k: jax.random.split(k, 6))(keys)  # (ns,6,2)
    all_ps = [jax.vmap(init_mlp, (None, 0))(P.LAYERS, sub_keys[:, i, :]) for i in range(3)]

    n_f, n_bc = P.N_F // 3, P.N_BC // 3
    batches = []
    for i, (t0_, t1_) in enumerate(DD_BOUNDS):
        k = sub_keys[:, 3 + i, :]
        Xf = jax.vmap(lambda kk: jax.random.uniform(jax.random.split(kk)[0], (n_f,)))(k)
        Tf = jax.vmap(lambda kk: t0_ + (t1_ - t0_) *
                      jax.random.uniform(jax.random.split(kk)[1], (n_f,)))(k)
        Tbc = jax.vmap(lambda kk: t0_ + (t1_ - t0_) *
                       jax.random.uniform(kk, (n_bc,)))(k[:, :])
        batches.append((Xf, Tf, Tbc))
    Xic = jax.vmap(lambda kk: jax.random.uniform(kk, (P.N_IC,)))(sub_keys[:, 0, :])
    Xif = jax.vmap(lambda kk: jax.random.uniform(kk, (P.DD_INTERFACE,)))(sub_keys[:, 1, :])

    state = opt.init(all_ps)

    def total_loss(all_ps):
        per = jax.vmap(lambda p0, p1, p2, b0, b1, b2, xi, xf:
                       _dd_loss((p0, p1, p2), (b0, b1, b2), xi, xf, Pe)[0]
                       )(all_ps[0], all_ps[1], all_ps[2],
                         batches[0], batches[1], batches[2], Xic, Xif)
        return jnp.sum(per), per

    @jax.jit
    def step(all_ps, state):
        (tot, per), gr = jax.value_and_grad(total_loss, has_aux=True)(all_ps)
        upd, state = opt.update(gr, state)
        return optax.apply_updates(all_ps, upd), state, per

    ep0, elapsed, hist = 0, 0.0, []
    ck = load_ckpt(ckpt_file)
    if ck is not None:
        all_ps, state = ck["ps"], ck["state"]
        ep0, elapsed, hist = ck["epoch"], ck["elapsed"], ck["hist"]
        print(f"  [dd Pe={Pe}] resuming from epoch {ep0} "
              f"({elapsed/60:.1f} min already spent)", flush=True)

    t0 = time.time()
    for ep in range(ep0, epochs):
        all_ps, state, per = step(all_ps, state)
        if ep % log_every == 0 or ep == epochs - 1:
            hist.append((ep, np.asarray(per)))
            if verbose:
                print(f"  [dd Pe={Pe}] ep {ep:6d}  loss {float(jnp.mean(per)):.3e}", flush=True)

        if ckpt_file and (ep + 1) % save_every == 0 and ep + 1 < epochs:
            save_ckpt(ckpt_file, ps=all_ps, state=state, key=None, batch=None,
                      epoch=ep + 1, elapsed=elapsed + (time.time() - t0), hist=hist)

    return all_ps, dict(time=elapsed + (time.time() - t0), seeds=list(seeds),
                        history=hist, variant="dd", Pe=Pe)


# ============================================================== evaluation
def pick_seed(ps, i):
    return jax.tree_util.tree_map(lambda a: a[i], ps)


def predict_grid(variant, ps, i, X, T, Pe):
    """Evaluate seed i on a flattened (X,T) grid; returns (U, PDE residual)."""
    Xj, Tj = jnp.asarray(X), jnp.asarray(T)
    if variant == "dd":
        nets = tuple(pick_seed(p, i) for p in ps)
        U = dd_predict(nets, Xj, Tj)
        r = jnp.where(Tj < P.TON,
                      RES_SOFT(nets[0], Xj, Tj, Pe),
                      jnp.where(Tj < P.TOFF,
                                RES_SOFT(nets[1], Xj, Tj, Pe),
                                RES_SOFT(nets[2], Xj, Tj, Pe)))
        return np.asarray(U), np.asarray(r)
    net = pick_seed(ps, i)
    Ub, Rb, _ = ops_for(variant)
    return np.asarray(Ub(net, Xj, Tj)), np.asarray(Rb(net, Xj, Tj, Pe))
