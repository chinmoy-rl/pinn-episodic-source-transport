"""
Inverse problem: recover the reactivation time Toff from noisy breakthrough
curves at three downstream observation wells.

    python3 inverse.py                         # full grid (P.INV_GRID)
    python3 inverse.py --workers 2 --worker 0  # shard 0 of 2
    python3 inverse.py --smoke                 # 300 epochs, 2 seeds

Requires the reference cache (run `analyze.py --stage reference` first). The
reference is read from disk rather than recomputed here, because this process
runs JAX in float32 and the reference must be float64.

Toff is a trainable scalar per seed, bounded to (Ton + 0.05, T_END) through a
sigmoid. The inlet step is smoothed with a sigmoid of width eps: with a hard
step the gradient of the loss with respect to Toff is zero almost everywhere,
so Toff would never move. eps is annealed from INV_EPS0 to INV_EPS1, which
widens the basin of attraction early and sharpens the estimate late.

Each seed sets both the network initialisation and the noise realisation, so
the spread across seeds is the total uncertainty from both sources.
"""
import argparse
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

import params as P
import pinn

OUT = "results_inverse"
LO = P.TON + 0.05


def toff_of(z):
    return LO + (P.T_END - LO) * jax.nn.sigmoid(z)


def z_of(toff):
    p = (toff - LO) / (P.T_END - LO)
    return float(np.log(p / (1.0 - p)))


def g_smooth(T, toff, eps):
    """Sigmoid-smoothed intermittent inlet; tends to g(T) as eps -> 0."""
    s_on = jax.nn.sigmoid((T - P.TON) / eps)
    s_off = jax.nn.sigmoid((T - toff) / eps)
    return P.U0 * (jnp.exp(-P.H * T) * (1.0 - s_on)
                   + jnp.exp(-P.H * (T - toff)) * s_off)


def load_ref(Pe, cache_dir="cache"):
    f = os.path.join(cache_dir, f"fdm_pe{Pe}.npz")
    if not os.path.exists(f):
        raise SystemExit(f"{f} missing -- run `python analyze.py --stage reference` first")
    d = np.load(f)
    if "signature" not in d.files or str(d["signature"]) != P.signature():
        raise SystemExit(f"{f} was built with different parameters -- "
                         f"re-run `python analyze.py --stage reference`")
    return d["U"]


def observations(Pe, wells, noise, seeds):
    ref = load_ref(Pe)
    Xw = np.asarray(P.INV_WELLS[wells], dtype=float)
    To = np.round(np.arange(P.INV_DT_OBS, P.T_END + 1e-9, P.INV_DT_OBS), 6)
    dt = P.T_END / (P.NT_EVAL - 1)
    kt = np.round(To / dt).astype(int)
    clean = np.stack([np.interp(Xw, P.X_EVAL, ref[k]) for k in kt])   # (nT, nW)
    Xo = np.tile(Xw, len(To))
    Tob = np.repeat(To, len(Xw))
    Uc = clean.ravel()
    sigma = noise * np.abs(Uc).max()
    Uo = np.stack([Uc + sigma * np.random.default_rng(s + 7919).standard_normal(Uc.shape)
                   for s in seeds]).astype(np.float32)
    return Xo, Tob, Uc, Uo, ref


def fname(Pe, wells, noise):
    return os.path.join(OUT, f"inv_pe{Pe}_{wells}_n{noise:g}.npz")


def train_inverse(Pe, wells, noise, seeds, epochs, ckpt_file=None,
                  save_every=1000, log_every=500, verbose=True):
    Xo, Tob, Uc, Uo, ref = observations(Pe, wells, noise, seeds)
    Xo_j, To_j = jnp.asarray(Xo), jnp.asarray(Tob)
    ns = len(seeds)

    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    ik, bk = jax.vmap(lambda k: jax.random.split(k))(keys).transpose(1, 0, 2)
    prm = {"net": jax.vmap(pinn.init_mlp, (None, 0))(P.LAYERS, ik),
           "z": jnp.full((ns,), z_of(P.INV_TOFF0))}
    batch = jax.vmap(pinn.make_batch, (0, None))(bk, False)
    # replace the inlet points with a denser set (see P.INV_N_BC)
    bck = jax.vmap(lambda k: jax.random.fold_in(k, 17))(bk)
    Tbc = jax.vmap(lambda k: jax.random.uniform(k, (P.INV_N_BC,)) * P.T_END)(bck)
    batch = (batch[0], batch[1], batch[2], Tbc)
    Uo_j = jnp.asarray(Uo)

    sched = optax.piecewise_constant_schedule(
        P.LR, boundaries_and_scales={int(0.75 * epochs): 0.1})
    opt = optax.adam(sched)
    state = opt.init(prm)

    def loss_one(p, b, uo, eps):
        net, toff = p["net"], toff_of(p["z"])
        Xf, Tf, Xic, Tbc = b
        l_pde = jnp.mean(pinn.RES_SOFT(net, Xf, Tf, Pe) ** 2)
        l_ic = jnp.mean(pinn.U_SOFT(net, Xic, jnp.zeros_like(Xic)) ** 2)
        l_bc0 = jnp.mean((pinn.U_SOFT(net, jnp.zeros_like(Tbc), Tbc)
                          - g_smooth(Tbc, toff, eps)) ** 2)
        l_bc1 = jnp.mean(pinn.DX_SOFT(net, jnp.ones_like(Tbc), Tbc) ** 2)
        l_dat = jnp.mean((pinn.U_SOFT(net, Xo_j, To_j) - uo) ** 2)
        tot = l_pde + l_ic + l_bc0 + l_bc1 + P.INV_W_DATA * l_dat
        return tot, jnp.stack([l_pde, l_ic, l_bc0, l_bc1, l_dat])

    def total(p, eps):
        tot, parts = jax.vmap(loss_one, (0, 0, 0, None))(p, batch, Uo_j, eps)
        return jnp.sum(tot), parts

    @jax.jit
    def step(p, state, eps):
        (_, parts), g = jax.value_and_grad(total, has_aux=True)(p, eps)
        upd, state = opt.update(g, state)
        return optax.apply_updates(p, upd), state, parts

    anneal_end = max(1, int(P.INV_ANNEAL * epochs))

    def eps_at(ep):
        if ep >= anneal_end:
            return P.INV_EPS1
        return P.INV_EPS0 * (P.INV_EPS1 / P.INV_EPS0) ** (ep / anneal_end)

    ep0, elapsed, hist = 0, 0.0, []
    ck = pinn.load_ckpt(ckpt_file)
    if ck is not None:
        prm, state = ck["ps"], ck["state"]
        ep0, elapsed, hist = ck["epoch"], ck["elapsed"], ck["hist"]
        if verbose:
            print(f"  resuming from epoch {ep0} ({elapsed/60:.1f} min spent)", flush=True)

    t0 = time.time()
    for ep in range(ep0, epochs):
        eps = eps_at(ep)
        prm, state, parts = step(prm, state, jnp.float32(eps))
        if ep % log_every == 0 or ep == epochs - 1:
            toff_now = np.asarray(toff_of(prm["z"]))
            hist.append((ep, eps, toff_now, np.asarray(parts)))
            if verbose and ep % (log_every * 10) == 0:
                print(f"  ep {ep:6d}  eps {eps:.3f}  Toff {toff_now.mean():.4f} "
                      f"+/- {toff_now.std():.4f}", flush=True)
        if ckpt_file and (ep + 1) % save_every == 0 and ep + 1 < epochs:
            pinn.save_ckpt(ckpt_file, ps=prm, state=state, key=None, batch=None,
                           epoch=ep + 1, elapsed=elapsed + time.time() - t0, hist=hist)

    # field accuracy on the evaluation grid
    XX, TT = np.meshgrid(P.X_EVAL, P.T_EVAL)
    Xg, Tg = jnp.asarray(XX.ravel()), jnp.asarray(TT.ravel())
    rmse = []
    for i in range(ns):
        net = jax.tree_util.tree_map(lambda a: a[i], prm["net"])
        U = np.asarray(pinn.U_SOFT(net, Xg, Tg)).reshape(ref.shape)
        rmse.append(float(np.sqrt(np.mean((U - ref) ** 2))))

    return dict(toff=np.asarray(toff_of(prm["z"])), rmse=np.array(rmse),
                time=elapsed + time.time() - t0, hist=hist,
                Xo=Xo, To=Tob, Uc=Uc, Uo=Uo)


def run(job, seeds, epochs, force=False, save_every=1000):
    Pe, wells, noise = job
    os.makedirs(OUT, exist_ok=True)
    f = fname(Pe, wells, noise)
    if os.path.exists(f) and not force:
        try:
            if str(np.load(f)["signature"]) == P.signature():
                print(f"[skip] {os.path.basename(f)}", flush=True)
                return
        except Exception:
            pass
    ck = f.replace(".npz", ".ckpt.pkl")
    print(f"[inverse] {os.path.basename(f)}  seeds={len(seeds)} epochs={epochs}", flush=True)
    r = train_inverse(Pe, wells, noise, seeds, epochs, ckpt_file=ck, save_every=save_every)
    np.savez_compressed(
        f, toff=r["toff"], rmse=r["rmse"], toff_true=P.TOFF, toff0=P.INV_TOFF0,
        Pe=Pe, wells=wells, noise=noise, seeds=np.array(seeds), epochs=epochs,
        train_time_s=r["time"], signature=P.signature(),
        hist_ep=np.array([h[0] for h in r["hist"]]),
        hist_eps=np.array([h[1] for h in r["hist"]]),
        hist_toff=np.array([h[2] for h in r["hist"]]),
        hist_loss=np.array([h[3] for h in r["hist"]]),
        Xo=r["Xo"], To=r["To"], Uc=r["Uc"], Uo=r["Uo"])
    if os.path.exists(ck):
        os.remove(ck)
    err = np.abs(r["toff"] - P.TOFF)
    print(f"[done   ] {os.path.basename(f)}  Toff = {r['toff'].mean():.4f} +/- "
          f"{r['toff'].std():.4f} (true {P.TOFF})  median |err| {np.median(err):.4f}  "
          f"{r['time']:.0f}s", flush=True)


def main():
    import train as TR
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=P.INV_EPOCHS)
    ap.add_argument("--seeds", type=int, default=len(P.INV_SEEDS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    jobs = [(Pe, w, n, None, 1.0) for (Pe, w, n) in P.INV_GRID]
    mine = TR.shard(jobs, a.workers, a.worker) if a.workers > 1 else jobs
    seeds, epochs = P.INV_SEEDS[:a.seeds], a.epochs
    if a.smoke:
        seeds, epochs = P.INV_SEEDS[:2], 300
        print("SMOKE MODE: 300 epochs, 2 seeds -- results are NOT publishable")

    if a.list:
        for j in mine:
            print("  ", os.path.basename(fname(*j[:3])))
        print(f"{len(mine)} jobs")
        return
    for j in mine:
        run(j[:3], seeds, epochs, force=a.force or a.smoke,
            save_every=a.save_every or 10 ** 9)


if __name__ == "__main__":
    main()
