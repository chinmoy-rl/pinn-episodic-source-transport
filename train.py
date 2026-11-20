"""
Training driver.

    python3 train.py                          # main sweep (7 variants x 3 Pe)
    python3 train.py --sweep all              # main + beta_w sweep + lift study
    python3 train.py --sweep betaw            # just the weighting-strength study
    python3 train.py --list                   # print the job list and exit
    python3 train.py --workers 3 --worker 0   # run shard 0 of 3
    python3 train.py --smoke                  # 300 epochs, 2 seeds

Each job writes results/pinn_<variant>[_<tag>]_pe<Pe>.npz holding, for every
seed, the predicted field and PDE residual on the evaluation grid, plus
wall-clock time, parameter count and loss history. Finished jobs are skipped;
unfinished ones resume from a checkpoint written every --save-every epochs.
"""
import argparse
import os
import numpy as np
import jax

import params as P
import pinn

OUT = "results"


# ------------------------------------------------------------------- jobs
def jobs_for(sweep):
    """Each job: (variant, Pe, overrides, tag, weight)."""
    J = []
    if sweep in ("main", "all"):
        for Pe in P.PE_LIST:
            for v in P.VARIANTS:
                J.append((v, Pe, {}, "", P.COST_WEIGHT.get(v, 1.0)))
    if sweep in ("betaw", "all"):
        for Pe in P.PE_LIST:
            for bw in P.BETA_W_SWEEP:
                J.append(("pw", Pe, {"BETA_W": bw}, f"bw{bw:g}", 1.0))
    if sweep in ("corner", "all"):
        for Pe in P.PE_LIST:
            for tau in P.RAMP_SWEEP:
                J.append(("base", Pe, {"RAMP_TAU": tau}, f"tau{tau:g}", 1.0))
    if sweep in ("lift", "all"):
        for lift, ell in P.HC_LIFT_SWEEP:
            tag = "quad" if lift == "quad" else f"exp{ell:g}"
            J.append(("hc", 2.0, {"HC_LIFT": lift, "HC_LENGTH": ell}, tag, 1.0))
    return J


def fname(variant, tag, Pe):
    t = f"_{tag}" if tag else ""
    return os.path.join(OUT, f"pinn_{variant}{t}_pe{Pe}.npz")


def shard(jobs, n, i):
    """Greedy longest-first partition, deterministic, so workers finish together."""
    order = sorted(range(len(jobs)), key=lambda k: (-jobs[k][4], k))
    bins, loads = [[] for _ in range(n)], [0.0] * n
    for k in order:
        b = loads.index(min(loads))
        bins[b].append(k)
        loads[b] += jobs[k][4]
    return [jobs[k] for k in sorted(bins[i])]


def param_count(ps, variant):
    leaves = ([l for sub in ps for l in jax.tree_util.tree_leaves(sub)]
              if variant == "dd" else jax.tree_util.tree_leaves(ps))
    return int(sum(np.prod(l.shape[1:]) for l in leaves))   # axis 0 = seeds


# -------------------------------------------------------------------- run
def run(job, seeds, epochs, force=False, save_every=1000):
    variant, Pe, over, tag, _ = job
    os.makedirs(OUT, exist_ok=True)
    f = fname(variant, tag, Pe)
    if os.path.exists(f) and not force:
        try:
            ok = str(np.load(f)["signature"]) == P.signature()
        except Exception:
            ok = False
        if ok:
            print(f"[skip] {os.path.basename(f)}", flush=True)
            return
        print(f"[stale] {os.path.basename(f)} has different physics -- retraining",
              flush=True)

    saved = {k: getattr(P, k) for k in over}
    for k, v in over.items():
        setattr(P, k, v)
    try:
        ckpt = f.replace(".npz", ".ckpt.pkl")
        print(f"[train] {os.path.basename(f)}  seeds={len(seeds)} epochs={epochs}",
              flush=True)
        ps, info = pinn.train(variant, Pe, seeds=seeds, epochs=epochs,
                              ckpt_file=ckpt, save_every=save_every)

        XX, TT = np.meshgrid(P.X_EVAL, P.T_EVAL)
        Xf, Tf = XX.ravel(), TT.ravel()
        shape = (len(P.T_EVAL), len(P.X_EVAL))
        U = np.empty((len(seeds), *shape), dtype=np.float32)
        Rr = np.empty_like(U)
        for i in range(len(seeds)):
            u, r = pinn.predict_grid(variant, ps, i, Xf, Tf, Pe)
            U[i], Rr[i] = u.reshape(shape), r.reshape(shape)

        np.savez_compressed(
            f, X=P.X_EVAL, T=P.T_EVAL, U=U, res=Rr,
            seeds=np.array(seeds), Pe=Pe, variant=variant, tag=tag,
            signature=P.signature(),
            overrides=str(over), train_time_s=info["time"], epochs=epochs,
            n_params=param_count(ps, variant),
            hist_ep=np.array([h[0] for h in info["history"]]),
            hist_loss=np.array([h[1] for h in info["history"]]))
        if os.path.exists(ckpt):
            os.remove(ckpt)
        print(f"[done ] {os.path.basename(f)}  {info['time']:.0f}s", flush=True)
    finally:
        for k, v in saved.items():
            setattr(P, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="main", choices=["main", "betaw", "lift", "corner", "all"])
    ap.add_argument("--variants", default=None, help="comma list, filters the job list")
    ap.add_argument("--pe", default=None, help="comma list, filters the job list")
    ap.add_argument("--epochs", type=int, default=P.EPOCHS)
    ap.add_argument("--seeds", type=int, default=len(P.SEEDS), help="how many of P.SEEDS")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    jobs = jobs_for(a.sweep)
    if a.variants:
        keep = set(a.variants.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    if a.pe:
        keep = {float(x) for x in a.pe.split(",")}
        jobs = [j for j in jobs if j[1] in keep]

    seeds, epochs = P.SEEDS[:a.seeds], a.epochs
    if a.smoke:
        seeds, epochs = P.SEEDS[:2], 300
        print("SMOKE MODE: 300 epochs, 2 seeds -- results are NOT publishable")

    mine = shard(jobs, a.workers, a.worker) if a.workers > 1 else jobs

    if a.list:
        for j in mine:
            done = "done" if os.path.exists(fname(j[0], j[3], j[1])) else "    "
            print(f"  [{done}] {os.path.basename(fname(j[0], j[3], j[1]))}  weight {j[4]:g}")
        print(f"{len(mine)} jobs")
        return

    for j in mine:
        run(j, seeds, epochs, force=a.force or a.smoke,
            save_every=a.save_every or 10 ** 9)


if __name__ == "__main__":
    main()
