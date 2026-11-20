"""
Analysis: every table and figure in the manuscript.

    python3 analyze.py --stage reference   # Table 2, Figs 1-2
    python3 analyze.py --stage pinn        # forward-study tables and figures
    python3 analyze.py --stage inverse     # switching-time identification
    python3 analyze.py                     # all three

All errors are ABSOLUTE RMSE (U is already scaled by U0). Every comparison
between variants is PAIRED: the same seeds are used for every variant, so
per-seed differences remove initialisation variability from the comparison.
Tables report distributions (mean, std, quartiles) because per-seed errors are
skewed.
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import params as P
import reference as REF

OUT, RES, INV = "out", "results", "results_inverse"
for d in (f"{OUT}/tables", f"{OUT}/figures"):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "lines.linewidth": 1.3,
})
CMAP = {v: plt.cm.tab10(i) for i, v in enumerate(P.VARIANTS)}
PE_DETAIL = 2.0
TT, XX = P.T_EVAL, P.X_EVAL
M_ON = np.abs(TT - P.TON) <= P.DELTA_SWITCH
M_OFF = np.abs(TT - P.TOFF) <= P.DELTA_SWITCH
M_SW = M_ON | M_OFF
M_BULK = ~M_SW & (TT > 0.05)


# ================================================================ helpers
def w(path, text):
    with open(path, "w") as f:
        f.write(text)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def dist(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                q25=float(np.percentile(v, 25)), median=float(np.median(v)),
                q75=float(np.percentile(v, 75)), n=int(v.size))


def tex_table(header, rows, align):
    return "\n".join([r"\begin{tabular}{" + align + "}", r"\toprule", header + r" \\",
                      r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"])


def e3(x):
    return f"{1e3 * x:.2f}"


def seed_metrics(U, ref):
    """Per-seed metrics for one (variant, Pe). U: (ns, NT, NX)."""
    out = {k: [] for k in ("glob", "on", "off", "sw", "bulk", "corner", "r2", "emax")}
    corner = (TT[:, None] <= P.CORNER_RADIUS) & (XX[None, :] <= P.CORNER_RADIUS)
    ss = np.sum((ref - ref.mean()) ** 2)
    for u in U:
        out["glob"].append(rmse(u, ref))
        out["on"].append(rmse(u[M_ON], ref[M_ON]))
        out["off"].append(rmse(u[M_OFF], ref[M_OFF]))
        out["sw"].append(rmse(u[M_SW], ref[M_SW]))
        out["bulk"].append(rmse(u[M_BULK], ref[M_BULK]))
        out["corner"].append(rmse(u[corner], ref[corner]))
        out["r2"].append(float(1 - np.sum((u - ref) ** 2) / ss))
        out["emax"].append(float(np.max(np.abs(u - ref))))
    return {k: np.array(v) for k, v in out.items()}


# ============================================================== reference
def stage_reference():
    payload, rows = {}, []
    for Pe in P.PE_LIST:
        lad = REF.grid_ladder(Pe)
        payload[str(Pe)] = lad
        for i, r in enumerate(lad):
            rate = "--" if np.isnan(r["rate"]) else f"{r['rate']:.2f}"
            rows.append((f"${Pe}$" if i == 0 else "") +
                        f" & {r['Nx']} & ${r['dX']:.2e}$ & ${r['dT']:.2e}$ & "
                        f"${r['err']:.2e}$ & {rate} \\\\")
            print(f"  Pe={Pe} Nx={r['Nx']:>4} L2={r['err']:.3e} rate={r['rate']:.2f}")
        rows.append(r"\midrule")
    rows.pop()
    w(f"{OUT}/tables/table2_grid.tex",
      tex_table(r"$Pe$ & $N_x$ & $\Delta X$ & $\Delta T$ & Rel.\ $L_2$ & Order", rows, "llrrrr"))
    w(f"{OUT}/tables/table2_grid.json", json.dumps(payload, indent=2, default=float))

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.45, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    T = np.linspace(0, P.T_END, 4000)
    g = P.inlet(T)
    for lo, hi in [(0, P.TON), (P.TON, P.TOFF), (P.TOFF, P.T_END)]:
        m = (T >= lo) & ((T <= hi) if hi == P.T_END else (T < hi))
        ax.plot(T[m], g[m], color="C0")
    for tc, lab in [(P.TON, r"$T_{\rm on}$"), (P.TOFF, r"$T_{\rm off}$")]:
        ax.axvline(tc, ls="--", c="0.55", lw=0.9)
        ax.text(tc, 1.05, lab, ha="center", fontsize=8)
    ax.set_xlabel("$T$"); ax.set_ylabel("$g(T)$"); ax.set_ylim(-0.05, 1.18)
    ax.set_title("(a) intermittent inlet forcing", fontsize=8, loc="left")
    for j, Pe in enumerate(P.PE_LIST):
        a = fig.add_subplot(gs[1, j])
        U, _ = REF.reference_field(Pe)
        im = a.pcolormesh(XX, TT, U, shading="auto", cmap="viridis", vmin=0, vmax=1,
                          rasterized=True)
        for tc in (P.TON, P.TOFF):
            a.axhline(tc, c="w", ls="--", lw=0.7)
        a.set_xlabel("$X$"); a.set_title(f"({chr(98 + j)}) $Pe={Pe}$", fontsize=8, loc="left")
        if j == 0:
            a.set_ylabel("$T$")
        plt.colorbar(im, ax=a, fraction=0.046)
    fig.savefig(f"{OUT}/figures/fig01_problem.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for Pe in P.PE_LIST:
        X, _, U, _ = REF.fdm(Pe, save_times=[0.5])
        sub = X[::40]
        axes[0].plot(X, U[0], label=f"$Pe={Pe}$")
        axes[0].plot(sub, REF.laplace(sub, 0.5, Pe), "o", ms=3, mfc="none", color="k")
        lad = payload[str(Pe)]
        axes[1].loglog([r["dX"] for r in lad], [r["err"] for r in lad], "o-", label=f"$Pe={Pe}$")
    dx = np.array([r["dX"] for r in payload[str(P.PE_LIST[-1])]])
    e0 = payload[str(P.PE_LIST[-1])][0]["err"]
    axes[1].loglog(dx, e0 * (dx / dx[0]) ** 2, "k--", lw=1, label=r"$O(\Delta X^2)$")
    axes[0].set_xlabel("$X$"); axes[0].set_ylabel("$U(X,0.5)$"); axes[0].legend()
    axes[1].set_xlabel(r"$\Delta X$"); axes[1].set_ylabel(r"relative $L_2$"); axes[1].legend()
    axes[1].set_xticks(dx); axes[1].set_xticklabels([f"{v:.1e}" for v in dx], fontsize=6.5)
    axes[1].minorticks_off()
    axes[0].set_title("(a) FDM (lines) vs de Hoog (markers)", fontsize=8, loc="left")
    axes[1].set_title("(b) grid convergence", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig02_verification.pdf"); plt.close(fig)
    return payload


# ================================================================== load
def load(variant, Pe, tag=""):
    t = f"_{tag}" if tag else ""
    f = f"{RES}/pinn_{variant}{t}_pe{Pe}.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f)
    if "signature" not in d.files or str(d["signature"]) != P.signature():
        print(f"  [stale, ignored] {os.path.basename(f)}")
        return None
    return {k: d[k] for k in d.files}


def collect():
    ref = {Pe: REF.reference_field(Pe)[0] for Pe in P.PE_LIST}
    store = {v: {} for v in P.VARIANTS}
    for v in P.VARIANTS:
        for Pe in P.PE_LIST:
            d = load(v, Pe)
            if d is None:
                print(f"  missing {v} Pe={Pe}")
                continue
            d["m"] = seed_metrics(d["U"], ref[Pe])
            store[v][Pe] = d
    return store, ref


# ================================================================ tables
def table3_ablation():
    rows = [f"{P.LABELS[v]} & {a} & {b} & {c} & {d} \\\\" for v, (a, b, c, d) in P.ABLATION.items()]
    w(f"{OUT}/tables/table3_ablation.tex",
      tex_table(r"Variant & Sampling & Residual weighting & Architecture & Uses $T_{\rm on},T_{\rm off}$",
                rows, "lllll"))


def table4_main(store):
    """Median [IQR] of RMSE by region (x10^3), with cost."""
    rows, js = [], {}
    for Pe in P.PE_LIST:
        js[str(Pe)] = {}
        first = True
        for v in P.VARIANTS:
            d = store[v].get(Pe)
            if d is None:
                continue
            m = d["m"]
            js[str(Pe)][v] = {k: dist(m[k]) for k in m}
            js[str(Pe)][v]["time"] = float(d["train_time_s"])
            js[str(Pe)][v]["n_params"] = int(d["n_params"])

            def cell(k):
                return (f"{e3(np.median(m[k]))} "
                        f"[{e3(np.percentile(m[k], 25))}, {e3(np.percentile(m[k], 75))}]")
            rows.append((f"${Pe}$" if first else "") + f" & {P.LABELS[v]} & " +
                        " & ".join(cell(k) for k in ("glob", "sw", "bulk")) +
                        f" & {int(d['n_params'])} & {float(d['train_time_s']):.0f} \\\\")
            first = False
        if not first:
            rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    w(f"{OUT}/tables/table4_main.tex",
      tex_table(r"$Pe$ & Variant & global & switching & bulk & $N_\theta$ & $t$ (s)", rows, "lllllrr"))
    w(f"{OUT}/tables/table4_main.json", json.dumps(js, indent=2, default=float))
    return js


def table5_distribution(store, Pe=PE_DETAIL):
    """Distributional statistics of switching-window RMSE (x10^3)."""
    rows = []
    for v in P.VARIANTS:
        d = store[v].get(Pe)
        if d is None:
            continue
        s = dist(d["m"]["sw"])
        rows.append(f"{P.LABELS[v]} & " +
                    " & ".join(e3(s[k]) for k in ("mean", "std", "q25", "median", "q75")) + r" \\")
    w(f"{OUT}/tables/table5_distribution.tex",
      tex_table(r"Variant & mean & std & $q_{25}$ & median & $q_{75}$", rows, "lrrrrr"))


def table6_paired(store):
    """Paired comparison against the baseline: median per-seed change and win rate."""
    rows, js = [], {}
    for Pe in P.PE_LIST:
        b = store["base"].get(Pe)
        if b is None:
            continue
        js[str(Pe)] = {}
        first = True
        for v in P.VARIANTS:
            d = store[v].get(Pe)
            if v == "base" or d is None:
                continue
            dsw = d["m"]["sw"] - b["m"]["sw"]
            dgl = d["m"]["glob"] - b["m"]["glob"]
            win = float(np.mean(d["m"]["sw"] < b["m"]["sw"]))
            rel = float(100 * np.median(dsw / b["m"]["sw"]))
            js[str(Pe)][v] = dict(d_sw=dist(dsw), d_glob=dist(dgl), win=win, rel_pct=rel)
            rows.append((f"${Pe}$" if first else "") + f" & {P.LABELS[v]} & "
                        f"{e3(np.median(dsw))} & {rel:+.1f}\\% & {e3(np.median(dgl))} & "
                        f"{100 * win:.0f}\\% \\\\")
            first = False
        if not first:
            rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    w(f"{OUT}/tables/table6_paired.tex",
      tex_table(r"$Pe$ & Variant & $\Delta$ switching & rel. & $\Delta$ global & win rate", rows, "llrrrr"))
    w(f"{OUT}/tables/table6_paired.json", json.dumps(js, indent=2, default=float))
    return js


# =============================================================== figures
def fig03_profiles(store, ref, Pe=PE_DETAIL):
    show = [v for v in ("base", "pw", "hc") if store[v].get(Pe) is not None]
    snaps = [0.75, P.TON, 1.25, 1.6, P.TOFF, 2.25]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.2), sharex=True, sharey=True)
    for ax, Ts in zip(axes.ravel(), snaps):
        k = int(np.argmin(np.abs(TT - Ts)))
        ax.plot(XX, ref[Pe][k], "k-", lw=1.6, label="reference", zorder=5)
        for v in show:
            U = store[v][Pe]["U"][:, k, :]
            ax.fill_between(XX, np.percentile(U, 5, axis=0), np.percentile(U, 95, axis=0),
                            color=CMAP[v], alpha=0.18, lw=0)
            ax.plot(XX, np.median(U, axis=0), "--", color=CMAP[v], lw=1.1, label=P.LABELS[v])
        ax.set_title(f"$T={Ts:g}$", fontsize=8)
    axes[0, 0].legend(fontsize=6.5)
    for a in axes[1]:
        a.set_xlabel("$X$")
    for a in axes[:, 0]:
        a.set_ylabel("$U$")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig03_profiles.pdf"); plt.close(fig)


def fig04_parity(store, ref, Pe=PE_DETAIL):
    vs = [v for v in P.VARIANTS if store[v].get(Pe) is not None]
    if not vs:
        return
    ncol = (len(vs) + 1) // 2
    fig, axes = plt.subplots(2, ncol, figsize=(1.9 * ncol, 4.0), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    idx = np.random.default_rng(0).choice(ref[Pe].size, 4000, replace=False)
    r = ref[Pe].ravel()[idx]
    sw = np.repeat(M_SW[:, None], len(XX), axis=1).ravel()[idx]
    for ax, v in zip(axes.ravel(), vs):
        u = np.median(store[v][Pe]["U"], axis=0).ravel()[idx]
        ax.scatter(r[~sw], u[~sw], s=1.2, color="0.6", rasterized=True)
        ax.scatter(r[sw], u[sw], s=1.6, color="C3", rasterized=True)
        ax.plot([0, 1], [0, 1], "k-", lw=0.8)
        ax.set_title(f"{P.LABELS[v]}  RMSE {e3(rmse(u, r))}e-3", fontsize=7)
    for ax in axes.ravel()[len(vs):]:
        ax.axis("off")
    fig.supxlabel("reference $U$", fontsize=9); fig.supylabel("PINN $U$ (seed median)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig04_parity.pdf"); plt.close(fig)


def fig05_temporal(store, ref):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), sharey=True)
    for ax, Pe in zip(axes, P.PE_LIST):
        for v in P.VARIANTS:
            d = store[v].get(Pe)
            if d is None:
                continue
            c = np.sqrt(np.mean((d["U"] - ref[Pe][None]) ** 2, axis=2))   # (ns, NT)
            ax.semilogy(TT, np.median(c, axis=0), color=CMAP[v], lw=1.0, label=P.LABELS[v])
            ax.fill_between(TT, np.percentile(c, 25, axis=0), np.percentile(c, 75, axis=0),
                            color=CMAP[v], alpha=0.12, lw=0)
        for tc in (P.TON, P.TOFF):
            ax.axvspan(tc - P.DELTA_SWITCH, tc + P.DELTA_SWITCH, color="0.88", zorder=0)
        ax.set_xlabel("$T$"); ax.set_title(f"$Pe={Pe}$", fontsize=8)
    axes[0].set_ylabel("RMSE at time $T$")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, ncol=2); break
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig05_temporal.pdf"); plt.close(fig)


def fig06_triplets(store, ref, Pe=PE_DETAIL):
    show = [v for v in ("base", "pw", "hc") if store[v].get(Pe) is not None]
    if not show:
        return
    fig, axes = plt.subplots(len(show), 3, figsize=(7.4, 2.2 * len(show)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    errs = {v: np.abs(np.median(store[v][Pe]["U"], axis=0) - ref[Pe]) for v in show}
    emax = max(float(np.percentile(e, 99.5)) for e in errs.values())
    for i, v in enumerate(show):
        U = np.median(store[v][Pe]["U"], axis=0)
        for j, (F, cm, vmax, t) in enumerate([(ref[Pe], "viridis", 1, "reference"),
                                              (U, "viridis", 1, P.LABELS[v]),
                                              (errs[v], "magma", emax,
                                               "|error| (99.5th pct scale)")]):
            im = axes[i, j].pcolormesh(XX, TT, F, shading="auto", cmap=cm, vmin=0, vmax=vmax,
                                       rasterized=True)
            axes[i, j].set_title(t, fontsize=8)
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)
        axes[i, 0].set_ylabel("$T$")
    for a in axes[-1]:
        a.set_xlabel("$X$")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig06_triplets.pdf"); plt.close(fig)


def fig07_boxes(store):
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    for ax, Pe in zip(axes, P.PE_LIST):
        vs = [v for v in P.VARIANTS if store[v].get(Pe) is not None]
        if not vs:
            continue
        data = [1e3 * store[v][Pe]["m"]["sw"] for v in vs]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, v in zip(bp["boxes"], vs):
            patch.set_facecolor(CMAP[v]); patch.set_alpha(0.55)
        for k, dd in enumerate(data):
            jit = np.random.default_rng(k).uniform(-0.15, 0.15, dd.size)
            ax.scatter(k + 1 + jit, dd, s=3, color="k", alpha=0.5, zorder=3)
        ax.set_xticks(range(1, len(vs) + 1))
        ax.set_xticklabels([P.LABELS[v].split("-")[1] for v in vs], fontsize=7)
        ax.set_yscale("log"); ax.set_title(f"$Pe={Pe}$", fontsize=8)
    axes[0].set_ylabel(r"switching-window RMSE ($\times10^{-3}$)")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig07_distributions.pdf"); plt.close(fig)


def fig08_paired(store, paired):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    vs = [v for v in P.VARIANTS if v != "base"]
    width = 0.26
    for j, Pe in enumerate(P.PE_LIST):
        b = store["base"].get(Pe)
        if b is None:
            continue
        pos = np.arange(len(vs)) + (j - 1) * width
        data = []
        for v in vs:
            d = store[v].get(Pe)
            data.append(100 * (d["m"]["sw"] - b["m"]["sw"]) / b["m"]["sw"]
                        if d is not None else np.array([np.nan]))
        bp = axes[0].boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                             showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(plt.cm.viridis(j / 2)); patch.set_alpha(0.6)
        wins = [100 * paired.get(str(Pe), {}).get(v, {}).get("win", np.nan) for v in vs]
        axes[1].bar(pos, wins, width * 0.9, color=plt.cm.viridis(j / 2), label=f"$Pe={Pe}$")
    for a in axes:
        a.set_xticks(np.arange(len(vs)))
        a.set_xticklabels([P.LABELS[v].split("-")[1] for v in vs], fontsize=7)
    axes[0].axhline(0, c="k", lw=0.8)
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].set_ylabel("paired change, switching RMSE (%)")
    axes[1].axhline(50, c="k", lw=0.8, ls="--")
    axes[1].set_ylabel("win rate vs baseline (%)"); axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=6.5)
    axes[0].set_title("(a) paired change (negative = better)", fontsize=8, loc="left")
    axes[1].set_title("(b) seeds where variant beats baseline", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig08_paired.pdf"); plt.close(fig)


def fig09_betaw(store, ref):
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    js = {}
    for j, Pe in enumerate(P.PE_LIST):
        pts = []
        b, pw = store["base"].get(Pe), store["pw"].get(Pe)
        if b is not None:
            pts.append((0.0, b["m"]["sw"]))
        for bw in P.BETA_W_SWEEP:
            d = load("pw", Pe, f"bw{bw:g}")
            if d is not None:
                pts.append((bw, seed_metrics(d["U"], ref[Pe])["sw"]))
        if pw is not None:
            pts.append((P.BETA_W, pw["m"]["sw"]))
        if len(pts) < 2:
            continue
        pts.sort(key=lambda t: t[0])
        x = np.array([p[0] for p in pts]); xp = np.where(x == 0, 0.3, x)
        med = np.array([np.median(p[1]) for p in pts])
        lo = np.array([np.percentile(p[1], 25) for p in pts])
        hi = np.array([np.percentile(p[1], 75) for p in pts])
        c = plt.cm.viridis(j / 2)
        ax.plot(xp, 1e3 * med, "o-", color=c, label=f"$Pe={Pe}$")
        ax.fill_between(xp, 1e3 * lo, 1e3 * hi, color=c, alpha=0.18, lw=0)
        js[str(Pe)] = {f"{bw:g}": dist(v) for bw, v in pts}
    ax.set_xscale("log"); ax.set_xlabel(r"$\beta_w$  ($\beta_w=0$ plotted at 0.3)")
    ax.set_ylabel(r"switching RMSE ($\times10^{-3}$)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig09_betaw.pdf"); plt.close(fig)
    w(f"{OUT}/tables/betaw.json", json.dumps(js, indent=2, default=float))
    return js


def fig10_loss(store, Pe=PE_DETAIL):
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for v in P.VARIANTS:
        d = store[v].get(Pe)
        if d is None:
            continue
        ax.semilogy(d["hist_ep"], np.median(d["hist_loss"], axis=1), color=CMAP[v], label=P.LABELS[v])
    ax.axvline(P.LR_DECAY_AT, c="0.5", ls=":", lw=0.9)
    ax.set_xlabel("epoch"); ax.set_ylabel("total loss (seed median)"); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig10_loss.pdf"); plt.close(fig)


def fig11_cost(main_js):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    mk = dict(zip(P.PE_LIST, ["o", "s", "^"]))
    for Pe in P.PE_LIST:
        for v in P.VARIANTS:
            d = main_js.get(str(Pe), {}).get(v)
            if d:
                ax.scatter(d["time"], 1e3 * d["sw"]["median"], marker=mk[Pe], s=34,
                           color=CMAP[v], edgecolor="k", linewidth=0.3)
    for v in P.VARIANTS:
        ax.scatter([], [], color=CMAP[v], label=P.LABELS[v], s=30)
    for Pe in P.PE_LIST:
        ax.scatter([], [], marker=mk[Pe], color="w", edgecolor="k", label=f"$Pe={Pe}$")
    ax.set_yscale("log")
    ax.set_xlabel("training time (s)"); ax.set_ylabel(r"median switching RMSE ($\times10^{-3}$)")
    ax.legend(fontsize=5.5, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig11_cost.pdf"); plt.close(fig)


def fig12_lift(store, ref, Pe=2.0):
    entries = []
    b = store["base"].get(Pe)
    if b is not None:
        entries.append(("soft (baseline)", b["m"]))
    for lift, ell in P.HC_LIFT_SWEEP + [(P.HC_LIFT, P.HC_LENGTH)]:
        if (lift, ell) == (P.HC_LIFT, P.HC_LENGTH):
            d = store["hc"].get(Pe)
            m = d["m"] if d is not None else None
        else:
            d = load("hc", Pe, "quad" if lift == "quad" else f"exp{ell:g}")
            m = seed_metrics(d["U"], ref[Pe]) if d is not None else None
        if m is not None:
            entries.append(("(1-X)^2" if lift == "quad" else f"exp(-X/{ell:g})", m))
    if len(entries) < 2:
        return {}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    for ax, (k, t) in zip(axes, [("glob", "global"), ("sw", "switching windows"), ("corner", "corner")]):
        ax.boxplot([1e3 * m[k] for _, m in entries], showfliers=False)
        ax.set_xticks(range(1, len(entries) + 1))
        ax.set_xticklabels([n for n, _ in entries], fontsize=6, rotation=25)
        ax.set_yscale("log"); ax.set_title(t, fontsize=8)
    axes[0].set_ylabel(r"RMSE ($\times10^{-3}$)")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig12_lift.pdf"); plt.close(fig)
    js = {n: {k: dist(m[k]) for k in ("glob", "sw", "corner")} for n, m in entries}
    w(f"{OUT}/tables/lift.json", json.dumps(js, indent=2, default=float))
    return js


def figS1_residual(store, Pe=PE_DETAIL):
    show = [v for v in ("base", "hc") if store[v].get(Pe) is not None]
    if not show:
        return
    fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 2.8), sharey=True)
    for ax, v in zip(np.atleast_1d(axes), show):
        Rr = np.median(np.abs(store[v][Pe]["res"]), axis=0)
        im = ax.pcolormesh(XX, TT, np.log10(Rr + 1e-12), shading="auto", cmap="cividis",
                           rasterized=True)
        for tc in (P.TON, P.TOFF):
            ax.axhline(tc, c="w", ls="--", lw=0.7)
        ax.set_xlabel("$X$"); ax.set_title(P.LABELS[v], fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\log_{10}|\mathcal{R}|$")
    np.atleast_1d(axes)[0].set_ylabel("$T$")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/figS1_residual.pdf"); plt.close(fig)


# ========================================================= corner experiment
def stage_corner():
    """
    Isolate the corner incompatibility from the switching events.

    tau ramps the inlet up from zero over 0 <= T <= tau, removing the conflict
    between U(X,0)=0 and g(0)=U0 while leaving both switches sharp. If the
    corner error collapses with tau and the switching error does not, the two
    effects are separable and the corner is a property of the problem
    statement rather than of the method.

    Each tau has its own reference solution, since the ramp changes the exact
    solution near T=0.
    """
    taus = [0.0] + list(P.RAMP_SWEEP)
    saved = P.RAMP_TAU
    data, rows = {}, []
    try:
        for tau in taus:
            P.RAMP_TAU = tau
            for Pe in P.PE_LIST:
                ref, _ = REF.reference_field(Pe)
                d = load("base", Pe) if tau == 0 else load("base", Pe, f"tau{tau:g}")
                if d is None:
                    print(f"  missing base tau={tau:g} Pe={Pe}")
                    continue
                data[(tau, Pe)] = seed_metrics(d["U"], ref)
    finally:
        P.RAMP_TAU = saved
    if not data:
        print("  no corner-experiment results")
        return {}

    js = {}
    for (tau, Pe), m in data.items():
        js[f"tau{tau:g}_Pe{Pe}"] = {k: dist(m[k]) for k in ("glob", "sw", "corner", "bulk")}
    for Pe in P.PE_LIST:
        first = True
        for tau in taus:
            m = data.get((tau, Pe))
            if m is None:
                continue
            rows.append((f"${Pe}$" if first else "") + f" & {tau:g} & " +
                        " & ".join(e3(np.median(m[k])) for k in ("glob", "sw", "corner", "bulk"))
                        + f" & {np.median(m['corner']) / np.median(m['bulk']):.1f} \\\\")
            first = False
        if not first:
            rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    w(f"{OUT}/tables/table8_corner.tex",
      tex_table(r"$Pe$ & $\tau$ & global & switching & corner & bulk & corner/bulk",
                rows, "llrrrrr"))
    w(f"{OUT}/tables/table8_corner.json", json.dumps(js, indent=2, default=float))

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9), sharex=True)
    for ax, (k, t) in zip(axes, [("corner", "corner region"),
                                 ("sw", "switching windows"), ("bulk", "bulk")]):
        for j, Pe in enumerate(P.PE_LIST):
            xs = [tau for tau in taus if (tau, Pe) in data]
            if not xs:
                continue
            med = [1e3 * np.median(data[(tau, Pe)][k]) for tau in xs]
            lo = [1e3 * np.percentile(data[(tau, Pe)][k], 25) for tau in xs]
            hi = [1e3 * np.percentile(data[(tau, Pe)][k], 75) for tau in xs]
            xp = [x if x > 0 else 0.003 for x in xs]
            c = plt.cm.viridis(j / 2)
            ax.plot(xp, med, "o-", color=c, label=f"$Pe={Pe}$")
            ax.fill_between(xp, lo, hi, color=c, alpha=0.18, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\tau$   ($\tau=0$ plotted at 0.003)")
        ax.set_title(t, fontsize=8)
    axes[0].set_ylabel(r"RMSE ($\times10^{-3}$)"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig15_corner_ramp.pdf"); plt.close(fig)
    return js


# ================================================================ inverse
def stage_inverse():
    runs = {}
    for (Pe, wl, n) in P.INV_GRID:
        f = f"{INV}/inv_pe{Pe}_{wl}_n{n:g}.npz"
        if not os.path.exists(f):
            print(f"  missing {os.path.basename(f)}")
            continue
        d = np.load(f)
        if str(d["signature"]) != P.signature():
            print(f"  [stale, ignored] {os.path.basename(f)}")
            continue
        runs[(Pe, wl, n)] = {k: d[k] for k in d.files}
    if not runs:
        print("  no inverse results")
        return {}

    rows, js = [], {}
    for (Pe, wl, n), d in runs.items():
        t = d["toff"]; err = np.abs(t - P.TOFF)
        js[f"Pe{Pe}_{wl}_n{n:g}"] = dict(toff=dist(t), abs_err=dist(err), rmse=dist(d["rmse"]),
                                          time=float(d["train_time_s"]))
        sd = t.std(ddof=1) if t.size > 1 else 0.0
        rows.append(f"${Pe}$ & {wl} & {100 * n:g}\\% & ${t.mean():.3f} \\pm {sd:.3f}$ & "
                    f"{np.median(err):.3f} & {e3(np.median(d['rmse']))} \\\\")
    w(f"{OUT}/tables/table7_inverse.tex",
      tex_table(r"$Pe$ & wells & noise & $\hat T_{\rm off}$ & median $|\hat T_{\rm off}-T_{\rm off}|$ & "
                r"field RMSE ($\times10^{-3}$)", rows, "lllrrr"))
    w(f"{OUT}/tables/table7_inverse.json", json.dumps(js, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    wells = list(P.INV_WELLS)
    off = np.linspace(-0.25, 0.25, len(P.INV_NOISE))
    for j, n in enumerate(P.INV_NOISE):
        keep = [(i, wl) for i, wl in enumerate(wells) if (2.0, wl, n) in runs]
        if not keep:
            continue
        data = [runs[(2.0, wl, n)]["toff"] for _, wl in keep]
        pos = [i + off[j] for i, _ in keep]
        if all(dd.size > 1 for dd in data):
            vp = ax.violinplot(data, positions=pos, widths=0.22, showmedians=True)
            for bb in vp["bodies"]:
                bb.set_facecolor(plt.cm.plasma(j / 2.5)); bb.set_alpha(0.6)
        else:
            ax.scatter(pos, [np.median(dd) for dd in data], color=plt.cm.plasma(j / 2.5))
        ax.scatter([], [], color=plt.cm.plasma(j / 2.5), label=f"noise {100 * n:g}%")
    ax.axhline(P.TOFF, c="k", ls="--", lw=1, label=r"true $T_{\rm off}$")
    allv = np.concatenate([runs[k]["toff"] for k in runs if k[0] == 2.0])
    pad = max(0.02, 3 * float(allv.std()))
    ax.set_ylim(min(allv.min(), P.TOFF) - pad, max(allv.max(), P.TOFF) + pad)
    ax.text(0.99, 0.02, f"initial guess {P.INV_TOFF0} (off scale)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color="0.4")
    ax.set_xticks(range(len(wells)))
    ax.set_xticklabels([f"{wl}\n{P.INV_WELLS[wl]}" for wl in wells], fontsize=7)
    ax.set_ylabel(r"recovered $\hat T_{\rm off}$"); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig13_inverse_toff.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for j, n in enumerate(P.INV_NOISE):
        d = runs.get((2.0, "spread", n))
        if d is None:
            continue
        Hh = d["hist_toff"]; c = plt.cm.plasma(j / 2.5)
        ax.plot(d["hist_ep"], np.median(Hh, axis=1), color=c, label=f"noise {100 * n:g}%")
        ax.fill_between(d["hist_ep"], np.percentile(Hh, 25, axis=1), np.percentile(Hh, 75, axis=1),
                        color=c, alpha=0.18, lw=0)
    ax.axhline(P.TOFF, c="k", ls="--", lw=1)
    ax2 = ax.twinx()
    d0 = next(iter(runs.values()))
    ax2.semilogy(d0["hist_ep"], d0["hist_eps"], c="0.5", ls=":", lw=1)
    ax2.set_ylabel(r"smoothing width $\varepsilon$", color="0.4"); ax2.grid(False)
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\hat T_{\rm off}$ (median, IQR)"); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig14_inverse_convergence.pdf"); plt.close(fig)
    return js


# ================================================================ inserts
def inserts(t2, main_js, paired, betaw, lift, inv, corner=None):
    # merge with keys written by earlier stages
    o = {}
    if os.path.exists(f"{OUT}/tables/inserts.json"):
        try:
            o = json.load(open(f"{OUT}/tables/inserts.json"))
        except Exception:
            o = {}
    ns = [v["glob"]["n"] for d in main_js.values() for v in d.values()]
    if ns:
        o["n_seeds_used"] = int(max(ns))
    if t2:
        o["ref_worst_grid_error"] = max(t2[str(Pe)][-1]["err"] for Pe in P.PE_LIST)
        o["ref_orders"] = {str(Pe): t2[str(Pe)][-1]["rate"] for Pe in P.PE_LIST}
    for Pe in P.PE_LIST:
        d = main_js.get(str(Pe), {})
        if not d:
            continue
        o[f"best_switch_Pe{Pe}"] = min(d, key=lambda v: d[v]["sw"]["median"])
        o[f"best_global_Pe{Pe}"] = min(d, key=lambda v: d[v]["glob"]["median"])
        if "base" in d:
            b = d["base"]
            o[f"base_switch_over_bulk_Pe{Pe}"] = b["sw"]["median"] / b["bulk"]["median"]
            o[f"base_off_over_on_Pe{Pe}"] = b["off"]["median"] / b["on"]["median"]
        for v, r in paired.get(str(Pe), {}).items():
            o[f"paired_rel_pct_{v}_Pe{Pe}"] = r["rel_pct"]
            o[f"win_rate_{v}_Pe{Pe}"] = r["win"]
    if betaw:
        o["betaw_median_switch"] = {Pe: {bw: s["median"] for bw, s in v.items()}
                                    for Pe, v in betaw.items()}
    if lift:
        o["lift_median_switch"] = {k: v["sw"]["median"] for k, v in lift.items()}
    if corner:
        o["corner_ramp"] = {k: {r: v[r]["median"] for r in ("glob", "sw", "corner", "bulk")}
                            for k, v in corner.items()}
    if inv:
        o["inverse"] = {k: {"toff_median": v["toff"]["median"],
                            "abs_err_median": v["abs_err"]["median"]} for k, v in inv.items()}
    w(f"{OUT}/tables/inserts.json", json.dumps(o, indent=2, default=float))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["reference", "pinn", "corner", "inverse", "all"])
    a = ap.parse_args()

    t2 = None
    if a.stage in ("reference", "all"):
        print("[reference]")
        t2 = stage_reference()
    if t2 is None and os.path.exists(f"{OUT}/tables/table2_grid.json"):
        t2 = json.load(open(f"{OUT}/tables/table2_grid.json"))

    main_js, paired, betaw, lift, inv = {}, {}, {}, {}, {}
    if a.stage in ("pinn", "all"):
        print("[pinn]")
        store, ref = collect()
        table3_ablation()
        main_js = table4_main(store)
        table5_distribution(store)
        paired = table6_paired(store)
        fig03_profiles(store, ref); fig04_parity(store, ref); fig05_temporal(store, ref)
        fig06_triplets(store, ref); fig07_boxes(store); fig08_paired(store, paired)
        betaw = fig09_betaw(store, ref); fig10_loss(store); fig11_cost(main_js)
        lift = fig12_lift(store, ref); figS1_residual(store)
    corner = {}
    if a.stage in ("corner", "all"):
        print("[corner]")
        corner = stage_corner()
    if a.stage in ("inverse", "all"):
        print("[inverse]")
        inv = stage_inverse()

    if a.stage != "reference":
        o = inserts(t2, main_js, paired, betaw, lift, inv, corner)
        print("\n--- key numbers (out/tables/inserts.json) ---")
        for k, v in o.items():
            if not isinstance(v, dict):
                print(f"  {k:<40} {v}")
    print(f"\n-> {OUT}/tables/  and  {OUT}/figures/")


if __name__ == "__main__":
    main()
