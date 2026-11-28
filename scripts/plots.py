#!/usr/bin/env python3
"""Makes every figure and table of the report from the raw measurements in results/.

    python3 scripts/plots.py [results_dir] [--backend openmp] [--cores 8]

Reads   results/timings.csv        (bench.sh: one row per graph x backend x threads, medians of 5 runs)
        results/leafsweep.csv      (tsne_seq -m ... -csv)
        results/crossover-omp.txt  (tsne_omp -crossover), and results/crossover-seq.txt if present
        results/direct-omp.txt     (tsne_omp -direct)
Writes  results/figures/*.pdf and *.png, and results/speedups.csv, results/unit_costs.csv

Phases:  P = beta + symmetrize,  tree = keys + sort + build,  attr,  cm (centers of mass),  walk.
Speedup is always measured against p = 1 of the SAME backend (the OpenCilk build uses another
compiler, so comparing it with the sequential gcc build would mix two effects).
Edges = nnz / 2 (every undirected edge is stored twice).
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("res", nargs="?", default="results")
ap.add_argument("--backend", default="openmp", help="backend used in the per-phase and across-graph figures")
ap.add_argument("--cores", type=int, default=8, help="physical cores (vertical line in the speedup plots)")
args = ap.parse_args()
RES, OUT = args.res, os.path.join(args.res, "figures")
os.makedirs(OUT, exist_ok=True)

PH = [("P", "P (beta + symmetrize)"), ("attr", "attraction (SpMM)"), ("tree", "quadtree build"),
      ("cm", "centers of mass"), ("walk", "Barnes-Hut walk")]
BK = {"pthreads": ("tab:blue", "o"), "openmp": ("tab:orange", "s"), "cilk": ("tab:green", "^")}
PCOL = {"P": "tab:purple", "attr": "tab:red", "tree": "tab:brown", "cm": "tab:gray", "walk": "tab:blue", "total": "black"}


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".pdf"))


def ht_region(ax, ps):
    """grey band where threads > physical cores: 2 hyperthreads share a core, ideal speedup no longer applies"""
    if max(ps) > args.cores:
        ax.axvspan(args.cores, max(ps) * 1.15, color="0.93", zorder=0)
        ax.text(args.cores * 1.06, 1.05, "hyper-\nthreads", fontsize=6, color="0.4", va="bottom")


def kfmt(v):
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"


def pow2_axis(ax, ps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps)
    ax.set_xticklabels([str(p) for p in ps])
    ax.minorticks_off()


# ------------------------------------------------------------------ load
d = pd.read_csv(os.path.join(RES, "timings.csv"))
d["P"] = d["beta"] + d["symm"]
d["tree"] = d["keys"] + d["sort"] + d["build"]
gs = d.drop_duplicates("graph").sort_values("n")["graph"].tolist()
par = d[d.backend != "seq"]
ps = sorted(par["threads"].unique())
bks = [b for b in BK if b in set(par.backend)]
cols = [c for c, _ in PH] + ["total"]


def row(g, b, p):
    return d[(d.graph == g) & (d.backend == b) & (d.threads == p)].iloc[0]


def nnz_of(g):
    return int(d[d.graph == g].iloc[0]["nnz"])


# ------------------------------------------------------------------ table: speedups
rows = []
for g in gs:
    for b in bks:
        r1 = row(g, b, 1)
        for p in ps:
            r = row(g, b, p)
            rows.append(dict(graph=g, backend=b, threads=p, **{c: r1[c] / r[c] for c in cols}))
pd.DataFrame(rows).round(3).to_csv(os.path.join(RES, "speedups.csv"), index=False)
print("wrote", os.path.join(RES, "speedups.csv"))

# ------------------------------------------------------------------ table: per-unit costs (sequential) for predict-then-measure
uc = []
for g in gs:
    r = d[(d.graph == g) & (d.backend == "seq")].iloc[0]
    uc.append(dict(graph=g, n=int(r.n), nnz=int(r.nnz),
                   attr_ns_per_nnz=r.attr / r.nnz * 1e9, walk_us_per_point=r.walk / r.n * 1e6,
                   sort_ns_per_point=r["sort"] / r.n * 1e9, beta_ns_per_nnz=r.beta / r.nnz * 1e9,
                   symm_ns_per_nnz=r.symm / r.nnz * 1e9))
pd.DataFrame(uc).round(3).to_csv(os.path.join(RES, "unit_costs.csv"), index=False)
print("wrote", os.path.join(RES, "unit_costs.csv"))

# ------------------------------------------------------------------ table: predict coPapersDBLP from k49 and com-Amazon
# Each phase of the target graph is predicted from the sequential cost per unit (per nonzero or per point)
# of each of the two smaller graphs; for the walk also with an n log2 n model. Nothing is fitted to the target.
PM_SRC, PM_TGT = ["k49_norm_10NN", "com-Amazon"], "coPapersDBLP"
if all(g in gs for g in PM_SRC + [PM_TGT]):
    pm_seq = {g: d[(d.graph == g) & (d.backend == "seq")].iloc[0] for g in PM_SRC + [PM_TGT]}
    pm_t = pm_seq[PM_TGT]
    pm_models = [("attr", "per nnz", lambda r: r.nnz), ("walk", "per point", lambda r: r.n),
              ("walk", "n log2 n", lambda r: r.n * np.log2(r.n)), ("sort", "per point", lambda r: r.n),
              ("beta", "per nnz", lambda r: r.nnz), ("symm", "per nnz", lambda r: r.nnz)]
    pm_rows = []
    for pm_ph, pm_model, pm_unit in pm_models:
        pm_rec = dict(phase=pm_ph, model=pm_model, measured_s=pm_t[pm_ph])
        for g in PM_SRC:
            pm_rec["pred_from_" + g] = pm_seq[g][pm_ph] / pm_unit(pm_seq[g]) * pm_unit(pm_t)
        pm_rows.append(pm_rec)
    pd.DataFrame(pm_rows).round(4).to_csv(os.path.join(RES, "prediction.csv"), index=False)
    print("wrote", os.path.join(RES, "prediction.csv"))

# ------------------------------------------------------------------ table: share of each phase, p = 1 and max p
# share of the sum of the phase medians (P = beta + symm, tree = keys + sort + build)
sh_d = d[d.backend == args.backend]
sh_rows = []
for g in gs:
    for sh_p in (1, int(sh_d.threads.max())):
        sh_r = sh_d[(sh_d.graph == g) & (sh_d.threads == sh_p)].iloc[0]
        sh_parts = {"P": sh_r.beta + sh_r.symm, "attr": sh_r.attr, "tree": sh_r["keys"] + sh_r["sort"] + sh_r.build, "cm": sh_r.cm,
                 "walk": sh_r.walk, "assemble": sh_r.assemble}
        sh_sum = sum(sh_parts.values())
        sh_rows.append(dict(graph=g, backend=args.backend, threads=sh_p, **{k: 100 * v / sh_sum for k, v in sh_parts.items()}))
pd.DataFrame(sh_rows).round(1).to_csv(os.path.join(RES, "phase_shares.csv"), index=False)
print("wrote", os.path.join(RES, "phase_shares.csv"))

# ------------------------------------------------------------------ fig 1: total speedup, per graph, per backend
fig, axes = plt.subplots(1, len(gs), figsize=(3.0 * len(gs) + 0.6, 2.9), sharey=True)
for ax, g in zip(np.atleast_1d(axes), gs):
    ax.plot(ps, ps, "--", color="lightgray", label="ideal (= p)")
    for b in bks:
        t1 = row(g, b, 1)["total"]
        ax.plot(ps, [t1 / row(g, b, p)["total"] for p in ps], marker=BK[b][1], color=BK[b][0], label=b)
    ht_region(ax, ps)
    ax.set_title(g, fontsize=9)
    ax.set_yscale("log", base=2)
    ax.set_yticks(ps)
    ax.set_yticklabels([str(p) for p in ps])
    ax.minorticks_off()
    pow2_axis(ax, ps)
    ax.set_xlabel("threads p")
np.atleast_1d(axes)[0].set_ylabel("speedup of the whole gradient\n(vs p = 1 of the same backend)")
np.atleast_1d(axes)[0].legend(fontsize=7)
save(fig, "fig1_total_speedup")

# ------------------------------------------------------------------ fig 2: run time of each phase vs p (graphs x phases)
fig, axes = plt.subplots(len(gs), len(PH), figsize=(2.3 * len(PH), 2.0 * len(gs) + 0.4), sharex=True)
for i, g in enumerate(gs):
    seq = d[(d.graph == g) & (d.backend == "seq")].iloc[0]
    for j, (c, title) in enumerate(PH):
        ax = axes[i][j]
        for b in bks:
            ax.plot(ps, [row(g, b, p)[c] for p in ps], marker=BK[b][1], ms=3, lw=1, color=BK[b][0], label=b)
        ax.axhline(seq[c], color="k", ls=":", lw=0.8, label="sequential")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ps)
        ax.set_xticklabels([str(p) for p in ps], fontsize=6)
        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=6)
        if i == 0:
            ax.set_title(title, fontsize=8)
        if j == 0:
            ax.set_ylabel(f"{g}\ntime [s]", fontsize=7)
        if i == len(gs) - 1:
            ax.set_xlabel("threads p", fontsize=7)
axes[0][0].legend(fontsize=6)
save(fig, "fig2_phase_times_vs_p")

# ------------------------------------------------------------------ fig 3: speedup of each phase (one backend)
b = args.backend
fig, axes = plt.subplots(1, len(gs), figsize=(3.0 * len(gs) + 0.6, 2.9), sharey=True)
for ax, g in zip(np.atleast_1d(axes), gs):
    ax.plot(ps, ps, "--", color="lightgray", label="ideal (= p)")
    r1 = row(g, b, 1)
    for c in cols:
        ax.plot(ps, [r1[c] / row(g, b, p)[c] for p in ps], marker="o", ms=3, lw=1.6 if c == "total" else 1,
                color=PCOL[c], label=dict(PH).get(c, "total"))
    ht_region(ax, ps)
    ax.set_title(f"{g} ({b})", fontsize=9)
    ax.set_yscale("log", base=2)
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.set_yticklabels(["1", "2", "4", "8", "16"])
    ax.minorticks_off()
    pow2_axis(ax, ps)
    ax.set_xlabel("threads p")
np.atleast_1d(axes)[0].set_ylabel("speedup vs p = 1")
np.atleast_1d(axes)[0].legend(fontsize=6)
save(fig, "fig3_phase_speedups")

# ------------------------------------------------------------------ fig 4: across the graphs (time at p=1 vs n and vs edges)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
ns = [int(row(g, b, 1)["n"]) for g in gs]
es = [nnz_of(g) / 2 for g in gs]
for ax, xs, lab in ((axes[0], ns, "nodes n"), (axes[1], es, "edges |E|")):
    for c in cols:
        ax.plot(xs, [row(g, b, 1)[c] for g in gs], marker="o", ms=4, lw=1.6 if c == "total" else 1,
                color=PCOL[c], label=dict(PH).get(c, "total"))
    pmax = max(ps)
    ax.plot(xs, [row(g, b, pmax)["total"] for g in gs], "k--", marker="s", ms=3, lw=1, label=f"total, p = {pmax}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([kfmt(v) for v in xs], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel(lab + "  (k49, com-Amazon, coPapersDBLP)", fontsize=8)
axes[0].set_ylabel(f"time [s] ({b}, p = 1)")
axes[1].legend(fontsize=6)
save(fig, "fig4_across_graphs")

# ------------------------------------------------------------------ fig 5: leaf capacity
lf = pd.read_csv(os.path.join(RES, "leafsweep.csv"))
lf["bh"] = lf["keys"] + lf["sort"] + lf["build"] + lf["cm"] + lf["walk"]
fig, ax = plt.subplots(figsize=(3.6, 2.9))
for g in lf.drop_duplicates("graph").sort_values("n")["graph"]:
    s = lf[lf.graph == g].sort_values("leaf_m")
    ax.plot(s["leaf_m"], s["bh"] / s["bh"].min(), marker="o", ms=4, label=f"{g} (min {s['bh'].min():.3f} s)")
    print(f"leaf sweep {g}: best m = {int(s.loc[s['bh'].idxmin(), 'leaf_m'])}  "
          + "  ".join(f"m={int(m)}: {t:.3f}s" for m, t in zip(s["leaf_m"], s["bh"])))
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(lf.leaf_m.unique()))
ax.set_xticklabels([str(m) for m in sorted(lf.leaf_m.unique())])
ax.minorticks_off()
ax.set_xlabel("leaf capacity m")
ax.set_ylabel("tree + centers + walk\n(relative to the best m)")
ax.legend(fontsize=6)
save(fig, "fig5_leaf_size")

# ------------------------------------------------------------------ fig 6: Barnes-Hut vs direct
pat = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$")


def crossover(fname):
    txt = open(fname).read()
    m = re.search(r"threads=(\d+)", txt)
    cr = np.array([tuple(map(float, m.groups())) for m in map(pat.match, txt.splitlines()) if m])
    n_, bh, di = cr[:, 0], cr[:, 1], cr[:, 2]
    ratio = di / bh
    x = None
    for i in range(1, len(n_)):
        if ratio[i - 1] < 1 <= ratio[i]:  # log-log interpolation between the two measured n around ratio 1
            t = -np.log(ratio[i - 1]) / (np.log(ratio[i]) - np.log(ratio[i - 1]))
            x = float(np.exp(np.log(n_[i - 1]) + t * (np.log(n_[i]) - np.log(n_[i - 1]))))
            break
    return int(m.group(1)) if m else None, n_, bh, di, x


runs = [crossover(os.path.join(RES, f)) for f in ("crossover-omp.txt", "crossover-seq.txt")
        if os.path.exists(os.path.join(RES, f))]
fig, ax = plt.subplots(figsize=(3.8, 3.0))
for k, (p, n_, bh, di, x) in enumerate(runs):
    ls = "-" if k == 0 else "--"
    ax.loglog(n_, di, "o" + ls, ms=3.5, color="tab:blue", label=f"direct O(n²), p = {p}")
    ax.loglog(n_, bh, "s" + ls, ms=3.5, color="tab:orange", label=f"Barnes-Hut, p = {p}")
    if x:
        ax.axvline(x, color="gray", ls=":" if k == 0 else "-.", lw=0.8)
        print(f"p = {p}: crossover (log-log interpolation between measured n) at n ~ {x:,.0f}")
    print(f"p = {p}: at n = {int(n_[-1])}: direct {di[-1]:.4f} s, BH {bh[-1]:.4f} s, ratio {di[-1] / bh[-1]:.1f}x")
ax.set_xlabel("n (strided sample of Y0, k49)")
ax.set_ylabel("time [s]")
ax.set_title("vertical lines mark the interpolated crossover", fontsize=7)
ax.legend(fontsize=6)
save(fig, "fig6_bh_vs_direct")
