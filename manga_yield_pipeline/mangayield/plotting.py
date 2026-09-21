"""
Diagnostic figures. Uses a non-interactive Matplotlib backend and wraps each
figure so one failure never sinks the rest.
"""
from __future__ import annotations

from typing import List, Dict
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import rar as rar_mod


def _save(fig, path: str, figures: List[str]) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append(path)


def fig_recovery(records, outdir, figures):
    import numpy as np
    panels = [("alpha_med", "alpha_true", r"IMF slope $\alpha$"),
              ("y_eff_obs", "y_eff_true", r"effective yield $y_{\rm eff}$"),
              ("grad", "grad_true", r"O/H gradient [dex/$R_e$]"),
              ("g_dagger", "g_dagger_true", r"$g_\dagger$ [m s$^{-2}$]")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (ok, tk, lab) in zip(axes.ravel(), panels):
        o = np.array([r.get(ok, np.nan) for r in records], float)
        t = np.array([r.get(tk, np.nan) for r in records], float)
        m = np.isfinite(o) & np.isfinite(t)
        if m.sum() >= 2:
            ax.scatter(t[m], o[m], s=22, alpha=0.7, edgecolor="k", linewidth=0.3)
            lo = np.nanmin([t[m].min(), o[m].min()])
            hi = np.nanmax([t[m].max(), o[m].max()])
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            r = np.corrcoef(t[m], o[m])[0, 1]
            ax.set_title(f"{lab}\nr={r:.2f}, N={m.sum()}", fontsize=10)
        ax.set_xlabel("injected truth")
        ax.set_ylabel("recovered")
    fig.suptitle("Mock recovery: inferred vs injected", fontsize=13)
    _save(fig, os.path.join(outdir, "fig_recovery.png"), figures)


def fig_alpha_kinematics(records, outdir, figures):
    a = np.array([r.get("alpha_med", np.nan) for r in records], float)
    v = np.array([r.get("v_flat", np.nan) for r in records], float)
    lm = np.array([r.get("logmstar", np.nan) for r in records], float)
    alo = np.array([r.get("alpha_lo", np.nan) for r in records], float)
    ahi = np.array([r.get("alpha_hi", np.nan) for r in records], float)
    m = np.isfinite(a) & np.isfinite(v)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if m.sum() >= 2:
        yerr = np.vstack([a[m] - alo[m], ahi[m] - a[m]])
        ax.errorbar(v[m], a[m], yerr=yerr, fmt="none", ecolor="0.7",
                    elinewidth=0.6, zorder=1)
        sc = ax.scatter(v[m], a[m], c=lm[m], cmap="viridis", s=32,
                        edgecolor="k", linewidth=0.3, zorder=2)
        fig.colorbar(sc, ax=ax, label=r"$\log M_\star$")
    ax.set_xlabel(r"outer-disk rotation $V_{\rm flat}$ [km/s]")
    ax.set_ylabel(r"inferred SN/IMF slope $\alpha$ (median $\pm1\sigma$)")
    ax.set_title("Supernova mass distribution vs outer-disk kinematics")
    _save(fig, os.path.join(outdir, "fig_alpha_vs_kinematics.png"), figures)


def fig_rar(records, cfg, outdir, figures):
    gbar, gobs = [], []
    gdags = []
    for r in records:
        rr = r.get("_rar")
        if rr is None:
            continue
        gbar.append(rr.g_bar)
        gobs.append(rr.g_obs)
        if np.isfinite(rr.g_dagger):
            gdags.append(rr.g_dagger)
    if not gbar:
        return
    gbar = np.concatenate(gbar)
    gobs = np.concatenate(gobs)
    m = np.isfinite(gbar) & np.isfinite(gobs) & (gbar > 0) & (gobs > 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(np.log10(gbar[m]), np.log10(gobs[m]), s=8, alpha=0.35,
               edgecolor="none")
    xs = np.linspace(np.log10(gbar[m]).min(), np.log10(gbar[m]).max(), 100)
    gd = float(np.median(gdags)) if gdags else cfg.rar.gdagger_guess
    ax.plot(xs, np.log10(rar_mod.rar_function(10 ** xs, gd)), "r-", lw=2,
            label=fr"RAR fit, $g_\dagger$={gd:.2e}")
    ax.plot(xs, xs, "k--", lw=1, label="1:1 (no dark matter)")
    ax.set_xlabel(r"$\log g_{\rm bar}$ [m s$^{-2}$]")
    ax.set_ylabel(r"$\log g_{\rm obs}$ [m s$^{-2}$]")
    ax.set_title("Radial Acceleration Relation (all galaxies, all radii)")
    ax.legend(fontsize=9)
    _save(fig, os.path.join(outdir, "fig_rar.png"), figures)


def fig_degeneracy(records, outdir, figures):
    rec = next((r for r in records if r.get("_inf") is not None
                and r["_inf"].samples is not None), None)
    if rec is None:
        return
    s = rec["_inf"].samples
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.hist2d(s[:, 0], s[:, 1], bins=45, cmap="magma")
    ax.set_xlabel(r"IMF slope $\alpha$")
    ax.set_ylabel(r"$\log_{10}\eta$ (outflow loading)")
    tr = rec.get("alpha_true")
    if tr is not None:
        ax.axvline(tr, color="cyan", ls="--", lw=1.5, label=r"injected $\alpha$")
        ax.legend()
    ax.set_title(f"alpha-eta posterior ({rec['plateifu']})\n"
                 f"corr={rec['_inf'].alpha_eta_corr:.2f}")
    _save(fig, os.path.join(outdir, "fig_alpha_eta_degeneracy.png"), figures)


def fig_correlations(corrs, outdir, figures):
    corrs = [c for c in corrs if np.isfinite(c.r)]
    if not corrs:
        return
    labels = [f"{c.y} vs {c.x}" + (f"|{c.control}" if c.control else "")
              for c in corrs]
    r = np.array([c.r for c in corrs])
    lo = np.array([c.ci_lo for c in corrs])
    hi = np.array([c.ci_hi for c in corrs])
    y = np.arange(len(corrs))
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(corrs) + 2))
    err = np.vstack([r - lo, hi - r])
    colors = ["crimson" if c.significant() else "0.6" for c in corrs]
    ax.errorbar(r, y, xerr=err, fmt="o", ecolor="0.7", markersize=5)
    for yi, ci, col in zip(y, corrs, colors):
        ax.plot(ci.r, yi, "o", color=col, markersize=6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("correlation coefficient (partial where noted) with 68% CI")
    ax.set_title("Population correlations (red = p<0.05)")
    _save(fig, os.path.join(outdir, "fig_correlations.png"), figures)


def make_all(records, corrs, recov, cfg, outdir, is_mock=True) -> List[str]:
    figures: List[str] = []
    if is_mock:
        try:
            fig_recovery(records, outdir, figures)
        except Exception:
            pass
    for fn in (lambda: fig_alpha_kinematics(records, outdir, figures),
               lambda: fig_rar(records, cfg, outdir, figures),
               lambda: fig_degeneracy(records, outdir, figures),
               lambda: fig_correlations(corrs, outdir, figures)):
        try:
            fn()
        except Exception:
            pass
    return figures
