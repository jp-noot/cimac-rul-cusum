"""Pipeline complet pour un filtre (CUSUM + RUL) et figures NORMALISÉES.

Normalisation des figures (aucune valeur physique n'apparaît) :
  temps  : t / T_end   (T_end = durée de vie du filtre)
  Δp     : Δp / Δp_fin
  RUL    : RUL / T_end
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cusum_rul import CusumDetector, RulEstimator

# paramètres CUSUM par défaut (Δp normalisé par Δp_fin, temps en heures) : à ajuster à vos données
ALPHA_REF, WINDOW, H, N_EXCEED = 0.005, 10.0, 20.0, 2


def _rmse(pred, true):
    m = np.isfinite(pred) & (true >= 0)
    return float(np.sqrt(np.mean((pred[m] - true[m]) ** 2))) if m.any() else np.nan


def process_filter(s, alpha_ref=ALPHA_REF, window=WINDOW, h=H, n_exceed=N_EXCEED, rul_step=2.0, bin_h=0.5):
    """s : dict renvoyé par prep.prepare. Retourne un dict de résultats (grandeurs en heures)."""
    t, y = s["t"], s["dp"] / s["dp_end"]
    T_end, T_clog = s["T_end"], s["T_clog"]
    det = CusumDetector(alpha_ref, window, h, n_exceed=n_exceed)
    tc, _ = det.run(t, y, breaks=s["breaks"])
    res = dict(t=t, y=y, det=det, tc=tc, T_end=T_end, T_clog=T_clog, thr_end=1.0,
               thr_clog=None if T_clog is None else s["dp_clog"] / s["dp_end"])
    if tc is None:
        return res
    est_e = RulEstimator(threshold=1.0)
    est_c = None if T_clog is None else RulEstimator(threshold=res["thr_clog"])
    tps = t[t >= tc][::max(1, int(round(rul_step / bin_h)))]
    res.update(est_e=est_e, tp=tps, fits=[], rul_end=[], rul_clog=[])
    for tp in tps:
        r, p = est_e.predict_detail(t, y, 0.0, tc, tp)
        res["rul_end"].append(r); res["fits"].append((tp, p))
        if est_c is not None:
            res["rul_clog"].append(est_c.predict_detail(t, y, 0.0, tc, tp)[0])
    res["rul_end"], res["rul_clog"] = np.array(res["rul_end"]), np.array(res["rul_clog"])
    res["rmse_end"] = _rmse(res["rul_end"], T_end - tps)
    if T_clog is not None:
        res["rmse_clog"] = _rmse(res["rul_clog"], T_clog - tps)
    res["lead_end"] = T_end - tc
    return res


def plot_filter(name, r, path):
    """Figure normalisée : RUL | Δp + projections exponentielles | score CUSUM."""
    T = r["T_end"]; tc = r["tc"]
    fig, ax = plt.subplots(3, 1, figsize=(8, 9.5), height_ratios=[1, 1.3, 0.8])
    ax[1].plot(r["t"] / T, r["y"], ".", ms=3, color="navy", label="Δp (normalisé)")
    ax[1].axhline(1.0, color="grey", ls=":", label="seuil fin de vie")
    if r["thr_clog"] is not None:
        ax[1].axhline(r["thr_clog"], color="r", ls=":", label="seuil colmatage (clogged)")
        ax[1].axvline(r["T_clog"] / T, color="r", lw=1, label="début colmatage")
    if tc is None:
        ax[1].set_title(f"{name} : aucune rupture détectée"); ax[0].axis("off"); ax[2].axis("off")
    else:
        ax[1].axvline(tc / T, color="orange", label="rupture CUSUM")
        tp = r["tp"]
        ax[0].plot(tp / T, (T - tp) / T, "--", color="grey", label="RUL vraie (fin de vie)")
        ax[0].plot(tp / T, r["rul_end"] / T, ".-", color="tab:green", ms=3, label=f"RUL estimée (RMSE {r['rmse_end'] / T:.3f})")
        if r["T_clog"] is not None:
            ax[0].plot(tp / T, np.maximum(r["T_clog"] - tp, 0) / T, color="lightblue", lw=3, label="RUL vraie (colmatage)")
            ax[0].plot(tp / T, r["rul_clog"] / T, "k.-", ms=3, label=f"RUL estimée seuil colmatage (RMSE {r['rmse_clog'] / T:.3f})")
        ax[0].axvline(tc / T, color="orange"); ax[0].set_ylabel("RUL / durée de vie"); ax[0].legend(fontsize=7)
        ax[0].set_title(f"{name} — RUL (temps normalisé)")
        k = max(1, len(r["fits"]) // 8)
        for j, (tpk, ps) in enumerate(r["fits"][::k]):
            if ps:
                tt = np.linspace(max(tpk - 40, 0), T * 1.02, 300)
                ax[1].plot(tt / T, r["est_e"].curve(ps[0], tt), color=plt.cm.viridis(j / 8), lw=1.2)
        ax[2].plot(np.array(r["det"].window_ends) / T, r["det"].cusum, "g.-", label="S_i")
        ax[2].axhline(r["det"].threshold, color="r", ls="--", label="seuil h"); ax[2].axvline(tc / T, color="orange")
        ax[2].set_ylabel("score CUSUM"); ax[2].set_xlabel("temps normalisé"); ax[2].legend(fontsize=7)
    ax[1].set_ylim(0, 1.4); ax[1].set_ylabel("Δp normalisé"); ax[1].legend(fontsize=7, loc="upper left")
    ax[1].set_title("Δp et projections exponentielles")
    for a in ax:
        a.set_xlim(0, 1.05); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def plot_overview(items, path):
    """items : liste (nom, résultat). Vue d'ensemble Δp normalisé + rupture CUSUM."""
    n = len(items); nc = 3; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(13, 3.2 * nr), squeeze=False); axs = axs.ravel()
    for a, (nm, r) in zip(axs, items):
        a.plot(r["t"] / r["T_end"], r["y"], ".", ms=2, color="navy")
        if r["tc"] is not None: a.axvline(r["tc"] / r["T_end"], color="orange")
        if r["T_clog"] is not None: a.axvline(r["T_clog"] / r["T_end"], color="r", lw=1)
        a.set_title(nm, fontsize=10); a.set_xlabel("temps normalisé"); a.set_ylabel("Δp normalisé"); a.grid(alpha=.3)
    for a in axs[n:]: a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
