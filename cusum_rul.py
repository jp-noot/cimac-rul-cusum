"""
Estimation de la RUL de filtres moteur : détection de rupture CUSUM (pente) +
ajustement exponentiel contraint.
Réimplémentation de : J.-P. Noot, M. Martin, E. Birmele, "Change point detection method
for estimating the remaining useful lifetime of engine filters", CIMAC Congress 2025, paper 471.

Classes : CusumDetector (section 3.2), RulEstimator (section 4), fonction run_filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------
# 1. CUSUM adapté à la pente (section 3.2, éq. 2 à 6)
# --------------------------------------------------------------------------
def _gauss_loglik(resid: np.ndarray) -> float:
    """Log-vraisemblance gaussienne, variance estimée par maximum de vraisemblance."""
    n = resid.size
    var = max(float(np.mean(resid**2)), 1e-12)
    return -0.5 * n * (np.log(2 * np.pi * var) + 1.0)


def window_score(t: np.ndarray, y: np.ndarray, alpha_ref: float) -> float:
    """Score s_i = log(L1/L0) d'une fenêtre (éq. 5).

    M0 : y = beta + eps                      (pente nulle)
    M1 : y = alpha* t + beta + eps, alpha* = max(alpha_ref, alpha_hat)  (éq. 3-4)
    beta et la variance sont estimés dans les deux modèles.
    """
    t = t - t.mean()  # centrage : n'affecte pas la pente, découple beta et alpha
    # M0
    ll0 = _gauss_loglik(y - y.mean())
    # M1 : pente estimée par MCO puis planchée à alpha_ref
    alpha_hat = float(np.dot(t, y - y.mean()) / np.dot(t, t))
    alpha_star = max(alpha_ref, alpha_hat)
    beta = float(np.mean(y - alpha_star * t))
    ll1 = _gauss_loglik(y - alpha_star * t - beta)
    return ll1 - ll0


@dataclass
class CusumDetector:
    """Détection en ligne du point de changement de pente.

    alpha_ref    : pente de référence (unité de y par unité de t)
    window       : durée d'une fenêtre (même unité que t, ex. heures moteur)
    threshold    : seuil h sur la somme cumulée S_i
    step         : intervalle de création des fenêtres (par défaut = window,
                   fenêtres disjointes ; < window = fenêtres glissantes)
    min_points   : nb minimum de points pour évaluer une fenêtre
    n_exceed     : nb de fenêtres CONSÉCUTIVES avec S_i > h exigé pour valider
                   la rupture (1 = règle du papier). Le compteur repart à 0 si
                   S_i retombe sous h. Extension anti fausses alertes.
    """

    alpha_ref: float
    window: float
    threshold: float
    step: float | None = None
    min_points: int = 5
    n_exceed: int = 1

    scores: list = field(default_factory=list, init=False)   # s_i
    cusum: list = field(default_factory=list, init=False)    # S_i
    window_ends: list = field(default_factory=list, init=False)

    def run(self, t: np.ndarray, y: np.ndarray, t_max: float | None = None, breaks=None):
        """Rejoue la série jusqu'à t_max. Retourne (t_change, index) ou (None, None).

        Le point de changement est le temps de fin de la 1re fenêtre dont
        S_i dépasse le seuil (la détection n'utilise que le passé).
        breaks : instants de redémarrage ; une fenêtre n'enjambe jamais un redémarrage
        (elle est tronquée au dernier redémarrage), S_i continue de s'accumuler.
        """
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        step = self.step or self.window
        self.scores, self.cusum, self.window_ends = [], [], []

        S = 0.0
        run_count, first_end = 0, None
        end = t[0] + self.window
        t_stop = t[-1] if t_max is None else min(t_max, t[-1])
        while end <= t_stop + 1e-12:
            lo = end - self.window
            if breaks is not None and len(breaks):
                b = breaks[breaks < end]
                if len(b): lo = max(lo, b.max())
            m = (t > lo) & (t <= end)
            if m.sum() >= self.min_points:
                s = window_score(t[m], y[m], self.alpha_ref)
                S = max(S + s, 0.0)  # éq. 6
                self.scores.append(s)
                self.cusum.append(S)
                self.window_ends.append(end)
                if S > self.threshold:
                    if run_count == 0:
                        first_end = end
                    run_count += 1
                    if run_count >= self.n_exceed:
                        # rupture = 1re fenêtre de la série de dépassements, confirmée après n_exceed
                        idx = int(np.searchsorted(t, first_end, side="right") - 1)
                        return first_end, idx
                else:
                    run_count = 0
            end += step
        return None, None


# --------------------------------------------------------------------------
# 2. Ajustement exponentiel contraint + RUL (section 4, éq. 7)
# --------------------------------------------------------------------------
@dataclass
class RulEstimator:
    """Modèle : dP(t) = a * exp(b * (t/c_norm)^2) + c,  a >= 0, b >= 0.

    Paramètres (Tableau 2 du papier) : c_norm=750, c_begin=0.2,
    h_begin=2 h, h_end=40 h.
    Initialisation : c = moyenne des premiers points ; a ~ [1e-20, 1] ;
    b ~ [1, 150] ; n_init tirages, moyenne des n_best meilleurs (10 / 6).
    """

    threshold: float                 # seuil de colmatage sur dP
    c_norm: float = 750.0
    c_begin: float = 0.2
    h_begin: float = 2.0
    h_end: float = 40.0
    n_init: int = 10
    n_best: int = 6
    seed: int | None = 0

    # -- modèle
    def _f(self, p, tn):
        a, b, c = p
        return a * np.exp(np.clip(b * tn**2, None, 700.0)) + c

    def _select(self, t, y, t0, t_change, t_present):
        t_begin = t_change - self.c_begin * (t_change - t0)
        m1 = (t >= t_begin) & (t <= t_begin + self.h_begin)
        m2 = (t >= t_present - self.h_end) & (t <= t_present)
        m = m1 | m2
        return m, m1

    def fit(self, t, y, t0, t_change, t_present):
        """Ajuste le modèle avec les données jusqu'à t_present.
        Retourne (liste des (loss, params)) triée, ou None si pas assez de points."""
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        past = t <= t_present
        t, y = t[past], y[past]
        m, m1 = self._select(t, y, t0, t_change, t_present)
        if m.sum() < 4:
            return None
        tn, yy = t[m] / self.c_norm, y[m]
        c_init = float(y[m1].mean()) if m1.any() else float(yy.min())

        rng = np.random.default_rng(self.seed)
        loss = lambda p: float(np.mean((self._f(p, tn) - yy) ** 2))
        fits = []
        for _ in range(self.n_init):
            a0 = 10 ** rng.uniform(-20, 0)
            b0 = rng.uniform(1, 150)
            # BFGS du papier ne gère pas les bornes -> L-BFGS-B (BFGS à mémoire
            # limitée) avec a>=0, b>=0 ; c libre.
            r = minimize(loss, [a0, b0, c_init], method="L-BFGS-B",
                         bounds=[(0, None), (0, None), (None, None)])
            fits.append((r.fun, r.x))
        fits.sort(key=lambda f: f[0])
        return fits

    def predict_detail(self, t, y, t0, t_change, t_present):
        """Retourne (RUL moyenne, liste des paramètres (a,b,c) retenus)."""
        fits = self.fit(t, y, t0, t_change, t_present)
        if fits is None:
            return np.nan, []
        ruls, params = [], []
        for _, (a, b, c) in fits[: self.n_best]:
            if a <= 0 or b <= 0:
                continue  # courbe plate : pas de croisement du seuil
            params.append((a, b, c))
            if self.threshold <= c + a:
                ruls.append(0.0)  # seuil déjà atteint à t=0 du modèle
                continue
            t_hit = self.c_norm * np.sqrt(np.log((self.threshold - c) / a) / b)
            ruls.append(max(t_hit - t_present, 0.0))
        return (float(np.mean(ruls)) if ruls else np.nan), params

    def predict_rul(self, t, y, t0, t_change, t_present):
        """RUL estimée à t_present (moyenne des n_best meilleurs ajustements)."""
        return self.predict_detail(t, y, t0, t_change, t_present)[0]

    def curve(self, params, t):
        return self._f(params, np.asarray(t, float) / self.c_norm)


# --------------------------------------------------------------------------
# 3. Pipeline complet sur un filtre
# --------------------------------------------------------------------------
def run_filter(t, dp, detector: CusumDetector, estimator: RulEstimator, rul_every=1.0):
    """Simule l'exécution en ligne. Retourne (t_change, [(t_present, RUL), ...])."""
    t = np.asarray(t, float)
    dp = np.asarray(dp, float)
    t_change, _ = detector.run(t, dp)
    if t_change is None:
        return None, []
    out, last = [], -np.inf
    for tp in t[t >= t_change]:
        if tp - last < rul_every:
            continue
        last = tp
        out.append((tp, estimator.predict_rul(t, dp, t[0], t_change, tp)))
    return t_change, out


def rmse_vs_true(preds, t_end):
    """RMSE entre RUL estimée et RUL vraie (t_end - t) ; t_end = fin de vie réelle."""
    p = np.array([(r, t_end - tp) for tp, r in preds if np.isfinite(r)])
    return float(np.sqrt(np.mean((p[:, 0] - p[:, 1]) ** 2))) if len(p) else np.nan


def grid_search_threshold(filters, annotated_changes, alpha_ref, window, thresholds):
    """Choisit h minimisant la distance moyenne |t_detecté - t_annoté| (section 5.2).

    filters : liste de (t, dp) ; annotated_changes : liste de t annotés.
    Une non-détection est pénalisée par (t_fin - t_annoté).
    """
    best = (np.inf, None)
    for h in thresholds:
        d = []
        for (t, y), ta in zip(filters, annotated_changes):
            tc, _ = CusumDetector(alpha_ref, window, h).run(t, y)
            d.append(abs(tc - ta) if tc is not None else abs(t[-1] - ta))
        if np.mean(d) < best[0]:
            best = (float(np.mean(d)), h)
    return best[1], best[0]
