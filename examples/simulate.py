"""Génère des filtres SIMULÉS à 3 phases (constante / linéaire / hausse exponentielle),
avec bruit, chutes transitoires et redémarrages. Aucune donnée réelle."""
import numpy as np
import pandas as pd


def simulate_filter(fid, life_h=400.0, u0=0.8, seed=0, n_restarts=2):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, life_h, int(life_h * 8)))          # heures moteur
    u = t / life_h
    base = rng.uniform(0.8, 1.2)
    dp = base * (1 + 0.15 * u) + base * 0.5 * (np.exp(4.0 * np.clip(u - u0, 0, None) / (1 - u0)) - 1) * (u > u0)
    dp += rng.normal(0, 0.015 * base, t.size)
    # redémarrages : trou d'horloge de 48 h, chute transitoire de ~8 h, décalage de niveau
    stamp = pd.Timestamp("2024-01-01") + pd.to_timedelta(t, unit="h")
    for tr in np.sort(rng.uniform(0.1 * life_h, 0.6 * life_h, n_restarts)):
        m = t >= tr
        stamp = stamp.where(~m, stamp + pd.Timedelta(hours=48))
        dp[(t >= tr) & (t < tr + 8)] -= 0.35 * base * np.exp(-(t[(t >= tr) & (t < tr + 8)] - tr) / 3)
        dp[m] += rng.normal(0, 0.05 * base)
    out = pd.DataFrame(dict(filter_id=fid, time=t, delta_p=dp, timestamp=stamp))
    # fin de vie = Δp atteint 3 × base ; colmatage = Δp ≥ 2 × base
    end = np.argmax(dp > 3.0 * base) if (dp > 3.0 * base).any() else len(dp) - 1
    out = out.iloc[: end + 1].copy()
    out["clogged"] = (out.delta_p >= 2.0 * base).cummax().astype(int)
    out["stable"] = 1
    return out


def simulate_dataset():
    return pd.concat([simulate_filter(1, 400, 0.80, 1), simulate_filter(2, 300, 0.75, 2),
                      simulate_filter(3, 500, 0.85, 3)], ignore_index=True)
