"""Préparation générique d'une série Δp (un filtre) : nettoyage des redémarrages + binning.

Entrée : DataFrame d'UN filtre avec colonnes (noms configurables)
  time   : heures moteur (croissant)           delta_p : pression différentielle
  stamp  : horodatage (optionnel, sert à repérer les redémarrages = longs trous d'horloge)
  stable : 1 si régime stable (optionnel)       clogged : 0/1 (optionnel, début de colmatage)
"""
import numpy as np
import pandas as pd

GAP_H = 24.0      # trou d'horloge (h) considéré comme un redémarrage
TRANSIT = 15.0    # heures retirées après un redémarrage (chute transitoire)
DROP_RATIO = 0.8  # chute isolée : dp < ratio * médiane glissante (~20 h)


def prepare(x, time="time", dp="delta_p", stamp=None, stable=None, clogged=None, bin_h=0.5, clean=True):
    x = x.sort_values(time).reset_index(drop=True)
    eh, y = x[time].values.astype(float), x[dp].values.astype(float)
    t0, t_end = eh[0], eh[-1]
    T_clog = dp_clog = None
    if clogged is not None and (x[clogged] == 1).any():
        T_clog = x.loc[x[clogged] == 1, time].min()
        dp_clog = float(np.median(y[eh >= T_clog][:5]))
    dp_end = float(np.median(y[-5:]))

    keep, breaks = np.ones(len(x), bool), []
    if clean and stamp is not None:
        wall = pd.to_datetime(x[stamp], utc=True, format="mixed").diff().dt.total_seconds().values / 3600
        for r in np.where(wall > GAP_H)[0]:
            breaks.append(eh[r] - t0)
            keep[(eh >= eh[r]) & (eh < eh[r] + TRANSIT)] = False
    if stable is not None:
        keep &= (x[stable] == 1).values
    age = eh[keep] - t0
    g = pd.Series(y[keep]).groupby((age // bin_h) * bin_h).median()
    t, v = g.index.values + bin_h / 2, g.values
    if clean:  # chutes isolées
        med = pd.Series(v).rolling(41, center=True, min_periods=8).median().values
        ok = ~(v < DROP_RATIO * med)
        t, v = t[ok], v[ok]
    return dict(t=t, dp=v, T_end=t_end - t0, T_clog=None if T_clog is None else T_clog - t0,
                dp_clog=dp_clog, dp_end=dp_end, breaks=np.array(breaks))
