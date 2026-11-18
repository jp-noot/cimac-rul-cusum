"""Démo complète sur données simulées. Usage (depuis la racine du dépôt) :
    python examples/demo_simulated.py          -> images/simulated_*.png"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from simulate import simulate_dataset
from prep import prepare
from pipeline import process_filter, plot_filter, plot_overview

df = simulate_dataset()
os.makedirs("images", exist_ok=True)
items, rows = [], []
for fid, x in df.groupby("filter_id"):
    s = prepare(x, time="time", dp="delta_p", stamp="timestamp", stable="stable", clogged="clogged")
    r = process_filter(s)
    name = f"Filtre simulé {fid}"
    plot_filter(name, r, f"images/simulated_filter_{fid}.png")
    items.append((name, r))
    rows.append(dict(filtre=fid, duree_h=round(r["T_end"]), rupture_h=None if r["tc"] is None else round(r["tc"]),
                     RMSE_RUL_h=None if r["tc"] is None else round(r["rmse_end"], 1)))
plot_overview(items, "images/simulated_overview.png")
print(pd.DataFrame(rows).to_string(index=False))
