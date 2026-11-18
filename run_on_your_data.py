"""Applique CUSUM + RUL à VOS données (CSV non fournis dans ce dépôt) et produit des figures normalisées.

Exemple :
  python run_on_your_data.py --csv mes_donnees.csv --id-col filter_id --time-col time \\
      --dp-col delta_p --stamp-col timestamp --stable-col stable --clogged-col clogged \\
      --ids 1 2 3 --out images
Le CSV doit avoir une ligne par mesure, avec une colonne identifiant le filtre.
"""
import argparse, os
import pandas as pd
from prep import prepare
from pipeline import process_filter, plot_filter, plot_overview, ALPHA_REF, WINDOW, H, N_EXCEED

a = argparse.ArgumentParser()
a.add_argument("--csv", required=True); a.add_argument("--id-col", required=True)
a.add_argument("--time-col", required=True); a.add_argument("--dp-col", required=True)
a.add_argument("--stamp-col"); a.add_argument("--stable-col"); a.add_argument("--clogged-col")
a.add_argument("--ids", nargs="*", help="identifiants de filtres (défaut : tous)")
a.add_argument("--alpha-ref", type=float, default=ALPHA_REF); a.add_argument("--window", type=float, default=WINDOW)
a.add_argument("--threshold", type=float, default=H); a.add_argument("--n-exceed", type=int, default=N_EXCEED)
a.add_argument("--out", default="images")
args = a.parse_args()

cols = [c for c in [args.id_col, args.time_col, args.dp_col, args.stamp_col, args.stable_col, args.clogged_col] if c]
df = pd.read_csv(args.csv, usecols=cols, low_memory=False)
ids = args.ids or sorted(df[args.id_col].dropna().unique())
os.makedirs(args.out, exist_ok=True)
items, rows = [], []
for fid in ids:
    x = df[df[args.id_col].astype(str).str.replace(r"\.0$", "", regex=True) == str(fid)]
    if x.empty: continue
    s = prepare(x, args.time_col, args.dp_col, args.stamp_col, args.stable_col, args.clogged_col)
    r = process_filter(s, args.alpha_ref, args.window, args.threshold, args.n_exceed)
    plot_filter(f"Filtre {fid}", r, os.path.join(args.out, f"filter_{fid}.png"))
    items.append((f"Filtre {fid}", r))
    rows.append(dict(filtre=fid, rupture_norm=None if r["tc"] is None else round(r["tc"] / r["T_end"], 3),
                     RMSE_norm=None if r["tc"] is None else round(r["rmse_end"] / r["T_end"], 3)))
plot_overview(items, os.path.join(args.out, "overview.png"))
print(pd.DataFrame(rows).to_string(index=False))   # grandeurs normalisées uniquement
