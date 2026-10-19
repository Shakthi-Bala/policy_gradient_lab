"""Print the latest CSV observation for every running or completed experiment."""

import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for path in sorted((root / "exp").glob("*/log.csv")):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        continue
    row = rows[-1]
    flags_path = path.parent / "flags.json"
    budget = (
        json.loads(flags_path.read_text())["n_iter"] if flags_path.exists() else "?"
    )
    iteration = int(float(row["Iteration"])) + 1
    peak = max(float(r["Eval_AverageReturn"]) for r in rows)
    print(
        f"{path.parent.name.split('_sd')[0]:52} {iteration:3}/{budget!s:3} "
        f"eval={float(row['Eval_AverageReturn']):8.1f} peak={peak:8.1f} "
        f"steps={int(float(row['Train_EnvstepsSoFar'])):8,}"
    )
