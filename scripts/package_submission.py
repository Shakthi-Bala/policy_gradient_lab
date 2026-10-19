"""Build the assignment's <100 MB archive from completed, validated runs only."""

import argparse
import csv
import json
import zipfile
from pathlib import Path

from experiments import experiments

ROOT = Path(__file__).resolve().parents[1]


def validate_run(directory):
    for name in ("agent.pt", "flags.json", "log.csv", "log.pkl", "provenance.json"):
        if not (directory / name).is_file():
            raise ValueError(f"Missing {directory.name}/{name}")
    flags = json.loads((directory / "flags.json").read_text())
    with (directory / "log.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(float(rows[-1]["Iteration"])) != flags["n_iter"] - 1:
        raise ValueError(f"Incomplete run: {directory.name}")
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submit.zip")
    args = parser.parse_args()
    directories = []
    names = set()
    for directory in sorted((ROOT / "exp").iterdir()):
        if directory.is_dir():
            flags = validate_run(directory)
            directories.append(directory)
            names.add(flags["exp_name"])
    missing = {job["name"] for job in experiments()} - names
    if missing:
        raise ValueError(
            f"Missing required experiment configurations: {sorted(missing)}"
        )
    files = []
    for directory in directories:
        files.extend(
            directory / name
            for name in (
                "agent.pt",
                "flags.json",
                "log.csv",
                "log.pkl",
                "provenance.json",
            )
        )
    files.extend((ROOT / "src").rglob("*.py"))
    files.extend(
        ROOT / name
        for name in ("pyproject.toml", "uv.lock", "README.md", ".python-version")
    )
    files.extend((ROOT / "scripts").glob("*.py"))
    files.extend((ROOT / "tests").glob("*.py"))
    files.extend(path for path in (ROOT / "docs").iterdir() if path.is_file())
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    if args.output.stat().st_size >= 100_000_000:
        args.output.unlink()
        raise ValueError("Submission exceeds the 100 MB limit")
    print(
        f"{args.output.name}: {len(directories)} completed runs, {args.output.stat().st_size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
