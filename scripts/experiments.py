"""Reproduce the required ablations without changing assignment hyperparameters."""

import argparse
import csv
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infrastructure.notify import send_update


def experiments():
    jobs = []
    for batch in (1000, 4000):
        for rtg, na in ((False, False), (True, False), (False, True), (True, True)):
            name = (
                "cartpole"
                + ("_lb" if batch == 4000 else "")
                + ("_rtg" if rtg else "")
                + ("_na" if na else "")
            )
            args = ["--env_name", "CartPole-v0", "-n", "100", "-b", str(batch)]
            args += (["-rtg"] if rtg else []) + (["-na"] if na else [])
            jobs.append({"group": "cartpole", "name": name, "args": args})
    for name, baseline, steps in [
        ("cheetah", False, 5),
        ("cheetah_baseline", True, 5),
        ("cheetah_baseline_bgs1", True, 1),
    ]:
        args = [
            "--env_name",
            "HalfCheetah-v4",
            "-n",
            "100",
            "-b",
            "5000",
            "-eb",
            "3000",
            "-rtg",
            "--discount",
            "0.95",
            "-lr",
            "0.01",
        ]
        if baseline:
            args += ["--use_baseline", "-blr", "0.01", "-bgs", str(steps)]
        jobs.append({"group": "cheetah", "name": name, "args": args})
    for lam in ("0", "0.95", "0.98", "0.99", "1"):
        args = [
            "--env_name",
            "LunarLander-v2",
            "--ep_len",
            "1000",
            "--discount",
            "0.99",
            "-n",
            "200",
            "-b",
            "2000",
            "-eb",
            "2000",
            "-l",
            "3",
            "-s",
            "128",
            "-lr",
            "0.001",
            "--use_reward_to_go",
            "--use_baseline",
            "--gae_lambda",
            lam,
            "--prune_lunar_particles",
        ]
        jobs.append(
            {"group": "lunar", "name": f"lunar_lander_lambda{lam}", "args": args}
        )
    jobs.append(
        {
            "group": "pendulum",
            "name": "pendulum",
            "args": [
                "--env_name",
                "InvertedPendulum-v4",
                "-n",
                "100",
                "-b",
                "5000",
                "-eb",
                "1000",
            ],
        }
    )
    for name, batch, lr, lam in [
        ("pendulum_tuned", 1000, "0.01", "0.95"),
        ("pendulum_tuned_small", 500, "0.005", "0.98"),
        ("pendulum_tuned_fast", 1000, "0.02", "0.95"),
    ]:
        jobs.append(
            {
                "group": "pendulum",
                "name": name,
                "args": [
                    "--env_name",
                    "InvertedPendulum-v4",
                    "-n",
                    "100",
                    "-b",
                    str(batch),
                    "-eb",
                    "1000",
                    "-rtg",
                    "-na",
                    "--discount",
                    "0.99",
                    "-lr",
                    lr,
                    "--use_baseline",
                    "--gae_lambda",
                    lam,
                    "-bgs",
                    "10",
                ],
            }
        )
    for job in jobs:
        job["args"] += [
            "--exp_name",
            job["name"],
            "--no_gpu",
            "--rollout_backend",
            "numpy",
        ]
    baseline = next(job for job in jobs if job["name"] == "cheetah_baseline")
    for seed in (2, 3):
        jobs.append(
            {
                "group": "cheetah",
                "name": "cheetah_baseline",
                "seed": seed,
                "args": [*baseline["args"], "--seed", str(seed)],
            }
        )
    normalized_args = baseline["args"].copy()
    normalized_args[normalized_args.index("--exp_name") + 1] = "cheetah_baseline_na"
    normalized_args.append("-na")
    jobs.append(
        {"group": "cheetah", "name": "cheetah_baseline_na", "args": normalized_args}
    )
    return jobs


def completed_directory(job):
    prefix = f"{job['args'][1]}_{job['name']}_sd{job.get('seed', 1)}_"
    for directory in sorted((ROOT / "exp").glob(prefix + "*"), reverse=True):
        flags_file = directory / "flags.json"
        if not flags_file.exists() or not (directory / "log.csv").exists():
            continue
        try:
            config = json.loads(flags_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        with (directory / "log.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if rows and int(float(rows[-1]["Iteration"])) == config["n_iter"] - 1:
            return directory
    return None


def running_process(job):
    """Adopt an identical live run after an orchestrator restart."""
    expected = ["src/scripts/run.py", *job["args"]]
    for process in psutil.process_iter(["cmdline", "cwd"]):
        try:
            command = process.info["cmdline"] or []
            if command[1:] == expected and process.info["cwd"] == str(ROOT):
                return process
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return None


def run_job(job, notify):
    # Two matrix invocations may overlap. Serialize each configuration, then
    # recheck completion so they never launch the same job twice.
    lock_dir = ROOT / "artifacts" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{job['name']}_sd{job.get('seed', 1)}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return run_locked_job(job, notify)


def run_locked_job(job, notify):
    name = job["name"]
    existing = completed_directory(job)
    if existing:
        return f"SKIP {name}: completed at {existing.name}"
    log_dir = ROOT / "artifacts" / "console"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy",
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
    )
    active = running_process(job)
    returncode = 0
    if active is not None:
        print(f"ADOPT {name}: waiting for existing PID {active.pid}", flush=True)
        while True:
            try:
                if not active.is_running() or active.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(2)
    else:
        with (log_dir / f"{name}_sd{job.get('seed', 1)}.log").open("w") as handle:
            result = subprocess.run(
                [sys.executable, "src/scripts/run.py", *job["args"]],
                check=False,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        returncode = result.returncode
    duration = (time.monotonic() - started) / 60
    directory = completed_directory(job)
    if returncode or directory is None:
        message = f"FAILED {name} seed {job.get('seed', 1)}; inspect artifacts/console/{name}_sd{job.get('seed', 1)}.log"
    else:
        with (directory / "log.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        duration = float(rows[-1]["TimeSinceStart"]) / 60
        peak = max(float(row["Eval_AverageReturn"]) for row in rows)
        final = float(rows[-1]["Eval_AverageReturn"])
        message = (
            f"{name}: completed {len(rows)} iterations in {duration:.1f} min. "
            f"Eval return peak={peak:.1f}, final={final:.1f}; "
            f"training steps={int(float(rows[-1]['Train_EnvstepsSoFar'])):,}. Seed {job.get('seed', 1)}."
        )
    print(message, flush=True)
    if notify:
        send_update(message)
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=["all", "cartpole", "cheetah", "lunar", "pendulum"],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    jobs = [j for j in experiments() if args.group == "all" or j["group"] == args.group]
    if args.list:
        print("# Generated exact benchmark commands; run from the repository root.")
        print("export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")
        for job in jobs:
            print(shlex.join(["uv", "run", "src/scripts/run.py", *job["args"]]))
        return
    # Run longer tasks first so short CartPole jobs fill the remaining capacity.
    jobs.sort(
        key=lambda j: {"lunar": 0, "cheetah": 1, "pendulum": 2, "cartpole": 3}[
            j["group"]
        ]
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job, args.notify) for job in jobs]
        messages = [future.result() for future in as_completed(futures)]
    raise SystemExit(
        1 if any(message.startswith("FAILED") for message in messages) else 0
    )


if __name__ == "__main__":
    main()
