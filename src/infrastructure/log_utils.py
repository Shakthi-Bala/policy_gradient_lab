"""Local-first CSV/checkpoint logging; optional Weights & Biases mirroring."""

import copy
import csv
import hashlib
import json
import os
import pickle
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import gym
import imageio.v2 as imageio
import numpy as np
import torch

_wandb_run = None


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.file = None
        self.writer = None
        self.rows = []

    def log(self, row, step):
        row = {k: float(v) for k, v in row.items()}
        row["step"] = step
        if self.file is None:
            self.file = self.path.open("w", newline="")
            self.writer = csv.DictWriter(
                self.file, fieldnames=list(row), extrasaction="raise"
            )
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()
        self.rows.append(copy.deepcopy(row))
        if _wandb_run is not None:
            _wandb_run.log(row, step=step)

    def log_trajs_as_videos(
        self, trajs, step, max_videos_to_save=2, fps=30, video_title="eval"
    ):
        directory = self.path.parent / "videos"
        directory.mkdir(exist_ok=True)
        for i, traj in enumerate(trajs[:max_videos_to_save]):
            destination = directory / f"{video_title}_{step:04d}_{i}.mp4"
            imageio.mimsave(
                destination, traj["image_obs"], fps=int(fps), macro_block_size=1
            )
            if _wandb_run is not None:
                import wandb

                _wandb_run.log(
                    {video_title: wandb.Video(str(destination), format="mp4")},
                    step=step,
                )

    def close(self):
        if self.file is not None:
            self.file.close()


def dump_log(agent, logger, args, save_dir):
    config = vars(args).copy()
    data = {
        "log": logger.rows,
        "config": config,
        "log_hash": hashlib.sha256(
            json.dumps(logger.rows, sort_keys=True).encode()
        ).hexdigest(),
        "time": datetime.now(timezone.utc).isoformat(),
    }
    directory = Path(save_dir)
    (directory / "flags.json").write_text(json.dumps(config, indent=2) + "\n")
    with (directory / "log.pkl").open("wb") as handle:
        pickle.dump(data, handle)
    # Atomic replacement prevents interrupted writes corrupting the last checkpoint.
    torch.save(agent.state_dict(), directory / "agent.tmp")
    os.replace(directory / "agent.tmp", directory / "agent.pt")


def write_provenance(directory, argv):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except subprocess.CalledProcessError:
        revision, dirty = None, None
    data = {
        "command": argv,
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gym": gym.__version__,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (Path(directory) / "provenance.json").write_text(json.dumps(data, indent=2) + "\n")


def setup_wandb(
    entity=None,
    project="policy-gradient-lab",
    group=None,
    name=None,
    mode="disabled",
    config=None,
):
    global _wandb_run
    if mode != "disabled":
        import wandb

        _wandb_run = wandb.init(
            entity=entity,
            project=project,
            group=group,
            name=name,
            mode=mode,
            config=config,
        )
    return _wandb_run
