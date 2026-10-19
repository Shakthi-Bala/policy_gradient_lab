# Development and reproduction

Tested on Linux x86_64 with Python 3.10 and a one-core CPU quota. Dependencies
are isolated in `.venv` and pinned in `uv.lock`.

```bash
uv sync --locked
uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

## Full benchmark

```bash
uv run python scripts/experiments.py --list
uv run python scripts/experiments.py --workers 4 --notify
uv run python scripts/status.py
uv run python scripts/analyze.py
uv run python scripts/package_submission.py
```

The runner launches independent processes, caps BLAS threads to one, and skips
completed configurations. Per-configuration file locks prevent duplicate jobs
across overlapping runner invocations. If the orchestrator is restarted while a
matching training process is still alive, it adopts that process and waits for
its completed artifacts instead of starting over. `--group cartpole`, `cheetah`, `lunar`, or `pendulum`
limits its scope. Console output is saved in ignored `artifacts/console/`; all
metrics/configs/checkpoints needed for reproduction are versioned under `exp/`.
The LunarLander commands enable a tested visual-particle cleanup option; see
[PROVENANCE.md](PROVENANCE.md) for exact-equivalence checks and a CPU benchmark.
The NumPy inference backend is specified in the generated commands, so it is
also recorded in `flags.json`. The default training CLI uses PyTorch inference.

A batch can exceed its nominal step count because rollouts finish their episodes.
Every tenth iteration refreshes `agent.pt`, `flags.json`, and `log.pkl`.
Only the model checkpoint is replaced atomically. The final checkpoint contains network parameters, not optimizer
state or a resumable simulator/RNG snapshot. Partial CSVs can be inspected while
training; analysis and packaging only accept completed runs. If a process is
interrupted, rerunning the matrix starts that configuration from its initial seed
in a fresh directory. Archive incomplete directories under `artifacts/` before
packaging; they must not be mistaken for completed assignment deliverables.

## Policy demonstrations

Choose a completed run directory and export its **final** checkpoint:

```bash
uv run python scripts/render_policy.py exp/<run-directory> \
  --output media/cartpole_demo --seed 42 --notify
```

The exporter saves an MP4, a compact GIF, and JSON with its seed, episodic return,
checkpoint identity, and reproduction command. Policies remain stochastic. Demo
scores are single episodes, not average benchmark returns. MuJoCo uses EGL and
classic-control/Box2D use pygame's dummy display for headless rendering. Override
`MUJOCO_GL` if the local host requires a different supported rendering backend.

## Cloud tracking (optional)

```bash
uv sync --extra tracking
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  -rtg -na --exp_name cartpole_rtg_na --wandb_mode online
```

The upstream `src/scripts/modal_run.py` is preserved for compatibility. Install
`uv sync --extra cloud --extra tracking` to use it; it is not needed or used for
these experiments. The reproduced results use local CPU execution.

## Notifications

`.env` is private, ignored by Git, and not copied into submission archives. Its
only supported value is `DISCORD_WEBHOOK_URL`. No other secrets are read by the
notification code. The experiment runner posts one summary per completed or failed
run when `--notify` is present; rendering posts the corresponding media. Missing
credentials or failed delivery do not invalidate training results. Notification
exceptions print only exception classes to avoid exposing URLs in logs.

GIF exports default to a 12-second preview; MP4s retain the full episode. Use
`--gif_seconds` to change preview duration. Large original GIFs are retained locally
in ignored `artifacts/media_full/`. The assignment archive excludes media assets
but includes nested JSON evaluation evidence and the self-contained PDF report.
