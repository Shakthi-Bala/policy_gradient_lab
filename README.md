# Policy Gradient Lab

**Learning control from experience with REINFORCE, value baselines, and generalized advantage estimation.**

A reproducible reinforcement learning study spanning discrete control, continuous
locomotion, and sample efficiency. Built from the Berkeley CS285 Spring 2026
[policy gradient assignment](hw2.pdf), with implemented learning algorithms,
local experiment tracking, correctness tests, and automated comparisons.

## Engineering highlights

- **Policy learning:** categorical and diagonal Gaussian neural policies trained
  with Monte Carlo policy gradients.
- **Variance reduction:** trajectory returns, reward-to-go, advantage
  normalization, learned state-value baselines, and GAE with tunable λ.
- **Reproducibility:** locked CPU dependencies, explicit seeds, independent
  evaluation environments, actual transition counts, source provenance, and
  recoverable checkpoints.
- **Experiment workflow:** CSV metrics and model artifacts without cloud accounts;
  optional Weights & Biases tracking and private Discord updates.
- **Verification:** numerical tests for discounted returns, GAE episode boundaries,
  probability gradients, baseline regression, and rollout budgets.

## Quick start

Requires Linux/macOS, Python 3.10, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 \
  -rtg -na --exp_name cartpole_rtg_na --no_gpu
```

The lockfile uses CPU PyTorch. MuJoCo and Box2D install through `uv`; the build
configuration supplies SWIG for Box2D automatically. No API key is needed.

## Experiments

| Study | Environment | Question |
|---|---|---|
| Return estimator ablation | CartPole-v0 | How do reward-to-go, normalization, and batch size affect learning? |
| Value baseline | HalfCheetah-v4 | Does learning a baseline improve continuous control? |
| Bias–variance tradeoff | LunarLander-v2 | How does GAE λ affect policy performance? |
| Sample efficiency | InvertedPendulum-v4 | Can tuning reach return 1,000 within 100,000 training transitions? |

Experiments and the empirical report are being prepared. Results will be linked
here after completed runs; the assignment's target scores are not claimed as
observed performance.

## Repository

```text
src/agents/pg_agent.py      Return estimators, advantage calculation, actor/critic updates
src/networks/              Categorical/Gaussian policies and value regression
src/infrastructure/        Rollouts, PyTorch helpers, logging, private notifications
src/scripts/run.py         Assignment-compatible training CLI
scripts/                   Experiment orchestration, plots, media and packaging
exp/                       Reproducible logs, flags and trained checkpoints
media/                     Generated learning curves and policy demonstrations
tests/                     Numerical and rollout correctness checks
docs/                      Protocol, results and source attribution
```

Each completed run saves `agent.pt`, `flags.json`, `log.csv`, `log.pkl`, and
`provenance.json`. Plot x-axes count actual training transitions, excluding
evaluation. See [source attribution and protocol](docs/PROVENANCE.md) for estimator,
evaluation, time-limit, and dependency details.

## Optional integrations

For W&B, run `uv sync --extra tracking`, then pass `--wandb_mode online` (or
`offline`) to training. The default is local-only logging.

For Discord, copy `.env.example` to `.env`, set `DISCORD_WEBHOOK_URL`, and use:

```bash
uv run python -m infrastructure.notify "Experiment milestone" --attach media/cartpole_small.png
```

Credentials stay in the ignored `.env` file. Training artifacts and exported
media contain no webhook credentials.

## Attribution

Based on [Berkeley Deep RL's Spring 2026 HW2 starter](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw2).
This repository documents its additions and experimental assumptions in
[PROVENANCE.md](docs/PROVENANCE.md).
