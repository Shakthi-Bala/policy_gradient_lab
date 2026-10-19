# Source and experimental protocol

This project extends Berkeley CS185/285 Spring 2026 Homework 2, supplied as
[`hw2.pdf`](../hw2.pdf). Starter source:
https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw2

Imported revision: `59e40fc035374fd9b29195a4c3bef02690b515b5`.
The course's agent, network, infrastructure, and training entry-point paths are
preserved. The implementations of policy distributions, score-function updates,
return estimators, value regression, GAE, and rollout collection were completed
in this repository. Local logging, tests, experiment automation, analysis,
media export, notifications, and documentation extend the starter.

## Protocol

- CPU PyTorch, one numerical thread per training process, seed 1 unless stated.
- Each batch contains complete episodes until at least the requested number of
  training transitions is reached. Actual counts can exceed the nominal batch.
- Learning curves use `Train_EnvstepsSoFar`, excluding evaluation transitions,
  matching the assignment. This is not the total simulator interaction cost.
- Training and evaluation use separate environments with seeds `seed` and
  `seed + 10000`. Policies are stochastic during evaluation, as in the starter.
- Discounted trajectory returns are constant within each episode; reward-to-go
  discounts relative to each timestep. The actor averages loss over transitions.
- Continuous actions use a diagonal Gaussian. The logged probability is for the
  raw sampled action; MuJoCo applies actuator control limits in the simulator.
- The baseline regresses raw Monte Carlo targets with MSE. Advantages are
  calculated before critic updates. No target normalization is hidden in the code.
- GAE follows equation 22 and resets at every episode boundary, including time
  limits. The tail value is zero, per the assignment's finite-horizon convention;
  this is not continuing-task time-limit bootstrapping.
- The PDF calls LunarLander a version with noisy actions, but the official starter
  directly instantiates standard `LunarLander-v2` without a noise wrapper. This
  repository follows that executable protocol and does not invent a noise model.
- Gym 0.25.2 and the assignment's environment IDs are retained, with NumPy <2.
  MuJoCo 2.3.x is installed explicitly because Gym's optional MuJoCo extra pins an
  older runtime; the exact installed versions are recorded in `uv.lock`.
- Run flags, source revision, dirty status, library versions, CSV metrics,
  checkpoints and pickled logs are recorded per run. Evaluation metrics are
  collected after an actor update; training metrics describe its preceding batch.
- Single-seed ablations satisfy the assignment's minimum but do not establish
  statistical significance. Best-run peaks and final returns are reported
  separately. Tuning attempts remain visible rather than being discarded.

No empirical threshold is considered achieved until verified from a completed
run's CSV. Generated charts and results are derived from those CSVs.

## CPU rollout inference

The reproducible experiment matrix explicitly selects `--rollout_backend numpy`.
For each single-observation action query, this backend evaluates the same frozen
float32 tanh MLP weights with NumPy, then samples the categorical or diagonal
Gaussian distribution with NumPy's seeded RNG. Parameter-version counters refresh
cached weights after every optimizer update or checkpoint load. Tests compare
network outputs and distribution parameters to PyTorch and verify categorical
sample frequencies. All gradient calculations and optimization still use PyTorch.
The default CLI backend remains `torch`; different backends have different random
streams, so a seed does not imply identical trajectories across backends.

A short initial PyTorch-backend throughput pilot was stopped before completion
when the host's one-core CPU quota became apparent. Its partial logs are retained
locally under ignored `artifacts/pilot_torch/`; those partial runs are excluded
from experimental comparisons. All reported runs use the explicitly recorded
NumPy backend with `OPENBLAS_NUM_THREADS=1` (set by the experiment runner).
