# Policy gradients: implementation notes

The policy is an MLP with tanh hidden activations. Discrete control uses categorical
logits; continuous control uses independent normal coordinates with a learned,
state-independent log standard deviation. `torch.distributions.Independent`
sums continuous-action log probabilities across coordinates so there is exactly
one score-function term per transition.

For each update, collect complete trajectories using the current frozen policy,
then compute one of two targets (discount γ):

- **Trajectory return:** every action in an episode receives the same target
  `Q[t] = r[0] + γ r[1] + … + γ^(T−1) r[T−1]`.
- **Reward-to-go:** `Q[t] = r[t] + γ Q[t+1]`, with a zero tail.

Without a value baseline, `A[t] = Q[t]`. With a baseline and no GAE,
`A[t] = Q[t] − V(s[t])`. A separate MLP learns `V` with mean squared error against
raw Monte Carlo targets, using the configured number of Adam steps per batch.

With GAE, work backward over the flattened batch:

```text
continue[t] = 1 − rollout_boundary[t]
δ[t] = r[t] + γ continue[t] V(s[t+1]) − V(s[t])
A[t] = δ[t] + γ λ continue[t] A[t+1]
```

Every trajectory boundary masks **both** the next value and next advantage;
otherwise concatenating episodes silently leaks values between unrelated states.
At λ=0 this is one-step TD. At λ=1 the sum telescopes to reward-to-go minus the
baseline, under the assignment's zero-tail convention. Time limits also terminate
the finite rollout; we intentionally do not bootstrap a continuing-task value.

If requested, normalize advantages over the entire sampled batch using
`(A − mean(A)) / (std(A) + 1e−8)`. Constant batches then produce zero, finite
advantages. The actor minimizes

```text
L_actor = −mean(log π(action | observation) × stop_gradient(advantage))
```

A single Adam update applies the actor gradient. Values used in advantages are
computed before updating the critic. Returns and advantages are NumPy values,
so actor backpropagation cannot inadvertently optimize the value baseline.

The algorithm averages over transitions, following the official starter's
vectorized convention. It is on-policy: after one actor update, the collected
batch is discarded and the updated policy gathers new experience.

## Numerical tests

`tests/test_policy_gradient.py` verifies hand-computed discounted targets,
independence across episodes, GAE endpoints and masks, constant-batch
normalization, categorical/Gaussian event shapes, the direction of a
positive-advantage policy update, value regression, and complete-episode sampling
budgets. Additional tests compare CPU inference outputs with PyTorch, exercise
cache invalidation after optimization, and compare empirical categorical action
frequencies with their distribution probabilities.

The optional NumPy rollout backend is inference only. It reuses the same trained
weights and distributions, while PyTorch remains responsible for gradients and
Adam updates. See [PROVENANCE.md](PROVENANCE.md) for seeded-backend differences.
