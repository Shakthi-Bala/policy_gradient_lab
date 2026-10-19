# Experimental results

Generated from completed run CSVs by `uv run python scripts/analyze.py`.

All primary ablations use **seed 1**. Peak return is the best single evaluation batch; final and trailing-ten means are also reported to expose instability. Evaluation samples are stochastic and limited; these are not confidence intervals or multi-seed guarantees.

| Experiment | Seed | Peak eval | Final eval | Last 10 mean | Training steps | First target steps |
|---|---:|---:|---:|---:|---:|---:|
| [cartpole_lb_na](../exp/CartPole-v0_cartpole_lb_na_sd1_20261019_165747/log.csv) | 1 | 200.0 | 200.0 | 200.0 | 401,861 | 44,274 |
| [cartpole_lb_rtg_na](../exp/CartPole-v0_cartpole_lb_rtg_na_sd1_20261019_170021/log.csv) | 1 | 200.0 | 200.0 | 200.0 | 403,417 | 69,020 |
| [cartpole_lb_rtg](../exp/CartPole-v0_cartpole_lb_rtg_sd1_20261019_165513/log.csv) | 1 | 200.0 | 200.0 | 200.0 | 403,846 | 73,115 |
| [cartpole_lb](../exp/CartPole-v0_cartpole_lb_sd1_20261019_165237/log.csv) | 1 | 200.0 | 200.0 | 165.0 | 406,820 | 77,340 |
| [cartpole_na](../exp/CartPole-v0_cartpole_na_sd1_20261019_165027/log.csv) | 1 | 200.0 | 200.0 | 200.0 | 102,080 | 21,151 |
| [cartpole_rtg_na](../exp/CartPole-v0_cartpole_rtg_na_sd1_20261019_165125/log.csv) | 1 | 200.0 | 200.0 | 165.5 | 105,992 | 18,004 |
| [cartpole_rtg](../exp/CartPole-v0_cartpole_rtg_sd1_20261019_164922/log.csv) | 1 | 200.0 | 200.0 | 186.7 | 105,319 | 41,159 |
| [cartpole](../exp/CartPole-v0_cartpole_sd1_20261019_164817/log.csv) | 1 | 200.0 | 122.0 | 126.3 | 104,024 | 20,711 |
| [cheetah_baseline_bgs1](../exp/HalfCheetah-v4_cheetah_baseline_bgs1_sd1_20261019_170144/log.csv) | 1 | -54.8 | -54.8 | -154.3 | 500,000 | — |
| [cheetah_baseline_na](../exp/HalfCheetah-v4_cheetah_baseline_na_sd1_20261019_172158/log.csv) | 1 | 610.4 | 370.9 | 503.3 | 500,000 | 385,000 |
| [cheetah_baseline](../exp/HalfCheetah-v4_cheetah_baseline_sd1_20261019_170143/log.csv) | 1 | 389.2 | 104.9 | 127.0 | 500,000 | 435,000 |
| [cheetah_baseline](../exp/HalfCheetah-v4_cheetah_baseline_sd2_20261019_171650/log.csv) | 2 | 263.8 | 208.1 | 208.2 | 500,000 | — |
| [cheetah_baseline](../exp/HalfCheetah-v4_cheetah_baseline_sd3_20261019_171650/log.csv) | 3 | 99.0 | -60.6 | -11.5 | 500,000 | — |
| [cheetah_baseline_torch](../exp/HalfCheetah-v4_cheetah_baseline_torch_sd1_20261019_173442/log.csv) | 1 | 386.6 | 374.2 | 324.3 | 500,000 | 475,000 |
| [cheetah](../exp/HalfCheetah-v4_cheetah_sd1_20261019_170145/log.csv) | 1 | -204.9 | -231.2 | -235.2 | 500,000 | — |
| [pendulum](../exp/InvertedPendulum-v4_pendulum_sd1_20261019_171447/log.csv) | 1 | 1000.0 | 6.3 | 71.7 | 509,717 | 431,837 |
| [pendulum_tuned_fast](../exp/InvertedPendulum-v4_pendulum_tuned_fast_sd1_20261019_171751/log.csv) | 1 | 1000.0 | 142.9 | 138.9 | 108,688 | 26,939 |
| [pendulum_tuned](../exp/InvertedPendulum-v4_pendulum_tuned_sd1_20261019_171455/log.csv) | 1 | 1000.0 | 222.4 | 372.7 | 106,469 | 40,028 |
| [pendulum_tuned_small](../exp/InvertedPendulum-v4_pendulum_tuned_small_sd1_20261019_171505/log.csv) | 1 | 1000.0 | 1000.0 | 932.3 | 73,468 | 36,372 |
| [lunar_lander_lambda0.95](../exp/LunarLander-v2_lunar_lander_lambda0.95_sd1_20261019_171212/log.csv) | 1 | 93.9 | -19.9 | -40.3 | 421,288 | — |
| [lunar_lander_lambda0.98](../exp/LunarLander-v2_lunar_lander_lambda0.98_sd1_20261019_171212/log.csv) | 1 | 169.2 | 68.4 | 50.4 | 435,109 | 268,736 |
| [lunar_lander_lambda0.99](../exp/LunarLander-v2_lunar_lander_lambda0.99_sd1_20261019_171212/log.csv) | 1 | 236.4 | 227.6 | 99.2 | 448,355 | 241,257 |
| [lunar_lander_lambda0](../exp/LunarLander-v2_lunar_lander_lambda0_sd1_20261019_171212/log.csv) | 1 | 62.5 | -97.3 | -64.0 | 437,102 | — |
| [lunar_lander_lambda1](../exp/LunarLander-v2_lunar_lander_lambda1_sd1_20261019_171212/log.csv) | 1 | 180.2 | 51.6 | 101.5 | 446,908 | 315,719 |

Targets: CartPole 200; HalfCheetah 300; LunarLander 150; InvertedPendulum 1,000.

## 1. CartPole: returns, normalization and batch size

**Small batch (1,000).** Last-ten mean evaluation returns: trajectory 126.3, reward-to-go 186.7, normalized trajectory 200.0, normalized reward-to-go 165.5. Without normalization, reward-to-go performed better by this endpoint metric.

Normalization changed the trajectory last-ten evaluation mean by +73.7 points.

Normalization changed the reward-to-go last-ten evaluation mean by -21.2 points.

**Large batch (4,000).** Last-ten mean evaluation returns: trajectory 165.0, reward-to-go 200.0, normalized trajectory 200.0, normalized reward-to-go 200.0. Without normalization, reward-to-go performed better by this endpoint metric.

Normalization changed the trajectory last-ten evaluation mean by +35.0 points.

Normalization changed the reward-to-go last-ten evaluation mean by +0.0 points.

With normalized reward-to-go, increasing the batch changed the last-ten mean from 165.5 to 200.0. The first 200-return evaluations occurred at 18004 and 69020 training steps respectively (None means never reached).

Reward-to-go is generally preferred because an action cannot influence past rewards. Removing those past terms preserves the expected score-function gradient while usually reducing its variance. Relative discounting weights future rewards from each decision.

Normalization rescales the update and subtracts a batch mean; it is a useful heuristic but introduces finite-batch bias. Compare normalized and unnormalized rows above rather than assuming it always helps. Batch size trades gradient noise against interaction cost: 100 iterations at 4,000 steps use roughly four times the data of 1,000-step batches. The charts compare actual training steps, not iteration counts. CartPole plots show training-batch average return, as in the assignment; the comparison table uses evaluation return.

![Small-batch CartPole](../media/cartpole_small.png)

![Large-batch CartPole](../media/cartpole_large.png)

## 2. HalfCheetah: learned baseline

**cheetah:** peak -204.9; final -231.2.

**cheetah_baseline:** peak 389.2; final 104.9. Baseline MSE (mean over optimizer steps on each training batch) changed from 178.8 to 22.2.

**cheetah_baseline_bgs1:** peak -54.8; final -54.8. Baseline MSE (mean over optimizer steps on each training batch) changed from 193.6 to 30.3.

Adding a baseline changed the last-ten evaluation mean from -235.2 to 127.0. Reducing its updates to one changed that mean to -154.3; final training-batch baseline MSE was 22.2 with five updates and 30.3 with one.

### Baseline seed repeats

The seed-1 baseline reached the peak target but missed the final-return target. Two additional seeds use exactly the same learning hyperparameters. All attempts are retained.

Seed 1: peak 389.2, final 104.9, last-ten mean 127.0.

Seed 2: peak 263.8, final 208.1, last-ten mean 208.2.

Seed 3: peak 99.0, final -60.6, last-ten mean -11.5.

### Optional advantage normalization

The optional normalized-baseline run reached peak 610.4, final 370.9, and last-ten mean 503.3. It adds only advantage normalization to the seed-1 baseline configuration. This additional setting is reported separately from the prescribed unnormalized ablation.

### Original-sampler reference

A further seed-1 baseline trial uses the original PyTorch action sampler, with all learning hyperparameters unchanged: peak 386.6, final 374.2, last-ten mean 324.3. The NumPy and PyTorch backends have equivalent policy distributions but different seeded action streams; this diagnostic checks whether low final scores persist with the reference sampler. It is labeled separately from the three NumPy seed trials.

**Unnormalized final-return target:** met in the recorded trials; best final evaluation 374.2 (cheetah_baseline_torch, seed 1). The normalized optional result is kept separate from this check.

The one-gradient-step ablation keeps the baseline learning rate at 0.01 and reduces value updates from five to one. This slows fitting to the current batch; policy outcomes above show whether that change helped this seed. Value targets change with the policy, so training MSE need not decrease monotonically, and lower training MSE alone does not prove better control. The baseline uses raw targets.

![HalfCheetah return](../media/cheetah_returns.png)

![Baseline training loss](../media/cheetah_baseline_loss.png)

## 3. LunarLander: generalized advantage estimation

Best peak: **lunar_lander_lambda0.99**, 236.4. Best trailing-ten mean: **lunar_lander_lambda1**, 101.5.

At λ=0 the final evaluation was -97.3; at λ=1 it was 51.6. These endpoint measurements and the intermediate settings above describe how the chosen horizon affected this particular training seed. The comparison is descriptive; each setting needs more training seeds for a reliable ranking.

λ=0 uses one-step TD residuals, reducing variance while depending heavily on value accuracy. λ=1 telescopes to Monte Carlo reward-to-go minus the baseline under the zero-tail boundary convention, reducing bootstrap bias while increasing variance. Intermediate λ mixes horizons. The observed ranking above and curves below reflect this seed, not a universal monotonic relationship. All five runs retain the PDF’s fixed batch size, learning rate, architecture, discount, and iteration budget.

The official starter uses standard LunarLander-v2 without an additional action-noise wrapper; this protocol is preserved. Headless runs explicitly prune visual-only exhaust particles after each original environment step. Observations, rewards and termination were bitwise identical in 5,000 comparison steps; learning hyperparameters are unchanged. See [PROVENANCE.md](PROVENANCE.md) and the recorded cleanup flag.

![GAE comparison](../media/lunar_gae.png)

## 4. InvertedPendulum: sample efficiency

Earliest observed 1,000-return evaluation: **pendulum_tuned_fast**, at **26,939 training steps**. The below-100K target was met.

```json
{
  "batch_size": 1000,
  "discount": 0.99,
  "learning_rate": 0.02,
  "n_layers": 2,
  "layer_size": 64,
  "use_reward_to_go": true,
  "normalize_advantages": true,
  "use_baseline": true,
  "baseline_gradient_steps": 10,
  "gae_lambda": 0.95
}
```

The earliest-crossing run ended at 142.9. The highest trailing-ten mean belonged to pendulum_tuned_small (932.3), whose final evaluation was 1000.0. First threshold crossing and sustained performance are different criteria.

The matched 1,000-step-batch candidates differ only in actor learning rate: 0.01 first reached 1,000 at 40028 steps and ended at 222.4; 0.02 first reached it at 26939 steps and ended at 142.9. The larger rate accelerated the initial crossing in this seed, while both candidates later lost performance.

Tuning reduces batches from 5,000 to 500 or 1,000 transitions to allow more frequent policy updates per interaction budget, and combines reward-to-go, normalization, a value baseline, γ=0.99 and GAE. The candidates also vary learning rate and λ. These coupled changes are a configuration search, not a controlled causal attribution of gains to any one hyperparameter. All attempts are retained.

![Pendulum sample efficiency](../media/pendulum_efficiency.png)

## Reproduction and limitations

Exact command lines are in [commands.sh](commands.sh); per-run flags and source revision are beside each CSV. The eight CartPole, three HalfCheetah, five LunarLander, and four InvertedPendulum configurations, plus two additional HalfCheetah baseline seeds the optional normalized-baseline run, and a reference PyTorch-sampler baseline, are generated by `scripts/experiments.py`.

Single-seed comparisons and peak selection can be optimistic. A larger study should repeat every setting over independent seeds and evaluate saved policies over many held-out episodes. Training steps exclude evaluation interactions as required by the assignment. Time-limit tails are zero, not bootstrapped. Videos show individual rollouts of final checkpoints and do not establish average performance.

## Fresh-episode evaluation of selected final checkpoints

Configurations are selected from the benchmark comparisons. Each frozen final policy is then evaluated stochastically on 30 new, individually seeded episodes. These bootstrap intervals measure episode variation for one trained policy, not training-seed uncertainty.

| Selected policy | Episodes | Mean return | Standard deviation | Bootstrap 95% CI |
|---|---:|---:|---:|---| 
| [cartpole_na](evaluation/cartpole.json) | 30 | 200.0 | 0.0 | [200.0, 200.0] |
| [cheetah_baseline_na](evaluation/cheetah.json) | 30 | 354.5 | 71.8 | [329.2, 379.9] |
| [cheetah_baseline_torch](evaluation/cheetah_reference.json) | 30 | 409.0 | 55.9 | [388.4, 428.3] |
| [cheetah_baseline](evaluation/cheetah_seed1.json) | 30 | 96.1 | 111.4 | [52.3, 131.6] |
| [lunar_lander_lambda0.99](evaluation/lunar.json) | 30 | 120.0 | 143.7 | [66.3, 167.4] |
| [pendulum_tuned_small](evaluation/pendulum.json) | 30 | 949.8 | 199.2 | [871.1, 1000.0] |

## Demonstration selection

GIFs are short previews; MP4s retain full episodes. The LunarLander showcase is an explicitly selected successful example, not a representative average. The selection log records every candidate tried, and the original seed-42 failure video is retained. The fresh-episode table above includes all 30 benchmark outcomes independently of demo selection.
