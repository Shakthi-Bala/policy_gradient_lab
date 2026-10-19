"""Evaluate one frozen final checkpoint on fresh episodes with uncertainty estimates."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from render_policy import load_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 2:
        parser.error("at least two episodes are required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent, env, flags = load_agent(args.run)
    cap = flags["ep_len"] or env.spec.max_episode_steps
    returns, lengths = [], []
    for episode in range(args.episodes):
        result = env.reset(seed=args.seed + episode)
        obs = result[0] if isinstance(result, tuple) else result
        total = 0.0
        for step in range(cap):
            action = agent.actor.get_action(obs)
            action = int(action) if agent.actor.discrete else action
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                obs, reward, done, _ = result
            total += reward
            if done:
                break
        returns.append(float(total))
        lengths.append(step + 1)
    env.close()
    rng = np.random.default_rng(args.seed)
    bootstrap_means = rng.choice(
        returns, size=(5000, args.episodes), replace=True
    ).mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    data = {
        "run": str(args.run),
        "environment": flags["env_name"],
        "experiment": flags["exp_name"],
        "training_seed": flags["seed"],
        "evaluation_seed": args.seed,
        "episodes": args.episodes,
        "checkpoint": "final",
        "stochastic": True,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=1)),
        "mean_return_bootstrap_95ci": [float(low), float(high)],
        "episode_returns": returns,
        "episode_lengths": lengths,
        "note": "Within-policy episode uncertainty; does not measure variability across training seeds.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"{flags['exp_name']}: {np.mean(returns):.1f} ± {np.std(returns, ddof=1):.1f}; "
        f"bootstrap 95% CI [{low:.1f}, {high:.1f}], {args.episodes} fresh episodes"
    )


if __name__ == "__main__":
    main()
