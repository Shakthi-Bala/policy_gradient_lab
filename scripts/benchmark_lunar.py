"""Compare exact LunarLander dynamics and CPU cost with visual-particle cleanup."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from gym.envs.box2d.lunar_lander import heuristic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from infrastructure.envs import make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/lunar_runtime.json")
    args = parser.parse_args()
    reference = make_env("LunarLander-v2", disable_env_checker=True)
    optimized = make_env(
        "LunarLander-v2", prune_lunar_particles=True, disable_env_checker=True
    )
    obs = reference.reset(seed=args.seed)
    np.testing.assert_array_equal(obs, optimized.reset(seed=args.seed))
    costs, peaks = [0.0, 0.0], [0, 0]
    try:
        for step in range(args.steps):
            # A deterministic heuristic targets a hover rather than landing, to
            # expose the legacy headless renderer's long-episode particle leak.
            target_obs = obs.copy()
            target_obs[1] -= 0.8
            action = heuristic(reference.unwrapped, target_obs)
            start = time.process_time()
            expected = reference.step(action)
            costs[0] += time.process_time() - start
            start = time.process_time()
            actual = optimized.step(action)
            costs[1] += time.process_time() - start
            np.testing.assert_array_equal(expected[0], actual[0])
            assert expected[1:3] == actual[1:3]
            peaks[0] = max(peaks[0], len(reference.unwrapped.particles))
            peaks[1] = max(peaks[1], len(optimized.unwrapped.particles))
            obs = expected[0]
            if expected[2]:
                obs = reference.reset(seed=args.seed + step + 1)
                np.testing.assert_array_equal(
                    obs, optimized.reset(seed=args.seed + step + 1)
                )
    finally:
        reference.close()
        optimized.close()
    data = {
        "environment": "LunarLander-v2",
        "gym": "0.25.2",
        "seed": args.seed,
        "steps_compared": args.steps,
        "observations_rewards_terminals_bitwise_identical": True,
        "reference_step_cpu_seconds": costs[0],
        "cleanup_step_cpu_seconds": costs[1],
        "step_cpu_speedup": costs[0] / costs[1],
        "peak_particles": peaks,
        "policy": "deterministic course heuristic with normalized hover height offset 0.8",
        "note": "Environment-step microbenchmark only; excludes neural inference and optimization.",
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
