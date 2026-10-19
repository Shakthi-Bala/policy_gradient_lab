"""Export a seeded final-checkpoint rollout as a captioned GIF and MP4."""

import argparse
import json
import os
import sys
from pathlib import Path

# Headless defaults must be set before importing simulator libraries.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agents.pg_agent import PGAgent
from infrastructure.notify import send_update


def load_agent(directory):
    flags = json.loads((directory / "flags.json").read_text())
    env = gym.make(flags["env_name"], disable_env_checker=True)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    agent = PGAgent(
        ob_dim=env.observation_space.shape[0],
        ac_dim=env.action_space.n if discrete else env.action_space.shape[0],
        discrete=discrete,
        n_layers=flags["n_layers"],
        layer_size=flags["layer_size"],
        gamma=flags["discount"],
        learning_rate=flags["learning_rate"],
        use_baseline=flags["use_baseline"],
        use_reward_to_go=flags["use_reward_to_go"],
        baseline_learning_rate=flags["baseline_learning_rate"],
        baseline_gradient_steps=flags["baseline_gradient_steps"],
        gae_lambda=flags["gae_lambda"],
        normalize_advantages=flags["normalize_advantages"],
    )
    agent.load_state_dict(
        torch.load(directory / "agent.pt", map_location="cpu", weights_only=True)
    )
    agent.actor.rollout_backend = flags.get("rollout_backend", "torch")
    agent.eval()
    return agent, env, flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="Output path without extension"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    agent, env, flags = load_agent(args.run)
    result = env.reset(seed=args.seed)
    obs = result[0] if isinstance(result, tuple) else result
    fps = int(env.metadata.get("render_fps", 30))
    cap = min(args.max_steps, flags["ep_len"] or env.spec.max_episode_steps)
    frames = []
    total_return = 0.0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for step in range(cap):
        frame = env.render(mode="rgb_array")
        action = agent.actor.get_action(obs)
        action = int(action) if agent.actor.discrete else action
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, _ = result
            done = terminated or truncated
        else:
            obs, reward, done, _ = result
        total_return += reward
        image = Image.fromarray(frame).convert("RGB")
        image.thumbnail((640, 420))
        canvas = Image.new("RGB", (640, 480), "#0f172a")
        canvas.paste(image, ((640 - image.width) // 2, 52 + (420 - image.height) // 2))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (14, 8),
            f"POLICY GRADIENT LAB  |  {flags['env_name']}",
            font=font,
            fill="#f8fafc",
        )
        draw.text(
            (14, 29),
            f"Final checkpoint · stochastic rollout · seed {args.seed} · step {step + 1} · return {total_return:.1f}",
            font=font,
            fill="#a5b4fc",
        )
        frames.append(np.asarray(canvas))
        if done:
            break
    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mp4 = args.output.with_suffix(".mp4")
    gif = args.output.with_suffix(".gif")
    imageio.mimsave(mp4, frames, fps=fps, macro_block_size=1)
    # GIF downsampling preserves playback speed and keeps Git/Discord artifacts small.
    stride = max(1, round(fps / 15))
    gif_frames = [
        Image.fromarray(frame).resize((480, 360)).quantize(colors=96)
        for frame in frames[::stride]
    ]
    gif_frames[0].save(
        gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1000 * stride / fps,
        loop=0,
        optimize=True,
    )
    metadata = {
        "run": str(args.run),
        "seed": args.seed,
        "steps": len(frames),
        "return": total_return,
        "checkpoint": "final",
        "stochastic": True,
        "fps": fps,
        "command": ["uv", "run", "python", "scripts/render_policy.py", *sys.argv[1:]],
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    message = (
        f"{flags['env_name']} policy demo: final checkpoint ({flags['exp_name']}), "
        f"seed {args.seed}, stochastic rollout, {len(frames)} steps, return {total_return:.1f}. "
        "This is one rollout, not the average evaluation score."
    )
    print(message)
    if args.notify:
        send_update(message, [mp4, gif])


if __name__ == "__main__":
    main()
