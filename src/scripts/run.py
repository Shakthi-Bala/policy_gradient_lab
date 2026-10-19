"""Train policy gradients using the command-line interface from CS285 HW2."""

import argparse
import os
import sys
import time
from datetime import datetime

import gym
import numpy as np
import torch

from agents.pg_agent import PGAgent
from infrastructure import pytorch_util as ptu
from infrastructure import utils
from infrastructure.envs import make_env
from infrastructure.log_utils import Logger, dump_log, setup_wandb, write_provenance

MAX_NVIDEO = 2


def run_training_loop(logger, args):
    # set random seeds
    torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = make_env(
        args.env_name,
        prune_lunar_particles=args.prune_lunar_particles,
        render_mode=None,
        disable_env_checker=True,
    )
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    # Evaluation uses a separate RNG stream and never changes training observations.
    eval_env = make_env(
        args.env_name,
        prune_lunar_particles=args.prune_lunar_particles,
        render_mode=None,
        disable_env_checker=True,
    )
    eval_env.reset(seed=args.seed + 10000)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    max_ep_len = args.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.dt
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    agent.actor.rollout_backend = args.rollout_backend

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(
            env, agent.actor, args.batch_size, max_ep_len
        )
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        train_info = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminal"],
        )

        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, _eval_envsteps_this_batch = utils.sample_trajectories(
                eval_env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            logs["Iteration"] = itr

            # perform the logging
            for key, value in logs.items():
                print(f"{key} : {value}")
            logger.log(logs, itr)
            print("Done logging...\n\n", flush=True)

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                eval_env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

        if (itr + 1) % args.checkpoint_freq == 0:
            dump_log(agent, logger, args, args.save_dir)

    dump_log(agent, logger, args, args.save_dir)
    logger.close()
    env.close()
    eval_env.close()


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v0")
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    parser.add_argument(
        "--wandb_mode", choices=["disabled", "offline", "online"], default="disabled"
    )
    parser.add_argument(
        "--rollout_backend", choices=["torch", "numpy"], default="torch"
    )
    parser.add_argument("--prune_lunar_particles", action="store_true")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    args = parser.parse_args(args=args)
    for field in (
        "n_iter",
        "batch_size",
        "eval_batch_size",
        "scalar_log_freq",
        "checkpoint_freq",
        "torch_threads",
    ):
        if getattr(args, field) < 1:
            parser.error(f"{field} must be positive")

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    exp_name = f"{args.env_name}_{args.exp_name}_sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config = vars(args)
    setup_wandb(
        project="policy-gradient-lab",
        name=exp_name,
        config=config,
        mode=args.wandb_mode,
    )
    args.save_dir = os.path.join(logdir_prefix, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    write_provenance(args.save_dir, ["uv", "run", "src/scripts/run.py", *sys.argv[1:]])
    logger = Logger(os.path.join(args.save_dir, "log.csv"))

    run_training_loop(logger, args)


if __name__ == "__main__":
    args = setup_arguments()
    main(args)
