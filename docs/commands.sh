# Generated exact benchmark commands; run from the repository root.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 1 --exp_name cheetah_baseline_bgs1 --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0 --prune_lunar_particles --exp_name lunar_lander_lambda0 --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.95 --prune_lunar_particles --exp_name lunar_lander_lambda0.95 --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.98 --prune_lunar_particles --exp_name lunar_lander_lambda0.98 --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.99 --prune_lunar_particles --exp_name lunar_lander_lambda0.99 --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 1 --prune_lunar_particles --exp_name lunar_lander_lambda1 --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 -rtg -na --discount 0.99 -lr 0.01 --use_baseline --gae_lambda 0.95 -bgs 10 --exp_name pendulum_tuned --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 500 -eb 1000 -rtg -na --discount 0.99 -lr 0.005 --use_baseline --gae_lambda 0.98 -bgs 10 --exp_name pendulum_tuned_small --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 1000 -eb 1000 -rtg -na --discount 0.99 -lr 0.02 --use_baseline --gae_lambda 0.95 -bgs 10 --exp_name pendulum_tuned_fast --no_gpu --rollout_backend numpy
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline --no_gpu --rollout_backend numpy --seed 2
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline --no_gpu --rollout_backend numpy --seed 3
