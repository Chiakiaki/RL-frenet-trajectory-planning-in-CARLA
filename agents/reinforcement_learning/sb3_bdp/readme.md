# Stable-Baselines3 BDP/BDPL Candidate-Action Runner

This folder contains the standalone PyTorch/Stable-Baselines3 migration of the
old TensorFlow Stable Baselines BDPL/TRPO candidate-action idea.  The original
CARLA/TensorFlow code is not modified; `run_BDPL_sb3.py` remains the launch
entrypoint, and the implementation details live in this package.

## File Layout

`envs.py`
: Gymnasium adapters.  `BDPCandidateEnv` wraps the existing CARLA environment
with its planner `external_sampler()`.  `GenericDiscreteCandidateEnv` wraps
ordinary Gymnasium/Gym environments with `Discrete(n)` actions.

`samplers.py`
: External candidate samplers.  `DiscreteOneHotExternalSampler` enumerates each
discrete action and represents it as a one-hot candidate vector.

`policies.py`
: `BDPBoltzmannPolicy`, the PyTorch actor-critic policy.  It scores each
`(state, candidate)` pair with a goodness network and uses the scores as
categorical logits.

`model.py`
: SB3 algorithm factory.  It selects `TRPO` from `sb3-contrib` or built-in SB3
`PPO`.

`callbacks.py`
: Checkpoint saving callback.

`spaces.py`
: Small Gym/Gymnasium space conversion helpers.

## Log Directory Layout

For training, `--log_path` is treated as a parent experiment directory.  The
runner creates a timestamped child folder for each run:

```text
--log_path=logs/sb3_bdp_cartpole_ppo

actual run folder:
logs/sb3_bdp_cartpole_ppo/YYYYmmdd_HHMMSS/
```

This keeps repeated launches with the same `--log_path` separated instead of
overwriting each other.  The model files are saved below:

```text
logs/sb3_bdp_cartpole_ppo/YYYYmmdd_HHMMSS/models/
```

For testing, pass `--log_path` to the specific timestamped run folder you want
to load.

## Candidate Observation Format

SB3 expects fixed-shape observation spaces, but BDPL candidate sets can be
conceptually variable-length.  This implementation stores candidates in a dict
observation with padding and a mask:

```text
obs["obs"]            : raw environment observation
obs["candidates"]     : (K, Da)
obs["candidate_mask"] : (K,)
action                : candidate row index in [0, K)
```

Shape names:

```text
B  = SB3/PyTorch batch size
K  = max_candidates
N  = number of real candidates for one state, N <= K
Ds = flattened state dimension
Da = candidate feature dimension
```

For one environment step:

```text
candidates[0:N]      = real candidate action features
candidates[N:K]      = zero padding
candidate_mask[0:N]  = 1
candidate_mask[N:K]  = 0
```

Inside the policy after SB3 batches rollout samples:

```text
state          : (B, Ds)
candidates     : (B, K, Da)
candidate_mask : (B, K)
logits         : (B, K)
values         : (B, 1)
```

The old TensorFlow implementation flattened all `(state, action)` pairs into
one long tensor and needed `grouping_mn` to recover which candidates belonged
to the same state.  Here, every state keeps its own candidate table inside the
dict observation.  PyTorch then receives logits shaped `(B, K)`, so the
categorical distribution applies one softmax per batch row.  Masked padding
slots get a very negative logit and are not selected.

## Using Off-the-Shelf Gymnasium/Gym Environments

The generic path currently supports:

- `Discrete(n)` action spaces.
- `Box` observations, such as `CartPole-v1`.
- `Discrete(n)` observations, such as `FrozenLake-v1`, converted to one-hot
  state vectors.

Continuous-action environments such as `Pendulum-v1` are not supported by the
generic wrapper yet because there is no discrete candidate set to enumerate.

For a normal discrete-action environment, the external sampler creates one
candidate per action label:

```text
action labels:       0       1       2
candidate vectors:  [1,0,0] [0,1,0] [0,0,1]
```

The policy still learns a BDPL-style goodness function over `(state,
candidate)` pairs.  The only difference from CARLA is that the candidate
generator is a simple deterministic one-hot sampler instead of a Frenet
trajectory planner.

## Installation Notes

Use a PyTorch/SB3 environment, not the old TensorFlow 1.14 environment.

For PPO smoke tests:

```bash
pip install stable-baselines3 gymnasium[classic-control]
```

For TRPO:

```bash
pip install stable-baselines3 sb3-contrib gymnasium[classic-control]
```

Legacy Gym environments can also be used with `--env_source=gym` if `gym` is
installed.  The wrapper includes a small compatibility adapter for classic
Box/Discrete environments.

## Train on Gymnasium CartPole

Run from the repository root:

```bash
python3 run_BDPL_sb3.py \
  --env_source=gymnasium \
  --env=CartPole-v1 \
  --sb3_algorithm=PPO \
  --num_timesteps=20000 \
  --trpo_timesteps_per_batch=1024 \
  --batch_size=128 \
  --learning_rate=7e-4 \
  --log_path=logs/sb3_bdp_cartpole_ppo
```

TRPO uses the same candidate-action policy but the optimizer comes from
`sb3-contrib`:

```bash
python3 run_BDPL_sb3.py \
  --env_source=gymnasium \
  --env=CartPole-v1 \
  --sb3_algorithm=TRPO \
  --num_timesteps=20000 \
  --trpo_timesteps_per_batch=1024 \
  --batch_size=128 \
  --learning_rate=7e-4 \
  --target_kl=0.01 \
  --log_path=logs/sb3_bdp_cartpole_trpo
```

`--trpo_timesteps_per_batch` maps to SB3 `n_steps`.  For PPO it is still used
as the rollout length because both PPO and TRPO are on-policy algorithms.

## Train on Legacy Gym CartPole

If the environment is registered in old `gym` rather than Gymnasium:

```bash
python3 run_BDPL_sb3.py \
  --env_source=gym \
  --env=CartPole-v1 \
  --sb3_algorithm=PPO \
  --num_timesteps=20000 \
  --log_path=logs/sb3_bdp_legacy_cartpole_ppo
```

## Train on FrozenLake

FrozenLake has discrete observations.  The wrapper converts the scalar state id
to a one-hot vector before passing it to the policy:

```bash
python3 run_BDPL_sb3.py \
  --env_source=gymnasium \
  --env=FrozenLake-v1 \
  --sb3_algorithm=PPO \
  --num_timesteps=50000 \
  --trpo_timesteps_per_batch=2048 \
  --batch_size=128 \
  --log_path=logs/sb3_bdp_frozenlake_ppo
```

## Parallel Gymnasium Training

For generic Gym/Gymnasium environments, use `--n_envs` to create multiple
copies of the environment.  With SB3 on-policy algorithms, the rollout batch
size becomes:

```text
rollout batch size = n_steps * n_envs
```

In this runner, `n_steps` is set by `--trpo_timesteps_per_batch`.

Use `--vec_env=dummy` for multiple envs in one process, or `--vec_env=subproc`
for process-level parallelism:

```bash
python3 run_BDPL_sb3.py \
  --env_source=gymnasium \
  --env=CartPole-v1 \
  --sb3_algorithm=PPO \
  --num_timesteps=20000 \
  --trpo_timesteps_per_batch=256 \
  --n_envs=4 \
  --vec_env=subproc \
  --batch_size=128 \
  --learning_rate=7e-4 \
  --log_interval=1 \
  --log_path=logs/sb3_bdp_cartpole_ppo_4env
```

## Test a Saved Model

Use the same `--env_source`, `--env`, and `--sb3_algorithm` as training.  Point
`--log_path` at the specific timestamped training run directory:

```bash
python3 run_BDPL_sb3.py \
  --test \
  --env_source=gymnasium \
  --env=CartPole-v1 \
  --sb3_algorithm=PPO \
  --log_path=logs/sb3_bdp_cartpole_ppo/20260514_153000 \
  --num_test_episode=10
```

By default, testing loads the newest `best_*.zip` model.  Use `--test_last` to
prefer `step_*.zip`, or pass an explicit model with `--test_model`.

## Useful Arguments

`--max_candidates`
: Fixed candidate-table size `K`.  For off-the-shelf discrete envs this should
usually be omitted, so it defaults to the number of actions.

`--policy_layers 64 64`
: Hidden layers for the candidate goodness network.

`--value_layers 64 64`
: Hidden layers for the value network.

`--activation tanh`
: Activation used by both networks.  Current choices are `tanh` and `relu`.

`--save_freq 5000`
: Save periodic `step_*_steps.zip` checkpoints every N total timesteps with
SB3 `CheckpointCallback`.  Use `0` to disable periodic checkpoints.

`--eval_freq 10000`
: Evaluate every N total timesteps with SB3 `EvalCallback` and save a single
`best_model.zip` when the evaluation mean reward improves.  Use `0` to disable
best-model evaluation/saving.

`--n_eval_episodes 5`
: Number of episodes used by each `EvalCallback` evaluation.

`--device cpu`
: Force CPU.  The default `auto` lets SB3 choose.

## Current Limitations

- Generic off-the-shelf environments must have discrete actions.
- The generic one-hot sampler proposes all actions every step; it does not yet
  implement learned, filtered, or state-dependent candidate generation.
- CARLA training still needs the original CARLA environment dependencies and a
  config file.  The generic Gym/Gymnasium path does not need CARLA or a YAML
  config.
- `sb3-contrib` TRPO is not the old TensorFlow TRPO implementation; it is the
  SB3/PyTorch TRPO optimizer using this custom candidate-action policy.
