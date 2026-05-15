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
`PPO`.  It can build either the BDP candidate-action policy or an ordinary SB3
policy through `policy_mode`.

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

Each training run also writes a resolved SB3 runner config:

```text
logs/sb3_bdp_cartpole_ppo/YYYYmmdd_HHMMSS/config.yaml
```

If you passed `--config_file`, that exact config file is copied to
`config.yaml` in the run folder.  If you did not pass `--config_file`, the
runner writes a generated `config.yaml` from the active CLI arguments.  The run
folder also contains `resolved_config.yaml`, which records the final effective
Gym/SB3 environment, model architecture, algorithm, and hyper-parameters after
CLI overrides.

The SB3 config does not include the old CARLA YAML contents.  It only stores
the legacy CARLA config path under `legacy_carla.cfg_file`, and that path is
ignored when `env_source` is `gymnasium` or `gym`.

## BDP Mode vs Built-In SB3 Mode

The default mode is the migrated BDP candidate-action algorithm:

```yaml
model_architecture:
  policy_mode: bdp
  policy_class: BDPBoltzmannPolicy
  candidate_sampler: DiscreteOneHotExternalSampler
```

In this mode, Gym/Gymnasium observations are wrapped into the candidate-action
Dict observation:

```text
obs["obs"], obs["candidates"], obs["candidate_mask"]
```

The action is a candidate row index, and `BDPBoltzmannPolicy` scores
`(state, candidate)` pairs.

To run the original off-the-shelf SB3 algorithm instead, set:

```yaml
model_architecture:
  policy_mode: builtin
  builtin_policy: MlpPolicy
```

`policy_mode: builtin` uses the raw Gym/Gymnasium observation space and action
space directly.  No candidate table is created, no one-hot external sampler is
used, and SB3 receives the policy name exactly like a normal PPO/TRPO script.
Use `MlpPolicy` for vector observations such as CartPole and `CnnPolicy` for
image observations such as CarRacing.

`policy_layers`, `value_layers`, and `activation` are passed as SB3
`policy_kwargs` in both modes.  In BDP mode, `policy_layers` builds the
candidate goodness network and `value_layers` builds the critic.  In built-in
mode, they become SB3's standard actor/critic `net_arch`.

When `builtin_policy: CnnPolicy`, both modes also share the same configurable
CNN feature extractor:

```yaml
model_architecture:
  builtin_policy: CnnPolicy
  features_extractor: NatureCNN     # SB3 default
  features_dim:
  features_extractor_kwargs: {}
```

Use the lighter custom extractor in `feature_extractors.py` with:

```yaml
model_architecture:
  builtin_policy: CnnPolicy
  features_extractor: EfficientCNN
  features_dim: 128
```

The same switch is available from the command line:

```bash
python3 run_BDPL_sb3.py \
  --env_source=gymnasium \
  --env=CartPole-v1 \
  --sb3_algorithm=PPO \
  --policy_mode=builtin \
  --builtin_policy=MlpPolicy \
  --num_timesteps=20000 \
  --log_path=logs/sb3_builtin_cartpole_ppo
```

For discrete CarRacing with SB3's built-in CNN policy:

```bash
python3 run_BDPL_sb3.py \
  --env_source=gymnasium \
  --env=CarRacing-v3 \
  --gym_make_kwargs='{"continuous": false}' \
  --sb3_algorithm=PPO \
  --policy_mode=builtin \
  --builtin_policy=CnnPolicy \
  --num_timesteps=20000 \
  --log_path=logs/sb3_builtin_racing_ppo
```

## YAML Config Entrypoint

You can launch training from a YAML file:

```bash
python3 run_BDPL_sb3.py \
  --config_file=agents/reinforcement_learning/sb3_bdp/config.yaml
```

The YAML keys correspond to the same CLI options used by the runner.  Command
line arguments can still override values from the file:

```bash
python3 run_BDPL_sb3.py \
  --config_file=agents/reinforcement_learning/sb3_bdp/config_cartpole_ppo_demo.yaml \
  --num_timesteps=5000
```
```bash
python3 run_BDPL_sb3.py \
  --config_file=agents/reinforcement_learning/sb3_bdp/config_racing_ppo_demo.yaml \
  --num_timesteps=5000
```


For testing, if `--config_file` is omitted, the runner automatically loads the
run folder config:

```text
<log_path>/config.yaml
```

so this is enough, and test is always set n_env=1:

```bash
python3 run_BDPL_sb3.py \
  --test \
  --log_path=logs/sb3_bdp_racing_ppo_4env/20260514_163142\
  --num_test_episode=10 \
  --play_mode=1 \
  --test_last

```bash
python3 run_BDPL_sb3.py \
  --test \
  --log_path=logs/sb3_bdp_racing_ppo_4env/20260514_163142 \
  --num_test_episode=10 \
  --test_last
```

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

Some Gymnasium environments need extra arguments in `gym.make()`.  Put them
under `environment.gym_make_kwargs`.  For example, CarRacing is continuous by
default, but Gymnasium can expose a discrete 5-action version:

```yaml
environment:
  env_source: gymnasium
  env: CarRacing-v3
  gym_make_kwargs:
    continuous: false
```

This runner passes those key/value pairs into:

```python
gym.make(env, **gym_make_kwargs)
```

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

For more env like car racing:
'''
pip install swig
pip install "gymnasium[box2d]"
'''

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
note: SB3 num_timesteps=20000 means total env steps across all parallel envs, not 20000 per env.

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
  --log_path=logs/sb3_bdp_cartpole_ppo_4env/20260514_115621 \
  --num_test_episode=10
```

By default, testing loads the newest `best_*.zip` model.  Use `--test_last` to
prefer `step_*.zip`, or pass an explicit model with `--test_model`.

## Rendering

Gymnasium environments expose rendering through `render_mode` at environment
creation time.  `render_mode` says how the environment should render, while
`play_mode` and `render_train` decide whether rendering is enabled for test or
training.

For Gymnasium classic-control envs such as CartPole, `render_mode: human` can
draw automatically during `env.step()`.  The runner therefore only passes
`render_mode` into the training env when `render_train: true`, and only passes
it into the test/eval env when `play_mode: 1`.

For visual testing, use:

```yaml
render:
  play_mode: 1
  render_mode: human
  render_train: false
  render_freq: 1
```

Then run test with the config file:

```bash
python3 run_BDPL_sb3.py \
  --test \
  --config_file=agents/reinforcement_learning/sb3_bdp/config.yaml \
  --log_path=logs/sb3_bdp_cartpole_ppo_4env/20260514_115621 \
  --num_test_episode=3
```

For generic Gymnasium/Gym envs, `play_mode: 1` automatically fills
`render_mode: human` if `render_mode` is empty, but keeping it explicit in the
config is clearer.

For training render, keep one environment and enable the render callback in
`config.yaml`:

```yaml
algorithm:
  n_envs: 1
  vec_env: dummy

render:
  play_mode: 0
  render_mode: human
  render_train: true
  render_freq: 1
```

Then start training from the config:

```bash
python3 run_BDPL_sb3.py \
  --config_file=agents/reinforcement_learning/sb3_bdp/config.yaml
```
```bash
python3 run_BDPL_sb3.py \
  --config_file=agents/reinforcement_learning/sb3_bdp/config_racing_ppo_demo.yaml
```


Rendering during training is useful for debugging, but it slows training down.
Use non-rendered training for normal experiments.

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

`--features_extractor EfficientCNN`
: CNN state feature extractor used when `--builtin_policy=CnnPolicy`.
Current choices are `NatureCNN` and `EfficientCNN`, and the setting applies to
both `--policy_mode=builtin` and `--policy_mode=bdp`.

`--features_dim 128`
: Optional CNN feature output dimension.  Omit it to keep the selected
extractor's default.

`--save_freq 5000`
: Save `step_*.zip` checkpoints every N timesteps.  Best checkpoints are saved
from SB3 monitor episode rewards.

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
