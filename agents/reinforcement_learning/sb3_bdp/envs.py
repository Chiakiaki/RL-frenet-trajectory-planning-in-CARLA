"""Environment adapters for the SB3 BDP candidate-action runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .samplers import DiscreteOneHotExternalSampler, ExternalCandidateSampler
from .spaces import as_gymnasium_box, convert_to_gymnasium_space


class LegacyGymCompatibilityEnv(gym.Env):
    """
    Minimal Gym -> Gymnasium adapter for Box/Discrete classic-control style envs.

    SB3 2.x is Gymnasium-first.  The official solution is ``shimmy``, but this
    wrapper is enough for the discrete-action BDP smoke-test path and avoids
    adding a dependency just to run CartPole-like legacy Gym tasks.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, legacy_env: Any):
        super().__init__()
        self.legacy_env = legacy_env
        self.render_mode = getattr(legacy_env, "render_mode", None)
        self.observation_space = convert_to_gymnasium_space(legacy_env.observation_space)
        self.action_space = convert_to_gymnasium_space(legacy_env.action_space)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        try:
            reset_result = self.legacy_env.reset(seed=seed, options=options)
        except TypeError:
            if seed is not None and hasattr(self.legacy_env, "seed"):
                self.legacy_env.seed(seed)
            reset_result = self.legacy_env.reset()

        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return reset_result[0], dict(reset_result[1] or {})
        return reset_result, {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        step_result = self.legacy_env.step(action)
        if len(step_result) == 5:
            raw_obs, reward, terminated, truncated, info = step_result
            return raw_obs, float(reward), bool(terminated), bool(truncated), dict(info or {})

        raw_obs, reward, done, info = step_result
        return raw_obs, float(reward), bool(done), False, dict(info or {})

    def render(self) -> Any:
        return self.legacy_env.render()

    def close(self) -> None:
        self.legacy_env.close()


class BDPCandidateEnv(gym.Env):
    """
    Gymnasium adapter for the old CARLA env.

    SB3 policies can only see the current observation, so candidate trajectories
    are included in a dict observation.  The action is a discrete candidate
    index; the wrapped legacy environment still consumes that index exactly as
    the original BDPL/TRPO runner did.

    Remarks about the "undefined number of candidates" challenge:

    OpenAI Gym/SB3 spaces must have a fixed shape, while ``external_sampler()``
    is conceptually variable-length.  The old TF code solved that by flattening
    candidates and passing a ``grouping_mn`` matrix.  Here we solve it in the
    more SB3-native way:

        candidates[0:n]      = real candidate trajectory features
        candidates[n:]       = zero padding
        candidate_mask[0:n]  = 1
        candidate_mask[n:]   = 0

    ``max_candidates`` is the fixed upper bound.  In this CARLA code path it is
    normally just ``env.num_traj`` (3, 9, 15, ...), but an explicit
    ``--max_candidates`` can be supplied if a future sampler can return fewer
    or more candidates under the same experiment.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, legacy_env: Any, max_candidates: Optional[int] = None):
        super().__init__()
        self.legacy_env = legacy_env
        self.render_mode = getattr(legacy_env, "render_mode", None)
        self.max_candidates = int(max_candidates or getattr(legacy_env, "num_traj", 0))
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be provided when legacy_env.num_traj is unavailable")

        if not hasattr(legacy_env, "external_sampler"):
            raise TypeError("BDPCandidateEnv requires a legacy env with external_sampler()")

        # raw_observation_space: original CARLA observation shape, e.g. lidar
        # (362,) or sequence state (TIME_STEP + 1, 9).  The policy later
        # flattens it to Ds.
        self.raw_observation_space = as_gymnasium_box(legacy_env.observation_space)
        legacy_action_space = legacy_env.action_space
        if not hasattr(legacy_action_space, "low") or not hasattr(legacy_action_space, "high"):
            raise TypeError("The candidate-action wrapper expects a Box-like legacy action space")

        action_low = np.asarray(legacy_action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(legacy_action_space.high, dtype=np.float32).reshape(-1)
        # candidate_dim Da: length of one trajectory-action feature vector.
        # For bdp mode this is usually T_ACTION_CANDIDATES / DT + 1.
        self.candidate_dim = int(action_low.shape[0])

        # observation_space is rectangular and SB3-friendly:
        #   obs:            raw_observation_space, later flattened to Ds
        #   candidates:     (K, Da)
        #   candidate_mask: (K,)
        # Even if the sampler returns N<K, the observation still has shape K.
        candidate_low = np.tile(action_low, (self.max_candidates, 1))
        candidate_high = np.tile(action_high, (self.max_candidates, 1))
        self.observation_space = spaces.Dict(
            {
                "obs": self.raw_observation_space,
                "candidates": spaces.Box(
                    low=candidate_low,
                    high=candidate_high,
                    shape=(self.max_candidates, self.candidate_dim),
                    dtype=np.float32,
                ),
                "candidate_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_candidates,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(self.max_candidates)

        self._last_num_candidates = 0
        self._closed = False

    def _format_observation(
        self, raw_obs: np.ndarray, candidates: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        # SB3 rollout buffers store rectangular arrays.  We therefore keep a
        # rectangular candidate tensor and use a mask to remember which rows are
        # real.  Padding rows are not actions the policy may choose; the policy
        # will turn their logits into a very negative number before softmax.
        # candidate_buffer: (K, Da), padded candidate feature matrix.
        # candidate_mask:   (K,), 1 for real rows and 0 for padded rows.
        candidate_buffer = np.zeros((self.max_candidates, self.candidate_dim), dtype=np.float32)
        candidate_mask = np.zeros((self.max_candidates,), dtype=np.float32)

        if candidates is not None:
            candidates = np.asarray(candidates, dtype=np.float32)
            # candidates from legacy external_sampler: (N, Da), where
            # N == len(self.bdpl_path_list) in the CARLA env.
            if candidates.ndim != 2:
                raise ValueError(f"external_sampler() must return a 2D candidate array, got {candidates.shape}")
            if candidates.shape[1] != self.candidate_dim:
                raise ValueError(
                    f"Candidate dim {candidates.shape[1]} does not match env action dim {self.candidate_dim}"
                )
            if candidates.shape[0] > self.max_candidates:
                raise ValueError(
                    f"Got {candidates.shape[0]} candidates but max_candidates={self.max_candidates}"
                )

            n_candidates = int(candidates.shape[0])
            # Copy the real rows into [0:N].  Rows [N:K] remain zero padding.
            candidate_buffer[:n_candidates] = candidates
            candidate_mask[:n_candidates] = 1.0
            self._last_num_candidates = n_candidates
        else:
            self._last_num_candidates = 0

        # Returned unbatched dict shapes:
        #   "obs":            raw observation shape, e.g. (362,)
        #   "candidates":     (K, Da)
        #   "candidate_mask": (K,)
        return {
            "obs": np.asarray(raw_obs, dtype=np.float32),
            "candidates": candidate_buffer,
            "candidate_mask": candidate_mask,
        }

    def _sample_candidates_for_obs(self, raw_obs: np.ndarray) -> Dict[str, np.ndarray]:
        # Same call order as the original runner:
        #   1. ask the planner for candidate trajectories;
        #   2. let the policy choose a candidate index;
        #   3. pass that index to legacy_env.step().
        # The only difference is that step 1 is now encoded in the observation
        # instead of being an extra argument to model.step().
        candidates, n_candidates = self.legacy_env.external_sampler()
        # candidates: (N, Da), n_candidates: scalar N
        candidates = np.asarray(candidates, dtype=np.float32)
        if int(n_candidates) != int(candidates.shape[0]):
            raise ValueError(
                f"external_sampler() reported {n_candidates} candidates but returned {candidates.shape[0]}"
            )
        return self._format_observation(raw_obs, candidates)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None and hasattr(self.legacy_env, "seed"):
            self.legacy_env.seed(seed)

        raw_reset = self.legacy_env.reset()
        raw_obs = raw_reset[0] if isinstance(raw_reset, tuple) else raw_reset
        return self._sample_candidates_for_obs(raw_obs), {}

    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action_idx = int(np.asarray(action).reshape(-1)[0])
        if self._last_num_candidates and action_idx >= self._last_num_candidates:
            raise ValueError(
                f"Invalid candidate index {action_idx}; only {self._last_num_candidates} candidates are valid"
            )

        raw_obs, reward, done, info = self.legacy_env.step(action_idx)
        terminated = bool(done)
        truncated = bool(info.pop("TimeLimit.truncated", False)) if isinstance(info, dict) else False
        info = dict(info or {})

        if terminated or truncated:
            obs = self._format_observation(raw_obs, candidates=None)
        else:
            obs = self._sample_candidates_for_obs(raw_obs)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Any:
        return self.legacy_env.render()

    def close(self) -> None:
        if not self._closed and hasattr(self.legacy_env, "destroy"):
            self.legacy_env.destroy()
        self._closed = True


class GenericDiscreteCandidateEnv(gym.Env):
    """
    Candidate-action wrapper for ordinary light-weight Gymnasium/Gym envs.

    This is the non-CARLA version of ``BDPCandidateEnv``.  It is meant for
    environments like CartPole-v1 or FrozenLake-v1 where the action space is
    already ``Discrete(n)``.  The wrapper creates candidate action features with
    an external sampler, stores them in a Dict observation, and translates the
    selected candidate index back to the raw environment action label.

    This lets the same ``BDPBoltzmannPolicy`` train on off-the-shelf envs:

        raw env action space:  Discrete(3)
        BDP action space:      Discrete(3), meaning "which candidate row"
        candidates:            one-hot rows for raw labels 0, 1, 2

    Shape example for CartPole-v1:

        raw_obs                : (4,)
        formatted obs["obs"]   : (4,)       -> Ds=4
        action_space           : Discrete(2)
        obs["candidates"]      : (2, 2)     -> K=2, Da=2
        obs["candidate_mask"]  : (2,)

    Shape example for FrozenLake-v1:

        raw_obs                : scalar discrete state id
        formatted obs["obs"]   : (16,) one-hot state vector
        action_space           : Discrete(4)
        obs["candidates"]      : (4, 4) one-hot action rows
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env: gym.Env,
        sampler: Optional[ExternalCandidateSampler] = None,
        max_candidates: Optional[int] = None,
    ):
        super().__init__()
        self.env = env
        self.render_mode = getattr(env, "render_mode", None)
        self._raw_observation_is_discrete = isinstance(convert_to_gymnasium_space(env.observation_space), spaces.Discrete)
        self.raw_observation_space = self._convert_observation_space(env.observation_space)
        self.raw_action_space = self._convert_action_space(env.action_space)

        if not isinstance(self.raw_action_space, spaces.Discrete):
            raise TypeError(
                "GenericDiscreteCandidateEnv currently supports only Discrete action spaces. "
                f"Got {self.raw_action_space}."
            )

        self.sampler = sampler or DiscreteOneHotExternalSampler(self.raw_action_space)
        self.max_candidates = int(max_candidates or self.raw_action_space.n)
        if self.max_candidates < int(self.raw_action_space.n):
            raise ValueError(
                f"max_candidates={self.max_candidates} is smaller than Discrete({self.raw_action_space.n})"
            )
        # Da for one-hot discrete actions is n_actions.
        self.candidate_dim = int(self.raw_action_space.n)
        # Observation dict shapes:
        #   obs:            (Ds,) for Box obs or one-hot Discrete obs
        #   candidates:     (K, Da) = (max_candidates, n_actions)
        #   candidate_mask: (K,)
        self.observation_space = spaces.Dict(
            {
                "obs": self.raw_observation_space,
                "candidates": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_candidates, self.candidate_dim),
                    dtype=np.float32,
                ),
                "candidate_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_candidates,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(self.max_candidates)
        self._candidate_labels: List[Any] = []

    def _convert_observation_space(self, observation_space: spaces.Space) -> spaces.Space:
        # SB3 can train on Dict observations, but this policy expects the "obs"
        # entry to flatten into a numeric tensor.  Most classic-control and
        # Box2D tasks are Box spaces; Discrete observations are converted to
        # one-hot Boxes so FrozenLake-like envs also work.
        observation_space = convert_to_gymnasium_space(observation_space)
        if isinstance(observation_space, spaces.Box):
            if is_image_space(observation_space):
                # Keep image observations as uint8 [0, 255] so SB3 can apply
                # the same VecTransposeImage and normalization path used by
                # built-in CnnPolicy/NatureCNN.
                return spaces.Box(
                    low=np.asarray(observation_space.low, dtype=observation_space.dtype),
                    high=np.asarray(observation_space.high, dtype=observation_space.dtype),
                    shape=observation_space.shape,
                    dtype=observation_space.dtype,
                )
            return spaces.Box(
                low=np.asarray(observation_space.low, dtype=np.float32),
                high=np.asarray(observation_space.high, dtype=np.float32),
                shape=observation_space.shape,
                dtype=np.float32,
            )
        if isinstance(observation_space, spaces.Discrete):
            return spaces.Box(low=0.0, high=1.0, shape=(int(observation_space.n),), dtype=np.float32)
        raise TypeError(
            "GenericDiscreteCandidateEnv currently supports Box or Discrete observations. "
            f"Got {observation_space}."
        )

    def _convert_action_space(self, action_space: spaces.Space) -> spaces.Space:
        action_space = convert_to_gymnasium_space(action_space)
        if isinstance(action_space, spaces.Discrete):
            return action_space
        raise TypeError(
            "GenericDiscreteCandidateEnv currently supports only Discrete actions. "
            f"Got {action_space}."
        )

    def _format_raw_obs(self, raw_obs: Any) -> np.ndarray:
        if self._raw_observation_is_discrete:
            # Discrete observation id -> one-hot vector:
            #   raw_obs: scalar i
            #   obs:     (n_states,), obs[i] = 1
            obs = np.zeros(self.raw_observation_space.shape, dtype=np.float32)
            obs[int(raw_obs)] = 1.0
            return obs
        # Image Boxes stay uint8 for CnnPolicy compatibility.  Other Box
        # observations are float32 and are flattened by the BDP MLP policy.
        return np.asarray(raw_obs, dtype=self.raw_observation_space.dtype)

    def _format_observation(self, raw_obs: Any) -> Dict[str, np.ndarray]:
        # formatted_obs: (obs_shape), later flattened by policy to (Ds,)
        formatted_obs = self._format_raw_obs(raw_obs)
        # candidates: (N, Da); for DiscreteOneHotExternalSampler, N=Da=n_actions.
        # labels: length N, candidate row i -> raw env action label labels[i].
        candidates, n_candidates, labels = self.sampler.sample(formatted_obs)
        candidates = np.asarray(candidates, dtype=np.float32)
        if candidates.ndim != 2:
            raise ValueError(f"Sampler must return a 2D candidate array, got {candidates.shape}")
        if candidates.shape[1] != self.candidate_dim:
            raise ValueError(
                f"Candidate dim {candidates.shape[1]} does not match expected dim {self.candidate_dim}"
            )
        if int(n_candidates) != int(candidates.shape[0]):
            raise ValueError(f"Sampler reported {n_candidates} candidates but returned {candidates.shape[0]}")
        if len(labels) != int(n_candidates):
            raise ValueError(f"Sampler returned {len(labels)} labels for {n_candidates} candidates")
        if int(n_candidates) > self.max_candidates:
            raise ValueError(f"Sampler returned {n_candidates} candidates but max_candidates={self.max_candidates}")

        # candidate_buffer: (K, Da), padded to fixed K for SB3 rollout storage.
        # candidate_mask:   (K,), 1 for candidates [0:N], 0 for padding [N:K].
        candidate_buffer = np.zeros((self.max_candidates, self.candidate_dim), dtype=np.float32)
        candidate_mask = np.zeros((self.max_candidates,), dtype=np.float32)
        candidate_buffer[: int(n_candidates)] = candidates
        candidate_mask[: int(n_candidates)] = 1.0
        self._candidate_labels = list(labels)
        # Returned unbatched dict shapes:
        #   "obs":            (obs_shape)
        #   "candidates":     (K, Da)
        #   "candidate_mask": (K,)
        return {
            "obs": formatted_obs,
            "candidates": candidate_buffer,
            "candidate_mask": candidate_mask,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        reset_result = self.env.reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            raw_obs, info = reset_result
        else:
            raw_obs, info = reset_result, {}
        return self._format_observation(raw_obs), dict(info or {})

    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # action is the policy output: a scalar candidate index in [0, K).
        # It is not necessarily the same as the raw environment label when the
        # sampler reorders or filters candidates, so use _candidate_labels.
        candidate_idx = int(np.asarray(action).reshape(-1)[0])
        if candidate_idx >= len(self._candidate_labels):
            raise ValueError(
                f"Invalid candidate index {candidate_idx}; only {len(self._candidate_labels)} candidates are valid"
            )

        raw_action = self._candidate_labels[candidate_idx]
        step_result = self.env.step(raw_action)
        if len(step_result) == 5:
            raw_obs, reward, terminated, truncated, info = step_result
        else:
            raw_obs, reward, done, info = step_result
            terminated = bool(done)
            truncated = False

        return self._format_observation(raw_obs), float(reward), bool(terminated), bool(truncated), dict(info or {})

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()


def make_legacy_carla_env(args: argparse.Namespace) -> Any:
    import gym as legacy_gym
    import carla_gym  # noqa: F401  Registers the CARLA gym environments.

    env_kwargs = dict(
        mode=args.planner_mode,
        is_finish_traj=args.is_finish_traj,
        use_lidar=args.use_lidar,
        num_traj=args.num_traj,
        scale_yaw=args.scale_yaw,
        scale_v=args.scale_v,
        debug=args.bdp_debug,
        short_hard=args.short_hard,
        env_change=args.env_change,
    )

    try:
        env = legacy_gym.make(args.env, disable_env_checker=True, **env_kwargs)
    except TypeError:
        env = legacy_gym.make(args.env, **env_kwargs)

    env = getattr(env, "unwrapped", env)
    if args.play_mode:
        env.enable_auto_render()
    env.begin_modules(args)
    return env


def make_raw_generic_env(args: argparse.Namespace, is_train: bool = True) -> gym.Env:
    """
    Create an ordinary Gymnasium/Gym env without the BDP candidate wrapper.

    This is used for ``--policy_mode=builtin`` so SB3 sees the same observation
    and action spaces it would see in a normal PPO/TRPO script.
    """

    requested_render_mode = getattr(args, "render_mode", None)
    should_render = bool(getattr(args, "render_train", False)) if is_train else bool(getattr(args, "play_mode", 0))
    render_mode = requested_render_mode if should_render else None
    gym_make_kwargs = dict(getattr(args, "gym_make_kwargs", {}) or {})
    if render_mode is not None:
        gym_make_kwargs["render_mode"] = render_mode

    if args.env_source == "gymnasium":
        env = gym.make(args.env, **gym_make_kwargs)
    elif args.env_source == "gym":
        import gym as legacy_gym

        try:
            legacy_env = legacy_gym.make(args.env, **gym_make_kwargs)
        except TypeError:
            gym_make_kwargs.pop("render_mode", None)
            legacy_env = legacy_gym.make(args.env, **gym_make_kwargs)
        env = LegacyGymCompatibilityEnv(legacy_env)
    else:
        raise ValueError(f"Unsupported generic env source: {args.env_source}")

    return env


def make_generic_discrete_env(args: argparse.Namespace, is_train: bool = True) -> gym.Env:
    env = make_raw_generic_env(args, is_train=is_train)
    sampler = DiscreteOneHotExternalSampler(env.action_space)
    return GenericDiscreteCandidateEnv(env, sampler=sampler, max_candidates=args.max_candidates)


def make_monitored_sb3_env(
    args: argparse.Namespace,
    log_dir: Path,
    is_train: bool,
    rank: Optional[int] = None,
) -> Monitor:
    if args.env_source == "carla":
        if getattr(args, "policy_mode", "bdp") == "builtin":
            raise ValueError(
                "--policy_mode=builtin is currently implemented for Gym/Gymnasium envs. "
                "Use --policy_mode=bdp for the legacy CARLA candidate-action wrapper."
            )
        if rank is not None:
            raise ValueError("Parallel CARLA envs are not enabled by this SB3 runner")
        legacy_env = make_legacy_carla_env(args)
        if hasattr(legacy_env, "open_log_saver"):
            legacy_env.open_log_saver(str(log_dir), is_train=is_train)

        candidate_env = BDPCandidateEnv(legacy_env, max_candidates=args.max_candidates)
        return Monitor(candidate_env, str(log_dir), info_keywords=("reserved",))

    if getattr(args, "policy_mode", "bdp") == "builtin":
        env = make_raw_generic_env(args, is_train=is_train)
    else:
        env = make_generic_discrete_env(args, is_train=is_train)
    monitor_dir = log_dir if is_train else log_dir / "test"
    if rank is not None:
        monitor_dir = monitor_dir / f"env_{rank}"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(env, str(monitor_dir))


def make_sb3_env(args: argparse.Namespace, log_dir: Path, is_train: bool) -> Any:
    n_envs = int(getattr(args, "n_envs", 1))
    if not is_train or n_envs <= 1:
        return make_monitored_sb3_env(args, log_dir, is_train=is_train)

    if args.env_source == "carla":
        raise ValueError(
            "Parallel training is currently implemented only for generic Gym/Gymnasium envs. "
            "CARLA needs separate ports/modules per process before it can be vectorized safely."
        )

    if n_envs < 1:
        raise ValueError(f"--n_envs must be >= 1, got {n_envs}")

    def make_env(rank: int) -> Callable[[], Monitor]:
        def _init() -> Monitor:
            return make_monitored_sb3_env(args, log_dir, is_train=True, rank=rank)

        return _init

    env_fns = [make_env(rank) for rank in range(n_envs)]
    if args.vec_env == "subproc":
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)
