import numpy as np
import gymnasium as gym
import torchvision.transforms as T
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional


class PreprocessObsWrapper(gym.ObservationWrapper):
    """Observation preprocessing.

    Modes:
    - "fixed" (default): resize to 224, normalize to [-1, 1], output numpy CHW float32
    - "timm": build torchvision transform from a timm model's pretrained config
      via timm.data.resolve_data_config + timm.data.create_transform
    """

    def __init__(
        self,
        env,
        mode: str = "fixed",
        timm_model: Optional[Any] = None,
        timm_model_name: Optional[str] = None,
        timm_data_cfg: Optional[dict[str, Any]] = None,
    ):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("PreprocessObsWrapper requires a Box observation space")

        if mode not in {"fixed", "timm"}:
            raise ValueError("mode must be 'fixed' or 'timm'")

        if mode == "fixed":
            self.resize_shape = (224, 224)
            c = obs_space.shape[2]
            low = np.full((c, 224, 224), -1.0, dtype=np.float32)
            high = np.full((c, 224, 224), 1.0, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

            transforms = []
            transforms.append(T.ToTensor())  # -> float CHW in [0,1]
            transforms.append(T.Resize(self.resize_shape, antialias=True))
            transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            transforms.append(T.Lambda(lambda x: x.numpy()))
            self.pipeline = T.Compose(transforms)
            self._use_timm = False
            return

        # === timm mode ===
        try:
            import timm  # local import so fixed mode doesn't depend on timm
            import timm.data
        except Exception as e:
            raise ImportError("timm mode requested but timm is not available") from e

        if timm_data_cfg is None:
            if timm_model is None:
                if not timm_model_name:
                    raise ValueError("timm mode requires timm_data_cfg, timm_model, or timm_model_name")
                # Note: envs are typically created once (n_envs=1), so creating a model here is acceptable.
                timm_model = timm.create_model(timm_model_name, pretrained=True)

            pretrained_cfg = getattr(timm_model, "pretrained_cfg", None) or getattr(timm_model, "default_cfg", {})
            timm_data_cfg = timm.data.resolve_data_config(pretrained_cfg, model=timm_model)

        input_size = timm_data_cfg.get("input_size", (3, 224, 224))
        if not (isinstance(input_size, (tuple, list)) and len(input_size) == 3):
            raise ValueError(f"unexpected timm data_cfg.input_size: {input_size!r}")
        c, h, w = int(input_size[0]), int(input_size[1]), int(input_size[2])
        self.resize_shape = (h, w)

        # timm normalization uses mean/std (typically ImageNet); bounds are unbounded after normalize.
        low = np.full((c, h, w), -np.inf, dtype=np.float32)
        high = np.full((c, h, w), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.pipeline = timm.data.create_transform(**timm_data_cfg, is_training=False)
        self._use_timm = True

    def observation(self, observation):
        if not self._use_timm:
            return self.pipeline(observation)

        # timm transforms are usually torchvision transforms that expect a PIL image.
        if isinstance(observation, np.ndarray):
            img = Image.fromarray(observation)
        elif isinstance(observation, Image.Image):
            img = observation
        else:
            # Best effort: pass through (some transforms accept tensors)
            img = observation

        out = self.pipeline(img)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        return out



class DiscreteActionWrapper(gym.ActionWrapper):
    """
    change action space from MultiBinary to Discrete with predefined button combos
    """
    def __init__(self, env, combos):
        super().__init__(env)


        if not hasattr(env.unwrapped, "buttons"):
            raise ValueError("unsupported env, must have 'buttons' attribute")

        self.buttons = list(env.unwrapped.buttons)  # e.g. ['B','Y','SELECT',...]
        self.button_to_idx = {b: i for i, b in enumerate(self.buttons)}

        # Get combos
        self.combos = combos
        self.action_space = gym.spaces.Discrete(len(combos))

        self._mapped = []
        n = env.action_space.n  # MultiBinary(n)
        for keys in combos:
            a = np.zeros(n, dtype=np.int8)
            for k in keys:
                if k not in self.button_to_idx:
                    raise ValueError(f"unsupported buttons in this env.buttons: {self.buttons}")
                a[self.button_to_idx[k]] = 1
            self._mapped.append(a)

    def action(self, act):
        return self._mapped[int(act)].copy()
    

class LifeTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._prev_lives = None

    def _get_lives(self, info):
        if not isinstance(info, dict):
            return None
        if "lives" in info:
            return int(info["lives"])
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_lives = self._get_lives(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        lives = self._get_lives(info)

        died = False
        if lives is not None and self._prev_lives is not None:
            if lives < self._prev_lives:
                died = True
        self._prev_lives = lives

        if died:
            terminated = True
            if isinstance(info, dict):
                info = dict(info)
                info["death"] = True

        return obs, reward, terminated, truncated, info


class ExtraInfoWrapper(gym.Wrapper):
    """
    Attach extra RAM-derived signals (HUD timer, x-position) to info.
    """

    TIMER_HUNDREDS = 0x0F31
    TIMER_TENS = 0x0F32
    TIMER_ONES = 0x0F33
    # In SMW RAM, $0094 stores the low byte and $0095 stores the high byte.
    X_POS_LOW = 0x0094
    X_POS_HIGH = 0x0095
    # In SMW RAM, $0096 stores the low byte and $0097 stores the high byte.
    Y_POS_LOW = 0x0096
    Y_POS_HIGH = 0x0097

    def __init__(self, env):
        super().__init__(env)
        self._episode_start_x = None
        self._episode_start_y = None

    def _get_ram(self):
        base_env = self.env.unwrapped
        if not hasattr(base_env, "get_ram"):
            return None
        return base_env.get_ram()

    def _read_time_left(self, ram):
        if ram is None:
            return None
        hundreds = int(ram[self.TIMER_HUNDREDS]) & 0x0F
        tens = int(ram[self.TIMER_TENS]) & 0x0F
        ones = int(ram[self.TIMER_ONES]) & 0x0F
        return hundreds * 100 + tens * 10 + ones

    def _read_x_pos(self, ram):
        if ram is None:
            return None
        low = int(ram[self.X_POS_LOW])
        high = int(ram[self.X_POS_HIGH])
        return (high << 8) | low

    def _read_y_pos(self, ram):
        if ram is None:
            return None
        low = int(ram[self.Y_POS_LOW])
        high = int(ram[self.Y_POS_HIGH])
        return (high << 8) | low

    def _inject_extra(self, info):
        ram = self._get_ram()
        time_left = self._read_time_left(ram)
        x_pos = self._read_x_pos(ram)
        y_pos = self._read_y_pos(ram)
        if time_left is None and x_pos is None and y_pos is None:
            return info
        if not isinstance(info, dict):
            info = {}
        # copy to avoid mutating shared dict instances
        info = dict(info)
        if time_left is not None:
            info["time_left"] = time_left
        if x_pos is not None:
            if self._episode_start_x is None:
                self._episode_start_x = x_pos
            info["x_pos"] = max(0, x_pos - self._episode_start_x)
        if y_pos is not None:
            # Keep both raw and delta (signed) so downstream shaping can decide direction.
            if self._episode_start_y is None:
                self._episode_start_y = y_pos
            info["y_pos_raw"] = int(y_pos)
            info["y_pos_delta"] = float(y_pos - self._episode_start_y)
        return info

    def reset(self, **kwargs):
        self._episode_start_x = None
        self._episode_start_y = None
        obs, info = self.env.reset(**kwargs)
        info = self._inject_extra(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._inject_extra(info)
        return obs, reward, terminated, truncated, info


class AuxObservationWrapper(gym.Wrapper):
    """
    Convert image observations into a dict that also exposes scalar features (step/time).
    """

    def __init__(self, env, step_normalizer: float = 18000.0, time_normalizer: float = 300.0):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("AuxObservationWrapper expects a Box observation space as the image input")
        self.image_space = env.observation_space
        self.step_normalizer = max(step_normalizer, 1.0)
        self.time_normalizer = max(time_normalizer, 1.0)
        scalar_low = np.full((2,), -np.inf, dtype=np.float32)
        scalar_high = np.full((2,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "image": self.image_space,
                "scalars": gym.spaces.Box(low=scalar_low, high=scalar_high, dtype=np.float32),
            }
        )
        self._step_count = 0

    def _make_obs(self, obs, info):
        time_left = float(info.get("time_left", 0.0)) if isinstance(info, dict) else 0.0
        time_feat = np.clip(time_left / self.time_normalizer, 0.0, 1.0)
        step_feat = np.clip(self._step_count / self.step_normalizer, 0.0, 1.0)
        scalars = np.array([step_feat, time_feat], dtype=np.float32)
        return {"image": obs, "scalars": scalars}

    def reset(self, **kwargs):
        self._step_count = 0
        obs, info = self.env.reset(**kwargs)
        return self._make_obs(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        return self._make_obs(obs, info), reward, terminated, truncated, info


class RewardOverrideWrapper(gym.Wrapper):
    """
        Baseline reward (docs/rd_f.md):

        Per-step reward:
            r = v + c + d

        where:
        - v = x1 - x0          (horizontal progress)
        - c = clock0 - clock1  (time bonus; usually ticks once per second)
        - d = death penalty    (applied on death step)

        Then clip reward to [-15, 15] for stability.

        Requirements:
        - ExtraInfoWrapper provides info['x_pos'] and info['time_left'].
        - LifeTerminationWrapper provides info['death']=True on death.
    """

    def __init__(
        self,
        env,
        reward_scale: float = 1.0,
        # === 終局處理 ===
        death_penalty: float = -15.0,
        # === Secret path bonus (optional; default off) ===
        secret_bonus: float = 0.0,
        secret_x_min: Optional[float] = None,
        secret_x_max: Optional[float] = None,
        secret_y_delta: Optional[float] = None,
        secret_y_mode: str = "down",  # 'down' (y_delta>=thr), 'up' (y_delta<=-thr), 'any' (abs>=thr)
        # === Secret path shaping (staged; optional; default off) ===
        # Stage 1: reach above entrance (x window + y_raw window) -> one-time bonus
        secret_stage1_bonus: float = 0.0,
        secret_stage1_x_min: Optional[float] = None,
        secret_stage1_x_max: Optional[float] = None,
        secret_stage1_y_raw_min: Optional[int] = None,
        secret_stage1_y_raw_max: Optional[int] = None,
        # Stage 2: after stage1, reward first N spin attempts (spin inferred by button name inside combo)
        secret_stage2_spin_bonus: float = 0.0,
        secret_stage2_spin_button: str = "A",
        secret_stage2_spin_required: int = 2,
        # Stage 3: actually enter/fall (x window + y_delta threshold) -> one-time bonus
        secret_stage3_bonus: float = 0.0,
        secret_stage3_x_min: Optional[float] = None,
        secret_stage3_x_max: Optional[float] = None,
        secret_stage3_y_delta: Optional[float] = None,
        secret_stage3_y_mode: str = "down",
    ):
        super().__init__(env)

        # Keep signature compatibility; only the following are used for baseline reward.
        self.reward_scale = float(reward_scale)
        self.death_penalty = float(death_penalty)

        self.secret_bonus = float(secret_bonus)
        self.secret_x_min = None if secret_x_min is None else float(secret_x_min)
        self.secret_x_max = None if secret_x_max is None else float(secret_x_max)
        self.secret_y_delta = None if secret_y_delta is None else float(secret_y_delta)
        if secret_y_mode not in {"down", "up", "any"}:
            raise ValueError("secret_y_mode must be one of: 'down', 'up', 'any'")
        self.secret_y_mode = str(secret_y_mode)

        # Staged shaping config (preferred). If stage3 fields are left at defaults,
        # legacy secret_* fields act as a fallback.
        self.secret_stage1_bonus = float(secret_stage1_bonus)
        self.secret_stage1_x_min = None if secret_stage1_x_min is None else float(secret_stage1_x_min)
        self.secret_stage1_x_max = None if secret_stage1_x_max is None else float(secret_stage1_x_max)
        self.secret_stage1_y_raw_min = None if secret_stage1_y_raw_min is None else int(secret_stage1_y_raw_min)
        self.secret_stage1_y_raw_max = None if secret_stage1_y_raw_max is None else int(secret_stage1_y_raw_max)

        self.secret_stage2_spin_bonus = float(secret_stage2_spin_bonus)
        self.secret_stage2_spin_button = str(secret_stage2_spin_button)
        self.secret_stage2_spin_required = max(0, int(secret_stage2_spin_required))

        self.secret_stage3_bonus = float(secret_stage3_bonus)
        self.secret_stage3_x_min = None if secret_stage3_x_min is None else float(secret_stage3_x_min)
        self.secret_stage3_x_max = None if secret_stage3_x_max is None else float(secret_stage3_x_max)
        self.secret_stage3_y_delta = None if secret_stage3_y_delta is None else float(secret_stage3_y_delta)
        if secret_stage3_y_mode not in {"down", "up", "any"}:
            raise ValueError("secret_stage3_y_mode must be one of: 'down', 'up', 'any'")
        self.secret_stage3_y_mode = str(secret_stage3_y_mode)

        self._secret_stage1_done = False
        self._secret_stage2_count = 0
        self._secret_stage3_done = False

        self._secret_given = False

        self._prev_x_pos: Optional[float] = None
        self._prev_time_left: Optional[float] = None

    def _reset_trackers(self, info: dict[str, Any]):
        x_pos = info.get("x_pos", None)
        time_left = info.get("time_left", None)
        self._prev_x_pos = float(x_pos) if x_pos is not None else None
        self._prev_time_left = float(time_left) if time_left is not None else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict):
            info = {}
        self._reset_trackers(info)
        self._secret_given = False
        self._secret_stage1_done = False
        self._secret_stage2_count = 0
        self._secret_stage3_done = False
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}

        # Baseline: r = v + c + d
        reward = 0.0

        # v = x1 - x0
        x_pos = info.get("x_pos", None)
        if x_pos is not None:
            x_pos_f = float(x_pos)
            if self._prev_x_pos is not None:
                reward += x_pos_f - float(self._prev_x_pos)
            self._prev_x_pos = x_pos_f

        # c = clock0 - clock1
        time_left = info.get("time_left", None)
        if time_left is not None:
            time_left_f = float(time_left)
            if self._prev_time_left is not None:
                reward += float(self._prev_time_left) - time_left_f
            self._prev_time_left = time_left_f

        # d = death penalty (on death step)
        if info.get("death", False):
            reward += self.death_penalty

        # -----------------
        # Secret path shaping (staged)
        # -----------------
        x_pos = info.get("x_pos", None)
        y_raw = info.get("y_pos_raw", None)
        y_delta = info.get("y_pos_delta", None)

        # Stage 1: reach above entrance (x + y_raw window)
        if (
            self.secret_stage1_bonus != 0.0
            and not self._secret_stage1_done
            and x_pos is not None
            and y_raw is not None
        ):
            x = float(x_pos)
            y = int(y_raw)
            x_ok = True
            if self.secret_stage1_x_min is not None and x < self.secret_stage1_x_min:
                x_ok = False
            if self.secret_stage1_x_max is not None and x > self.secret_stage1_x_max:
                x_ok = False
            y_ok = True
            if self.secret_stage1_y_raw_min is not None and y < self.secret_stage1_y_raw_min:
                y_ok = False
            if self.secret_stage1_y_raw_max is not None and y > self.secret_stage1_y_raw_max:
                y_ok = False

            if x_ok and y_ok:
                reward += self.secret_stage1_bonus
                self._secret_stage1_done = True
                info = dict(info)
                info["secret_stage1"] = True
                info["secret_stage1_bonus"] = float(self.secret_stage1_bonus)

        # Stage 2: reward first N spin attempts after stage1
        if (
            self.secret_stage2_spin_bonus != 0.0
            and self._secret_stage1_done
            and self.secret_stage2_spin_required > 0
            and self._secret_stage2_count < self.secret_stage2_spin_required
        ):
            act_i = int(action) if np.isscalar(action) else int(np.asarray(action).item())
            try:
                combo = COMBOS[act_i]
            except Exception:
                combo = None
            if isinstance(combo, (list, tuple)) and (self.secret_stage2_spin_button in combo):
                reward += self.secret_stage2_spin_bonus
                self._secret_stage2_count += 1
                info = dict(info)
                info["secret_stage2_spin"] = True
                info["secret_stage2_spin_count"] = int(self._secret_stage2_count)
                info["secret_stage2_spin_bonus"] = float(self.secret_stage2_spin_bonus)

        # Stage 3: enter/fall into secret path.
        # Prefer staged config; fall back to legacy secret_* if staged left unset.
        stage3_bonus = self.secret_stage3_bonus if self.secret_stage3_bonus != 0.0 else self.secret_bonus
        stage3_x_min = self.secret_stage3_x_min if self.secret_stage3_x_min is not None else self.secret_x_min
        stage3_x_max = self.secret_stage3_x_max if self.secret_stage3_x_max is not None else self.secret_x_max
        stage3_y_delta = self.secret_stage3_y_delta if self.secret_stage3_y_delta is not None else self.secret_y_delta
        stage3_y_mode = self.secret_stage3_y_mode if self.secret_stage3_bonus != 0.0 or self.secret_stage3_y_delta is not None else self.secret_y_mode

        if (
            stage3_bonus != 0.0
            and not self._secret_stage3_done
            and (stage3_y_delta is not None)
            and (y_delta is not None)
        ):
            x_ok = True
            if (stage3_x_min is not None or stage3_x_max is not None) and (x_pos is not None):
                x = float(x_pos)
                if stage3_x_min is not None and x < stage3_x_min:
                    x_ok = False
                if stage3_x_max is not None and x > stage3_x_max:
                    x_ok = False
            thr = float(stage3_y_delta)
            yd = float(y_delta)
            if stage3_y_mode == "down":
                y_ok = yd >= thr
            elif stage3_y_mode == "up":
                y_ok = yd <= -thr
            else:
                y_ok = abs(yd) >= thr

            if x_ok and y_ok:
                reward += float(stage3_bonus)
                self._secret_stage3_done = True
                info = dict(info)
                info["secret_stage3"] = True
                info["secret_stage3_bonus"] = float(stage3_bonus)

        # Optional: one-time bonus when entering a "secret" path region.
        # This is intentionally simple and uses only info fields produced by ExtraInfoWrapper.
        if (
            self.secret_bonus != 0.0
            and not self._secret_given
            and (self.secret_y_delta is not None)
            and ("y_pos_delta" in info)
        ):
            x_ok = True
            if (self.secret_x_min is not None or self.secret_x_max is not None) and ("x_pos" in info):
                x = float(info.get("x_pos", 0.0))
                if self.secret_x_min is not None and x < self.secret_x_min:
                    x_ok = False
                if self.secret_x_max is not None and x > self.secret_x_max:
                    x_ok = False
            y_delta = float(info.get("y_pos_delta", 0.0))
            thr = float(self.secret_y_delta)
            if self.secret_y_mode == "down":
                y_ok = y_delta >= thr
            elif self.secret_y_mode == "up":
                y_ok = y_delta <= -thr
            else:
                y_ok = abs(y_delta) >= thr

            if x_ok and y_ok:
                reward += self.secret_bonus
                self._secret_given = True
                info = dict(info)
                info["secret_trigger"] = True
                info["secret_bonus"] = float(self.secret_bonus)

        # clip to [-15, 15]
        reward = float(np.clip(reward, -15.0, 15.0))

        reward *= self.reward_scale
        return obs, reward, terminated, truncated, info


class _RunningMeanVar:
    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.eps = float(eps)

    def update(self, x: float) -> None:
        x_f = float(x)
        self.count += 1
        if self.count == 1:
            self.mean = x_f
            self.var = 0.0
            return
        delta = x_f - self.mean
        self.mean += delta / self.count
        delta2 = x_f - self.mean
        m2 = self.var * (self.count - 1) + delta * delta2
        self.var = m2 / max(self.count - 1, 1)

    def std(self) -> float:
        return float(np.sqrt(max(self.var, 0.0) + self.eps))


class IntrinsicRewardWrapper(gym.Wrapper):
    """Add intrinsic rewards: curiosity (RND), novelty (episodic), state surprise.

    Outputs to info:
      - r_intrinsic, r_intrinsic_curiosity, r_intrinsic_novelty, r_intrinsic_surprise
      - r_extrinsic (the incoming reward before adding intrinsic)

    If intrinsic_scale != 0, the env reward is: r_total = r_extrinsic + intrinsic_scale * r_intrinsic
    """

    def __init__(
        self,
        env,
        *,
        intrinsic_scale: float = 0.0,
        w_curiosity: float = 1.0,
        w_novelty: float = 1.0,
        w_surprise: float = 1.0,
        embed_hw: tuple[int, int] = (16, 16),
        rnd_hidden: int = 128,
        rnd_out_dim: int = 64,
        rnd_lr: float = 1e-4,
        novelty_hash_bytes: int = 64,
        max_episode_novelty_states: int = 50_000,
        intrinsic_clip: float = 10.0,
    ):
        super().__init__(env)

        self.intrinsic_scale = float(intrinsic_scale)
        self.w_curiosity = float(w_curiosity)
        self.w_novelty = float(w_novelty)
        self.w_surprise = float(w_surprise)
        self.embed_hw = (int(embed_hw[0]), int(embed_hw[1]))
        self.novelty_hash_bytes = int(novelty_hash_bytes)
        self.max_episode_novelty_states = int(max_episode_novelty_states)
        self.intrinsic_clip = float(intrinsic_clip)

        self._prev_embed: Optional[torch.Tensor] = None  # shape (D,)
        self._visit_counts: dict[bytes, int] = {}

        # RND networks (CPU by default; keep it lightweight and independent of SB3 device)
        self._rnd_in_dim: Optional[int] = None
        self._rnd_target: Optional[nn.Module] = None
        self._rnd_pred: Optional[nn.Module] = None
        self._rnd_opt: Optional[torch.optim.Optimizer] = None
        self._rnd_rms = _RunningMeanVar()
        self._rnd_hidden = int(rnd_hidden)
        self._rnd_out_dim = int(rnd_out_dim)
        self._rnd_lr = float(rnd_lr)

        # Surprise stats (action-conditioned on delta magnitude)
        self._surprise_rms = _RunningMeanVar()
        self._per_action_stats: dict[int, _RunningMeanVar] = {}

    def _extract_image(self, obs: Any) -> Optional[np.ndarray]:
        if isinstance(obs, dict):
            if "image" in obs:
                return obs["image"]
            return None
        return obs

    def _obs_to_tensor(self, obs: Any) -> Optional[torch.Tensor]:
        img = self._extract_image(obs)
        if img is None:
            return None
        if isinstance(img, torch.Tensor):
            x = img.detach().float()
            if x.ndim == 3:
                x = x.unsqueeze(0)
        else:
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                # CHW
                chw = arr
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                # HWC -> CHW
                chw = np.transpose(arr, (2, 0, 1))
            else:
                return None
            x = torch.from_numpy(chw).float().unsqueeze(0)

        # Ensure (B, C, H, W)
        if x.ndim != 4:
            return None
        return x

    def _embed(self, obs: Any) -> Optional[torch.Tensor]:
        x = self._obs_to_tensor(obs)
        if x is None:
            return None
        # (B,C,H,W) -> pooled -> flat
        pooled = F.adaptive_avg_pool2d(x, self.embed_hw)
        flat = pooled.flatten(1)
        return flat[0].contiguous()

    def _maybe_init_rnd(self, embed_dim: int) -> None:
        if self._rnd_in_dim is not None:
            return
        self._rnd_in_dim = int(embed_dim)
        self._rnd_target = nn.Sequential(
            nn.Linear(self._rnd_in_dim, self._rnd_hidden),
            nn.ReLU(),
            nn.Linear(self._rnd_hidden, self._rnd_hidden),
            nn.ReLU(),
            nn.Linear(self._rnd_hidden, self._rnd_out_dim),
        )
        self._rnd_pred = nn.Sequential(
            nn.Linear(self._rnd_in_dim, self._rnd_hidden),
            nn.ReLU(),
            nn.Linear(self._rnd_hidden, self._rnd_hidden),
            nn.ReLU(),
            nn.Linear(self._rnd_hidden, self._rnd_out_dim),
        )
        for p in self._rnd_target.parameters():
            p.requires_grad_(False)
        self._rnd_opt = torch.optim.Adam(self._rnd_pred.parameters(), lr=self._rnd_lr)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visit_counts = {}
        self._prev_embed = self._embed(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}
        else:
            info = dict(info)

        r_extrinsic = float(reward)

        embed_next = self._embed(obs)
        embed_prev = self._prev_embed
        self._prev_embed = embed_next

        r_curiosity = 0.0
        r_novelty = 0.0
        r_surprise = 0.0

        if embed_next is not None:
            # --- novelty (episodic) ---
            try:
                bits = (embed_next.detach().cpu().numpy() > 0).astype(np.uint8)
                packed = np.packbits(bits)
                key = packed.tobytes()[: self.novelty_hash_bytes]
            except Exception:
                key = None
            if key is not None:
                if len(self._visit_counts) < self.max_episode_novelty_states:
                    c = self._visit_counts.get(key, 0) + 1
                    self._visit_counts[key] = c
                else:
                    c = 1
                r_novelty = 1.0 / float(np.sqrt(c))

            # --- curiosity (RND) ---
            if self.w_curiosity != 0.0:
                self._maybe_init_rnd(int(embed_next.numel()))
                assert self._rnd_target is not None and self._rnd_pred is not None and self._rnd_opt is not None
                x = embed_next.detach().unsqueeze(0)
                with torch.no_grad():
                    y_t = self._rnd_target(x)
                y_p = self._rnd_pred(x)
                loss = torch.mean((y_p - y_t) ** 2)
                r_curiosity_raw = float(loss.detach().cpu().item())
                self._rnd_rms.update(r_curiosity_raw)
                r_curiosity = r_curiosity_raw / self._rnd_rms.std()

                self._rnd_opt.zero_grad(set_to_none=True)
                loss.backward()
                self._rnd_opt.step()

        # --- state surprise (delta magnitude conditioned on action) ---
        if embed_prev is not None and embed_next is not None and self.w_surprise != 0.0:
            delta = (embed_next - embed_prev).detach()
            delta_norm2 = float(torch.mean(delta * delta).cpu().item())
            a = int(action) if np.isscalar(action) else int(np.asarray(action).item())
            stats = self._per_action_stats.get(a)
            if stats is None:
                stats = _RunningMeanVar()
                self._per_action_stats[a] = stats

            # compute surprise before updating stats
            z = (delta_norm2 - stats.mean) / stats.std()
            r_surprise_raw = float(z * z)
            stats.update(delta_norm2)

            self._surprise_rms.update(r_surprise_raw)
            r_surprise = r_surprise_raw / self._surprise_rms.std()

        r_intrinsic = (
            self.w_curiosity * float(r_curiosity)
            + self.w_novelty * float(r_novelty)
            + self.w_surprise * float(r_surprise)
        )
        if self.intrinsic_clip > 0:
            r_intrinsic = float(np.clip(r_intrinsic, -self.intrinsic_clip, self.intrinsic_clip))

        # Write info fields
        info["r_extrinsic"] = r_extrinsic
        info["r_intrinsic"] = float(r_intrinsic)
        info["r_intrinsic_curiosity"] = float(r_curiosity)
        info["r_intrinsic_novelty"] = float(r_novelty)
        info["r_intrinsic_surprise"] = float(r_surprise)

        if self.intrinsic_scale != 0.0:
            reward = float(r_extrinsic + self.intrinsic_scale * r_intrinsic)

        return obs, reward, terminated, truncated, info


class InfoLogger(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if reward > 0 or terminated:
            print(info)
        return obs, reward, terminated, truncated, info

COMBOS = [
    [],                  # 0: NOOP
    ["RIGHT"],           # 1: 走右
    ["LEFT"],            # 2: 走左(可選)
    ["DOWN"],            # 3: 下蹲
    ["A"],               # 4: A (often spin jump in SMW)
    ["B"],               # 5: 跑
    ["RIGHT", "A"],      # 6: 右 + 跳
    ["RIGHT", "B"],      # 7: 右 + 跑
    ["RIGHT", "A", "B"], # 8: 右 + 跳 + 跑
    ["LEFT", "A"],       # 9: 左 + A
    ["LEFT", "B"],       # 10: 左 + B
    ["LEFT", "A", "B"],  # 11: 左 + A + B
]
import retro
def make_base_env(
    game: str,
    state: str,
    *,
    preprocess_mode: str = "fixed",
    timm_model_name: Optional[str] = None,
    timm_data_cfg: Optional[dict[str, Any]] = None,
    reward_scale: float = 1.0,
    # Secret path shaping (optional)
    secret_bonus: float = 0.0,
    secret_x_min: Optional[float] = None,
    secret_x_max: Optional[float] = None,
    secret_y_delta: Optional[float] = None,
    secret_y_mode: str = "down",
    # Secret path shaping (staged; optional)
    secret_stage1_bonus: float = 0.0,
    secret_stage1_x_min: Optional[float] = None,
    secret_stage1_x_max: Optional[float] = None,
    secret_stage1_y_raw_min: Optional[int] = None,
    secret_stage1_y_raw_max: Optional[int] = None,
    secret_stage2_spin_bonus: float = 0.0,
    secret_stage2_spin_button: str = "A",
    secret_stage2_spin_required: int = 2,
    secret_stage3_bonus: float = 0.0,
    secret_stage3_x_min: Optional[float] = None,
    secret_stage3_x_max: Optional[float] = None,
    secret_stage3_y_delta: Optional[float] = None,
    secret_stage3_y_mode: str = "down",
    intrinsic_enable: bool = False,
    intrinsic_scale: float = 0.0,
    intrinsic_w_curiosity: float = 1.0,
    intrinsic_w_novelty: float = 1.0,
    intrinsic_w_surprise: float = 1.0,
):
    env = retro.make(game=game, state=state, render_mode="rgb_array")
    env = PreprocessObsWrapper(
        env,
        mode=preprocess_mode,
        timm_model_name=timm_model_name,
        timm_data_cfg=timm_data_cfg,
    )
    env = DiscreteActionWrapper(env, COMBOS)
    env = ExtraInfoWrapper(env)
    env = LifeTerminationWrapper(env)
    env = RewardOverrideWrapper(
        env,
        reward_scale=reward_scale,
        secret_bonus=secret_bonus,
        secret_x_min=secret_x_min,
        secret_x_max=secret_x_max,
        secret_y_delta=secret_y_delta,
        secret_y_mode=secret_y_mode,
        secret_stage1_bonus=secret_stage1_bonus,
        secret_stage1_x_min=secret_stage1_x_min,
        secret_stage1_x_max=secret_stage1_x_max,
        secret_stage1_y_raw_min=secret_stage1_y_raw_min,
        secret_stage1_y_raw_max=secret_stage1_y_raw_max,
        secret_stage2_spin_bonus=secret_stage2_spin_bonus,
        secret_stage2_spin_button=secret_stage2_spin_button,
        secret_stage2_spin_required=secret_stage2_spin_required,
        secret_stage3_bonus=secret_stage3_bonus,
        secret_stage3_x_min=secret_stage3_x_min,
        secret_stage3_x_max=secret_stage3_x_max,
        secret_stage3_y_delta=secret_stage3_y_delta,
        secret_stage3_y_mode=secret_stage3_y_mode,
    )
    if intrinsic_enable or intrinsic_scale != 0.0:
        env = IntrinsicRewardWrapper(
            env,
            intrinsic_scale=intrinsic_scale,
            w_curiosity=intrinsic_w_curiosity,
            w_novelty=intrinsic_w_novelty,
            w_surprise=intrinsic_w_surprise,
        )
    env = AuxObservationWrapper(env)
    return env
