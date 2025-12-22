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
                timm_model = timm.create_model(timm_model_name, pretrained=True)

            pretrained_cfg = getattr(timm_model, "pretrained_cfg", None) or getattr(timm_model, "default_cfg", {})
            timm_data_cfg = timm.data.resolve_data_config(pretrained_cfg, model=timm_model)

        input_size = timm_data_cfg.get("input_size", (3, 224, 224))
        if not (isinstance(input_size, (tuple, list)) and len(input_size) == 3):
            raise ValueError(f"unexpected timm data_cfg.input_size: {input_size!r}")
        c, h, w = int(input_size[0]), int(input_size[1]), int(input_size[2])
        self.resize_shape = (h, w)

        low = np.full((c, h, w), -np.inf, dtype=np.float32)
        high = np.full((c, h, w), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.pipeline = timm.data.create_transform(**timm_data_cfg, is_training=False)
        self._use_timm = True

    def observation(self, observation):
        if not self._use_timm:
            return self.pipeline(observation)

        if isinstance(observation, np.ndarray):
            img = Image.fromarray(observation)
        elif isinstance(observation, Image.Image):
            img = observation
        else:
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
    """Reward shaping + optional secret shaping.

    Baseline reward:
      r = v + c + d

    - v = x1 - x0 (horizontal progress)
    - c = time_left_t-1 - time_left_t (time bonus)
    - d = death penalty on death step

    Then clip to [-15, 15] and multiply by reward_scale.
    """

    def __init__(
        self,
        env,
        reward_scale: float = 1.0,
        death_penalty: float = -15.0,
        secret_bonus: float = 0.0,
        secret_x_min: Optional[float] = None,
        secret_x_max: Optional[float] = None,
        secret_y_delta: Optional[float] = None,
        secret_y_mode: str = "down",
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
    ):
        super().__init__(env)
        self.reward_scale = float(reward_scale)
        self.death_penalty = float(death_penalty)

        self.secret_bonus = float(secret_bonus)
        self.secret_x_min = None if secret_x_min is None else float(secret_x_min)
        self.secret_x_max = None if secret_x_max is None else float(secret_x_max)
        self.secret_y_delta = None if secret_y_delta is None else float(secret_y_delta)
        if secret_y_mode not in {"down", "up", "any"}:
            raise ValueError("secret_y_mode must be one of: 'down', 'up', 'any'")
        self.secret_y_mode = str(secret_y_mode)

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

    def _in_range(self, v: float, vmin: Optional[float], vmax: Optional[float]) -> bool:
        if vmin is not None and v < vmin:
            return False
        if vmax is not None and v > vmax:
            return False
        return True

    def _y_delta_ok(self, y_delta: float, thr: float, mode: str) -> bool:
        if mode == "down":
            return y_delta >= thr
        if mode == "up":
            return y_delta <= -thr
        return abs(y_delta) >= thr

    def _action_has_button(self, action: Any, button: str) -> bool:
        # Best-effort: reach into DiscreteActionWrapper if present
        combos = getattr(self.env, "combos", None)
        if combos is None:
            combos = getattr(getattr(self.env, "unwrapped", None), "combos", None)
        try:
            idx = int(action)
        except Exception:
            return False
        if isinstance(combos, list) and 0 <= idx < len(combos):
            return button in combos[idx]
        return False

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}

        reward = 0.0

        # v = x1 - x0
        x_pos = info.get("x_pos", None)
        if x_pos is not None:
            x_pos_f = float(x_pos)
            if self._prev_x_pos is not None:
                reward += x_pos_f - self._prev_x_pos
            self._prev_x_pos = x_pos_f

        # c = time_left0 - time_left1
        time_left = info.get("time_left", None)
        if time_left is not None:
            time_left_f = float(time_left)
            if self._prev_time_left is not None:
                reward += self._prev_time_left - time_left_f
            self._prev_time_left = time_left_f

        # d = death penalty
        if bool(info.get("death", False)):
            reward += self.death_penalty

        # staged secret shaping
        if not self._secret_stage1_done and self.secret_stage1_bonus != 0.0:
            y_raw = info.get("y_pos_raw", None)
            x = info.get("x_pos", None)
            if x is not None and y_raw is not None:
                if self._in_range(float(x), self.secret_stage1_x_min, self.secret_stage1_x_max):
                    y_ok = True
                    if self.secret_stage1_y_raw_min is not None and int(y_raw) < self.secret_stage1_y_raw_min:
                        y_ok = False
                    if self.secret_stage1_y_raw_max is not None and int(y_raw) > self.secret_stage1_y_raw_max:
                        y_ok = False
                    if y_ok:
                        reward += self.secret_stage1_bonus
                        self._secret_stage1_done = True

        if self._secret_stage1_done and not self._secret_stage3_done and self.secret_stage2_spin_bonus != 0.0:
            if self._secret_stage2_count < self.secret_stage2_spin_required:
                if self._action_has_button(action, self.secret_stage2_spin_button):
                    reward += self.secret_stage2_spin_bonus
                    self._secret_stage2_count += 1

        if not self._secret_stage3_done and self.secret_stage3_bonus != 0.0:
            x = info.get("x_pos", None)
            y_delta = info.get("y_pos_delta", None)
            if x is not None and y_delta is not None and self.secret_stage3_y_delta is not None:
                if self._in_range(float(x), self.secret_stage3_x_min, self.secret_stage3_x_max) and self._y_delta_ok(
                    float(y_delta), self.secret_stage3_y_delta, self.secret_stage3_y_mode
                ):
                    reward += self.secret_stage3_bonus
                    self._secret_stage3_done = True

        # legacy secret bonus (one-time)
        if not self._secret_given and self.secret_bonus != 0.0:
            x = info.get("x_pos", None)
            y_delta = info.get("y_pos_delta", None)
            if (
                x is not None
                and y_delta is not None
                and self.secret_y_delta is not None
                and self._in_range(float(x), self.secret_x_min, self.secret_x_max)
                and self._y_delta_ok(float(y_delta), self.secret_y_delta, self.secret_y_mode)
            ):
                reward += self.secret_bonus
                self._secret_given = True

        reward = float(np.clip(reward, -15.0, 15.0))
        reward *= self.reward_scale

        if terminated or truncated:
            self._reset_trackers(info)

        return obs, reward, terminated, truncated, info


class IntrinsicRewardWrapper(gym.Wrapper):
    """Simple intrinsic reward signals.

    - curiosity: random-projection prediction error (lightweight, online)
    - novelty: episodic count-based bonus on discretized x_pos/time_left
    - surprise: abs(delta x_pos) (progress surprise)
    """

    def __init__(
        self,
        env,
        intrinsic_scale: float = 0.0,
        w_curiosity: float = 1.0,
        w_novelty: float = 1.0,
        w_surprise: float = 1.0,
        proj_dim: int = 128,
        seed: int = 0,
    ):
        super().__init__(env)
        self.intrinsic_scale = float(intrinsic_scale)
        self.w_curiosity = float(w_curiosity)
        self.w_novelty = float(w_novelty)
        self.w_surprise = float(w_surprise)
        self.rng = np.random.RandomState(seed)

        # for curiosity: random projection target + trainable linear predictor (torch)
        self.proj_dim = int(proj_dim)
        self._pred: Optional[nn.Module] = None
        self._opt: Optional[torch.optim.Optimizer] = None
        self._target_w: Optional[torch.Tensor] = None

        self._counts: dict[tuple[int, int], int] = {}
        self._prev_x: Optional[float] = None

    def _flat_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            img = obs.get("image")
            sc = obs.get("scalars")
            parts = []
            if img is not None:
                parts.append(np.asarray(img, dtype=np.float32).reshape(-1))
            if sc is not None:
                parts.append(np.asarray(sc, dtype=np.float32).reshape(-1))
            if parts:
                return np.concatenate(parts, axis=0)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _ensure_models(self, dim: int):
        if self._pred is not None:
            return
        w = self.rng.normal(0, 1.0 / np.sqrt(dim), size=(dim, self.proj_dim)).astype(np.float32)
        self._target_w = torch.tensor(w)
        self._pred = nn.Linear(dim, self.proj_dim)
        self._opt = torch.optim.Adam(self._pred.parameters(), lr=1e-3)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(info, dict):
            x = info.get("x_pos", None)
            self._prev_x = float(x) if x is not None else None
        else:
            self._prev_x = None
        self._counts.clear()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}

        intrinsic = 0.0

        # novelty: discretize (x_pos,time_left)
        if self.w_novelty != 0.0:
            x = int(info.get("x_pos", 0) // 16) if info.get("x_pos", None) is not None else 0
            t = int(info.get("time_left", 0) // 10) if info.get("time_left", None) is not None else 0
            key = (x, t)
            c = self._counts.get(key, 0) + 1
            self._counts[key] = c
            intrinsic += self.w_novelty * (1.0 / np.sqrt(c))

        # surprise: abs(delta x)
        if self.w_surprise != 0.0:
            x_now = info.get("x_pos", None)
            if x_now is not None:
                x_now_f = float(x_now)
                if self._prev_x is not None:
                    intrinsic += self.w_surprise * abs(x_now_f - self._prev_x)
                self._prev_x = x_now_f

        # curiosity: online predictor error on random projection of obs
        if self.w_curiosity != 0.0:
            flat = self._flat_obs(obs)
            self._ensure_models(int(flat.shape[0]))
            x_t = torch.tensor(flat).unsqueeze(0)
            with torch.no_grad():
                target = x_t @ self._target_w  # type: ignore[operator]
            pred = self._pred(x_t)  # type: ignore[operator]
            loss = F.mse_loss(pred, target)
            self._opt.zero_grad()  # type: ignore[union-attr]
            loss.backward()
            self._opt.step()  # type: ignore[union-attr]
            intrinsic += self.w_curiosity * float(loss.detach().cpu().item())

        intrinsic *= self.intrinsic_scale
        if intrinsic != 0.0:
            info = dict(info)
            info["intrinsic_reward"] = intrinsic

        return obs, float(reward) + float(intrinsic), terminated, truncated, info


class InfoLogger(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if reward > 0 or terminated:
            print(info)
        return obs, reward, terminated, truncated, info

COMBOS = [
    [],                  # 0: NOOP
    ["RIGHT"],           # 1: 走右
    ["LEFT"],            # 2: 走左（可選）
    ["DOWN"],            # 3: 下蹲
    ["A"],               # 4: 跳
    ["B"],               # 5: 跑
    ["RIGHT", "A"],      # 6: 右 + 跳
    ["RIGHT", "B"],      # 7: 右 + 跑
    ["RIGHT", "A", "B"], # 8: 右 + 跳 + 跑
    ["LEFT", "A"],       # 10: 左 + 跳
    ["LEFT", "B"],       # 11: 左 + 跑
    ["LEFT", "A", "B"],  # 12: 左 + 跳 + 跑
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
    secret_bonus: float = 0.0,
    secret_x_min: Optional[float] = None,
    secret_x_max: Optional[float] = None,
    secret_y_delta: Optional[float] = None,
    secret_y_mode: str = "down",
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
