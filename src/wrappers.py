import numpy as np
import gymnasium as gym
import torchvision.transforms as T
from PIL import Image
import torch
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

    def __init__(self, env):
        super().__init__(env)
        self._episode_start_x = None

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

    def _inject_extra(self, info):
        ram = self._get_ram()
        time_left = self._read_time_left(ram)
        x_pos = self._read_x_pos(ram)
        if time_left is None and x_pos is None:
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
        return info

    def reset(self, **kwargs):
        self._episode_start_x = None
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
        Dense reward shaping for Super Mario World.

        目前 shaping 條件:
        - 前進主 reward(唯一 dense): x_delta * distance_scale
        - 新最遠位置 bonus(低頻事件): 刷新 max_x 時給 new_max_bonus (預設為小常數)
        - 存活 reward 服從前進: 只有在 x_delta > 0 時才給 survival_reward
        - 停滯 soft constraint: 超過 stagnation_threshold 後，縮小 forward reward(避免額外負獎勵疊加)
        - 動作多樣性(小、短期): 最近 10 步動作種類 >= 4 時給 action_diversity_reward
        - 終局型獎懲: win_reward / death_penalty 只在 episode 結束時套用

        其餘項目(距離指數、每秒移動範圍懲罰、分數/金幣)預設關閉，僅供 debug/curriculum。

        備註:若 info 沒有 x_pos(RAM 取不到),會 fallback 使用 env_reward + 多樣性,避免 reward 退化。
    """

    def __init__(
        self,
        env,
        reward_scale: float = 1.0,
        # === 距離獎勵 (最重要！) ===
        distance_scale: float = 0.1,        # 唯一 dense 前進獎勵
        new_max_bonus: float = 1.0,         # 刷新最遠位置的低頻探索提示(建議小)
        # (debug/curriculum) 距離指數獎勵：預設關閉，避免後期爆炸
        distance_exp_bonus: float = 0.0,
        
        # === 存活獎勵 (服從前進) ===
        survival_reward: float = 0.02,      # 只有 x_delta>0 時才給
        survival_scaling: bool = False,     # 若要用 scaling，仍會在 x_delta>0 時才套用
        
        # === 動作多樣性獎勵 ===
        action_diversity_reward: float = 0.1,  # 獎勵使用不同動作
        
        # === 終局處理 ===
        death_penalty: float = -250.0,       # 終局死亡懲罰(負值)
        # (debug/curriculum) 死亡時根據距離給獎勵：預設關閉
        distance_on_death_bonus: float = 0.0,

        # === (debug/curriculum) 每秒位移範圍懲罰(逼迫往前走) ===
        steps_per_second: int = 60,
        min_movement_range: float = 0.0,
        movement_range_penalty: float = 0.0,
        
        # === 停滯 soft constraint ===
        stagnation_threshold: int = 15,
        forward_reward_scale_when_stagnant: float = 0.2,
        # (debug/curriculum) 若仍想額外施加停滯負獎勵，可手動設為非 0
        stagnation_penalty: float = 0.0,
        
        # === (debug/curriculum) 遊戲指標：預設關閉 ===
        coin_reward: float = 0.0,
        score_reward_scale: float = 0.0,
        
        # === 通關(終局) ===
        win_reward: float = 500.0,
    ):
        super().__init__(env)

        self.reward_scale = float(reward_scale)
        
        self.distance_scale = distance_scale
        self.distance_exp_bonus = distance_exp_bonus
        self.new_max_bonus = new_max_bonus
        self.survival_reward = survival_reward
        self.survival_scaling = survival_scaling
        self.action_diversity_reward = action_diversity_reward
        self.death_penalty = death_penalty
        self.distance_on_death_bonus = distance_on_death_bonus

        self.steps_per_second = max(int(steps_per_second), 1)
        self.min_movement_range = float(min_movement_range)
        self.movement_range_penalty = float(movement_range_penalty)

        self.stagnation_threshold = stagnation_threshold
        self.forward_reward_scale_when_stagnant = float(np.clip(forward_reward_scale_when_stagnant, 0.0, 1.0))
        self.stagnation_penalty = stagnation_penalty
        self.coin_reward = coin_reward
        self.score_reward_scale = score_reward_scale
        self.win_reward = win_reward
        
        # 追蹤變數
        self._prev_x_pos = 0
        self._prev_score = 0
        self._prev_coins = 0
        self._stagnation_counter = 0
        self._max_x_pos = 0
        self._step_count = 0
        self._action_history = []
        self._episode_total_distance = 0
        self._x_pos_window = []

    def _reset_trackers(self, info):
        self._prev_x_pos = info.get("x_pos", 0)
        self._prev_score = info.get("score", 0)
        self._prev_coins = info.get("coins", 0)
        self._stagnation_counter = 0
        self._max_x_pos = info.get("x_pos", 0)
        self._step_count = 0
        self._action_history = []
        self._episode_total_distance = 0
        self._x_pos_window = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict):
            info = {}
        self._reset_trackers(info)
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}

        self._step_count += 1
        reward = 0.0
        
        # If x_pos is missing (e.g., RAM unavailable or address mismatch),
        # fall back to the underlying env reward to avoid a degenerate constant-return signal.
        x_pos = info.get("x_pos", None)
        if x_pos is None:
            shaped = 0.0

            # retain native reward when x_pos is unavailable
            shaped += float(env_reward)

            # action diversity still applies
            if isinstance(action, np.ndarray):
                action_int = int(action[0]) if len(action.shape) > 0 else int(action)
            else:
                action_int = int(action)
            self._action_history.append(action_int)
            if len(self._action_history) > 20:
                self._action_history.pop(0)
            if len(self._action_history) >= 10:
                unique_actions = len(set(self._action_history[-10:]))
                if unique_actions >= 4:
                    shaped += self.action_diversity_reward

            if terminated or truncated:
                self._reset_trackers(info)

            shaped *= self.reward_scale
            return obs, shaped, terminated, truncated, info

        x_pos = float(x_pos)

        # update 1-second x_pos window
        self._x_pos_window.append(x_pos)
        if len(self._x_pos_window) > self.steps_per_second:
            self._x_pos_window.pop(0)

        # ============ 1) 前進主 reward (唯一 dense) ============
        x_delta = 0.0
        if self._prev_x_pos is not None:
            x_delta = float(x_pos - float(self._prev_x_pos))

        forward_reward = 0.0
        if x_delta > 0:
            forward_reward = x_delta * float(self.distance_scale)
            self._episode_total_distance += x_delta
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

        # 停滯 soft constraint：超過 threshold 後縮小 forward reward
        if self._stagnation_counter > self.stagnation_threshold and forward_reward > 0:
            forward_reward *= self.forward_reward_scale_when_stagnant

        reward += forward_reward

        # ============ 2) 新最遠位置 bonus (低頻事件，避免與 x_delta 雙倍計算) ============
        if x_pos > self._max_x_pos:
            reward += float(self.new_max_bonus)
            self._max_x_pos = x_pos

        # ============ 3) (debug/curriculum) 距離指數加成：預設關閉 ============
        if self.distance_exp_bonus and x_pos > 50:
            exp_bonus = (x_pos ** float(self.distance_exp_bonus)) * 0.00001
            reward += float(min(exp_bonus, 5.0))

        # ============ 4) 存活獎勵服從前進 ============
        if x_delta > 0:
            if self.survival_scaling:
                survival = float(self.survival_reward) * (1.0 + self._step_count / 500.0)
                reward += float(min(survival, 0.2))
            else:
                reward += float(self.survival_reward)

        self._prev_x_pos = x_pos
        
        # ============ 3. 動作多樣性獎勵 ============
        # 記錄最近 20 個動作,獎勵使用不同動作
        # 將 action 轉為整數 (如果是 array)
        if isinstance(action, np.ndarray):
            action_int = int(action[0]) if len(action.shape) > 0 else int(action)
        else:
            action_int = int(action)
        
        self._action_history.append(action_int)
        if len(self._action_history) > 20:
            self._action_history.pop(0)
        if len(self._action_history) >= 10:
            unique_actions = len(set(self._action_history[-10:]))
            if unique_actions >= 4:  # 使用了至少 4 種不同動作
                reward += self.action_diversity_reward
        
        # ============ 4.5 停滯(額外負獎勵，僅供 debug/curriculum) ============
        if self.stagnation_penalty != 0 and self._stagnation_counter > self.stagnation_threshold:
            reward += float(self.stagnation_penalty)

        # ============ 4.5 每秒移動範圍懲罰(強迫往前走) ============
        # 以 steps_per_second 步近似 1 秒；若 1 秒內移動範圍不足則扣分。
        if self.min_movement_range > 0 and self.movement_range_penalty > 0:
            if len(self._x_pos_window) >= self.steps_per_second:
                movement_range = float(max(self._x_pos_window) - min(self._x_pos_window))
                if movement_range < self.min_movement_range:
                    deficit_ratio = (self.min_movement_range - movement_range) / self.min_movement_range
                    reward -= self.movement_range_penalty * float(np.clip(deficit_ratio, 0.0, 1.0))
        
        # ============ 5. 分數和金幣 ============
        score = info.get("score", 0)
        if self._prev_score is not None:
            score_delta = score - self._prev_score
            if score_delta > 0:
                reward += score_delta * self.score_reward_scale
        self._prev_score = score
        
        coins = info.get("coins", 0)
        if self._prev_coins is not None:
            coins_delta = coins - self._prev_coins
            if coins_delta > 0:
                reward += coins_delta * self.coin_reward
        self._prev_coins = coins
        
        # ============ 6. 終局型獎懲(只在 episode 結束時套用) ============
        if terminated or truncated:
            if info.get("death", False):
                reward += float(self.death_penalty)

                # (debug/curriculum) 死亡時依距離給 bonus
                if self.distance_on_death_bonus != 0:
                    reward += float(self._max_x_pos) * float(self.distance_on_death_bonus)

            # 通關：best-effort 用 x_pos 門檻判斷
            if x_pos > 4800:
                reward += float(self.win_reward)
        
        # Episode 結束
        if terminated or truncated:
            self._reset_trackers(info)

        reward *= self.reward_scale
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
    env = RewardOverrideWrapper(env, reward_scale=reward_scale)
    env = AuxObservationWrapper(env)
    return env
