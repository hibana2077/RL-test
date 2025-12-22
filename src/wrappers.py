import numpy as np
import gymnasium as gym
import torchvision.transforms as T


class PreprocessObsWrapper(gym.ObservationWrapper):
    """Fixed preprocessing: resize to 224, normalize to [-1,1], output CHW float32."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("PreprocessObsWrapper requires a Box observation space")
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

    def observation(self, observation):
        return self.pipeline(observation)



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
    Replace environment reward with custom shaping
    """

    def __init__(
        self,
        env,
        win_reward: float = 400.0,
    ):
        super().__init__(env)
        self.win_reward = win_reward
        self._prev_score = None

    def _reset_trackers(self, info):
        self._prev_score = info.get("score", 0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict):
            info = {}
        self._reset_trackers(info)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}

        reward = 0.0
        # Reward for score increments
        score = info.get("score")
        if score is not None:
            if self._prev_score is None:
                self._prev_score = score
            else:
                score_delta = score - self._prev_score
                if score_delta > 0:
                    reward += 10.0
                self._prev_score = score
        time_left = info.get("time_left", None)
        
        if terminated or truncated:
            self._reset_trackers(info)

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
def make_base_env(game: str, state: str):
    env = retro.make(game=game, state=state, render_mode="rgb_array")
    env = PreprocessObsWrapper(env)
    env = DiscreteActionWrapper(env, COMBOS)
    env = ExtraInfoWrapper(env)
    env = LifeTerminationWrapper(env)
    env = RewardOverrideWrapper(env)
    env = AuxObservationWrapper(env)
    return env
