import os
import retro
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

# 重新載入模組以確保使用最新的代碼
import importlib
import wrappers
import custom_policy
import eval as eval_module
importlib.reload(wrappers)
importlib.reload(custom_policy)
importlib.reload(eval_module)

from wrappers import make_base_env
from custom_policy import VisionBackbonePolicy, CustomPPO
from eval import evaluate_policy, record_video

# 檢查 GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Gym Retro (SMW) with configurable hyperparameters")

    # Game
    parser.add_argument("--game", type=str, default="SuperMarioWorld-Snes", help="Retro game id")
    parser.add_argument("--state", type=str, default="YoshiIsland1", help="Retro state")

    # Training (defaults keep the original constants)
    parser.add_argument("--total-steps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--train-chunk", type=int, default=25_000, help="Timesteps per training round")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs (retro usually requires 1)")
    parser.add_argument("--n-steps", type=int, default=1024, help="Steps collected per rollout")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--kl-coef", type=float, default=0.1, help="KL coefficient (CustomPPO)")
    parser.add_argument("--ent-coef", type=float, default=0.1, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--clip-range", type=float, default=0.3, help="PPO clip range")

    # Eval / record
    parser.add_argument("--eval-episodes", type=int, default=2, help="Episodes per evaluation")
    parser.add_argument("--eval-max-steps", type=int, default=6000, help="Max steps per eval episode")
    parser.add_argument("--record-steps", type=int, default=3000, help="Steps to record per round")

    # Logging
    parser.add_argument("--log-dir", type=str, default="./runs_smw", help="Base output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device string, e.g. 'cpu' or 'cuda:0'",
    )

    # Vision backbone (timm model name)
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="timm backbone model name for VisionBackboneExtractor (default: resnet18)",
    )

    # Reward scaling
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Multiply shaped rewards by this factor (default: 1.0)",
    )

    # Secret path shaping (optional)
    parser.add_argument(
        "--secret-bonus",
        type=float,
        default=0.0,
        help="[legacy] One-time bonus when entering secret path region (default: 0.0 = off)",
    )
    parser.add_argument(
        "--secret-x-min",
        type=float,
        default=None,
        help="Min x_pos (relative) for secret trigger (default: None)",
    )
    parser.add_argument(
        "--secret-x-max",
        type=float,
        default=None,
        help="Max x_pos (relative) for secret trigger (default: None)",
    )
    parser.add_argument(
        "--secret-y-delta",
        type=float,
        default=None,
        help="Threshold on y_pos_delta for secret trigger (default: None)",
    )
    parser.add_argument(
        "--secret-y-mode",
        type=str,
        default="down",
        choices=["down", "up", "any"],
        help="[legacy] How to interpret y_pos_delta vs threshold: down/up/any (default: down)",
    )

    # Secret path shaping (staged; recommended)
    parser.add_argument("--secret-stage1-bonus", type=float, default=0.0, help="Stage1 bonus on reaching above entrance")
    parser.add_argument("--secret-stage1-x-min", type=float, default=None, help="Stage1 x_pos min")
    parser.add_argument("--secret-stage1-x-max", type=float, default=None, help="Stage1 x_pos max")
    parser.add_argument("--secret-stage1-y-raw-min", type=int, default=None, help="Stage1 y_pos_raw min")
    parser.add_argument("--secret-stage1-y-raw-max", type=int, default=None, help="Stage1 y_pos_raw max")

    parser.add_argument("--secret-stage2-spin-bonus", type=float, default=0.0, help="Stage2 bonus per spin attempt")
    parser.add_argument(
        "--secret-stage2-spin-button",
        type=str,
        default="A",
        help="Button name that indicates spin jump in COMBOS (default: A)",
    )
    parser.add_argument("--secret-stage2-spin-required", type=int, default=2, help="How many spin attempts to reward")

    parser.add_argument("--secret-stage3-bonus", type=float, default=0.0, help="Stage3 bonus when actually entering/falling")
    parser.add_argument("--secret-stage3-x-min", type=float, default=None, help="Stage3 x_pos min")
    parser.add_argument("--secret-stage3-x-max", type=float, default=None, help="Stage3 x_pos max")
    parser.add_argument("--secret-stage3-y-delta", type=float, default=None, help="Stage3 threshold on y_pos_delta")
    parser.add_argument(
        "--secret-stage3-y-mode",
        type=str,
        default="down",
        choices=["down", "up", "any"],
        help="Stage3 y_pos_delta mode (down/up/any)",
    )

    # Intrinsic rewards (curiosity/novelty/surprise)
    parser.add_argument(
        "--intrinsic-enable",
        action="store_true",
        help="Enable intrinsic rewards wrapper (curiosity/novelty/surprise)",
    )
    parser.add_argument(
        "--intrinsic-scale",
        type=float,
        default=0.0,
        help="Total intrinsic reward scale added to env reward (default: 0.0 = off)",
    )
    parser.add_argument("--intrinsic-w-curiosity", type=float, default=1.0, help="Weight for curiosity (RND)")
    parser.add_argument("--intrinsic-w-novelty", type=float, default=1.0, help="Weight for novelty (episodic)")
    parser.add_argument("--intrinsic-w-surprise", type=float, default=1.0, help="Weight for state surprise")

    return parser.parse_args()


ARGS = parse_args()

# Game Settings
GAME = ARGS.game
STATE = ARGS.state

# ==========================================
# 強力抗局部最優配置 V2 (可透過參數覆寫)
# ==========================================
TOTAL_STEPS = ARGS.total_steps     # 總訓練步數
TRAIN_CHUNK = ARGS.train_chunk     # 每輪訓練步數
N_ENVS = ARGS.n_envs               # 並行環境數 (retro 限制)
N_STEPS = ARGS.n_steps             # 每次收集步數
BATCH_SIZE = ARGS.batch_size       # Batch size
N_EPOCHS = ARGS.n_epochs           # PPO epochs
LEARNING_RATE = ARGS.learning_rate # 學習率
GAMMA = ARGS.gamma                 # 折扣因子
KL_COEF = ARGS.kl_coef             # KL coefficient
VF_COEF = ARGS.vf_coef             # Value function coefficient
ENT_COEF = ARGS.ent_coef           # Entropy coefficient
CLIP_RANGE = ARGS.clip_range       # PPO clip range

# Evaluation & Recording Settings
EVAL_EPISODES = ARGS.eval_episodes
EVAL_MAX_STEPS = ARGS.eval_max_steps
RECORD_STEPS = ARGS.record_steps

# Directories
LOG_DIR = ARGS.log_dir
VIDEO_DIR = os.path.join(LOG_DIR, "videos")
CKPT_DIR = os.path.join(LOG_DIR, "checkpoints")
TENSORBOARD_LOG = os.path.join(LOG_DIR, "tb")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print("=" * 60)
print("強力抗局部最優配置 V2")
print("=" * 60)
print(f"★ ENT_COEF = {ENT_COEF} (超高探索!)")
print(f"★ KL_COEF = {KL_COEF} (允許大更新)")
print(f"★ GAMMA = {GAMMA} (重視長期)")
print(f"★ 死亡懲罰 = 0 (完全不怕死)")
print(f"★ 存活獎勵 = 隨時間增加")
print(f"★ GAME/STATE = {GAME} / {STATE}")
print(f"★ TOTAL_STEPS = {TOTAL_STEPS:,} | TRAIN_CHUNK = {TRAIN_CHUNK:,}")
print(f"★ N_STEPS = {N_STEPS} | BATCH_SIZE = {BATCH_SIZE} | N_EPOCHS = {N_EPOCHS}")
print(f"★ LEARNING_RATE = {LEARNING_RATE} | CLIP_RANGE = {CLIP_RANGE}")
print(f"★ DEVICE = {ARGS.device}")
print(f"★ BACKBONE = {ARGS.backbone}")
print("=" * 60)

# 由於 retro 限制每個進程只能有一個環境實例
# 我們使用單一環境並用 DummyVecEnv 包裝

def create_single_env(game: str, state: str):
    """創建單一環境並用 DummyVecEnv 包裝 (retro 限制)"""
    env = make_base_env(
        game,
        state,
        preprocess_mode="timm",
        timm_model_name=ARGS.backbone,
        reward_scale=ARGS.reward_scale,
        # legacy
        secret_bonus=ARGS.secret_bonus,
        secret_x_min=ARGS.secret_x_min,
        secret_x_max=ARGS.secret_x_max,
        secret_y_delta=ARGS.secret_y_delta,
        secret_y_mode=ARGS.secret_y_mode,
        # staged
        secret_stage1_bonus=ARGS.secret_stage1_bonus,
        secret_stage1_x_min=ARGS.secret_stage1_x_min,
        secret_stage1_x_max=ARGS.secret_stage1_x_max,
        secret_stage1_y_raw_min=ARGS.secret_stage1_y_raw_min,
        secret_stage1_y_raw_max=ARGS.secret_stage1_y_raw_max,
        secret_stage2_spin_bonus=ARGS.secret_stage2_spin_bonus,
        secret_stage2_spin_button=ARGS.secret_stage2_spin_button,
        secret_stage2_spin_required=ARGS.secret_stage2_spin_required,
        secret_stage3_bonus=ARGS.secret_stage3_bonus,
        secret_stage3_x_min=ARGS.secret_stage3_x_min,
        secret_stage3_x_max=ARGS.secret_stage3_x_max,
        secret_stage3_y_delta=ARGS.secret_stage3_y_delta,
        secret_stage3_y_mode=ARGS.secret_stage3_y_mode,
        intrinsic_enable=ARGS.intrinsic_enable,
        intrinsic_scale=ARGS.intrinsic_scale,
        intrinsic_w_curiosity=ARGS.intrinsic_w_curiosity,
        intrinsic_w_novelty=ARGS.intrinsic_w_novelty,
        intrinsic_w_surprise=ARGS.intrinsic_w_surprise,
    )
    vec_env = DummyVecEnv([lambda: env])
    return vec_env, env  # 返回兩者以便後續使用

print("Environment functions defined.")

# 0. Close existing environment if exists
try:
    if "train_env" in globals():
        train_env.close()
except Exception:
    pass

# 1. Create Training Environment
print("Creating training environment...")
if N_ENVS != 1:
    print(f"[WARN] retro 環境通常限制每個進程只能有一個實例；已將 n_envs 從 {N_ENVS} 覆寫為 1")
    N_ENVS = 1

def _make_train_env():
    return make_base_env(
        GAME,
        STATE,
        preprocess_mode="timm",
        timm_model_name=ARGS.backbone,
        reward_scale=ARGS.reward_scale,
        # legacy
        secret_bonus=ARGS.secret_bonus,
        secret_x_min=ARGS.secret_x_min,
        secret_x_max=ARGS.secret_x_max,
        secret_y_delta=ARGS.secret_y_delta,
        secret_y_mode=ARGS.secret_y_mode,
        # staged
        secret_stage1_bonus=ARGS.secret_stage1_bonus,
        secret_stage1_x_min=ARGS.secret_stage1_x_min,
        secret_stage1_x_max=ARGS.secret_stage1_x_max,
        secret_stage1_y_raw_min=ARGS.secret_stage1_y_raw_min,
        secret_stage1_y_raw_max=ARGS.secret_stage1_y_raw_max,
        secret_stage2_spin_bonus=ARGS.secret_stage2_spin_bonus,
        secret_stage2_spin_button=ARGS.secret_stage2_spin_button,
        secret_stage2_spin_required=ARGS.secret_stage2_spin_required,
        secret_stage3_bonus=ARGS.secret_stage3_bonus,
        secret_stage3_x_min=ARGS.secret_stage3_x_min,
        secret_stage3_x_max=ARGS.secret_stage3_x_max,
        secret_stage3_y_delta=ARGS.secret_stage3_y_delta,
        secret_stage3_y_mode=ARGS.secret_stage3_y_mode,
        intrinsic_enable=ARGS.intrinsic_enable,
        intrinsic_scale=ARGS.intrinsic_scale,
        intrinsic_w_curiosity=ARGS.intrinsic_w_curiosity,
        intrinsic_w_novelty=ARGS.intrinsic_w_novelty,
        intrinsic_w_surprise=ARGS.intrinsic_w_surprise,
    )

train_env = DummyVecEnv([_make_train_env])
print(f"Environment: {GAME} - {STATE}")
print(f"Observation: {train_env.observation_space}")
print(f"Actions: {train_env.action_space}")

# 2. Initialize Model with anti-local-optimum settings
print("\n" + "=" * 60)
print("Initializing PPO...")
print("=" * 60)

model = CustomPPO(
    VisionBackbonePolicy,
    train_env,
    policy_kwargs=dict(
        normalize_images=False,
        features_extractor_kwargs=dict(backbone_name=ARGS.backbone),
    ),
    n_epochs=N_EPOCHS,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    verbose=1,
    gamma=GAMMA,
    kl_coef=KL_COEF,
    ent_coef=ENT_COEF,
    clip_range=CLIP_RANGE,
    tensorboard_log=TENSORBOARD_LOG,
    device=ARGS.device,
)

# 模型資訊
total_params = sum(p.numel() for p in model.policy.parameters())
model_size_mb = total_params * 4 / 1024**2
print(f"\nModel: {total_params:,} params ({model_size_mb:.2f} MB)")
print("✓ Model size < 1 GB")

# Training Loop - 使用同一個環境進行訓練和評估
best_mean = -1e18
trained = 0
round_idx = 0

print("=" * 60)
print("Starting Training...")
print("=" * 60)

try:
    while trained < TOTAL_STEPS:
        round_idx += 1
        chunk = min(TRAIN_CHUNK, TOTAL_STEPS - trained)

        print(f"\n=== Round {round_idx} | Learn {chunk:,} steps (Total: {trained:,}/{TOTAL_STEPS:,}) ===")
        
        # --- Train ---
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=True)
        trained += chunk

        # --- Save Checkpoint ---
        ckpt_path = os.path.join(CKPT_DIR, f"CustomPPO_step_{trained}.zip")
        model.save(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # NOTE: Do NOT reuse the same env for train+eval+record.
        # SB3 keeps internal last_obs between learn() calls (reset_num_timesteps=False).
        # If we step/reset the training env outside model.learn(), rollouts become inconsistent
        # and losses can blow up while rewards appear stuck.
        #
        # Gym Retro is also finicky about multiple env instances per process, so we
        # temporarily close the training env before eval/video, then recreate it.
        train_env.close()

        # --- Evaluate ---
        print("Evaluating...")
        mean_ret, best_ret = evaluate_policy(
            model,
            GAME,
            STATE,
            n_episodes=EVAL_EPISODES,
            max_steps=EVAL_MAX_STEPS,
            preprocess_mode="timm",
            timm_model_name=ARGS.backbone,
            reward_scale=ARGS.reward_scale,
        )
        print(f"[EVAL] Mean Return: {mean_ret:.3f}, Best Return: {best_ret:.3f}")

        # --- Save Best Model ---
        if mean_ret > best_mean:
            best_mean = mean_ret
            best_path = os.path.join(LOG_DIR, "best_model.zip")
            model.save(best_path)
            print(f"*** New best record! Saved to {best_path} ***")

        # --- Record Video ---
        print("Recording video...")
        record_video(
            model,
            GAME,
            STATE,
            VIDEO_DIR,
            video_len=RECORD_STEPS,
            prefix=f"step_{trained}_mean_{mean_ret:.2f}",
            preprocess_mode="timm",
            timm_model_name=ARGS.backbone,
            reward_scale=ARGS.reward_scale,
        )

        # Re-create training env after eval/video
        train_env = DummyVecEnv([_make_train_env])
        model.set_env(train_env)
        
        # GPU 記憶體狀態
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[GPU] Memory allocated: {allocated:.2f} GB")

except KeyboardInterrupt:
    print("\n*** Training interrupted manually ***")

finally:
    print("\n" + "=" * 60)
    print(f"Training finished!")
    print(f"Total steps trained: {trained:,}")
    print(f"Best mean return: {best_mean:.3f}")
    print("=" * 60)

    try:
        train_env.close()
    except Exception:
        pass

print(f"You can now review the recorded videos in the '{VIDEO_DIR}' directory.")

# from IPython.display import Video
# import glob

# # Find the latest video file
# list_of_files = glob.glob(os.path.join(VIDEO_DIR, '*.mp4')) 
# if list_of_files:
#     latest_file = max(list_of_files, key=os.path.getctime)
#     print(f"Playing: {latest_file}")
#     display(Video(latest_file, embed=True, width=600))
# else:
#     print("No videos found yet.")

# # 訓練結束後關閉環境
# try:
#     base_env.close()
#     train_env.close()
#     print("Environment closed successfully.")
# except:
#     pass