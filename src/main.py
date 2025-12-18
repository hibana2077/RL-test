import os
import retro
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

# 重新載入模組以確保使用最新的代碼
import importlib
import wrappers
import policy
import eval as eval_module
importlib.reload(wrappers)
importlib.reload(policy)
importlib.reload(eval_module)

from wrappers import make_base_env
from policy import VisionBackbonePolicy, CustomPPO
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
print("=" * 60)

# 由於 retro 限制每個進程只能有一個環境實例
# 我們使用單一環境並用 DummyVecEnv 包裝

def create_single_env(game: str, state: str):
    """創建單一環境並用 DummyVecEnv 包裝 (retro 限制)"""
    env = make_base_env(game, state)
    vec_env = DummyVecEnv([lambda: env])
    return vec_env, env  # 返回兩者以便後續使用

print("Environment functions defined.")

# 0. Close existing environment if exists
try:
    if 'base_env' in globals():
        base_env.close()
    if 'train_env' in globals():
        train_env.close()
except:
    pass

# 1. Create Training Environment
print("Creating training environment...")
base_env = make_base_env(GAME, STATE)
if N_ENVS != 1:
    print(f"[WARN] retro 環境通常限制每個進程只能有一個實例；已將 n_envs 從 {N_ENVS} 覆寫為 1")
    N_ENVS = 1

train_env = DummyVecEnv([lambda: base_env])
print(f"Environment: {GAME} - {STATE}")
print(f"Observation: {train_env.observation_space}")
print(f"Actions: {train_env.action_space}")

# 2. Initialize Model with anti-local-optimum settings
print("\n" + "=" * 60)
print("Initializing PPO with 強力抗局部最優配置...")
print("=" * 60)

model = CustomPPO(
    VisionBackbonePolicy,
    train_env,
    policy_kwargs=dict(normalize_images=False),
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

        # --- Evaluate (使用同一個環境) ---
        print("Evaluating...")
        mean_ret, best_ret = evaluate_policy(
            model,
            GAME,
            STATE,
            n_episodes=EVAL_EPISODES,
            max_steps=EVAL_MAX_STEPS,
            env=base_env,  # 使用同一個環境
        )
        print(f"[EVAL] Mean Return: {mean_ret:.3f}, Best Return: {best_ret:.3f}")

        # --- Save Best Model ---
        if mean_ret > best_mean:
            best_mean = mean_ret
            best_path = os.path.join(LOG_DIR, "best_model.zip")
            model.save(best_path)
            print(f"*** New best record! Saved to {best_path} ***")

        # --- Record Video (使用同一個環境) ---
        print("Recording video...")
        record_video(
            model,
            GAME,
            STATE,
            VIDEO_DIR,
            video_len=RECORD_STEPS,
            prefix=f"step_{trained}_mean_{mean_ret:.2f}",
            env=base_env,  # 使用同一個環境
        )
        
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