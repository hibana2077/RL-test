import os
import argparse
from typing import Tuple, Optional
import numpy as np
from policy import CustomPPO
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from wrappers import make_base_env


def evaluate_policy(
    model: CustomPPO,
    game: str,
    state: str,
    n_episodes: int,
    max_steps: int,
    env=None,
    *,
    preprocess_mode: str = "fixed",
    timm_model_name: Optional[str] = None,
):
    """
    評估策略性能。如果提供了 env 則使用該環境，否則創建新環境。
    """
    close_env = False
    if env is None:
        env = make_base_env(game, state, preprocess_mode=preprocess_mode, timm_model_name=timm_model_name)
        close_env = True
    
    returns = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
            steps += 1

        returns.append(ep_ret)

    if close_env:
        env.close()
    
    mean_ret = float(np.mean(returns)) if returns else 0.0
    best_ret = float(np.max(returns)) if returns else 0.0
    return mean_ret, best_ret

def _format_info(info: dict, max_len: int = 120) -> str:
    if not isinstance(info, dict) or not info:
        return "{}"
    parts = []
    total_len = 0
    for key, value in info.items():
        fragment = f"{key}={value}"
        if total_len + len(fragment) > max_len:
            parts.append("...")
            break
        parts.append(fragment)
        total_len += len(fragment) + 2
    return "{" + ", ".join(parts) + "}"


def _annotate_frame(frame: np.ndarray, cumulative_reward: float, last_reward: float, info: dict, font: ImageFont.ImageFont) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    info_str = _format_info(info)
    lines = [
        f"reward={last_reward:.3f}",
        f"cum_reward={cumulative_reward:.3f}",
        f"info: {info_str}",
    ]
    padding = 4
    bbox_sample = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox_sample[3] - bbox_sample[1]
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
    box_width = max(line_widths) + padding * 2
    box_height = line_height * len(lines) + padding * (len(lines) + 1)
    draw.rectangle([0, 0, box_width, box_height], fill=(0, 0, 0, 200))
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height + padding
    return np.array(img)


def record_video(
    model: CustomPPO,
    game: str,
    state: str,
    out_dir: str,
    video_len: int,
    prefix: str,
    env=None,
    *,
    preprocess_mode: str = "fixed",
    timm_model_name: Optional[str] = None,
):
    """
    錄製遊戲影片。如果提供了 env 則使用該環境，否則創建新環境。
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{prefix}.mp4")

        close_env = False
        if env is None:
            env = make_base_env(game, state, preprocess_mode=preprocess_mode, timm_model_name=timm_model_name)
            close_env = True
        
        fps = env.metadata.get("render_fps", 60)
        # Force the ffmpeg backend; otherwise imageio may fall back to a TIFF writer
        # (e.g., when imageio-ffmpeg isn't installed), which doesn't accept fps/codec.
        writer = imageio.get_writer(out_path, format="ffmpeg", fps=fps, codec="libx264")
        font = ImageFont.load_default()

        obs, info = env.reset()
        cumulative_reward = 0.0
        for _ in range(video_len):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is None:
                continue
            cumulative_reward += float(reward)
            annotated = _annotate_frame(frame, cumulative_reward, float(reward), info, font)
            writer.append_data(annotated)
            if terminated or truncated:
                obs, info = env.reset()
                cumulative_reward = 0.0

        writer.close()
        if close_env:
            env.close()
        print(f"Saved video to {out_path}")
    except Exception as e:
        print(f"Warning: Could not record video: {e}")
        print("If you're on Windows, ensure FFmpeg is available and install: pip install imageio-ffmpeg")
        print("Skipping video recording...")