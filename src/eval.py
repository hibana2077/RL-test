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
    reward_scale: float = 1.0,
    intrinsic_enable: bool = False,
    intrinsic_scale: float = 0.0,
    intrinsic_w_curiosity: float = 1.0,
    intrinsic_w_novelty: float = 1.0,
    intrinsic_w_surprise: float = 1.0,
    # Secret path shaping (optional)
    secret_bonus: float = 0.0,
    secret_x_min: Optional[float] = None,
    secret_x_max: Optional[float] = None,
    secret_y_delta: Optional[float] = None,
    secret_y_mode: str = "down",
    # Secret path shaping (staged; recommended)
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
    """
    評估策略性能。如果提供了 env 則使用該環境，否則創建新環境。
    """
    close_env = False
    if env is None:
        env = make_base_env(
            game,
            state,
            preprocess_mode=preprocess_mode,
            timm_model_name=timm_model_name,
            reward_scale=reward_scale,
            # legacy
            secret_bonus=secret_bonus,
            secret_x_min=secret_x_min,
            secret_x_max=secret_x_max,
            secret_y_delta=secret_y_delta,
            secret_y_mode=secret_y_mode,
            # staged
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
            intrinsic_enable=intrinsic_enable,
            intrinsic_scale=intrinsic_scale,
            intrinsic_w_curiosity=intrinsic_w_curiosity,
            intrinsic_w_novelty=intrinsic_w_novelty,
            intrinsic_w_surprise=intrinsic_w_surprise,
        )
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


def _format_info_lines(
    info: dict,
    *,
    max_pairs_per_line: int = 2,
    max_lines: int = 6,
) -> list[str]:
    """Format env `info` as multiple lines.

    Requirement: each line contains at most `max_pairs_per_line` key=value fragments.
    """
    if not isinstance(info, dict) or not info:
        return ["{}"]

    fragments = [f"{k}={v}" for k, v in info.items()]
    lines: list[str] = []
    i = 0
    while i < len(fragments) and len(lines) < max_lines:
        chunk = fragments[i : i + max_pairs_per_line]
        lines.append(", ".join(chunk))
        i += max_pairs_per_line

    if i < len(fragments):
        if lines:
            lines[-1] = lines[-1] + ", ..."
        else:
            lines.append("...")
    return lines


def _clip_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if max_width <= 0:
        return ""
    bbox = draw.textbbox((0, 0), text, font=font)
    if (bbox[2] - bbox[0]) <= max_width:
        return text

    ellipsis = "…"
    # Binary search for the longest prefix that fits when suffixed with ellipsis.
    lo, hi = 0, len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ellipsis
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _annotate_frame(frame: np.ndarray, cumulative_reward: float, last_reward: float, info: dict, font: ImageFont.ImageFont) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    info_lines = _format_info_lines(info, max_pairs_per_line=2)
    lines = [
        f"reward={last_reward:.3f}",
        f"cum_reward={cumulative_reward:.3f}",
    ]
    if info_lines == ["{}"]:
        lines.append("info: {}")
    else:
        lines.append(f"info: {info_lines[0]}")
        for extra in info_lines[1:]:
            lines.append(f"      {extra}")

    # Ensure the overlay never renders beyond the frame width.
    max_text_width = max(1, img.size[0] - (4 * 2))
    lines = [_clip_text_to_width(draw, line, font, max_text_width) for line in lines]
    padding = 4
    bbox_sample = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox_sample[3] - bbox_sample[1]
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
    box_width = min(max(line_widths) + padding * 2, img.size[0])
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
    reward_scale: float = 1.0,
    intrinsic_enable: bool = False,
    intrinsic_scale: float = 0.0,
    intrinsic_w_curiosity: float = 1.0,
    intrinsic_w_novelty: float = 1.0,
    intrinsic_w_surprise: float = 1.0,
    # Secret path shaping (optional)
    secret_bonus: float = 0.0,
    secret_x_min: Optional[float] = None,
    secret_x_max: Optional[float] = None,
    secret_y_delta: Optional[float] = None,
    secret_y_mode: str = "down",
    # Secret path shaping (staged; recommended)
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
    """
    錄製遊戲影片。如果提供了 env 則使用該環境，否則創建新環境。
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{prefix}.mp4")

        close_env = False
        if env is None:
            env = make_base_env(
                game,
                state,
                preprocess_mode=preprocess_mode,
                timm_model_name=timm_model_name,
                reward_scale=reward_scale,
                # legacy
                secret_bonus=secret_bonus,
                secret_x_min=secret_x_min,
                secret_x_max=secret_x_max,
                secret_y_delta=secret_y_delta,
                secret_y_mode=secret_y_mode,
                # staged
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
                intrinsic_enable=intrinsic_enable,
                intrinsic_scale=intrinsic_scale,
                intrinsic_w_curiosity=intrinsic_w_curiosity,
                intrinsic_w_novelty=intrinsic_w_novelty,
                intrinsic_w_surprise=intrinsic_w_surprise,
            )
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