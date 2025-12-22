import os
import argparse
from typing import Any, Optional, Tuple
import numpy as np
from custom_policy import CustomPPO
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from wrappers import make_base_env


def evaluate_policy(
    model: CustomPPO,
    game: str,
    state: str,
    n_episodes: int,
    max_steps: int,
    *,
    preprocess_mode: str = "fixed",
    timm_model_name: Optional[str] = None,
    reward_scale: float = 1.0,
    env_kwargs: Optional[dict[str, Any]] = None,
):
    env = make_base_env(
        game,
        state,
        preprocess_mode=preprocess_mode,
        timm_model_name=timm_model_name,
        reward_scale=reward_scale,
        **(env_kwargs or {}),
    )
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

    env.close()
    mean_ret = float(np.mean(returns)) if returns else 0.0
    best_ret = float(np.max(returns)) if returns else 0.0
    return mean_ret, best_ret


def _pair_lines(kvs: list[str], max_per_line: int = 2) -> list[str]:
    lines: list[str] = []
    i = 0
    while i < len(kvs):
        chunk = kvs[i : i + max_per_line]
        lines.append("  ".join(chunk))
        i += max_per_line
    return lines


def _format_kv(k: str, v: Any) -> str:
    if isinstance(v, float):
        return f"{k}={v:.3f}"
    return f"{k}={v}"


def _select_info_items(info: dict) -> list[str]:
    if not isinstance(info, dict) or not info:
        return []

    preferred = [
        "x_pos",
        "x_pos_delta",
        "time_left",
        "y_pos_raw",
        "y_pos_delta",
        "y_pos_delta_step",
        "lives",
        "score",
        "coins",
        "player_dir",
        "slope_type",
        "speed",
        "accel",
        "intrinsic_reward",
        "death",
    ]
    out: list[str] = []
    seen = set()
    for k in preferred:
        if k in info:
            out.append(_format_kv(k, info.get(k)))
            seen.add(k)
    # add a few remaining keys deterministically
    for k in sorted(info.keys()):
        if k in seen:
            continue
        if len(out) >= 10:
            break
        out.append(_format_kv(k, info.get(k)))
    return out


def _annotate_frame(
    frame: np.ndarray,
    cumulative_reward: float,
    last_reward: float,
    info: dict,
    font: ImageFont.ImageFont,
    bar_height: int = 64,
) -> np.ndarray:
    # Create a new canvas with a bottom bar (text area) so we don't cover the game view.
    h, w = int(frame.shape[0]), int(frame.shape[1])
    bar_h = max(32, int(bar_height))
    canvas = Image.new("RGB", (w, h + bar_h), color=(0, 0, 0))
    canvas.paste(Image.fromarray(frame), (0, 0))

    draw = ImageDraw.Draw(canvas)
    padding = 6
    # First line: rewards (two items max)
    lines = [f"reward={last_reward:.3f}  cum_reward={cumulative_reward:.3f}"]

    info_items = _select_info_items(info)
    lines.extend(_pair_lines(info_items, max_per_line=2))

    # Fit lines into bar area
    bbox_sample = draw.textbbox((0, 0), "Ag", font=font)
    line_h = bbox_sample[3] - bbox_sample[1]
    max_lines = max(1, (bar_h - padding * 2) // max(1, line_h + 2))
    lines = lines[:max_lines]

    y = h + padding
    for line in lines:
        # If a line is too long, trim it.
        while True:
            bbox = draw.textbbox((0, 0), line, font=font)
            if (bbox[2] - bbox[0]) <= (w - padding * 2) or len(line) <= 8:
                break
            line = line[:-4] + "..."
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_h + 2

    return np.array(canvas)


def record_video(
    model: CustomPPO,
    game: str,
    state: str,
    out_dir: str,
    video_len: int,
    prefix: str,
    *,
    preprocess_mode: str = "fixed",
    timm_model_name: Optional[str] = None,
    reward_scale: float = 1.0,
    env_kwargs: Optional[dict[str, Any]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.mp4")

    env = make_base_env(
        game,
        state,
        preprocess_mode=preprocess_mode,
        timm_model_name=timm_model_name,
        reward_scale=reward_scale,
        **(env_kwargs or {}),
    )
    fps = env.metadata.get("render_fps", 60)
    writer = imageio.get_writer(out_path, fps=fps)
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
    env.close()
    print(f"Saved video to {out_path}")