from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance
class VisionBackboneExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, backbone_name: str = "resnet18"):
        channels, height, width = observation_space.shape
        # Temporary value, will be updated after backbone is created
        super().__init__(observation_space, features_dim=1)
        self.backbone = timm.create_model(
            backbone_name,  # e.g. resnet18, convnext_small.dinov3_lvd1689m, convnext_base.clip_laion2b
            pretrained=True,
            in_chans=channels,
            features_only=True,
            out_indices=[-1],
        )
        self.output_dim = self.backbone.feature_info[-1]["num_chs"]

        self._features_dim = self.output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        features = self.backbone(x)[0]
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return pooled


class ScalarAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 32, num_heads: int = 4, num_layers: int = 2, output_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        # 將每個 scalar 特徵視為一個 token，投影到 embed_dim
        self.feature_embedding = nn.Linear(1, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * embed_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, input_dim)
        # Reshape to (Batch, input_dim, 1)
        x = x.unsqueeze(-1)
        # Embedding: (Batch, input_dim, embed_dim)
        x = self.feature_embedding(x)
        # Transformer Encoder
        x = self.transformer_encoder(x)
        # Output projection
        return self.output_net(x)


class VisionScalarExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, backbone_name: str = "resnet18"):
        assert isinstance(observation_space, spaces.Dict), "VisionScalarExtractor expects a Dict observation space"
        image_space = observation_space["image"]
        scalar_space = observation_space["scalars"]
        super().__init__(observation_space, features_dim=1)
        self.image_extractor = VisionBackboneExtractor(image_space, backbone_name=backbone_name)
        scalar_dim = int(np.prod(scalar_space.shape))
        ######################################
        # 定義簡單的 MLP 處理 scalar 輸入 (步數、時間等資訊)
        # 輸入維度: scalar_dim (2: step_feat, time_feat)
        # 輸出維度: 64，與 image features 拼接
        ######################################
        # 使用 Attention Encoder 處理 scalar 輸入
        self.scalar_net = ScalarAttentionEncoder(input_dim=scalar_dim, output_dim=64)
        ######################################
        ######################################
        self._features_dim = self.image_extractor.features_dim + 64

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        image_feats = self.image_extractor(observations["image"])
        scalar_feats = self.scalar_net(observations["scalars"])
        return torch.cat([image_feats, scalar_feats], dim=1)


class VisionBackbonePolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["features_extractor_class"] = VisionScalarExtractor
        super().__init__(*args, **kwargs)
        
class CustomPPO(OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        kl_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cuda:0",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if normalize_advantage:
            assert batch_size > 1, "Cannot normalize advantage with batch_size=1"
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.kl_coef = kl_coef
    def train(self):
        ##################################
        # Every RolloutBuffer sample contains:
        # class RolloutBufferSamples(NamedTuple):
        #   observations: th.Tensor
        #   actions: th.Tensor
        #   old_values: th.Tensor
        #   old_log_prob: th.Tensor
        #   advantages: th.Tensor
        #   returns: th.Tensor
        ###################################
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        total_losses = []
        grad_norms_pre_clip = []
        continue_training = True
        for epoch in range(self.n_epochs):
            epoch_losses = []
            epoch_grad_norms = []
            epoch_pg_losses = []
            epoch_value_losses = []
            epoch_entropy_losses = []
            epoch_clip_fractions = []
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss
                ##############################
                # PPO Clipped Surrogate Loss 實作
                # 1. 計算 unclipped 的 surrogate: ratio * advantages
                # 2. 計算 clipped 的 surrogate: clamp(ratio, 1-ε, 1+ε) * advantages
                # 3. 取兩者較小值，並取負號（因為 optimizer 做 minimize）
                ##############################
                # Unclipped surrogate objective
                policy_loss_1 = advantages * ratio
                # Clipped surrogate objective
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                # 取較小值並取負號 (gradient ascent -> minimize negative)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                ##############################
                # Logging
                pg_losses.append(policy_loss.item())
                epoch_pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                epoch_clip_fractions.append(clip_fraction)
                values_pred = values
                
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                epoch_value_losses.append(value_loss.item())
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                epoch_entropy_losses.append(entropy_loss.item())
                
                ################################
                # 計算 KL Divergence 與總損失
                # KL(old || new) ≈ (old_log_prob - new_log_prob) 的期望值
                # 當 kl_coef > 0 時，加入 KL penalty 穩定訓練
                ################################
                if self.kl_coef == 0:
                    # 不使用 KL penalty，直接記錄 KL 用於監控
                    with torch.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                        approx_kl_divs.append(approx_kl)
                    # Total Loss = Policy Loss + Value Loss * vf_coef + Entropy Loss * ent_coef
                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                else:
                    # 計算 KL divergence 作為 penalty term
                    log_ratio = log_prob - rollout_data.old_log_prob
                    # 使用更穩定的 KL 近似: E[(exp(r) - 1) - r]
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                    approx_kl_divs.append(approx_kl.item())
                    # Total Loss 加入 KL penalty
                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss + self.kl_coef * approx_kl
                ################################
                ################################
                self.policy.optimizer.zero_grad()
                loss.backward()

                # === Debug: Gradient norm (pre-clip) ===
                total_norm = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms_pre_clip.append(total_norm)
                epoch_grad_norms.append(total_norm)
                # ======================================

                total_losses.append(loss.item())
                epoch_losses.append(loss.item())

                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            # Print once per epoch-update (not per minibatch)
            if self._n_updates % 10 == 0:
                def _mean(xs):
                    return float(np.mean(xs)) if len(xs) else 0.0

                print(
                    "[DEBUG] Update "
                    f"{self._n_updates} | "
                    f"Loss(mean)={_mean(epoch_losses):.4f} | "
                    f"PG={_mean(epoch_pg_losses):.4f} | "
                    f"VF={_mean(epoch_value_losses):.4f} | "
                    f"Ent={_mean(epoch_entropy_losses):.4f} | "
                    f"KL={_mean(approx_kl_divs):.6f} | "
                    f"ClipFrac={_mean(epoch_clip_fractions):.3f} | "
                    f"GradNorm(preclip,mean)={_mean(epoch_grad_norms):.4f}"
                )

            self._n_updates += 1
            if not continue_training:
                break
        explained_variance_ = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Scale diagnostics (helps interpret big value loss/loss)
        returns = self.rollout_buffer.returns.flatten()
        values_buf = self.rollout_buffer.values.flatten()
        advantages_buf = self.rollout_buffer.advantages.flatten()

        self.logger.record("train/returns_mean", float(np.mean(returns)))
        self.logger.record("train/returns_std", float(np.std(returns)))
        self.logger.record("train/values_mean", float(np.mean(values_buf)))
        self.logger.record("train/values_std", float(np.std(values_buf)))
        self.logger.record("train/advantages_mean", float(np.mean(advantages_buf)))
        self.logger.record("train/advantages_std", float(np.std(advantages_buf)))
        self.logger.record("train/grad_norm_pre_clip", float(np.mean(grad_norms_pre_clip)) if len(grad_norms_pre_clip) else 0.0)
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", float(np.mean(total_losses)) if len(total_losses) else float(loss.item()))
        self.logger.record("train/explained_variance", explained_variance_)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", self.clip_range)

        
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MyPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = True,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
