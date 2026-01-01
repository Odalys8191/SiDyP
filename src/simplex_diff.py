'''
Simplex diffusion model
Modified from https://github.com/allenai/tess-diffusion
'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from diffusers.utils import BaseOutput

from simplex_utils import convert_to_simplex, logits_projection, nested_detach


"""DDPM scheduler for the simplex diffusion model."""

from diffusers import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from dataclasses import dataclass
from typing import Union, Tuple, Optional
import torch
import numpy as np
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
import math
import pdb


@dataclass
class SimplexDDPMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.
    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        projected_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            The projected logits sample (x_{0}) based on the model output from the current timestep.
    """

    prev_sample: torch.FloatTensor
    projected_logits: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, device, max_beta=0.999, improved_ddpm=False):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].
    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.
    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def default_alpha_bar(time_step):
        return math.cos((time_step + 1e-4) / (1 + 1e-4) * math.pi / 2) ** 2

    if improved_ddpm:
        # Implements eqn. 17 in https://arxiv.org/pdf/2102.09672.pdf.
        alpha_bar = lambda x: (default_alpha_bar(x) / default_alpha_bar(0.0))
        alphas_cumprod = []
    else:
        alpha_bar = default_alpha_bar
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        alpha_bar_t1 = alpha_bar(t1)
        betas.append(min(1 - alpha_bar(t2) / alpha_bar_t1, max_beta))
        if improved_ddpm:
            alphas_cumprod.append(alpha_bar_t1)
    # TODO(rabeeh): maybe this cause memory issue.
    betas = torch.tensor(betas, dtype=torch.float32, device=device)
    if improved_ddpm:
        return betas, torch.tensor(alphas_cumprod, dtype=torch.torch.float32, device=device)
    return betas


class SimplexDDPMScheduler(DDPMScheduler):
    @register_to_config
    def __init__(
        self,
        device,
        simplex_value: float,
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = False,
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32, device=device)
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps, device=device)
        elif beta_schedule == "squaredcos_improved_ddpm":
            self.betas, self.alphas_cumprod = betas_for_alpha_bar(num_train_timesteps, device=device, improved_ddpm=True)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps, device=device)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        if beta_schedule == "squaredcos_improved_ddpm":
            self.alphas = None
        else:
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.one = torch.tensor(1.0, device=device)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        # TODO(rabeeh): if memory issue, we can not add this to GPU and convert them iteratively.
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy()).to(device=device)

        self.variance_type = variance_type

    def step(
        self,
        projected_logits: torch.FloatTensor,
        timestep: int,
        noise: torch.FloatTensor,
        generator=None,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            projected_logits (`torch.FloatTensor`): projected logits from the diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            noise (`torch.FloatTensor`): a random noise with simplex_value standard deviation.
            generator: random number generator.
        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] resulted values.
        """
        t = timestep

        # 1. compute alphas, betas
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            projected_logits = torch.clamp(projected_logits, -1, 1)

        # See algorithm 2 in Figure 3 in https://arxiv.org/pdf/2210.17432.pdf.
        predicted_logits_coeff = alpha_prod_t_prev ** (0.5)
        noise_coeff = (1 - alpha_prod_t_prev) ** (0.5)
        pred_prev_sample = predicted_logits_coeff * projected_logits + noise_coeff * noise

        return SimplexDDPMSchedulerOutput(prev_sample=pred_prev_sample, projected_logits=projected_logits)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        # timesteps = timesteps.to(original_samples.device)

        alphas_cumprod_timesteps = self.alphas_cumprod[timesteps].view(-1, 1)
        sqrt_alpha_prod = alphas_cumprod_timesteps**0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod_timesteps) ** 0.5
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples



class SimplexDiffusionPipelineOutput(BaseOutput):
    simplex: np.ndarray
    logits: np.ndarray
    # loss: np.ndarray




class SimplexDiffusion(nn.Module):
    def __init__(self, args, noise_scheduler, inference_noise_scheduler, num_classes=10, w_dim=128):
        super(SimplexDiffusion, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.simplex_value = args.simplex_value

        self._denoise_fn = ConditionalModel(
            self.args.train_timesteps, y_dim=self.num_classes, w_dim=w_dim,
            eps=20, guidance=True, x_dim=768).to(self.args.device)


        self.noise_scheduler = noise_scheduler
        self.inference_noise_scheduler = inference_noise_scheduler
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.timestep_embed = nn.Linear(1, 768, bias=True)
        

    
    def forward_t(self, y, w, x_embed, noisy_y):
        # 检查y是否是连续标签分布
        if y.dim() > 1 and y.size(-1) == self.num_classes and y.dtype in [torch.float32, torch.float64]:
            # 连续标签分布，直接使用
            simplex_y = convert_to_simplex(y, self.simplex_value, self.num_classes)
            noisy_labels = torch.argmax(y, dim=-1)  # 用于计算损失
        else:
            # 离散标签，转换为one-hot
            noisy_labels = torch.argmax(y, dim=-1) if y.dim() > 1 else y
            simplex_y = convert_to_simplex(noisy_labels, self.simplex_value, self.num_classes)
        
        # 处理noisy_y
        if noisy_y.dim() > 1 and noisy_y.size(-1) == self.num_classes:
            # 连续标签分布，直接使用
            simplex_noisy_y = convert_to_simplex(noisy_y, self.simplex_value, self.num_classes)
        else:
            # 离散标签，转换为one-hot
            simplex_noisy_y = convert_to_simplex(noisy_y.long(), self.simplex_value, self.num_classes)
        
        noise = self.simplex_value * torch.randn(simplex_y.shape, device=simplex_y.device, dtype=simplex_y.dtype)

        batch_size = simplex_y.size(0)
        timesteps = torch.randint(0, len(self.noise_scheduler), (batch_size,), device=simplex_y.device, dtype=torch.int64)
        noisy_simplex = self.noise_scheduler.add_noise(simplex_y, noise, timesteps)
        
        # if self.args.self_condition:
        #     if previous_pred is None:
        #         previous_pred = torch.zeros_like(noisy_simplex, device=noisy_simplex.device)
        #     noisy_simplex = mix_values_based_on_self_condition(self.args.self_condition_type, noisy_simplex, previous_pred)

        logits = self._denoise_fn(noisy_simplex, timesteps, w, x_embed, simplex_noisy_y)

        loss = self.loss(logits, noisy_labels)

        return loss, logits
    

    def reverse_t(self, y, w, x_embed, generator):
        simplex_shape = (w.size(0), self.num_classes)
        simplex = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.args.device)
        
        # 处理y，支持连续标签分布
        if y.dim() > 1 and y.size(-1) == self.num_classes:
            # 连续标签分布，直接使用
            simplex_y = convert_to_simplex(y, self.simplex_value, self.num_classes)
        else:
            # 离散标签，转换为one-hot
            simplex_y = convert_to_simplex(y.long(), self.simplex_value, self.num_classes)

        logits_projection_fct = lambda x: logits_projection(
            x, self.simplex_value
        )


        for t in self.inference_noise_scheduler.timesteps:
            logits = self._denoise_fn(simplex, t, w, x_embed, simplex_y)

            projected_logits = logits_projection_fct(logits)

            # compute previous logits: x_t -> x_t-1
            noise = self.simplex_value * torch.randn(simplex_shape, generator=generator, device=self.args.device)
            simplex = self.inference_noise_scheduler.step(projected_logits, t, noise, generator=generator).prev_sample

        logits = nested_detach(logits)
        simplex = nested_detach(simplex)
        return SimplexDiffusionPipelineOutput(simplex=simplex, logits=logits)



class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out

class ConditionalModel(nn.Module):
    def __init__(self, n_steps, y_dim=10, w_dim=128, eps=20, guidance=True, x_dim=768):
        super(ConditionalModel, self).__init__()
        n_steps = n_steps + 1
        self.y_dim = y_dim
        self.guidance = guidance
        self.norm = nn.BatchNorm1d(w_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim + y_dim, w_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, w_dim, n_steps)
            
        self.unetnorm1 = nn.BatchNorm1d(w_dim)
        self.lin2 = ConditionalLinear(w_dim, w_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(w_dim)
        self.lin3 = ConditionalLinear(w_dim, w_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(w_dim)
        self.lin4 = nn.Linear(w_dim, y_dim)

    def forward(self, y, t, w, x_embed, noisy_y):
        w = self.norm(w)
        if self.guidance:
            y = torch.cat([y, noisy_y], dim=-1)

        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = y * w
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)
