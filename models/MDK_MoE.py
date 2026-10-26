import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from RevIN.RevIN import RevIN
import scipy.ndimage
import matplotlib.pyplot as plt
import os

from layers.Embed import PositionalEmbedding

import math
from typing import Optional

import torch
import torch.nn as nn

import numpy as np
from scipy.fftpack import fft, ifft


class PatchEmbed(nn.Module):
    def __init__(self, seq_len=96, pos=True):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(self.seq_len, self.seq_len)
        pos_emb_theta = 10000
        self.pos = pos
        if self.pos:
            self.pe = PositionalEmbedding(self.seq_len, pos_emb_theta)
    
    def forward(self, x):
        x = self.proj(x)
        if self.pos:
            x += self.pe(x) 
        return x

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class WaveletDecomposer(nn.Module):
    def __init__(self, kernel_size=21, init_scale=24.0, init_f0=0.04):
        super().__init__()
        self.kernel_size = kernel_size
        self.scale = init_scale
        self.f0 = init_f0

    def create_morlet_kernel(self, device):
        t = torch.linspace(-self.kernel_size // 2, self.kernel_size // 2, steps=self.kernel_size, device=device)
        
        s = self.scale
        f = self.f0
        
        wavelet = torch.exp(-t**2 / (2 * s**2)) * torch.cos(2 * math.pi * f * t)
        kernel = wavelet.view(1, 1, self.kernel_size)
        kernel = kernel / torch.norm(kernel, p=2)
        return kernel

    def forward(self, x):
        B, L, C = x.shape
        x_ = x.permute(0, 2, 1)  # [B, C, L]

        kernel = self.create_morlet_kernel(x.device)
        kernel = kernel.repeat(C, 1, 1)  

        pad = self.kernel_size // 2
        low_part = F.conv1d(x_, kernel, padding=pad, groups=C)
        high_part = x_ - low_part

        return low_part.permute(0, 2, 1), high_part.permute(0, 2, 1)

class FrequencyDecomposer(nn.Module):
    def __init__(self, threshold_ratio=0.1):
        super().__init__()

        self.threshold_ratio = threshold_ratio

    def batch_fft(self, x):
        B, T, D = x.shape
        x_freq = torch.fft.fft(x, dim=1)  

        threshold = int(T * self.threshold_ratio)
        mask = torch.zeros(T, device=x.device)
        mask[:threshold] = 1
        mask[-threshold:] = 1
        mask = mask.view(1, T, 1) 

        low_freq = x_freq * mask
        low_part = torch.fft.ifft(low_freq, dim=1).real
        high_part = x.real - low_part

        return low_part, high_part

class FrequencyContrastiveLoss(nn.Module):
    def __init__(self, hidden_dim=128, temperature=0.2, top_k=None, eps=1e-8, neighbor_positive=False,C=1):
        super(FrequencyContrastiveLoss, self).__init__()
        self.hidden_dim = hidden_dim
        self.base_temperature = temperature
        self.eps = eps
        self.neighbor_positive = neighbor_positive
        self.proj = None  
        self.C =C
        if self.C <= 7:
            self.top_k = None
        else:
            self.top_k = min(20, C//2)   

    def sim_con(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        tau = self.base_temperature * 2 if self.C <= 16 else self.base_temperature
        return torch.matmul(z1, z2.transpose(1, 2)) / tau

    def calc_lower_bound(self, z1, z2, pos):
        sim_matrix = torch.exp(self.sim_con(z1, z2))
        sim_matrix_1 = sim_matrix / (torch.sum(sim_matrix, dim=2, keepdim=True) + self.eps)
        sim_matrix_2 = sim_matrix.transpose(1, 2) / (torch.sum(sim_matrix.transpose(1, 2), dim=2, keepdim=True) + self.eps)

        # Soft top-k mask
        if self.top_k is not None and self.top_k < sim_matrix.size(-1):
            weights_1 = torch.zeros_like(sim_matrix_1)
            weights_2 = torch.zeros_like(sim_matrix_2)
            top_vals_1, top_idx_1 = torch.topk(sim_matrix_1, self.top_k, dim=-1)
            top_vals_2, top_idx_2 = torch.topk(sim_matrix_2, self.top_k, dim=-1)
            weights_1.scatter_(-1, top_idx_1, top_vals_1)
            weights_2.scatter_(-1, top_idx_2, top_vals_2)
            sim_matrix_1 = sim_matrix_1 * (weights_1 / (top_vals_1.max(dim=-1, keepdim=True)[0] + self.eps))
            sim_matrix_2 = sim_matrix_2 * (weights_2 / (top_vals_2.max(dim=-1, keepdim=True)[0] + self.eps))

        lori_1 = -torch.log((sim_matrix_1 * pos).sum(dim=-1) + self.eps).mean()
        lori_2 = -torch.log((sim_matrix_2 * pos).sum(dim=-1) + self.eps).mean()
        return (lori_1 + lori_2) / 2

    def forward(self, X_f_low, X_w_low, X_f_high, X_w_high, X_seasonal, X_trend):
        B, C, L = X_f_low.shape
        hidden_dim = self.hidden_dim
        z_f_low   = X_f_low
        z_w_low   = X_w_low
        z_f_high  = X_f_high
        z_w_high  = X_w_high
        z_season  = X_seasonal
        z_trend   = X_trend

        pos = torch.eye(C, device=X_f_low.device).unsqueeze(0)
        if self.neighbor_positive and C > 2:
            pos_shift = torch.eye(C, device=X_f_low.device).roll(1, dims=1)
            pos = torch.clamp(pos + 0.5 * pos_shift, max=1.0)

        loss_low = self.calc_lower_bound(z_f_low, z_w_low, pos)

        loss_high = self.calc_lower_bound(z_f_high, z_w_high, pos)

        loss_season_f = self.calc_lower_bound(z_season, z_f_low, pos)
        loss_season_w = self.calc_lower_bound(z_season, z_w_low, pos)


        loss_trend_f = self.calc_lower_bound(z_trend, z_f_high, pos)
        loss_trend_w = self.calc_lower_bound(z_trend, z_w_high, pos)

        total_loss = (loss_low + loss_high + loss_season_f + loss_season_w + loss_trend_f + loss_trend_w) / 6
        return total_loss

@torch.no_grad()
def _power_iteration(A, iters=10):
    C = A.shape[0]
    u = F.normalize(torch.randn(C, device=A.device), dim=0, eps=1e-12)
    for _ in range(iters):
        v = F.normalize(A.T @ u, dim=0, eps=1e-12)
        u = F.normalize(A @ v, dim=0, eps=1e-12)
    sigma = u @ (A @ v)
    return sigma.abs().clamp(min=1e-12)

def _scale_to_spectral_radius(A, alpha=0.99, iters=10):
    sigma = _power_iteration(A, iters=iters)
    scale = (alpha / sigma).clamp(max=1.0)
    return A * scale

class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=None, noise_epsilon=1e-5):
        super(TopKGating, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k if top_k is not None else max(1, num_experts // 2)
        self.gate = nn.Linear(input_dim, num_experts)
        self.noise_epsilon = noise_epsilon
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

    def decompostion_tp(self, x, alpha=10.0):
        # x: [B, C, num_experts]
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1, dim=-1)
        kth_largest_mat = kth_largest_val.unsqueeze(-1).expand(-1, -1, self.num_experts)
        mask = x < kth_largest_mat
        x_soft = self.softmax(x)
        output = torch.zeros_like(x)
        output[mask] = alpha * torch.log(x_soft[mask] + 1.0)
        output[~mask] = alpha * (torch.exp(x_soft[~mask]) - 1.0)
        return output

    def forward(self, x):
        logits = self.gate(x)  # [B, C, num_experts]

        if self.training:
            raw_noise_stddev = logits @ self.w_noise  # [B, C, num_experts]
            noise_stddev = (self.softplus(raw_noise_stddev) + self.noise_epsilon)
            logits = logits + (torch.randn_like(logits) * noise_stddev)

        logits = self.decompostion_tp(logits)   
        gates = self.softmax(logits)            # [B, C, num_experts]
        return gates


class StableRecursiveKalmanFusion(nn.Module):
    def __init__(self, C: int,
                 D_feat: int,
                 low_dim_threshold: int = 8,
                 a_max: float = 1.2,
                 alpha: float = 0.95,
                 k_min: float = 0.05,
                 k_max: float = 0.95,
                 use_moe_gating: bool = True,
                 gating_topk: Optional[int] = None):
        super().__init__()
        assert 0.0 < alpha < 1.0
        assert 0.0 < k_min < k_max < 1.0

        self.C = C
        self.D_feat = D_feat
        self.low_dim_threshold = low_dim_threshold
        self.a_max = a_max
        self.alpha = alpha
        self.k_min = k_min
        self.k_max = k_max
        self.use_moe_gating = use_moe_gating
        self.num_inputs = 6  
        self.moe_weight_sum = torch.zeros(self.num_inputs)
        self.moe_weight_count = torch.tensor(0.0)

        self.K_raw = nn.Parameter(torch.zeros(1, C))

        if C <= self.low_dim_threshold:
            self.A_diag_raw = nn.Parameter(torch.zeros(C))
        else:
            self.A_raw = nn.Parameter(torch.eye(C))

        if self.use_moe_gating:
            self.gater = TopKGating(input_dim=D_feat, num_experts=self.num_inputs, top_k=gating_topk)
        else:
            self.obs_logits = nn.Parameter(torch.zeros(self.num_inputs, C))

    @torch.no_grad()
    def _accumulate_moe_weights(self, gates):
        if self.moe_weight_sum.device != gates.device:
            self.moe_weight_sum = self.moe_weight_sum.to(gates.device)
            self.moe_weight_count = self.moe_weight_count.to(gates.device)
        self.moe_weight_sum += gates.mean(dim=(0, 1)).detach()
        self.moe_weight_count += 1.0

    @torch.no_grad()
    def get_moe_weight_stats(self):
        if float(self.moe_weight_count.item()) <= 0:
            return None
        return (self.moe_weight_sum / self.moe_weight_count).detach().clone()

    def _get_K(self):
        return self.k_min + (self.k_max - self.k_min) * torch.sigmoid(self.K_raw)  # [1, C]

    def _get_A(self):
        C = self.C
        if C <= self.low_dim_threshold:
            a = self.a_max * torch.tanh(self.A_diag_raw)  # [C]
            return torch.diag(a)                          # [C, C]
        else:
            return _scale_to_spectral_radius(self.A_raw, alpha=self.alpha)

    def _get_obs_weights_simple(self):
        weights = F.softmax(self.obs_logits, dim=0)  # [6, C]
        w_f_low = weights[0].view(1, self.C, 1)
        w_w_low = weights[1].view(1, self.C, 1)
        w_f_high = weights[2].view(1, self.C, 1)
        w_w_high = weights[3].view(1, self.C, 1)
        w_s_low = weights[4].view(1, self.C, 1)
        w_s_high = weights[5].view(1, self.C, 1)
        return w_f_low, w_w_low, w_f_high, w_w_high, w_s_low, w_s_high

    def load_balancing_loss_from_gates(self, gates):
        # gates: [B, C, 6]
        mean_gates = gates.mean(dim=(0,1))  # [6]
        loss = 6.0 * ((mean_gates - (1.0/6.0))**2).sum()
        return loss

    def forward(self, X_f_low, X_w_low, X_f_high, X_w_high, X_s_low, X_s_high, return_gates=False):
        inputs = [X_f_low, X_w_low, X_f_high, X_w_high, X_s_low, X_s_high]
        for x in inputs:
            B, C, D = x.shape
            assert C == self.C and D == self.D_feat, "none"

        A = self._get_A()  # [C, C]
        K = self._get_K()  # [1, C]

        if self.use_moe_gating:
            x_avg = torch.stack(inputs).mean(dim=0)  # [B, C, D]
            gates = self.gater(x_avg)                # [B, C, 6]
            if self.training:
                self._accumulate_moe_weights(gates)
            lb_loss = self.load_balancing_loss_from_gates(gates)
            w_f_low = gates[..., 0].unsqueeze(-1)    # [B, C, 1]
            w_w_low = gates[..., 1].unsqueeze(-1)
            w_f_high = gates[..., 2].unsqueeze(-1)
            w_w_high = gates[..., 3].unsqueeze(-1)
            w_s_low = gates[..., 4].unsqueeze(-1)
            w_s_high = gates[..., 5].unsqueeze(-1)
        else:
            w_f_low, w_w_low, w_f_high, w_w_high, w_s_low, w_s_high = self._get_obs_weights_simple()
            B = X_f_low.shape[0]
            w_f_low = w_f_low.expand(B, -1, -1)
            w_w_low = w_w_low.expand(B, -1, -1)
            w_f_high = w_f_high.expand(B, -1, -1)
            w_w_high = w_w_high.expand(B, -1, -1)
            w_s_low = w_s_low.expand(B, -1, -1)
            w_s_high = w_s_high.expand(B, -1, -1)
            lb_loss = torch.tensor(0.0, device=X_f_low.device)

        z = (w_f_low * X_f_low + 
             w_w_low * X_w_low + 
             w_f_high * X_f_high + 
             w_w_high * X_w_high + 
             w_s_low * X_s_low + 
             w_s_high * X_s_high)  # [B, C, D]

        x_prev = torch.stack(inputs).mean(dim=0)  # [B, C, D]

        x_pred = torch.einsum('bid,ji->bjd', x_prev, A)   # [B, C, D]

        # K shape [1, C] -> [B, C, D] broadcast
        K_broadcast = K.unsqueeze(-1).expand(B, -1, D)    # [B, C, D]

        inov = z - x_pred                                 # [B, C, D]
        x_new = x_pred + K_broadcast * inov               # [B, C, D]

        if return_gates:
            if self.use_moe_gating:
                return x_new, lb_loss, gates  # gates per-sample [B,C,6]
            else:
                gates_simple = torch.cat(
                    [w_f_low, w_w_low, w_f_high, w_w_high, w_s_low, w_s_high], 
                    dim=-1
                )  # [B,C,6]
                return x_new, lb_loss, gates_simple
        else:
            return x_new, lb_loss




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.level = configs.level
        self.threshold_ratio = configs.threshold_ratio
        self.moving_avg = configs.moving_avg
        self.dropout =  nn.Dropout(configs.dropout) 
        self.pos = configs.pos
        self.temperature = configs.temperature
        self.gating_topk = configs.gating_topk

        self.low_dim_threshold = 8
        self.is_low_dim = self.enc_in <= self.low_dim_threshold
        
        if self.enc_in <= 7 and self.pred_len <=96 : 
            self.threshold_ratio = max(0.1, self.threshold_ratio)
            self.init_scale = 12.0
        else:
            self.threshold_ratio = self.threshold_ratio
            self.init_scale = 24.0

        self.FFTdecomposer = FrequencyDecomposer(self.threshold_ratio)
        self.WaveletDecomposer = WaveletDecomposer(self.level, self.init_scale, init_f0=0.1)
        self.AvgDecomposer = moving_avg(kernel_size=self.moving_avg,stride=1)
      
        self.kalman = StableRecursiveKalmanFusion(self.enc_in,D_feat=self.d_model,low_dim_threshold=8,a_max=1.2,alpha=0.5,k_min=0.05,k_max=0.95,use_moe_gating= True,gating_topk = self.gating_topk)

        self.embed = PatchEmbed(self.seq_len, self.pos)

        self.mi_loss_fn = FrequencyContrastiveLoss(hidden_dim=self.d_model, temperature=self.temperature, top_k=5, neighbor_positive=True,C=self.enc_in)

        self.Linear_f_low = nn.Linear(self.seq_len, self.d_model)  
        self.Linear_w_low = nn.Linear(self.seq_len, self.d_model)  
        self.Linear_s_low = nn.Linear(self.seq_len, self.d_model)  
        self.Linear_f_high = nn.Linear(self.seq_len, self.d_model)  
        self.Linear_w_high = nn.Linear(self.seq_len, self.d_model)  
        self.Linear_s_high = nn.Linear(self.seq_len, self.d_model) 
        self.Linear_x = nn.Linear(self.seq_len, self.d_model)

        self.normalize_layers = RevIN(configs.enc_in, affine=False)

        self.MLP1 = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, self.pred_len)
        )

    def forward(self, x):
        x = self.normalize_layers(x, 'norm')
        x = self.embed(x.permute(0, 2, 1))  # [B, C, D]
        x = x.permute(0, 2, 1)              # [B, C, D]-> [B, D, C] 
        X_f_low, X_f_high = self.FFTdecomposer.batch_fft(x)
        X_w_low, X_w_high = self.WaveletDecomposer(x)
        X_Trend = self.AvgDecomposer(x)
        X_Seasonal = x - X_Trend   

        x, X_f_low, X_f_high, X_w_low, X_w_high, X_Seasonal, X_Trend = x.permute(0, 2, 1), X_f_low.permute(0, 2, 1), X_f_high.permute(0, 2, 1), \
        X_w_low.permute(0, 2, 1), X_w_high.permute(0, 2, 1),  X_Seasonal.permute(0, 2, 1), X_Trend.permute(0, 2, 1)

        X_f_low = self.Linear_f_low(X_f_low)
        X_f_high = self.Linear_f_high(X_f_high)
        X_w_low = self.Linear_w_low(X_w_low)
        X_w_high = self.Linear_w_high(X_w_high)
        X_s_low = self.Linear_s_low(X_Trend)
        X_s_high = self.Linear_s_high(X_Seasonal)
        x =self.Linear_x(x)

        loss_mi = self.mi_loss_fn(X_f_low, X_w_low, X_f_high, X_w_high, X_s_low, X_s_high)

        fused_X, loss_moe = self.kalman(X_f_low, X_w_low, X_f_high, X_w_high, X_s_low, X_s_high,False)
        fused_X = self.dropout(fused_X+x)
        x_out = self.MLP1(fused_X)
          
        x_out = self.normalize_layers(x_out.permute(0, 2, 1), 'denorm')
        return loss_moe, loss_mi, x_out

    def get_moe_weight_stats(self):
        if hasattr(self, 'kalman') and hasattr(self.kalman, 'get_moe_weight_stats'):
            return self.kalman.get_moe_weight_stats()
        return None
    