import torch
import torch.nn as nn
from typing import Tuple

class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        beta = 10.0
        sigma = torch.sigmoid(beta * x)
        return grad_output * beta * sigma * (1.0 - sigma)

spike_fn = SurrogateHeaviside.apply


class SimpleLIF(nn.Module):
    """
    Minimal discrete-time LIF-like cell.
    """
    def __init__(self, tau: float = 2.0, v_th: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        self.v_th = float(v_th)

    def forward(self, x_t: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = v + (-v + x_t) / self.tau
        # surrogate spike: forward=hard step, backward=smooth
        spk = spike_fn(v - self.v_th).to(x_t.dtype)
        v = v * (1.0 - spk)  # reset
        return spk, v
    
class SNN_DVS(nn.Module):
    def __init__(
        self,
        time_steps: int,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        tau: float = 2.0,
        tau_out: float = 2.0,
    ):
        super().__init__()
        self.T = int(time_steps)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.lif_h = SimpleLIF(tau=tau)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.lif_o = SimpleLIF(tau=tau_out, v_th=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 2, H, W] from DVSGesture loader
        returns output spikes: [T, B, C]
        """
        B, T, C, H, W = x.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        x = x.float()
        x = x.view(B, T, -1)  # [B, T, N]
        v_h = torch.zeros((B, self.fc_in.out_features), device=x.device, dtype=x.dtype)
        v_o = torch.zeros((B, self.fc_out.out_features), device=x.device, dtype=x.dtype)

        spk_out_all = []

        for t in range(T):
            h = self.fc_in(x[:, t])
            spk_h, v_h = self.lif_h(h, v_h)

            o = self.fc_out(spk_h)
            spk_o, v_o = self.lif_o(o, v_o)

            spk_out_all.append(spk_o)

        spk_out = torch.stack(spk_out_all, dim=0)  # [T, B, num_classes]
        return spk_out