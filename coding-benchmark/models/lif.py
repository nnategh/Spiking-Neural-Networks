import torch
import torch.nn as nn


class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = (x > 0).float()
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sigma = torch.sigmoid(x)
        return grad_output * sigma * (1.0 - sigma)


spike_fn = SurrogateHeaviside.apply


class LIFLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, tau: float = 2.0, v_th: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.tau = tau
        self.v_th = v_th

    def forward(self, x: torch.Tensor, v=None):
        if v is None:
            v = torch.zeros(x.size(0), self.fc.out_features, device=x.device)

        I = self.fc(x)
        v = v + (I - v) / self.tau
        spikes = spike_fn(v - self.v_th)
        v = v * (1.0 - spikes)

        return spikes, v