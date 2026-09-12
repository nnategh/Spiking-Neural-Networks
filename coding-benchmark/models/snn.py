import torch
import torch.nn as nn

from models.lif import LIFLayer
from encoders.rate import rate_encode
from encoders.ttfs import ttfs_encode
from encoders.phase import phase_encode


class SNNClassifier(nn.Module):
    """
    SNN classifier for static frame-based datasets such as EMNIST and CIFAR-10.

    Pipeline:
        image [B,C,H,W]
        -> encoder
        -> input spike sequence [T,B,N]
        -> LIF hidden layer
        -> linear output layer
        -> output current sequence [T,B,num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        time_steps: int = 32,
        hidden_dim: int = 256,
        coding: str = "rate",
        tau_hidden: float = 2.0,
        tau_out: float = 2.0,
        v_th: float = 0.1,
        encoding_args: dict = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.coding = coding.lower()
        self.tau_out = tau_out
        self.encoding_args = encoding_args or {}

        self.lif = LIFLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            tau=tau_hidden,
            v_th=v_th,
        )

        self.readout_layer = nn.Linear(hidden_dim, num_classes)

    def encode(self, x: torch.Tensor):
        if self.coding == "rate":
            return rate_encode(
                x,
                time_steps=self.time_steps,
                **self.encoding_args,
            )

        if self.coding == "ttfs":
            return ttfs_encode(
                x,
                time_steps=self.time_steps,
                **self.encoding_args,
            )

        if self.coding == "phase":
            return phase_encode(
                x,
                time_steps=self.time_steps,
                **self.encoding_args,
            )

        raise ValueError(f"Unknown coding scheme: {self.coding}")

    def forward(self, x: torch.Tensor, return_seq: bool = False):
        encoded = self.encode(x)

        phase_index = None
        if isinstance(encoded, tuple):
            input_spikes, phase_index = encoded
        else:
            input_spikes = encoded

        self.last_input_spike_count = input_spikes.detach().sum()

        T, B, _ = input_spikes.shape

        v_hidden = None
        v_out = torch.zeros(B, self.num_classes, device=x.device)

        hidden_spike_count = 0.0
        output_seq = []

        for t in range(T):
            hidden_spikes, v_hidden = self.lif(input_spikes[t], v_hidden)

            hidden_spike_count += hidden_spikes.detach().sum()

            current = self.readout_layer(hidden_spikes)
            v_out = v_out + (current - v_out) / self.tau_out

            if return_seq:
                output_seq.append(v_out)

        if return_seq:
            output_seq = torch.stack(output_seq, dim=0)  # [T,B,num_classes]

            if phase_index is not None:
                return v_out, output_seq, hidden_spike_count, phase_index

            return v_out, output_seq, hidden_spike_count

        return v_out