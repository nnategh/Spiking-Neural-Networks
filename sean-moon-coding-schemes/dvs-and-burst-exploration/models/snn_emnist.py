import torch
import torch.nn as nn
from models.lif import LIFLayer
from encoding.ttfs import ttfs_encode_flat
from encoding.rate import rate_encode_flat
from encoding.phase import phase_encode_flat
from encoding.burst import burst_encode_flat

class SNN_EMNIST(nn.Module):
    def __init__(self, time_steps=32, hidden_dim=256, num_classes=47,
                coding="rate", tau_out=2.0, encoding_args = None):
        super().__init__()
        self.time_steps = time_steps
        self.coding = coding.lower()
        self.hidden_dim = hidden_dim
        self.tau_out = tau_out
        # scheme-specific encoding parameters live here
        self.encoding_args = encoding_args or {}

        self.lif = LIFLayer(28*28, hidden_dim)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_seq: bool = False):
        B = x.size(0)
        C = self.readout.out_features
        v_out = torch.zeros(B, C, device=x.device)
        v_hid = None

        # For metrics
        v_out_seq = []
        # total spikes over all timesteps & batch
        spike_count = 0.0 

        # Encoding stage
        if self.coding == "rate":
            seq = rate_encode_flat(x, self.time_steps, **self.encoding_args)
        
        elif self.coding == "ttfs":
            seq = ttfs_encode_flat(x, self.time_steps, **self.encoding_args)

        elif self.coding == "phase":
            seq = phase_encode_flat(x, self.time_steps, **self.encoding_args)

        elif self.coding == "burst":
            seq = burst_encode_flat(x,self.time_steps, **self.encoding_args)
        else:
            raise ValueError(f"Unknown coding scheme: {self.coding}")
        
        # SNN dynamics 
        for t in range(self.time_steps):
                spikes, v_hid = self.lif(seq[t], v_hid)

                spike_count += spikes.detach().sum()
                I = self.readout(spikes)
                v_out = v_out + (I - v_out) / self.tau_out
                if return_seq:
                    v_out_seq.append(v_out.detach())

        if return_seq:
                v_out_seq = torch.stack(v_out_seq, dim=0)  # [T, B, C]
                return v_out, v_out_seq, spike_count
        

        return v_out