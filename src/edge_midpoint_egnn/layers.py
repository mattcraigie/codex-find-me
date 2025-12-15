from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2
    )


def rotate_vectors(vectors: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    rot = rotation_matrix(theta).transpose(-1, -2)
    return torch.matmul(vectors, rot)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, *src.shape[1:]), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class EdgeMidpointEGNNLayer(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        gate_in_dim = 2 * scalar_dim + 3
        self.gate_mlp = make_mlp(gate_in_dim, hidden_dim, 3 * vector_dim)
        self.psi = make_mlp(scalar_dim, hidden_dim, vector_dim)

        scalar_in_dim = 2 * scalar_dim + 2 + 2 * vector_dim + 1
        self.scalar_mlp = make_mlp(scalar_in_dim, hidden_dim, scalar_dim)

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor,
        midpoint_pos: torch.Tensor,
        midpoint_theta: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d = midpoint_pos[senders] - midpoint_pos[receivers]
        r = torch.linalg.norm(d, dim=-1, keepdim=True)
        u = d / (r + 1e-8)

        theta_j = midpoint_theta[receivers]
        theta_i = midpoint_theta[senders]
        delta_theta = theta_i - theta_j

        u_receiver_frame = rotate_vectors(u.unsqueeze(1), -theta_j).squeeze(1)
        v_i = v[senders]
        v_i_to_j = rotate_vectors(v_i, delta_theta)

        gate_inputs = torch.cat([h[senders], h[receivers], r, torch.cos(delta_theta).unsqueeze(-1), torch.sin(delta_theta).unsqueeze(-1)], dim=-1)
        gates = self.gate_mlp(gate_inputs)
        a, b, c = gates.split(self.vector_dim, dim=-1)

        psi_h = self.psi(h[senders])

        vector_message = (
            a.unsqueeze(-1) * v_i_to_j
            + b.unsqueeze(-1) * u_receiver_frame.unsqueeze(1)
            + c.unsqueeze(-1) * psi_h.unsqueeze(-1) * u_receiver_frame.unsqueeze(1)
        )
        v_update = scatter_sum(vector_message, receivers, h.shape[0])
        v_new = v + v_update

        v_norm = torch.linalg.norm(v_i_to_j, dim=-1)
        v_dot = (v_i_to_j * u_receiver_frame.unsqueeze(1)).sum(dim=-1)

        scalar_inputs = torch.cat(
            [
                h[senders],
                h[receivers],
                r,
                torch.cos(delta_theta).unsqueeze(-1),
                torch.sin(delta_theta).unsqueeze(-1),
                v_norm,
                v_dot,
            ],
            dim=-1,
        )
        delta_h = self.scalar_mlp(scalar_inputs)
        h_new = h + scatter_sum(delta_h, receivers, h.shape[0])
        return h_new, v_new
