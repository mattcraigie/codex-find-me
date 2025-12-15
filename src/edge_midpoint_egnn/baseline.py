from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .graph import _knn_edges
from .layers import make_mlp, scatter_sum


class BaselineEGNNLayer(nn.Module):
    """A minimal EGNN-style layer that uses only pairwise distances.

    This layer intentionally avoids any oriented information so that it cannot
    separate mirror-related point clouds when given only coordinate distances.
    """

    def __init__(self, scalar_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        gate_in_dim = 2 * scalar_dim + 1
        self.gate_mlp = make_mlp(gate_in_dim, hidden_dim, scalar_dim)

    def forward(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        d = positions[senders] - positions[receivers]
        r = torch.linalg.norm(d, dim=-1, keepdim=True)

        gate_inputs = torch.cat([h[senders], h[receivers], r], dim=-1)
        messages = self.gate_mlp(gate_inputs)
        return h + scatter_sum(messages, receivers, h.shape[0])


class BaselineEGNN(nn.Module):
    """A simple EGNN baseline operating on the original points.

    The model uses kNN connectivity and processes only scalar features that are
    functions of pairwise distances, ensuring invariance to reflections.
    """

    def __init__(
        self,
        in_features: Optional[int],
        scalar_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 4,
        k: int = 8,
        include_readout: bool = False,
        readout_out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.k = k
        input_dim = in_features if in_features is not None else 0
        input_dim += 1  # edge length
        self.encoder = make_mlp(input_dim, hidden_dim, scalar_dim)
        self.layers = nn.ModuleList(
            [BaselineEGNNLayer(scalar_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.readout = (
            make_mlp(scalar_dim, hidden_dim, readout_out_dim)
            if include_readout
            else None
        )

    def build_edges(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edges = _knn_edges(positions, self.k)
        if not edges:
            raise ValueError("No edges were constructed; try increasing k or using more points")
        senders, receivers = zip(*edges)
        return torch.tensor(senders, device=positions.device), torch.tensor(
            receivers, device=positions.device
        )

    def forward(
        self,
        positions: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        return_graph: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        positions = positions.float()
        if features is not None:
            features = features.float()

        senders, receivers = self.build_edges(positions)
        d = positions[senders] - positions[receivers]
        lengths = torch.linalg.norm(d, dim=-1, keepdim=True)
        if features is None:
            feat_cat = lengths
        else:
            feat_cat = torch.cat([features[senders], lengths], dim=-1)
        h = self.encoder(feat_cat)

        for layer in self.layers:
            h = layer(h, positions, senders, receivers)

        out = self.readout(h.mean(dim=0, keepdim=True)) if self.readout is not None else None

        if return_graph:
            return h, (senders, receivers), out
        return h, out
