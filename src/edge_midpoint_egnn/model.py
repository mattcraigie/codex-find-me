from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .graph import EdgeMidpointGraphBuilder, MidpointGraph
from .layers import EdgeMidpointEGNNLayer, make_mlp


class EdgeMidpointEncoder(nn.Module):
    def __init__(self, in_features: Optional[int], scalar_dim: int, vector_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.in_features = in_features
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim

        input_dim = 1 if in_features is None else 3 * in_features + 1
        self.scalar_mlp = make_mlp(input_dim, hidden_dim, scalar_dim)
        self.vector_mlp = make_mlp(input_dim, hidden_dim, vector_dim)

    def forward(self, graph: MidpointGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        i, j = graph.endpoints.t()
        lengths = graph.edge_lengths
        if graph.endpoint_features is None:
            feat_i = torch.zeros((i.numel(), 0), device=i.device)
            feat_j = torch.zeros((i.numel(), 0), device=i.device)
        else:
            feat_i = graph.endpoint_features[i]
            feat_j = graph.endpoint_features[j]

        diff = torch.abs(feat_i - feat_j) if feat_i.numel() > 0 else torch.zeros_like(feat_i)
        inputs = torch.cat([feat_i, feat_j, diff, lengths], dim=-1)
        h0 = self.scalar_mlp(inputs)

        direction = torch.stack([torch.cos(graph.midpoint_theta), torch.sin(graph.midpoint_theta)], dim=-1)
        amplitude = self.vector_mlp(inputs).unsqueeze(-1)
        v0 = amplitude * direction.unsqueeze(1)
        return h0, v0


class InvariantReadout(nn.Module):
    def __init__(self, scalar_dim: int, hidden_dim: int = 128, out_dim: int = 1, pooling: str = "mean") -> None:
        super().__init__()
        self.pooling = pooling
        self.mlp = make_mlp(scalar_dim, hidden_dim, out_dim)

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            pooled = scalars.mean(dim=0, keepdim=True)
        elif self.pooling == "sum":
            pooled = scalars.sum(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.mlp(pooled)


class EdgeMidpointEGNN(nn.Module):
    def __init__(
        self,
        in_features: Optional[int],
        scalar_dim: int = 64,
        vector_dim: int = 16,
        hidden_dim: int = 128,
        n_layers: int = 4,
        graph_builder: Optional[EdgeMidpointGraphBuilder] = None,
        include_readout: bool = False,
        readout_out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.graph_builder = graph_builder or EdgeMidpointGraphBuilder()
        self.encoder = EdgeMidpointEncoder(in_features, scalar_dim, vector_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [EdgeMidpointEGNNLayer(scalar_dim, vector_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.readout = (
            InvariantReadout(scalar_dim, hidden_dim, readout_out_dim)
            if include_readout
            else None
        )

    def build_graph(self, positions: torch.Tensor, features: Optional[torch.Tensor] = None) -> MidpointGraph:
        return self.graph_builder(positions, features)

    def forward(
        self,
        positions: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        return_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[MidpointGraph]]:
        graph = self.build_graph(positions, features)
        h, v = self.encoder(graph)
        for layer in self.layers:
            h, v = layer(h, v, graph.midpoint_pos, graph.midpoint_theta, graph.senders, graph.receivers)

        if self.readout is not None:
            output = self.readout(h)
        else:
            output = None

        if return_graph:
            return h, v, graph, output
        return h, v, output
