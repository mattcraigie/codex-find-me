from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


def _unique_undirected_edges(edges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    unique = set()
    for i, j in edges:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        unique.add((a, b))
    return sorted(unique)


def _knn_edges(positions: torch.Tensor, k: int) -> List[Tuple[int, int]]:
    n = positions.shape[0]
    if n <= 1:
        return []
    k = min(k, n - 1)
    distances = torch.cdist(positions, positions, p=2)
    knn = distances.topk(k + 1, largest=False).indices[:, 1:]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in knn[i].tolist():
            edges.append((i, j))
    return _unique_undirected_edges(edges)


def _line_graph_adjacency(num_points: int, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    incident: List[List[int]] = [[] for _ in range(num_points)]
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)

    adjacency: set[Tuple[int, int]] = set()
    for edge_indices in incident:
        for i in range(len(edge_indices)):
            for j in range(len(edge_indices)):
                if i == j:
                    continue
                adjacency.add((edge_indices[i], edge_indices[j]))
    return sorted(adjacency)


def _midpoint_knn_adjacency(midpoints: torch.Tensor, k: int) -> List[Tuple[int, int]]:
    e = midpoints.shape[0]
    if e <= 1:
        return []
    k = min(k, e - 1)
    distances = torch.cdist(midpoints, midpoints, p=2)
    knn = distances.topk(k + 1, largest=False).indices[:, 1:]
    adjacency: set[Tuple[int, int]] = set()
    for i in range(e):
        for j in knn[i].tolist():
            if i == j:
                continue
            adjacency.add((i, j))
    return sorted(adjacency)


@dataclass
class MidpointGraph:
    midpoint_pos: torch.Tensor
    midpoint_theta: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    endpoints: torch.Tensor
    endpoint_pos: torch.Tensor
    endpoint_features: Optional[torch.Tensor] = None
    edge_lengths: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "MidpointGraph":
        return MidpointGraph(
            midpoint_pos=self.midpoint_pos.to(device),
            midpoint_theta=self.midpoint_theta.to(device),
            senders=self.senders.to(device),
            receivers=self.receivers.to(device),
            endpoints=self.endpoints.to(device),
            endpoint_pos=self.endpoint_pos.to(device),
            endpoint_features=None
            if self.endpoint_features is None
            else self.endpoint_features.to(device),
            edge_lengths=None if self.edge_lengths is None else self.edge_lengths.to(device),
        )


class EdgeMidpointGraphBuilder:
    def __init__(
        self,
        k: int = 8,
        midpoint_adjacency: str = "line_graph",
        midpoint_k: int = 4,
    ) -> None:
        self.k = k
        self.midpoint_adjacency = midpoint_adjacency
        self.midpoint_k = midpoint_k

    def __call__(
        self,
        positions: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> MidpointGraph:
        positions = positions.float()
        if features is not None:
            features = features.float()
        edges = _knn_edges(positions, self.k)
        endpoints = torch.tensor(edges, device=positions.device, dtype=torch.long)
        if endpoints.numel() == 0:
            raise ValueError("No edges were constructed; try increasing k or providing more points")
        i, j = endpoints.t()
        midpoint_pos = (positions[i] + positions[j]) / 2
        delta = positions[j] - positions[i]
        midpoint_theta = torch.atan2(delta[:, 1], delta[:, 0])
        edge_lengths = torch.linalg.norm(delta, dim=-1, keepdim=True)

        adjacency: List[Tuple[int, int]] = []
        if self.midpoint_adjacency == "line_graph" or self.midpoint_adjacency == "hybrid":
            adjacency.extend(_line_graph_adjacency(positions.shape[0], edges))
        if self.midpoint_adjacency == "midpoint_knn" or self.midpoint_adjacency == "hybrid":
            adjacency.extend(_midpoint_knn_adjacency(midpoint_pos, self.midpoint_k))

        if not adjacency:
            # fallback to self-loop-free identity adjacency so layers can still run
            adjacency = [(idx, idx) for idx in range(len(edges))]

        adjacency = sorted(set(adjacency))
        senders, receivers = zip(*adjacency)
        return MidpointGraph(
            midpoint_pos=midpoint_pos,
            midpoint_theta=midpoint_theta,
            senders=torch.tensor(senders, device=positions.device, dtype=torch.long),
            receivers=torch.tensor(receivers, device=positions.device, dtype=torch.long),
            endpoints=endpoints,
            endpoint_pos=positions,
            endpoint_features=features,
            edge_lengths=edge_lengths,
        )
