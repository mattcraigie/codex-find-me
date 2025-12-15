"""Parity-detection example comparing the frame-equivariant EGNN to a baseline.

This module builds a synthetic dataset where labels indicate whether a chiral
2D point cloud is presented in a left- or right-handed orientation. The
baseline EGNN operates only on pairwise distances and cannot distinguish the
reflections, while the edge-midpoint frame-equivariant EGNN can leverage
oriented frames to resolve the parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import nn

from edge_midpoint_egnn import BaselineEGNN, EdgeMidpointEGNN


@dataclass
class TrainingResult:
    edge_acc: float
    baseline_acc: float
    null_edge_acc: float
    null_baseline_acc: float


class ParityPointCloudDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        n_samples: int = 256,
        n_points: int = 4,
        noise: float = 0.01,
        null_labels: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_points = n_points
        self.noise = noise
        self.null_labels = null_labels
        self.base_seed = seed

        # A chiral template: triangle with a displaced fourth point.
        self.prototype = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.35, 0.8],
                [0.7, 0.4],
            ]
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.n_samples:  # guard against accidental reuse
            raise IndexError

        g = torch.Generator().manual_seed(self.base_seed + idx)
        label = torch.randint(0, 2, (1,), generator=g).item() if self.null_labels else idx % 2

        points = self.prototype.clone()
        points += torch.randn(points.shape, generator=g) * self.noise

        reflect = (
            torch.randint(0, 2, (1,), generator=g).item() if self.null_labels else label
        )
        if reflect == 1:
            points[:, 1] = -points[:, 1]

        theta = torch.rand((), generator=g) * (2 * torch.pi)
        cos, sin = torch.cos(theta), torch.sin(theta)
        rot = torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])])
        points = points @ rot.T
        translation = torch.randn(1, 2, generator=g) * 0.2
        points = points + translation

        return points.float(), torch.tensor(label, dtype=torch.long)


def _train_epoch(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    dataset: ParityPointCloudDataset,
    device: torch.device,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    for positions, labels in dataset:
        positions = positions.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = forward_fn(model, positions)
        loss = criterion(logits, labels.unsqueeze(0))
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    dataset: ParityPointCloudDataset,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    for positions, labels in dataset:
        positions = positions.to(device)
        labels = labels.to(device)
        logits = forward_fn(model, positions)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
    return correct / len(dataset)


def train_parity_classifiers(
    device: Optional[torch.device] = None,
    epochs: int = 12,
    train_size: int = 128,
    test_size: int = 64,
    null_size: int = 64,
) -> TrainingResult:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ParityPointCloudDataset(n_samples=train_size)
    test_ds = ParityPointCloudDataset(n_samples=test_size)
    null_ds = ParityPointCloudDataset(n_samples=null_size, null_labels=True, seed=123)

    def edge_model_fn() -> EdgeMidpointEGNN:
        return EdgeMidpointEGNN(
            in_features=None,
            scalar_dim=48,
            vector_dim=12,
            hidden_dim=96,
            n_layers=3,
            include_readout=True,
            readout_out_dim=2,
        )

    def baseline_model_fn() -> BaselineEGNN:
        return BaselineEGNN(
            in_features=None,
            scalar_dim=48,
            hidden_dim=96,
            n_layers=3,
            k=4,
            include_readout=True,
            readout_out_dim=2,
        )

    def edge_forward(model: EdgeMidpointEGNN, positions: torch.Tensor) -> torch.Tensor:
        result = model(positions, return_graph=True)
        # Older versions of the model may return only three outputs even when
        # ``return_graph`` is True; handle both shapes defensively to keep the
        # example robust across environments.
        if len(result) == 4:
            _, _, _, out = result
        elif len(result) == 3:
            _, _, out = result
        else:  # pragma: no cover - unexpected
            raise ValueError(f"Unexpected model output arity: {len(result)}")
        return out

    def baseline_forward(model: BaselineEGNN, positions: torch.Tensor) -> torch.Tensor:
        _, _, out = model(positions, return_graph=True)
        return out

    edge_model = edge_model_fn().to(device)
    baseline_model = baseline_model_fn().to(device)

    edge_opt = torch.optim.Adam(edge_model.parameters(), lr=3e-3)
    base_opt = torch.optim.Adam(baseline_model.parameters(), lr=3e-3)

    for _ in range(epochs):
        _train_epoch(edge_model, edge_forward, edge_opt, train_ds, device)
        _train_epoch(baseline_model, baseline_forward, base_opt, train_ds, device)

    edge_acc = _evaluate(edge_model, edge_forward, test_ds, device)
    baseline_acc = _evaluate(baseline_model, baseline_forward, test_ds, device)

    # Null test: labels are decoupled from reflection; accuracy should be near chance.
    null_edge_acc = _evaluate(edge_model, edge_forward, null_ds, device)
    null_baseline_acc = _evaluate(baseline_model, baseline_forward, null_ds, device)

    return TrainingResult(
        edge_acc=edge_acc,
        baseline_acc=baseline_acc,
        null_edge_acc=null_edge_acc,
        null_baseline_acc=null_baseline_acc,
    )


def main() -> None:  # pragma: no cover - example entrypoint
    torch.manual_seed(42)
    result = train_parity_classifiers()
    print("Edge-midpoint EGNN accuracy:", result.edge_acc)
    print("Baseline EGNN accuracy:", result.baseline_acc)
    print("Null test (edge-midpoint):", result.null_edge_acc)
    print("Null test (baseline):", result.null_baseline_acc)


if __name__ == "__main__":  # pragma: no cover
    main()
