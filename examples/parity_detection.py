"""Parity-violation detection with full-graph propagation + stochastic readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from edge_midpoint_egnn import BaselineEGNN, EdgeMidpointEGNN


# -----------------------------
# Framework primitives (new)
# -----------------------------


@dataclass
class TrainConfig:
    device: torch.device
    epochs: int = 50
    lr: float = 3e-3
    weight_decay: float = 1e-6

    # Global dataset size (fixed, generated once)
    n_points_total: int = 10_000
    test_points: int = 1_000  # held-out nodes from the same fixed cloud

    # Stochastic readout
    bag_size: int = 1_000  # subset size used to estimate mean statistic
    K: int = 4  # number of independent subsets per step
    steps_per_epoch: int = 128  # number of steps per epoch (each step = new subsets)

    # DataLoader
    batch_size: int = 1  # each item is a full-graph sample (train or test)
    num_workers: int = 0

    seed: int = 42


@dataclass
class Metrics:
    edge_acc: float
    baseline_acc: float
    null_edge_acc: float
    null_baseline_acc: float


class ExperimentModule(nn.Module):
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def fit(self, module: ExperimentModule, train_loader: DataLoader) -> None:
        module.to(self.cfg.device)
        opt = torch.optim.AdamW(
            module.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        for _ in range(self.cfg.epochs):
            module.train()
            for batch in train_loader:
                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                outputs = module(batch)
                loss = module.loss(outputs, batch)
                loss.backward()
                opt.step()

    @torch.no_grad()
    def evaluate(self, module: ExperimentModule, loader: DataLoader) -> float:
        module.eval()
        correct, total = 0, 0
        for batch in loader:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            outputs = module(batch)
            pred = module.predict(outputs)
            y = batch["label"]
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(1, total)


# -----------------------------
# Fixed synthetic point cloud
# -----------------------------


def _random_rotation_3d(g: torch.Generator, device: torch.device) -> torch.Tensor:
    a = torch.empty(3, 3, device=device).uniform_(-1.0, 1.0, generator=g)
    q, _ = torch.linalg.qr(a)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _build_fixed_global_pointset(
    *,
    n_points_total: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build exactly n_points_total points by placing (n_points_total/4) chiral
    tetrahedra on a jittered uniform grid (rare overlap), then apply ONE global
    random rotation+translation. No further stochasticity.
    """

    if n_points_total % 4 != 0:
        raise ValueError("n_points_total must be divisible by 4.")

    g = torch.Generator(device=device).manual_seed(seed)

    template = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.35, 0.9, 0.0],
            [0.55, 0.35, 0.65],
        ],
        device=device,
        dtype=torch.float32,
    )

    edge_scale = 1e-2
    spacing = 7e-2
    jitter = 1e-2  # uniform jitter

    n_tetra = n_points_total // 4

    side = int(torch.ceil(torch.tensor(float(n_tetra) ** (1 / 3))).item())
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(side, device=device),
            torch.arange(side, device=device),
            torch.arange(side, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)[:n_tetra].float()

    centers = coords * spacing
    centers = centers + torch.empty_like(centers).uniform_(-jitter, jitter, generator=g)

    tet = template * edge_scale
    points = (centers[:, None, :] + tet[None, :, :]).reshape(n_points_total, 3)

    R = _random_rotation_3d(g, device=device)
    points = points @ R.T
    translation = torch.empty(1, 3, device=device).uniform_(-0.5, 0.5, generator=g)
    points = points + translation

    return points


def _reflect_points_x(points: torch.Tensor) -> torch.Tensor:
    out = points.clone()
    out[..., 0] = -out[..., 0]
    return out


# -----------------------------
# Datasets: full-graph sample, with stochastic readout controlled by idx
# -----------------------------


class FullGraphSampleDataset(Dataset):
    """
    Produces items that each contain the FULL graph (train or test pool),
    a label, and a seed to drive stochastic readout (subset sampling) inside
    the module.

    We use length = steps so DataLoader yields multiple steps per epoch,
    while the underlying positions remain fixed.
    """

    def __init__(
        self,
        *,
        positions: torch.Tensor,  # (N, 3) fixed
        label: int,  # 0 or 1
        steps: int,
        base_seed: int,
    ) -> None:
        super().__init__()
        self.positions = positions.cpu()
        self.label = int(label)
        self.steps = int(steps)
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        step_seed = self.base_seed + idx
        return {
            "positions": self.positions.float(),
            "label": torch.tensor(self.label, dtype=torch.long),
            "step_seed": torch.tensor(step_seed, dtype=torch.long),
        }


class NullFullGraphDataset(Dataset):
    """
    Null: label and reflection are independent. Still uses full-graph propagation.
    """

    def __init__(
        self,
        *,
        positions: torch.Tensor,  # (N, 3) fixed
        steps: int,
        base_seed: int,
    ) -> None:
        super().__init__()
        self.positions = positions.cpu()
        self.steps = int(steps)
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        step_seed = self.base_seed + idx
        return {
            "positions": self.positions.float(),
            "label": torch.tensor(0, dtype=torch.long),
            "step_seed": torch.tensor(step_seed, dtype=torch.long),
            "null_mode": torch.tensor(1, dtype=torch.long),
        }


# -----------------------------
# Backbones (thin adapters): full-graph -> per-node scalar s_v
# -----------------------------


class EdgeMidpointNodeScalar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = EdgeMidpointEGNN(
            in_features=None,
            scalar_dim=48,
            vector_dim=12,
            hidden_dim=96,
            n_layers=3,
            include_readout=False,
        )
        self.to_scalar = nn.Linear(48, 1)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        node_scalar = self.model(positions, return_node_features=True)
        return self.to_scalar(node_scalar).squeeze(-1)


class BaselineNodeScalar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = BaselineEGNN(
            in_features=None,
            scalar_dim=48,
            hidden_dim=96,
            n_layers=3,
            k=16,
            include_readout=False,
        )
        self.to_scalar = nn.Linear(48, 1)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        node_scalar = self.model(positions, return_node_features=True)
        return self.to_scalar(node_scalar).squeeze(-1)


# -----------------------------
# Module: full-graph propagate + stochastic readout mean over subsets
# -----------------------------


class FullGraphStochasticReadoutClassifier(ExperimentModule):
    """
    For each step:
      - run backbone on ALL nodes -> s (N,)
      - sample K subsets of size bag_size (using step_seed)
      - compute T_hat_k = mean(s[idx_k])
      - classify each T_hat_k via head; average CE loss

    Prediction:
      - uses deterministic mean over ALL nodes: T_full = mean(s)
    """

    def __init__(self, backbone: nn.Module, bag_size: int, K: int, hidden: int = 64) -> None:
        super().__init__()
        self.backbone = backbone
        self.bag_size = int(bag_size)
        self.K = int(K)

        self.head = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.criterion = nn.CrossEntropyLoss()

    def _sample_subsets(self, N: int, step_seed: int, device: torch.device) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(int(step_seed))
        idx = []
        for k in range(self.K):
            gk = torch.Generator(device=device).manual_seed(int(step_seed) + 10_000 * (k + 1))
            idx_k = torch.randperm(N, generator=gk, device=device)[: self.bag_size]
            idx.append(idx_k)
        return torch.stack(idx, dim=0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = batch["positions"].squeeze(0)
        step_seed = int(batch["step_seed"].item())
        N = positions.shape[0]

        null_mode = int(batch.get("null_mode", torch.tensor(0, device=positions.device)).item())
        if null_mode == 1:
            g = torch.Generator(device=positions.device).manual_seed(step_seed + 777)
            label = int(torch.randint(0, 2, (1,), generator=g, device=positions.device).item())
            reflect = int(torch.randint(0, 2, (1,), generator=g, device=positions.device).item())
        else:
            label = int(batch["label"].item())
            reflect = label

        if reflect == 1:
            positions = _reflect_points_x(positions)

        s = self.backbone(positions)
        if s.dim() != 1 or s.shape[0] != N:
            raise ValueError(f"Expected backbone to return (N,), got {tuple(s.shape)}")

        subset_idx = self._sample_subsets(N, step_seed, device=positions.device)
        subset_means = (
            s.index_select(0, subset_idx.reshape(-1)).view(self.K, self.bag_size).mean(dim=1)
        )
        logits = self.head(subset_means.view(self.K, 1))

        T_full = s.mean()

        return {
            "logits_K": logits,
            "label": torch.tensor(label, device=positions.device, dtype=torch.long),
            "T_full": T_full,
        }

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        y = outputs["label"].view(1).expand(outputs["logits_K"].shape[0])
        return self.criterion(outputs["logits_K"], y)

    @torch.no_grad()
    def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.head(outputs["T_full"].view(1, 1))
        return logits.argmax(dim=-1)


# -----------------------------
# Build single fixed dataset and loaders
# -----------------------------


def make_loaders(cfg: TrainConfig) -> Dict[str, DataLoader]:
    cpu = torch.device("cpu")

    global_positions = _build_fixed_global_pointset(
        n_points_total=cfg.n_points_total,
        seed=cfg.seed,
        device=cpu,
    )

    g = torch.Generator(device=cpu).manual_seed(cfg.seed + 999)
    perm = torch.randperm(cfg.n_points_total, generator=g)
    test_idx = perm[: cfg.test_points]
    train_idx = perm[cfg.test_points :]

    train_positions = global_positions.index_select(0, train_idx)
    test_positions = global_positions.index_select(0, test_idx)

    train0 = FullGraphSampleDataset(
        positions=train_positions,
        label=0,
        steps=cfg.steps_per_epoch,
        base_seed=cfg.seed + 1000,
    )
    train1 = FullGraphSampleDataset(
        positions=train_positions,
        label=1,
        steps=cfg.steps_per_epoch,
        base_seed=cfg.seed + 2000,
    )
    train_ds = torch.utils.data.ConcatDataset([train0, train1])

    test0 = FullGraphSampleDataset(
        positions=test_positions,
        label=0,
        steps=max(64, cfg.steps_per_epoch // 2),
        base_seed=cfg.seed + 3000,
    )
    test1 = FullGraphSampleDataset(
        positions=test_positions,
        label=1,
        steps=max(64, cfg.steps_per_epoch // 2),
        base_seed=cfg.seed + 4000,
    )
    test_ds = torch.utils.data.ConcatDataset([test0, test1])

    null_ds = NullFullGraphDataset(
        positions=test_positions,
        steps=max(128, cfg.steps_per_epoch),
        base_seed=cfg.seed + 5000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    null_loader = DataLoader(
        null_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return {"train": train_loader, "test": test_loader, "null": null_loader}


# -----------------------------
# Run experiment
# -----------------------------


def run(cfg: Optional[TrainConfig] = None) -> Metrics:
    cfg = cfg or TrainConfig(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)

    loaders = make_loaders(cfg)
    trainer = Trainer(cfg)

    edge_module = FullGraphStochasticReadoutClassifier(
        backbone=EdgeMidpointNodeScalar(),
        bag_size=cfg.bag_size,
        K=cfg.K,
    )
    base_module = FullGraphStochasticReadoutClassifier(
        backbone=BaselineNodeScalar(),
        bag_size=cfg.bag_size,
        K=cfg.K,
    )

    trainer.fit(edge_module, loaders["train"])
    trainer.fit(base_module, loaders["train"])

    edge_acc = trainer.evaluate(edge_module, loaders["test"])
    baseline_acc = trainer.evaluate(base_module, loaders["test"])

    null_edge_acc = trainer.evaluate(edge_module, loaders["null"])
    null_baseline_acc = trainer.evaluate(base_module, loaders["null"])

    return Metrics(
        edge_acc=edge_acc,
        baseline_acc=baseline_acc,
        null_edge_acc=null_edge_acc,
        null_baseline_acc=null_baseline_acc,
    )


def main() -> None:
    cfg = TrainConfig(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    m = run(cfg)
    print("Edge-midpoint (full-prop + stochastic readout) acc:", m.edge_acc)
    print("Baseline (full-prop + stochastic readout) acc:", m.baseline_acc)
    print("Null test (edge-midpoint):", m.null_edge_acc)
    print("Null test (baseline):", m.null_baseline_acc)


if __name__ == "__main__":
    main()


