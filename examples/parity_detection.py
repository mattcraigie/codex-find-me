"""
Minimal 2D parity-violation toy example using simple triangle mocks.
The example mirrors the reference snippet with dlutils DataHandler/RegressionTrainer
and a straightforward batch-difference loss.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from dlutils.data import DataHandler, RotatedDataset
from dlutils.training import RegressionTrainer
from torch import nn

# ----------------------
# Triangle mock helpers
# ----------------------


def random_unit_vector_2d(num_vectors: int) -> np.ndarray:
    vec = np.random.randn(num_vectors, 2)
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    return vec


def get_random_orthog_vecs_2d(num_vectors: int) -> Tuple[np.ndarray, np.ndarray]:
    i = random_unit_vector_2d(num_vectors)
    j = np.stack([-i[:, 1], i[:, 0]], axis=1)
    return i, j


def add_triangle_to_grid(size: int, a: float, b: float, num_triangles: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)

    x1, y1 = np.random.uniform(0, size, (2, num_triangles))
    point_1 = np.stack([x1, y1], axis=1)

    direction_2, direction_3 = get_random_orthog_vecs_2d(num_triangles)

    point_2 = point_1 + a * direction_2
    point_3 = point_1 + b * direction_3

    for p in [point_1, point_2, point_3]:
        p_grid = np.round(p).astype(int) % size
        np.add.at(grid, tuple(p_grid.T), 1)

    return grid


def make_triangle_mocks(
    num_mocks: int,
    size: int,
    a: float,
    b: float,
    num_triangles: int,
    min_scale: float = 1.0,
    max_scale: float = 1.0,
) -> np.ndarray:
    try:
        min_scale = float(min_scale)
        max_scale = float(max_scale)
        if min_scale > max_scale or min_scale <= 0 or max_scale <= 0:
            raise ValueError
    except (ValueError, TypeError):
        min_scale = max_scale = 1.0

    all_mocks = np.zeros((num_mocks, size, size))

    for i in range(num_mocks):
        scale = np.random.uniform(min_scale, max_scale)
        scaled_a = a * scale
        scaled_b = b * scale
        resulting_grid = add_triangle_to_grid(size, scaled_a, scaled_b, num_triangles)
        all_mocks[i] = resulting_grid

    return all_mocks


def create_triangle_mock_set_2d(
    num_mocks: int,
    field_size: int,
    total_num_triangles: int,
    ratio_left: float,
    length_side1: float,
    length_side2: float,
    min_scale: float = 1.0,
    max_scale: float = 1.0,
) -> np.ndarray:
    num_left = round(total_num_triangles * ratio_left)
    num_right = round(total_num_triangles * (1 - ratio_left))

    fields_left = make_triangle_mocks(
        num_mocks,
        field_size,
        length_side1,
        length_side2,
        num_left,
        min_scale,
        max_scale,
    )
    fields_right = make_triangle_mocks(
        num_mocks,
        field_size,
        length_side1,
        -length_side2,
        num_right,
        min_scale,
        max_scale,
    )

    return fields_left + fields_right


# ----------------------
# Parity dataset helpers
# ----------------------


def create_parity_violating_mocks_2d(
    num_mocks: int,
    field_size: int = 16,
    total_num_triangles: int = 3,
    ratio_left: float = 0.5,
    length_side1: float = 2.0,
    length_side2: float = 1.5,
    min_scale: float = 1.0,
    max_scale: float = 1.0,
) -> np.ndarray:
    return create_triangle_mock_set_2d(
        num_mocks,
        field_size,
        total_num_triangles,
        ratio_left,
        length_side1,
        length_side2,
        min_scale,
        max_scale,
    )


# ----------------------
# Model + loss
# ----------------------


class SimpleParityCNN(nn.Module):
    def __init__(self, field_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, 1)
        self.field_size = field_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 1, self.field_size, self.field_size)
        feats = self.net(x).flatten(1)
        return self.head(feats).squeeze(-1)


def batch_difference_loss(model: nn.Module, data: torch.Tensor) -> torch.Tensor:
    gx = model(data)
    gPx = model(data.flip(dims=(-1,)))

    fx = gx - gPx
    mu_B = fx.mean(0)
    sigma_B = fx.std(0)
    return -mu_B / sigma_B


# ----------------------
# Training entrypoint
# ----------------------


model_lookup: Dict[str, nn.Module] = {
    "simple_cnn": SimpleParityCNN,
}


def get_parity_score(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            score = float(batch_difference_loss(model, batch).item())
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def train_and_test_model(
    model_type: str,
    model_name: str,
    model_kwargs: Dict,
    mock_kwargs: Dict,
    training_kwargs: Dict,
    output_root: str,
    repeats: int = 1,
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_class = model_lookup[model_type]
    except KeyError:
        raise KeyError(f"Unrecognized model type {model_type} for {model_name} analysis")

    os.makedirs(output_root, exist_ok=True)

    validation_scores = []
    test_scores = []

    test_mocks = create_parity_violating_mocks_2d(training_kwargs["num_test_mocks"], **mock_kwargs)
    test_mocks = torch.from_numpy(test_mocks).float().unsqueeze(1)
    test_loader = DataHandler(test_mocks).make_single_dataloader(batch_size=32)

    for repeat in range(repeats):
        start = time.time()

        np.random.seed(0)
        torch.manual_seed(repeat)

        train_val_mocks = create_parity_violating_mocks_2d(
            training_kwargs["num_train_val_mocks"], **mock_kwargs
        )
        train_val_mocks = torch.from_numpy(train_val_mocks).float().unsqueeze(1)

        data_handler = DataHandler(train_val_mocks)
        train_loader, val_loader = data_handler.make_dataloaders(
            batch_size=32,
            shuffle_split=False,
            shuffle_dataloaders=True,
            val_fraction=0.2,
            dataset_class=RotatedDataset,
        )

        model = model_class(**model_kwargs).to(device)

        trainer = RegressionTrainer(
            model,
            train_loader,
            val_loader,
            criterion=batch_difference_loss,
            no_targets=True,
            device=device,
        )
        trainer.run_training(
            epochs=training_kwargs["epochs"],
            lr=training_kwargs["lr"],
            print_progress=False,
            show_loss_plot=False,
        )

        trainer.get_best_model()

        val_scores = [float(batch_difference_loss(model, batch.to(device)).item()) for batch in val_loader]
        validation_scores.append(float(np.mean(val_scores)))

        test_scores.append(get_parity_score(model, test_loader, device))

        torch.save(model.state_dict(), os.path.join(output_root, f"{model_name}_repeat{repeat}.pt"))
        np.save(os.path.join(output_root, f"{model_name}_repeat{repeat}_losses.npy"), trainer.losses)

        print(f"Repeat {repeat} took {time.time() - start:.2f} seconds")

    return {
        "val_scores": np.array(validation_scores),
        "test_scores": np.array(test_scores),
    }


def main() -> None:  # pragma: no cover
    mock_kwargs = {
        "field_size": 16,
        "total_num_triangles": 4,
        "ratio_left": 0.6,
        "length_side1": 2.0,
        "length_side2": 1.5,
        "min_scale": 0.8,
        "max_scale": 1.2,
    }
    training_kwargs = {
        "num_train_val_mocks": 256,
        "num_test_mocks": 128,
        "epochs": 3,
        "lr": 1e-3,
    }
    results = train_and_test_model(
        model_type="simple_cnn",
        model_name="triangle_parity",
        model_kwargs={"field_size": mock_kwargs["field_size"]},
        mock_kwargs=mock_kwargs,
        training_kwargs=training_kwargs,
        output_root="./triangle_parity_runs",
        repeats=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print("Validation scores:", results["val_scores"])
    print("Test scores:", results["test_scores"])


if __name__ == "__main__":
    main()
