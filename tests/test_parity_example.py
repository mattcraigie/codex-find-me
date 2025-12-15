import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch is required; install via `pip install -r requirements.txt`"
)

from examples.parity_detection import TrainConfig, run


def test_parity_example_demonstrates_parity_detection():
    torch.manual_seed(123)
    cfg = TrainConfig(
        device=torch.device("cpu"),
        epochs=2,
        n_points_total=512,
        test_points=128,
        bag_size=64,
        K=2,
        steps_per_epoch=6,
    )

    metrics = run(cfg)

    for acc in (metrics.edge_acc, metrics.baseline_acc, metrics.null_edge_acc, metrics.null_baseline_acc):
        assert 0.0 <= acc <= 1.0

    # Edge model should not underperform baseline on average
    assert metrics.edge_acc >= metrics.baseline_acc - 0.2

    # Null datasets should stay near chance
    assert abs(metrics.null_edge_acc - 0.5) <= 0.3
    assert abs(metrics.null_baseline_acc - 0.5) <= 0.3
