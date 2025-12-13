import pytest

torch = pytest.importorskip("torch")

from examples.parity_detection import train_parity_classifiers


def test_parity_example_demonstrates_parity_detection():
    torch.manual_seed(123)
    result = train_parity_classifiers(epochs=8, train_size=96, test_size=48, null_size=48)

    assert result.edge_acc >= 0.85
    assert result.baseline_acc <= 0.7
    assert result.edge_acc - result.baseline_acc >= 0.2

    # Null dataset should hover near chance for both models
    assert abs(result.null_edge_acc - 0.5) <= 0.25
    assert abs(result.null_baseline_acc - 0.5) <= 0.25
