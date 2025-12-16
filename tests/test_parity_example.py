import pytest

# Skip if torch or dlutils are unavailable in the environment
torch = pytest.importorskip(
    "torch", reason="PyTorch is required; install via `pip install -r requirements.txt`"
)
dlutils = pytest.importorskip(
    "dlutils", reason="dlutils is required for the parity example; install via requirements"
)

from examples.parity_detection import (
    create_parity_violating_mocks_2d,
    train_and_test_model,
)


def test_parity_example_minimal_training(tmp_path):
    torch.manual_seed(0)

    mock_kwargs = {
        "field_size": 8,
        "total_num_triangles": 3,
        "ratio_left": 0.5,
        "length_side1": 2.0,
        "length_side2": 1.0,
        "min_scale": 1.0,
        "max_scale": 1.0,
    }
    training_kwargs = {
        "num_train_val_mocks": 32,
        "num_test_mocks": 16,
        "epochs": 1,
        "lr": 1e-3,
    }

    # Quick shape sanity check on mock generation
    mocks = create_parity_violating_mocks_2d(4, **mock_kwargs)
    assert mocks.shape == (4, mock_kwargs["field_size"], mock_kwargs["field_size"])

    results = train_and_test_model(
        model_type="simple_cnn",
        model_name="triangle_parity_test",
        model_kwargs={"field_size": mock_kwargs["field_size"]},
        mock_kwargs=mock_kwargs,
        training_kwargs=training_kwargs,
        output_root=str(tmp_path),
        repeats=1,
        device=torch.device("cpu"),
    )

    assert "val_scores" in results and "test_scores" in results
    assert results["val_scores"].shape[0] == 1
    assert results["test_scores"].shape[0] == 1
