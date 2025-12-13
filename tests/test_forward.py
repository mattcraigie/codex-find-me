import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch is required; install via `pip install -r requirements.txt`"
)

from edge_midpoint_egnn import EdgeMidpointEGNN


def test_forward_pass_runs():
    torch.manual_seed(0)
    positions = torch.randn(6, 2)
    features = torch.randn(6, 4)

    model = EdgeMidpointEGNN(
        in_features=features.shape[-1],
        scalar_dim=16,
        vector_dim=4,
        hidden_dim=32,
        n_layers=2,
        include_readout=True,
        readout_out_dim=3,
    )

    scalars, vectors, graph, output = model(positions, features, return_graph=True)

    assert scalars.shape[0] == graph.midpoint_pos.shape[0]
    assert vectors.shape[:2] == (graph.midpoint_pos.shape[0], model.encoder.vector_dim)
    assert output.shape[-1] == 3
