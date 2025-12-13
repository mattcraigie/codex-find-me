# Edge-Midpoint Frame-Equivariant EGNN

A PyTorch implementation of the edge-midpoint frame-equivariant EGNN described in the design notes below. The package constructs midpoint-centric graphs from 2D point clouds, orients each midpoint node by the incident edge direction, and performs EGNN-style message passing with vector latents that are transported between local frames. All operations are SE(2)-equivariant by construction.

## Installation

```bash
pip install -e .
```

This installs the `edge_midpoint_egnn` package with a minimal dependency on PyTorch.

## Quickstart

```python
import torch
from edge_midpoint_egnn import EdgeMidpointEGNN

# toy point cloud: 5 points in 2D with optional features
positions = torch.randn(5, 2)
features = torch.randn(5, 3)

model = EdgeMidpointEGNN(
    in_features=features.shape[-1],
    scalar_dim=32,
    vector_dim=8,
    hidden_dim=96,
    n_layers=3,
    include_readout=True,
    readout_out_dim=2,
)

scalar_latents, vector_latents, graph, output = model(positions, features, return_graph=True)
print("midpoint nodes:", graph.midpoint_pos.shape[0])
print("graph-level output:", output.shape)
```

### Configuring graph construction

`EdgeMidpointEGNN` uses `EdgeMidpointGraphBuilder` to lift an input point cloud to an edge-midpoint graph:

- **Initial neighbors**: kNN over input points (default `k=8`).
- **Midpoint connectivity**: choose `midpoint_adjacency` as `"line_graph"` (edges that share a vertex), `"midpoint_knn"` (kNN over midpoints), or `"hybrid"` (union).
- **Frames**: every midpoint stores an oriented frame angle derived from its incident edge direction.

You can pass a custom builder instance to the model constructor for fine-grained control.

## Package layout

- `edge_midpoint_egnn.graph` – graph builders and the `MidpointGraph` dataclass.
- `edge_midpoint_egnn.model` – the `EdgeMidpointEGNN` model, encoder, and invariant readout head.
- `edge_midpoint_egnn.layers` – an SE(2)-equivariant message-passing layer with vector transport between frames.

## Model summary

1. **Inputs and initial graph**
   - Points \(x_i \in \mathbb{R}^2\) with optional features \(f_i \in \mathbb{R}^{d_{in}}\).
   - Build an undirected neighbor graph (default kNN). Edges \(e = (i, j)\) may be deterministically ordered.

2. **Edge-midpoint nodes**
   - Midpoint position: \(m_e = (x_i + x_j)/2\).
   - Local frame angle: \(\theta_e = \operatorname{atan2}(x_j^y - x_i^y, x_j^x - x_i^x)\) (mod \(2\pi\)).

3. **Midpoint connectivity**
   - Line-graph adjacency (edges that share an endpoint) and/or kNN between midpoints.

4. **Latents per midpoint**
   - Scalar \(h_j \in \mathbb{R}^{d_s}\).
   - Vector \(V_j \in \mathbb{R}^{d_v \times 2}\) transforming as \(V_j \rightarrow V_j R^T\) under global rotation \(R\).

5. **Message passing (one layer)**
   - Relative displacement \(d_{ij}\), distance \(r_{ij}\), unit direction \(u_{ij}\).
   - Receiver-frame unit direction \(u_{ij}^{(j)} = R(-\theta_j) u_{ij}\) and relative frame rotation \(\Delta\theta_{ij} = \theta_i - \theta_j\).
   - Rotate sender vectors into the receiver frame: \(V_{i \rightarrow j} = V_i R(\Delta\theta_{ij})^T\).
   - Scalar gates from an MLP over \((h_i, h_j, r_{ij}, \cos \Delta\theta_{ij}, \sin \Delta\theta_{ij})\) produce per-channel coefficients \(a_{ij}, b_{ij}, c_{ij}\).
   - Vector update: \(V_j \leftarrow V_j + \sum_i [a_{ij} V_{i \rightarrow j} + b_{ij} (\mathbf{1}_{d_v} \otimes u_{ij}^{(j)}) + c_{ij} (\psi(h_i) \otimes u_{ij}^{(j)})]\).
   - Scalar update: use invariants (vector norms and dot products with \(u_{ij}^{(j)}\)) plus \((h_i, h_j, r_{ij}, \cos \Delta\theta_{ij}, \sin \Delta\theta_{ij})\) inside an MLP.

6. **Depth and readout**
   - Stack multiple layers for higher-order structure. A simple invariant head pools scalars for graph-level predictions; vector latents remain equivariant for downstream tasks.

## Notes

- All geometric operations use only relative distances and angles; absolute coordinates are never consumed by MLPs.
- Vector latents follow spin-1 transformation rules; if you need director (spin-2) behavior, restrict vector usage accordingly.
- The implementation favors clarity and modularity so you can swap neighbor builders, message-passing depths, or readout strategies as needed.
