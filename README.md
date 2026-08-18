# GRAD

This repository contains the implementation of "Curvature-Regularized Graph Autoencoders with Rewiring
for Node-Level Graph Anomaly Detection".

## Overview

The framework consists of two main components:

1. **Rewire**: Graph rewiring module that optimizes the graph structure using Ollivier-Ricci curvature and Fiedler vectors
2. **GAD (Graph Anomaly Detection)**: Anomaly detection module that uses a shared encoder-decoder architecture with curvature regularization

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Place your graph datasets in the `data/` directory.
Example data files expected:
- `gad/data/disney.pt`
- `gad/data/weibo.pt`

## Usage

### Step 1: Graph Rewiring

Run the graph rewiring process to optimize the graph structure:

```bash
cd rewire
python run_rewire.py --dataset disney --num_iterations 1 --batch_add 50 --batch_remove 10
```

Parameters:
- `--dataset`: Name of the dataset to process (e.g., 'books', 'disney')
- `--num_iterations`: Number of rewiring iterations (default: 1)
- `--batch_add`: Number of edges to add in each iteration (default: 50)
- `--batch_remove`: Number of edges to remove in each iteration (default: 10)


### Step 2: Anomaly Detection

After rewiring, run the anomaly detection with hyperparameter tuning using Optuna:

```bash
cd gad
python tune.py --model_type graph_transformer_curv --dataset disney --n_trials 30
```

Parameters:
- `--model_type`: Model architecture to use (see available models below)
- `--dataset`: Name of the dataset (e.g., 'weibo', 'disney')
- `--n_trials`: Number of Optuna trials for hyperparameter search
- `--search_type`: Search strategy ('tpe' for Bayesian, 'grid' for grid search)


## Model Architecture

### Available Models

The framework supports multiple model architectures:

| Model Type | Description |
|------------|-------------|
| `gcn` | Baseline GCN-based autoencoder (DOMINANT) |
| `standard_transformer` | Standard Transformer without graph bias (baseline) |
| `graph_transformer` | Graph Transformer with adjacency-based attention bias |
| `graph_transformer_lap` | Graph Transformer with Laplacian Positional Encoding |
| `graph_transformer_curv` | **Graph Transformer with Curvature Positional Encoding** |

### Model Details

#### GCN-based Model (DOMINANT)
- Shared GCN encoder with two layers
- Attribute decoder for feature reconstruction
- Structure decoder for adjacency reconstruction
- Curvature regularization in loss function

#### Graph Transformer Models

**1. Standard Transformer (Baseline)**
- Vanilla multi-head self-attention without graph structure information
- Used as baseline to demonstrate the importance of graph inductive bias

**2. Graph Transformer with Adjacency Bias**
- Adds learnable bias to attention scores based on adjacency matrix
- `Attention = softmax((Q @ K^T) / sqrt(d) + B)` where B is derived from adjacency

**3. Graph Transformer with Laplacian PE**
- Uses Laplacian eigenvectors as positional encoding
- Captures global graph structure information

**4. Graph Transformer with Curvature PE (Core Innovation)**
- Uses node curvature (Ollivier-Ricci curvature) as positional encoding
- Curvature encodes local graph topology:
  - **Positive curvature**: node is in a dense community (many triangles)
  - **Zero curvature**: tree-like or grid-like structure
  - **Negative curvature**: node is a bridge between communities
- Particularly effective for anomaly detection as anomalies often exhibit distinct curvature patterns

### Loss Function

The total loss combines reconstruction loss with curvature regularization:

```
loss = (1 - β_curv) * recon_loss + β_curv * curv_loss

where:
  recon_loss = α * attr_loss + (1 - α) * struct_loss
  curv_loss = curvature difference between original and reconstructed graph
```

## File Structure

```
├── gad/                           # Graph Anomaly Detection module
│   ├── model.py                   # GCN-based DOMINANT model
│   ├── model_transformer.py       # Graph Transformer variants
│   ├── layers.py                  # Graph convolutional layers
│   ├── loss.py                    # Loss function with curvature regularization
│   ├── curv_utils.py              # Curvature computation utilities
│   ├── utils.py                   # Data loading utilities
│   ├── tune.py                    # Optuna hyperparameter tuning
│   └── data/                      # Dataset directory
│       ├── disney.pt
│       └── weibo.pt
└── rewire/                        # Graph Rewiring module
    ├── run_rewire.py              # Rewiring execution script
    ├── preprocessing/
    │   └── rewire.py              # Core rewiring algorithm
    ├── GraphRicciCurvature/       # Ricci curvature computation library
    │   ├── OllivierRicci.py
    │   └── util.py
    └── data/                      # Dataset directory
        ├── books.pt
        └── disney.pt
```

## License

MIT License
