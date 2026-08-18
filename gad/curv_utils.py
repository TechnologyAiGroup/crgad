# curv_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_node_curvature(kappa: torch.Tensor, adj: torch.Tensor, eps=1e-8):
    """
    Compute node curvature from edge curvature.
    
    Node curvature = average of edge curvatures connected to the node.
    
    Args:
        kappa: Edge curvature matrix [N, N], symmetric, kappa[i,j] is curvature of edge (i,j)
        adj: Adjacency matrix [N, N], used to identify edges
        eps: small value for numerical stability
    
    Returns:
        node_curv: Node curvature [N], each node has one curvature value
    """
    n = kappa.shape[0]
    device = kappa.device
    
    # Edge mask: 1 for edges, 0 for non-edges (exclude self-loops)
    edge_mask = (adj.abs() > 1e-6).float()
    edge_mask = edge_mask * (1 - torch.eye(n, device=device))  # remove diagonal
    
    # Sum of edge curvatures for each node (sum over rows)
    curv_sum = (kappa * edge_mask).sum(dim=1) + (kappa * edge_mask).sum(dim=0)  # [N]
    # Note: since kappa is symmetric, we can just use sum(dim=1) * 2
    
    # Number of edges for each node (degree)
    degree = edge_mask.sum(dim=1)  # [N]
    degree_safe = torch.where(degree < eps, torch.ones_like(degree), degree)
    
    # Node curvature = average of connected edge curvatures
    node_curv = curv_sum / (2 * degree_safe)
    
    # For isolated nodes (degree=0), set curvature to 0
    node_curv = torch.where(degree < eps, torch.zeros_like(node_curv), node_curv)
    
    return node_curv


class CurvaturePositionalEncoding(nn.Module):
    """
    Curvature-based Positional Encoding.
    
    Uses node curvature (scalar) and maps it to a vector via MLP.
    
    Node curvature encodes local graph structure:
    - High positive curvature: node is in a dense community (many triangles)
    - Near zero curvature: node is in a tree-like or grid-like structure
    - Negative curvature: node is a bridge between communities
    
    Args:
        hidden_size: dimension of the output positional encoding
        use_mlp: if True, use MLP to map scalar to vector; if False, use linear projection
        activation: activation function for MLP ('relu', 'gelu', 'none')
    """
    def __init__(self, hidden_size, use_mlp=True, activation='gelu'):
        super(CurvaturePositionalEncoding, self).__init__()
        
        self.hidden_size = hidden_size
        self.use_mlp = use_mlp
        
        if use_mlp:
            # MLP: scalar -> hidden -> hidden_size
            mid_size = hidden_size // 2
            if activation == 'relu':
                self.proj = nn.Sequential(
                    nn.Linear(1, mid_size),
                    nn.ReLU(),
                    nn.Linear(mid_size, hidden_size)
                )
            elif activation == 'gelu':
                self.proj = nn.Sequential(
                    nn.Linear(1, mid_size),
                    nn.GELU(),
                    nn.Linear(mid_size, hidden_size)
                )
            else:
                self.proj = nn.Sequential(
                    nn.Linear(1, mid_size),
                    nn.Linear(mid_size, hidden_size)
                )
        else:
            # Simple linear projection
            self.proj = nn.Linear(1, hidden_size)
    
    def forward(self, adj):
        """
        Compute curvature positional encoding.
        
        Args:
            adj: Adjacency matrix [N, N]
        
        Returns:
            pe: Positional encoding [N, hidden_size]
        """
        # Compute edge curvature
        kappa = orc_approx_jost_liu(adj)  # [N, N]
        
        # Compute node curvature (average of connected edge curvatures)
        node_curv = compute_node_curvature(kappa, adj)  # [N]
        
        # Normalize: curvature is typically in [-2, 1] range
        # We scale it to roughly [-1, 1] for better learning
        node_curv_norm = node_curv / 2.0  # approximately normalize to [-1, 0.5]
        
        # Map scalar to vector: [N] -> [N, 1] -> [N, hidden_size]
        node_curv_expanded = node_curv_norm.unsqueeze(-1)  # [N, 1]
        pe = self.proj(node_curv_expanded)  # [N, hidden_size]
        
        return pe


class CurvaturePositionalEncodingRoPE(nn.Module):
    """
    Curvature-based Positional Encoding using Rotary Position Embedding (RoPE) style.
    
    Instead of adding positional encoding to input features, RoPE applies rotation
    to query and key vectors in attention mechanism based on curvature values.
    
    The key idea from RoPE (used in LLaMA, DeepSeek, etc.):
    - Position information is encoded through rotation matrices
    - For position p and dimension d: 
        - PE(p, 2i) = sin(p / 10000^(2i/d))
        - PE(p, 2i+1) = cos(p / 10000^(2i/d))
    
    We adapt this by using normalized curvature as the "position":
    - Curvature κ is normalized to a suitable range
    - Each dimension pair gets a rotation angle based on κ
    
    This allows the attention mechanism to naturally incorporate curvature information
    through multiplicative interactions rather than additive.
    
    Args:
        hidden_size: dimension of the model (must be even)
        max_curvature: expected maximum absolute curvature value for normalization
        base: base for frequency calculation (default 10000, same as original RoPE)
    """
    def __init__(self, hidden_size, max_curvature=2.0, base=10000):
        super(CurvaturePositionalEncodingRoPE, self).__init__()
        
        assert hidden_size % 2 == 0, "hidden_size must be even for RoPE"
        
        self.hidden_size = hidden_size
        self.max_curvature = max_curvature
        self.base = base
        
        # Precompute frequency bands (inverse frequencies)
        # dim_i uses frequency: 1 / (base^(2i/d))
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, adj):
        """
        Compute curvature-based rotary positional encoding.
        
        Args:
            adj: Adjacency matrix [N, N]
        
        Returns:
            cos_cache: Cosine values for rotation [N, hidden_size]
            sin_cache: Sine values for rotation [N, hidden_size]
        """
        # Compute edge curvature
        kappa = orc_approx_jost_liu(adj)  # [N, N]
        
        # Compute node curvature (average of connected edge curvatures)
        node_curv = compute_node_curvature(kappa, adj)  # [N]
        
        # Normalize curvature to positive range for rotation
        # Curvature is typically in [-2, 1], we map to [0, ~1.5] range
        # by adding offset and scaling
        curv_normalized = (node_curv + self.max_curvature) / (2 * self.max_curvature)
        # curv_normalized is now roughly in [0, 1] range
        
        # Scale up to get meaningful rotation angles
        # Higher curvature = more rotation
        curv_scaled = curv_normalized * 10.0  # scale factor for more pronounced rotation
        
        # Compute angles: [N] -> [N, hidden_size/2]
        # Each dimension pair gets a different frequency rotation
        angles = curv_scaled.unsqueeze(-1) * self.inv_freq.unsqueeze(0)  # [N, hidden_size/2]
        
        # Duplicate for sin/cos pairs: [N, hidden_size/2] -> [N, hidden_size]
        cos_cache = torch.cos(angles).repeat_interleave(2, dim=-1)  # [N, hidden_size]
        sin_cache = torch.sin(angles).repeat_interleave(2, dim=-1)  # [N, hidden_size]
        
        return cos_cache, sin_cache


def apply_rotary_pos_emb(x, cos, sin):
    """
    Apply rotary position embedding to input tensor.
    
    This rotates pairs of dimensions in x based on the cos/sin values.
    
    For each pair of dimensions (x1, x2):
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
    
    Args:
        x: Input tensor [N, num_heads, head_dim] or [N, hidden_size]
        cos: Cosine values [N, hidden_size] or [N, head_dim]
        sin: Sine values [N, hidden_size] or [N, head_dim]
    
    Returns:
        Rotated tensor with same shape as x
    """
    # x shape: [N, num_heads, head_dim] or [N, hidden_size]
    # cos/sin shape: [N, hidden_size]
    
    # Get the last dimension of x
    last_dim = x.shape[-1]
    
    # If cos/sin have larger dimension than x, slice to match
    # This handles the case where cos/sin are [N, hidden_size] but x is [N, num_heads, head_dim]
    if cos.shape[-1] != last_dim:
        cos = cos[..., :last_dim]
        sin = sin[..., :last_dim]
    
    # Reshape x to separate pairs: [..., last_dim] -> [..., last_dim/2, 2]
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    
    # Split into x1 and x2
    x1 = x_reshape[..., 0]  # [..., last_dim/2]
    x2 = x_reshape[..., 1]  # [..., last_dim/2]
    
    # Reshape cos and sin to match
    cos_reshape = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]  # [..., last_dim/2]
    sin_reshape = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]  # [..., last_dim/2]
    
    # Apply rotation
    # Expand cos/sin to match x's shape (for multi-head attention)
    while cos_reshape.dim() < x1.dim():
        cos_reshape = cos_reshape.unsqueeze(1)
        sin_reshape = sin_reshape.unsqueeze(1)
    
    rotated_x1 = x1 * cos_reshape - x2 * sin_reshape
    rotated_x2 = x1 * sin_reshape + x2 * cos_reshape
    
    # Stack back: [..., last_dim/2, 2] -> [..., last_dim]
    rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
    
    return rotated_x


def compute_neighbor_curvature_mean(node_curv: torch.Tensor, adj: torch.Tensor, eps=1e-8):
    """
    Compute the average curvature of neighboring nodes for each node.
    
    Args:
        node_curv: Node curvature [N]
        adj: Adjacency matrix [N, N]
        eps: small value for numerical stability
    
    Returns:
        neighbor_curv_mean: Average neighbor curvature [N]
    """
    n = node_curv.shape[0]
    device = node_curv.device
    
    # Edge mask (exclude self-loops)
    edge_mask = (adj.abs() > 1e-6).float()
    edge_mask = edge_mask * (1 - torch.eye(n, device=device))
    
    # Sum of neighbor curvatures: [N, N] @ [N] -> [N]
    # For each node i, sum of curvatures of its neighbors
    neighbor_curv_sum = edge_mask @ node_curv  # [N]
    
    # Number of neighbors (degree)
    degree = edge_mask.sum(dim=1)  # [N]
    degree_safe = torch.where(degree < eps, torch.ones_like(degree), degree)
    
    # Average neighbor curvature
    neighbor_curv_mean = neighbor_curv_sum / degree_safe
    
    # For isolated nodes, use global average curvature
    global_mean = node_curv.mean()
    neighbor_curv_mean = torch.where(degree < eps, 
                                      torch.full_like(neighbor_curv_mean, global_mean.item()),
                                      neighbor_curv_mean)
    
    return neighbor_curv_mean


class CurvatureDiffPositionalEncoding(nn.Module):
    """
    Curvature Difference-based Positional Encoding.
    
    Uses the DIFFERENCE between node's own curvature and the average curvature 
    of its neighbors as the positional encoding signal.
    
    This captures how "unusual" a node's local topology is compared to its neighborhood:
    - Positive difference: node is in a denser local region than its neighbors (local hub)
    - Negative difference: node is in a sparser region than neighbors (peripheral node)
    - Near zero: node's curvature is typical for its neighborhood
    
    This is particularly useful for anomaly detection:
    - Anomalies often have curvature patterns different from their neighbors
    - A node with unusual curvature relative to its neighborhood may be suspicious
    
    Args:
        hidden_size: dimension of the output positional encoding
        use_mlp: if True, use MLP to map scalar to vector; if False, use linear projection
        activation: activation function for MLP ('relu', 'gelu', 'none')
    """
    def __init__(self, hidden_size, use_mlp=True, activation='gelu'):
        super(CurvatureDiffPositionalEncoding, self).__init__()
        
        self.hidden_size = hidden_size
        self.use_mlp = use_mlp
        
        if use_mlp:
            mid_size = hidden_size // 2
            if activation == 'relu':
                self.proj = nn.Sequential(
                    nn.Linear(1, mid_size),
                    nn.ReLU(),
                    nn.Linear(mid_size, hidden_size)
                )
            elif activation == 'gelu':
                self.proj = nn.Sequential(
                    nn.Linear(1, mid_size),
                    nn.GELU(),
                    nn.Linear(mid_size, hidden_size)
                )
            else:
                self.proj = nn.Sequential(
                    nn.Linear(1, mid_size),
                    nn.Linear(mid_size, hidden_size)
                )
        else:
            self.proj = nn.Linear(1, hidden_size)
    
    def forward(self, adj):
        """
        Compute curvature difference positional encoding.
        
        Args:
            adj: Adjacency matrix [N, N]
        
        Returns:
            pe: Positional encoding [N, hidden_size]
        """
        # Compute edge curvature
        kappa = orc_approx_jost_liu(adj)  # [N, N]
        
        # Compute node curvature (average of connected edge curvatures)
        node_curv = compute_node_curvature(kappa, adj)  # [N]
        
        # Compute average neighbor curvature
        neighbor_curv_mean = compute_neighbor_curvature_mean(node_curv, adj)  # [N]
        
        # Curvature difference: own curvature - average neighbor curvature
        curv_diff = node_curv - neighbor_curv_mean  # [N]
        
        # Normalize: difference is typically small, scale up for better learning
        # The difference is usually in [-1, 1] range, we keep it as is
        curv_diff_norm = curv_diff  # no scaling needed, difference is already normalized
        
        # Map scalar to vector: [N] -> [N, 1] -> [N, hidden_size]
        curv_diff_expanded = curv_diff_norm.unsqueeze(-1)  # [N, 1]
        pe = self.proj(curv_diff_expanded)  # [N, hidden_size]
        
        return pe


def orc_approx_jost_liu(A: torch.Tensor, eps=1e-8):
    """
    Differentiable Jost-Liu ORC approximation.
    - Input A should be in [0, 1] (e.g., sigmoid output)
    - Uses soft degrees and soft common neighbors
    - Preserves gradients from A to kappa
    """
    device = A.device
    n = A.shape[0]
    
    # 1. Ensure symmetry (assume undirected graph)
    A_sym = (A + A.t()) / 2  # [n, n]
    
    # 2. Compute soft degrees (sum of edge weights)
    deg = A_sym.sum(dim=1)  # [n]
    deg_safe = torch.where(deg < eps, torch.tensor(eps, device=device), deg)
    
    # 3. Soft common neighbors: (A @ A)[i, j] = sum_k A[i,k] * A[j,k]
    comm_neigh = A_sym @ A_sym  # [n, n] — now differentiable!
    
    # 4. Get edge mask: consider all pairs (or only where A_sym > threshold)
    # For full differentiability, we compute curvature for ALL pairs,
    # but will zero out non-edges later if needed.
    # Alternatively, use a soft mask:
    edge_mask = A_sym  # use edge weight as mask (continuous)
    
    # 5. Extract diagonal and off-diagonal
    # We'll compute curvature for all (i,j), then apply mask
    di = deg_safe.unsqueeze(1)  # [n, 1]
    dj = deg_safe.unsqueeze(0)  # [1, n]
    
    max_d = torch.maximum(di, dj)  # [n, n]
    min_d = torch.minimum(di, dj)  # [n, n]
    
    tri = comm_neigh  # [n, n]
    
    # Upper bound
    upper = tri / max_d  # [n, n]
    
    # Lower bound components
    t1 = 1 - 1/di - 1/dj - tri / min_d  # [n, n]
    t2 = 1 - 1/di - 1/dj - tri / max_d  # [n, n]
    lower = -F.relu(t1) - F.relu(t2) + tri / max_d  # [n, n]
    
    # Approximate curvature
    kappa = 0.5 * (upper + lower)  # [n, n]
    
    # 6. Apply edge mask: only edges have meaningful curvature
    # Option 1: hard mask (not differentiable, but common in practice)
    #   kappa = kappa * (A_sym > 1e-8).float()
    # Option 2: soft mask (fully differentiable)
    kappa = kappa * A_sym  # weight by edge probability
    
    # 7. Zero diagonal
    kappa = kappa - torch.diag(torch.diag(kappa))
    
    return kappa