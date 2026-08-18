"""
Graph Transformer model for graph anomaly detection.
Uses attention bias from adjacency matrix to incorporate graph structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with adjacency bias.
    
    The key idea: add adjacency matrix as bias to attention scores
    Attention = softmax((Q @ K^T) / sqrt(d) + B)
    where B is derived from adjacency matrix A
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Learnable edge attention bias
        self.edge_bias = nn.Parameter(torch.zeros(1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        """
        Args:
            x: Node features [num_nodes, hidden_size]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Updated node features [num_nodes, hidden_size]
        """
        num_nodes = x.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.einsum('qhd,khd->hqk', Q, K) * self.scale
        
        # Add adjacency bias: edges get positive bias to increase attention
        # Use binary edge mask
        edge_mask = (adj.abs() > 1e-6).float()
        # Remove diagonal (self-loops get handled separately)
        edge_mask = edge_mask * (1 - torch.eye(num_nodes, device=adj.device))
        attn_bias = self.edge_bias * edge_mask
        attn_scores = attn_scores + attn_bias.unsqueeze(0)
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.einsum('hqk,khd->hqd', attn_probs, V)
        
        # Reshape back
        out = out.permute(1, 0, 2).contiguous().view(num_nodes, self.hidden_size)
        out = self.out_proj(out)
        
        return out


class GraphTransformerLayer(nn.Module):
    """
    Complete Graph Transformer Layer:
    - Multi-head Graph Attention
    - Feed-Forward Network
    - Layer Normalization
    - Residual Connections
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        
        self.attention = GraphAttentionLayer(hidden_size, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        # Pre-LN: normalize before attention
        attn_out = self.attention(self.norm1(x), adj)
        x = x + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class StructureDecoder(nn.Module):
    """
    Structure decoder that reconstructs the adjacency matrix.
    Uses bilinear product for better edge prediction.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super(StructureDecoder, self).__init__()
        
        # Bilinear weight for edge prediction: A_hat[i,j] = z_i @ W @ z_j^T
        self.edge_weight = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z):
        """
        Args:
            z: Node representations [num_nodes, hidden_size]
        Returns:
            A_hat: Reconstructed adjacency matrix [num_nodes, num_nodes]
        """
        z = self.dropout(z)
        # Bilinear product: [N, H] @ [H, H] @ [N, H]^T = [N, N]
        A_hat = z @ self.edge_weight @ z.T
        return A_hat


class AttributeDecoder(nn.Module):
    """
    Attribute decoder that reconstructs node features.
    """
    def __init__(self, hidden_size, feat_size, dropout=0.1):
        super(AttributeDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, feat_size)
        )
        
    def forward(self, z):
        """
        Args:
            z: Node representations [num_nodes, hidden_size]
        Returns:
            X_hat: Reconstructed features [num_nodes, feat_size]
        """
        return self.decoder(z)


# ============================================================================
# Standard Transformer (without graph bias) for comparison
# ============================================================================

class StandardAttentionLayer(nn.Module):
    """
    Standard Transformer Attention Layer WITHOUT adjacency bias.
    
    This is a vanilla multi-head self-attention layer for comparison with
    GraphAttentionLayer. It does NOT use any graph structure information.
    
    Attention = softmax((Q @ K^T) / sqrt(d))
    Note: NO bias term B from adjacency matrix!
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(StandardAttentionLayer, self).__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj=None):
        """
        Args:
            x: Node features [num_nodes, hidden_size]
            adj: Adjacency matrix [num_nodes, num_nodes] (NOT USED, kept for API consistency)
        
        Returns:
            Updated node features [num_nodes, hidden_size]
        """
        num_nodes = x.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores (NO bias from adjacency!)
        attn_scores = torch.einsum('qhd,khd->hqk', Q, K) * self.scale
        
        # Softmax (standard, no graph bias)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.einsum('hqk,khd->hqd', attn_probs, V)
        
        # Reshape back
        out = out.permute(1, 0, 2).contiguous().view(num_nodes, self.hidden_size)
        out = self.out_proj(out)
        
        return out


class StandardTransformerLayer(nn.Module):
    """
    Standard Transformer Layer (without graph bias).
    
    Same structure as GraphTransformerLayer but uses StandardAttentionLayer
    instead of GraphAttentionLayer.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(StandardTransformerLayer, self).__init__()
        
        self.attention = StandardAttentionLayer(hidden_size, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj=None):
        # Pre-LN: normalize before attention
        attn_out = self.attention(self.norm1(x), adj)
        x = x + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class DominantStandardTransformer(nn.Module):
    """
    Standard Transformer for graph anomaly detection (WITHOUT graph attention bias).
    
    This model serves as a baseline to compare against DominantGraphTransformer.
    It uses vanilla self-attention without incorporating adjacency matrix information.
    
    Key difference from DominantGraphTransformer:
    - NO attention bias from adjacency matrix
    - Relies purely on node features for attention
    - Graph structure is ONLY used in the decoder for reconstruction loss
    """
    def __init__(self, feat_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        super(DominantStandardTransformer, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(feat_size, hidden_size)
        
        # Standard Transformer encoder layers (NO graph bias)
        self.encoder_layers = nn.ModuleList([
            StandardTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final encoder norm
        self.encoder_norm = nn.LayerNorm(hidden_size)
        
        # Decoders (same as Graph Transformer)
        self.attr_decoder = AttributeDecoder(hidden_size, feat_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)
        
    def forward(self, x, adj=None):
        """
        Forward pass.
        
        Args:
            x: Node feature matrix [num_nodes, feat_size]
            adj: Adjacency matrix [num_nodes, num_nodes] (not used in encoder, 
                 kept for API consistency and used only in loss computation)
        
        Returns:
            A_hat: Reconstructed adjacency matrix [num_nodes, num_nodes]
            X_hat: Reconstructed attribute matrix [num_nodes, feat_size]
        """
        # Project input
        h = self.input_proj(x)
        
        # Standard Transformer encoding (no graph bias)
        for layer in self.encoder_layers:
            h = layer(h, adj)  # adj is ignored in StandardTransformerLayer
        
        # Final normalization
        z = self.encoder_norm(h)
        
        # Decode attributes
        X_hat = self.attr_decoder(z)
        
        # Decode structure with bilinear product
        A_hat = self.struct_decoder(z)
        
        return A_hat, X_hat


class DominantGraphTransformer(nn.Module):
    def __init__(self, feat_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        super(DominantGraphTransformer, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(feat_size, hidden_size)
        
        # Graph Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final encoder norm
        self.encoder_norm = nn.LayerNorm(hidden_size)
        
        # Decoders
        self.attr_decoder = AttributeDecoder(hidden_size, feat_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)
        
    def forward(self, x, adj):
        """
        Forward pass.
        
        Args:
            x: Node feature matrix [num_nodes, feat_size]
            adj: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            A_hat: Reconstructed adjacency matrix [num_nodes, num_nodes]
            X_hat: Reconstructed attribute matrix [num_nodes, feat_size]
        """
        # Project input
        h = self.input_proj(x)
        
        # Graph Transformer encoding
        for layer in self.encoder_layers:
            h = layer(h, adj)
        
        # Final normalization
        z = self.encoder_norm(h)
        
        # Decode attributes
        X_hat = self.attr_decoder(z)
        
        # Decode structure with bilinear product
        A_hat = self.struct_decoder(z)
        
        return A_hat, X_hat


# ============================================================================
# Alternative: Laplacian Positional Encoding version
# ============================================================================

class LaplacianPositionalEncoding(nn.Module):
    """
    Laplacian Positional Encoding for graph structure.
    """
    def __init__(self, hidden_size, max_k=16):
        super(LaplacianPositionalEncoding, self).__init__()
        self.max_k = max_k
        self.proj = nn.Linear(max_k, hidden_size)
        
    def forward(self, adj):
        num_nodes = adj.size(0)
        degree = adj.sum(dim=1)
        L = torch.diag(degree) - adj
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            k = min(self.max_k, num_nodes)
            pe = eigenvectors[:, :k]
            if k < self.max_k:
                pe = F.pad(pe, (0, self.max_k - k))
        except:
            pe = torch.zeros(num_nodes, self.max_k, device=adj.device)
        
        return self.proj(pe)


class GraphTransformerWithLapPE(nn.Module):
    """
    Graph Transformer with Laplacian Positional Encoding.
    """
    def __init__(self, feat_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        super(GraphTransformerWithLapPE, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.input_proj = nn.Linear(feat_size, hidden_size)
        self.lap_pe = LaplacianPositionalEncoding(hidden_size)
        
        self.encoder_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(hidden_size)
        
        self.attr_decoder = AttributeDecoder(hidden_size, feat_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)
        
    def forward(self, x, adj):
        h = self.input_proj(x)
        
        # Add Laplacian PE
        pe = self.lap_pe(adj)
        h = h + pe
        
        for layer in self.encoder_layers:
            h = layer(h, adj)
        
        z = self.encoder_norm(h)
        
        X_hat = self.attr_decoder(z)
        A_hat = self.struct_decoder(z)
        
        return A_hat, X_hat


# ============================================================================
# Curvature-based Positional Encoding version
# ============================================================================

class GraphTransformerWithCurvPE(nn.Module):
    """
    Graph Transformer with Curvature Positional Encoding.
    
    Uses node curvature (derived from Ollivier-Ricci curvature) as positional encoding.
    
    Node curvature encodes local graph topology:
    - Positive curvature: node is in a dense community (many triangles)
    - Zero curvature: tree-like or grid-like structure
    - Negative curvature: node is a bridge between communities
    
    This can be particularly useful for anomaly detection, as anomalies often
    exhibit distinct curvature patterns.
    """
    def __init__(self, feat_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        super(GraphTransformerWithCurvPE, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.input_proj = nn.Linear(feat_size, hidden_size)
        
        # Import and use curvature positional encoding
        from curv_utils import CurvaturePositionalEncoding
        self.curv_pe = CurvaturePositionalEncoding(hidden_size, use_mlp=True, activation='gelu')
        
        self.encoder_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(hidden_size)
        
        self.attr_decoder = AttributeDecoder(hidden_size, feat_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)
        
    def forward(self, x, adj):
        """
        Forward pass with curvature positional encoding.
        
        Args:
            x: Node features [N, F]
            adj: Adjacency matrix [N, N]
        
        Returns:
            A_hat: Reconstructed adjacency [N, N]
            X_hat: Reconstructed features [N, F]
        """
        # Project input features
        h = self.input_proj(x)
        
        # Compute and add curvature positional encoding
        pe = self.curv_pe(adj)
        h = h + pe
        
        # Graph Transformer encoding
        for layer in self.encoder_layers:
            h = layer(h, adj)
        
        z = self.encoder_norm(h)
        
        # Decode
        X_hat = self.attr_decoder(z)
        A_hat = self.struct_decoder(z)
        
        return A_hat, X_hat


# ============================================================================
# Curvature Difference-based Positional Encoding version
# ============================================================================

class GraphTransformerWithCurvPEDiff(nn.Module):
    """
    Graph Transformer with Curvature Difference Positional Encoding.
    
    Uses the DIFFERENCE between a node's own curvature and the average curvature
    of its neighbors as the positional encoding signal.
    
    Key insight for anomaly detection:
    - Anomalous nodes often have curvature patterns that differ from their neighbors
    - A node with unusual local topology relative to its neighborhood may be suspicious
    - This "deviation from neighborhood norm" signal can be more discriminative than
      absolute curvature values
    
    Curvature difference interpretation:
    - Positive: node is in a denser local region than neighbors (local hub, potential anomaly)
    - Negative: node is in a sparser region than neighbors (peripheral node)
    - Near zero: node's topology is typical for its neighborhood (normal behavior)
    
    This encoding is particularly useful for detecting:
    - Community outliers: nodes that don't fit their local community structure
    - Bridge nodes: nodes connecting different communities (often negative curvature)
    - Dense anomalies: nodes with unusually high local connectivity
    """
    def __init__(self, feat_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        super(GraphTransformerWithCurvPEDiff, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.input_proj = nn.Linear(feat_size, hidden_size)
        
        # Import and use curvature difference positional encoding
        from curv_utils import CurvatureDiffPositionalEncoding
        self.curv_pe_diff = CurvatureDiffPositionalEncoding(hidden_size, use_mlp=True, activation='gelu')
        
        self.encoder_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(hidden_size)
        
        self.attr_decoder = AttributeDecoder(hidden_size, feat_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)
        
    def forward(self, x, adj):
        """
        Forward pass with curvature difference positional encoding.
        
        Args:
            x: Node features [N, F]
            adj: Adjacency matrix [N, N]
        
        Returns:
            A_hat: Reconstructed adjacency [N, N]
            X_hat: Reconstructed features [N, F]
        """
        # Project input features
        h = self.input_proj(x)
        
        # Compute and add curvature DIFFERENCE positional encoding
        pe = self.curv_pe_diff(adj)
        h = h + pe
        
        # Graph Transformer encoding
        for layer in self.encoder_layers:
            h = layer(h, adj)
        
        z = self.encoder_norm(h)
        
        # Decode
        X_hat = self.attr_decoder(z)
        A_hat = self.struct_decoder(z)
        
        return A_hat, X_hat


# ============================================================================
# Curvature-based Rotary Position Embedding (RoPE) version
# ============================================================================

class GraphAttentionLayerWithRoPE(nn.Module):
    """
    Graph Attention Layer with Curvature-based Rotary Position Embedding (RoPE).
    
    This layer applies RoPE to Q and K vectors before computing attention,
    where the rotation angles are determined by node curvature values.
    
    Key differences from standard attention:
    1. RoPE is applied to Q and K (rotary embedding)
    2. Curvature provides the "position" signal for rotation
    3. Adjacency bias is still added to attention scores
    
    The RoPE mechanism:
    - Q_rot = rotate(Q, angle=curvature)
    - K_rot = rotate(K, angle=curvature)
    - Attention = softmax(Q_rot @ K_rot^T / sqrt(d) + B)
    
    This allows nodes with different curvatures to naturally attend differently,
    even if their features are similar.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(GraphAttentionLayerWithRoPE, self).__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert hidden_size % 2 == 0, "hidden_size must be even for RoPE"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Learnable edge attention bias
        self.edge_bias = nn.Parameter(torch.zeros(1))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj, cos_cache, sin_cache):
        """
        Args:
            x: Node features [num_nodes, hidden_size]
            adj: Adjacency matrix [num_nodes, num_nodes]
            cos_cache: Cosine values for RoPE [num_nodes, hidden_size]
            sin_cache: Sine values for RoPE [num_nodes, hidden_size]
        
        Returns:
            Updated node features [num_nodes, hidden_size]
        """
        from curv_utils import apply_rotary_pos_emb
        
        num_nodes = x.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        # Need to handle the multi-head dimension
        Q = apply_rotary_pos_emb(Q, cos_cache, sin_cache)  # [N, num_heads, head_dim]
        K = apply_rotary_pos_emb(K, cos_cache, sin_cache)  # [N, num_heads, head_dim]
        
        # Compute attention scores with rotated Q and K
        attn_scores = torch.einsum('qhd,khd->hqk', Q, K) * self.scale
        
        # Add adjacency bias
        edge_mask = (adj.abs() > 1e-6).float()
        edge_mask = edge_mask * (1 - torch.eye(num_nodes, device=adj.device))
        attn_bias = self.edge_bias * edge_mask
        attn_scores = attn_scores + attn_bias.unsqueeze(0)
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.einsum('hqk,khd->hqd', attn_probs, V)
        
        # Reshape back
        out = out.permute(1, 0, 2).contiguous().view(num_nodes, self.hidden_size)
        out = self.out_proj(out)
        
        return out


class GraphTransformerLayerWithRoPE(nn.Module):
    """
    Graph Transformer Layer with RoPE-based attention.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(GraphTransformerLayerWithRoPE, self).__init__()
        
        self.attention = GraphAttentionLayerWithRoPE(hidden_size, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj, cos_cache, sin_cache):
        # Pre-LN: normalize before attention
        attn_out = self.attention(self.norm1(x), adj, cos_cache, sin_cache)
        x = x + self.dropout(attn_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class GraphTransformerWithCurvPERoPE(nn.Module):
    """
    Graph Transformer with Curvature-based Rotary Position Embedding (RoPE).
    
    This model uses curvature values to determine rotary position embeddings,
    following the RoPE design used in LLaMA, DeepSeek, and other modern LLMs.
    
    Key innovation:
    - Instead of additive positional encoding, we use multiplicative rotation
    - Curvature values determine the rotation angles for each node
    - Q and K vectors are rotated before computing attention
    
    Advantages for anomaly detection:
    - Nodes with different curvatures naturally have different attention patterns
    - The rotation operation preserves relative distance relationships
    - Better captures the "structural position" of nodes in the graph
    
    How it works:
    1. Compute node curvature κ for each node
    2. Normalize κ and use it as the "position" signal
    3. For each dimension pair (d_i, d_{i+1}), apply rotation:
       - rotated_d_i = d_i * cos(θ) - d_{i+1} * sin(θ)
       - rotated_d_{i+1} = d_i * sin(θ) + d_{i+1} * cos(θ)
       - where θ = κ * freq_i
    4. Apply rotated Q and K in attention computation
    """
    def __init__(self, feat_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        super(GraphTransformerWithCurvPERoPE, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(feat_size, hidden_size)
        
        # Curvature-based RoPE
        from curv_utils import CurvaturePositionalEncodingRoPE
        self.rope_pe = CurvaturePositionalEncodingRoPE(hidden_size, max_curvature=2.0, base=10000)
        
        # Graph Transformer encoder layers with RoPE attention
        self.encoder_layers = nn.ModuleList([
            GraphTransformerLayerWithRoPE(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final encoder norm
        self.encoder_norm = nn.LayerNorm(hidden_size)
        
        # Decoders
        self.attr_decoder = AttributeDecoder(hidden_size, feat_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)
        
    def forward(self, x, adj):
        """
        Forward pass with curvature-based RoPE.
        
        Args:
            x: Node features [N, F]
            adj: Adjacency matrix [N, N]
        
        Returns:
            A_hat: Reconstructed adjacency [N, N]
            X_hat: Reconstructed features [N, F]
        """
        # Project input features
        h = self.input_proj(x)
        
        # Compute curvature-based RoPE
        cos_cache, sin_cache = self.rope_pe(adj)
        
        # Graph Transformer encoding with RoPE
        for layer in self.encoder_layers:
            h = layer(h, adj, cos_cache, sin_cache)
        
        z = self.encoder_norm(h)
        
        # Decode
        X_hat = self.attr_decoder(z)
        A_hat = self.struct_decoder(z)
        
        return A_hat, X_hat
