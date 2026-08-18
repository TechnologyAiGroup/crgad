import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random


def load_anomaly_detection_dataset(dataset, datadir='data'):
    adj_norm, feat, truth, adj = load_pt_dataset(dataset, datadir)
    
    return adj_norm, feat, truth, adj


def load_mat_dataset(dataset, datadir='data'):
    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    truth = data_mat['Label']
    truth = truth.flatten()

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    feat = feat.toarray()
    return adj_norm, feat, truth, adj


def load_pt_dataset(dataset, datadir='data'):
    if dataset.endswith('.pt'):
        file_path = f'{datadir}/{dataset}'
    else:
        file_path = f'{datadir}/{dataset}.pt'
    
    data = torch.load(file_path)
    
    edge_index = data.edge_index  # [2, num_edges]
    x = data.x  # [num_nodes, num_features]
    y = data.y if hasattr(data, 'y') else data.label if hasattr(data, 'label') else None
    
    num_nodes = x.shape[0]

    edge_index_cpu = edge_index.cpu()
    row, col = edge_index_cpu[0].numpy(), edge_index_cpu[1].numpy()
    
    adj_sparse = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    

    adj_sparse = adj_sparse + adj_sparse.T
    adj_sparse = (adj_sparse > 0).astype(int)
    
    feat = x.cpu().numpy()
    
    if y is not None:
        truth = y.cpu().numpy()
        truth = (truth != 0).astype(int)
    else:
        truth = np.zeros(num_nodes)
    
    adj_norm = normalize_adj(adj_sparse + sp.eye(adj_sparse.shape[0]))
    adj_norm = adj_norm.toarray()
    
    adj_with_selfloop = adj_sparse + sp.eye(adj_sparse.shape[0])
    adj = adj_with_selfloop.toarray()
    return adj_norm, feat, truth, adj


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()