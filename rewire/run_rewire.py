from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected

import time
import torch
import numpy as np
import pandas as pd
from preprocessing import rewire
import os
import argparse

def load_my_dataset(file_path):
    data = torch.load(file_path)
    
    if isinstance(data, dict):
        from torch_geometric.data import Data
        x = data['x']
        edge_index = data['edge_index']
        y = data['y']
        edge_attr = data.get('edge_attr', None)
    else:
        x = data.x
        edge_index = data.edge_index
        y = data.y
        edge_attr = getattr(data, 'edge_attr', None)
    
    if y is not None:
        if y.dtype == torch.bool:
            y = y.long()
        elif y.dtype.is_floating_point:
            if y.dim() > 1 and y.shape[1] > 1:
                y = y.argmax(dim=1).long()
            else:
                y = y.round().long()
        elif y.dtype != torch.long:
            y = y.long()
    else:
        raise ValueError(f"Dataset {file_path} has no labels (y is None)")

    from torch_geometric.data import Data
    pyg_data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    
    pyg_data.edge_index = to_undirected(pyg_data.edge_index)
    
    class DatasetWrapper:
        def __init__(self, data):
            self._data = data
            self._length = 1
        
        def __len__(self):
            return self._length
        
        def __getitem__(self, idx):
            if idx == 0:
                return self._data
            else:
                raise IndexError("DatasetWrapper only contains one graph")
        
        @property
        def data(self):
            return self._data
    
    return DatasetWrapper(pyg_data)

def save_graph_data(original_data, rewired_data, dataset_name):
    os.makedirs(f"graphData/{dataset_name}_rewired", exist_ok=True)
    
    original_dir = f"graphData/{dataset_name}_rewired/original"
    os.makedirs(original_dir, exist_ok=True)
    
    with open(f"{original_dir}/features.txt", "w") as f:
        f.write(f"# Node features for {dataset_name} (original)\n")
        f.write(f"# Number of nodes: {original_data.x.shape[0]}\n")
        f.write(f"# Feature dimension: {original_data.x.shape[1]}\n")
        f.write("# Format: node_id feature1 feature2 ... featureN\n")
        features = original_data.x.cpu().numpy()
        for i, feature in enumerate(features):
            feature_str = " ".join([str(x) for x in feature])
            f.write(f"{i} {feature_str}\n")
    
    with open(f"{original_dir}/edges.txt", "w") as f:
        f.write(f"# Edge list for {dataset_name} (original)\n")
        f.write(f"# Number of nodes: {original_data.x.shape[0]}\n")
        f.write(f"# Number of edges: {original_data.edge_index.shape[1]}\n")
        f.write("# Format: source_node target_node\n")
        edges = original_data.edge_index.cpu().t().numpy() 
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")
    
    with open(f"{original_dir}/labels.txt", "w") as f:
        f.write(f"# Node labels for {dataset_name} (original)\n")
        f.write(f"# Number of nodes: {original_data.x.shape[0]}\n")
        f.write("# Format: node_id label\n")
        labels = original_data.y.cpu().numpy() 
        for i, label in enumerate(labels):
            f.write(f"{i} {label}\n")
    
    rewired_dir = f"graphData/{dataset_name}_rewired/rewired"
    os.makedirs(rewired_dir, exist_ok=True)
    
    with open(f"{rewired_dir}/features.txt", "w") as f:
        f.write(f"# Node features for {dataset_name} (rewired)\n")
        f.write(f"# Number of nodes: {rewired_data.x.shape[0]}\n")
        f.write(f"# Feature dimension: {rewired_data.x.shape[1]}\n")
        f.write("# Format: node_id feature1 feature2 ... featureN\n")
        features = rewired_data.x.cpu().numpy() 
        for i, feature in enumerate(features):
            feature_str = " ".join([str(x) for x in feature])
            f.write(f"{i} {feature_str}\n")
    
    with open(f"{rewired_dir}/edges.txt", "w") as f:
        f.write(f"# Edge list for {dataset_name} (rewired)\n")
        f.write(f"# Number of nodes: {rewired_data.x.shape[0]}\n")
        f.write(f"# Number of edges: {rewired_data.edge_index.shape[1]}\n")
        f.write("# Format: source_node target_node\n")
        edges = rewired_data.edge_index.cpu().t().numpy()  
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")
    
    with open(f"{rewired_dir}/labels.txt", "w") as f:
        f.write(f"# Node labels for {dataset_name} (rewired)\n")
        f.write(f"# Number of nodes: {rewired_data.x.shape[0]}\n")
        f.write("# Format: node_id label\n")
        labels = rewired_data.y.cpu().numpy() 
        for i, label in enumerate(labels):
            f.write(f"{i} {label}\n")
    
    print(f"Graph data saved for {dataset_name}")
    print(f"  Original: {original_dir}")
    print(f"  Rewired: {rewired_dir}")

books = load_my_dataset("data/books.pt")
disney = load_my_dataset("data/disney.pt")
datasets = {
    "books": books,
    "disney": disney
}

for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

parser = argparse.ArgumentParser(description='rewire', argument_default=argparse.SUPPRESS)

parser.add_argument('--num_iterations', type=int)
parser.add_argument('--dataset', type=str, help='name of dataset to use')
parser.add_argument('--batch_add', type=int)
parser.add_argument('--batch_remove', type=int)
args = parser.parse_args()

if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    if name in datasets:
        datasets = {name: datasets[name]}
    else:
        print(f"Dataset {name} not found. Available datasets: {list(datasets.keys())}")
        exit(1)

for key in datasets:
    accuracies = []
    dataset = datasets[key]
    
    original_data = dataset.data.clone().cpu()
    start = time.time()
    print(f"[INFO]  hyper-parameter : num_iterations = {args.num_iterations}")
    print(f"[INFO]  hyper-parameter : batch_add = {args.batch_add}")
    print(f"[INFO]  hyper-parameter : batch_remove = {args.batch_remove}")
    dataset.data.edge_index, dataset.data.edge_type = rewire.rewireProcess(dataset.data, 
            loops=args.num_iterations, 
            remove_edges=False, 
            is_undirected=True,
            batch_add=args.batch_add,
            batch_remove=args.batch_remove,
            dataset_name=key,
            graph_index=0)
    print(len(dataset.data.edge_type))
    end = time.time()
    rewiring_duration = end - start

    rewired_data = dataset.data.cpu()
    
    save_graph_data(original_data, rewired_data, key)
    print("finish")