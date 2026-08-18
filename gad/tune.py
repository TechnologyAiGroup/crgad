"""
Optuna hyperparameter tuning for graph anomaly detection models.

Usage:
    python tune.py --model_type graph_transformer_curv --dataset gen_1000 --n_trials 50
    python tune.py --model_type gcn --dataset gen_1000 --n_trials 30
"""

import torch
import argparse
import optuna
from optuna.samplers import TPESampler, GridSampler
from sklearn.metrics import roc_auc_score
import json
import os
from datetime import datetime

from model import Dominant
from model_transformer import (
    DominantGraphTransformer, GraphTransformerWithLapPE, GraphTransformerWithCurvPE,
    GraphTransformerWithCurvPEDiff, GraphTransformerWithCurvPERoPE, DominantStandardTransformer
)
from utils import load_anomaly_detection_dataset
from loss import loss_func


# ============================================================================
# Model Factory
# ============================================================================

def get_model(model_type, feat_size, hidden_dim, num_heads, num_layers, dropout):
    """Create model based on model_type and hyperparameters."""
    if model_type == 'gcn':
        return Dominant(
            feat_size=feat_size, 
            hidden_size=hidden_dim, 
            dropout=dropout
        )
    elif model_type == 'graph_transformer':      
        return DominantGraphTransformer(
            feat_size=feat_size,
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == 'graph_transformer_lap':
        return GraphTransformerWithLapPE(
            feat_size=feat_size,
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == 'graph_transformer_curv':
        return GraphTransformerWithCurvPE(
            feat_size=feat_size,
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == 'standard_transformer':      
        return DominantStandardTransformer(
            feat_size=feat_size,
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == 'graph_transformer_curv_diff':
        return GraphTransformerWithCurvPEDiff(
            feat_size=feat_size,
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type == 'graph_transformer_curv_rope':
        return GraphTransformerWithCurvPERoPE(
            feat_size=feat_size,
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Hyperparameter Search Spaces

def get_search_space(model_type, use_grid_search=False):
    """
    Define hyperparameter search space based on model type.
    
    Args:
        model_type: Type of model
        use_grid_search: If True, return discrete choices for grid search
    
    Returns:
        Dictionary of parameter names and their search spaces
    """
    
    # Common parameters for all models
    common_params = {
        'hidden_dim': [32, 64, 128, 256] if use_grid_search else (32, 256),
        'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2] if use_grid_search else (1e-4, 1e-2),
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5] if use_grid_search else (0.1, 0.5),
        'alpha': [0.5, 0.6, 0.7, 0.8, 0.9] if use_grid_search else (0.5, 0.9),
        'epoch': [100, 200, 300] if use_grid_search else (100, 300),
    }
    
    # Transformer-specific parameters
    transformer_params = {
        'num_heads': [2, 4, 8] if use_grid_search else (2, 8),
        'num_layers': [1, 2, 3, 4] if use_grid_search else (1, 4),
    }
    
    # Curvature-specific parameters
    curv_params = {
        'beta_curv': [0.0, 0.01, 0.05, 0.1, 0.2] if use_grid_search else (0.0, 0.2),
        'curv_type': ['abs', 'square'] if use_grid_search else None,
    }
    
    # Build search space based on model type
    search_space = {
        'hidden_dim': common_params['hidden_dim'],
        'lr': common_params['lr'],
        'dropout': common_params['dropout'],
        'alpha': common_params['alpha'],
        'epoch': common_params['epoch'],
    }
    
    # Add transformer parameters for non-GCN models
    if model_type != 'gcn':
        search_space['num_heads'] = transformer_params['num_heads']
        search_space['num_layers'] = transformer_params['num_layers']
    
    # Add curvature parameters for curvature-based models
    if model_type in ['graph_transformer_curv', 'graph_transformer_curv_diff', 'graph_transformer_curv_rope']:
        search_space['beta_curv'] = curv_params['beta_curv']
        search_space['curv_type'] = curv_params['curv_type']
    else:
        # For non-curvature models, set beta_curv to 0
        search_space['beta_curv'] = [0.0] if use_grid_search else None
    
    return search_space


# Training Function for Optuna

def train_and_evaluate(model, adj, attrs, adj_label, adj_input, args, device, label):
    """
    Train model and return best AUC score.
    
    Args:
        model: The model to train
        adj, attrs, adj_label, adj_input: Data tensors
        args: Arguments containing hyperparameters
        device: torch device
        label: Ground truth labels for evaluation
    
    Returns:
        best_auc: Best AUC score achieved during training
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_auc = 0.0
    
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        
        A_hat, X_hat = model(attrs, adj_input)
        
        loss_per_node, scalar_loss, struct_loss, feat_loss, curv_loss = loss_func(
            adj_label, A_hat, attrs, X_hat, 
            alpha=args.alpha,
            beta_curv=args.beta_curv,
            curvature_type=args.curv_type
        )
        
        scalar_loss.backward()
        optimizer.step()
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == args.epoch - 1:
            model.eval()
            with torch.no_grad():
                A_hat, X_hat = model(attrs, adj_input)
                loss_per_node, _, _, _, _ = loss_func(
                    adj_label, A_hat, attrs, X_hat,
                    alpha=args.alpha,
                    beta_curv=0.0
                )
                score = loss_per_node.cpu().numpy()
                auc = roc_auc_score(label, score)
                
                if auc > best_auc:
                    best_auc = auc
    
    return best_auc


def objective(trial, model_type, dataset, device, data_cache):
    """
    Optuna objective function.
    
    Args:
        trial: Optuna trial object
        model_type: Type of model to optimize
        dataset: Dataset name
        device: torch device
        data_cache: Cached data to avoid reloading
    
    Returns:
        best_auc: The metric to optimize
    """
    # Get search space
    search_space = get_search_space(model_type, use_grid_search=False)
    
    # Sample hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', *search_space['hidden_dim'])
    lr = trial.suggest_float('lr', *search_space['lr'], log=True)
    dropout = trial.suggest_float('dropout', *search_space['dropout'])
    alpha = trial.suggest_float('alpha', *search_space['alpha'])
    epoch = trial.suggest_int('epoch', *search_space['epoch'])
    
    if model_type != 'gcn':
        num_heads = trial.suggest_int('num_heads', 2, 8)
        # Ensure hidden_dim is divisible by num_heads
        # Adjust hidden_dim if necessary
        if hidden_dim % num_heads != 0:
            # Round to nearest divisible value
            hidden_dim = (hidden_dim // num_heads) * num_heads
            hidden_dim = max(32, hidden_dim)  # minimum 32
        num_layers = trial.suggest_int('num_layers', *search_space['num_layers'])
    else:
        num_heads = 1
        num_layers = 1
    
    if model_type in ['graph_transformer_curv', 'graph_transformer_curv_diff', 'graph_transformer_curv_rope']:
        beta_curv = trial.suggest_float('beta_curv', *search_space['beta_curv'])
        curv_type = trial.suggest_categorical('curv_type', ['abs', 'square'])
    else:
        beta_curv = 0.0
        curv_type = 'square'
    
    # Create args object
    class Args:
        pass
    args = Args()
    args.hidden_dim = hidden_dim
    args.lr = lr
    args.dropout = dropout
    args.alpha = alpha
    args.epoch = epoch
    args.num_heads = num_heads
    args.num_layers = num_layers
    args.beta_curv = beta_curv
    args.curv_type = curv_type
    
    # Load data (use cache if available)
    if data_cache is None:
        adj, attrs, label, adj_label = load_anomaly_detection_dataset(dataset)
        adj = torch.FloatTensor(adj).to(device)
        adj_label = torch.FloatTensor(adj_label).to(device)
        attrs = torch.FloatTensor(attrs).to(device)
    else:
        adj, attrs, label, adj_label = data_cache
    
    # Get adj_input based on model type
    if model_type.startswith('graph_transformer') or model_type == 'standard_transformer':
        adj_input = adj_label
    else:
        adj_input = adj
    
    # Create model
    feat_size = attrs.size(1)
    model = get_model(model_type, feat_size, hidden_dim, num_heads, num_layers, dropout)
    
    # Print trial info
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: {model_type} on {dataset}")
    print(f"hidden_dim={hidden_dim}, lr={lr:.5f}, dropout={dropout:.2f}")
    print(f"alpha={alpha:.2f}, beta_curv={beta_curv:.3f}, epoch={epoch}")
    if model_type != 'gcn':
        print(f"num_heads={num_heads}, num_layers={num_layers}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"{'='*60}")
    
    # Train and evaluate
    best_auc = train_and_evaluate(model, adj, attrs, adj_label, adj_input, args, device, label)
    
    print(f"Trial {trial.number} finished. Best AUC: {best_auc:.5f}")
    
    return best_auc

# Grid Search with Optuna

def create_grid_search_space(model_type):
    """
    Create search space for GridSampler.
    GridSampler requires a dictionary with parameter names as keys 
    and lists of values as values.
    """
    search_space = get_search_space(model_type, use_grid_search=True)
    
    # Convert to GridSampler format
    grid_space = {}
    for param, values in search_space.items():
        if values is not None:
            grid_space[param] = values
    
    return grid_space


def objective_grid(trial, model_type, dataset, device, data_cache, grid_space):
    """
    Optuna objective function for grid search.
    """
    # Get parameters from grid using trial.suggest_categorical
    # GridSampler requires using suggest_categorical for each parameter
    hidden_dim = trial.suggest_categorical('hidden_dim', grid_space['hidden_dim'])
    lr = trial.suggest_categorical('lr', grid_space['lr'])
    dropout = trial.suggest_categorical('dropout', grid_space['dropout'])
    alpha = trial.suggest_categorical('alpha', grid_space['alpha'])
    epoch = trial.suggest_categorical('epoch', grid_space['epoch'])
    
    if model_type != 'gcn':
        num_heads = trial.suggest_categorical('num_heads', grid_space['num_heads'])
        num_layers = trial.suggest_categorical('num_layers', grid_space['num_layers'])
    else:
        num_heads = 1
        num_layers = 1
    
    if model_type in ['graph_transformer_curv', 'graph_transformer_curv_diff', 'graph_transformer_curv_rope']:
        beta_curv = trial.suggest_categorical('beta_curv', grid_space['beta_curv'])
        curv_type = trial.suggest_categorical('curv_type', grid_space['curv_type'])
    else:
        beta_curv = 0.0
        curv_type = 'square'
    
    # Create args object
    class Args:
        pass
    args = Args()
    args.hidden_dim = hidden_dim
    args.lr = lr
    args.dropout = dropout
    args.alpha = alpha
    args.epoch = epoch
    args.num_heads = num_heads
    args.num_layers = num_layers
    args.beta_curv = beta_curv
    args.curv_type = curv_type
    
    # Load data
    if data_cache is None:
        adj, attrs, label, adj_label = load_anomaly_detection_dataset(dataset)
        adj = torch.FloatTensor(adj).to(device)
        adj_label = torch.FloatTensor(adj_label).to(device)
        attrs = torch.FloatTensor(attrs).to(device)
    else:
        adj, attrs, label, adj_label = data_cache
    
    # Get adj_input
    if model_type.startswith('graph_transformer') or model_type == 'standard_transformer':
        adj_input = adj_label
    else:
        adj_input = adj
    
    # Create model
    feat_size = attrs.size(1)
    
    # Handle num_heads compatibility
    if model_type != 'gcn' and hidden_dim % num_heads != 0:
        # Skip this combination
        print(f"Skipping: hidden_dim={hidden_dim} not divisible by num_heads={num_heads}")
        return 0.0
    
    model = get_model(model_type, feat_size, hidden_dim, num_heads, num_layers, dropout)
    
    # Print trial info
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: {model_type} on {dataset}")
    print(f"Params: {trial.params}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"{'='*60}")
    
    # Train and evaluate
    best_auc = train_and_evaluate(model, adj, attrs, adj_label, adj_input, args, device, label)
    
    print(f"Trial {trial.number} finished. Best AUC: {best_auc:.5f}")
    
    return best_auc

# Main Tuning Function

def run_tuning(args):
    """
    Run hyperparameter tuning with Optuna.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data once and cache it
    adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)
    adj = torch.FloatTensor(adj).to(device)
    adj_label = torch.FloatTensor(adj_label).to(device)
    attrs = torch.FloatTensor(attrs).to(device)
    data_cache = (adj, attrs, label, adj_label)
    
    # Create study
    if args.search_type == 'grid':
        # Grid search
        grid_space = create_grid_search_space(args.model_type)
        
        # Filter out incompatible combinations for transformers
        if args.model_type != 'gcn':
            # Only keep num_heads that divide hidden_dim
            filtered_space = {}
            for param, values in grid_space.items():
                filtered_space[param] = values
            grid_space = filtered_space
        
        print(f"\nGrid Search Space: {grid_space}")
        
        # Calculate total combinations
        total_combinations = 1
        for v in grid_space.values():
            total_combinations *= len(v)
        print(f"Total combinations: {total_combinations}")
        
        sampler = GridSampler(grid_space)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f"{args.model_type}_{args.dataset}"
        )
        
        study.optimize(
            lambda trial: objective_grid(trial, args.model_type, args.dataset, device, data_cache, grid_space),
            n_trials=args.n_trials if args.n_trials > 0 else total_combinations,
            show_progress_bar=True
        )
    else:
        # TPE (Bayesian optimization)
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f"{args.model_type}_{args.dataset}"
        )
        
        study.optimize(
            lambda trial: objective(trial, args.model_type, args.dataset, device, data_cache),
            n_trials=args.n_trials,
            show_progress_bar=True
        )
    
    # Print results
    print("\n" + "="*60)
    print("TUNING COMPLETED!")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best AUC: {study.best_value:.5f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'model_type': args.model_type,
        'dataset': args.dataset,
        'best_auc': float(study.best_value),
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat(),
        'all_trials': [
            {
                'trial': t.number,
                'params': t.params,
                'auc': float(t.value) if t.value is not None else None
            }
            for t in study.trials
        ]
    }
    
    # Create results directory
    os.makedirs('tune_results', exist_ok=True)
    result_file = f"tune_results/{args.model_type}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    
    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    
    # Required arguments
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['gcn', 'graph_transformer', 'graph_transformer_lap', 
                                 'graph_transformer_curv', 'graph_transformer_curv_diff',
                                 'graph_transformer_curv_rope', 'standard_transformer'],
                        help='Model type to tune')
    parser.add_argument('--dataset', type=str, default='gen_1000',
                        help='Dataset name')
    
    # Tuning parameters
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials (for TPE) or max trials (for grid)')
    parser.add_argument('--search_type', type=str, default='tpe',
                        choices=['tpe', 'grid'],
                        help='Search type: tpe (Bayesian) or grid (exhaustive)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    print("Args:", args)
    
    run_tuning(args)
