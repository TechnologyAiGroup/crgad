import torch
from curv_utils import orc_approx_jost_liu
def loss_func(adj, A_hat, attrs, X_hat, alpha, beta_curv=0.0,
              curvature_type='abs', curvature_target='real'):
    # --- 1. Original recon losses ---
    diff_attribute = (X_hat - attrs) ** 2
    attr_err = torch.sqrt(torch.sum(diff_attribute, dim=1) + 1e-12)
    attr_cost = torch.mean(attr_err)
    
    diff_structure = (A_hat - adj) ** 2
    struct_err = torch.sqrt(torch.sum(diff_structure, dim=1) + 1e-12)
    struct_cost = torch.mean(struct_err)
    
    recon_loss_per_node = alpha * attr_err + (1 - alpha) * struct_err  # [n]
    recon_loss_scalar = alpha * attr_cost + (1 - alpha) * struct_cost  # scalar

    # --- 2. Curvature loss ---
    curv_loss_scalar = torch.tensor(0.0, device=A_hat.device, dtype=A_hat.dtype)
    node_curv_scores = torch.zeros(A_hat.shape[0], device=A_hat.device, dtype=A_hat.dtype)

    if beta_curv > 0:
        try:
            kappa_real = orc_approx_jost_liu(adj)
            kappa_recon = orc_approx_jost_liu(A_hat)
            
            edge_mask_real = adj.abs() > 1e-6
            edge_mask_recon = A_hat.abs() > 1e-6
            edge_mask = edge_mask_real & edge_mask_recon
            
            if curvature_target == 'real':
                if curvature_type == 'abs':
                    curv_diff = torch.abs(kappa_recon - kappa_real)
                elif curvature_type == 'square':
                    curv_diff = (kappa_recon - kappa_real) ** 2
                else:
                    raise ValueError(f"Unknown curvature_type {curvature_type}")
                
                curv_term = curv_diff[edge_mask]
                curv_loss_scalar = torch.mean(curv_term)
                
                kappa_masked = curv_diff * edge_mask
                node_degree = torch.sum(edge_mask, dim=1)
                node_curv_sum = torch.sum(kappa_masked, dim=1) + torch.sum(kappa_masked, dim=0)
                node_degree_safe = torch.where(node_degree == 0, torch.ones_like(node_degree), node_degree)
                node_curv_scores = node_curv_sum / (2 * node_degree_safe)
                
        except Exception as e:
            print(f"[CurvLoss Warning] Skipped: {e}")
            curv_loss_scalar = torch.tensor(0.0, device=A_hat.device)

    # --- 3. Total losses with beta_curv as global weight ---
    scalar_loss = (1 - beta_curv) * recon_loss_scalar + beta_curv * curv_loss_scalar
    
    total_loss_per_node = (1 - beta_curv) * recon_loss_per_node + beta_curv * node_curv_scores

    print(f"Recon: {recon_loss_scalar:.4f}, Curv: {curv_loss_scalar:.4f}, Beta: {beta_curv:.2f}")
    
    return total_loss_per_node, scalar_loss, struct_cost, attr_cost, curv_loss_scalar




