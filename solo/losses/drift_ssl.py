import torch
import torch.nn.functional as F


def drift_ssl_loss_func(
        p: torch.Tensor,
        z_target: torch.Tensor,
        labels: torch.Tensor,
        tau: float,
        adaptive_tau: bool = False,
        tau_alpha: float = 0.7
) -> tuple[torch.Tensor, float]:
    """
    Computes Tangent Drift SSL Loss.

    Args:
        p: [2B, D] L2-normalized predictor outputs
        z_target: [2B, D] L2-normalized target embeddings (momentum net output)
        labels: [2B] instance indices to identify positive pairs
        tau: (float) Bandwidth parameter for the Gaussian Kernel

    Returns:
        loss: The MSE loss matched to the Extrinsic Spherical Gradient.
        metrics: Dictionary containing detailed telemetry (drift magnitudes and effective dim).
    """
    M, D = z_target.shape

    # Ensure targets are detached (Stop-Gradient) and normalized
    z_sg = F.normalize(z_target.detach(), dim=-1)
    p = F.normalize(p, dim=-1)

    # 1. Compute Masks
    labels = labels.view(-1, 1)
    is_same_image = torch.eq(labels, labels.T)
    self_mask = torch.eye(M, dtype=torch.bool, device=z_sg.device)

    pos_mask = is_same_image & ~self_mask
    neg_mask = ~is_same_image

    # 2. Kernel & Drift Computation
    sim_matrix = torch.matmul(z_sg, z_sg.T)
    dist_matrix = torch.sqrt(torch.clamp(2.0 - 2.0 * sim_matrix, min=1e-8))

    # Calculate effective distance (mean distance between positive pairs)
    # Using detach() to prevent gradients from flowing through the tau calculation
    d_eff = dist_matrix[pos_mask].detach().mean()

    if adaptive_tau:
        # \tau = \alpha * d_{eff}. Clamped for numerical stability
        computed_tau = torch.clamp(tau_alpha * d_eff, min=0.02, max=2.0)
    else:
        computed_tau = tau

    kernel_weights = torch.exp(-dist_matrix / computed_tau)

    # Positives Centroid
    k_pos = kernel_weights.masked_fill(~pos_mask, 0.0)
    w_pos = k_pos / torch.clamp(k_pos.sum(dim=1, keepdim=True), min=1e-8)
    mu_pos = torch.matmul(w_pos, z_sg)

    # Negatives Centroid
    k_neg = kernel_weights.masked_fill(~neg_mask, 0.0)
    w_neg = k_neg / torch.clamp(k_neg.sum(dim=1, keepdim=True), min=1e-8)
    mu_neg = torch.matmul(w_neg, z_sg)

    # Component Forces: Pull towards positive, Pull towards negative (for tracking)
    v_pos = mu_pos - z_sg
    v_neg = mu_neg - z_sg

    # V(i) = mu+(i) - mu-(i) (The Total Drift Vector)
    V = mu_pos - mu_neg

    # 3. Tangent-Space Projection
    inner_product = (V * z_sg).sum(dim=1, keepdim=True)
    V_perp = V - inner_product * z_sg

    # Target for the predictor
    target = z_sg + V_perp

    # 4. MSE Loss
    loss = F.mse_loss(p, target)

    # 5. Calculate Advanced Metrics
    # Drift Magnitudes
    drift_mag = torch.norm(V, dim=1).mean().item()
    drift_pos_mag = torch.norm(v_pos, dim=1).mean().item()
    drift_neg_mag = torch.norm(v_neg, dim=1).mean().item()

    # Effective Dimensionality: ED = (Tr(C))^2 / Tr(C^2)
    z_centered = z_sg - z_sg.mean(dim=0, keepdim=True)
    C = torch.matmul(z_centered.T, z_centered) / (M - 1)
    tr_C = torch.trace(C)
    tr_C2 = torch.trace(torch.matmul(C, C))
    eff_dim = (tr_C ** 2 / torch.clamp(tr_C2, min=1e-8)).item()

    metrics = {
        "drift_mag": drift_mag,
        "drift_pos_mag": drift_pos_mag,
        "drift_neg_mag": drift_neg_mag,
        "eff_dim": eff_dim,
        "d_eff": d_eff.item(),
        "active_tau": computed_tau.item() if isinstance(computed_tau, torch.Tensor) else computed_tau
    }

    return loss, metrics
