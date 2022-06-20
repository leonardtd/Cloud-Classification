import torch


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims) 
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    # SQRT((x - x.T)^2 = X^2 -2XX.T + X.T^2)
    return x_square + x_inner + x_square.transpose(2, 1)


def dense_knn_matrix(x, k=16):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points ,k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.flatten(2).transpose(2, 1)

        batch_size, n_points, n_dims = x.shape
        _, nn_idx = torch.topk(-pairwise_distance(x.detach()), k=k)
        center_idx = (
            torch.arange(0, n_points, device=x.device)
            .expand(batch_size, k, -1)
            .transpose(2, 1)
        )

    # 2 (neighbor-src), BATCH_SIZE, NUM_PATCHES, K
    return torch.stack((nn_idx, center_idx), dim=0)
