import torch

def compute_grid_points(feature_volume: torch.Tensor):
    d, h, w = feature_volume.shape[-3:]
    grids = torch.meshgrid(
        torch.linspace(-1, 1, d),
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing="ij"
    )

    # NOTE: The 3D coordinates have to correspond to width, height and depth in this order.
    # This is what torch.grid_sample expects. So, we flip.
    return torch.stack(grids, dim=-1).to(feature_volume.device).flip(-1)

def compute_2d_grid_points(h, w, device=torch.device("cpu")):
    grids = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing="ij"
    )

    # NOTE: The 3D coordinates have to correspond to width, height and depth in this order.
    # This is what torch.grid_sample expects. So, we flip.
    return torch.stack(grids, dim=-1).to(device).flip(-1)
