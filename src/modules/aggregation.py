"""Aggregation strategies for CV-KAN.

The core CV-KAN mechanism is: aggregate → polar transform → broadcast.
These aggregation modules provide domain-specific strategies while keeping
the rest of the architecture unchanged.

Strategies:
- GlobalMeanAggregation: Standard global mean (text, general)
- MagnitudeWeightedAggregation: Magnitude-weighted mean (parameter-free attention)
- LocalWindowAggregation: 2D local windows (images, preserves spatial locality)
- CausalAggregation: Cumulative causal mean (time series, autoregressive)
- NeighborhoodAggregation: Graph neighborhood mean (GNNs)
"""

import torch
import torch.nn.functional as F
from torch import nn


class GlobalMeanAggregation(nn.Module):
    """Standard global mean aggregation.

    This is the default behavior from the original PolarizingBlock.
    Suitable for text classification, general sequence tasks.

    The mean is bounded regardless of sequence length, providing stability.
    """

    def forward(self, Z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute global mean over tokens.

        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_complex)
            mask: Optional binary mask (batch, n_tokens) where 1=valid, 0=pad

        Returns:
            Aggregated tensor of shape (batch, 1, d_complex)
        """
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, n_tokens, 1)
            sum_Z = (Z * mask_expanded).sum(dim=1, keepdim=True)
            count = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)
            return sum_Z / count
        return Z.mean(dim=1, keepdim=True)


class MagnitudeWeightedAggregation(nn.Module):
    """Magnitude-weighted mean aggregation (parameter-free).

    Uses token magnitudes as implicit attention weights, creating
    a feedback loop: polarized tokens contribute more to aggregates.

    This is attention-free in the sense that no new parameters are
    learned — it just reads the magnitude signal from previous layers.

    Theoretically: if polarization produces magnitude as an importance
    signal, this allows that signal to influence aggregation without
    introducing learned attention weights.
    """

    def forward(self, Z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute magnitude-weighted mean over tokens.

        Args:
            Z: Complex tensor of shape (batch, n_tokens, d_complex)
            mask: Optional binary mask (batch, n_tokens) where 1=valid, 0=pad

        Returns:
            Aggregated tensor of shape (batch, 1, d_complex)
        """
        # Mean magnitude across complex dims as weight
        mag = torch.abs(Z).mean(dim=-1, keepdim=True)  # (batch, n_tokens, 1)

        if mask is not None:
            mag = mag * mask.unsqueeze(-1).float()

        # Normalize weights to sum to 1
        weights = mag / (mag.sum(dim=1, keepdim=True) + 1e-6)

        return (Z * weights).sum(dim=1, keepdim=True)


class LocalWindowAggregation(nn.Module):
    """Local 2D window aggregation for images.

    Preserves spatial structure while keeping the polarization dynamic local.
    Aggregates over local windows instead of global mean.

    Args:
        kernel_size: Size of the local aggregation window (default: 7)
        stride: Stride of the window (default: 1, keeps same spatial size)
    """

    def __init__(self, kernel_size: int = 7, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2  # Same padding

    def forward(
        self,
        Z: torch.Tensor,
        mask: torch.Tensor | None = None,
        spatial_shape: tuple | None = None,
    ) -> torch.Tensor:
        """Compute local window mean for 2D spatial data.

        Args:
            Z: Complex tensor of shape (batch, H*W, d_complex) or (batch, H, W, d_complex)
            mask: Optional binary mask (not typically used for images)
            spatial_shape: (H, W) tuple if Z is flattened

        Returns:
            Aggregated tensor of same shape as input
        """
        original_shape = Z.shape

        # Handle flattened input
        if len(Z.shape) == 3:
            batch, n_tokens, d_complex = Z.shape
            if spatial_shape is None:
                # Assume square
                H = W = int(n_tokens**0.5)
                assert n_tokens == H * W, f"Cannot infer spatial dims from {n_tokens} tokens"
            else:
                H, W = spatial_shape
            Z = Z.view(batch, H, W, d_complex)
        else:
            batch, H, W, d_complex = Z.shape

        # Rearrange for avg_pool2d: (batch, d_complex, H, W)
        # Process real and imaginary parts separately
        Z_real = Z.real.permute(0, 3, 1, 2)  # (batch, d_complex, H, W)
        Z_imag = Z.imag.permute(0, 3, 1, 2)

        # Apply 2D average pooling with same padding
        A_real = F.avg_pool2d(
            F.pad(Z_real, (self.padding, self.padding, self.padding, self.padding), mode="reflect"),
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        A_imag = F.avg_pool2d(
            F.pad(Z_imag, (self.padding, self.padding, self.padding, self.padding), mode="reflect"),
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        # Reconstruct complex and restore shape
        A = torch.complex(A_real, A_imag)
        A = A.permute(0, 2, 3, 1)  # (batch, H, W, d_complex)

        # Return in original shape
        if len(original_shape) == 3:
            A = A.view(batch, -1, d_complex)

        return A


class CausalAggregation(nn.Module):
    """Causal cumulative mean aggregation for time series.

    At each position t, the aggregate A_t is the mean of all tokens from 1 to t.
    This enables autoregressive processing where future tokens don't influence past.

    Natural for time series forecasting, sequence generation, etc.
    """

    def forward(self, Z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute causal cumulative mean.

        A_t = mean(Z_1, Z_2, ..., Z_t) for each t

        Args:
            Z: Complex tensor of shape (batch, seq_len, d_complex)
            mask: Optional binary mask (batch, seq_len)

        Returns:
            Aggregated tensor of shape (batch, seq_len, d_complex)
            where A[:, t, :] = mean(Z[:, :t+1, :])
        """
        batch, seq_len, d_complex = Z.shape
        device = Z.device

        if mask is not None:
            # Masked causal cumsum
            mask_float = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            Z_masked = Z * mask_float

            # Cumulative sum of values
            cumsum = torch.cumsum(Z_masked, dim=1)

            # Cumulative count of valid tokens
            cumcount = torch.cumsum(mask_float, dim=1).clamp(min=1.0)

            return cumsum / cumcount
        # Simple causal cumsum
        cumsum = torch.cumsum(Z, dim=1)
        counts = torch.arange(1, seq_len + 1, device=device, dtype=Z.real.dtype)
        counts = counts.view(1, -1, 1)  # (1, seq_len, 1)
        return cumsum / counts


class NeighborhoodAggregation(nn.Module):
    """Graph neighborhood aggregation.

    For each node, aggregates features from its neighbors as defined by
    an adjacency matrix or edge index.

    Suitable for graph neural networks using CV-KAN layers.

    Args:
        normalize: Whether to normalize by neighbor count (mean) or not (sum)
        self_loop: Whether to include the node itself in aggregation
    """

    def __init__(self, normalize: bool = True, self_loop: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.self_loop = self_loop

    def forward(
        self,
        Z: torch.Tensor,
        adj: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute neighborhood mean for graph data.

        Args:
            Z: Complex tensor of shape (batch, n_nodes, d_complex) or (n_nodes, d_complex)
            adj: Adjacency matrix (batch, n_nodes, n_nodes) or (n_nodes, n_nodes)
                 Non-zero entries indicate edges.
            edge_index: Alternative edge representation (2, n_edges) with [src, dst] rows
            mask: Optional node mask

        Returns:
            Aggregated tensor of same shape as Z
        """
        # Handle unbatched input
        unbatched = len(Z.shape) == 2
        if unbatched:
            Z = Z.unsqueeze(0)
            if adj is not None:
                adj = adj.unsqueeze(0)

        batch, n_nodes, d_complex = Z.shape

        if adj is not None:
            # Adjacency matrix mode
            adj_float = adj.float()

            if self.self_loop:
                # Add self-loops
                eye = torch.eye(n_nodes, device=Z.device, dtype=adj_float.dtype)
                adj_float = adj_float + eye.unsqueeze(0)

            if self.normalize:
                # Compute degree for normalization
                degree = adj_float.sum(dim=-1, keepdim=True).clamp(min=1.0)
                adj_norm = adj_float / degree
            else:
                adj_norm = adj_float

            # Aggregate: A = adj_norm @ Z
            # Handle complex: (batch, n_nodes, n_nodes) @ (batch, n_nodes, d_complex)
            A_real = torch.bmm(adj_norm, Z.real)
            A_imag = torch.bmm(adj_norm, Z.imag)
            A = torch.complex(A_real, A_imag)

        elif edge_index is not None:
            # Edge index mode (COO format)
            src, dst = edge_index[0], edge_index[1]

            if self.self_loop:
                # Add self-loops
                self_loops = torch.arange(n_nodes, device=Z.device)
                src = torch.cat([src, self_loops])
                dst = torch.cat([dst, self_loops])

            # Gather source features for each edge
            # This is a simplified version assuming batch size 1
            if batch > 1:
                raise NotImplementedError("Batched edge_index aggregation not yet supported")

            Z_src = Z[0, src]  # (n_edges, d_complex)

            # Scatter-add to destinations
            A = torch.zeros_like(Z[0])

            # For complex tensors, we need to handle real and imag separately
            A_real = torch.zeros(n_nodes, d_complex, device=Z.device, dtype=Z.real.dtype)
            A_imag = torch.zeros(n_nodes, d_complex, device=Z.device, dtype=Z.imag.dtype)

            A_real.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d_complex), Z_src.real)
            A_imag.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d_complex), Z_src.imag)

            if self.normalize:
                # Count neighbors
                counts = torch.zeros(n_nodes, 1, device=Z.device, dtype=Z.real.dtype)
                ones = torch.ones(len(dst), 1, device=Z.device, dtype=Z.real.dtype)
                counts.scatter_add_(0, dst.unsqueeze(-1), ones)
                counts = counts.clamp(min=1.0)
                A_real = A_real / counts
                A_imag = A_imag / counts

            A = torch.complex(A_real, A_imag).unsqueeze(0)
        else:
            # No graph structure provided, fall back to global mean
            A = Z.mean(dim=1, keepdim=True).expand_as(Z)

        if unbatched:
            A = A.squeeze(0)

        return A
