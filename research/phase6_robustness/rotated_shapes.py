import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RotatedShapesDataset(Dataset):
    """Synthetic dataset of 2D geometric shapes (Square, Triangle)
    with controlled rotation for robustness testing.

    Images are 28x28 grayscale (1 channel).
    """

    def __init__(self, n_samples=5000, img_size=28, rotation_range=(0, 2 * np.pi), seed=42):
        self.n_samples = n_samples
        self.img_size = img_size
        self.rotation_range = rotation_range
        self.seed = seed

        # Pre-generate data
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        data = torch.zeros(self.n_samples, 1, self.img_size, self.img_size)
        labels = torch.zeros(self.n_samples, dtype=torch.long)

        # Center of image
        cx, cy = self.img_size / 2, self.img_size / 2
        # Scale of shapes (radius)
        r = self.img_size * 0.25

        for i in range(self.n_samples):
            # 0: Square, 1: Triangle
            label = i % 2
            labels[i] = label

            # Random rotation
            theta = np.random.uniform(self.rotation_range[0], self.rotation_range[1])

            # Generate vertices
            if label == 0:  # Square
                # 4 points at 45, 135, 225, 315 degrees
                angles = np.array([1, 3, 5, 7]) * np.pi / 4
            else:  # Triangle
                # 3 points at 90, 210, 330 degrees
                angles = np.array([1, 5, 9]) * np.pi / 6

            # Rotate vertices
            angles += theta

            vx = cx + r * np.cos(angles)
            vy = cy + r * np.sin(angles)

            # Rasterize
            # For 28x28, we can check every pixel against the polygon edges
            grid_y, grid_x = torch.meshgrid(
                torch.arange(self.img_size), torch.arange(self.img_size), indexing="ij"
            )

            mask = self._point_in_polygon(grid_x, grid_y, vx, vy)
            data[i, 0] = mask.float()

            # Add noise
            data[i, 0] += torch.randn(self.img_size, self.img_size) * 0.1

        return data, labels

    def _point_in_polygon(self, x, y, vx, vy):
        # Convex polygon containment check using cross product
        n_verts = len(vx)
        mask_ccw = torch.ones_like(x, dtype=torch.bool)
        mask_cw = torch.ones_like(x, dtype=torch.bool)

        # Check if point is to the left of all edges (CCW winding)
        # OR to the right of all edges (CW winding)

        for i in range(n_verts):
            j = (i + 1) % n_verts

            # Edge vector
            ex = vx[j] - vx[i]
            ey = vy[j] - vy[i]

            # Point vector
            px = x - vx[i]
            py = y - vy[i]

            # Cross product
            cross = ex * py - ey * px

            mask_ccw &= cross >= 0
            mask_cw &= cross <= 0

        return mask_ccw | mask_cw

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_rotated_shapes_loaders(train_size=4000, test_size=1000, batch_size=64):
    # Train on "Upright" (limited rotation +/- 30 deg)
    train_ds = RotatedShapesDataset(
        n_samples=train_size, rotation_range=(-np.pi / 6, np.pi / 6), seed=42
    )

    # Test on "Rotated" (full rotation 0-360 deg)
    test_ds = RotatedShapesDataset(n_samples=test_size, rotation_range=(0, 2 * np.pi), seed=123)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
