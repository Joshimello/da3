"""
Evaluation module for computing reconstruction accuracy metrics.

This module provides:
1. Accuracy metrics: Chamfer distance, Accuracy, Completeness, F-score
2. Sim3 transformation utilities for manual alignment
3. Correspondence-based Sim3 refinement using IRLS
"""

import numpy as np
import trimesh
from typing import Tuple, Optional, Dict
from scipy.spatial import cKDTree
import torch

from loop_utils.alignment_torch import robust_weighted_estimate_sim3_torch


def load_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load point cloud from PLY file and return points and colors as numpy arrays."""
    pcd = trimesh.load(path)
    points = np.array(pcd.vertices)

    if hasattr(pcd, 'colors') and pcd.colors is not None and len(pcd.colors) > 0:
        colors = np.array(pcd.colors)[:, :3]
    else:
        colors = None

    return points, colors


def compute_nearest_neighbor_distances_cpu(
    query: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Compute nearest neighbor distances from query to target points (CPU).

    Args:
        query: [N, 3] query points
        target: [M, 3] target points

    Returns:
        [N] distances to nearest neighbors
    """
    tree = cKDTree(target)
    distances, _ = tree.query(query, k=1)
    return distances


def compute_nearest_neighbor_distances_gpu(
    query: np.ndarray,
    target: np.ndarray,
    batch_size: int = 8192,
    target_batch_size: int = 50000,
) -> np.ndarray:
    """
    Compute nearest neighbor distances from query to target points (GPU).

    Args:
        query: [N, 3] query points
        target: [M, 3] target points
        batch_size: Batch size for query points
        target_batch_size: Batch size for target points

    Returns:
        [N] distances to nearest neighbors
    """
    device = torch.device("cuda")

    all_min_dists = []

    # Pre-load target batches to GPU
    target_batches = []
    for j in range(0, len(target), target_batch_size):
        target_batch = target[j:j + target_batch_size]
        target_t = torch.from_numpy(target_batch).float().to(device)
        target_batches.append(target_t)

    for i in range(0, len(query), batch_size):
        batch = query[i:i + batch_size]
        query_t = torch.from_numpy(batch).float().to(device)

        # Compute distances to all target batches
        batch_min_dists = []
        for target_t in target_batches:
            dists = torch.cdist(query_t, target_t)
            batch_min_dists.append(dists.min(dim=1).values)
            del dists

        # Get minimum distance across all target batches
        min_dists = torch.stack(batch_min_dists, dim=1).min(dim=1).values
        all_min_dists.append(min_dists.cpu().numpy())

    return np.concatenate(all_min_dists)


def compute_nearest_neighbor_distances(
    query: np.ndarray,
    target: np.ndarray,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute nearest neighbor distances, automatically choosing CPU or GPU.

    Args:
        query: [N, 3] query points
        target: [M, 3] target points
        use_gpu: Whether to use GPU acceleration

    Returns:
        [N] distances to nearest neighbors
    """
    if use_gpu and torch.cuda.is_available():
        return compute_nearest_neighbor_distances_gpu(query, target)
    else:
        return compute_nearest_neighbor_distances_cpu(query, target)


def find_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find nearest neighbor correspondences between source and target.

    Args:
        source: [N, 3] source points
        target: [M, 3] target points
        max_distance: Maximum correspondence distance

    Returns:
        source_corr: [K, 3] corresponding source points
        target_corr: [K, 3] corresponding target points
        distances: [K] distances between correspondences
        mask: [N] boolean mask of valid correspondences
    """
    tree = cKDTree(target)
    distances, indices = tree.query(source, k=1)

    # Filter by max distance
    mask = distances < max_distance

    source_corr = source[mask]
    target_corr = target[indices[mask]]
    distances_filtered = distances[mask]

    return source_corr, target_corr, distances_filtered, mask


def refine_alignment_sim3(
    source: np.ndarray,
    target: np.ndarray,
    max_correspondence_distance: float = np.inf,
    align_method: str = "sim3",
    irls_delta: float = 0.1,
    irls_max_iters: int = 20,
    irls_tol: float = 1e-9,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Refine alignment using correspondence-based IRLS Sim3/SE3 estimation.

    This uses the same robust_weighted_estimate_sim3_torch function used
    in the reconstruction pipeline for chunk alignment.

    Args:
        source: [N, 3] source points (will be transformed to match target)
        target: [M, 3] target points (reference, stays fixed)
        max_correspondence_distance: Maximum distance for valid correspondences
        align_method: "sim3" or "se3"
        irls_delta: Huber loss delta for IRLS
        irls_max_iters: Maximum IRLS iterations
        irls_tol: IRLS convergence tolerance
        verbose: If True, print progress

    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
        s: scale factor
        info: Dictionary with alignment information
    """
    info = {
        'num_correspondences': 0,
        'mean_distance_before': 0.0,
        'mean_distance_after': 0.0,
    }

    # Find nearest neighbor correspondences
    source_corr, target_corr, distances, mask = find_correspondences(
        source, target, max_correspondence_distance
    )

    if len(source_corr) < 10:
        raise ValueError(f"Not enough correspondences found ({len(source_corr)}). "
                        "Try adjusting the manual alignment first.")

    info['num_correspondences'] = len(source_corr)
    info['mean_distance_before'] = float(np.mean(distances))

    if verbose:
        print(f"[Refine] Found {len(source_corr)} correspondences")
        print(f"[Refine] Mean distance before: {info['mean_distance_before']:.6f}")

    # Use distance-based weights: closer points get higher weight
    # Weight = 1 / (1 + distance), normalized
    weights = 1.0 / (1.0 + distances)
    weights = weights / np.sum(weights)
    weights = weights.astype(np.float32)

    # Use robust_weighted_estimate_sim3_torch for IRLS estimation
    # Note: this transforms source to target, so we pass (source_corr, target_corr)
    if verbose:
        print(f"[Refine] Running IRLS {align_method.upper()} estimation...")

    s, R, t = robust_weighted_estimate_sim3_torch(
        source_corr.astype(np.float32),
        target_corr.astype(np.float32),
        weights,
        delta=irls_delta,
        max_iters=irls_max_iters,
        tol=irls_tol,
        align_method=align_method,
    )

    # Compute error after alignment
    source_transformed = apply_sim3(source_corr, R, t, s)
    distances_after = np.linalg.norm(target_corr - source_transformed, axis=1)
    info['mean_distance_after'] = float(np.mean(distances_after))

    if verbose:
        print(f"[Refine] Mean distance after: {info['mean_distance_after']:.6f}")
        print(f"[Refine] Scale: {s:.6f}")

    return R, t, s, info


def apply_sim3(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
) -> np.ndarray:
    """
    Apply Sim3 transformation: points_transformed = s * R @ points + t

    Args:
        points: [N, 3] points
        R: [3, 3] rotation matrix
        t: [3] translation vector
        s: scale factor

    Returns:
        [N, 3] transformed points
    """
    return s * (points @ R.T) + t


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles (in radians) from rotation matrix.

    Args:
        R: [3, 3] rotation matrix

    Returns:
        roll: Rotation around X axis
        pitch: Rotation around Y axis
        yaw: Rotation around Z axis
    """
    # Handle gimbal lock cases
    if abs(R[2, 0]) >= 1.0 - 1e-6:
        # Gimbal lock
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))

    return roll, pitch, yaw


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create rotation matrix from Euler angles (in radians).

    Args:
        roll: Rotation around X axis
        pitch: Rotation around Y axis
        yaw: Rotation around Z axis

    Returns:
        [3, 3] rotation matrix
    """
    # Rotation around X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation around Y (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation around Z (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def compute_accuracy_metrics(
    reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: list = [0.01, 0.02, 0.05, 0.1],
    use_gpu: bool = True,
) -> Dict:
    """
    Compute reconstruction accuracy metrics.

    Metrics:
    - Chamfer Distance: Average of mean distances in both directions
    - Accuracy: Mean distance from reconstruction to GT (lower is better)
    - Completeness: Mean distance from GT to reconstruction (lower is better)
    - Precision@t: % of reconstruction points within threshold t of GT
    - Recall@t: % of GT points within threshold t of reconstruction
    - F-score@t: Harmonic mean of Precision and Recall at threshold t

    Args:
        reconstruction: [N, 3] reconstructed point cloud
        ground_truth: [M, 3] ground truth point cloud
        thresholds: List of distance thresholds for Precision/Recall/F-score
        use_gpu: Whether to use GPU acceleration

    Returns:
        Dictionary with all metrics
    """
    # Compute distances: reconstruction -> GT (for accuracy/precision)
    dist_rec_to_gt = compute_nearest_neighbor_distances(
        reconstruction, ground_truth, use_gpu=use_gpu
    )

    # Compute distances: GT -> reconstruction (for completeness/recall)
    dist_gt_to_rec = compute_nearest_neighbor_distances(
        ground_truth, reconstruction, use_gpu=use_gpu
    )

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = float(np.mean(dist_rec_to_gt))
    metrics['completeness'] = float(np.mean(dist_gt_to_rec))
    metrics['chamfer_distance'] = (metrics['accuracy'] + metrics['completeness']) / 2

    # Median versions (more robust to outliers)
    metrics['accuracy_median'] = float(np.median(dist_rec_to_gt))
    metrics['completeness_median'] = float(np.median(dist_gt_to_rec))

    # Threshold-based metrics
    for t in thresholds:
        precision = np.mean(dist_rec_to_gt < t) * 100  # percentage
        recall = np.mean(dist_gt_to_rec < t) * 100     # percentage

        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = 0.0

        metrics[f'precision@{t}'] = float(precision)
        metrics[f'recall@{t}'] = float(recall)
        metrics[f'f_score@{t}'] = float(f_score)

    # Statistics
    metrics['num_reconstruction_points'] = len(reconstruction)
    metrics['num_gt_points'] = len(ground_truth)

    return metrics


def save_aligned_point_cloud(
    points: np.ndarray,
    output_path: str,
    colors: Optional[np.ndarray] = None,
):
    """
    Save aligned point cloud to PLY file.

    Args:
        points: [N, 3] points
        output_path: Output PLY path
        colors: Optional [N, 3] or [N, 4] colors (uint8)
    """
    if colors is not None:
        if colors.shape[1] == 3:
            colors = np.hstack([colors, np.full((len(colors), 1), 255, dtype=np.uint8)])
        pcd = trimesh.PointCloud(vertices=points, colors=colors)
    else:
        pcd = trimesh.PointCloud(vertices=points)

    pcd.export(output_path)
    print(f"[Eval] Saved aligned point cloud to: {output_path}")
