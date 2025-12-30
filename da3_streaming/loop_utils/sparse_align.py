"""
Sparse alignment module for point cloud registration using feature-based sampling.

This module provides functionality to:
1. Extract feature points (SIFT, ORB) from images as reliable sampling locations
2. Project feature points to 3D using depth maps
3. Use these sparse 3D-3D correspondences for SIM3/SE3 alignment with RANSAC
"""

import os
import cv2
import numpy as np
import trimesh
from typing import Tuple, Optional, List, Dict

from loop_utils.sim3utils import (
    robust_weighted_estimate_sim3,
    robust_weighted_estimate_sim3_numba,
)
from loop_utils.alignment_torch import robust_weighted_estimate_sim3_torch
from loop_utils.alignment_triton import robust_weighted_estimate_sim3_triton


class FeatureExtractor:
    """Feature extractor supporting SIFT and ORB."""

    def __init__(self, method: str = "sift", max_features: int = 5000):
        """
        Initialize feature extractor.

        Args:
            method: Feature extraction method ('sift' or 'orb')
            max_features: Maximum number of features to extract
        """
        self.method = method.lower()
        self.max_features = max_features

        if self.method == "sift":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif self.method == "orb":
            self.detector = cv2.ORB_create(nfeatures=max_features)
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract keypoint locations from an image.

        Args:
            image: Input image [H, W, 3] uint8 or [H, W] grayscale

        Returns:
            keypoints: [N, 2] array of (x, y) coordinates
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Detect keypoints
        kps = self.detector.detect(gray, None)

        if kps is None or len(kps) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Convert keypoints to numpy array and sort by response (strength)
        kps_sorted = sorted(kps, key=lambda x: x.response, reverse=True)
        keypoints = np.array([kp.pt for kp in kps_sorted], dtype=np.float32)

        return keypoints


def project_2d_to_3d(
    points_2d: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    depth_threshold: float = 0.1,
    max_depth: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 2D points to 3D world coordinates using depth map.

    Args:
        points_2d: [N, 2] array of (x, y) pixel coordinates
        depth: [H, W] depth map
        intrinsics: [3, 3] camera intrinsic matrix
        extrinsics: [3, 4] world-to-camera transform (w2c)
        depth_threshold: Minimum depth value to consider valid
        max_depth: Maximum depth value to consider valid

    Returns:
        points_3d: [M, 3] valid 3D points in world coordinates
        valid_mask: [N] boolean mask of valid points
    """
    N = len(points_2d)
    H, W = depth.shape

    # Round to nearest pixel
    px = np.round(points_2d[:, 0]).astype(np.int32)
    py = np.round(points_2d[:, 1]).astype(np.int32)

    # Check bounds
    valid_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    # Get depth values
    depths = np.zeros(N, dtype=np.float32)
    depths[valid_bounds] = depth[py[valid_bounds], px[valid_bounds]]

    # Check depth validity
    valid_depth = (depths > depth_threshold) & (depths < max_depth)
    valid_mask = valid_bounds & valid_depth

    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32), valid_mask

    # Get valid points
    valid_px = px[valid_mask]
    valid_py = py[valid_mask]
    valid_depths = depths[valid_mask]

    # Unproject to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x_cam = (valid_px - cx) * valid_depths / fx
    y_cam = (valid_py - cy) * valid_depths / fy
    z_cam = valid_depths

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # [M, 3]

    # Transform to world coordinates
    # extrinsics is w2c, we need c2w
    R_w2c = extrinsics[:3, :3]
    t_w2c = extrinsics[:3, 3]

    # c2w: p_world = R_w2c^T @ (p_cam - t_w2c)
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c

    points_world = (R_c2w @ points_cam.T).T + t_c2w

    return points_world.astype(np.float32), valid_mask


def get_sparse_correspondences_same_frames(
    images1: np.ndarray,
    images2: np.ndarray,
    depths1: np.ndarray,
    depths2: np.ndarray,
    confs1: np.ndarray,
    confs2: np.ndarray,
    intrinsics1: np.ndarray,
    intrinsics2: np.ndarray,
    extrinsics1: np.ndarray,
    extrinsics2: np.ndarray,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get sparse 3D-3D correspondences from overlapping frames.

    Since overlapping chunks process the SAME source images, we use feature
    detection to find good sampling locations, then project those same pixel
    locations through both depth predictions to get 3D correspondences.

    Args:
        images1: [N, H, W, 3] images from first chunk (overlap region)
        images2: [N, H, W, 3] images from second chunk (overlap region)
        depths1: [N, H, W] depth maps from first chunk
        depths2: [N, H, W] depth maps from second chunk
        confs1: [N, H, W] confidence maps from first chunk
        confs2: [N, H, W] confidence maps from second chunk
        intrinsics1: [N, 3, 3] camera intrinsics for first chunk
        intrinsics2: [N, 3, 3] camera intrinsics for second chunk
        extrinsics1: [N, 3, 4] camera extrinsics (w2c) for first chunk
        extrinsics2: [N, 3, 4] camera extrinsics (w2c) for second chunk
        config: Configuration dictionary

    Returns:
        pts1: [M, 3] 3D points from first chunk
        pts2: [M, 3] 3D points from second chunk
        weights: [M] confidence weights for each correspondence
    """
    sparse_config = config["Model"].get("Sparse_Align", {})
    method = sparse_config.get("feature_method", "sift")
    max_features = sparse_config.get("max_features", 2000)
    depth_threshold = sparse_config.get("depth_threshold", 0.1)
    max_depth = config["Model"].get("depth_threshold", 100.0)
    conf_threshold_ratio = sparse_config.get("conf_threshold_ratio", 0.5)

    extractor = FeatureExtractor(method=method, max_features=max_features)

    all_pts1 = []
    all_pts2 = []
    all_weights = []

    N = min(len(images1), len(images2))
    total_keypoints = 0
    total_valid = 0

    for i in range(N):
        # Extract keypoints from the image (use image1, they should be identical)
        keypoints = extractor.extract(images1[i])
        total_keypoints += len(keypoints)

        if len(keypoints) == 0:
            continue

        # Compute confidence threshold for this frame
        conf_threshold = min(
            np.median(confs1[i]) * conf_threshold_ratio,
            np.median(confs2[i]) * conf_threshold_ratio
        )

        # Get depth at keypoint locations for both predictions
        H, W = depths1[i].shape
        px = np.round(keypoints[:, 0]).astype(np.int32)
        py = np.round(keypoints[:, 1]).astype(np.int32)

        # Bounds check
        valid_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        for j, (x, y) in enumerate(zip(px, py)):
            if not valid_bounds[j]:
                continue

            d1 = depths1[i, y, x]
            d2 = depths2[i, y, x]
            c1 = confs1[i, y, x]
            c2 = confs2[i, y, x]

            # Check validity
            if d1 < depth_threshold or d1 > max_depth:
                continue
            if d2 < depth_threshold or d2 > max_depth:
                continue
            if c1 < conf_threshold or c2 < conf_threshold:
                continue

            # Project to 3D using each prediction
            kp = keypoints[j:j+1]

            pt1, valid1 = project_2d_to_3d(
                kp, depths1[i], intrinsics1[i], extrinsics1[i],
                depth_threshold, max_depth
            )
            pt2, valid2 = project_2d_to_3d(
                kp, depths2[i], intrinsics2[i], extrinsics2[i],
                depth_threshold, max_depth
            )

            if valid1[0] and valid2[0]:
                all_pts1.append(pt1[0])
                all_pts2.append(pt2[0])
                # Weight by combined confidence
                all_weights.append(np.sqrt(c1 * c2))
                total_valid += 1

    print(f"[Sparse Align] Extracted {total_keypoints} keypoints across {N} frames")
    print(f"[Sparse Align] {total_valid} valid 3D correspondences after filtering")

    if len(all_pts1) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    pts1 = np.array(all_pts1, dtype=np.float32)
    pts2 = np.array(all_pts2, dtype=np.float32)
    weights = np.array(all_weights, dtype=np.float32)

    return pts1, pts2, weights


def ransac_filter_correspondences(
    pts1: np.ndarray,
    pts2: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.5,
    max_iterations: int = 1000,
    min_inliers: int = 10,
    align_method: str = "sim3",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter correspondences using RANSAC with SIM3/SE3 model.

    Args:
        pts1: [N, 3] target points (from first chunk)
        pts2: [N, 3] source points (from second chunk)
        weights: [N] weights
        threshold: Inlier distance threshold
        max_iterations: Maximum RANSAC iterations
        min_inliers: Minimum number of inliers required
        align_method: 'sim3' or 'se3'

    Returns:
        pts1_filtered: [M, 3] filtered target points
        pts2_filtered: [M, 3] filtered source points
        weights_filtered: [M] filtered weights
        inlier_mask: [N] boolean mask of inliers
    """
    N = len(pts1)

    if N < 4:
        return pts1, pts2, weights, np.ones(N, dtype=bool)

    best_inliers = None
    best_num_inliers = 0
    best_transform = None

    # Adaptive threshold based on point cloud scale
    pts_combined = np.vstack([pts1, pts2])
    scale = np.std(pts_combined)
    adaptive_threshold = threshold * scale

    for iteration in range(max_iterations):
        # Sample 4 points (minimum for SIM3)
        indices = np.random.choice(N, size=min(4, N), replace=False)

        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]
        sample_weights = np.ones(len(indices), dtype=np.float32)

        try:
            # Estimate transform from sample
            s, R, t = weighted_estimate_sim3_minimal(
                sample_pts2, sample_pts1, sample_weights, align_method
            )

            # Transform all source points
            transformed = s * (pts2 @ R.T) + t

            # Compute residuals
            residuals = np.linalg.norm(pts1 - transformed, axis=1)

            # Find inliers
            inliers = residuals < adaptive_threshold
            num_inliers = np.sum(inliers)

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers
                best_transform = (s, R, t)

                # Early termination if we have enough inliers
                if num_inliers > 0.8 * N:
                    break

        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_inliers is None or best_num_inliers < min_inliers:
        print(f"[Sparse Align] RANSAC: insufficient inliers ({best_num_inliers}/{N}), using all points")
        return pts1, pts2, weights, np.ones(N, dtype=bool)

    print(f"[Sparse Align] RANSAC: {best_num_inliers}/{N} inliers ({100*best_num_inliers/N:.1f}%)")

    return (
        pts1[best_inliers],
        pts2[best_inliers],
        weights[best_inliers],
        best_inliers,
    )


def weighted_estimate_sim3_minimal(
    src: np.ndarray, tgt: np.ndarray, weights: np.ndarray, align_method: str = "sim3"
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate SIM3/SE3 transformation from weighted point correspondences.
    Minimal implementation for RANSAC sampling.

    Args:
        src: [N, 3] source points
        tgt: [N, 3] target points
        weights: [N] weights
        align_method: 'sim3' or 'se3'

    Returns:
        s: scale factor (1.0 for SE3)
        R: [3, 3] rotation matrix
        t: [3] translation vector
    """
    weights = weights / (np.sum(weights) + 1e-10)

    # Compute weighted centroids
    mu_src = np.sum(weights[:, None] * src, axis=0)
    mu_tgt = np.sum(weights[:, None] * tgt, axis=0)

    # Center the points
    src_centered = src - mu_src
    tgt_centered = tgt - mu_tgt

    # Compute weighted covariance
    H = np.zeros((3, 3))
    for i in range(len(src)):
        H += weights[i] * np.outer(src_centered[i], tgt_centered[i])

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    if align_method == "sim3":
        var_src = np.sum(weights * np.sum(src_centered**2, axis=1))
        s = np.sum(S) / (var_src + 1e-10) if var_src > 1e-10 else 1.0
        # Clamp scale to reasonable range
        s = np.clip(s, 0.1, 10.0)
    else:
        s = 1.0

    # Compute translation
    t = mu_tgt - s * (R @ mu_src)

    return s, R, t


def save_sparse_points_ply(
    pts1: np.ndarray,
    pts2: np.ndarray,
    weights: np.ndarray,
    save_path: str,
    inlier_mask: Optional[np.ndarray] = None,
):
    """
    Save sparse alignment points to PLY files for visualization.

    Args:
        pts1: [N, 3] points from first chunk
        pts2: [N, 3] points from second chunk
        weights: [N] confidence weights
        save_path: Base path for saving (without extension)
        inlier_mask: Optional [N] boolean mask indicating RANSAC inliers
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Normalize weights for color mapping
    if len(weights) > 0:
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            w_norm = (weights - w_min) / (w_max - w_min)
        else:
            w_norm = np.ones_like(weights)
    else:
        w_norm = np.array([])

    # Colors: pts1 = blue, pts2 = red, intensity based on weight
    colors1 = np.zeros((len(pts1), 4), dtype=np.uint8)
    colors1[:, 2] = 255  # Blue channel
    colors1[:, 0] = (w_norm * 100).astype(np.uint8)  # Some red based on weight
    colors1[:, 3] = 255  # Alpha

    colors2 = np.zeros((len(pts2), 4), dtype=np.uint8)
    colors2[:, 0] = 255  # Red channel
    colors2[:, 1] = (w_norm * 100).astype(np.uint8)  # Some green based on weight
    colors2[:, 3] = 255  # Alpha

    # Save pts1 (from first chunk) - blue points
    if len(pts1) > 0:
        pcd1 = trimesh.PointCloud(vertices=pts1, colors=colors1)
        pcd1.export(f"{save_path}_chunk1.ply")
        print(f"[Sparse Align] Saved {len(pts1)} points to {save_path}_chunk1.ply")

    # Save pts2 (from second chunk) - red points
    if len(pts2) > 0:
        pcd2 = trimesh.PointCloud(vertices=pts2, colors=colors2)
        pcd2.export(f"{save_path}_chunk2.ply")
        print(f"[Sparse Align] Saved {len(pts2)} points to {save_path}_chunk2.ply")

    # Save combined with correspondence lines (as a combined point cloud)
    if len(pts1) > 0 and len(pts2) > 0:
        # Combine both point sets
        combined_pts = np.vstack([pts1, pts2])
        combined_colors = np.vstack([colors1, colors2])
        pcd_combined = trimesh.PointCloud(vertices=combined_pts, colors=combined_colors)
        pcd_combined.export(f"{save_path}_combined.ply")
        print(f"[Sparse Align] Saved {len(combined_pts)} combined points to {save_path}_combined.ply")


def sparse_align_point_maps(
    images1: np.ndarray,
    images2: np.ndarray,
    depths1: np.ndarray,
    depths2: np.ndarray,
    confs1: np.ndarray,
    confs2: np.ndarray,
    intrinsics1: np.ndarray,
    intrinsics2: np.ndarray,
    extrinsics1: np.ndarray,
    extrinsics2: np.ndarray,
    config: dict,
    precompute_scale: Optional[float] = None,
    save_dir: Optional[str] = None,
    alignment_idx: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Align point maps using sparse feature-based sampling.

    This function handles the case where overlapping chunks process the SAME
    source images. It uses feature detection to find reliable sampling locations,
    then projects those pixels through both depth predictions to get 3D
    correspondences for alignment.

    Args:
        images1: [N, H, W, 3] images from first chunk (overlap region)
        images2: [N, H, W, 3] images from second chunk (overlap region)
        depths1: [N, H, W] depth maps from first chunk
        depths2: [N, H, W] depth maps from second chunk
        confs1: [N, H, W] confidence maps from first chunk
        confs2: [N, H, W] confidence maps from second chunk
        intrinsics1: [N, 3, 3] camera intrinsics for first chunk
        intrinsics2: [N, 3, 3] camera intrinsics for second chunk
        extrinsics1: [N, 3, 4] camera extrinsics (w2c) for first chunk
        extrinsics2: [N, 3, 4] camera extrinsics (w2c) for second chunk
        config: Configuration dictionary
        precompute_scale: Optional precomputed scale factor
        save_dir: Optional directory to save sparse points for visualization
        alignment_idx: Index of this alignment (for naming saved files)

    Returns:
        s: scale factor
        R: [3, 3] rotation matrix
        t: [3] translation vector
    """
    print("[Sparse Align] Starting sparse alignment...")

    # Get sparse correspondences using feature detection for sampling
    pts1, pts2, weights = get_sparse_correspondences_same_frames(
        images1,
        images2,
        depths1,
        depths2,
        confs1,
        confs2,
        intrinsics1,
        intrinsics2,
        extrinsics1,
        extrinsics2,
        config,
    )

    if len(pts1) < 4:
        raise ValueError(
            f"Not enough sparse correspondences found ({len(pts1)}). "
            "Try adjusting sparse alignment parameters:\n"
            "  - Increase max_features\n"
            "  - Lower conf_threshold_ratio\n"
            "  - Lower depth_threshold\n"
            "Or fall back to dense alignment."
        )

    # Apply precompute_scale if provided
    if precompute_scale is not None:
        pts2 = pts2 * precompute_scale

    # RANSAC filtering
    sparse_config = config["Model"].get("Sparse_Align", {})
    ransac_threshold = sparse_config.get("ransac_threshold", 0.5)
    ransac_iterations = sparse_config.get("ransac_iterations", 1000)
    min_inliers = sparse_config.get("min_inliers", 10)
    align_method = config["Model"]["align_method"]
    if align_method == "scale+se3":
        align_method = "se3"  # Use SE3 for RANSAC when scale is precomputed

    pts1_ransac, pts2_ransac, weights_ransac, inlier_mask = ransac_filter_correspondences(
        pts1,
        pts2,
        weights,
        threshold=ransac_threshold,
        max_iterations=ransac_iterations,
        min_inliers=min_inliers,
        align_method=align_method,
    )

    print(f"[Sparse Align] {len(pts1_ransac)} correspondences after RANSAC filtering")

    # Save sparse points for visualization if save_dir is provided
    if save_dir is not None:
        sparse_save_path = os.path.join(save_dir, "sparse_align", f"align_{alignment_idx:03d}")
        # Save pre-RANSAC points (all correspondences)
        save_sparse_points_ply(pts1, pts2, weights, f"{sparse_save_path}_pre_ransac")
        # Save post-RANSAC points (inliers only)
        save_sparse_points_ply(pts1_ransac, pts2_ransac, weights_ransac, f"{sparse_save_path}_post_ransac", inlier_mask)

    # Use RANSAC filtered points for final estimation
    pts1, pts2, weights = pts1_ransac, pts2_ransac, weights_ransac

    if len(pts1) < 4:
        raise ValueError(
            f"Not enough inliers after RANSAC ({len(pts1)}). "
            "Try increasing ransac_threshold or adjusting feature parameters."
        )

    # Final SIM3/SE3 estimation using IRLS
    align_lib = config["Model"]["align_lib"]
    align_method = config["Model"]["align_method"]

    print(f"[Sparse Align] Running final {align_method.upper()} estimation with {align_lib}...")

    if align_lib == "numba":
        s, R, t = robust_weighted_estimate_sim3_numba(
            pts2,
            pts1,
            weights,
            delta=config["Model"]["IRLS"]["delta"],
            max_iters=config["Model"]["IRLS"]["max_iters"],
            tol=eval(config["Model"]["IRLS"]["tol"]),
            align_method=align_method,
        )
    elif align_lib == "numpy":
        s, R, t = robust_weighted_estimate_sim3(
            pts2,
            pts1,
            weights,
            delta=config["Model"]["IRLS"]["delta"],
            max_iters=config["Model"]["IRLS"]["max_iters"],
            tol=eval(config["Model"]["IRLS"]["tol"]),
            align_method=align_method,
        )
    elif align_lib == "torch":
        s, R, t = robust_weighted_estimate_sim3_torch(
            pts2,
            pts1,
            weights,
            delta=config["Model"]["IRLS"]["delta"],
            max_iters=config["Model"]["IRLS"]["max_iters"],
            tol=eval(config["Model"]["IRLS"]["tol"]),
            align_method=align_method,
        )
    elif align_lib == "triton":
        s, R, t = robust_weighted_estimate_sim3_triton(
            pts2,
            pts1,
            weights,
            delta=config["Model"]["IRLS"]["delta"],
            max_iters=config["Model"]["IRLS"]["max_iters"],
            tol=eval(config["Model"]["IRLS"]["tol"]),
            align_method=align_method,
        )
    else:
        raise ValueError(f"Unknown align_lib: {align_lib}")

    if precompute_scale is not None:
        s = precompute_scale

    print(f"[Sparse Align] Completed. Scale: {s:.6f}")

    return s, R, t
