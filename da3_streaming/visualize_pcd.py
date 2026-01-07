import argparse
import glob
import os
import re
import time

import numpy as np
import trimesh
import viser

from eval_reconstruction import (
    compute_accuracy_metrics,
    apply_sim3,
    rotation_matrix_from_euler,
    load_point_cloud,
)


def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds from da3-streaming experiments.")
    parser.add_argument("--exp_folder", type=str, required=True, help="Path to the experiment folder.")
    parser.add_argument("--show_sparse", action="store_true", help="Show sparse alignment points.")
    parser.add_argument("--gt_pcd", type=str, default=None, help="Path to ground truth point cloud PLY file.")
    args = parser.parse_args()

    if not os.path.isdir(args.exp_folder):
        print(f"Error: Experiment folder not found at {args.exp_folder}")
        return

    server = viser.ViserServer()

    # --- GUI controls ---
    server.gui.add_markdown("### Point Cloud Visibility")
    show_chunks_gui = server.gui.add_checkbox("Show Chunks", initial_value=True)
    show_merged_gui = server.gui.add_checkbox("Show Merged", initial_value=True)
    show_cameras_gui = server.gui.add_checkbox("Show Cameras", initial_value=True)

    server.gui.add_markdown("### Sparse Alignment Points")
    show_sparse_gui = server.gui.add_checkbox("Show Sparse Points", initial_value=args.show_sparse)
    show_sparse_chunk1_gui = server.gui.add_checkbox("  - Chunk1 (Blue)", initial_value=True)
    show_sparse_chunk2_gui = server.gui.add_checkbox("  - Chunk2 (Red)", initial_value=True)
    show_sparse_combined_gui = server.gui.add_checkbox("  - Combined", initial_value=False)
    show_sparse_pre_ransac_gui = server.gui.add_checkbox("  - Pre-RANSAC", initial_value=False)
    show_sparse_post_ransac_gui = server.gui.add_checkbox("  - Post-RANSAC", initial_value=True)

    server.gui.add_markdown("### Point Sizes")
    merged_point_size_slider = server.gui.add_slider(
        "Merged Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
    )
    chunk_point_size_slider = server.gui.add_slider(
        "Chunk Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
    )
    camera_size_slider = server.gui.add_slider(
        "Camera Size", min=0.001, max=0.5, step=0.001, initial_value=0.05
    )
    sparse_point_size_slider = server.gui.add_slider(
        "Sparse Point Size", min=0.005, max=0.2, step=0.005, initial_value=0.03
    )

    # --- Ground Truth Alignment Controls ---
    server.gui.add_markdown("---")
    server.gui.add_markdown("### Ground Truth Alignment")
    show_gt_gui = server.gui.add_checkbox("Show Ground Truth", initial_value=True)
    gt_point_size_slider = server.gui.add_slider(
        "GT Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
    )

    # Translation controls
    server.gui.add_markdown("#### Translation")
    gt_tx_slider = server.gui.add_slider("Translate X", min=-50.0, max=50.0, step=0.1, initial_value=0.0)
    gt_ty_slider = server.gui.add_slider("Translate Y", min=-50.0, max=50.0, step=0.1, initial_value=0.0)
    gt_tz_slider = server.gui.add_slider("Translate Z", min=-50.0, max=50.0, step=0.1, initial_value=0.0)

    # Rotation controls (in degrees)
    server.gui.add_markdown("#### Rotation (degrees)")
    gt_rx_slider = server.gui.add_slider("Rotate X (Roll)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
    gt_ry_slider = server.gui.add_slider("Rotate Y (Pitch)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)
    gt_rz_slider = server.gui.add_slider("Rotate Z (Yaw)", min=-180.0, max=180.0, step=1.0, initial_value=0.0)

    # Scale control
    server.gui.add_markdown("#### Scale")
    gt_scale_slider = server.gui.add_slider("Scale", min=0.01, max=10.0, step=0.01, initial_value=1.0)

    # Reset button
    reset_gt_button = server.gui.add_button("Reset GT Transform")

    # --- Evaluation Controls ---
    server.gui.add_markdown("---")
    server.gui.add_markdown("### Evaluation")
    server.gui.add_markdown("#### Thresholds")
    threshold1_slider = server.gui.add_slider("Threshold 1", min=0.001, max=1.0, step=0.001, initial_value=0.01)
    threshold2_slider = server.gui.add_slider("Threshold 2", min=0.001, max=2.0, step=0.001, initial_value=0.05)
    threshold3_slider = server.gui.add_slider("Threshold 3", min=0.001, max=5.0, step=0.001, initial_value=0.1)

    run_eval_button = server.gui.add_button("Run Evaluation")
    results_text = server.gui.add_markdown("*Click 'Run Evaluation' to compute metrics*")

    # --- State variables ---
    eval_state = {
        'gt_points_original': None,  # Original GT points (before transform)
        'gt_colors': None,
        'gt_handle': None,
        'reconstruction_points': None,
        'reconstruction_colors': None,
    }

    # --- Load and add point clouds and cameras ---

    # Camera poses
    camera_poses_path = os.path.join(args.exp_folder, "camera_poses.ply")
    if os.path.exists(camera_poses_path):
        print(f"Loading camera poses: {camera_poses_path}")
        cam_pcd = trimesh.load(camera_poses_path)
        cam_points = np.array(cam_pcd.vertices)
        cam_colors = np.array(cam_pcd.colors)[:, :3] if hasattr(cam_pcd, 'colors') and cam_pcd.colors is not None and len(cam_pcd.colors) > 0 else np.array([255, 0, 0])

        camera_pcd_handle = server.scene.add_point_cloud(
            name="/cameras",
            points=cam_points,
            colors=cam_colors,
            point_size=camera_size_slider.value,
            visible=show_cameras_gui.value,
        )
        print(f"Loaded {len(cam_points)} camera poses.")

        @show_cameras_gui.on_update
        def _(_):
            camera_pcd_handle.visible = show_cameras_gui.value

        @camera_size_slider.on_update
        def _(_):
            camera_pcd_handle.point_size = camera_size_slider.value
    else:
        print(f"Warning: camera_poses.ply not found at {camera_poses_path}")


    # Merged point cloud
    pcd_dir = os.path.join(args.exp_folder, "pcd")
    merged_pcd_path = os.path.join(pcd_dir, "combined_pcd.ply")
    if os.path.exists(merged_pcd_path):
        print(f"Loading merged point cloud: {merged_pcd_path}")
        pcd = trimesh.load(merged_pcd_path)
        points = np.array(pcd.vertices)
        colors = np.array(pcd.colors)[:, :3] if hasattr(pcd, 'colors') and pcd.colors is not None and len(pcd.colors) > 0 and pcd.colors.shape[0] == points.shape[0] else np.random.randint(0, 255, size=(len(points), 3), dtype=np.uint8)

        merged_pcd_handle = server.scene.add_point_cloud(
            name="/merged",
            points=points,
            colors=colors,
            point_size=merged_point_size_slider.value,
            visible=show_merged_gui.value,
        )
        print(f"Loaded {len(points)} points for merged cloud.")

        # Store for evaluation
        eval_state['reconstruction_points'] = points.copy()
        eval_state['reconstruction_colors'] = colors.copy()

        @show_merged_gui.on_update
        def _(_):
            merged_pcd_handle.visible = show_merged_gui.value

        @merged_point_size_slider.on_update
        def _(_):
            merged_pcd_handle.point_size = merged_point_size_slider.value

    else:
        print(f"Warning: Merged point cloud not found at {merged_pcd_path}")

    # Chunk point clouds
    chunk_pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*_pcd.ply")))
    chunk_pcd_files = [f for f in chunk_pcd_files if "combined" not in os.path.basename(f)]

    chunk_handles = []
    if chunk_pcd_files:
        print(f"Found {len(chunk_pcd_files)} chunk point clouds.")
        for i, pcd_path in enumerate(chunk_pcd_files):
            chunk_name = os.path.basename(pcd_path).replace('.ply', '')
            print(f"Loading chunk {chunk_name}: {pcd_path}")
            pcd = trimesh.load(pcd_path)
            points = np.array(pcd.vertices)
            if hasattr(pcd, 'colors') and pcd.colors is not None and len(pcd.colors) > 0 and pcd.colors.shape[0] == points.shape[0]:
                colors = np.array(pcd.colors)[:, :3]
            else:
                # Assign random color if no color info
                colors = np.random.randint(0, 255, size=(len(points), 3), dtype=np.uint8)

            chunk_handle = server.scene.add_point_cloud(
                name=f"/chunks/{chunk_name}",
                points=points,
                colors=colors,
                point_size=chunk_point_size_slider.value,
                visible=show_chunks_gui.value
            )
            chunk_handles.append(chunk_handle)
            print(f"Loaded {len(points)} points for chunk {chunk_name}.")

        @show_chunks_gui.on_update
        def _(_):
            for handle in chunk_handles:
                handle.visible = show_chunks_gui.value

        @chunk_point_size_slider.on_update
        def _(_):
            for handle in chunk_handles:
                handle.point_size = chunk_point_size_slider.value

    else:
        print("No chunk point clouds found.")

    # Sparse alignment points
    sparse_align_dir = os.path.join(args.exp_folder, "sparse_align")
    sparse_handles = []

    if os.path.exists(sparse_align_dir):
        print(f"\nLoading sparse alignment points from: {sparse_align_dir}")

        # Find all sparse alignment files
        sparse_files = sorted(glob.glob(os.path.join(sparse_align_dir, "*.ply")))

        for sparse_path in sparse_files:
            filename = os.path.basename(sparse_path)

            # Parse filename to determine type
            # Format: align_XXX_pre_ransac_chunk1.ply, align_XXX_post_ransac_combined.ply, etc.
            is_pre_ransac = "pre_ransac" in filename
            is_post_ransac = "post_ransac" in filename
            is_chunk1 = "chunk1" in filename
            is_chunk2 = "chunk2" in filename
            is_combined = "combined" in filename

            # Extract alignment index
            match = re.search(r'align_(\d+)', filename)
            align_idx = match.group(1) if match else "unknown"

            try:
                pcd = trimesh.load(sparse_path)
                points = np.array(pcd.vertices)

                if len(points) == 0:
                    continue

                if hasattr(pcd, 'colors') and pcd.colors is not None and len(pcd.colors) > 0:
                    colors = np.array(pcd.colors)[:, :3]
                else:
                    # Default colors based on type
                    if is_chunk1:
                        colors = np.full((len(points), 3), [0, 0, 255], dtype=np.uint8)  # Blue
                    elif is_chunk2:
                        colors = np.full((len(points), 3), [255, 0, 0], dtype=np.uint8)  # Red
                    else:
                        colors = np.full((len(points), 3), [0, 255, 0], dtype=np.uint8)  # Green

                # Determine visibility based on GUI settings
                def get_visibility(is_pre, is_post, is_c1, is_c2, is_comb):
                    if not show_sparse_gui.value:
                        return False
                    if is_pre and not show_sparse_pre_ransac_gui.value:
                        return False
                    if is_post and not show_sparse_post_ransac_gui.value:
                        return False
                    if is_c1 and not show_sparse_chunk1_gui.value:
                        return False
                    if is_c2 and not show_sparse_chunk2_gui.value:
                        return False
                    if is_comb and not show_sparse_combined_gui.value:
                        return False
                    return True

                initial_visible = get_visibility(is_pre_ransac, is_post_ransac, is_chunk1, is_chunk2, is_combined)

                # Create descriptive name
                ransac_type = "pre_ransac" if is_pre_ransac else "post_ransac"
                point_type = "chunk1" if is_chunk1 else ("chunk2" if is_chunk2 else "combined")
                display_name = f"/sparse/align_{align_idx}/{ransac_type}/{point_type}"

                handle = server.scene.add_point_cloud(
                    name=display_name,
                    points=points,
                    colors=colors,
                    point_size=sparse_point_size_slider.value,
                    visible=initial_visible,
                )

                sparse_handles.append({
                    'handle': handle,
                    'is_pre_ransac': is_pre_ransac,
                    'is_post_ransac': is_post_ransac,
                    'is_chunk1': is_chunk1,
                    'is_chunk2': is_chunk2,
                    'is_combined': is_combined,
                })

                print(f"  Loaded {len(points)} sparse points: {filename}")

            except Exception as e:
                print(f"  Warning: Failed to load {filename}: {e}")

        print(f"Loaded {len(sparse_handles)} sparse alignment point clouds.")

        # Update handlers for sparse point visibility
        def update_sparse_visibility():
            for item in sparse_handles:
                h = item['handle']
                visible = show_sparse_gui.value
                if visible:
                    if item['is_pre_ransac'] and not show_sparse_pre_ransac_gui.value:
                        visible = False
                    elif item['is_post_ransac'] and not show_sparse_post_ransac_gui.value:
                        visible = False
                    if visible:
                        if item['is_chunk1'] and not show_sparse_chunk1_gui.value:
                            visible = False
                        elif item['is_chunk2'] and not show_sparse_chunk2_gui.value:
                            visible = False
                        elif item['is_combined'] and not show_sparse_combined_gui.value:
                            visible = False
                h.visible = visible

        @show_sparse_gui.on_update
        def _(_):
            update_sparse_visibility()

        @show_sparse_chunk1_gui.on_update
        def _(_):
            update_sparse_visibility()

        @show_sparse_chunk2_gui.on_update
        def _(_):
            update_sparse_visibility()

        @show_sparse_combined_gui.on_update
        def _(_):
            update_sparse_visibility()

        @show_sparse_pre_ransac_gui.on_update
        def _(_):
            update_sparse_visibility()

        @show_sparse_post_ransac_gui.on_update
        def _(_):
            update_sparse_visibility()

        @sparse_point_size_slider.on_update
        def _(_):
            for item in sparse_handles:
                item['handle'].point_size = sparse_point_size_slider.value
    else:
        print(f"\nNo sparse alignment points found at {sparse_align_dir}")

    # --- Load Ground Truth if provided ---
    def update_gt_transform():
        """Update GT point cloud with current transformation parameters."""
        if eval_state['gt_points_original'] is None or eval_state['gt_handle'] is None:
            return

        # Get transformation parameters
        tx, ty, tz = gt_tx_slider.value, gt_ty_slider.value, gt_tz_slider.value
        rx, ry, rz = np.radians(gt_rx_slider.value), np.radians(gt_ry_slider.value), np.radians(gt_rz_slider.value)
        scale = gt_scale_slider.value

        # Create rotation matrix
        R = rotation_matrix_from_euler(rx, ry, rz)
        t = np.array([tx, ty, tz])

        # Apply transformation: points_new = scale * R @ points + t
        transformed_points = apply_sim3(eval_state['gt_points_original'], R, t, scale)

        # Update the point cloud
        eval_state['gt_handle'].remove()
        eval_state['gt_handle'] = server.scene.add_point_cloud(
            name="/ground_truth",
            points=transformed_points,
            colors=eval_state['gt_colors'],
            point_size=gt_point_size_slider.value,
            visible=show_gt_gui.value,
        )

    if args.gt_pcd and os.path.exists(args.gt_pcd):
        print(f"\nLoading ground truth point cloud: {args.gt_pcd}")
        gt_points, gt_colors = load_point_cloud(args.gt_pcd)

        if gt_colors is None:
            # Default to gray for GT
            gt_colors = np.full((len(gt_points), 3), 180, dtype=np.uint8)

        eval_state['gt_points_original'] = gt_points.copy()
        eval_state['gt_colors'] = gt_colors

        gt_handle = server.scene.add_point_cloud(
            name="/ground_truth",
            points=gt_points,
            colors=gt_colors,
            point_size=gt_point_size_slider.value,
            visible=show_gt_gui.value,
        )
        eval_state['gt_handle'] = gt_handle

        print(f"Loaded {len(gt_points)} ground truth points.")

        # GT visibility and size handlers
        @show_gt_gui.on_update
        def _(_):
            if eval_state['gt_handle'] is not None:
                eval_state['gt_handle'].visible = show_gt_gui.value

        @gt_point_size_slider.on_update
        def _(_):
            if eval_state['gt_handle'] is not None:
                eval_state['gt_handle'].point_size = gt_point_size_slider.value

        # GT transform handlers
        @gt_tx_slider.on_update
        def _(_):
            update_gt_transform()

        @gt_ty_slider.on_update
        def _(_):
            update_gt_transform()

        @gt_tz_slider.on_update
        def _(_):
            update_gt_transform()

        @gt_rx_slider.on_update
        def _(_):
            update_gt_transform()

        @gt_ry_slider.on_update
        def _(_):
            update_gt_transform()

        @gt_rz_slider.on_update
        def _(_):
            update_gt_transform()

        @gt_scale_slider.on_update
        def _(_):
            update_gt_transform()

        @reset_gt_button.on_click
        def _(_):
            gt_tx_slider.value = 0.0
            gt_ty_slider.value = 0.0
            gt_tz_slider.value = 0.0
            gt_rx_slider.value = 0.0
            gt_ry_slider.value = 0.0
            gt_rz_slider.value = 0.0
            gt_scale_slider.value = 1.0
            update_gt_transform()

    elif args.gt_pcd:
        print(f"Warning: Ground truth file not found: {args.gt_pcd}")
    else:
        print("\nNo ground truth provided. Use --gt_pcd to enable evaluation.")

    # --- Evaluation callback ---
    @run_eval_button.on_click
    def _(_):
        if eval_state['reconstruction_points'] is None:
            results_text.content = "**Error:** No reconstruction loaded"
            return

        if eval_state['gt_points_original'] is None:
            results_text.content = "**Error:** No ground truth loaded. Use --gt_pcd argument."
            return

        results_text.content = "*Running evaluation...*"

        try:
            reconstruction = eval_state['reconstruction_points']

            # Get current GT transformation
            tx, ty, tz = gt_tx_slider.value, gt_ty_slider.value, gt_tz_slider.value
            rx, ry, rz = np.radians(gt_rx_slider.value), np.radians(gt_ry_slider.value), np.radians(gt_rz_slider.value)
            scale = gt_scale_slider.value

            R = rotation_matrix_from_euler(rx, ry, rz)
            t = np.array([tx, ty, tz])

            # Apply transformation to GT
            gt_transformed = apply_sim3(eval_state['gt_points_original'], R, t, scale)

            # Compute metrics
            thresholds = [threshold1_slider.value, threshold2_slider.value, threshold3_slider.value]
            print(f"[Eval] Computing accuracy metrics with thresholds: {thresholds}")

            metrics = compute_accuracy_metrics(
                reconstruction,
                gt_transformed,
                thresholds=thresholds,
                use_gpu=True,
            )

            # Format results
            t1, t2, t3 = thresholds
            results_md = f"""
**Transform Applied:**
- Translation: ({tx:.2f}, {ty:.2f}, {tz:.2f})
- Rotation: ({gt_rx_slider.value:.1f}°, {gt_ry_slider.value:.1f}°, {gt_rz_slider.value:.1f}°)
- Scale: {scale:.4f}

**Metrics:**
- Chamfer Distance: {metrics['chamfer_distance']:.6f}
- Accuracy (rec→GT): {metrics['accuracy']:.6f}
- Completeness (GT→rec): {metrics['completeness']:.6f}

**F-scores:**
- F@{t1:.3f}: {metrics[f'f_score@{t1}']:.2f}% (P={metrics[f'precision@{t1}']:.2f}%, R={metrics[f'recall@{t1}']:.2f}%)
- F@{t2:.3f}: {metrics[f'f_score@{t2}']:.2f}% (P={metrics[f'precision@{t2}']:.2f}%, R={metrics[f'recall@{t2}']:.2f}%)
- F@{t3:.3f}: {metrics[f'f_score@{t3}']:.2f}% (P={metrics[f'precision@{t3}']:.2f}%, R={metrics[f'recall@{t3}']:.2f}%)

**Point Counts:**
- Reconstruction: {metrics['num_reconstruction_points']:,}
- Ground Truth: {metrics['num_gt_points']:,}
"""
            results_text.content = results_md
            print("[Eval] Evaluation complete!")
            print(f"  Chamfer: {metrics['chamfer_distance']:.6f}")
            print(f"  Accuracy: {metrics['accuracy']:.6f}")
            print(f"  Completeness: {metrics['completeness']:.6f}")
            for t in thresholds:
                print(f"  F@{t}: {metrics[f'f_score@{t}']:.2f}%")

        except Exception as e:
            results_text.content = f"**Error:** {str(e)}"
            print(f"[Eval] Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nPoint cloud visualization loaded!")
    print("Open your browser to the viser URL to view the point clouds.")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
