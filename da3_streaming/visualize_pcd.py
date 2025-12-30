import argparse
import glob
import os
import time

import numpy as np
import trimesh
import viser

def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds from da3-streaming experiments.")
    parser.add_argument("--exp_folder", type=str, required=True, help="Path to the experiment folder.")
    args = parser.parse_args()

    if not os.path.isdir(args.exp_folder):
        print(f"Error: Experiment folder not found at {args.exp_folder}")
        return

    server = viser.ViserServer()

    # --- GUI controls ---
    show_chunks_gui = server.gui.add_checkbox("Show Chunks", initial_value=True)
    show_merged_gui = server.gui.add_checkbox("Show Merged", initial_value=True)
    show_cameras_gui = server.gui.add_checkbox("Show Cameras", initial_value=True)

    merged_point_size_slider = server.gui.add_slider(
        "Merged Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
    )
    chunk_point_size_slider = server.gui.add_slider(
        "Chunk Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01
    )
    camera_size_slider = server.gui.add_slider(
        "Camera Size", min=0.001, max=0.5, step=0.001, initial_value=0.05
    )

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

    print("\nPoint cloud visualization loaded!")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
