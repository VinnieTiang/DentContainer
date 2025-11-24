#!/usr/bin/env python3
"""
Compare Original and Dented Containers by Depth Analysis

This script:
1. Loads an original container mesh
2. Loads the corresponding dented container mesh
3. Renders both from the same camera position
4. Compares depth values pixel-by-pixel
5. Creates a black and white mask:
   - WHITE (255) = dented areas (different depth)
   - BLACK (0) = normal areas (same depth)
"""

import numpy as np
import trimesh
import pyrender
from pathlib import Path
import json
import imageio
import cv2
from typing import Optional, Tuple, Dict
import logging
from datetime import datetime

from config import ContainerConfig, RendererConfig
from camera_position import CameraPoseGenerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def look_at_matrix(eye: np.ndarray, at: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a pyrender camera pose (4x4 world_from_camera) from eye, at, up."""
    z = (eye - at).astype(np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    
    R = np.stack([x, y, z], axis=1)
    T = eye.reshape(3, 1)
    
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R
    mat[:3, 3:] = T
    return mat


class DentComparisonRenderer:
    """Renders and compares original vs dented containers to identify dented areas."""
    
    def __init__(self, image_size: int = 512, camera_fov: float = 75.0):
        """
        Initialize the comparison renderer.
        
        Args:
            image_size: Size of rendered images (square)
            camera_fov: Camera field of view in degrees
        """
        self.image_size = image_size
        self.camera_fov = camera_fov
        
        # Initialize pyrender offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=image_size,
            viewport_height=image_size
        )
        
        # Initialize camera pose generator
        self.container_config = ContainerConfig()
        self.renderer_config = RendererConfig()
        self.renderer_config.IMAGE_SIZE = image_size
        self.renderer_config.CAMERA_FOV = camera_fov
        self.pose_generator = CameraPoseGenerator(self.container_config, self.renderer_config)
        
        logger.info(f"DentComparisonRenderer initialized (image_size={image_size}, fov={camera_fov}°)")
    
    def render_depth(self, mesh: trimesh.Trimesh, pose: Dict) -> np.ndarray:
        """
        Render depth map for a mesh from a given camera pose.
        
        Args:
            mesh: Container mesh to render
            pose: Camera pose dictionary with 'eye', 'at', 'up' keys
            
        Returns:
            Depth map as numpy array
        """
        # Convert pose to numpy arrays
        eye = pose['eye'].cpu().numpy()[0] if hasattr(pose['eye'], 'cpu') else np.asarray(pose['eye'])
        at = pose['at'].cpu().numpy()[0] if hasattr(pose['at'], 'cpu') else np.asarray(pose['at'])
        up = pose['up'].cpu().numpy()[0] if hasattr(pose['up'], 'cpu') else np.asarray(pose['up'])
        
        # Ensure 1D arrays
        if eye.ndim > 1:
            eye = eye.flatten()
        if at.ndim > 1:
            at = at.flatten()
        if up.ndim > 1:
            up = up.flatten()
        
        # Ensure mesh has normals
        try:
            mesh.compute_vertex_normals()
        except:
            pass
        
        # Create pyrender mesh
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        
        # Create scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
        scene.add(pr_mesh, name="container")
        
        # Setup camera
        fov_y_rad = np.deg2rad(self.camera_fov)
        camera = pyrender.PerspectiveCamera(
            yfov=fov_y_rad,
            znear=self.renderer_config.ZNEAR,
            zfar=self.renderer_config.ZFAR
        )
        
        # Calculate camera pose matrix
        cam_world = look_at_matrix(eye, at, up)
        scene.add(camera, pose=cam_world)
        
        # Add lighting
        light = pyrender.PointLight(color=np.ones(3), intensity=10.0)
        scene.add(light, pose=cam_world)
        
        # Render depth only
        # pyrender.render() returns (color, depth) even with DEPTH_ONLY flag
        # Some versions may return additional values, so we capture all and take depth
        render_result = self.renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        
        # Extract depth (typically the second element, or last if more values returned)
        if isinstance(render_result, tuple):
            depth = render_result[1] if len(render_result) >= 2 else render_result[-1]
        else:
            depth = render_result
        
        return depth
    
    def compare_depths(self, original_depth: np.ndarray, dented_depth: np.ndarray, 
                      threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compare depth maps to identify dented areas.
        
        Compares pixel-by-pixel depth values between original and dented containers.
        Areas with different depth values (exceeding threshold) are marked as WHITE (dented).
        Areas with same/similar depth values are marked as BLACK (normal).
        
        Args:
            original_depth: Depth map from original container
            dented_depth: Depth map from dented container
            threshold: Depth difference threshold in meters to consider as dent
            
        Returns:
            Tuple of (difference_map, binary_mask)
            - difference_map: Absolute depth difference (meters)
            - binary_mask: Binary mask (WHITE=255 for dented areas with different depth, BLACK=0 for normal areas with same depth)
        """
        # Ensure same dimensions
        if original_depth.shape != dented_depth.shape:
            logger.warning(f"Depth shape mismatch: original {original_depth.shape} vs dented {dented_depth.shape}")
            # Resize to match
            h, w = original_depth.shape
            dented_depth = cv2.resize(dented_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate absolute depth difference between original and dented containers
        # This detects any change in depth (dents push surface inward/outward)
        depth_diff = np.abs(dented_depth - original_depth)
        
        # Identify valid pixels (both depths are valid and finite)
        valid_mask = (original_depth > 0) & (dented_depth > 0) & np.isfinite(original_depth) & np.isfinite(dented_depth)
        
        # Create binary mask:
        # - Start with all BLACK (0) = normal areas (same depth)
        # - Mark WHITE (255) = dented areas (different depth exceeding threshold)
        binary_mask = np.zeros_like(depth_diff, dtype=np.uint8)
        dented_pixels = valid_mask & (depth_diff > threshold)
        binary_mask[dented_pixels] = 255  # WHITE for areas with different depth
        
        return depth_diff, binary_mask
    
    def process_container_pair(self, original_path: Path, dented_path: Path, 
                              output_dir: Path, container_type: str = "20ft",
                              threshold: float = 0.01) -> Dict:
        """
        Process a pair of original and dented containers.
        
        Args:
            original_path: Path to original container OBJ file
            dented_path: Path to dented container OBJ file
            output_dir: Directory to save output images
            container_type: Container type ("20ft", "40ft", "40ft_hc")
            threshold: Depth difference threshold in meters
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing container pair:")
        logger.info(f"  Original: {original_path}")
        logger.info(f"  Dented: {dented_path}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load meshes
        logger.info("Loading meshes...")
        original_mesh = trimesh.load(original_path)
        dented_mesh = trimesh.load(dented_path)
        
        if isinstance(original_mesh, trimesh.Scene):
            original_mesh = list(original_mesh.geometry.values())[0]
        if isinstance(dented_mesh, trimesh.Scene):
            dented_mesh = list(dented_mesh.geometry.values())[0]
        
        logger.info(f"  Original: {len(original_mesh.vertices)} vertices, {len(original_mesh.faces)} faces")
        logger.info(f"  Dented: {len(dented_mesh.vertices)} vertices, {len(dented_mesh.faces)} faces")
        
        # Generate camera poses
        poses = self.pose_generator.generate_poses(container_type)
        logger.info(f"Generated {len(poses)} camera poses")
        
        # Process each camera pose
        results = []
        base_name = original_path.stem.replace("_dented", "").replace("container_", "")
        
        for pose_idx, pose in enumerate(poses):
            shot_name = pose['name']
            logger.info(f"  [{pose_idx+1}/{len(poses)}] Processing shot: {shot_name}")
            
            try:
                # Render depth maps
                original_depth = self.render_depth(original_mesh, pose)
                dented_depth = self.render_depth(dented_mesh, pose)
                
                # Compare depths
                depth_diff, dent_mask = self.compare_depths(original_depth, dented_depth, threshold)
                
                # Render RGB for visualization (optional)
                original_rgb, _ = self._render_rgb(original_mesh, pose)
                dented_rgb, _ = self._render_rgb(dented_mesh, pose)
                
                # Save outputs
                shot_output_dir = output_dir / shot_name
                shot_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save depth maps
                original_depth_path = shot_output_dir / f"{base_name}_original_depth.npy"
                dented_depth_path = shot_output_dir / f"{base_name}_dented_depth.npy"
                np.save(original_depth_path, original_depth.astype(np.float32))
                np.save(dented_depth_path, dented_depth.astype(np.float32))
                
                # Save depth difference
                depth_diff_path = shot_output_dir / f"{base_name}_depth_diff.npy"
                np.save(depth_diff_path, depth_diff.astype(np.float32))
                
                # Save normalized depth images for visualization
                original_depth_img = self._normalize_depth(original_depth)
                dented_depth_img = self._normalize_depth(dented_depth)
                depth_diff_img = self._normalize_depth(depth_diff)
                
                imageio.imwrite(shot_output_dir / f"{base_name}_original_depth.png", original_depth_img)
                imageio.imwrite(shot_output_dir / f"{base_name}_dented_depth.png", dented_depth_img)
                imageio.imwrite(shot_output_dir / f"{base_name}_depth_diff.png", depth_diff_img)
                
                # Save RGB images
                imageio.imwrite(shot_output_dir / f"{base_name}_original_rgb.png", original_rgb)
                imageio.imwrite(shot_output_dir / f"{base_name}_dented_rgb.png", dented_rgb)
                
                # Generate and save point clouds as PLY files
                try:
                    # Original point cloud with RGB colors
                    original_pcd = self._depth_to_pointcloud(original_depth, pose, original_rgb)
                    original_ply_path = shot_output_dir / f"{base_name}_original_pointcloud.ply"
                    original_pcd.export(original_ply_path)
                    logger.info(f"    ✓ Saved original point cloud: {original_ply_path.name} ({len(original_pcd.vertices)} points)")
                    
                    # Dented point cloud with RGB colors
                    dented_pcd = self._depth_to_pointcloud(dented_depth, pose, dented_rgb)
                    dented_ply_path = shot_output_dir / f"{base_name}_dented_pointcloud.ply"
                    dented_pcd.export(dented_ply_path)
                    logger.info(f"    ✓ Saved dented point cloud: {dented_ply_path.name} ({len(dented_pcd.vertices)} points)")
                except Exception as e:
                    logger.warning(f"    ⚠️  Failed to generate point clouds: {e}")
                
                # Save binary mask: WHITE (255) = dented areas (different depth), BLACK (0) = normal areas (same depth)
                imageio.imwrite(shot_output_dir / f"{base_name}_dent_mask.png", dent_mask)
                
                # Calculate statistics
                dent_pixels = np.sum(dent_mask > 0)
                total_pixels = dent_mask.size
                dent_percentage = (dent_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                
                max_diff = np.max(depth_diff[depth_diff > 0]) if np.any(depth_diff > 0) else 0
                mean_diff = np.mean(depth_diff[depth_diff > threshold]) if np.any(depth_diff > threshold) else 0
                
                logger.info(f"    ✓ Dent mask: {dent_pixels} pixels ({dent_percentage:.1f}%)")
                logger.info(f"    ✓ Max depth difference: {max_diff*1000:.2f}mm, Mean: {mean_diff*1000:.2f}mm")
                
                results.append({
                    'shot_name': shot_name,
                    'dent_pixels': int(dent_pixels),
                    'total_pixels': int(total_pixels),
                    'dent_percentage': float(dent_percentage),
                    'max_depth_diff_m': float(max_diff),
                    'max_depth_diff_mm': float(max_diff * 1000),
                    'mean_depth_diff_m': float(mean_diff),
                    'mean_depth_diff_mm': float(mean_diff * 1000),
                    'output_dir': str(shot_output_dir)
                })
                
            except Exception as e:
                logger.error(f"    ✗ Error processing shot {shot_name}: {e}", exc_info=True)
                continue
        
        # Save summary
        summary = {
            'original_file': str(original_path),
            'dented_file': str(dented_path),
            'container_type': container_type,
            'threshold_m': threshold,
            'threshold_mm': threshold * 1000,
            'timestamp': datetime.now().isoformat(),
            'shots': results
        }
        
        summary_path = output_dir / f"{base_name}_comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Comparison complete. Summary saved to: {summary_path}")
        
        return summary
    
    def _render_rgb(self, mesh: trimesh.Trimesh, pose: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Render RGB image for visualization."""
        # Convert pose to numpy arrays
        eye = pose['eye'].cpu().numpy()[0] if hasattr(pose['eye'], 'cpu') else np.asarray(pose['eye'])
        at = pose['at'].cpu().numpy()[0] if hasattr(pose['at'], 'cpu') else np.asarray(pose['at'])
        up = pose['up'].cpu().numpy()[0] if hasattr(pose['up'], 'cpu') else np.asarray(pose['up'])
        
        if eye.ndim > 1:
            eye = eye.flatten()
        if at.ndim > 1:
            at = at.flatten()
        if up.ndim > 1:
            up = up.flatten()
        
        try:
            mesh.compute_vertex_normals()
        except:
            pass
        
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
        scene.add(pr_mesh, name="container")
        
        fov_y_rad = np.deg2rad(self.camera_fov)
        camera = pyrender.PerspectiveCamera(
            yfov=fov_y_rad,
            znear=self.renderer_config.ZNEAR,
            zfar=self.renderer_config.ZFAR
        )
        
        cam_world = look_at_matrix(eye, at, up)
        scene.add(camera, pose=cam_world)
        
        light = pyrender.PointLight(color=np.ones(3), intensity=10.0)
        scene.add(light, pose=cam_world)
        
        color, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgb = color[..., :3].astype(np.uint8)
        
        return rgb, depth
    
    def _depth_to_pointcloud(self, depth: np.ndarray, pose: Dict, rgb: Optional[np.ndarray] = None) -> trimesh.PointCloud:
        """
        Convert depth map to 3D point cloud using camera intrinsics.
        
        Args:
            depth: Depth map (H, W) in meters
            pose: Camera pose dictionary with 'eye', 'at', 'up' keys
            rgb: Optional RGB image (H, W, 3) for colored point cloud
            
        Returns:
            trimesh.PointCloud object
        """
        height, width = depth.shape
        
        # Calculate camera intrinsics from FOV
        fov_y_rad = np.deg2rad(self.camera_fov)
        focal_length = (height / 2.0) / np.tan(fov_y_rad / 2.0)
        cx, cy = width / 2.0, height / 2.0
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to normalized camera coordinates
        x_norm = (u - cx) / focal_length
        y_norm = (v - cy) / focal_length
        
        # Get valid depth pixels (non-zero)
        valid_mask = depth > 0
        
        # Back-project to 3D points in camera space
        x_cam = x_norm[valid_mask] * depth[valid_mask]
        y_cam = y_norm[valid_mask] * depth[valid_mask]
        z_cam = depth[valid_mask]
        
        # Stack into Nx3 array
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Convert pose to numpy arrays
        eye = pose['eye'].cpu().numpy()[0] if hasattr(pose['eye'], 'cpu') else np.asarray(pose['eye'])
        at = pose['at'].cpu().numpy()[0] if hasattr(pose['at'], 'cpu') else np.asarray(pose['at'])
        up = pose['up'].cpu().numpy()[0] if hasattr(pose['up'], 'cpu') else np.asarray(pose['up'])
        
        if eye.ndim > 1:
            eye = eye.flatten()
        if at.ndim > 1:
            at = at.flatten()
        if up.ndim > 1:
            up = up.flatten()
        
        # Build camera-to-world transformation matrix
        # Camera looks along -Z axis, so we need to transform
        forward = (at - eye)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)
        
        # Rotation matrix: camera frame to world frame
        R_cam_to_world = np.stack([right, up_corrected, -forward], axis=1)
        
        # Transform points from camera space to world space
        points_world = (R_cam_to_world @ points_cam.T).T + eye
        
        # Extract colors if RGB provided
        colors = None
        if rgb is not None:
            colors = rgb[valid_mask].astype(np.uint8)
        
        # Create point cloud
        if colors is not None:
            pcd = trimesh.PointCloud(vertices=points_world, colors=colors)
        else:
            pcd = trimesh.PointCloud(vertices=points_world)
        
        return pcd
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-255 range for visualization."""
        valid_depths = depth[depth > 0]
        
        if len(valid_depths) == 0:
            return np.zeros_like(depth, dtype=np.uint8)
        
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        depth_range = max_depth - min_depth
        
        if depth_range > 0:
            normalized = (depth - min_depth) / depth_range
            normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        else:
            normalized = np.ones_like(depth, dtype=np.uint8) * 128
        
        normalized[depth <= 0] = 0
        return normalized
    
    def cleanup(self):
        """Clean up renderer resources."""
        if hasattr(self, 'renderer'):
            self.renderer.delete()


def main():
    """Main function to process container pairs."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare original and dented containers by depth analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare a single container pair
  python compare_dents_depth.py \\
    --original complete_containers/container_20ft_0005.obj \\
    --dented complete_containers_dented/container_20ft_0005_dented.obj \\
    --output comparison_output \\
    --container-type 20ft
  
  # Batch process all containers
  python compare_dents_depth.py --batch \\
    --original-dir complete_containers \\
    --dented-dir complete_containers_dented \\
    --output comparison_output
        """
    )
    
    parser.add_argument('--original', type=str, help='Path to original container OBJ file')
    parser.add_argument('--dented', type=str, help='Path to dented container OBJ file')
    parser.add_argument('--output', type=str, default='comparison_output', 
                       help='Output directory for comparison results')
    parser.add_argument('--container-type', type=str, default='20ft', 
                       choices=['20ft', '40ft', '40ft_hc'],
                       help='Container type')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Depth difference threshold in meters (default: 0.01 = 10mm)')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size for rendering (default: 512)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process all containers in directories')
    parser.add_argument('--original-dir', type=str, default='complete_containers',
                       help='Directory with original containers (for batch mode)')
    parser.add_argument('--dented-dir', type=str, default='complete_containers_dented',
                       help='Directory with dented containers (for batch mode)')
    
    args = parser.parse_args()
    
    # Initialize renderer
    renderer = DentComparisonRenderer(image_size=args.image_size)
    
    try:
        if args.batch:
            # Batch processing
            original_dir = Path(args.original_dir)
            dented_dir = Path(args.dented_dir)
            output_dir = Path(args.output)
            
            if not original_dir.exists():
                logger.error(f"Original directory not found: {original_dir}")
                return
            
            if not dented_dir.exists():
                logger.error(f"Dented directory not found: {dented_dir}")
                return
            
            # Find all OBJ files
            original_files = sorted(original_dir.glob("*.obj"))
            
            logger.info(f"Found {len(original_files)} original containers")
            logger.info("=" * 60)
            
            for original_file in original_files:
                # Find corresponding dented file
                base_name = original_file.stem
                # Try different naming patterns
                dented_file = dented_dir / f"{base_name}_dented.obj"
                if not dented_file.exists():
                    # Try alternative pattern
                    dented_file = dented_dir / f"{base_name.replace('container_', '')}_dented.obj"
                
                if not dented_file.exists():
                    logger.warning(f"No dented file found for {original_file.name}, skipping")
                    continue
                
                # Determine container type from filename
                container_type = "20ft"
                if "40ft_hc" in original_file.name:
                    container_type = "40ft_hc"
                elif "40ft" in original_file.name:
                    container_type = "40ft"
                
                # Process pair
                container_output_dir = output_dir / original_file.stem
                logger.info(f"\nProcessing: {original_file.name}")
                renderer.process_container_pair(
                    original_file,
                    dented_file,
                    container_output_dir,
                    container_type=container_type,
                    threshold=args.threshold
                )
        
        else:
            # Single file processing
            if not args.original or not args.dented:
                parser.error("--original and --dented are required in single file mode")
            
            original_path = Path(args.original)
            dented_path = Path(args.dented)
            
            if not original_path.exists():
                logger.error(f"Original file not found: {original_path}")
                return
            
            if not dented_path.exists():
                logger.error(f"Dented file not found: {dented_path}")
                return
            
            output_dir = Path(args.output)
            renderer.process_container_pair(
                original_path,
                dented_path,
                output_dir,
                container_type=args.container_type,
                threshold=args.threshold
            )
    
    finally:
        renderer.cleanup()
        logger.info("✓ Renderer cleanup complete")


if __name__ == "__main__":
    main()

