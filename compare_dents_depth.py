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
import subprocess
import sys

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
        
        # Load dent metadata from JSON file
        dent_metadata = self._load_dent_metadata(dented_path)
        
        # Generate camera poses
        poses = self.pose_generator.generate_poses(container_type)
        logger.info(f"Generated {len(poses)} camera poses")
        
        # Process each camera pose - Collect statistics and save summary JSON
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
                
                # Render RGB for saving (needed for visual output generation later)
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
                
                # Extract dent-depth values (where mask indicates dents)
                dent_depth_values = self._extract_dent_depths(depth_diff, dent_mask, threshold)
                
                # Calculate raw statistics (before filtering)
                raw_max_diff_m = np.max(dent_depth_values) if len(dent_depth_values) > 0 else 0.0
                raw_max_diff_mm = raw_max_diff_m * 1000.0
                
                # Filter unrealistic dent depths using robust outlier rejection
                filtered_dent_depths = self._filter_dent_depths_outliers(dent_depth_values)
                
                # Calculate filtered statistics (after filtering)
                filtered_max_diff_m = np.max(filtered_dent_depths) if len(filtered_dent_depths) > 0 else 0.0
                filtered_max_diff_mm = filtered_max_diff_m * 1000.0
                
                # Check if filtered max depth exceeds 200mm threshold
                # If it does, set output as "unknown" instead of capping
                if filtered_max_diff_mm > 200.0:
                    filtered_max_diff_m = "unknown"
                    filtered_max_diff_mm = "unknown"
                    logger.info(f"    ✓ Dent mask: {dent_pixels} pixels ({dent_percentage:.1f}%)")
                    logger.info(f"    ✓ Raw max depth difference: {raw_max_diff_mm:.2f}mm")
                    logger.info(f"    ✓ Filtered max depth difference: unknown (exceeds 200mm threshold)")
                else:
                    logger.info(f"    ✓ Dent mask: {dent_pixels} pixels ({dent_percentage:.1f}%)")
                    logger.info(f"    ✓ Raw max depth difference: {raw_max_diff_mm:.2f}mm")
                    logger.info(f"    ✓ Filtered max depth difference: {filtered_max_diff_mm:.2f}mm")
                
                # Calculate camera distance from panel
                # For regular shots: distance from camera to panel
                # For corner shots: distance from panel towards corner pole
                camera_distance_m = self._calculate_camera_distance(pose)
                
                # Calculate dent area in cm² from mask pixels
                # Count pixels where mask = 1 (dented pixels) and convert to real-world area
                dent_area_cm2 = self._calculate_dent_area(dent_mask, dented_depth)
                
                # Prepare filtered statistics for JSON (handle "unknown" case)
                filtered_max_diff_m_json = filtered_max_diff_m if filtered_max_diff_m == "unknown" else float(filtered_max_diff_m)
                filtered_max_diff_mm_json = filtered_max_diff_mm if filtered_max_diff_mm == "unknown" else float(filtered_max_diff_mm)
                
                results.append({
                    'shot_name': shot_name,
                    'dent_pixels': int(dent_pixels),
                    'total_pixels': int(total_pixels),
                    'dent_percentage': float(dent_percentage),
                    # Raw statistics (before filtering)
                    'max_depth_diff_m_raw': float(raw_max_diff_m),
                    'max_depth_diff_mm_raw': float(raw_max_diff_mm),
                    # Filtered statistics (after outlier removal, set to "unknown" if exceeds 200mm)
                    'max_depth_diff_m': filtered_max_diff_m_json,
                    'max_depth_diff_mm': filtered_max_diff_mm_json,
                    'dent_area_cm2': float(dent_area_cm2),
                    'camera_distance_m': float(camera_distance_m),
                    'camera_distance_cm': float(camera_distance_m * 100),
                    'output_dir': str(shot_output_dir)
                })
                
            except Exception as e:
                logger.error(f"    ✗ Error processing shot {shot_name}: {e}", exc_info=True)
                continue
        
        # Save summary JSON first
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
        
        # Automatically generate visual outputs
        self._generate_visual_outputs(summary_path)
        
        return summary
    
    def _generate_visual_outputs(self, summary_path: Path) -> None:
        """
        Automatically run the visual output generation script after summary JSON is created.
        
        Args:
            summary_path: Path to the comparison summary JSON file
        """
        try:
            # Get the path to the visual output script (same directory as this script)
            script_dir = Path(__file__).parent
            visual_output_script = script_dir / "compare_dents_depth_visual_output.py"
            
            if not visual_output_script.exists():
                logger.warning(f"Visual output script not found: {visual_output_script}")
                logger.warning("Skipping visual output generation")
                return
            
            logger.info("Generating visual overlay images...")
            
            # Run the visual output script
            result = subprocess.run(
                [sys.executable, str(visual_output_script), "--summary", str(summary_path)],
                capture_output=True,
                text=True,
                cwd=str(script_dir)
            )
            
            if result.returncode == 0:
                logger.info("✓ Visual overlay images generated successfully")
                if result.stdout:
                    # Log any output from the script
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            logger.info(f"  {line}")
            else:
                logger.error(f"✗ Visual output generation failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                if result.stdout:
                    logger.error(f"Standard output: {result.stdout}")
        
        except Exception as e:
            logger.error(f"✗ Error generating visual outputs: {e}", exc_info=True)
            logger.warning("Continuing without visual outputs...")
    
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
    
    def _load_dent_metadata(self, dented_path: Path) -> Optional[Dict]:
        """
        Load dent metadata from JSON file corresponding to the dented container OBJ file.
        
        Args:
            dented_path: Path to dented container OBJ file
            
        Returns:
            Dictionary with dent metadata or None if file not found
        """
        # Try to find corresponding JSON file
        json_path = dented_path.with_suffix('.json')
        
        if not json_path.exists():
            logger.warning(f"Dent metadata JSON not found: {json_path}")
            return None
        
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded dent metadata from: {json_path}")
            return metadata
        except Exception as e:
            logger.warning(f"Failed to load dent metadata from {json_path}: {e}")
            return None
    
    def _extract_dent_area_from_metadata(self, dent_metadata: Optional[Dict]) -> float:
        """
        Extract total dent area in cm² from metadata.
        
        Args:
            dent_metadata: Dictionary with dent metadata
            
        Returns:
            Total dent area in cm²
        """
        if not dent_metadata or 'dents' not in dent_metadata:
            return 0.0
        
        total_area_m2 = 0.0
        
        for dent in dent_metadata['dents']:
            dent_type = dent.get('type', '')
            
            if dent_type == 'elliptical':
                # Area = π * radius_x * radius_y
                radius_x = dent.get('radius_x', 0)
                radius_y = dent.get('radius_y', 0)
                area = np.pi * radius_x * radius_y
                total_area_m2 += area
            elif dent_type == 'corner':
                # Area = π * radius²
                radius = dent.get('radius', 0)
                area = np.pi * radius * radius
                total_area_m2 += area
            elif dent_type == 'crease':
                # Area = length * width
                length = dent.get('length', 0)
                width = dent.get('width', 0)
                area = length * width
                total_area_m2 += area
            elif dent_type == 'circular':
                # Area = π * radius²
                radius = dent.get('radius', 0)
                area = np.pi * radius * radius
                total_area_m2 += area
        
        # Convert to cm²
        total_area_cm2 = total_area_m2 * 10000.0
        return total_area_cm2
    
    def _extract_dent_depth_from_metadata(self, dent_metadata: Optional[Dict]) -> float:
        """
        Extract maximum dent depth in mm from metadata.
        
        Args:
            dent_metadata: Dictionary with dent metadata
            
        Returns:
            Maximum dent depth in mm
        """
        if not dent_metadata or 'dents' not in dent_metadata:
            return 0.0
        
        max_depth_mm = 0.0
        
        for dent in dent_metadata['dents']:
            # Prefer depth_mm if available, otherwise convert depth (meters) to mm
            depth_mm = dent.get('depth_mm', 0)
            if depth_mm == 0:
                depth_m = dent.get('depth', 0)
                depth_mm = depth_m * 1000.0
            
            max_depth_mm = max(max_depth_mm, depth_mm)
        
        return max_depth_mm
    
    def _calculate_camera_distance(self, pose: Dict) -> float:
        """
        Calculate camera distance from the panel or corner pole.
        
        For regular shots: distance from camera (eye) to panel (at)
        For corner shots: distance from camera (eye) to corner pole (at)
        
        Args:
            pose: Camera pose dictionary with 'eye' and 'at' keys
            
        Returns:
            Distance in meters
        """
        # Convert pose to numpy arrays
        eye = pose['eye'].cpu().numpy()[0] if hasattr(pose['eye'], 'cpu') else np.asarray(pose['eye'])
        at = pose['at'].cpu().numpy()[0] if hasattr(pose['at'], 'cpu') else np.asarray(pose['at'])
        
        # Ensure 1D arrays
        if eye.ndim > 1:
            eye = eye.flatten()
        if at.ndim > 1:
            at = at.flatten()
        
        # Calculate Euclidean distance from camera to target
        distance = np.linalg.norm(eye - at)
        
        return float(distance)
    
    def _calculate_dent_area(self, dent_mask: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Calculate dent area in cm² by counting pixels where mask = 1 (dented) 
        and converting pixel count to real-world area using depth information.
        
        Args:
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas (mask = 1)
            depth_map: Depth map (H, W) in meters
            
        Returns:
            Dent area in cm²
        """
        h, w = dent_mask.shape
        # Count pixels where mask = 1 (dented pixels, represented as 255 in the mask)
        dent_pixels = (dent_mask > 0)  # Count all non-zero pixels (mask = 1 means dented)
        
        if not np.any(dent_pixels):
            return 0.0
        
        # Calculate camera intrinsics from FOV
        fov_y_rad = np.deg2rad(self.camera_fov)
        focal_length = (h / 2.0) / np.tan(fov_y_rad / 2.0)
        
        # For square images, aspect ratio is 1:1
        fov_x_rad = fov_y_rad
        
        # Calculate pixel size in world coordinates for each dent pixel
        # pixel_width = 2 * depth * tan(fov_x/2) / image_width
        # pixel_height = 2 * depth * tan(fov_y/2) / image_height
        # pixel_area = pixel_width * pixel_height
        
        # Get depths for dent pixels
        dent_depths = depth_map[dent_pixels]
        valid_depths = dent_depths[dent_depths > 0]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Calculate average depth for dent region
        avg_depth = np.mean(valid_depths)
        
        # Calculate pixel dimensions in meters at average depth
        pixel_width_m = 2 * avg_depth * np.tan(fov_x_rad / 2.0) / w
        pixel_height_m = 2 * avg_depth * np.tan(fov_y_rad / 2.0) / h
        pixel_area_m2 = pixel_width_m * pixel_height_m
        
        # Calculate total area
        num_dent_pixels = np.sum(dent_pixels)
        total_area_m2 = num_dent_pixels * pixel_area_m2
        
        # Convert to cm²
        total_area_cm2 = total_area_m2 * 10000.0
        
        return total_area_cm2
    
    def _extract_dent_depths(self, depth_diff: np.ndarray, dent_mask: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Extract depth difference values for pixels identified as dents.
        
        Args:
            depth_diff: Depth difference map (H, W) in meters
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            threshold: Minimum depth difference threshold in meters
            
        Returns:
            Array of dent depth values in meters (1D array)
        """
        # Get pixels where mask indicates dents (white pixels)
        dent_pixels = (dent_mask > 0)
        
        if not np.any(dent_pixels):
            return np.array([])
        
        # Extract depth differences for dent pixels
        dent_depths = depth_diff[dent_pixels]
        
        # Filter to valid, finite, positive values above threshold
        valid_depths = dent_depths[
            np.isfinite(dent_depths) & 
            (dent_depths > 0) & 
            (dent_depths > threshold)
        ]
        
        return valid_depths
    
    def _filter_dent_depths_outliers(self, dent_depths: np.ndarray, 
                                     method: str = 'percentile',
                                     percentile: float = 99.0,
                                     iqr_multiplier: float = 1.5) -> np.ndarray:
        """
        Filter unrealistic dent depths using robust outlier rejection.
        
        Supports two methods:
        1. Percentile clipping: clip values above the specified percentile
        2. IQR filtering: remove values > Q3 + multiplier * IQR
        
        Args:
            dent_depths: Array of dent depth values in meters
            method: 'percentile' or 'iqr'
            percentile: Percentile to clip at (for percentile method)
            iqr_multiplier: Multiplier for IQR method (default: 1.5)
            
        Returns:
            Filtered array of dent depth values in meters
        """
        if len(dent_depths) == 0:
            return dent_depths
        
        if method == 'percentile':
            # Percentile clipping: clip above specified percentile
            clip_value = np.percentile(dent_depths, percentile)
            filtered = dent_depths[dent_depths <= clip_value]
            return filtered
        
        elif method == 'iqr':
            # IQR filtering: remove values > Q3 + multiplier * IQR
            q1 = np.percentile(dent_depths, 25.0)
            q3 = np.percentile(dent_depths, 75.0)
            iqr = q3 - q1
            upper_bound = q3 + iqr_multiplier * iqr
            filtered = dent_depths[dent_depths <= upper_bound]
            return filtered
        
        else:
            logger.warning(f"Unknown filtering method: {method}. Using percentile method.")
            return self._filter_dent_depths_outliers(dent_depths, method='percentile', percentile=percentile)
    
    def _calculate_max_depth_diff_robust(self, depth_diff: np.ndarray, percentile: float = 99.0) -> float:
        """
        Calculate maximum depth difference while ignoring outliers.
        
        Uses percentile-based approach to filter out extreme outliers.
        
        Args:
            depth_diff: Depth difference map (H, W) in meters
            percentile: Percentile to use for max calculation (default: 99.0, meaning 99th percentile)
            
        Returns:
            Maximum depth difference in meters (ignoring outliers)
        """
        # Get all valid depth differences (positive and finite)
        valid_depths = depth_diff[(depth_diff > 0) & np.isfinite(depth_diff)]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Use percentile to ignore outliers (e.g., 99th percentile instead of absolute max)
        max_depth_m = np.percentile(valid_depths, percentile)
        
        return float(max_depth_m)
    
    def _calculate_dent_depth(self, depth_diff: np.ndarray, dent_mask: np.ndarray) -> float:
        """
        Calculate maximum dent depth in mm.
        
        Args:
            depth_diff: Depth difference map (H, W) in meters
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            
        Returns:
            Maximum dent depth in mm
        """
        dent_pixels = (dent_mask > 127)
        
        if not np.any(dent_pixels):
            return 0.0
        
        # Get depth differences for dent pixels
        dent_depths = depth_diff[dent_pixels]
        valid_depths = dent_depths[np.isfinite(dent_depths) & (dent_depths > 0)]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Calculate maximum depth difference
        max_depth_m = np.max(valid_depths)
        
        # Convert to mm
        max_depth_mm = max_depth_m * 1000.0
        
        return max_depth_mm
    
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

