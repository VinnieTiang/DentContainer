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
from sklearn.linear_model import RANSACRegressor
from scipy.ndimage import gaussian_filter

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
        
        # Pre-compute camera intrinsics from FOV (matching pyrender PerspectiveCamera)
        # These intrinsics are used for area calculations and point cloud generation
        fov_y_rad = np.deg2rad(camera_fov)
        self.focal_length = (image_size / 2.0) / np.tan(fov_y_rad / 2.0)
        self.cx = image_size / 2.0  # Principal point x
        self.cy = image_size / 2.0  # Principal point y
        self.fov_x_rad = fov_y_rad  # Square images: fov_x = fov_y
        
        logger.info(f"DentComparisonRenderer initialized (image_size={image_size}, fov={camera_fov}°)")
        logger.info(f"  Camera intrinsics: focal_length={self.focal_length:.2f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
    
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
                      threshold: float = 0.035, morphology_opening_size: int = 9,
                      morphology_closing_size: int = 11,
                      gap_fill_threshold_ratio: float = 0.5, gap_fill_distance: int = 7,
                      internal_fill_threshold_ratio: float = 0.3, internal_fill_iterations: int = 3,
                      min_dent_area: int = 200, skip_morphology: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compare depth maps to identify dented areas and their direction.
        
        Compares pixel-by-pixel depth values between original and dented containers.
        Areas with different depth values (exceeding threshold) are marked as WHITE (dented).
        Areas with same/similar depth values are marked as BLACK (normal).
        
        Args:
            original_depth: Depth map from original container
            dented_depth: Depth map from dented container
            threshold: Depth difference threshold in meters to consider as dent
            morphology_opening_size: Kernel size for morphological opening operation to remove noise (default: 9).
                                    Must be odd (3, 5, 7, etc.). Set to 0 to disable opening.
            morphology_closing_size: Kernel size for morphological closing operation to fill gaps (default: 7).
                                    Must be odd (3, 5, 7, etc.). Set to 0 to disable closing.
            gap_fill_threshold_ratio: Ratio of threshold to use for gap-filling near existing dents (default: 0.5).
                                     Lower values fill more gaps but may include more false positives.
            gap_fill_distance: Distance in pixels to search around existing dent regions for gap-filling (default: 7).
            internal_fill_threshold_ratio: Lower threshold ratio for pixels inside white regions to fill black lines (default: 0.3).
                                         Very low values fill more internal gaps but may include false positives.
            internal_fill_iterations: Number of iterations to apply internal fill (default: 3). More iterations fill larger gaps.
            min_dent_area: Minimum area (in pixels) required to keep a dent region (default: 200).
                          Smaller regions are removed as noise or misclassifications.
            
        Returns:
            Tuple of (difference_map, binary_mask, pure_binary_mask, direction_map)
            - difference_map: Absolute depth difference (meters)
            - binary_mask: Binary mask after morphology operations (WHITE=255 for dented areas with different depth, BLACK=0 for normal areas with same depth)
            - pure_binary_mask: Pure binary mask before morphology operations (after gap-filling, before opening/closing)
            - direction_map: Direction map (1.0 = inward dent, -1.0 = outward dent, 0.0 = no dent)
        """
        # Ensure same dimensions
        if original_depth.shape != dented_depth.shape:
            logger.warning(f"Depth shape mismatch: original {original_depth.shape} vs dented {dented_depth.shape}")
            # Resize to match
            h, w = original_depth.shape
            dented_depth = cv2.resize(dented_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate signed depth difference to track direction
        # For INTERNAL camera (camera inside container):
        # Positive = dented_depth < original_depth (inward dent, surface pushed toward camera/closer)
        # Negative = dented_depth > original_depth (outward dent, surface pushed away from camera/farther)
        signed_depth_diff = dented_depth - original_depth
        
        # Calculate absolute depth difference between original and dented containers
        # This detects any change in depth (dents push surface inward/outward)
        depth_diff = np.abs(signed_depth_diff)
        
        # Identify valid pixels (both depths are valid and finite)
        valid_mask = (original_depth > 0) & (dented_depth > 0) & np.isfinite(original_depth) & np.isfinite(dented_depth)
        
        # Optional: Mask out edge regions to reduce perspective distortion artifacts
        # Edge pixels are more sensitive to small geometric differences due to perspective projection
        # This helps reduce V-shaped patterns at image edges
        edge_mask_percent = 0.05  # Mask 5% from each edge (configurable, set to 0 to disable)
        if edge_mask_percent > 0:
            h, w = depth_diff.shape
            edge_pixels_h = int(h * edge_mask_percent)
            edge_pixels_w = int(w * edge_mask_percent)
            # Create edge exclusion mask (True = exclude, False = keep)
            edge_exclusion_mask = np.zeros_like(valid_mask, dtype=bool)
            # Mark edges as excluded
            edge_exclusion_mask[:edge_pixels_h, :] = True  # Top edge
            edge_exclusion_mask[h-edge_pixels_h:, :] = True  # Bottom edge
            edge_exclusion_mask[:, :edge_pixels_w] = True  # Left edge
            edge_exclusion_mask[:, w-edge_pixels_w:] = True  # Right edge
            # Exclude edge pixels from valid mask
            valid_mask = valid_mask & ~edge_exclusion_mask
        
        # Create binary mask:
        # - Start with all BLACK (0) = normal areas (same depth)
        # - Mark WHITE (255) = dented areas (different depth exceeding threshold)
        binary_mask = np.zeros_like(depth_diff, dtype=np.uint8)
        dented_pixels = valid_mask & (depth_diff > threshold)
        binary_mask[dented_pixels] = 255  # WHITE for areas with different depth
        
        # Gap-filling step: Use a lower threshold for pixels near existing dent regions
        # This fills gaps between white segments by marking nearby pixels with smaller depth differences as dented
        if gap_fill_distance > 0 and gap_fill_threshold_ratio > 0:
            # Create a dilated version of existing dent regions to identify nearby pixels
            dilation_kernel = np.ones((gap_fill_distance * 2 + 1, gap_fill_distance * 2 + 1), np.uint8)
            dilated_mask = cv2.dilate(binary_mask, dilation_kernel, iterations=1)
            
            # Find pixels that are near existing dents but not yet marked as dents
            nearby_pixels = (dilated_mask > 0) & (binary_mask == 0) & valid_mask
            
            # Use a lower threshold for these nearby pixels to fill gaps
            gap_fill_threshold = threshold * gap_fill_threshold_ratio
            gap_fill_pixels = nearby_pixels & (depth_diff > gap_fill_threshold)
            binary_mask[gap_fill_pixels] = 255
        
        # Internal fill step: Use an even lower threshold for pixels inside white dented regions
        # This fills black lines and gaps within white segments by iteratively expanding dent regions
        if internal_fill_threshold_ratio > 0 and internal_fill_iterations > 0:
            internal_fill_threshold = threshold * internal_fill_threshold_ratio
            
            for iteration in range(internal_fill_iterations):
                # Create a dilated mask to find pixels inside or very close to white regions
                # Use a small kernel to gradually expand inward
                small_kernel = np.ones((3, 3), np.uint8)
                dilated_internal = cv2.dilate(binary_mask, small_kernel, iterations=1)
                
                # Find pixels that are inside white regions (surrounded by white) but not yet marked
                # These are pixels that are within the dilated region but currently black
                internal_pixels = (dilated_internal > 0) & (binary_mask == 0) & valid_mask
                
                # Use convolution to efficiently check if pixels are mostly surrounded by white
                # Count white neighbors using a 3x3 kernel (excluding center)
                kernel_3x3 = np.ones((3, 3), np.float32)
                kernel_3x3[1, 1] = 0  # Don't count center pixel
                neighbor_count = cv2.filter2D((binary_mask > 0).astype(np.float32), -1, kernel_3x3)
                
                # Pixels with 6+ white neighbors (out of 8) are considered mostly surrounded
                # Fill them regardless of depth difference
                mostly_surrounded = internal_pixels & (neighbor_count >= 6)
                binary_mask[mostly_surrounded] = 255
                
                # For other internal pixels, use a very low threshold
                remaining_internal = internal_pixels & ~mostly_surrounded
                internal_fill_pixels = remaining_internal & (depth_diff > internal_fill_threshold)
                binary_mask[internal_fill_pixels] = 255
                
                # Stop early if no new pixels were filled
                if not (np.any(mostly_surrounded) or np.any(internal_fill_pixels)):
                    break
        
        # Create direction map: 1.0 = inward dent, -1.0 = outward dent, 0.0 = no dent
        # For INTERNAL camera: flip sign because inward dent (closer) has negative signed_diff, outward (farther) has positive
        direction_map = np.zeros_like(signed_depth_diff, dtype=np.float32)
        direction_map[valid_mask & (depth_diff > threshold)] = -np.sign(signed_depth_diff[valid_mask & (depth_diff > threshold)])
        
        # Save pure binary mask before morphology operations (after gap-filling, before opening/closing)
        pure_binary_mask = binary_mask.copy()
        
        # Apply morphology operations only if not skipped
        if not skip_morphology:
            # Apply morphological opening (erosion followed by dilation) to remove noise
            # Opening removes small isolated white regions and smooths boundaries
            if morphology_opening_size > 0:
                # Ensure kernel size is odd (required by OpenCV)
                opening_size = morphology_opening_size
                if opening_size % 2 == 0:
                    opening_size += 1
                    logger.debug(f"Morphology opening kernel size adjusted to {opening_size} (must be odd)")
                
                opening_kernel = np.ones((opening_size, opening_size), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, opening_kernel)
            
            # Apply morphological closing (dilation followed by erosion) to fill gaps and connect regions
            # Closing fills small gaps and thin black lines inside dented regions
            if morphology_closing_size > 0:
                # Ensure kernel size is odd (required by OpenCV)
                closing_size = morphology_closing_size
                if closing_size % 2 == 0:
                    closing_size += 1
                    logger.debug(f"Morphology closing kernel size adjusted to {closing_size} (must be odd)")
                
                closing_kernel = np.ones((closing_size, closing_size), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)
            
            # Post-morphology aggressive fill: Fill black pixels inside white regions regardless of depth
            # This catches any remaining black lines that weren't filled by morphological operations
            binary_mask = self._aggressive_internal_fill(binary_mask, valid_mask)
            
            # Fill black holes inside white dented regions
            binary_mask = self._fill_holes(binary_mask)
            
            # Remove small and thin false-positive regions using connected-component analysis
            binary_mask = self._filter_thin_components(binary_mask, min_area=min_dent_area)
        else:
            # If morphology is skipped, binary_mask remains as pure_binary_mask
            # This allows caller to apply custom morphology after depth filtering
            pass
        
        # Update direction map to match final binary mask (only where mask is white)
        direction_map = direction_map * (binary_mask > 0).astype(np.float32)
        
        return depth_diff, binary_mask, pure_binary_mask, direction_map
    
    def _depth_to_points_camera_space(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to 3D points in camera space.
        
        Args:
            depth: Depth map (H, W) in meters
            
        Returns:
            Tuple of (points_3d, valid_mask)
            - points_3d: Nx3 array of 3D points in camera space (x, y, z)
            - valid_mask: Boolean mask indicating valid pixels (H, W)
        """
        height, width = depth.shape
        
        # Use pre-computed camera intrinsics if depth map matches expected size
        # Otherwise calculate from actual dimensions (should be rare)
        if height == self.image_size and width == self.image_size:
            focal_length = self.focal_length
            cx, cy = self.cx, self.cy
        else:
            # Fallback: calculate for non-standard depth map dimensions
            fov_y_rad = np.deg2rad(self.camera_fov)
            focal_length = (height / 2.0) / np.tan(fov_y_rad / 2.0)
            cx, cy = width / 2.0, height / 2.0
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to normalized camera coordinates
        x_norm = (u - cx) / focal_length
        y_norm = (v - cy) / focal_length
        
        # Get valid depth pixels (non-zero and finite)
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        # Back-project to 3D points in camera space
        x_cam = x_norm[valid_mask] * depth[valid_mask]
        y_cam = y_norm[valid_mask] * depth[valid_mask]
        z_cam = depth[valid_mask]
        
        # Stack into Nx3 array
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        return points_cam, valid_mask
    
    def _fit_plane_ransac(self, points: np.ndarray, 
                         residual_threshold: float = 0.01,
                         max_trials: int = 1000,
                         min_samples: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a plane to 3D points using RANSAC.
        
        Args:
            points: Nx3 array of 3D points
            residual_threshold: Maximum distance from a point to the plane to be considered an inlier (meters)
            max_trials: Maximum number of RANSAC iterations
            min_samples: Minimum number of samples to fit a plane
            
        Returns:
            Tuple of (inlier_mask, plane_coefficients)
            - inlier_mask: Boolean array indicating inlier points
            - plane_coefficients: [a, b, c, d] where ax + by + cz + d = 0
        """
        if len(points) < min_samples:
            logger.warning(f"Not enough points for RANSAC: {len(points)} < {min_samples}")
            return np.zeros(len(points), dtype=bool), np.array([0, 0, 1, 0])
        
        # Use sklearn's RANSACRegressor for plane fitting
        # We'll fit z as a function of x and y: z = ax + by + c
        # This is equivalent to fitting the plane ax + by - z + c = 0
        
        X = points[:, :2]  # x, y coordinates
        y = points[:, 2]    # z coordinates
        
        # Create RANSAC regressor
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            min_samples=min_samples,
            random_state=42
        )
        
        try:
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            
            # Get plane coefficients: z = ax + by + c
            # Convert to plane equation: ax + by - z + c = 0
            # So plane_coefficients = [a, b, -1, c]
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
            plane_coefficients = np.array([a, b, -1.0, c])
            
            logger.info(f"RANSAC plane fitting: {np.sum(inlier_mask)}/{len(points)} inliers "
                       f"({100*np.sum(inlier_mask)/len(points):.1f}%)")
            
            return inlier_mask, plane_coefficients
            
        except Exception as e:
            logger.warning(f"RANSAC fitting failed: {e}. Using all points as inliers.")
            return np.ones(len(points), dtype=bool), np.array([0, 0, -1, 0])
    
    def _calculate_adaptive_sigma(self, depth: np.ndarray, 
                                  min_sigma: float = 4.0, 
                                  max_sigma: float = 18.0,
                                  base_sigma: float = 8.0) -> float:
        """
        Calculate adaptive sigma for Gaussian smoothing based on depth variance.
        
        Higher variance indicates more corrugation/variation, requiring more smoothing.
        Lower variance indicates flatter surfaces, requiring less smoothing.
        
        Args:
            depth: Depth map (H, W) in meters
            min_sigma: Minimum sigma value (default: 3.0)
            max_sigma: Maximum sigma value (default: 12.0)
            base_sigma: Base sigma value for normalization (default: 6.0)
            
        Returns:
            Adaptive sigma value for Gaussian smoothing
        """
        # Get valid depth pixels
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return base_sigma
        
        valid_depths = depth[valid_mask]
        
        # Calculate depth variance
        depth_variance = np.var(valid_depths)
        depth_std = np.std(valid_depths)
        
        # Calculate mean depth for normalization
        depth_mean = np.mean(valid_depths)
        
        # Normalize variance by mean depth to get relative variation
        # This accounts for different camera distances
        relative_variance = depth_variance / (depth_mean ** 2 + 1e-6)
        relative_std = depth_std / (depth_mean + 1e-6)
        
        # Scale sigma based on relative standard deviation
        # Higher relative std = more corrugation = higher sigma needed
        # Use a scaling factor: sigma = base_sigma * (1 + relative_std * scale_factor)
        scale_factor = 2.0  # Adjust this to control sensitivity
        adaptive_sigma = base_sigma * (1.0 + relative_std * scale_factor)
        
        # Clamp to min/max bounds
        adaptive_sigma = np.clip(adaptive_sigma, min_sigma, max_sigma)
        
        logger.info(f"    Depth variance: {depth_variance:.6f}, relative std: {relative_std:.4f}, "
                   f"adaptive sigma: {adaptive_sigma:.2f}")
        
        return float(adaptive_sigma)
    
    def _apply_gaussian_smoothing(self, depth: np.ndarray, sigma: Optional[float] = None, 
                                 adaptive: bool = True) -> Tuple[np.ndarray, float]:
        """
        Apply Gaussian smoothing to depth map to flatten corrugation patterns.
        
        Args:
            depth: Depth map (H, W) in meters
            sigma: Standard deviation for Gaussian kernel (if None and adaptive=True, will be calculated)
            adaptive: If True, automatically tune sigma based on depth variance
            
        Returns:
            Tuple of (smoothed_depth_map, sigma_used)
        """
        # Calculate adaptive sigma if requested
        if adaptive and sigma is None:
            sigma = self._calculate_adaptive_sigma(depth)
        elif sigma is None:
            sigma = 6.0  # Default fallback
        
        # Only smooth valid depth pixels (non-zero and finite)
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return depth.copy(), sigma
        
        # Create a copy and apply Gaussian filter
        smoothed_depth = depth.copy()
        
        # Apply Gaussian filter to valid regions
        # Use gaussian_filter which handles NaN/inf gracefully by only filtering valid pixels
        smoothed = gaussian_filter(depth, sigma=sigma, mode='constant', cval=0.0)
        
        # Preserve invalid pixels (set smoothed invalid pixels back to original)
        smoothed_depth[valid_mask] = smoothed[valid_mask]
        smoothed_depth[~valid_mask] = depth[~valid_mask]
        
        return smoothed_depth, sigma
    
    def _calculate_adaptive_residual_threshold(self, depth: np.ndarray,
                                             base_threshold: float = 0.02,
                                             min_threshold: float = 0.015,
                                             max_threshold: float = 0.05) -> float:
        """
        Calculate adaptive residual threshold for RANSAC based on corrugation depth.
        
        Deeper corrugations require a more lenient threshold to include all panel sections.
        
        Args:
            depth: Depth map (H, W) in meters
            base_threshold: Base threshold value (default: 0.02m = 2cm)
            min_threshold: Minimum threshold value (default: 0.015m = 1.5cm)
            max_threshold: Maximum threshold value (default: 0.05m = 5cm)
            
        Returns:
            Adaptive residual threshold in meters
        """
        # Get valid depth pixels
        valid_mask = (depth > 0) & np.isfinite(depth)
        
        if not np.any(valid_mask):
            return base_threshold
        
        valid_depths = depth[valid_mask]
        
        # Calculate depth range (corrugation depth)
        depth_min = np.min(valid_depths)
        depth_max = np.max(valid_depths)
        depth_range = depth_max - depth_min
        
        # Calculate mean depth for normalization
        depth_mean = np.mean(valid_depths)
        
        # Normalize range by mean depth to get relative corrugation depth
        relative_range = depth_range / (depth_mean + 1e-6)
        
        # Scale threshold based on relative corrugation depth
        # Higher relative range = deeper corrugations = higher threshold needed
        scale_factor = 1.5  # Adjust this to control sensitivity
        adaptive_threshold = base_threshold * (1.0 + relative_range * scale_factor)
        
        # Clamp to min/max bounds
        adaptive_threshold = np.clip(adaptive_threshold, min_threshold, max_threshold)
        
        logger.info(f"    Depth range: {depth_range:.4f}m, relative range: {relative_range:.4f}, "
                   f"adaptive residual threshold: {adaptive_threshold*1000:.1f}mm")
        
        return float(adaptive_threshold)
    
    def _generate_panel_mask_ransac(self, depth: np.ndarray,
                                   residual_threshold: Optional[float] = None,
                                   adaptive_threshold: bool = True,
                                   max_trials: int = 1000) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate a binary panel mask using RANSAC plane fitting.
        
        Applies RANSAC to detect the main container panel surface and creates a binary mask
        from the inliers (pixels belonging to the main panel).
        
        Args:
            depth: Depth map (H, W) in meters
            residual_threshold: Maximum distance from plane to be considered an inlier (meters)
                                If None and adaptive_threshold=True, will be calculated adaptively
            adaptive_threshold: If True, automatically tune residual threshold based on corrugation depth
            max_trials: Maximum number of RANSAC iterations
            
        Returns:
            Tuple of (panel_mask, plane_coefficients)
            - panel_mask: Binary mask (H, W) where True/1.0 = panel pixels (inliers), False/0.0 = other pixels (outliers)
            - plane_coefficients: [a, b, -1, c] where z = ax + by + c in camera space, or None if RANSAC failed
        """
        # Calculate adaptive residual threshold if requested
        if adaptive_threshold and residual_threshold is None:
            residual_threshold = self._calculate_adaptive_residual_threshold(depth)
        elif residual_threshold is None:
            residual_threshold = 0.02  # Default fallback (2cm)
        
        # Convert depth map to 3D points
        points_3d, valid_mask = self._depth_to_points_camera_space(depth)
        
        if len(points_3d) == 0:
            logger.warning("No valid points found in depth map for RANSAC")
            return np.zeros_like(depth, dtype=np.float32), None
        
        # Apply RANSAC plane fitting
        inlier_mask_points, plane_coefficients = self._fit_plane_ransac(
            points_3d, 
            residual_threshold=residual_threshold,
            max_trials=max_trials
        )
        
        # Create binary mask image from inlier mask (0 or 1 for multiplication)
        panel_mask = np.zeros_like(depth, dtype=np.float32)
        
        # Map inlier points back to image coordinates
        valid_indices = np.where(valid_mask)
        inlier_indices = np.where(inlier_mask_points)[0]
        
        # Create mapping from point index to pixel coordinates
        if len(inlier_indices) > 0:
            # Get pixel coordinates for inlier points
            inlier_pixels = (valid_indices[0][inlier_indices], valid_indices[1][inlier_indices])
            panel_mask[inlier_pixels] = 1.0  # 1.0 for panel pixels (for multiplication)
        
        logger.info(f"Panel mask: {np.sum(panel_mask > 0)}/{panel_mask.size} pixels "
                   f"({100*np.sum(panel_mask > 0)/panel_mask.size:.1f}%)")
        
        return panel_mask, plane_coefficients
    
    def _fill_internal_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill black holes (dents) inside the white panel mask using external contour filling.
        
        RANSAC correctly identifies dents as outliers (not part of the flat plane), creating
        holes in the mask. This method finds the outer boundary of the panel and fills
        everything inside it, effectively including dents back into the panel ROI.
        
        Strategy: Finds the outer-most boundary (external contour) and fills it solid,
        which patches all internal holes (dents) instantly.
        
        Args:
            mask: Binary mask (H, W) where 1.0 = panel pixels, 0.0 = background
            
        Returns:
            Filled mask (H, W) with holes filled (float32, 0.0 or 1.0)
        """
        # Ensure mask is uint8 (0 or 255) for OpenCV operations
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours - RETR_EXTERNAL = Only find the outer-most boundary (ignores inner holes)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No contours found in panel mask, returning original mask")
            return mask
        
        # Create a new solid mask
        filled_mask = np.zeros_like(mask_uint8)
        
        # Find the largest contour (The Container Panel)
        # This filters out tiny noise blobs outside the container
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw it SOLID (thickness = cv2.FILLED or -1)
        # This paints the entire shape white, erasing any internal black holes
        cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Count how many pixels were filled
        pixels_before = np.sum(mask > 0)
        pixels_after = np.sum(filled_mask > 0)
        pixels_filled = pixels_after - pixels_before
        
        if pixels_filled > 0:
            logger.info(f"    Filled {pixels_filled} pixels ({100*pixels_filled/pixels_before:.1f}% increase) "
                       f"to patch holes in panel mask")
        
        # Convert back to float (0.0 / 1.0) for consistency
        return (filled_mask > 0).astype(np.float32)
    
    def _get_main_panel_mask(self, ransac_mask: np.ndarray,
                            erosion_kernel_size: int = 7, 
                            erosion_iterations: int = 3) -> np.ndarray:
        """
        Refines the RANSAC mask to keep ONLY the Main Panel, 
        removing connected bars/frames at the edges.
        
        This method implements a 'Largest Connected Component' filter:
        1. Erodes the mask to break connections between panel and frame
        2. Finds all connected components
        3. Keeps only the largest blob (the main panel)
        4. Discards all smaller blobs (bars/frames)
        
        Args:
            ransac_mask: Binary mask (H, W) from RANSAC, where 1.0 = panel pixels
            erosion_kernel_size: Size of erosion kernel to break connections (default: 7x7)
            erosion_iterations: Number of erosion iterations (default: 3)
        
        Returns:
            main_panel_mask: Binary mask (H, W) containing only the largest connected component
        """
        # Step 1: Convert to uint8 for OpenCV operations
        mask_uint8 = (ransac_mask > 0).astype(np.uint8) * 255
        
        # Step 2: ERODE - Break the links between Panel and Frame
        # A larger kernel ensures we snap thin connections between panel and frame
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        # Aggressive shrinking to ensure separation
        eroded = cv2.erode(mask_uint8, kernel, iterations=erosion_iterations)
        
        # Step 3: CONNECTED COMPONENTS - Find all separate objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        
        # Step 4: FILTER - Find the Largest Component (The Panel)
        # stats[:, cv2.CC_STAT_AREA] is the Area
        # Label 0 is background, so we skip it
        if num_labels < 2:
            logger.warning("No objects found after erosion, using original RANSAC mask as fallback")
            return ransac_mask
        
        # Find index of largest blob (skipping background at index 0)
        # stats[1:, cv2.CC_STAT_AREA] gets areas of all non-background components
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_area = stats[largest_label, cv2.CC_STAT_AREA]
        
        logger.info(f"Found {num_labels-1} connected components after erosion. "
                   f"Largest component (label {largest_label}) has {largest_area} pixels")
        
        # Create mask for ONLY the largest blob
        main_panel_mask = (labels == largest_label).astype(np.float32)
        
        # Log refinement statistics
        original_pixels = np.sum(ransac_mask > 0)
        refined_pixels = np.sum(main_panel_mask > 0)
        logger.info(f"Panel mask refinement: {original_pixels} -> {refined_pixels} pixels "
                   f"({100*refined_pixels/max(original_pixels,1):.1f}% retained)")
        
        # Note: We do NOT dilate back to restore size
        # Keeping it eroded acts as a natural "Safety Margin" away from the bars/frames
        # This ensures we are definitely on the main panel, not near frame connections
        
        return main_panel_mask
    
    def _apply_mask_to_depth(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply binary mask to depth map using multiplication.

        Args:
            depth: Depth map (H, W) in meters
            mask: Binary mask (H, W) where 1.0 = keep, 0.0 = remove

        Returns:
            Masked depth map (H, W) where non-masked pixels are set to 0
        """
        # Use multiplication: masked_depth = depth * mask
        masked_depth = depth * mask
        return masked_depth

    def _create_sanitized_mask(self, panel_mask: np.ndarray, original_depth: np.ndarray, 
                               dented_depth: np.ndarray, erosion_kernel_size: int = 5, 
                               depth_consistency_threshold: float = 0.1,
                               opening_kernel_size: int = 5,
                               final_erosion_iterations: int = 1) -> np.ndarray:
        """
        Create a sanitized mask using intersection logic to handle misalignment artifacts at container edges.
        
        This method implements a 'Sanitization Intersection' step that:
        1. Starts with the RANSAC-generated panel_mask as the base
        2. Erodes edges to remove edge noise
        3. Filters out pixels where dented_depth <= 0 (panel moved and exposed background)
        4. Filters out pixels where depth difference exceeds threshold (ghost dents from background wall)
        5. Applies morphological opening to smooth edges and remove noise islands
        6. Applies final safety erosion to create a clean buffer away from edge artifacts
        
        Args:
            panel_mask: Binary mask (H, W) from RANSAC plane fitting, where 1.0 = panel pixels
            original_depth: Original depth map (H, W) in meters
            dented_depth: Dented depth map (H, W) in meters
            erosion_kernel_size: Size of initial erosion kernel (default: 5x5)
            depth_consistency_threshold: Maximum allowed depth difference in meters (default: 0.5)
            opening_kernel_size: Size of kernel for morphological opening (default: 5x5)
            final_erosion_iterations: Number of iterations for final safety erosion (default: 1)
        
        Returns:
            final_safe_mask: Binary mask (H, W) where 1.0 = valid overlapping surface pixels
        """
        # Step 1: Start with the original RANSAC panel mask
        final_safe_mask = panel_mask.copy()
        
        # Step 2: Erode edges to remove edge noise
        # Convert mask to uint8 for cv2 operations (0 or 255)
        mask_uint8 = (final_safe_mask * 255).astype(np.uint8)
        erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        eroded_mask = cv2.erode(mask_uint8, erosion_kernel, iterations=1)
        # Convert back to float (0.0 or 1.0)
        final_safe_mask = (eroded_mask / 255.0).astype(np.float32)
        
        # Step 3: Check dented_depth > 0 (filters out where panel moved and exposed background)
        valid_dented_mask = (dented_depth > 0) & np.isfinite(dented_depth)
        final_safe_mask = final_safe_mask * valid_dented_mask.astype(np.float32)
        
        # Step 4: Check depth consistency ONLY at edges (not in internal regions where dents are)
        # This removes 'Ghost Dents' at edges caused by camera seeing distant background wall through a gap
        # But preserves internal holes (dents) that were filled by _fill_internal_holes()
        depth_diff = np.abs(original_depth - dented_depth)
        
        # Identify edge regions: pixels near the boundary of the panel mask
        # Edge regions are where ghost dents occur, internal regions contain real dents
        mask_uint8_for_edge = (final_safe_mask * 255).astype(np.uint8)
        # Dilate to find edge region (boundary + small buffer)
        edge_buffer = 10  # pixels from edge
        edge_kernel = np.ones((edge_buffer * 2 + 1, edge_buffer * 2 + 1), np.uint8)
        dilated_mask = cv2.dilate(mask_uint8_for_edge, edge_kernel, iterations=1)
        # Edge region = dilated area - original mask (boundary region)
        edge_region = (dilated_mask > 0) & (mask_uint8_for_edge == 0)
        
        # Apply depth consistency check ONLY to edge regions
        # Internal regions (where dents are) are preserved regardless of depth difference
        depth_consistent_mask = np.ones_like(final_safe_mask, dtype=bool)
        if np.any(edge_region):
            # At edges: filter out large depth differences (ghost dents from background)
            edge_depth_consistent = (depth_diff < depth_consistency_threshold) & np.isfinite(depth_diff)
            depth_consistent_mask[edge_region] = edge_depth_consistent[edge_region]
            # Internal regions: keep all pixels (preserve filled dents)
            # depth_consistent_mask[~edge_region] remains True (already set above)
        
        final_safe_mask = final_safe_mask * depth_consistent_mask.astype(np.float32)
        
        # Ensure original_depth is also valid (finite and > 0) for the final mask
        valid_original_mask = (original_depth > 0) & np.isfinite(original_depth)
        final_safe_mask = final_safe_mask * valid_original_mask.astype(np.float32)
        
        # Step 5: Morphological smoothing to remove noise islands and smooth jagged edges
        # Convert to uint8 for OpenCV operations
        raw_safe_mask_uint8 = (final_safe_mask * 255).astype(np.uint8)
        
        # Step 5A: Morphological Opening (Erosion -> Dilation)
        # This removes small "islands" (broken floating pixels) and smooths sharp spikes
        # Opening removes noise islands and creates clean, smooth edges
        opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)
        smoothed_mask = cv2.morphologyEx(raw_safe_mask_uint8, cv2.MORPH_OPEN, opening_kernel)
        
        # Step 5B: Final Safety Erosion
        # Shrink the mask slightly to create a safety buffer away from edge artifacts
        # This ensures we are definitely on solid metal, not near dangerous edge regions
        if final_erosion_iterations > 0:
            final_clean_mask = cv2.erode(smoothed_mask, opening_kernel, iterations=final_erosion_iterations)
        else:
            final_clean_mask = smoothed_mask
        
        # Convert back to float (0.0 or 1.0)
        final_safe_mask = (final_clean_mask / 255.0).astype(np.float32)
        
        return final_safe_mask

    def _aggressive_internal_fill(self, binary_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Aggressively fill black pixels inside white dented regions regardless of depth difference.
        
        This method fills any black pixels that are inside or mostly surrounded by white regions,
        treating them as part of the dented area. This is applied after morphological operations
        to catch any remaining black lines that weren't filled.
        
        Args:
            binary_mask: Input binary mask (uint8), with dent pixels = 255, background = 0
            valid_mask: Boolean mask indicating valid pixels
            
        Returns:
            Binary mask with internal black pixels filled
        """
        filled_mask = binary_mask.copy()
        
        # Use convolution to efficiently count white neighbors
        # Count white neighbors using a 3x3 kernel (excluding center)
        kernel_3x3 = np.ones((3, 3), np.float32)
        kernel_3x3[1, 1] = 0  # Don't count center pixel
        neighbor_count = cv2.filter2D((binary_mask > 0).astype(np.float32), -1, kernel_3x3)
        
        # Find black pixels that are inside white regions
        # A pixel is considered "inside" if it has 5+ white neighbors (out of 8)
        # This catches black lines and gaps within white regions
        black_pixels = (binary_mask == 0) & valid_mask
        mostly_surrounded = black_pixels & (neighbor_count >= 5)
        
        # Fill these pixels with white
        filled_mask[mostly_surrounded] = 255
        
        return filled_mask

    def _fill_holes(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Fill black holes (black regions) inside white dented regions.
        
        This method identifies black regions that are completely surrounded by white regions
        and fills them with white, treating them as part of the dented area.
        
        Args:
            binary_mask: Input binary mask (uint8), with dent pixels = 255, background = 0
            
        Returns:
            Binary mask with holes filled (black regions inside white regions become white)
        """
        # Create a copy to avoid modifying the original
        filled_mask = binary_mask.copy()
        
        # Invert the mask to find black regions (holes)
        # In inverted mask: white (255) = holes/background, black (0) = dented regions
        inverted_mask = 255 - binary_mask
        
        # Find connected components in the inverted mask
        # This will identify all black regions (holes and background)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
        
        h, w = binary_mask.shape
        
        # Iterate through all components (skip label 0, which is the background in inverted mask)
        for label in range(1, num_labels):
            x, y, w_comp, h_comp, area = stats[label]
            
            # Check if this component touches any border
            # If it touches the border, it's background, not a hole
            touches_border = (
                x <= 0 or y <= 0 or 
                (x + w_comp) >= w or 
                (y + h_comp) >= h
            )
            
            # If it doesn't touch the border, it's a hole inside a white region
            # Fill it with white (255)
            if not touches_border:
                filled_mask[labels == label] = 255
        
        return filled_mask

    def _filter_thin_components(self, binary_mask: np.ndarray,
                            min_area: int = 80,
                            min_thickness: int = 5,
                            max_aspect_ratio: float = 12.0) -> np.ndarray:
        """
        Remove thin or small false-positive components from the binary dent mask.

        Args:
            binary_mask: Input binary mask (uint8), with dent pixels = 255
            min_area: Minimum area (in pixels) required to keep a component
            min_thickness: Minimum width/height allowed (removes 1–4 pixel thin lines)
            max_aspect_ratio: Maximum allowed bounding-box aspect ratio
                            (components longer than this ratio are removed)

        Returns:
            Cleaned binary mask with thin/small components removed.
        """
        # Copy to avoid modifying original directly
        cleaned = np.zeros_like(binary_mask)

        # Connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        total_components = num_labels - 1  # Exclude background
        removed_count = 0
        kept_count = 0
        removed_by_area = 0
        removed_by_thickness = 0
        removed_by_aspect = 0

        for label in range(1, num_labels):   # Skip label 0 (background)
            x, y, w, h, area = stats[label]

            # --- RULE 1: Remove small noisy regions ---
            if area < min_area:
                removed_by_area += 1
                removed_count += 1
                continue

            # --- RULE 2: Remove very thin lines (1–4 px wide) ---
            if w < min_thickness or h < min_thickness:
                removed_by_thickness += 1
                removed_count += 1
                continue

            # --- RULE 3: Remove long straight lines (very elongated) ---
            aspect_ratio = max(w / float(h), h / float(w))
            if aspect_ratio > max_aspect_ratio:
                removed_by_aspect += 1
                removed_count += 1
                continue

            # Passed all filters → keep component
            cleaned[labels == label] = 255
            kept_count += 1

        # Log filtering statistics if components were removed
        if removed_count > 0:
            logger.debug(f"Component filtering: {kept_count} kept, {removed_count} removed "
                        f"(area<{min_area}: {removed_by_area}, thin: {removed_by_thickness}, "
                        f"elongated: {removed_by_aspect})")

        return cleaned

    
    def process_container_pair(self, original_path: Path, dented_path: Path, 
                              output_dir: Path, container_type: str = "20ft",
                              threshold: float = 0.035, min_area_cm2: float = 1.0,
                              dataset_dir: Path = None, save_rgb_to_dataset: bool = False,
                              is_testset: bool = False) -> Dict:
        """
        Process a pair of original and dented containers.
        
        Args:
            original_path: Path to original container OBJ file
            dented_path: Path to dented container OBJ file
            output_dir: Directory to save output images
            container_type: Container type ("20ft", "40ft", "40ft_hc")
            threshold: Depth difference threshold in meters
            min_area_cm2: Minimum area threshold in cm² for dent segments (default: 1.0 cm²)
            dataset_dir: Optional custom dataset directory (defaults to "output_scene_dataset")
            save_rgb_to_dataset: Whether to save RGB images to dataset folder (deprecated, use is_testset instead)
            is_testset: If True, saves additional files (raw depth, RGB) for testset generation.
                       If False (regular rendering), only saves _dent_mask.png and _dented_depth.npy
            
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
                
                # Save raw depth maps (before RANSAC masking) - these are the raw camera captures
                shot_output_dir = output_dir / shot_name
                shot_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save raw depth maps to output_scene folder
                original_depth_raw_path = shot_output_dir / f"{base_name}_original_depth_raw.npy"
                dented_depth_raw_path = shot_output_dir / f"{base_name}_dented_depth_raw.npy"
                np.save(original_depth_raw_path, original_depth.astype(np.float32))
                np.save(dented_depth_raw_path, dented_depth.astype(np.float32))
                
                # Apply Gaussian smoothing to flatten corrugation pattern before RANSAC
                # Automatically tune sigma based on depth variance
                logger.info(f"    Applying Gaussian smoothing with adaptive sigma to flatten corrugation pattern...")
                original_depth_smoothed, sigma_used = self._apply_gaussian_smoothing(original_depth, adaptive=True)
                logger.info(f"    Using sigma={sigma_used:.2f} for Gaussian smoothing")
                
                # Apply RANSAC plane fitting to detect main panel surface (using smoothed original depth map)
                # Use adaptive residual threshold to include all corrugations (upper and lower)
                logger.info(f"    Applying RANSAC plane fitting to detect main panel surface...")
                raw_panel_mask, plane_coefficients = self._generate_panel_mask_ransac(
                    original_depth_smoothed,
                    adaptive_threshold=True,  # Automatically tune threshold based on corrugation depth
                    max_trials=1000
                )
                
                # Fill internal holes (dents) that RANSAC correctly identified as outliers
                # RANSAC creates holes at dent locations because dents are not part of the flat plane
                # This step fills those holes so dents are included in the panel ROI
                logger.info(f"    Filling internal holes (dents) in panel mask...")
                filled_panel_mask = self._fill_internal_holes(raw_panel_mask)
                
                # Refine filled mask to keep ONLY the main panel, excluding bars/frames at edges
                logger.info(f"    Refining panel mask to exclude edge cases (bars/frames)...")
                panel_mask = self._get_main_panel_mask(
                    filled_panel_mask,  # Use filled mask (with dents included)
                    erosion_kernel_size=7,
                    erosion_iterations=3
                )
                
                # Create sanitized mask using intersection logic to handle misalignment artifacts at edges
                logger.info(f"    Creating sanitized mask to handle edge misalignment artifacts...")
                final_safe_mask = self._create_sanitized_mask(
                    panel_mask, original_depth, dented_depth,
                    erosion_kernel_size=5,
                    depth_consistency_threshold=0.1
                )
                
                # Log mask statistics
                panel_pixels = np.sum(panel_mask > 0)
                safe_pixels = np.sum(final_safe_mask > 0)
                logger.info(f"    Panel mask: {panel_pixels} pixels, Safe mask: {safe_pixels} pixels "
                           f"({100*safe_pixels/max(panel_pixels,1):.1f}% retained)")
                
                # Apply sanitized mask to both original (unsmoothed) depth maps for accurate dent measurement
                # This ensures we calculate difference only on valid overlapping surface
                original_depth_masked = self._apply_mask_to_depth(original_depth, final_safe_mask)
                dented_depth_masked = self._apply_mask_to_depth(dented_depth, final_safe_mask)
                
                # Compare depths using masked depth maps to detect dent locations
                # This initial comparison identifies where dents are
                # Skip morphology here since we'll apply it after depth filtering
                depth_diff_initial, dent_mask_unused, pure_dent_mask, direction_map = self.compare_depths(
                    original_depth_masked, dented_depth_masked, threshold, skip_morphology=True
                )
                
                # Calculate depth relative to median panel depth using pure_mask (before morphology)
                # This provides accurate depth measurement for filtering segments
                # Use RANSAC plane depth to eliminate perspective distortion
                # Use original depth map to calculate median for accurate dent depth measurement
                logger.info(f"    Calculating depth relative to median panel depth (for pure mask filtering)...")
                empty_mask = np.zeros_like(pure_dent_mask, dtype=np.uint8)
                median_panel_depth = self._calculate_median_panel_depth(original_depth_masked, empty_mask, final_safe_mask, plane_coefficients)
                logger.info(f"    Median panel depth: {median_panel_depth:.4f}m ({median_panel_depth*1000:.2f}mm)")
                depth_diff_median = self._calculate_depth_diff_from_median(dented_depth_masked, pure_dent_mask, final_safe_mask, plane_coefficients, original_depth_map=original_depth_masked)
                
                # Filter pure_mask segments by minimum depth threshold (10mm) before morphology operations
                # Compare each segment to median depth of its local corrugation wall
                logger.info(f"    Filtering pure mask segments by minimum depth threshold: 10mm (relative to local corrugation wall)")
                valid_mask = (dented_depth_masked > 0) & np.isfinite(dented_depth_masked)
                filtered_pure_mask, depth_filtered_segments = self._filter_segments_by_depth(
                    pure_dent_mask, dented_depth_masked, depth_diff_median, direction_map, 
                    panel_mask=final_safe_mask, min_depth_mm=10.0, shot_name=shot_name
                )
                
                # Log depth filtering results
                num_segments_before_depth = len(self._analyze_dent_segments(
                    pure_dent_mask, dented_depth_masked, depth_diff_median, direction_map,
                    shot_name=shot_name
                ))
                num_segments_after_depth = len(depth_filtered_segments)
                logger.info(f"    Segments before depth filtering: {num_segments_before_depth}, after depth filtering (>=10mm): {num_segments_after_depth}")
                
                # Apply morphology operations to depth-filtered pure_mask
                logger.info(f"    Applying morphology operations to depth-filtered mask...")
                dent_mask = self._apply_morphology_operations(
                    filtered_pure_mask, valid_mask,
                    morphology_opening_size=9,
                    morphology_closing_size=11,
                    min_dent_area=200
                )
                
                # Filter dent segments by minimum area threshold (after morphology)
                logger.info(f"    Filtering dent segments by minimum area threshold: {min_area_cm2} cm²")
                filtered_dent_mask, dent_segments = self._filter_segments_by_area(
                    dent_mask, dented_depth_masked, depth_diff_median, direction_map, min_area_cm2,
                    shot_name=shot_name
                )
                
                # Log filtering results
                num_segments_before_area = len(self._analyze_dent_segments(
                    dent_mask, dented_depth_masked, depth_diff_median, direction_map,
                    shot_name=shot_name
                ))
                num_segments_after_area = len(dent_segments)
                logger.info(f"    Segments before area filtering: {num_segments_before_area}, after filtering: {num_segments_after_area}")
                
                # Use filtered mask for all subsequent operations
                dent_mask = filtered_dent_mask
                
                # Calculate raw depth difference: |dented_depth - original_depth|
                # This is the direct comparison between original and dented NPY files
                logger.info(f"    Calculating raw depth difference (original vs dented)...")
                depth_diff_raw = np.abs(dented_depth_masked - original_depth_masked)
                
                # Also calculate depth difference relative to median panel depth (for accurate dent measurement)
                # Use RANSAC plane depth to eliminate perspective distortion
                # Use original depth map to calculate median for accurate dent depth measurement
                logger.info(f"    Calculating depth relative to median panel depth (for dent analysis)...")
                empty_mask = np.zeros_like(dent_mask, dtype=np.uint8)
                median_panel_depth = self._calculate_median_panel_depth(original_depth_masked, empty_mask, final_safe_mask, plane_coefficients)
                logger.info(f"    Median panel depth: {median_panel_depth:.4f}m ({median_panel_depth*1000:.2f}mm)")
                depth_diff_median = self._calculate_depth_diff_from_median(dented_depth_masked, dent_mask, final_safe_mask, plane_coefficients, original_depth_map=original_depth_masked)
                
                # Use raw depth difference for visualization (what user wants to see)
                depth_diff = depth_diff_raw
                
                # Analyze segments with all depth measurements calculated per segment
                dent_segments = self._analyze_dent_segments(
                    dent_mask=dent_mask,
                    dented_depth_map=dented_depth_masked,
                    depth_diff_median=depth_diff_median,  # Use median-based depth diff for analysis
                    direction_map=direction_map,
                    original_depth_map=original_depth_masked,
                    depth_diff_original_vs_dented=depth_diff_initial,  # Raw difference for comparison
                    median_panel_depth=median_panel_depth,
                    panel_mask=final_safe_mask,
                    dilation_radius=30,
                    shot_name=shot_name
                )
                # Filter segments again by area threshold
                filtered_segments = [seg for seg in dent_segments if seg['area_cm2'] >= min_area_cm2]
                dent_segments = filtered_segments
                
                # Calculate max_depth_diff_mm for this snapshot (for dataset filtering)
                max_depth_diff_mm_snapshot = 0.0
                if len(dent_segments) > 0:
                    max_depth_diffs = [seg.get('max_depth_diff_mm', 0) for seg in dent_segments]
                    if max_depth_diffs:
                        max_depth_diff_mm_snapshot = float(max(max_depth_diffs))
                
                # Render RGB for saving (needed for visual output generation later)
                original_rgb, _ = self._render_rgb(original_mesh, pose)
                dented_rgb, _ = self._render_rgb(dented_mesh, pose)
                
                # Save outputs (shot_output_dir already created above when saving raw depth maps)
                
                # Save masked depth maps (after RANSAC panel detection)
                original_depth_path = shot_output_dir / f"{base_name}_original_depth.npy"
                dented_depth_path = shot_output_dir / f"{base_name}_dented_depth.npy"
                np.save(original_depth_path, original_depth_masked.astype(np.float32))
                np.save(dented_depth_path, dented_depth_masked.astype(np.float32))
                
                # Also save to dataset folder for training (directly in dataset_dir, no subfolders)
                # Filter: Skip saving to dataset if max_depth_diff_mm > 80mm
                if max_depth_diff_mm_snapshot <= 80.0:
                    if dataset_dir is None:
                        dataset_dir = Path("output_scene_dataset")
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save raw dented depth map to dataset folder ONLY for testset generation
                    if is_testset:
                        dataset_dented_depth_raw_path = dataset_dir / f"{base_name}_{shot_name}_dented_depth_raw.npy"
                        np.save(dataset_dented_depth_raw_path, dented_depth.astype(np.float32))
                        logger.info(f"    ✓ Saved raw dented depth map (before RANSAC): {base_name}_{shot_name}_dented_depth_raw.npy")
                    
                    # Save files directly in dataset_dir with full name including shot_name
                    dataset_dented_depth_path = dataset_dir / f"{base_name}_{shot_name}_dented_depth.npy"
                    np.save(dataset_dented_depth_path, dented_depth_masked.astype(np.float32))
                
                # Save raw panel mask from RANSAC (before refinement) for debugging
                raw_panel_mask_path = shot_output_dir / f"{base_name}_panel_mask_raw.npy"
                np.save(raw_panel_mask_path, raw_panel_mask.astype(np.float32))  # Save as float (0 or 1)
                # Convert to uint8 (0 or 255) for PNG visualization
                raw_panel_mask_uint8 = (raw_panel_mask * 255).astype(np.uint8)
                imageio.imwrite(shot_output_dir / f"{base_name}_panel_mask_raw.png", raw_panel_mask_uint8)
                
                # Save refined panel mask (after largest connected component filtering)
                panel_mask_path = shot_output_dir / f"{base_name}_panel_mask.npy"
                np.save(panel_mask_path, panel_mask.astype(np.float32))  # Save as float (0 or 1)
                # Convert to uint8 (0 or 255) for PNG visualization
                panel_mask_uint8 = (panel_mask * 255).astype(np.uint8)
                imageio.imwrite(shot_output_dir / f"{base_name}_panel_mask.png", panel_mask_uint8)
                
                # Save sanitized safe mask for debugging
                safe_mask_path = shot_output_dir / f"{base_name}_safe_mask.npy"
                np.save(safe_mask_path, final_safe_mask.astype(np.float32))  # Save as float (0 or 1)
                # Convert to uint8 (0 or 255) for PNG visualization
                safe_mask_uint8 = (final_safe_mask * 255).astype(np.uint8)
                imageio.imwrite(shot_output_dir / f"{base_name}_safe_mask.png", safe_mask_uint8)
                
                # Save debug visualization images for panel extraction pipeline
                logger.info(f"    Saving debug visualization images...")
                # 1. Original depth (unsmoothed)
                original_depth_img_debug = self._normalize_depth(original_depth)
                imageio.imwrite(shot_output_dir / f"{base_name}_debug_original_depth.png", original_depth_img_debug)
                
                # 2. Smoothed depth (used for RANSAC)
                smoothed_depth_img = self._normalize_depth(original_depth_smoothed)
                imageio.imwrite(shot_output_dir / f"{base_name}_debug_smoothed_depth.png", smoothed_depth_img)
                
                # Save depth difference
                depth_diff_path = shot_output_dir / f"{base_name}_depth_diff.npy"
                np.save(depth_diff_path, depth_diff.astype(np.float32))
                
                # Save normalized depth images for visualization (using masked depth maps)
                original_depth_img = self._normalize_depth(original_depth_masked)
                dented_depth_img = self._normalize_depth(dented_depth_masked)
                depth_diff_img = self._normalize_depth(depth_diff)
                
                imageio.imwrite(shot_output_dir / f"{base_name}_original_depth.png", original_depth_img)
                imageio.imwrite(shot_output_dir / f"{base_name}_dented_depth.png", dented_depth_img)
                imageio.imwrite(shot_output_dir / f"{base_name}_depth_diff.png", depth_diff_img)
                
                # Save RGB images
                imageio.imwrite(shot_output_dir / f"{base_name}_original_rgb.png", original_rgb)
                imageio.imwrite(shot_output_dir / f"{base_name}_dented_rgb.png", dented_rgb)
                
                # Generate and save point clouds as PLY files (using masked depth maps)
                try:
                    # Original point cloud with RGB colors
                    original_pcd = self._depth_to_pointcloud(original_depth_masked, pose, original_rgb)
                    original_ply_path = shot_output_dir / f"{base_name}_original_pointcloud.ply"
                    original_pcd.export(original_ply_path)
                    logger.info(f"    ✓ Saved original point cloud: {original_ply_path.name} ({len(original_pcd.vertices)} points)")
                    
                    # Dented point cloud with RGB colors
                    dented_pcd = self._depth_to_pointcloud(dented_depth_masked, pose, dented_rgb)
                    dented_ply_path = shot_output_dir / f"{base_name}_dented_pointcloud.ply"
                    dented_pcd.export(dented_ply_path)
                    logger.info(f"    ✓ Saved dented point cloud: {dented_ply_path.name} ({len(dented_pcd.vertices)} points)")
                except Exception as e:
                    logger.warning(f"    ⚠️  Failed to generate point clouds: {e}")
                
                # Validity check: ensure dent_mask never labels empty space (depth=0) as a dent
                # Map dent_mask onto dented_depth_masked and check for invalid pixels
                logger.info(f"    Performing validity check on dent_mask...")
                invalid_pixels = (dent_mask > 0) & (dented_depth_masked == 0)
                num_invalid_pixels = np.sum(invalid_pixels)
                if num_invalid_pixels > 0:
                    logger.warning(f"    ⚠️  Found {num_invalid_pixels} invalid pixels (dent_mask=1 but dented_depth=0), removing from mask")
                    dent_mask[invalid_pixels] = 0
                    logger.info(f"    ✓ Validity check complete: removed {num_invalid_pixels} invalid pixels")
                else:
                    logger.info(f"    ✓ Validity check passed: no invalid pixels found")
                
                # Apply sanitized safe_mask to dent_mask to ensure we don't label non-panel regions or edge artifacts as dents
                logger.info(f"    Applying sanitized safe_mask to dent_mask...")
                dent_mask_before_panel = dent_mask.copy()
                # Apply final_safe_mask: set dent_mask to 0 where final_safe_mask is 0 (non-panel regions or edge artifacts)
                dent_mask[final_safe_mask == 0] = 0
                num_removed_by_panel = np.sum((dent_mask_before_panel > 0) & (dent_mask == 0))
                if num_removed_by_panel > 0:
                    logger.info(f"    ✓ Safe mask applied: removed {num_removed_by_panel} pixels outside safe region")
                else:
                    logger.info(f"    ✓ Safe mask applied: all dent pixels are within safe region")
                
                # Save binary mask: WHITE (255) = dented areas (different depth), BLACK (0) = normal areas (same depth)
                imageio.imwrite(shot_output_dir / f"{base_name}_dent_mask.png", dent_mask)
                
                # Save mask as NPY for DL training (binary float32: 0.0 = background, 1.0 = dent)
                # Convert from uint8 (0/255) to float32 (0.0/1.0) for consistency with depth maps
                dent_mask_npy_path = shot_output_dir / f"{base_name}_dent_mask.npy"
                dent_mask_binary = (dent_mask > 127).astype(np.float32)
                np.save(dent_mask_npy_path, dent_mask_binary)
                logger.info(f"    ✓ Saved dent mask (NPY): {dent_mask_npy_path.name}")
                
                # Save pure binary mask (before morphology operations)
                imageio.imwrite(shot_output_dir / f"{base_name}_dent_mask_pure.png", pure_dent_mask)
                
                # Save dent segment information to JSON
                segment_json_path = shot_output_dir / f"{base_name}_dent_segments.json"
                segment_data = {
                    'shot_name': shot_name,
                    'container_type': container_type,
                    'min_area_threshold_cm2': min_area_cm2,
                    'depth_threshold_m': threshold,
                    'depth_threshold_mm': threshold * 1000.0,
                    'num_segments': len(dent_segments),
                    'segments': dent_segments,
                    'timestamp': datetime.now().isoformat()
                }
                with open(segment_json_path, 'w') as f:
                    json.dump(segment_data, f, indent=2)
                logger.info(f"    ✓ Saved dent segment information: {segment_json_path.name} ({len(dent_segments)} segments)")
                
                # Save to dataset folder for training (directly in dataset_dir, no subfolders)
                if dataset_dir is None:
                    dataset_dir = Path("output_scene_dataset")
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                # Filter: Skip saving to dataset if max_depth_diff_mm > 80mm
                if max_depth_diff_mm_snapshot > 80.0:
                    logger.info(f"    ⚠️  Skipping dataset save: max_depth_diff_mm ({max_depth_diff_mm_snapshot:.2f}mm) > 80mm")
                else:
                    # Always save dent mask and masked depth for training (both regular and testset)
                    dataset_dent_mask_path = dataset_dir / f"{base_name}_{shot_name}_dent_mask.png"
                    imageio.imwrite(dataset_dent_mask_path, dent_mask)
                    
                    # Save mask as NPY for DL training (binary float32: 0.0 = background, 1.0 = dent)
                    # Convert from uint8 (0/255) to float32 (0.0/1.0) for consistency with depth maps
                    dataset_dent_mask_npy_path = dataset_dir / f"{base_name}_{shot_name}_dent_mask.npy"
                    dent_mask_binary = (dent_mask > 127).astype(np.float32)
                    np.save(dataset_dent_mask_npy_path, dent_mask_binary)
                    logger.info(f"    ✓ Saved dent mask (NPY) to dataset: {base_name}_{shot_name}_dent_mask.npy")
                    
                    # Save RGB image to dataset folder ONLY for testset generation
                    if is_testset or save_rgb_to_dataset:
                        dataset_rgb_path = dataset_dir / f"{base_name}_{shot_name}_rgb.png"
                        imageio.imwrite(dataset_rgb_path, dented_rgb)
                        logger.info(f"    ✓ Saved RGB image to dataset: {base_name}_{shot_name}_rgb.png")
                    
                    # Save dent segment JSON to dataset folder (always save, matches other dataset files)
                    dataset_segment_json_path = dataset_dir / f"{base_name}_{shot_name}_dent_segments.json"
                    with open(dataset_segment_json_path, 'w') as f:
                        json.dump(segment_data, f, indent=2)
                    logger.info(f"    ✓ Saved dent segment JSON to dataset: {base_name}_{shot_name}_dent_segments.json")
                
                # Calculate basic statistics
                dent_pixels = np.sum(dent_mask > 0)
                total_pixels = dent_mask.size
                dent_percentage = (dent_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                
                # Calculate camera distance from panel
                camera_distance_m = self._calculate_camera_distance(pose)
                
                # Calculate total dent area in cm²
                dent_area_cm2 = self._calculate_dent_area(dent_mask, dented_depth_masked)
                
                # Calculate max dent depth across all segments in this snapshot
                max_dent_depth_mm = 0.0
                if len(dent_segments) > 0:
                    max_depths = [seg.get('max_depth_diff_vs_median_panel_mm', 0) for seg in dent_segments]
                    if max_depths:
                        max_dent_depth_mm = float(max(max_depths))
                
                # Log summary statistics
                logger.info(f"    ✓ Dent mask: {dent_pixels} pixels ({dent_percentage:.1f}%)")
                logger.info(f"    ✓ Total dent area: {dent_area_cm2:.2f} cm²")
                logger.info(f"    ✓ Number of segments: {len(dent_segments)}")
                if max_dent_depth_mm > 0:
                    logger.info(f"    ✓ Max dent depth (vs median panel): {max_dent_depth_mm:.2f}mm")
                
                # Add to results summary (basic stats only, detailed depth measurements are in segment JSON)
                results.append({
                    'shot_name': shot_name,
                    'dent_pixels': int(dent_pixels),
                    'total_pixels': int(total_pixels),
                    'dent_percentage': float(dent_percentage),
                    'median_panel_depth_m': float(median_panel_depth),
                    'median_panel_depth_mm': float(median_panel_depth * 1000),
                    'max_dent_depth_mm': float(max_dent_depth_mm),
                    'dent_area_cm2': float(dent_area_cm2),
                    'num_segments': len(dent_segments),
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
            'min_area_threshold_cm2': min_area_cm2,
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
        
        # Use pre-computed camera intrinsics if depth map matches expected size
        # Otherwise calculate from actual dimensions (should be rare)
        if height == self.image_size and width == self.image_size:
            focal_length = self.focal_length
            cx, cy = self.cx, self.cy
        else:
            # Fallback: calculate for non-standard depth map dimensions
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
    
    def _calculate_segment_area(self, segment_mask: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Calculate area of a single dent segment in cm² using camera intrinsics.
        
        Args:
            segment_mask: Binary mask (H, W) where True/255 = segment pixels
            depth_map: Depth map (H, W) in meters
            
        Returns:
            Segment area in cm²
        """
        h, w = segment_mask.shape
        segment_pixels = (segment_mask > 0)
        
        if not np.any(segment_pixels):
            return 0.0
        
        # Get depths for segment pixels
        segment_depths = depth_map[segment_pixels]
        valid_depths = segment_depths[segment_depths > 0]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Calculate average depth for segment region
        avg_depth = np.mean(valid_depths)
        
        # Calculate pixel dimensions in meters at average depth using camera intrinsics
        # Using pre-computed FOV (square images: fov_x = fov_y)
        pixel_width_m = 2 * avg_depth * np.tan(self.fov_x_rad / 2.0) / w
        pixel_height_m = 2 * avg_depth * np.tan(self.fov_x_rad / 2.0) / h
        pixel_area_m2 = pixel_width_m * pixel_height_m
        
        # Calculate total area
        num_segment_pixels = np.sum(segment_pixels)
        total_area_m2 = num_segment_pixels * pixel_area_m2
        
        # Convert to cm²
        total_area_cm2 = total_area_m2 * 10000.0
        
        return total_area_cm2
    
    def _calculate_segment_dimensions(self, segment_mask: np.ndarray, depth_map: np.ndarray, 
                                     bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Calculate width and length of a dent segment in cm using camera intrinsics.
        
        Uses the bounding box dimensions and converts to real-world measurements
        based on the average depth of the segment.
        
        Args:
            segment_mask: Binary mask (H, W) where True/255 = segment pixels
            depth_map: Depth map (H, W) in meters
            bbox: Bounding box as (x, y, width_pixels, height_pixels)
            
        Returns:
            Tuple of (width_cm, length_cm) in centimeters
        """
        h, w = segment_mask.shape
        x, y, width_px, height_px = bbox
        
        segment_pixels = (segment_mask > 0)
        
        if not np.any(segment_pixels) or width_px == 0 or height_px == 0:
            return 0.0, 0.0
        
        # Get depths for segment pixels
        segment_depths = depth_map[segment_pixels]
        valid_depths = segment_depths[segment_depths > 0]
        
        if len(valid_depths) == 0:
            return 0.0, 0.0
        
        # Calculate average depth for segment region
        avg_depth = np.mean(valid_depths)
        
        # Calculate pixel dimensions in meters at average depth using camera intrinsics
        # Using pre-computed FOV (square images: fov_x = fov_y)
        pixel_width_m = 2 * avg_depth * np.tan(self.fov_x_rad / 2.0) / w
        pixel_height_m = 2 * avg_depth * np.tan(self.fov_x_rad / 2.0) / h
        
        # Calculate real-world dimensions from bounding box
        # Note: width_px and height_px are in image coordinates
        # We need to determine which is actually width vs length based on orientation
        # For simplicity, use the larger dimension as length and smaller as width
        width_m = width_px * pixel_width_m
        height_m = height_px * pixel_height_m
        
        # Determine width (smaller) and length (larger)
        if width_m >= height_m:
            length_m = width_m
            width_m = height_m
        else:
            length_m = height_m
        
        # Convert to cm
        width_cm = width_m * 100.0
        length_cm = length_m * 100.0
        
        return width_cm, length_cm
    
    def _analyze_dent_segments(self, dent_mask: np.ndarray, dented_depth_map: np.ndarray, 
                               depth_diff_median: np.ndarray, direction_map: np.ndarray,
                               original_depth_map: Optional[np.ndarray] = None,
                               depth_diff_original_vs_dented: Optional[np.ndarray] = None,
                               median_panel_depth: Optional[float] = None,
                               panel_mask: Optional[np.ndarray] = None,
                               dilation_radius: int = 30,
                               shot_name: Optional[str] = None) -> list:
        """
        Analyze dent segments (connected components) and calculate properties for each segment.
        
        Calculates three types of depth measurements for each segment:
        1. Original vs Dented depth difference
        2. Depth difference vs Median Panel Depth (raw and filtered)
        3. Depth difference vs Local Corrugation Median
        
        Args:
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            dented_depth_map: Depth map (H, W) in meters (dented depth)
            depth_diff_median: Depth difference map (H, W) in meters (relative to median panel)
            direction_map: Direction map (H, W) where 1.0 = inward, -1.0 = outward, 0.0 = no dent
            original_depth_map: Optional original depth map for original vs dented comparison
            depth_diff_original_vs_dented: Optional depth difference map (original vs dented)
            median_panel_depth: Optional median panel depth value
            panel_mask: Optional panel mask (H, W) where True/1.0 = panel region
            dilation_radius: Radius in pixels for local corrugation calculation (default: 30)
            shot_name: Optional shot name (e.g., "internal_back_wall") - used to determine depth calculation method
            
        Returns:
            List of dictionaries containing segment information with all depth measurements
        """
        h, w = dent_mask.shape
        
        # Calculate global median depth of entire wall (for wave classification)
        global_median_depth = self._calculate_global_median_depth(
            dented_depth_map, dent_mask, panel_mask
        )
        logger.debug(f"Global median depth: {global_median_depth:.4f}m ({global_median_depth*1000:.2f}mm)")
        
        # Find connected components (segments)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dent_mask, connectivity=8
        )
        
        segments = []
        
        for label_id in range(1, num_labels):  # Skip label 0 (background)
            # Create mask for this segment
            segment_mask = (labels == label_id)
            
            # Get segment statistics
            x, y, width, height, area_pixels = stats[label_id]
            centroid_x, centroid_y = centroids[label_id]
            
            # Calculate area in cm² using camera intrinsics
            area_cm2 = self._calculate_segment_area(segment_mask.astype(np.uint8) * 255, dented_depth_map)
            
            # Calculate width and length in cm using camera intrinsics
            width_cm, length_cm = self._calculate_segment_dimensions(
                segment_mask.astype(np.uint8) * 255, dented_depth_map, (x, y, width, height)
            )
            
            # Get depth values for this segment
            segment_depths = dented_depth_map[segment_mask]
            valid_depths = segment_depths[(segment_depths > 0) & np.isfinite(segment_depths)]
            avg_depth_m = np.mean(valid_depths) if len(valid_depths) > 0 else 0.0
            
            # ====================================================================
            # CALCULATE THREE TYPES OF DEPTH DIFFERENCES FOR THIS SEGMENT:
            # ====================================================================
            
            # 1. Original vs Dented depth difference (direct comparison)
            max_depth_diff_original_vs_dented_m = 0.0
            max_depth_diff_original_vs_dented_mm = 0.0
            if depth_diff_original_vs_dented is not None:
                segment_depth_diffs_original = depth_diff_original_vs_dented[segment_mask]
                valid_diffs_original = segment_depth_diffs_original[
                    np.isfinite(segment_depth_diffs_original) & (segment_depth_diffs_original > 0)
                ]
                if len(valid_diffs_original) > 0:
                    max_depth_diff_original_vs_dented_m = float(np.max(valid_diffs_original))
                    max_depth_diff_original_vs_dented_mm = max_depth_diff_original_vs_dented_m * 1000.0
            
            # 2. Depth difference vs Median Panel Depth (raw and filtered)
            segment_depth_diffs_median = depth_diff_median[segment_mask]
            valid_diffs_median = segment_depth_diffs_median[
                np.isfinite(segment_depth_diffs_median) & (segment_depth_diffs_median > 0)
            ]
            
            max_depth_diff_vs_median_panel_m_raw = 0.0
            max_depth_diff_vs_median_panel_mm_raw = 0.0
            max_depth_diff_vs_median_panel_m = 0.0
            max_depth_diff_vs_median_panel_mm = 0.0
            
            if len(valid_diffs_median) > 0:
                # Raw maximum
                max_depth_diff_vs_median_panel_m_raw = float(np.max(valid_diffs_median))
                max_depth_diff_vs_median_panel_mm_raw = max_depth_diff_vs_median_panel_m_raw * 1000.0
                
                # Filtered maximum (using percentile clipping to remove outliers)
                filtered_diffs = self._filter_dent_depths_outliers(valid_diffs_median, method='percentile', percentile=99.0)
                if len(filtered_diffs) > 0:
                    max_depth_diff_vs_median_panel_m = float(np.max(filtered_diffs))
                    max_depth_diff_vs_median_panel_mm = max_depth_diff_vs_median_panel_m * 1000.0
                    # If filtered exceeds 200mm, use raw value
                    if max_depth_diff_vs_median_panel_mm > 200.0:
                        max_depth_diff_vs_median_panel_m = max_depth_diff_vs_median_panel_m_raw
                        max_depth_diff_vs_median_panel_mm = max_depth_diff_vs_median_panel_mm_raw
                else:
                    max_depth_diff_vs_median_panel_m = max_depth_diff_vs_median_panel_m_raw
                    max_depth_diff_vs_median_panel_mm = max_depth_diff_vs_median_panel_mm_raw
            
            # 3. Depth difference vs Local Corrugation Median (dilation_radius pixels)
            max_depth_diff_vs_local_corrugation_m = 0.0
            max_depth_diff_vs_local_corrugation_mm = 0.0
            local_corrugation_median_m = 0.0
            
            if panel_mask is not None:
                # Calculate median depth of local corrugation wall around this segment
                local_corrugation_median = self._calculate_local_corrugation_median(
                    segment_mask, dented_depth_map, panel_mask, dent_mask, dilation_radius=dilation_radius
                )
                
                if local_corrugation_median > 0:
                    local_corrugation_median_m = local_corrugation_median
                    # Calculate depth difference relative to local corrugation median
                    depth_diffs_from_local = np.abs(valid_depths - local_corrugation_median)
                    if len(depth_diffs_from_local) > 0:
                        max_depth_diff_vs_local_corrugation_m = float(np.max(depth_diffs_from_local))
                        max_depth_diff_vs_local_corrugation_mm = max_depth_diff_vs_local_corrugation_m * 1000.0
            
            # Determine dent direction for this segment
            segment_directions = direction_map[segment_mask]
            inward_pixels = np.sum(segment_directions > 0.5)  # > 0.5 means inward
            outward_pixels = np.sum(segment_directions < -0.5)  # < -0.5 means outward
            total_direction_pixels = inward_pixels + outward_pixels
            
            if total_direction_pixels > 0:
                direction_ratio = inward_pixels / total_direction_pixels
                # Determine dominant direction (> 50% threshold)
                if direction_ratio > 0.5:
                    direction = 'inward'
                else:
                    direction = 'outward'
            else:
                # Fallback: use average direction value
                avg_direction = np.mean(segment_directions[segment_directions != 0])
                if avg_direction > 0:
                    direction = 'inward'
                    direction_ratio = 1.0
                elif avg_direction < 0:
                    direction = 'outward'
                    direction_ratio = 0.0
                else:
                    direction = 'unknown'
                    direction_ratio = 0.5
            
            # Calculate local plane-fitting depth using Geometric Continuity (Stripes Method)
            # Uses Directional RANSAC to extend healthy stripes across the dent
            # Robust to: Warping, Bowing, Tilting, and Corrugation shape
            # Ensure depth map is in meters (it should already be)
            max_depth_local_plane_m = self._calculate_max_dent_depth_stripes(
                dented_depth_map, segment_mask.astype(np.uint8)
            )
            max_depth_local_plane_mm = max_depth_local_plane_m * 1000.0
            
            # For compatibility, set local_plane_depth_m to segment average
            # (The stripes method doesn't return this, but we keep it for backward compatibility)
            local_plane_depth_m = avg_depth_m
            
            # Wave location is determined by the stripes method internally
            # Set to a generic value since the new method doesn't classify explicitly
            wave_location = "STRIPES_METHOD"
            wave_location_description = "Geometric Continuity (Stripes Method - Directional RANSAC)"
            depth_diff_from_global_m = local_plane_depth_m - global_median_depth
            depth_diff_from_global_mm = depth_diff_from_global_m * 1000.0
            
            # Build segment info dictionary with all depth measurements
            segment_info = {
                'segment_id': int(label_id),
                'area_cm2': float(area_cm2),
                'width_cm': float(width_cm),
                'length_cm': float(length_cm),
                'pixel_count': int(area_pixels),
                'centroid': [float(centroid_x), float(centroid_y)],
                'bbox': [int(x), int(y), int(width), int(height)],
                'avg_depth_m': float(avg_depth_m),
                'direction': direction,
                'direction_ratio': float(direction_ratio),
                # Depth measurements
                'max_depth_diff_original_vs_dented_m': float(max_depth_diff_original_vs_dented_m),
                'max_depth_diff_original_vs_dented_mm': float(max_depth_diff_original_vs_dented_mm),
                'max_depth_diff_vs_median_panel_m_raw': float(max_depth_diff_vs_median_panel_m_raw),
                'max_depth_diff_vs_median_panel_mm_raw': float(max_depth_diff_vs_median_panel_mm_raw),
                'max_depth_diff_vs_median_panel_m': float(max_depth_diff_vs_median_panel_m),
                'max_depth_diff_vs_median_panel_mm': float(max_depth_diff_vs_median_panel_mm),
                'max_depth_diff_vs_local_corrugation_m': float(max_depth_diff_vs_local_corrugation_m),
                'max_depth_diff_vs_local_corrugation_mm': float(max_depth_diff_vs_local_corrugation_mm),
                'local_corrugation_median_m': float(local_corrugation_median_m),
                'local_corrugation_dilation_radius_pixels': dilation_radius,
                # Local plane-fitting depth (dominant surface method)
                'max_depth_local_plane_m': float(max_depth_local_plane_m),
                'max_depth_local_plane_mm': float(max_depth_local_plane_mm),
                'local_plane_depth_m': float(local_plane_depth_m),
                'local_plane_depth_mm': float(local_plane_depth_m * 1000.0),
                # Wave classification (Single Rail vs Multi-Rail)
                'wave_location': wave_location,
                'wave_location_description': wave_location_description,
                'global_median_depth_m': float(global_median_depth),
                'global_median_depth_mm': float(global_median_depth * 1000.0),
                'depth_diff_from_global_m': float(depth_diff_from_global_m),
                'depth_diff_from_global_mm': float(depth_diff_from_global_mm),
                # Legacy fields for backward compatibility
                # Updated to use local plane-fitting depth (rim-based classification method)
                'max_depth_diff_m': float(max_depth_local_plane_m),
                'max_depth_diff_mm': float(max_depth_local_plane_mm)
            }
            
            segments.append(segment_info)
        
        return segments
    
    def _calculate_global_median_depth(self, depth_map: np.ndarray, dent_mask: np.ndarray,
                                      panel_mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate global median depth of the entire wall (excluding dent pixels).
        This represents the "middle line" of the corrugation pattern.
        
        Args:
            depth_map: Depth map (H, W) in meters
            dent_mask: Binary mask (H, W) where True/255 = dent pixels
            panel_mask: Optional panel mask (H, W) where True/1.0 = panel region
            
        Returns:
            Global median depth in meters
        """
        # Find all wall pixels (non-dent pixels with valid depth)
        if panel_mask is not None:
            wall_pixels_mask = (depth_map > 0) & (dent_mask == 0) & (panel_mask > 0) & np.isfinite(depth_map)
        else:
            wall_pixels_mask = (depth_map > 0) & (dent_mask == 0) & np.isfinite(depth_map)
        
        wall_pixels = depth_map[wall_pixels_mask]
        
        if len(wall_pixels) == 0:
            return 0.0
        
        return float(np.median(wall_pixels))
    
    def _detect_corrugation_orientation(self, depth_map: np.ndarray) -> str:
        """
        --- A. Helper: Detect Wall Orientation ---
        Determines if stripes run Vertical (Side Walls) or Horizontal (Roof).
        
        Logic: Uses Sobel gradients to find direction of highest change.
        - Vertical Walls: Depth changes rapidly Left-to-Right (High dX), constant Up-Down (Low dY)
        - Horizontal Roofs: Depth changes rapidly Up-Down (High dY), constant Left-Right (Low dX)
        
        Args:
            depth_map: Depth map (H, W) in meters
            
        Returns:
            "VERTICAL" for side walls, "HORIZONTAL" for roofs, "UNKNOWN" for flat/complex
        """
        # Sobel gradients to find direction of highest change
        dx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
        
        # If Horizontal variance (dx) is higher, waves go Up-Down (Vertical)
        if np.var(dx) > np.var(dy) * 1.2:
            return "VERTICAL"
        # If Vertical variance (dy) is higher, waves go Left-Right (Horizontal)
        elif np.var(dy) > np.var(dx) * 1.2:
            return "HORIZONTAL"
        
        return "UNKNOWN"
    
    def _calculate_max_dent_depth_stripes(self, depth_map: np.ndarray, mask_binary: np.ndarray) -> float:
        """
        --- B. Helper: Directional RANSAC (The "Stripes" Logic) ---
        Measures depth by extending the "healthy stripes" across the dent.
        Robust to: Warping, Bowing, Tilting, and Corrugation shape.
        
        This implements the Geometric Continuity approach (Stripes Method) which mimics
        IICL inspection standards by following container rails directionally.
        
        Args:
            depth_map: Depth map (H, W) in meters
            mask_binary: Binary mask (H, W) where True/1.0 = dent pixels
            
        Returns:
            Maximum dent depth in meters (95th percentile)
        """
        if np.sum(mask_binary) == 0:
            return 0.0
        
        # 1. Orientation
        orientation = self._detect_corrugation_orientation(depth_map)
        
        # 2. Process Each Dent Individually
        num_labels, labels = cv2.connectedComponents((mask_binary > 0).astype(np.uint8))
        max_severity_total = 0.0
        H, W = depth_map.shape
        
        # Grid for RANSAC
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        for i in range(1, num_labels):
            dent_mask = (labels == i)
            ys, xs = np.where(dent_mask)
            if len(ys) == 0:
                continue
                
            y_min, y_max = np.min(ys), np.max(ys)
            x_min, x_max = np.min(xs), np.max(xs)
            
            # --- 3. EXTRACT THE STRIPE (Scanning) ---
            # Look "Up/Down" or "Left/Right" to find the original rail.
            scan_margin = 150  # Look far to find healthy metal
            
            if orientation == "VERTICAL":  # Side Walls
                y_scan_min = max(0, y_min - scan_margin)
                y_scan_max = min(H, y_max + scan_margin)
                # CRITICAL: Keep width strict (don't look left/right at other rails)
                x_scan_min, x_scan_max = x_min, x_max
                
            elif orientation == "HORIZONTAL":  # Roofs
                y_scan_min, y_scan_max = y_min, y_max
                # CRITICAL: Keep height strict
                x_scan_min = max(0, x_min - scan_margin)
                x_scan_max = min(W, x_max + scan_margin)
                
            else:  # Unknown (Box Search fallback)
                y_scan_min = max(0, y_min - 50)
                y_scan_max = min(H, y_max + 50)
                x_scan_min = max(0, x_min - 50)
                x_scan_max = min(W, x_max + 50)

            strip_depth = depth_map[y_scan_min:y_scan_max, x_scan_min:x_scan_max]
            strip_mask = mask_binary[y_scan_min:y_scan_max, x_scan_min:x_scan_max]
            
            # Identify Healthy Neighbors in this stripe
            neighbor_mask = (strip_depth > 0) & (strip_mask == 0) & np.isfinite(strip_depth)
            if np.sum(neighbor_mask) < 50:
                continue

            # --- 4. SURFACE LOGIC (The Bowing Fix) ---
            neighbor_depths = strip_depth[neighbor_mask]
            
            # Coordinate Grids for this strip
            h_c, w_c = strip_depth.shape
            yy_c, xx_c = np.meshgrid(np.arange(h_c), np.arange(w_c), indexing='ij')
            X_candidates = np.stack([xx_c[neighbor_mask], yy_c[neighbor_mask]], axis=1)
            y_candidates = neighbor_depths

            # Check Variance: Does this stripe contain both Hills and Valleys?
            # Corrugation depth is ~36mm. If range > 15mm, we likely have mixed surfaces.
            depth_range = np.percentile(neighbor_depths, 95) - np.percentile(neighbor_depths, 5)
            
            if depth_range > 0.015:
                # Case: Multi-Wave Stripe.
                # ACTION: Filter for "Peaks" (Closer points) to bridge the gap.
                # This works even if wall is bowed, because it's relative to local strip.
                threshold = np.percentile(neighbor_depths, 50)
                is_peak = neighbor_depths <= threshold
                
                X_train = X_candidates[is_peak]
                y_train = y_candidates[is_peak]
            else:
                # Case: Single Rail (Pure Hill or Pure Valley).
                # ACTION: Use everything.
                X_train = X_candidates
                y_train = y_candidates

            # --- 5. CREATE GHOST STRIPES (RANSAC Fit) ---
            try:
                reg = RANSACRegressor(random_state=42, residual_threshold=0.005)
                reg.fit(X_train, y_train)
                
                # --- 6. MEASURE PERPENDICULAR DROP ---
                dent_pixels_mask = (strip_mask > 0)
                X_dent = np.stack([xx_c[dent_pixels_mask], yy_c[dent_pixels_mask]], axis=1)
                actual_depths = strip_depth[dent_pixels_mask]
                
                # Predict "Should Be" Depth
                ideal_depths = reg.predict(X_dent)
                
                # Calculate Difference
                diffs = np.abs(actual_depths - ideal_depths)
                valid_diffs = diffs[np.isfinite(diffs)]
                
                if len(valid_diffs) == 0:
                    continue
                
                # Robust Max (95th Percentile)
                sev = np.percentile(valid_diffs, 95)
                if sev > max_severity_total:
                    max_severity_total = sev
            except Exception as e:
                logger.debug(f"RANSAC fitting failed for dent segment {i}: {e}")
                continue

        return float(max_severity_total)
    
    def _calculate_max_depth_diff_m(self, shot_name: Optional[str],
                                    max_depth_diff_original_vs_dented_m: float,
                                    max_depth_diff_vs_median_panel_m: float,
                                    max_depth_diff_vs_local_corrugation_m: float) -> float:
        """
        Calculate max_depth_diff_m based on shot_name.
        For back panels, use the minimum of the three depth measurements.
        For all other shots (side walls, roof, doors, etc.), use the smaller of:
        max_depth_diff_original_vs_dented_m and max_depth_diff_vs_median_panel_m.
        
        Args:
            shot_name: Name of the shot (e.g., "internal_back_wall")
            max_depth_diff_original_vs_dented_m: Depth difference vs original dented (meters)
            max_depth_diff_vs_median_panel_m: Depth difference vs median panel (meters)
            max_depth_diff_vs_local_corrugation_m: Depth difference vs local corrugation (meters)
            
        Returns:
            Maximum depth difference in meters
        """
        # Check if this is a back panel shot
        if shot_name and 'back' in shot_name.lower():
            # For back panels, use the minimum of the three values
            values = [
                max_depth_diff_original_vs_dented_m,
                max_depth_diff_vs_median_panel_m,
                max_depth_diff_vs_local_corrugation_m
            ]
            # Filter out zero values (which indicate measurement wasn't available)
            valid_values = [v for v in values if v > 0]
            if valid_values:
                return float(min(valid_values))
            # Fallback to median panel if no valid values
            return float(max_depth_diff_vs_median_panel_m)
        else:
            # For non-back panels, use the smaller of original_vs_dented and vs_median_panel
            values = [
                max_depth_diff_original_vs_dented_m,
                max_depth_diff_vs_median_panel_m
            ]
            # Filter out zero values (which indicate measurement wasn't available)
            valid_values = [v for v in values if v > 0]
            if valid_values:
                return float(min(valid_values))
            # Fallback to median panel if no valid values
            return float(max_depth_diff_vs_median_panel_m)
    
    def _calculate_local_corrugation_median(self, segment_mask: np.ndarray, depth_map: np.ndarray,
                                            panel_mask: Optional[np.ndarray], dent_mask: np.ndarray,
                                            dilation_radius: int = 30) -> float:
        """
        Calculate median depth of the corrugation wall region around a dent segment.
        
        Args:
            segment_mask: Binary mask (H, W) for a single dent segment (True = segment pixels)
            depth_map: Depth map (H, W) in meters
            panel_mask: Panel mask (H, W) where True/1.0 = panel region
            dent_mask: Full dent mask (H, W) to exclude all dent pixels
            dilation_radius: Radius in pixels to search around segment for corrugation region
            
        Returns:
            Median depth of local corrugation wall in meters
        """
        h, w = segment_mask.shape
        
        # Create dilated region around segment to find nearby corrugation wall
        segment_uint8 = segment_mask.astype(np.uint8) * 255
        kernel_size = dilation_radius * 2 + 1
        dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_region = cv2.dilate(segment_uint8, dilation_kernel, iterations=1)
        
        # Find panel pixels in dilated region (excluding dent pixels)
        if panel_mask is not None:
            # Panel region in dilated area, excluding dent pixels
            local_panel_region = (dilated_region > 0) & (panel_mask > 0) & (dent_mask == 0)
        else:
            # Use entire dilated area excluding dent pixels
            local_panel_region = (dilated_region > 0) & (dent_mask == 0)
        
        # Get valid depths from local corrugation wall region
        local_panel_depths = depth_map[local_panel_region & (depth_map > 0) & np.isfinite(depth_map)]
        
        if len(local_panel_depths) == 0:
            # Fallback: use global panel median if no local region found
            if panel_mask is not None:
                panel_region = (panel_mask > 0) & (dent_mask == 0)
            else:
                panel_region = (dent_mask == 0)
            fallback_depths = depth_map[panel_region & (depth_map > 0) & np.isfinite(depth_map)]
            if len(fallback_depths) > 0:
                return float(np.median(fallback_depths))
            return 0.0
        
        return float(np.median(local_panel_depths))
    
    def _filter_segments_by_depth(self, dent_mask: np.ndarray, depth_map: np.ndarray,
                                  depth_diff: np.ndarray, direction_map: np.ndarray,
                                  panel_mask: Optional[np.ndarray] = None,
                                  min_depth_mm: float = 10.0,
                                  shot_name: Optional[str] = None) -> Tuple[np.ndarray, list]:
        """
        Filter dent segments by minimum depth threshold.
        Compares each segment to the median depth of its local corrugation wall (not entire panel).
        
        Args:
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            depth_map: Depth map (H, W) in meters (dented depth)
            depth_diff: Depth difference map (H, W) in meters (relative to median panel depth)
            direction_map: Direction map (H, W) where 1.0 = inward, -1.0 = outward, 0.0 = no dent
            panel_mask: Panel mask (H, W) where True/1.0 = panel region
            min_depth_mm: Minimum depth threshold in mm (default: 10.0 mm)
            
        Returns:
            Tuple of (filtered_mask, filtered_segments_info):
            - filtered_mask: Binary mask with only segments above depth threshold
            - filtered_segments_info: List of segment information for kept segments
        """
        # Find connected components (segments)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dent_mask, connectivity=8
        )
        
        filtered_segments = []
        
        # Process each segment individually
        for label_id in range(1, num_labels):  # Skip label 0 (background)
            # Create mask for this segment
            segment_mask = (labels == label_id)
            
            # Calculate median depth of local corrugation wall around this segment
            local_corrugation_median = self._calculate_local_corrugation_median(
                segment_mask, depth_map, panel_mask, dent_mask, dilation_radius=30
            )
            
            if local_corrugation_median <= 0:
                # Skip if no valid corrugation region found
                continue
            
            # Get depth values for this segment
            segment_depths = depth_map[segment_mask & (depth_map > 0) & np.isfinite(depth_map)]
            
            if len(segment_depths) == 0:
                continue
            
            # Calculate depth difference relative to local corrugation median
            # Use maximum depth difference in the segment
            depth_diffs_from_local = np.abs(segment_depths - local_corrugation_median)
            max_depth_diff_m = np.max(depth_diffs_from_local)
            max_depth_diff_mm = max_depth_diff_m * 1000.0
            
            # Filter: keep only segments with depth >= min_depth_mm relative to local corrugation
            if max_depth_diff_mm >= min_depth_mm:
                # Get segment info using existing analysis function
                # Create single-segment mask for analysis
                single_segment_mask = (segment_mask.astype(np.uint8) * 255)
                segment_info_list = self._analyze_dent_segments(
                    single_segment_mask, depth_map, depth_diff, direction_map,
                    shot_name=shot_name
                )
                if len(segment_info_list) > 0:
                    # Get segment info (should be only one segment)
                    segment_info = segment_info_list[0].copy()
                    # Update segment_id to match original labels
                    segment_info['segment_id'] = label_id
                    # Update with local corrugation-based depth measurement
                    segment_info['max_depth_diff_m'] = max_depth_diff_m
                    segment_info['max_depth_diff_mm'] = max_depth_diff_mm
                    segment_info['local_corrugation_median_m'] = local_corrugation_median
                    filtered_segments.append(segment_info)
        
        # Create new mask with only filtered segments
        filtered_mask = np.zeros_like(dent_mask)
        
        if len(filtered_segments) > 0:
            # Keep only segments that passed the depth filter
            for seg in filtered_segments:
                segment_id = seg['segment_id']
                if segment_id < num_labels:
                    filtered_mask[labels == segment_id] = 255
        
        return filtered_mask, filtered_segments
    
    def _apply_morphology_operations(self, binary_mask: np.ndarray, valid_mask: np.ndarray,
                                     morphology_opening_size: int = 9,
                                     morphology_closing_size: int = 11,
                                     min_dent_area: int = 200) -> np.ndarray:
        """
        Apply morphology operations to a binary mask.
        
        Args:
            binary_mask: Input binary mask (uint8), with dent pixels = 255, background = 0
            valid_mask: Boolean mask indicating valid pixels
            morphology_opening_size: Kernel size for morphological opening
            morphology_closing_size: Kernel size for morphological closing
            min_dent_area: Minimum area for filtering thin components
            
        Returns:
            Binary mask after morphology operations
        """
        result_mask = binary_mask.copy()
        
        # Apply morphological opening (erosion followed by dilation) to remove noise
        if morphology_opening_size > 0:
            opening_size = morphology_opening_size
            if opening_size % 2 == 0:
                opening_size += 1
            opening_kernel = np.ones((opening_size, opening_size), np.uint8)
            result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, opening_kernel)
        
        # Apply morphological closing (dilation followed by erosion) to fill gaps
        if morphology_closing_size > 0:
            closing_size = morphology_closing_size
            if closing_size % 2 == 0:
                closing_size += 1
            closing_kernel = np.ones((closing_size, closing_size), np.uint8)
            result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, closing_kernel)
        
        # Post-morphology aggressive fill
        result_mask = self._aggressive_internal_fill(result_mask, valid_mask)
        
        # Fill black holes inside white dented regions
        result_mask = self._fill_holes(result_mask)
        
        # Remove small and thin false-positive regions
        result_mask = self._filter_thin_components(result_mask, min_area=min_dent_area)
        
        return result_mask
    
    def _filter_segments_by_area(self, dent_mask: np.ndarray, depth_map: np.ndarray,
                                 depth_diff: np.ndarray, direction_map: np.ndarray,
                                 min_area_cm2: float = 1.0,
                                 shot_name: Optional[str] = None) -> Tuple[np.ndarray, list]:
        """
        Filter dent segments by minimum area threshold.
        
        Args:
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            depth_map: Depth map (H, W) in meters
            depth_diff: Depth difference map (H, W) in meters
            direction_map: Direction map (H, W) where 1.0 = inward, -1.0 = outward, 0.0 = no dent
            min_area_cm2: Minimum area threshold in cm² (default: 1.0 cm²)
            
        Returns:
            Tuple of (filtered_mask, filtered_segments_info):
            - filtered_mask: Binary mask with only segments above threshold
            - filtered_segments_info: List of segment information for kept segments
        """
        # Analyze all segments
        all_segments = self._analyze_dent_segments(dent_mask, depth_map, depth_diff, direction_map,
                                                   shot_name=shot_name)
        
        # Filter segments by area
        filtered_segments = [seg for seg in all_segments if seg['area_cm2'] >= min_area_cm2]
        
        # Create new mask with only filtered segments
        filtered_mask = np.zeros_like(dent_mask)
        
        if len(filtered_segments) > 0:
            # Find connected components again
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                dent_mask, connectivity=8
            )
            
            # Keep only segments that passed the area filter
            for seg in filtered_segments:
                segment_id = seg['segment_id']
                if segment_id < num_labels:
                    filtered_mask[labels == segment_id] = 255
        
        return filtered_mask, filtered_segments
    
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
        
        # Use pre-computed camera intrinsics from renderer (matching pyrender PerspectiveCamera)
        # These intrinsics are derived from the same FOV used to create the camera in render_depth()
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
        
        # Calculate pixel dimensions in meters at average depth using camera intrinsics
        # Using pre-computed FOV (square images: fov_x = fov_y)
        pixel_width_m = 2 * avg_depth * np.tan(self.fov_x_rad / 2.0) / w
        pixel_height_m = 2 * avg_depth * np.tan(self.fov_x_rad / 2.0) / h
        pixel_area_m2 = pixel_width_m * pixel_height_m
        
        # Calculate total area
        num_dent_pixels = np.sum(dent_pixels)
        total_area_m2 = num_dent_pixels * pixel_area_m2
        
        # Convert to cm²
        total_area_cm2 = total_area_m2 * 10000.0
        
        return total_area_cm2
    
    def _extract_dent_depths(self, depth_diff: np.ndarray, dent_mask: np.ndarray, threshold: float = 0.035) -> np.ndarray:
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
    
    def _calculate_plane_depth_map(self, depth_map: np.ndarray, plane_coefficients: np.ndarray) -> np.ndarray:
        """
        Calculate plane depth map from RANSAC plane coefficients.
        
        Args:
            depth_map: Depth map (H, W) in meters (used for shape and valid mask)
            plane_coefficients: [a, b, -1, c] where z = ax + by + c in camera space
            
        Returns:
            Plane depth map (H, W) in meters
        """
        height, width = depth_map.shape
        a, b, _, c = plane_coefficients
        
        # Get camera intrinsics
        if height == self.image_size and width == self.image_size:
            focal_length = self.focal_length
            cx, cy = self.cx, self.cy
        else:
            fov_y_rad = np.deg2rad(self.camera_fov)
            focal_length = (height / 2.0) / np.tan(fov_y_rad / 2.0)
            cx, cy = width / 2.0, height / 2.0
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to normalized camera coordinates
        x_norm = (u - cx) / focal_length
        y_norm = (v - cy) / focal_length
        
        # Calculate plane depth: z_plane = c / (1 - a*x_norm - b*y_norm)
        # This solves z = a*x_norm*z + b*y_norm*z + c for z
        denominator = 1.0 - a * x_norm - b * y_norm
        # Avoid division by zero and handle near-zero denominators
        denominator = np.where(np.abs(denominator) < 1e-6, np.sign(denominator) * 1e-6, denominator)
        plane_depth = c / denominator
        
        # Clamp plane depth to reasonable values (within depth map range)
        valid_mask = (depth_map > 0) & np.isfinite(depth_map)
        if np.any(valid_mask):
            min_depth = np.min(depth_map[valid_mask])
            max_depth = np.max(depth_map[valid_mask])
            # Allow some margin beyond the depth range
            margin = (max_depth - min_depth) * 0.1
            plane_depth = np.clip(plane_depth, min_depth - margin, max_depth + margin)
        
        # Set invalid depths to 0 (where original depth map is invalid)
        plane_depth[~valid_mask] = 0.0
        
        # Filter out unrealistic depths (negative or infinite)
        plane_depth[plane_depth <= 0] = 0.0
        plane_depth[~np.isfinite(plane_depth)] = 0.0
        
        return plane_depth
    
    def _calculate_median_panel_depth(self, depth_map: np.ndarray, dent_mask: np.ndarray, 
                                     panel_mask: Optional[np.ndarray] = None,
                                     plane_coefficients: Optional[np.ndarray] = None) -> float:
        """
        Calculate median depth of the panel region, excluding dent areas.
        
        If plane_coefficients are provided, uses RANSAC-fitted plane depth instead of raw depth.
        This eliminates perspective distortion effects.
        
        Args:
            depth_map: Depth map (H, W) in meters
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            panel_mask: Optional panel mask (H, W) where True/1.0 = panel region.
                       If None, uses entire image.
            plane_coefficients: Optional RANSAC plane coefficients [a, b, -1, c]. If provided,
                              uses plane depth instead of raw depth.
            
        Returns:
            Median panel depth in meters
        """
        # If plane coefficients are provided, use plane depth instead of raw depth
        if plane_coefficients is not None:
            plane_depth_map = self._calculate_plane_depth_map(depth_map, plane_coefficients)
            depth_to_use = plane_depth_map
            logger.info("    Using RANSAC-fitted plane depth for median calculation (eliminates perspective distortion)")
        else:
            depth_to_use = depth_map
            logger.info("    Using raw depth for median calculation (no plane coefficients provided)")
        
        # Combine masks: panel region excluding dent areas
        if panel_mask is not None:
            # Panel region (excluding dents)
            panel_region = (panel_mask > 0) & (dent_mask == 0)
        else:
            # Use entire image excluding dents
            panel_region = (dent_mask == 0)
        
        # Get valid panel depths (non-zero, finite, excluding dents)
        panel_depths = depth_to_use[panel_region & (depth_to_use > 0) & np.isfinite(depth_to_use)]
        
        if len(panel_depths) == 0:
            # Fallback: use median of all valid depths if no panel region found
            valid_depths = depth_to_use[(depth_to_use > 0) & np.isfinite(depth_to_use) & (dent_mask == 0)]
            if len(valid_depths) == 0:
                logger.warning("No valid panel depths found for median calculation, using mean of all depths")
                valid_depths = depth_to_use[(depth_to_use > 0) & np.isfinite(depth_to_use)]
                if len(valid_depths) == 0:
                    return 0.0
            return float(np.median(valid_depths))
        
        return float(np.median(panel_depths))
    
    def _calculate_depth_diff_from_median(self, depth_map: np.ndarray, dent_mask: np.ndarray,
                                          panel_mask: Optional[np.ndarray] = None,
                                          plane_coefficients: Optional[np.ndarray] = None,
                                          original_depth_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate depth difference map relative to median panel depth.
        
        Args:
            depth_map: Depth map (H, W) in meters (dented depth)
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            panel_mask: Optional panel mask (H, W) where True/1.0 = panel region
            plane_coefficients: Optional RANSAC plane coefficients [a, b, -1, c]. If provided,
                              uses plane depth for median calculation.
            original_depth_map: Optional original depth map (H, W) in meters. If provided,
                              calculates median from original panel instead of dented panel.
                              This gives accurate dent depth relative to original panel median.
            
        Returns:
            Depth difference map (H, W) in meters, where values represent
            deviation from median panel depth
        """
        # Calculate median from original depth map if provided, otherwise from dented depth map
        if original_depth_map is not None:
            # Use original depth map to calculate median (excludes any dents)
            # Use empty mask for original since we want to exclude dents from median calculation
            empty_mask = np.zeros_like(dent_mask, dtype=np.uint8)
            median_depth = self._calculate_median_panel_depth(original_depth_map, empty_mask, panel_mask, plane_coefficients)
        else:
            # Fallback to original behavior: calculate median from dented depth map
            median_depth = self._calculate_median_panel_depth(depth_map, dent_mask, panel_mask, plane_coefficients)
        
        # Calculate absolute depth difference from median (using dented depth map)
        depth_diff = np.abs(depth_map - median_depth)
        
        # Set invalid depths to 0
        valid_mask = (depth_map > 0) & np.isfinite(depth_map)
        depth_diff[~valid_mask] = 0.0
        
        return depth_diff
    
    def _calculate_dent_depth(self, depth_diff: np.ndarray, dent_mask: np.ndarray) -> float:
        """
        Calculate maximum dent depth in mm relative to median panel depth.
        
        Args:
            depth_diff: Depth difference map (H, W) in meters (relative to median panel depth)
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
    parser.add_argument('--threshold', type=float, default=0.035,
                       help='Depth difference threshold in meters (default: 0.035 = 35mm)')
    parser.add_argument('--min-area-cm2', type=float, default=1.0,
                       help='Minimum dent area threshold in cm² (default: 1.0 cm²)')
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
                    threshold=args.threshold,
                    min_area_cm2=args.min_area_cm2
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
                threshold=args.threshold,
                min_area_cm2=args.min_area_cm2
            )
    
    finally:
        renderer.cleanup()
        logger.info("✓ Renderer cleanup complete")


if __name__ == "__main__":
    main()

