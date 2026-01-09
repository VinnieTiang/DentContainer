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
                      min_dent_area: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        # Positive = dented_depth > original_depth (inward dent, surface pushed away from camera)
        # Negative = dented_depth < original_depth (outward dent, surface pushed toward camera)
        signed_depth_diff = dented_depth - original_depth
        
        # Calculate absolute depth difference between original and dented containers
        # This detects any change in depth (dents push surface inward/outward)
        depth_diff = np.abs(signed_depth_diff)
        
        # Identify valid pixels (both depths are valid and finite)
        valid_mask = (original_depth > 0) & (dented_depth > 0) & np.isfinite(original_depth) & np.isfinite(dented_depth)
        
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
        direction_map = np.zeros_like(signed_depth_diff, dtype=np.float32)
        direction_map[valid_mask & (depth_diff > threshold)] = np.sign(signed_depth_diff[valid_mask & (depth_diff > threshold)])
        
        # Save pure binary mask before morphology operations (after gap-filling, before opening/closing)
        pure_binary_mask = binary_mask.copy()
        
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
                                   max_trials: int = 1000) -> np.ndarray:
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
            Binary mask (H, W) where True/1.0 = panel pixels (inliers), False/0.0 = other pixels (outliers)
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
            return np.zeros_like(depth, dtype=np.float32)
        
        # Apply RANSAC plane fitting
        inlier_mask_points, _ = self._fit_plane_ransac(
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
        
        return panel_mask
    
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
                
                # Save raw dented depth map to dataset folder ONLY for testset generation
                if is_testset:
                    if dataset_dir is None:
                        dataset_dir = Path("output_scene_dataset")
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    
                    dataset_dented_depth_raw_path = dataset_dir / f"{base_name}_{shot_name}_dented_depth_raw.npy"
                    np.save(dataset_dented_depth_raw_path, dented_depth.astype(np.float32))
                    logger.info(f"    ✓ Saved raw dented depth map (before RANSAC): {base_name}_{shot_name}_dented_depth_raw.npy")
                
                # Apply Gaussian smoothing to flatten corrugation pattern before RANSAC
                # Automatically tune sigma based on depth variance
                logger.info(f"    Applying Gaussian smoothing with adaptive sigma to flatten corrugation pattern...")
                original_depth_smoothed, sigma_used = self._apply_gaussian_smoothing(original_depth, adaptive=True)
                logger.info(f"    Using sigma={sigma_used:.2f} for Gaussian smoothing")
                
                # Apply RANSAC plane fitting to detect main panel surface (using smoothed original depth map)
                # Use adaptive residual threshold to include all corrugations (upper and lower)
                logger.info(f"    Applying RANSAC plane fitting to detect main panel surface...")
                panel_mask = self._generate_panel_mask_ransac(
                    original_depth_smoothed,
                    adaptive_threshold=True,  # Automatically tune threshold based on corrugation depth
                    max_trials=1000
                )
                
                # Apply panel mask to both original (unsmoothed) depth maps for accurate dent measurement
                original_depth_masked = self._apply_mask_to_depth(original_depth, panel_mask)
                dented_depth_masked = self._apply_mask_to_depth(dented_depth, panel_mask)
                
                # Compare depths using masked depth maps to detect dent locations
                # This initial comparison identifies where dents are
                depth_diff_initial, dent_mask, pure_dent_mask, direction_map = self.compare_depths(original_depth_masked, dented_depth_masked, threshold)
                
                # Filter dent segments by minimum area threshold
                logger.info(f"    Filtering dent segments by minimum area threshold: {min_area_cm2} cm²")
                filtered_dent_mask, dent_segments = self._filter_segments_by_area(
                    dent_mask, dented_depth_masked, depth_diff_initial, direction_map, min_area_cm2
                )
                
                # Log filtering results
                num_segments_before = len(self._analyze_dent_segments(dent_mask, dented_depth_masked, depth_diff_initial, direction_map))
                num_segments_after = len(dent_segments)
                logger.info(f"    Segments before filtering: {num_segments_before}, after filtering: {num_segments_after}")
                
                # Use filtered mask for all subsequent operations
                dent_mask = filtered_dent_mask
                
                # Recalculate depth difference relative to median panel depth
                # This provides more accurate depth measurement by comparing to the "normal" panel surface
                logger.info(f"    Calculating depth relative to median panel depth...")
                median_panel_depth = self._calculate_median_panel_depth(dented_depth_masked, dent_mask, panel_mask)
                logger.info(f"    Median panel depth: {median_panel_depth:.4f}m ({median_panel_depth*1000:.2f}mm)")
                depth_diff = self._calculate_depth_diff_from_median(dented_depth_masked, dent_mask, panel_mask)
                
                # Re-analyze segments with median-based depth_diff for accurate depth measurements
                dent_segments = self._analyze_dent_segments(dent_mask, dented_depth_masked, depth_diff, direction_map)
                # Filter segments again by area threshold (using median-based depth_diff)
                filtered_segments = [seg for seg in dent_segments if seg['area_cm2'] >= min_area_cm2]
                dent_segments = filtered_segments
                
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
                if dataset_dir is None:
                    dataset_dir = Path("output_scene_dataset")
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                # Save files directly in dataset_dir with full name including shot_name
                dataset_dented_depth_path = dataset_dir / f"{base_name}_{shot_name}_dented_depth.npy"
                np.save(dataset_dented_depth_path, dented_depth_masked.astype(np.float32))
                
                # Save panel mask from RANSAC
                panel_mask_path = shot_output_dir / f"{base_name}_panel_mask.npy"
                np.save(panel_mask_path, panel_mask.astype(np.float32))  # Save as float (0 or 1)
                # Convert to uint8 (0 or 255) for PNG visualization
                panel_mask_uint8 = (panel_mask * 255).astype(np.uint8)
                imageio.imwrite(shot_output_dir / f"{base_name}_panel_mask.png", panel_mask_uint8)
                
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
                
                # Save binary mask: WHITE (255) = dented areas (different depth), BLACK (0) = normal areas (same depth)
                imageio.imwrite(shot_output_dir / f"{base_name}_dent_mask.png", dent_mask)
                
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
                
                # Always save dent mask and masked depth for training (both regular and testset)
                dataset_dent_mask_path = dataset_dir / f"{base_name}_{shot_name}_dent_mask.png"
                imageio.imwrite(dataset_dent_mask_path, dent_mask)
                
                # Save RGB image to dataset folder ONLY for testset generation
                if is_testset or save_rgb_to_dataset:
                    dataset_rgb_path = dataset_dir / f"{base_name}_{shot_name}_rgb.png"
                    imageio.imwrite(dataset_rgb_path, dented_rgb)
                    logger.info(f"    ✓ Saved RGB image to dataset: {base_name}_{shot_name}_rgb.png")
                
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
                # If it does, use the raw detected depth instead of the filtered value
                if filtered_max_diff_mm > 200.0:
                    filtered_max_diff_m = raw_max_diff_m
                    filtered_max_diff_mm = raw_max_diff_mm
                    logger.info(f"    ✓ Dent mask: {dent_pixels} pixels ({dent_percentage:.1f}%)")
                    logger.info(f"    ✓ Max depth (vs median panel): {raw_max_diff_mm:.2f}mm")
                    logger.info(f"    ✓ Filtered max depth: {filtered_max_diff_mm:.2f}mm (using raw value, filtered exceeded 200mm)")
                else:
                    logger.info(f"    ✓ Dent mask: {dent_pixels} pixels ({dent_percentage:.1f}%)")
                    logger.info(f"    ✓ Max depth (vs median panel): {raw_max_diff_mm:.2f}mm")
                    logger.info(f"    ✓ Filtered max depth: {filtered_max_diff_mm:.2f}mm")
                
                # Calculate camera distance from panel
                # For regular shots: distance from camera to panel
                # For corner shots: distance from panel towards corner pole
                camera_distance_m = self._calculate_camera_distance(pose)
                
                # Calculate dent area in cm² from mask pixels
                # Count pixels where mask = 1 (dented pixels) and convert to real-world area
                # Use masked depth map for area calculation
                dent_area_cm2 = self._calculate_dent_area(dent_mask, dented_depth_masked)
                
                results.append({
                    'shot_name': shot_name,
                    'dent_pixels': int(dent_pixels),
                    'total_pixels': int(total_pixels),
                    'dent_percentage': float(dent_percentage),
                    # Depth measurement reference (median panel depth)
                    'median_panel_depth_m': float(median_panel_depth),
                    'median_panel_depth_mm': float(median_panel_depth * 1000),
                    # Raw statistics (before filtering) - depth relative to median panel
                    'max_depth_diff_m_raw': float(raw_max_diff_m),
                    'max_depth_diff_mm_raw': float(raw_max_diff_mm),
                    # Filtered statistics (after outlier removal, uses raw value if filtered exceeds 200mm)
                    'max_depth_diff_m': float(filtered_max_diff_m),
                    'max_depth_diff_mm': float(filtered_max_diff_mm),
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
    
    def _analyze_dent_segments(self, dent_mask: np.ndarray, depth_map: np.ndarray, 
                               depth_diff: np.ndarray, direction_map: np.ndarray) -> list:
        """
        Analyze dent segments (connected components) and calculate properties for each.
        
        Args:
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            depth_map: Depth map (H, W) in meters (dented depth)
            depth_diff: Depth difference map (H, W) in meters
            direction_map: Direction map (H, W) where 1.0 = inward, -1.0 = outward, 0.0 = no dent
            
        Returns:
            List of dictionaries containing segment information:
            - segment_id: Unique ID for the segment
            - area_cm2: Area in cm²
            - pixel_count: Number of pixels in segment
            - centroid: (x, y) centroid in pixel coordinates
            - bbox: Bounding box (x, y, width, height)
            - avg_depth_m: Average depth in meters
            - max_depth_diff_m: Maximum depth difference in meters
            - max_depth_diff_mm: Maximum depth difference in mm
            - direction: 'inward' or 'outward' (dominant direction in segment)
            - direction_ratio: Ratio of inward pixels (1.0 = all inward, 0.0 = all outward)
        """
        h, w = dent_mask.shape
        
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
            area_cm2 = self._calculate_segment_area(segment_mask.astype(np.uint8) * 255, depth_map)
            
            # Calculate width and length in cm using camera intrinsics
            width_cm, length_cm = self._calculate_segment_dimensions(
                segment_mask.astype(np.uint8) * 255, depth_map, (x, y, width, height)
            )
            
            # Get depth values for this segment
            segment_depths = depth_map[segment_mask]
            valid_depths = segment_depths[segment_depths > 0]
            avg_depth_m = np.mean(valid_depths) if len(valid_depths) > 0 else 0.0
            
            # Get depth difference values for this segment
            segment_depth_diffs = depth_diff[segment_mask]
            valid_diffs = segment_depth_diffs[np.isfinite(segment_depth_diffs) & (segment_depth_diffs > 0)]
            max_depth_diff_m = np.max(valid_diffs) if len(valid_diffs) > 0 else 0.0
            max_depth_diff_mm = max_depth_diff_m * 1000.0
            
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
            
            segment_info = {
                'segment_id': int(label_id),
                'area_cm2': float(area_cm2),
                'width_cm': float(width_cm),
                'length_cm': float(length_cm),
                'pixel_count': int(area_pixels),
                'centroid': [float(centroid_x), float(centroid_y)],
                'bbox': [int(x), int(y), int(width), int(height)],
                'avg_depth_m': float(avg_depth_m),
                'max_depth_diff_m': float(max_depth_diff_m),
                'max_depth_diff_mm': float(max_depth_diff_mm),
                'direction': direction,
                'direction_ratio': float(direction_ratio)
            }
            
            segments.append(segment_info)
        
        return segments
    
    def _filter_segments_by_area(self, dent_mask: np.ndarray, depth_map: np.ndarray,
                                 depth_diff: np.ndarray, direction_map: np.ndarray,
                                 min_area_cm2: float = 1.0) -> Tuple[np.ndarray, list]:
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
        all_segments = self._analyze_dent_segments(dent_mask, depth_map, depth_diff, direction_map)
        
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
    
    def _calculate_median_panel_depth(self, depth_map: np.ndarray, dent_mask: np.ndarray, 
                                     panel_mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate median depth of the panel region, excluding dent areas.
        
        This represents the "normal" panel surface depth, which is used as reference
        for measuring dent depth.
        
        Args:
            depth_map: Depth map (H, W) in meters
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            panel_mask: Optional panel mask (H, W) where True/1.0 = panel region.
                       If None, uses entire image.
            
        Returns:
            Median panel depth in meters
        """
        # Combine masks: panel region excluding dent areas
        if panel_mask is not None:
            # Panel region (excluding dents)
            panel_region = (panel_mask > 0) & (dent_mask == 0)
        else:
            # Use entire image excluding dents
            panel_region = (dent_mask == 0)
        
        # Get valid panel depths (non-zero, finite, excluding dents)
        panel_depths = depth_map[panel_region & (depth_map > 0) & np.isfinite(depth_map)]
        
        if len(panel_depths) == 0:
            # Fallback: use median of all valid depths if no panel region found
            valid_depths = depth_map[(depth_map > 0) & np.isfinite(depth_map) & (dent_mask == 0)]
            if len(valid_depths) == 0:
                logger.warning("No valid panel depths found for median calculation, using mean of all depths")
                valid_depths = depth_map[(depth_map > 0) & np.isfinite(depth_map)]
                if len(valid_depths) == 0:
                    return 0.0
            return float(np.median(valid_depths))
        
        return float(np.median(panel_depths))
    
    def _calculate_depth_diff_from_median(self, depth_map: np.ndarray, dent_mask: np.ndarray,
                                          panel_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate depth difference map relative to median panel depth.
        
        Args:
            depth_map: Depth map (H, W) in meters (dented depth)
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            panel_mask: Optional panel mask (H, W) where True/1.0 = panel region
            
        Returns:
            Depth difference map (H, W) in meters, where values represent
            deviation from median panel depth
        """
        median_depth = self._calculate_median_panel_depth(depth_map, dent_mask, panel_mask)
        
        # Calculate absolute depth difference from median
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

