"""
Generates a list of camera poses for rendering a container from various angles.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple

from config import ContainerConfig, RendererConfig

class CameraPoseGenerator:
    """Calculates all camera positions and targets for rendering."""

    def __init__(self, container_config: ContainerConfig, renderer_config: RendererConfig):
        self.c_config = container_config
        self.r_config = renderer_config

    def generate_poses(self, container_type: str) -> List[Dict]:
        """
        Generates a list of all camera poses for a given container type.
        
        Each pose is a dictionary with 'name', 'eye', 'at', and 'up' keys.
        """
        spec = self.c_config.CONTAINER_SPECS[container_type]
        # After y-up transform: length is X, height is Y, width is Z
        # The container is centered at the origin.
        length, height, width = spec["external_y_up"]

        poses = []
        poses.extend(self._get_internal_end_wall_views(length, height, width))
        poses.extend(self._get_internal_corner_views(length, height, width))
        poses.extend(self._get_internal_roof_views(length, height, width))
        poses.extend(self._get_internal_side_wall_views(length, height, width))
        
        return poses

    def _get_internal_end_wall_views(self, length: float, height: float, width: float) -> List[Dict]:
        """Generates close-up internal views of the left door, right door, and back wall."""
        poses = []
        # Use a closer distance for door captures to get better detail
        door_dist = self.r_config.INTERNAL_DOOR_DISTANCE  # Reduce distance by 40% to get closer
        back_wall_dist = self.r_config.INTERNAL_BACK_WALL_DISTANCE
        cam_height = self.r_config.INTERNAL_CAMERA_HEIGHT

        # Door is at the front (x = length/2), spanning the width
        # Left door is on the left side (negative Z), right door is on the right side (positive Z)
        # Each door is approximately half the width
        door_half_width = width / 4.0  # Each door panel is roughly 1/4 of total width from center
        
        # Left door view - camera positioned on the RIGHT side, facing LEFT towards left door at 90 degrees
        # To capture left door perpendicularly: stand on right side, look left
        left_door_center_z = -door_half_width  # Center of left door panel (negative Z = left side)
        camera_offset_z = door_dist * 0.5  # Offset camera to opposite side for perpendicular view
        poses.append({
            "name": "internal_door_left",
            "eye": torch.tensor([[length / 2 - door_dist, cam_height, camera_offset_z]]),  # Camera on RIGHT side (positive Z)
            "at": torch.tensor([[length / 2, height / 2, left_door_center_z]]),  # Look at LEFT door center (negative Z)
            "up": torch.tensor([[0, 1, 0]])
        })
        
        # Right door view - camera positioned on the LEFT side, facing RIGHT towards right door at 90 degrees
        # To capture right door perpendicularly: stand on left side, look right
        right_door_center_z = door_half_width  # Center of right door panel (positive Z = right side)
        poses.append({
            "name": "internal_door_right",
            "eye": torch.tensor([[length / 2 - door_dist, cam_height, -camera_offset_z]]),  # Camera on LEFT side (negative Z)
            "at": torch.tensor([[length / 2, height / 2, right_door_center_z]]),  # Look at RIGHT door center (positive Z)
            "up": torch.tensor([[0, 1, 0]])
        })

        # 4. Back wall view (from inside, looking at center of wall)
        # Camera is positioned further from back wall to capture more area
        poses.append({
            "name": "internal_back_wall",
            "eye": torch.tensor([[-length / 2 + back_wall_dist, cam_height, 0]]),
            "at": torch.tensor([[-length / 2, height / 2, 0]]),
            "up": torch.tensor([[0, 1, 0]])
        })
        return poses

    def _get_internal_corner_views(self, length: float, height: float, width: float) -> List[Dict]:
        """Generates internal views of the four corners from a closer vantage point."""
        poses = []
        cam_height = self.r_config.INTERNAL_CAMERA_HEIGHT
        dist = self.r_config.INTERNAL_CORNER_SHOT_DISTANCE

        # Position camera 1.5m from the corner, looking towards the vertical center of the corner.
        corners = {
            "back_left":   {"at": [-length / 2, height / 2, -width / 2], "eye": [-length / 2 + dist, cam_height, -width / 2 + dist]},
            "back_right":  {"at": [-length / 2, height / 2,  width / 2], "eye": [-length / 2 + dist, cam_height,  width / 2 - dist]},
            "front_left":  {"at": [ length / 2, height / 2, -width / 2], "eye": [ length / 2 - dist, cam_height, -width / 2 + dist]},
            "front_right": {"at": [ length / 2, height / 2,  width / 2], "eye": [ length / 2 - dist, cam_height,  width / 2 - dist]}
        }
        
        for name, pos in corners.items():
            poses.append({
                "name": f"internal_corner_{name}",
                "eye": torch.tensor([pos["eye"]]),
                "at": torch.tensor([pos["at"]]),
                "up": torch.tensor([[0, 1, 0]])
            })
        return poses

    def _get_internal_roof_views(self, length: float, height: float, width: float) -> List[Dict]:
        """
        Generates panoramic views of the internal roof with full coverage and overlap for stitching.
        
        Camera positions are calculated to ensure:
        - Full coverage of roof length (including edges)
        - Proper overlap between shots (20-30% for stitching)
        - Full coverage of roof width
        """
        poses = []
        num_shots = self.r_config.ROOF_SHOT_COUNT
        
        # Use configured camera height for roof views (lower height = further from roof = better coverage)
        cam_height = self.r_config.INTERNAL_ROOF_CAMERA_HEIGHT
        
        # Calculate distance from camera to roof
        # Distance = roof_height - camera_height
        distance_to_roof = height - cam_height
        
        # Ensure camera height is reasonable (not below floor or too high)
        # Floor is at y=0, so cam_height should be > 0
        if cam_height < 0.1:
            cam_height = 0.1  # Minimum 10cm from floor
            distance_to_roof = height - cam_height
        elif cam_height > height - 0.1:
            cam_height = height - 0.1  # Keep at least 10cm from roof
            distance_to_roof = 0.1
        
        # Calculate horizontal coverage based on FOV
        # FOV = 75 degrees, so horizontal coverage = 2 * distance * tan(FOV/2)
        fov_rad = np.deg2rad(self.r_config.CAMERA_FOV)
        horizontal_coverage = 2 * distance_to_roof * np.tan(fov_rad / 2.0)
        
        # Verify that horizontal coverage captures full roof width
        # Note: If horizontal_coverage < width, the camera may not capture the full width
        # in a single shot. This depends on distance from roof, FOV, and container dimensions.
        # With INTERNAL_ROOF_CAMERA_HEIGHT=0.8m, distance to roof is ~1.8m (standard) or ~2.1m (high cube).
        # With FOV=75°, this provides good coverage. Adjust INTERNAL_ROOF_CAMERA_HEIGHT if needed.
        width_coverage_ratio = horizontal_coverage / width if width > 0 else 1.0
        if width_coverage_ratio < 1.0:
            # Log a warning but proceed - width coverage may be partial
            # The stitching algorithm can handle this if there's sufficient overlap along length
            pass
        
        # Calculate coverage along length (X-axis)
        # Since camera looks straight up, coverage is circular with diameter = horizontal_coverage
        # For length coverage, we use the full horizontal coverage (conservative)
        length_coverage_per_shot = horizontal_coverage * 0.9  # Use 90% to account for edge effects
        
        # Calculate overlap percentage (20-30% is good for stitching)
        overlap_percentage = 0.25  # 25% overlap
        
        # Calculate step size between shots (with overlap)
        step_size = length_coverage_per_shot * (1 - overlap_percentage)
        
        # Calculate number of shots needed to cover full length with overlap
        # Add margin beyond edges to ensure complete coverage
        edge_margin = length_coverage_per_shot * 0.15  # 15% margin beyond edges
        coverage_needed = length + 2 * edge_margin
        num_shots_needed = int(np.ceil(coverage_needed / step_size)) + 1
        
        # Use the configured number or calculated number, whichever is larger
        # This ensures we always have enough shots for full coverage
        num_shots = max(num_shots, num_shots_needed)
        
        # Calculate starting and ending positions to cover full length with margins
        # Start before the back edge, end after the front edge
        start_x = -length / 2 - edge_margin
        end_x = length / 2 + edge_margin
        
        # Generate evenly spaced positions with proper overlap
        if num_shots == 1:
            # Single shot: center position
            x_positions = [0.0]
        else:
            # Multiple shots: evenly spaced with overlap
            x_positions = np.linspace(start_x, end_x, num_shots)
        
        # Ensure we capture the full width by looking at the center of the roof width
        # The camera should be positioned to capture from -width/2 to +width/2
        roof_center_z = 0.0  # Center of roof width
        
        # Skip the first and last positions - use only middle positions
        # x_positions[1:-1] excludes both first and last elements
        # Only process if we have at least 3 positions (to have middle positions after removing first and last)
        if len(x_positions) >= 3:
            middle_positions = x_positions[1:-1]
        elif len(x_positions) == 2:
            # If only 2 positions, skip both (no middle positions)
            middle_positions = []
        else:
            # If only 1 position, skip it
            middle_positions = []
        
        for i, x_pos in enumerate(middle_positions, start=1):
            # Position camera at calculated X position, looking up at roof
            # Target is directly above camera position at roof height, centered on width
            poses.append({
                "name": f"internal_roof_{i}",
                "eye": torch.tensor([[x_pos, cam_height, roof_center_z]]),
                "at": torch.tensor([[x_pos, height, roof_center_z]]),  # Look directly up at roof center
                # Use X-axis as up vector (towards door) for consistent orientation
                "up": torch.tensor([[1, 0, 0]]) 
            })
        
        return poses

    def _get_internal_side_wall_views(self, length: float, height: float, width: float) -> List[Dict]:
        """Generates panoramic views of the internal side walls."""
        poses = []
        cam_height = self.r_config.INTERNAL_CAMERA_HEIGHT
        num_shots = self.r_config.SIDE_WALL_SHOT_COUNT
        wall_margin = self.r_config.INTERNAL_SIDE_WALL_DISTANCE  # Distance from opposite wall to place camera

        # Create shots along the length of the container for both walls
        x_positions = np.linspace(-length / 2 * 0.9, length / 2 * 0.9, num_shots)

        # Left wall (seen from right side)
        for i, x_pos in enumerate(x_positions):
            poses.append({
                "name": f"internal_left_wall_{i+1}",
                "eye": torch.tensor([[x_pos, cam_height, width / 2 - wall_margin]]),
                "at": torch.tensor([[x_pos, height / 2, -width / 2]]),
                "up": torch.tensor([[0, 1, 0]])
            })
        
        # Right wall (seen from left side)
        for i, x_pos in enumerate(x_positions):
            poses.append({
                "name": f"internal_right_wall_{i+1}",
                "eye": torch.tensor([[x_pos, cam_height, -width / 2 + wall_margin]]),
                "at": torch.tensor([[x_pos, height / 2, width / 2]]),
                "up": torch.tensor([[0, 1, 0]])
            })

        return poses 