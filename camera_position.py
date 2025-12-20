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
    
    def _get_randomized_camera_height(self, base_height: float, variation_range: float = 0.08) -> float:
        """
        Returns a randomized camera height to simulate human variation.
        
        Args:
            base_height: Base camera height in meters
            variation_range: Maximum variation in meters (±variation_range/2)
                             Default 0.08m (±4cm) simulates natural human variation
        
        Returns:
            Randomized camera height within [base_height - variation_range/2, base_height + variation_range/2]
        """
        variation = np.random.uniform(-variation_range / 2, variation_range / 2)
        return base_height + variation
    
    def _get_randomized_distance(self, base_distance: float, variation_percentage: float = 0.15) -> float:
        """
        Returns a randomized distance to simulate human variation in positioning.
        
        Args:
            base_distance: Base distance in meters
            variation_percentage: Maximum variation as percentage (±variation_percentage/2)
                                 Default 0.15 (±7.5%) simulates natural human variation
        
        Returns:
            Randomized distance within [base_distance * (1 - variation_percentage/2), 
                                       base_distance * (1 + variation_percentage/2)]
        """
        variation_factor = np.random.uniform(1 - variation_percentage / 2, 1 + variation_percentage / 2)
        return base_distance * variation_factor

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
        # poses.extend(self._get_internal_corner_views(length, height, width))  # Corner captures commented out
        poses.extend(self._get_internal_roof_views(length, height, width))
        poses.extend(self._get_internal_side_wall_views(length, height, width))
        
        return poses

    def _get_internal_end_wall_views(self, length: float, height: float, width: float) -> List[Dict]:
        """Generates close-up internal views of the left door, right door, and back wall."""
        poses = []
        # Use a closer distance for door captures to maximize door panel area in frame
        # Reduced distance to get closer and capture bigger area of door panel
        door_dist = self.r_config.INTERNAL_DOOR_DISTANCE * 0.8  # Reduce distance by 40% to get closer
        # Randomize back wall distance to simulate human variation in positioning
        back_wall_dist = self._get_randomized_distance(self.r_config.INTERNAL_BACK_WALL_DISTANCE)

        # Get door opening angle from config
        # Note: In generation coords, door rotates around Z-axis (vertical)
        # In camera coords (y-up): X=length, Y=height, Z=width
        # Rotation around vertical axis in generation becomes rotation around Y-axis in camera coords
        door_open_angle = self.c_config.DOOR_OPEN_ANGLE  # Opening angle in radians
        
        # Door is at the front (x = length/2), spanning the width
        # Left door is on the left side (negative Z), right door is on the right side (positive Z)
        # Each door is approximately half the width
        door_half_width = width / 4.0  # Each door panel is roughly 1/4 of total width from center
        
        # Calculate door panel height (approximate, accounting for frame)
        # Door opening height from config, but we'll use container height as reference
        # Upper and lower target heights for maximizing door panel capture
        door_bottom = 0.0  # Floor level
        door_top = height  # Top of container
        door_upper_target = door_bottom + height * 0.7  # Upper portion (70% from bottom)
        door_lower_target = door_bottom + height * 0.3  # Lower portion (30% from bottom)
        
        # Door hinge positions (approximate - at the edge of door panel)
        # Left door hinge is at left edge, right door hinge is at right edge
        # Hinge is at door_x_pos, and at the edge of the door panel
        door_x_pos = length / 2  # Door is at front of container
        
        # Calculate door center positions after rotation
        # Left door (side=-1): rotates by -door_open_angle around vertical axis (Y-axis in camera coords)
        # Right door (side=1): rotates by +door_open_angle around vertical axis (Y-axis in camera coords)
        # Rotation is around the hinge point at the edge of the door
        
        # Left door: rotates around its left edge (hinge at z = -width/2 approximately)
        left_hinge_z = -width / 2  # Approximate hinge position at left edge
        left_door_center_z_closed = -door_half_width  # Center when closed
        left_open_angle = -door_open_angle  # Left door opens inward (negative angle)
        
        # Right door: rotates around its right edge (hinge at z = width/2 approximately)
        right_hinge_z = width / 2  # Approximate hinge position at right edge
        right_door_center_z_closed = door_half_width  # Center when closed
        right_open_angle = door_open_angle  # Right door opens inward (positive angle)
        
        # Calculate rotated door center positions
        # Rotation around Y-axis (vertical) in camera coordinates
        # Rotation formula around Y-axis: 
        #   x' = x*cos(θ) + z*sin(θ)
        #   z' = -x*sin(θ) + z*cos(θ)
        # But we rotate around hinge point, so translate to hinge, rotate, translate back
        
        # Left door: rotate around hinge at (door_x_pos, left_hinge_z)
        # Door center when closed: (door_x_pos, left_door_center_z_closed)
        left_center_offset_z = left_door_center_z_closed - left_hinge_z
        left_center_offset_x = 0  # Door center is at same X as hinge
        # Rotate offset vector
        left_center_offset_x_rotated = left_center_offset_x * np.cos(left_open_angle) + left_center_offset_z * np.sin(left_open_angle)
        left_center_offset_z_rotated = -left_center_offset_x * np.sin(left_open_angle) + left_center_offset_z * np.cos(left_open_angle)
        # Translate back to world coordinates
        left_door_center_x_rotated = door_x_pos + left_center_offset_x_rotated
        left_door_center_z_rotated = left_hinge_z + left_center_offset_z_rotated
        
        # Right door: rotate around hinge at (door_x_pos, right_hinge_z)
        # Door center when closed: (door_x_pos, right_door_center_z_closed)
        right_center_offset_z = right_door_center_z_closed - right_hinge_z
        right_center_offset_x = 0  # Door center is at same X as hinge
        # Rotate offset vector
        right_center_offset_x_rotated = right_center_offset_x * np.cos(right_open_angle) + right_center_offset_z * np.sin(right_open_angle)
        right_center_offset_z_rotated = -right_center_offset_x * np.sin(right_open_angle) + right_center_offset_z * np.cos(right_open_angle)
        # Translate back to world coordinates
        right_door_center_x_rotated = door_x_pos + right_center_offset_x_rotated
        right_door_center_z_rotated = right_hinge_z + right_center_offset_z_rotated
        
        # Calculate door normal direction after rotation (perpendicular to door surface)
        # Door normal rotates with the door around Y-axis
        # Original normal: (1, 0, 0) pointing outward
        # After rotation by θ around Y-axis: (cos(θ), 0, sin(θ)) - still pointing outward
        # For camera inside container looking at door, we need inward normal (negated)
        # Left door normal (inward): (-cos(left_open_angle), 0, -sin(left_open_angle))
        left_door_normal_x = -np.cos(left_open_angle)
        left_door_normal_z = -np.sin(left_open_angle)
        
        # Right door normal (inward): (-cos(right_open_angle), 0, -sin(right_open_angle))
        right_door_normal_x = -np.cos(right_open_angle)
        right_door_normal_z = -np.sin(right_open_angle)
        
        # Position camera along the rotated door normal direction to be perpendicular to door surface
        # Camera should be inside container, looking at door
        # Camera position = door_center - normal * distance (moving inward from door)
        
        # Left door - Upper shot
        cam_height_left_upper = self._get_randomized_camera_height(door_upper_target)
        # Camera positioned along rotated door normal (inward), at distance door_dist from door surface
        # Since normal points inward, camera = door_center + normal * distance (moving inward)
        left_cam_eye_x = left_door_center_x_rotated + left_door_normal_x * door_dist
        left_cam_eye_z = left_door_center_z_rotated + left_door_normal_z * door_dist
        # Target point on door surface (rotated position)
        left_target_x = left_door_center_x_rotated
        left_target_z = left_door_center_z_rotated
        poses.append({
            "name": "internal_door_left_upper",
            "eye": torch.tensor([[left_cam_eye_x, cam_height_left_upper, left_cam_eye_z]]),
            "at": torch.tensor([[left_target_x, door_upper_target, left_target_z]]),
            "up": torch.tensor([[0, 1, 0]])
        })
        
        # Left door - Lower shot
        cam_height_left_lower = self._get_randomized_camera_height(door_lower_target)
        poses.append({
            "name": "internal_door_left_lower",
            "eye": torch.tensor([[left_cam_eye_x, cam_height_left_lower, left_cam_eye_z]]),
            "at": torch.tensor([[left_target_x, door_lower_target, left_target_z]]),
            "up": torch.tensor([[0, 1, 0]])
        })
        
        # Right door - Upper shot
        cam_height_right_upper = self._get_randomized_camera_height(door_upper_target)
        # Camera positioned along rotated door normal (inward), at distance door_dist from door surface
        # Since normal points inward, camera = door_center + normal * distance (moving inward)
        right_cam_eye_x = right_door_center_x_rotated + right_door_normal_x * door_dist
        right_cam_eye_z = right_door_center_z_rotated + right_door_normal_z * door_dist
        # Target point on door surface (rotated position)
        right_target_x = right_door_center_x_rotated
        right_target_z = right_door_center_z_rotated
        poses.append({
            "name": "internal_door_right_upper",
            "eye": torch.tensor([[right_cam_eye_x, cam_height_right_upper, right_cam_eye_z]]),
            "at": torch.tensor([[right_target_x, door_upper_target, right_target_z]]),
            "up": torch.tensor([[0, 1, 0]])
        })
        
        # Right door - Lower shot
        cam_height_right_lower = self._get_randomized_camera_height(door_lower_target)
        poses.append({
            "name": "internal_door_right_lower",
            "eye": torch.tensor([[right_cam_eye_x, cam_height_right_lower, right_cam_eye_z]]),
            "at": torch.tensor([[right_target_x, door_lower_target, right_target_z]]),
            "up": torch.tensor([[0, 1, 0]])
        })

        # 4. Back wall view (from inside, looking at center of wall)
        # Camera is positioned further from back wall to capture more area
        # Distance is randomized to simulate human variation
        cam_height_back = self._get_randomized_camera_height(self.r_config.INTERNAL_CAMERA_HEIGHT)
        poses.append({
            "name": "internal_back_wall",
            "eye": torch.tensor([[-length / 2 + back_wall_dist, cam_height_back, 0]]),
            "at": torch.tensor([[-length / 2, height / 2, 0]]),
            "up": torch.tensor([[0, 1, 0]])
        })
        return poses

    # Corner camera captures commented out
    # def _get_internal_corner_views(self, length: float, height: float, width: float) -> List[Dict]:
    #     """Generates internal views of the four corners from a closer vantage point."""
    #     poses = []
    #     cam_height = self.r_config.INTERNAL_CAMERA_HEIGHT
    #     dist = self.r_config.INTERNAL_CORNER_SHOT_DISTANCE
    #
    #     # Position camera 1.5m from the corner, looking towards the vertical center of the corner.
    #     corners = {
    #         "back_left":   {"at": [-length / 2, height / 2, -width / 2], "eye": [-length / 2 + dist, cam_height, -width / 2 + dist]},
    #         "back_right":  {"at": [-length / 2, height / 2,  width / 2], "eye": [-length / 2 + dist, cam_height,  width / 2 - dist]},
    #         "front_left":  {"at": [ length / 2, height / 2, -width / 2], "eye": [ length / 2 - dist, cam_height, -width / 2 + dist]},
    #         "front_right": {"at": [ length / 2, height / 2,  width / 2], "eye": [ length / 2 - dist, cam_height,  width / 2 - dist]}
    #     }
    #     
    #     for name, pos in corners.items():
    #         poses.append({
    #             "name": f"internal_corner_{name}",
    #             "eye": torch.tensor([pos["eye"]]),
    #             "at": torch.tensor([pos["at"]]),
    #             "up": torch.tensor([[0, 1, 0]])
    #         })
    #     return poses

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
        # Use base height for coverage calculations, then randomize per shot
        base_cam_height = self.r_config.INTERNAL_ROOF_CAMERA_HEIGHT
        
        # Calculate distance from camera to roof using base height for coverage calculations
        # Distance = roof_height - camera_height
        base_distance_to_roof = height - base_cam_height
        
        # Ensure base camera height is reasonable (not below floor or too high)
        # Floor is at y=0, so cam_height should be > 0
        if base_cam_height < 0.1:
            base_cam_height = 0.1  # Minimum 10cm from floor
            base_distance_to_roof = height - base_cam_height
        elif base_cam_height > height - 0.1:
            base_cam_height = height - 0.1  # Keep at least 10cm from roof
            base_distance_to_roof = 0.1
        
        # Calculate horizontal coverage based on FOV using base distance
        # FOV = 75 degrees, so horizontal coverage = 2 * distance * tan(FOV/2)
        fov_rad = np.deg2rad(self.r_config.CAMERA_FOV)
        horizontal_coverage = 2 * base_distance_to_roof * np.tan(fov_rad / 2.0)
        
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
            # Randomize camera height for each shot to simulate human variation
            cam_height = self._get_randomized_camera_height(base_cam_height)
            # Ensure randomized height stays within reasonable bounds
            cam_height = max(0.1, min(cam_height, height - 0.1))
            
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
        """
        Generates panoramic views of the internal side walls with full coverage.
        
        Ensures that all shots together cover the full length of the container with proper overlap.
        Uses base distance for coverage calculations, then randomizes per shot.
        """
        poses = []
        num_shots = self.r_config.SIDE_WALL_SHOT_COUNT
        
        # Use base distance for coverage calculations (before randomization)
        base_side_wall_dist = self.r_config.INTERNAL_SIDE_WALL_DISTANCE
        
        # Calculate horizontal coverage based on FOV and distance from wall
        # FOV = 75 degrees, so horizontal coverage = 2 * distance * tan(FOV/2)
        fov_rad = np.deg2rad(self.r_config.CAMERA_FOV)
        horizontal_coverage = 2 * base_side_wall_dist * np.tan(fov_rad / 2.0)
        
        # Use 90% of coverage to account for edge effects
        length_coverage_per_shot = horizontal_coverage * 0.9
        
        # Calculate overlap percentage (20-30% is good for stitching)
        overlap_percentage = 0.25  # 25% overlap
        
        # Calculate step size between shots (with overlap)
        step_size = length_coverage_per_shot * (1 - overlap_percentage)
        
        # Calculate total coverage needed with margins
        edge_margin = length_coverage_per_shot * 0.15  # 15% margin beyond edges
        coverage_needed = length + 2 * edge_margin
        
        # Calculate number of shots needed for full coverage
        num_shots_needed = int(np.ceil(coverage_needed / step_size)) + 1
        
        # Use configured number or calculated number, whichever is larger
        # This ensures we always have enough shots for full coverage
        num_shots = max(num_shots, num_shots_needed)
        
        # Calculate starting and ending positions to cover full length with margins
        start_x = -length / 2 - edge_margin
        end_x = length / 2 + edge_margin
        
        # Generate evenly spaced positions with proper overlap
        if num_shots == 1:
            x_positions = [0.0]
        else:
            x_positions = np.linspace(start_x, end_x, num_shots)
        
        # Skip first and last positions to avoid capturing exterior surfaces
        # Only process middle positions (similar to roof shots)
        if len(x_positions) >= 3:
            middle_positions = x_positions[1:-1]
        elif len(x_positions) == 2:
            # If only 2 positions, skip both (no middle positions)
            middle_positions = []
        else:
            # If only 1 position, skip it
            middle_positions = []
        
        # Now randomize distance per shot (for variation, but keep coverage calculations based on base)
        side_wall_dist = self._get_randomized_distance(self.r_config.INTERNAL_SIDE_WALL_DISTANCE)

        # Left wall: camera positioned at distance from left wall (z = -width/2), same logic as back wall
        # Camera is on right side (positive Z), looking at left wall
        camera_z_left = -width / 2 + side_wall_dist
        for i, x_pos in enumerate(middle_positions):
            cam_height = self._get_randomized_camera_height(self.r_config.INTERNAL_CAMERA_HEIGHT)
            poses.append({
                "name": f"internal_left_wall_{i+1}",
                "eye": torch.tensor([[x_pos, cam_height, camera_z_left]]),
                "at": torch.tensor([[x_pos, height / 2, -width / 2]]),
                "up": torch.tensor([[0, 1, 0]])
            })
        
        # Right wall: camera positioned at distance from right wall (z = width/2), same logic as back wall
        # Camera is on left side (negative Z), looking at right wall
        camera_z_right = width / 2 - side_wall_dist
        for i, x_pos in enumerate(middle_positions):
            cam_height = self._get_randomized_camera_height(self.r_config.INTERNAL_CAMERA_HEIGHT)
            poses.append({
                "name": f"internal_right_wall_{i+1}",
                "eye": torch.tensor([[x_pos, cam_height, camera_z_right]]),
                "at": torch.tensor([[x_pos, height / 2, width / 2]]),
                "up": torch.tensor([[0, 1, 0]])
            })

        return poses 