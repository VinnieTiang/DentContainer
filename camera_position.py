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
        poses.extend(self._get_overall_views(length, height, width))
        poses.extend(self._get_internal_end_wall_views(length, height, width))
        poses.extend(self._get_internal_corner_views(length, height, width))
        poses.extend(self._get_internal_roof_views(length, height, width))
        poses.extend(self._get_internal_side_wall_views(length, height, width))
        
        return poses

    def _get_overall_views(self, length: float, height: float, width: float) -> List[Dict]:
        """Generates the overall internal and external aerial views."""
        poses = []
        cam_height = self.r_config.INTERNAL_CAMERA_HEIGHT
        
        # 1. Overall internal view (from back, looking at center of door wall)
        poses.append({
            "name": "internal_overall",
            "eye": torch.tensor([[-length / 2 + self.r_config.SHOT_FROM_BACK_OFFSET, cam_height, 0]]),
            "at": torch.tensor([[length / 2, height / 2, 0]]), # Look at vertical center
            "up": torch.tensor([[0, 1, 0]])
        })
        
        # 2. Aerial view from outside
        aerial_dist = length * self.r_config.AERIAL_VIEW_DISTANCE_MULTIPLIER
        poses.append({
            "name": "external_aerial",
            "eye": torch.tensor([[-aerial_dist / 2, height + aerial_dist / 2, -aerial_dist / 2]]),
            "at": torch.tensor([[0, height / 2, 0]]),
            "up": torch.tensor([[0, 1, 0]])
        })
        return poses

    def _get_internal_end_wall_views(self, length: float, height: float, width: float) -> List[Dict]:
        """Generates close-up internal views of the door and back wall."""
        poses = []
        door_dist = self.r_config.INTERNAL_DOOR_DISTANCE  # Use specific distance for door view
        back_wall_dist = self.r_config.INTERNAL_BACK_WALL_DISTANCE
        cam_height = self.r_config.INTERNAL_CAMERA_HEIGHT

        # 3. Door view (from inside, looking at center of wall)
        # Camera is positioned further from door to capture more area
        poses.append({
            "name": "internal_door",
            "eye": torch.tensor([[length / 2 - door_dist, cam_height, 0]]),
            "at": torch.tensor([[length / 2, height / 2, 0]]),
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
        """Generates panoramic views of the internal roof."""
        poses = []
        cam_height = self.r_config.INTERNAL_CAMERA_HEIGHT
        num_shots = self.r_config.ROOF_SHOT_COUNT
        
        # Create shots along the length of the container
        x_positions = np.linspace(-length / 2 * 0.9, length / 2 * 0.9, num_shots)
        
        for i, x_pos in enumerate(x_positions):
            poses.append({
                "name": f"internal_roof_{i+1}",
                "eye": torch.tensor([[x_pos, cam_height, 0]]),
                "at": torch.tensor([[x_pos, height, 0]]),
                # Use a stable 'up' vector. Since the view direction is along Y,
                # we can use a vector along X (towards door) or Z.
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