"""
Configuration classes for container rendering and generation.
"""

import numpy as np


class ContainerConfig:
    """Configuration for container dimensions and parameters."""
    
    # ISO standard container specifications
    CONTAINER_SPECS = {
        "20ft": {
            "external": (6.058, 2.438, 2.591),  # L, W, H in meters
            "external_y_up": (6.058, 2.591, 2.438),  # After y-up transform: length is X, height is Y, width is Z
            "door_opening": (2.286, 2.261)  # W, H
        },
        "40ft": {
            "external": (12.192, 2.438, 2.591),
            "external_y_up": (12.192, 2.591, 2.438),  # After y-up transform: length is X, height is Y, width is Z
            "door_opening": (2.286, 2.261)
        },
        "40ft_hc": {  # High Cube
            "external": (12.192, 2.438, 2.896),
            "external_y_up": (12.192, 2.896, 2.438),  # After y-up transform: length is X, height is Y, width is Z
            "door_opening": (2.286, 2.566)
        }
    }
    
    # Common container colors (name: RGB float values)
    CONTAINER_COLORS = {
        "red": (0.7, 0.1, 0.1),
        "blue": (0.1, 0.3, 0.7),
        "green": (0.1, 0.5, 0.2),
        "orange": (0.9, 0.5, 0.1),
        "gray": (0.5, 0.5, 0.5),
        "white": (0.9, 0.9, 0.9),
        "yellow": (0.9, 0.8, 0.1),
        "brown": (0.5, 0.3, 0.2)
    }
    
    # Color probabilities for random selection
    COLOR_PROBABILITIES = [0.20, 0.20, 0.15, 0.10, 0.15, 0.10, 0.05, 0.05]
    
    # Structural parameters
    WALL_THICKNESS = 0.002
    FLOOR_THICKNESS = 0.028
    DOOR_THICKNESS = 0.050
    
    # Corrugation parameters
    CORRUGATION_DEPTH = 0.025
    CORRUGATION_PITCH = 0.20
    
    # Floor parameters
    FLOOR_PLANK_WIDTH = 0.15  # Width of wooden planks
    FLOOR_PLANK_GAP = 0.002   # Gap between planks
    TIE_DOWN_SPACING = 0.6    # Spacing between tie-down points
    
    # Roof parameters
    ROOF_ARCH_HEIGHT = 0.015  # Slight arch for water drainage
    VENTILATION_HOLE_RADIUS = 0.05  # Ventilation holes
    
    # Corner casting parameters
    CORNER_CASTING_APERTURE_SIZE = 0.100  # ISO standard aperture size
    CORNER_CASTING_APERTURE_DEPTH = 0.050
    
    # Door parameters
    DOOR_HINGE_COUNT = 3  # Number of hinges per door
    DOOR_SEAL_THICKNESS = 0.015
    DOOR_HANDLE_SIZE = 0.12
    DOOR_OPEN_ANGLE = np.radians(25)  # Opening angle in radians (~15 degrees)


class RendererConfig:
    """Configuration for rendering parameters."""
    
    # Image settings
    IMAGE_SIZE = 512  # Square images
    
    # Camera settings
    CAMERA_FOV = 75.0  # Field of view in degrees (increased to see more of the wall panels)
    ZNEAR = 0.01  # Near clipping plane
    ZFAR = 100.0  # Far clipping plane
    
    # Camera positioning parameters
    INTERNAL_CAMERA_HEIGHT = 1.5  # Height of camera inside container (meters)
    SHOT_FROM_BACK_OFFSET = 0.5  # Offset from back wall for overall view
    AERIAL_VIEW_DISTANCE_MULTIPLIER = 1.5  # Multiplier for aerial view distance
    EXTERNAL_SHOT_DISTANCE = 1.5  # Distance for external shots
    INTERNAL_DOOR_DISTANCE = 2.5  # Distance from door for internal door view (increased to see more area)
    INTERNAL_BACK_WALL_DISTANCE = 2.5  # Distance from back wall for back wall view (increased to see more area)
    INTERNAL_CORNER_SHOT_DISTANCE = 1.5  # Distance from corner for corner views
    INTERNAL_SIDE_WALL_DISTANCE = 0.05  # Distance from opposite wall for side wall views (smaller = camera closer to center = further from target wall = better view of whole panel)
    
    # Panoramic shot counts
    ROOF_SHOT_COUNT = 5  # Number of roof shots along container length
    SIDE_WALL_SHOT_COUNT = 5  # Number of side wall shots along container length
    
    # Annotation settings
    ANNOTATION_EXCLUSION_LIST = []  # List of shot names to exclude from annotations

