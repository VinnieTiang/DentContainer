#!/usr/bin/env python3
"""
ISO Shipping Container 3D Model Generator (Ultra Realistic Version)

This enhanced version includes:
- Realistic trapezoidal corrugated panels
- Detailed corner castings with apertures for twist locks
- Structural rails and cross-members
- Advanced door locking mechanisms with handles and cam retainers
- Proper frame construction with door hinges
- Multiple container types (20ft, 40ft, 40ft HC)
- Realistic floor with wooden plank pattern and tie-down points
- Roof ventilation details
- Identification plates and markings
- Forklift pockets
- Interior tie-down points
- Door seals and gaskets
- Enhanced structural details

Based on ISO 668:2020 standards and real-world container specifications
"""

import numpy as np
import trimesh
from pathlib import Path
import logging
import os
import shutil
import random
from typing import List, Tuple, Dict

try:
    import shapely.geometry
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not available for advanced corrugations. Install with: pip install shapely")

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper geometry builders
# -----------------------------------------------------------------------------
def _colored(mesh: trimesh.Trimesh, color_rgba: Tuple[int, int, int, int]) -> trimesh.Trimesh:
    """Return a copy of mesh with vertex_colors set to color_rgba."""
    mesh = mesh.copy()
    mesh.visual.vertex_colors = np.tile(color_rgba, (len(mesh.vertices), 1))
    return mesh


def _box(extents: Tuple[float, float, float],
         transform: np.ndarray = None,
         color: Tuple[int, int, int, int] = None) -> trimesh.Trimesh:
    """Create a box with optional transform and color."""
    m = trimesh.creation.box(extents)
    if transform is not None:
        m.apply_transform(transform)
    if color is not None:
        m = _colored(m, color)
    return m


def _cylinder(radius: float, height: float,
              sections: int = 20,
              transform: np.ndarray = None,
              color: Tuple[int, int, int, int] = None) -> trimesh.Trimesh:
    """Create a cylinder with optional transform and color."""
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    if transform is not None:
        m.apply_transform(transform)
    if color is not None:
        m = _colored(m, color)
    return m


# -----------------------------------------------------------------------------
# Container Specifications and Configuration
# -----------------------------------------------------------------------------
class ContainerConfig:
    """Configuration for container dimensions and parameters."""
    
    # ISO standard container specifications
    CONTAINER_SPECS = {
        "20ft": {
            "external": (6.058, 2.438, 2.591),  # L, W, H in meters
            "door_opening": (2.286, 2.261)  # W, H
        },
        "40ft": {
            "external": (12.192, 2.438, 2.591),
            "door_opening": (2.286, 2.261)
        },
        "40ft_hc": {  # High Cube
            "external": (12.192, 2.438, 2.896),
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
    
    # Structural parameters - ISO standard thicknesses (in meters)
    SIDE_WALL_THICKNESS = 0.0016  # 1.6mm - Corten steel side panels
    ROOF_THICKNESS = 0.002         # 2.0mm - Roof panels for additional strength
    DOOR_THICKNESS = 0.002         # 2.0mm - Front door panels (was incorrectly 50mm)
    BACK_WALL_THICKNESS = 0.002    # 2.0mm - Back wall panels
    FLOOR_THICKNESS = 0.028        # 28mm - Plywood floor panels
    # Legacy parameter for backward compatibility (used for general wall thickness)
    WALL_THICKNESS = 0.0016        # Default to side wall thickness
    
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


# -----------------------------------------------------------------------------
# Main Generator Class
# -----------------------------------------------------------------------------
class ShippingContainerGenerator:
    """Generate highly detailed ISO shipping container meshes."""
    
    def __init__(self):
        self.config = ContainerConfig()
        self.body_color_rgba: Tuple[int, int, int, int] = None
        logger.info("ShippingContainerGenerator (Advanced) initialized")
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, output_path="shipping_container.obj", 
                 container_type="20ft", color=None):
        """Generate the complete container model."""
        
        # Ensure output directory exists
        output_dir = Path("complete_containers")
        output_dir.mkdir(exist_ok=True)
        
        # Convert output_path to Path and ensure it's in the output directory
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = output_dir / output_path.name
        
        if container_type not in self.config.CONTAINER_SPECS:
            logger.warning(f"Unknown container_type {container_type} – defaulting to 20ft")
            container_type = "20ft"
        
        # Select color
        if color is None or color not in self.config.CONTAINER_COLORS:
            color_name = self._pick_body_color()
        else:
            color_name = color
        
        color_rgb_float = self.config.CONTAINER_COLORS[color_name]
        self.body_color_rgba = tuple(int(c * 255) for c in color_rgb_float) + (255,)
        
        specs = self.config.CONTAINER_SPECS[container_type]
        ext_l, ext_w, ext_h = specs["external"]
        
        logger.info(f"Building {container_type} container ({ext_l:.3f}m × {ext_w:.3f}m × {ext_h:.3f}m)")
        
        meshes: List[trimesh.Trimesh] = []
        
        # Build structural components
        logger.info("Creating main structure...")
        meshes.extend(self._create_side_walls(ext_l, ext_h, ext_w))
        meshes.append(self._create_back_wall(ext_w, ext_h, -ext_l / 2))
        meshes.append(self._create_roof(ext_l, ext_w, ext_h))
        meshes.append(self._create_floor(ext_l, ext_w))
        
        logger.info("Adding corner castings...")
        meshes.extend(self._create_corner_castings(ext_l, ext_w, ext_h))
        
        logger.info("Creating structural details...")
        meshes.extend(self._create_side_rails(ext_l, ext_w, ext_h))
        # Roof cross members removed - no horizontal poles on roof
        meshes.extend(self._create_floor_cross_members(ext_l, ext_w))
        
        logger.info("Adding roof-to-side connection details...")
        meshes.extend(self._create_roof_side_connections(ext_l, ext_w, ext_h))
        
        logger.info("Adding back wall structural frame...")
        meshes.extend(self._create_back_wall_frame(ext_w, ext_h, -ext_l / 2))
        
        logger.info("Creating doors...")
        meshes.extend(self._create_doors_and_frame(ext_l, ext_w, ext_h, specs["door_opening"]))
        
        logger.info("Adding floor details...")
        meshes.extend(self._create_floor_details(ext_l, ext_w))
        
        logger.info("Adding forklift pockets...")
        meshes.extend(self._create_forklift_pockets(ext_l, ext_w))
        
        # Interior tie-down points removed - no dots on floor panel
        # meshes.extend(self._create_interior_tie_downs(ext_l, ext_w, ext_h))
        
        # Combine meshes
        logger.info("Combining meshes...")
        container_mesh = trimesh.util.concatenate(meshes)
        container_mesh.remove_duplicate_faces()
        container_mesh.remove_unreferenced_vertices()
        
        logger.info(f"Model generated successfully!")
        logger.info(f"Main file: {output_path}")
        logger.info(f"\nModel statistics:")
        logger.info(f"  Vertices: {len(container_mesh.vertices)}")
        logger.info(f"  Faces: {len(container_mesh.faces)}")
        logger.info(f"  Bounding box: {container_mesh.bounds}")
        logger.info(f"  Volume: {container_mesh.volume:.2f} m³")
        
        # Convert to Y-up for compatibility
        y_up_transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        container_mesh.apply_transform(y_up_transform)
        
        # Export
        container_mesh.export(str(output_path))
        
        return container_mesh
    
    # ------------------------------------------------------------------
    # Corrugated panels with trapezoidal profile
    # ------------------------------------------------------------------
    def _create_corrugated_panel(self, dim_a: float, dim_b: float, *,
                                 normal: Tuple[int, int, int],
                                 offset: Tuple[float, float, float],
                                 shallow: bool = False,
                                 thickness: float = None) -> trimesh.Trimesh:
        """Create corrugated panel with trapezoidal profile.
        
        Args:
            dim_a: First dimension (corrugation direction)
            dim_b: Second dimension (extrusion direction)
            normal: Panel normal vector
            offset: Panel position offset
            shallow: Whether to use shallow corrugation
            thickness: Panel thickness in meters (uses WALL_THICKNESS if None)
        """
        
        if not SHAPELY_AVAILABLE:
            # Fallback to simple sinusoidal corrugation
            return self._create_corrugated_panel_simple(dim_a, dim_b, normal, offset, shallow, thickness)
        
        # Increase number of corrugations for higher detail
        na = max(1, int(dim_a / self.config.CORRUGATION_PITCH))
        pitch = dim_a / na
        depth = self.config.CORRUGATION_DEPTH * (0.4 if shallow else 1.0)
        wall_t = thickness if thickness is not None else self.config.WALL_THICKNESS
        
        # Generate trapezoidal profile with more subdivision points
        crest_ratio, valley_ratio = 0.35, 0.35
        slope_ratio = (1.0 - crest_ratio - valley_ratio) / 2.0
        
        # Add more points along each segment for smoother curves
        points_per_segment = 8  # Subdivide each corrugation segment
        
        outer_profile = []
        current_a = -dim_a / 2
        for i in range(na):
            # Valley start
            outer_profile.append([current_a, 0])
            current_a += pitch * valley_ratio
            
            # Valley end (with subdivision)
            valley_end = current_a
            for sub in range(1, points_per_segment):
                sub_a = current_a - pitch * valley_ratio + (pitch * valley_ratio * sub / points_per_segment)
                outer_profile.append([sub_a, 0])
            
            # Slope up (with subdivision)
            slope_start = current_a
            current_a += pitch * slope_ratio
            slope_end = current_a
            for sub in range(1, points_per_segment):
                sub_a = slope_start + (slope_end - slope_start) * sub / points_per_segment
                sub_depth = depth * (sub / points_per_segment)
                outer_profile.append([sub_a, sub_depth])
            
            # Crest (with subdivision)
            crest_start = current_a
            current_a += pitch * crest_ratio
            for sub in range(1, points_per_segment):
                sub_a = crest_start + (current_a - crest_start) * sub / points_per_segment
                outer_profile.append([sub_a, depth])
            
            # Slope down (with subdivision)
            slope_start = current_a
            current_a += pitch * slope_ratio
            slope_end = current_a
            for sub in range(1, points_per_segment):
                sub_a = slope_start + (slope_end - slope_start) * sub / points_per_segment
                sub_depth = depth * (1.0 - sub / points_per_segment)
                outer_profile.append([sub_a, sub_depth])
        
        outer_profile.append([dim_a / 2, 0])
        
        # Clean duplicate points
        outer_profile = [p for i, p in enumerate(outer_profile) 
                        if i == 0 or not np.allclose(p, outer_profile[i-1])]
        
        # Create closed polygon with thickness
        outer_xy = np.array(outer_profile)
        inner_xy = outer_xy - [0, wall_t]
        polygon_verts = np.vstack((outer_xy, inner_xy[::-1]))
        
        poly_verts_centered = polygon_verts - np.mean(polygon_verts, axis=0)
        poly = shapely.geometry.Polygon(poly_verts_centered)
        
        # Extrude with high resolution
        # Use more segments for smoother surfaces
        panel = trimesh.creation.extrude_polygon(poly, height=dim_b)
        panel.apply_translation([0, 0, -dim_b / 2])
        
        # Subdivide the mesh to increase polygon count for smoother appearance
        # This adds more triangles to the mesh for better detail
        try:
            # Subdivide each face to increase mesh density
            panel = panel.subdivide()
            # Optionally subdivide again for even more detail
            panel = panel.subdivide()
        except:
            # If subdivision fails, use original mesh
            pass
        
        # Apply orientation transform
        transform = self._get_panel_transform(normal)
        panel.apply_transform(transform)
        panel.apply_translation(offset)
        
        return _colored(panel, self.body_color_rgba)
    
    def _create_corrugated_panel_simple(self, dim_a: float, dim_b: float,
                                       normal: Tuple[int, int, int],
                                       offset: Tuple[float, float, float],
                                       shallow: bool = False,
                                       thickness: float = None) -> trimesh.Trimesh:
        """Fallback: simple sinusoidal corrugation with high mesh density."""
        num_corrugations = int(dim_a / self.config.CORRUGATION_PITCH)
        depth = self.config.CORRUGATION_DEPTH * (0.4 if shallow else 1.0)
        wall_t = thickness if thickness is not None else self.config.WALL_THICKNESS
        
        vertices = []
        faces = []
        # Significantly increased mesh density for smoother, more detailed panels
        # More segments along corrugation direction (dim_a)
        segments_a = max(150, int(dim_a / self.config.CORRUGATION_PITCH * 30))
        # More segments along perpendicular direction (dim_b)
        segments_b = max(100, int(dim_b * 40))
        
        for i in range(segments_a + 1):
            u = i / segments_a
            x_local = u * dim_a - dim_a / 2
            wave_position = u * num_corrugations
            z_offset = np.sin(wave_position * 2 * np.pi) * depth
            
            for j in range(segments_b + 1):
                v = j / segments_b
                y_local = v * dim_b - dim_b / 2
                vertices.append([x_local, y_local, z_offset])
        
        for i in range(segments_a):
            for j in range(segments_b):
                v1 = i * (segments_b + 1) + j
                v2 = v1 + 1
                v3 = (i + 1) * (segments_b + 1) + j
                v4 = v3 + 1
                faces.append([v1, v2, v4])
                faces.append([v1, v4, v3])
        
        panel = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        
        # Subdivide the mesh to increase polygon count for smoother, more detailed panels
        try:
            # Subdivide once to increase mesh density
            panel = panel.subdivide()
            # Subdivide again for even higher detail
            panel = panel.subdivide()
        except:
            # If subdivision fails, use original mesh
            pass
        
        # Apply orientation
        transform = self._get_panel_transform(normal)
        panel.apply_transform(transform)
        panel.apply_translation(offset)
        
        return _colored(panel, self.body_color_rgba)
    
    def _get_panel_transform(self, normal: Tuple[int, int, int]) -> np.ndarray:
        """Get transformation matrix for panel orientation."""
        normal_tuple = tuple(np.round(normal).astype(int))
        transform = np.eye(4)
        
        if normal_tuple[2] == 0:  # Vertical panels
            if normal_tuple == (0, 1, 0):  # Left wall
                transform = np.eye(4)
            elif normal_tuple == (0, -1, 0):  # Right wall
                transform = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
            elif normal_tuple == (1, 0, 0):  # Front (doors)
                transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
            elif normal_tuple == (-1, 0, 0):  # Back wall
                transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        else:  # Horizontal panels
            ry_neg_90 = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0])
            if normal_tuple == (0, 0, 1):  # Roof
                rx_90 = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
                transform = np.dot(rx_90, ry_neg_90)
            elif normal_tuple == (0, 0, -1):  # Floor
                rx_neg_90 = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                transform = np.dot(rx_neg_90, ry_neg_90)
        
        return transform
    
    # ------------------------------------------------------------------
    # Main structural components
    # ------------------------------------------------------------------
    def _create_side_walls(self, length: float, height: float, width: float) -> List[trimesh.Trimesh]:
        """Create left and right corrugated walls.
        
        Side panels are continuously welded to side rails and corner posts.
        The top edge meets the top side rail where it connects to the roof panel.
        """
        # Top rail height (60mm = 0.06m)
        top_rail_height = 0.06
        # Side wall extends from floor to top rail (not all the way to roof)
        # The top rail serves as the connection point
        wall_height = height - top_rail_height / 2  # Extend to middle of top rail
        offset_z = wall_height / 2
        
        left = self._create_corrugated_panel(length, wall_height, normal=(0, 1, 0),
                                            offset=(0, -width / 2, offset_z),
                                            thickness=self.config.SIDE_WALL_THICKNESS)
        right = self._create_corrugated_panel(length, wall_height, normal=(0, -1, 0),
                                             offset=(0, width / 2, offset_z),
                                             thickness=self.config.SIDE_WALL_THICKNESS)
        return [left, right]
    
    def _create_back_wall(self, width: float, height: float, x_pos: float) -> trimesh.Trimesh:
        """Create back corrugated wall.
        
        Back wall extends to meet the top side rails where it connects to the roof.
        The top rail serves as the connection point between roof and back wall.
        """
        top_rail_height = 0.06  # 60mm rails
        # Back wall extends to meet the top rail (similar to side walls)
        wall_height = height - top_rail_height / 2
        offset_z = wall_height / 2
        return self._create_corrugated_panel(width, wall_height, normal=(-1, 0, 0),
                                            offset=(x_pos, 0, offset_z),
                                            thickness=self.config.BACK_WALL_THICKNESS)
    
    def _create_roof(self, length: float, width: float, height: float) -> trimesh.Trimesh:
        """Create corrugated roof with slight arch for drainage.
        
        Roof panels are welded to top side rails and front/rear headers.
        The roof sits on top of the top side rails, creating a proper edge connection.
        """
        # Top rail height (60mm = 0.06m)
        top_rail_height = 0.06
        # Roof panel sits on top of the rails
        # Adjust width to account for rails, roof extends slightly over rails for proper connection
        roof_width = width  # Full width, will overlap rails slightly
        roof_z_position = height - top_rail_height / 2  # Sit on top of rails
        
        # Horizontal corrugation: swap dimensions so corrugation runs along length (horizontal)
        roof = self._create_corrugated_panel(length, roof_width, normal=(0, 0, 1),
                                            offset=(0, 0, roof_z_position), shallow=True,
                                            thickness=self.config.ROOF_THICKNESS)
        
        # Rotate roof panel 90 degrees around Z-axis so corrugation runs horizontally (across width)
        # This ensures the corrugation runs perpendicular to the container length
        rotation_z = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        roof.apply_transform(rotation_z)
        
        # Add slight arch for water drainage (center is slightly higher)
        vertices = roof.vertices.copy()
        center_x = 0
        center_y = 0
        max_dist = np.sqrt((length/2)**2 + (width/2)**2)
        
        for i, v in enumerate(vertices):
            dist_from_center = np.sqrt((v[0] - center_x)**2 + (v[1] - center_y)**2)
            arch_factor = 1.0 - (dist_from_center / max_dist) ** 2
            vertices[i, 2] += arch_factor * self.config.ROOF_ARCH_HEIGHT
        
        roof.vertices = vertices
        return roof
    
    def _create_roof_details(self, length: float, width: float, height: float) -> List[trimesh.Trimesh]:
        """Create roof ventilation and structural details."""
        parts = []
        vent_radius = self.config.VENTILATION_HOLE_RADIUS
        vent_color = (40, 40, 40, 255)  # Dark gray for ventilation
        
        # Add ventilation holes along the roof (simplified as cylinders)
        num_vents = 4
        vent_spacing = length / (num_vents + 1)
        
        for i in range(1, num_vents + 1):
            x_pos = -length / 2 + i * vent_spacing
            for y_offset in (-width / 3, width / 3):
                # Ventilation cover (raised ring)
                vent_cover = _cylinder(vent_radius + 0.02, 0.01, color=vent_color)
                vent_cover.apply_translation((x_pos, y_offset, height + 0.01))
                parts.append(vent_cover)
        
        return parts
    
    def _create_floor(self, length: float, width: float) -> trimesh.Trimesh:
        """Create corrugated floor base (dark brown)."""
        # Floor uses plywood thickness, not steel panel thickness
        # The corrugated panel represents the steel base, but actual floor thickness is plywood
        floor_panel = self._create_corrugated_panel(
            width, length, normal=(0, 0, -1),
            offset=(0, 0, self.config.FLOOR_THICKNESS / 2),
            shallow=True,
            thickness=0.002  # Steel base layer under plywood
        )
        dark_brown_rgba = (80, 50, 20, 255)
        return _colored(floor_panel, dark_brown_rgba)
    
    def _create_floor_details(self, length: float, width: float) -> List[trimesh.Trimesh]:
        """Create realistic floor with wooden plank pattern."""
        parts = []
        plank_w = self.config.FLOOR_PLANK_WIDTH
        gap = self.config.FLOOR_PLANK_GAP
        floor_z = self.config.FLOOR_THICKNESS
        plank_thickness = 0.025
        
        # Wood colors (varying shades for realism)
        wood_colors = [
            (70, 45, 25, 255),  # Dark wood
            (85, 55, 30, 255),  # Medium wood
            (95, 65, 35, 255),  # Light wood
        ]
        
        # Create planks running along the length (across width)
        num_planks = int(width / (plank_w + gap))
        current_y = -width / 2
        
        for i in range(num_planks):
            plank_width_actual = min(plank_w, width / 2 - current_y)
            if plank_width_actual <= 0:
                break
            
            # Vary wood color slightly for realism
            wood_color = wood_colors[i % len(wood_colors)]
            
            plank = _box((length, plank_width_actual, plank_thickness), color=wood_color)
            plank.apply_translation((0, current_y + plank_width_actual / 2, floor_z + plank_thickness / 2))
            parts.append(plank)
            
            current_y += plank_width_actual + gap
        
        return parts
    
    # ------------------------------------------------------------------
    # Structural details
    # ------------------------------------------------------------------
    def _create_corner_castings(self, length: float, width: float, height: float) -> List[trimesh.Trimesh]:
        """Create detailed corner castings with apertures for twist locks."""
        cc_l, cc_w, cc_h = 0.178, 0.162, 0.118
        casting_color = (200, 200, 50, 255)  # ISO standard yellow/gold color
        aperture_size = self.config.CORNER_CASTING_APERTURE_SIZE
        aperture_depth = self.config.CORNER_CASTING_APERTURE_DEPTH
        castings = []
        
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (0, 1):
                    # Main casting body
                    casting = _box((cc_l, cc_w, cc_h), color=casting_color)
                    z_pos = sz * (height - cc_h) + cc_h / 2
                    x_pos = sx * (length / 2 - cc_l / 2)
                    y_pos = sy * (width / 2 - cc_w / 2)
                    casting.apply_translation((x_pos, y_pos, z_pos))
                    castings.append(casting)
                    
                    # Create aperture (simplified as a recessed box)
                    # Real apertures are complex 3D shapes, but we'll represent them
                    aperture = _box((aperture_size, aperture_size, aperture_depth), 
                                   color=(50, 50, 50, 255))
                    # Position aperture at the corner face
                    if sx == 1:  # Front/back faces
                        aperture.apply_translation((x_pos + cc_l/2 - aperture_depth/2, y_pos, z_pos))
                    else:  # Back face
                        aperture.apply_translation((x_pos - cc_l/2 + aperture_depth/2, y_pos, z_pos))
                    castings.append(aperture)
        
        return castings
    
    def _create_side_rails(self, length: float, width: float, height: float) -> List[trimesh.Trimesh]:
        """Create top and bottom side rails (ISO standard: 60x60mm square steel pipes)."""
        # ISO standard top side rail: 60x60mm square steel pipe (0.06m x 0.06m)
        rail_t = 0.06  # 60mm square rails
        rail_color = (90, 90, 95, 255)  # Dark metallic gray for structural rails
        rails = []
        
        # Top rails - positioned at the top edge where roof and side panels meet
        # These serve as the connection point between roof and side panels
        for y in (-width / 2 + rail_t / 2, width / 2 - rail_t / 2):
            rail = _box((length, rail_t, rail_t), color=rail_color)
            # Position rail so top edge aligns with container height
            rail.apply_translation((0, y, height - rail_t / 2))
            rails.append(rail)
        
        # Bottom rails
        for y in (-width / 2 + rail_t / 2, width / 2 - rail_t / 2):
            rail = _box((length, rail_t, rail_t), color=rail_color)
            rail.apply_translation((0, y, rail_t / 2))
            rails.append(rail)
        
        return rails
    
    def _create_roof_side_connections(self, length: float, width: float, height: float) -> List[trimesh.Trimesh]:
        """Create connection details where roof panels meet side panels at top side rails.
        
        In real containers, roof panels are continuously welded to top side rails.
        This method adds visual details to represent the welded connection.
        """
        parts = []
        top_rail_height = 0.06  # 60mm rails
        rail_t = top_rail_height
        
        # Welding seam color - slightly darker/metallic to represent welded joint
        weld_color = (70, 70, 75, 255)  # Dark metallic gray for welding seams
        weld_width = 0.005  # 5mm weld bead width
        weld_height = 0.002  # 2mm weld bead height
        
        # Create welding seams along top rails where roof meets side panels
        # These represent continuous welding along the connection
        for y_side in (-1, 1):
            y_pos = y_side * (width / 2 - rail_t / 2)
            
            # Welding seam along the top edge of the rail (where roof sits)
            # This represents the continuous weld between roof panel and top rail
            weld_seam = _box((length, weld_width, weld_height), color=weld_color)
            weld_seam.apply_translation((0, y_pos, height - rail_t / 2 + weld_height / 2))
            parts.append(weld_seam)
            
            # Additional connection detail: small reinforcement at corners
            # These represent reinforcement plates at corner castings
            corner_reinforcement_size = 0.08
            corner_reinforcement_thickness = 0.003
            
            for x_side in (-1, 1):
                x_pos = x_side * (length / 2 - corner_reinforcement_size / 2)
                # Small reinforcement plate at corner
                corner_plate = _box((corner_reinforcement_size, corner_reinforcement_thickness, 
                                     corner_reinforcement_size), color=weld_color)
                corner_plate.apply_translation((x_pos, y_pos, height - rail_t / 2))
                parts.append(corner_plate)
        
        # Front and rear headers (where roof meets front/back walls)
        # These are part of the end frames but ensure proper roof connection
        header_t = 0.10  # Header thickness (from end frame)
        header_color = (90, 90, 95, 255)
        
        # Front header (at door end)
        front_header = _box((header_t, width, rail_t), color=header_color)
        front_header.apply_translation((length / 2 - header_t / 2, 0, height - rail_t / 2))
        parts.append(front_header)
        
        # Rear header (at back wall)
        rear_header = _box((header_t, width, rail_t), color=header_color)
        rear_header.apply_translation((-length / 2 + header_t / 2, 0, height - rail_t / 2))
        parts.append(rear_header)
        
        return parts
    
    def _create_roof_cross_members(self, length: float, width: float, height: float) -> List[trimesh.Trimesh]:
        """Create roof structural cross members."""
        spacing = 0.6
        num = int(np.floor(length / spacing))
        bar_t = 0.03
        bar_color = (90, 90, 90, 255)
        members = []
        
        for i in range(1, num):
            x = -length / 2 + i * spacing
            bar = _box((bar_t, width - 0.2, bar_t), color=bar_color)
            bar.apply_translation((x, 0, height - bar_t / 2))
            members.append(bar)
        
        return members
    
    def _create_floor_cross_members(self, length: float, width: float) -> List[trimesh.Trimesh]:
        """Create floor structural cross members."""
        spacing = 0.305
        num = int(np.floor(length / spacing))
        rail_t = 0.04
        rail_color = (80, 80, 80, 255)
        rails = []
        
        for i in range(num + 1):
            x = -length / 2 + i * spacing
            rail = _box((rail_t, width - 0.2, rail_t), color=rail_color)
            rail.apply_translation((x, 0, rail_t / 2))
            rails.append(rail)
        
        return rails
    
    def _create_back_wall_frame(self, width: float, height: float, x_pos: float) -> List[trimesh.Trimesh]:
        """Create structural frame for back wall with vertical posts, horizontal bars, and L-brackets."""
        parts = []
        # Hard metal color - darker metallic gray for structural posts
        frame_color = (80, 80, 85, 255)  # Dark metallic gray for hard metal
        post_size = 0.08  # Size of cuboid posts (was radius, now full dimension)
        bar_radius = 0.03   # Radius of horizontal bars
        bracket_size = 0.06  # Size of L-shaped brackets
        floor_z = self.config.FLOOR_THICKNESS
        wall_thickness = self.config.WALL_THICKNESS
        
        # Position frame slightly forward from back wall (on interior side)
        frame_x = x_pos + wall_thickness + post_size / 2
        
        # Vertical posts at corners (cuboid/box shape for hard metal appearance)
        for y_side in (-1, 1):
            y_pos = y_side * (width / 2 - post_size / 2)
            # Full height post - cuboid shape instead of cylinder
            post = _box((post_size, post_size, height - floor_z), color=frame_color)
            post.apply_translation((frame_x, y_pos, floor_z + (height - floor_z) / 2))
            parts.append(post)
        
        # Horizontal bars at top and bottom
        bar_length = width - 2 * (post_size / 2)
        
        # Top horizontal bar
        top_bar = _cylinder(bar_radius, bar_length, sections=16, color=frame_color)
        # Rotate to horizontal (along Y axis)
        rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        top_bar.apply_transform(rotation)
        top_bar.apply_translation((frame_x, 0, height - bar_radius))
        parts.append(top_bar)
        
        # Bottom horizontal bar
        bottom_bar = _cylinder(bar_radius, bar_length, sections=16, color=frame_color)
        bottom_bar.apply_transform(rotation)
        bottom_bar.apply_translation((frame_x, 0, floor_z + bar_radius))
        parts.append(bottom_bar)
        
        # L-shaped brackets connecting bars to posts
        bracket_color = (70, 70, 75, 255)  # Darker metallic for brackets
        bracket_thickness = 0.015
        
        for y_side in (-1, 1):
            y_pos = y_side * (width / 2 - post_size / 2)
            
            # Top bracket - L-shaped (vertical arm + horizontal arm)
            # Vertical arm (along post)
            top_bracket_vert = _box((bracket_size, bracket_thickness, bracket_size * 0.8), color=bracket_color)
            top_bracket_vert.apply_translation((frame_x - bracket_size/2, y_pos, height - bracket_size * 0.4))
            parts.append(top_bracket_vert)
            
            # Horizontal arm (along bar)
            top_bracket_horiz = _box((bracket_thickness, bracket_size * 0.8, bracket_size), color=bracket_color)
            top_bracket_horiz.apply_translation((frame_x - bracket_thickness/2, y_pos, height - bracket_size/2))
            parts.append(top_bracket_horiz)
            
            # Bottom bracket - L-shaped
            # Vertical arm (along post)
            bottom_bracket_vert = _box((bracket_size, bracket_thickness, bracket_size * 0.8), color=bracket_color)
            bottom_bracket_vert.apply_translation((frame_x - bracket_size/2, y_pos, floor_z + bracket_size * 0.4))
            parts.append(bottom_bracket_vert)
            
            # Horizontal arm (along bar)
            bottom_bracket_horiz = _box((bracket_thickness, bracket_size * 0.8, bracket_size), color=bracket_color)
            bottom_bracket_horiz.apply_translation((frame_x - bracket_thickness/2, y_pos, floor_z + bracket_size/2))
            parts.append(bottom_bracket_horiz)
        
        return parts
    
    # ------------------------------------------------------------------
    # Doors and locking mechanisms
    # ------------------------------------------------------------------
    def _create_doors_and_frame(self, length: float, width: float, height: float,
                                door_opening: Tuple[float, float]) -> List[trimesh.Trimesh]:
        """Create detailed door system with frames, hinges, seals, and locking mechanisms."""
        frame_t = 0.10
        door_th = self.config.DOOR_THICKNESS
        floor_z = self.config.FLOOR_THICKNESS
        frame_x = length / 2 - frame_t / 2
        frame_color = (90, 90, 90, 255)
        gasket_color = (30, 30, 30, 255)  # Dark rubber color
        hinge_color = (120, 120, 120, 255)
        
        parts = []
        
        # Front frame
        parts.extend(self._create_end_frame(frame_x, width, height, floor_z, frame_t, frame_color))
        
        # Door panels
        door_panel_w = (width - 2 * frame_t) / 2
        door_panel_h = height - frame_t - floor_z
        door_x_pos = frame_x - frame_t / 2 - door_th / 2
        door_base_z = floor_z
        
        for side in (-1, 1):
            door_center_y = side * (door_panel_w / 2)
            door_panel_center_z = door_base_z + door_panel_h / 2
            
            # Door hinges (robust hinges on outer edge)
            hinge_y = door_center_y + side * (door_panel_w / 2 - 0.05)
            hinge_x = door_x_pos - door_th / 2  # Hinge pivot point X position
            
            # Door panel
            door_panel = self._create_corrugated_panel(
                dim_a=door_panel_w, dim_b=door_panel_h,
                normal=(1, 0, 0),
                offset=(door_x_pos, door_center_y, door_panel_center_z),
                shallow=False,
                thickness=self.config.DOOR_THICKNESS
            )
            
            # Apply rotation to open door slightly around vertical hinge axis (Z-axis)
            # Coordinate system: X=length (front/back), Y=width (left/right), Z=height (up/down)
            # Doors are at front (positive X), normal=(1,0,0) facing outward
            # To open OUTWARD: doors swing away from container center
            # Left door (side=-1): hinge on left edge, swings right (positive Y) = CCW = positive angle
            # Right door (side=1): hinge on right edge, swings left (negative Y) = CW = negative angle
            # Try opposite: left=negative, right=positive
            open_angle = self.config.DOOR_OPEN_ANGLE * side  # left=negative, right=positive
            
            # Translate to hinge point, rotate around Z-axis (vertical), translate back
            door_panel.apply_translation((-hinge_x, -hinge_y, -door_panel_center_z))
            rotation_matrix = trimesh.transformations.rotation_matrix(open_angle, [0, 0, 1])
            door_panel.apply_transform(rotation_matrix)
            door_panel.apply_translation((hinge_x, hinge_y, door_panel_center_z))
            
            parts.append(door_panel)
            
            parts.extend(self._create_door_hinges(
                door_panel_h, door_x_pos, hinge_y, door_base_z, hinge_color
            ))
            
            # Door seals/gaskets (rubber seals around perimeter)
            seal_th = self.config.DOOR_SEAL_THICKNESS
            door_seal_parts = []
            # Top and bottom seals
            for z_mult in (0, 1):
                z_pos = door_base_z + z_mult * door_panel_h
                gasket = _box((door_th + seal_th, door_panel_w + seal_th * 2, seal_th), 
                             color=gasket_color)
                gasket.apply_translation((door_x_pos - seal_th/2, door_center_y, z_pos))
                door_seal_parts.append(gasket)
            
            # Side seals
            for y_mult in (-1, 1):
                y_pos = door_center_y + y_mult * door_panel_w / 2
                gasket = _box((door_th + seal_th, seal_th, door_panel_h), 
                             color=gasket_color)
                gasket.apply_translation((door_x_pos - seal_th/2, y_pos, door_panel_center_z))
                door_seal_parts.append(gasket)
            
            # Apply rotation to door seals (same as door panel)
            for seal_part in door_seal_parts:
                seal_part.apply_translation((-hinge_x, -hinge_y, -door_panel_center_z))
                seal_part.apply_transform(rotation_matrix)
                seal_part.apply_translation((hinge_x, hinge_y, door_panel_center_z))
                parts.append(seal_part)
            
            # Locking assemblies - two vertical bars per door (as seen in real containers)
            # Position bars symmetrically on each door
            bar_positions = [-0.30, 0.30]  # Relative positions from door center
            for bar_offset_ratio in bar_positions:
                rod_y = door_center_y + bar_offset_ratio * door_panel_w
                assembly = self._create_locking_assembly(door_panel_h, door_x_pos, rod_y,
                                                        door_base_z, frame_x, door_panel_w)
                # Apply rotation to locking assemblies (same as door panel)
                for assembly_part in assembly:
                    assembly_part.apply_translation((-hinge_x, -hinge_y, -door_panel_center_z))
                    assembly_part.apply_transform(rotation_matrix)
                    assembly_part.apply_translation((hinge_x, hinge_y, door_panel_center_z))
                    parts.append(assembly_part)
        
        return parts
    
    def _create_door_hinges(self, door_h: float, door_x: float, hinge_y: float,
                            base_z: float, hinge_color: Tuple[int, int, int, int]) -> List[trimesh.Trimesh]:
        """Create robust door hinges."""
        parts = []
        hinge_count = self.config.DOOR_HINGE_COUNT
        hinge_spacing = door_h / (hinge_count + 1)
        hinge_radius = 0.025
        hinge_length = 0.08
        
        for i in range(1, hinge_count + 1):
            z_pos = base_z + i * hinge_spacing
            
            # Hinge pin (vertical cylinder)
            pin = _cylinder(hinge_radius, hinge_length, color=hinge_color)
            pin.apply_translation((door_x - self.config.DOOR_THICKNESS/2, hinge_y, z_pos))
            parts.append(pin)
            
            # Hinge plates (simplified as boxes)
            plate = _box((0.01, hinge_length * 0.6, hinge_length), color=hinge_color)
            plate.apply_translation((door_x - self.config.DOOR_THICKNESS/2 - 0.02, hinge_y, z_pos))
            parts.append(plate)
        
        return parts
    
    def _create_end_frame(self, x_pos: float, width: float, height: float,
                         floor_z: float, frame_t: float, frame_color) -> List[trimesh.Trimesh]:
        """Create structural end frame."""
        parts = []
        
        # Header
        header = _box((frame_t, width, frame_t), color=frame_color)
        header.apply_translation((x_pos, 0, height - frame_t / 2))
        parts.append(header)
        
        # Sill
        sill = _box((frame_t, width, frame_t), color=frame_color)
        sill.apply_translation((x_pos, 0, floor_z / 2))
        parts.append(sill)
        
        # Posts
        post_h = height - (floor_z / 2) - (frame_t / 2)
        post_center_z = (height + floor_z) / 2
        for y_mult in (-1, 1):
            y_pos = y_mult * (width / 2 - frame_t / 2)
            post = _box((frame_t, frame_t, post_h), color=frame_color)
            post.apply_translation((x_pos, y_pos, post_center_z))
            parts.append(post)
        
        return parts
    
    def _create_locking_assembly(self, door_h: float, door_x: float, rod_y: float,
                                 base_z: float, frame_x: float, door_w: float) -> List[trimesh.Trimesh]:
        """Create detailed locking bar assembly with cam retainers - matching real container doors."""
        rod_radius = 0.020  # Thicker bars for more prominence
        rod_h = door_h * 0.92  # Bars extend almost full height
        bar_color = (150, 150, 150, 255)  # Metallic gray
        handle_color = (180, 180, 180, 255)  # Lighter gray for handles
        cam_color = (130, 130, 130, 255)  # Darker gray for cams
        
        parts = []
        rod_x_pos = door_x + self.config.DOOR_THICKNESS / 2 + rod_radius + 0.01
        
        # Main vertical locking rod (more prominent)
        rod = _cylinder(rod_radius, rod_h, sections=16, color=bar_color)
        rod_center_z = base_z + door_h / 2
        rod.apply_translation((rod_x_pos, rod_y, rod_center_z))
        parts.append(rod)
        
        # Handle (operating lever) - attached directly to the rod at lower third
        # The handle rotates with the rod to operate the cam locks
        handle_mount_z = base_z + door_h * 0.35  # About 1/3 from bottom - ergonomic height
        handle_length = 0.20  # Length of handle extending from rod
        handle_width = 0.05   # Width of handle
        handle_thickness = 0.04  # Thickness of handle
        
        # Handle attachment point (connects handle to rod)
        handle_attachment = _box((rod_radius * 2.5, rod_radius * 2.5, rod_radius * 3), color=bar_color)
        handle_attachment.apply_translation((rod_x_pos, rod_y, handle_mount_z))
        parts.append(handle_attachment)
        
        # Main handle lever (extends horizontally from rod for gripping and rotation)
        handle_body = _box((handle_length, handle_width, handle_thickness), color=handle_color)
        # Position handle so it extends from the rod outward
        handle_body.apply_translation((rod_x_pos + handle_length/2, rod_y, handle_mount_z))
        parts.append(handle_body)
        
        # Handle grip end (vertical grip at the end of handle for better grip)
        grip_height = handle_thickness * 2.5
        handle_grip = _box((handle_width * 1.2, handle_width * 1.2, grip_height), color=handle_color)
        handle_grip.apply_translation((rod_x_pos + handle_length, rod_y, handle_mount_z))
        parts.append(handle_grip)
        
        # Cam locks at top and bottom (lock into frame slots)
        # These rotate with the rod when handle is turned
        cam_width = rod_radius * 6
        cam_depth = rod_radius * 4
        cam_height = rod_radius * 5
        
        for z_mult in (-1, 1):
            z_pos = rod_center_z + z_mult * (rod_h / 2 - rod_radius * 1.5)
            
            # Cam lock body (rectangular, rotates with rod to lock/unlock)
            cam = _box((cam_depth, cam_width, cam_height), color=cam_color)
            cam.apply_translation((rod_x_pos, rod_y, z_pos))
            parts.append(cam)
            
            # Cam lock engagement point on frame (where cam locks into when rotated)
            retainer_slot = _box((rod_radius * 2, cam_width * 1.2, cam_height * 0.8), color=(100, 100, 100, 255))
            retainer_slot.apply_translation((frame_x - 0.015, rod_y, z_pos))
            parts.append(retainer_slot)
        
        return parts
    
    # ------------------------------------------------------------------
    # Additional realistic details
    # ------------------------------------------------------------------
    def _create_identification_plates(self, length: float, width: float, height: float) -> List[trimesh.Trimesh]:
        """Create identification plates and markings."""
        parts = []
        plate_color = (240, 240, 240, 255)  # White/light gray
        
        # Front door identification plate (simplified as a box)
        plate_thickness = 0.005
        plate_width = 0.8
        plate_height = 0.2
        
        plate = _box((plate_thickness, plate_width, plate_height), color=plate_color)
        plate.apply_translation((length / 2 - 0.01, 0, height - 0.4))
        parts.append(plate)
        
        # Side identification plate
        side_plate = _box((plate_thickness, plate_height, 0.6), color=plate_color)
        side_plate.apply_translation((0, width / 2 + 0.001, height / 2))
        parts.append(side_plate)
        
        return parts
    
    def _create_forklift_pockets(self, length: float, width: float) -> List[trimesh.Trimesh]:
        """Create forklift pockets on the bottom of the container."""
        parts = []
        pocket_color = (60, 60, 60, 255)
        pocket_width = 0.15
        pocket_depth = 0.20
        pocket_height = 0.10
        
        # Forklift pockets are typically on the sides
        num_pockets = 2
        pocket_spacing = length / (num_pockets + 1)
        
        for i in range(1, num_pockets + 1):
            x_pos = -length / 2 + i * pocket_spacing
            for y_side in (-1, 1):
                y_pos = y_side * (width / 2 - pocket_width / 2)
                
                # Create recessed pocket
                pocket = _box((pocket_depth, pocket_width, pocket_height), color=pocket_color)
                pocket.apply_translation((x_pos, y_pos, pocket_height / 2))
                parts.append(pocket)
        
        return parts
    
    def _create_interior_tie_downs(self, length: float, width: float, height: float) -> List[trimesh.Trimesh]:
        """Create interior tie-down points for securing cargo."""
        parts = []
        tie_down_color = (100, 100, 100, 255)
        tie_down_radius = 0.02
        tie_down_height = 0.05
        spacing = self.config.TIE_DOWN_SPACING
        floor_z = self.config.FLOOR_THICKNESS
        
        # Create tie-down rings along the floor only
        num_tie_downs_x = int(length / spacing)
        num_tie_downs_y = int(width / spacing)
        
        for i in range(1, num_tie_downs_x):
            for j in range(1, num_tie_downs_y):
                x_pos = -length / 2 + i * spacing
                y_pos = -width / 2 + j * spacing
                
                # Tie-down ring (simplified as a cylinder)
                ring = _cylinder(tie_down_radius, tie_down_height, color=tie_down_color)
                ring.apply_translation((x_pos, y_pos, floor_z + tie_down_height / 2))
                parts.append(ring)
        
        return parts
    
    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _pick_body_color(self) -> str:
        """Pick a random color based on probabilities."""
        colors = list(self.config.CONTAINER_COLORS.keys())
        return np.random.choice(colors, p=self.config.COLOR_PROBABILITIES)


# -----------------------------------------------------------------------------
# Cleanup functions
# -----------------------------------------------------------------------------
def cleanup_complete_containers():
    """Clean up all complete container outputs before creating new ones"""
    print("\n🧹 Cleaning up previous complete container outputs...")
    
    # Directory to clean
    complete_dir = Path("complete_containers")
    
    files_removed = 0
    dirs_removed = 0
    
    # Remove complete_containers directory and all contents
    if complete_dir.exists():
        try:
            shutil.rmtree(complete_dir)
            print(f"  ✓ Removed directory: {complete_dir}")
            dirs_removed += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {complete_dir}: {e}")
    
    # Remove any stray container .obj files in root directory
    stray_files = [
        'container_20ft_red.obj',
        'container_20ft_blue.obj',
        'shipping_container.obj'
    ]
    
    for filename in stray_files:
        if Path(filename).exists():
            try:
                os.remove(filename)
                print(f"  ✓ Removed: {filename}")
                files_removed += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {filename}: {e}")
    
    print(f"  📊 Complete container cleanup: {files_removed} files, {dirs_removed} directories removed")
    return files_removed + dirs_removed > 0


# -----------------------------------------------------------------------------
# Main function for testing
# -----------------------------------------------------------------------------
def main():
    """Generate complete shipping containers."""
    # Clean up previous outputs first
    cleanup_complete_containers()
    
    # Create output directory
    output_dir = Path("complete_containers")
    output_dir.mkdir(exist_ok=True)
    print(f"Created folder: {output_dir}")
    
    # Get user input for number of containers
    while True:
        try:
            num_containers = int(input("\nEnter number of complete containers to generate (1-500): ").strip())
            if 1 <= num_containers <= 500:
                break
            else:
                print("Please enter a number between 1 and 500")
        except ValueError:
            print("Please enter a valid number")
    
    # Get user input for container type
    print("\nSelect container type:")
    print("1. 20ft container")
    print("2. 40ft container")
    print("3. 40ft High Cube container")
    print("4. Random (mix of all types)")
    
    while True:
        container_choice = input("Enter choice (1-4): ").strip()
        if container_choice in ['1', '2', '3', '4']:
            break
        else:
            print("Please enter a valid choice (1-4)")
    
    # Map choice to container type
    container_types_map = {
        '1': ['20ft'],
        '2': ['40ft'],
        '3': ['40ft_hc'],
        '4': ['20ft', '40ft', '40ft_hc']
    }
    container_types = container_types_map[container_choice]
    
    generator = ShippingContainerGenerator()
    
    print("\n" + "=" * 60)
    print(f"Generating {num_containers} complete container(s)...")
    print("=" * 60)
    
    generated_count = 0
    
    for i in range(num_containers):
        # Select container type (random if multiple types available)
        if len(container_types) > 1:
            container_type = random.choice(container_types)
        else:
            container_type = container_types[0]
        
        # Generate with random color
        container_num = i + 1
        output_filename = f"container_{container_type}_{container_num:04d}.obj"
        
        print(f"\n[{container_num}/{num_containers}] Generating {container_type} container...")
        try:
            generator.generate(
                output_path=output_filename,
                container_type=container_type,
                color=None  # None means random color
            )
            generated_count += 1
            print(f"  ✓ Container {container_num} generated successfully")
        except Exception as e:
            print(f"  ✗ Failed to generate container {container_num}: {e}")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Successfully generated {generated_count} out of {num_containers} container(s)")
    print(f"All files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

