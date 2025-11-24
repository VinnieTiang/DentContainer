#!/usr/bin/env python3
"""
Container Dent Generator

Adds realistic dents to container models with physics-based logic:
- Random sizes and locations
- Multiple dent types (circular, elliptical, crease, edge)
- Realistic placement (edges, corners, surfaces)
- Smooth falloff for natural appearance
- Respects container structure (avoids interior, preserves functionality)

Dent types:
1. Circular dents: Impact from round objects (forklift wheels, cargo)
2. Elliptical dents: Oblique impacts or sliding collisions
3. Crease dents: Edge impacts (most common in real containers)
4. Corner dents: Corner-to-corner impacts during stacking
"""

import numpy as np
import trimesh
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import random
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DentGenerator:
    """Generate realistic dents on container meshes."""
    
    def __init__(self, mesh: trimesh.Trimesh, highlight_color: Tuple[int, int, int, int] = None):
        """
        Initialize dent generator with a container mesh.
        
        Args:
            mesh: trimesh.Trimesh object of the container
            highlight_color: RGBA color for highlighting dents (default: darker red/orange)
        """
        self.mesh = mesh.copy()
        self.original_vertices = self.mesh.vertices.copy()
        self.bounds = self.mesh.bounds
        self.center = self.mesh.centroid
        
        # Container dimensions (approximate from bounds)
        self.length = self.bounds[1][0] - self.bounds[0][0]
        self.width = self.bounds[1][1] - self.bounds[0][1]
        self.height = self.bounds[1][2] - self.bounds[0][2]
        
        # Initialize vertex colors if not present
        if not hasattr(self.mesh.visual, 'vertex_colors') or self.mesh.visual.vertex_colors is None:
            # Default to gray if no colors exist
            self.mesh.visual.vertex_colors = np.tile([128, 128, 128, 255], (len(self.mesh.vertices), 1))
        
        # Store original colors
        self.original_colors = self.mesh.visual.vertex_colors.copy()
        
        # Set highlight color for dents (darker red/orange by default)
        if highlight_color is None:
            self.highlight_color = np.array([180, 60, 40, 255])  # Dark red/orange for dents
        else:
            self.highlight_color = np.array(highlight_color)
        
        # Track dented vertices
        self.dented_vertices = np.zeros(len(self.mesh.vertices), dtype=bool)
        
        # Track dent specifications for metadata
        self.dent_specs_list = []
        
        # Identify structural elements (rails, bars, brackets) by color
        # Structural elements typically have gray colors (80-180 range)
        self.structural_mask = self._identify_structural_elements()
        
        logger.info(f"Container dimensions: {self.length:.2f}m × {self.width:.2f}m × {self.height:.2f}m")
        logger.info(f"Structural elements identified: {np.sum(self.structural_mask)} vertices")
    
    def _highlight_dented_vertices(self, mask: np.ndarray, normalized_dist: np.ndarray):
        """
        Highlight dented vertices by changing their colors.
        Deeper dents (closer to center) get more intense highlighting.
        
        Args:
            mask: Boolean array indicating which vertices are dented
            normalized_dist: Normalized distance from dent center (0 = center, 1 = edge)
        """
        if not np.any(mask):
            return
        
        # Mark vertices as dented
        self.dented_vertices[mask] = True
        
        # Get current colors for affected vertices
        colors = self.mesh.visual.vertex_colors[mask].copy()
        
        # Calculate highlight intensity based on depth (closer to center = more intense)
        # Use inverse of normalized distance (1.0 at center, 0.0 at edge)
        intensity = 1.0 - normalized_dist
        
        # Blend original color with highlight color
        # More intense at center, fading to original at edges
        for i, (orig_color, intensity_val) in enumerate(zip(colors, intensity)):
            # Blend between original color and highlight color
            blend_factor = intensity_val * 0.7  # 70% max blend for visibility
            highlighted_color = (
                orig_color * (1 - blend_factor) + 
                self.highlight_color * blend_factor
            ).astype(np.uint8)
            colors[i] = highlighted_color
        
        # Apply colors
        self.mesh.visual.vertex_colors[mask] = colors
    
    def _get_surface_normal(self, vertex: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Estimate surface normal at a vertex using physics-based approach.
        ALWAYS returns outward-pointing normal for proper dent direction.
        
        Args:
            vertex: Vertex position
            k: Number of nearest neighbors to consider
            
        Returns:
            Normalized OUTWARD-pointing surface normal vector
        """
        # Find nearest vertices
        distances = np.linalg.norm(self.mesh.vertices - vertex, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        
        # Get faces containing these vertices
        face_mask = np.any(np.isin(self.mesh.faces, nearest_indices), axis=1)
        if not np.any(face_mask):
            # Fallback: use direction from container center to point
            normal = vertex - self.center
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            return normal
        
        # Average face normals (weighted by face area for better accuracy)
        face_normals = self.mesh.face_normals[face_mask]
        
        # Weight by distance (closer faces have more influence)
        face_centers = np.mean(self.mesh.vertices[self.mesh.faces[face_mask]], axis=1)
        face_distances = np.linalg.norm(face_centers - vertex, axis=1)
        weights = 1.0 / (face_distances + 1e-6)
        weights = weights / np.sum(weights)
        
        # Weighted average of normals
        normal = np.sum(face_normals * weights[:, np.newaxis], axis=0)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        
        # CRITICAL: Ensure normal points OUTWARD from container center
        # This prevents spikes from pointing inward or in wrong directions
        to_center = self.center - vertex
        if np.dot(normal, to_center) > 0:
            # Normal is pointing inward, flip it
            normal = -normal
        
        return normal
    
    def _is_on_edge(self, vertex: np.ndarray, threshold: float = 0.05) -> bool:
        """
        Check if a vertex is near an edge of the container.
        
        Args:
            vertex: Vertex position
            threshold: Distance threshold (in meters) to consider as edge
            
        Returns:
            True if vertex is near an edge
        """
        # Check proximity to bounding box edges
        min_bounds = self.bounds[0]
        max_bounds = self.bounds[1]
        
        # Check each axis
        for i in range(3):
            dist_to_min = abs(vertex[i] - min_bounds[i])
            dist_to_max = abs(vertex[i] - max_bounds[i])
            if dist_to_min < threshold or dist_to_max < threshold:
                return True
        
        return False
    
    def _is_on_floor(self, vertex: np.ndarray, threshold: float = 0.05) -> bool:
        """
        Check if a vertex is on the floor panel.
        
        Args:
            vertex: Vertex position
            threshold: Distance threshold (in meters) to consider as floor
            
        Returns:
            True if vertex is on the floor
        """
        # Floor is at the minimum z-coordinate
        min_z = self.bounds[0][2]
        return abs(vertex[2] - min_z) < threshold
    
    def _is_on_corner(self, vertex: np.ndarray, threshold: float = 0.08) -> bool:
        """
        Check if a vertex is near a corner of the container.
        
        Args:
            vertex: Vertex position
            threshold: Distance threshold (in meters) to consider as corner
            
        Returns:
            True if vertex is near a corner
        """
        min_bounds = self.bounds[0]
        max_bounds = self.bounds[1]
        
        # Count how many axes are near boundaries
        near_boundary_count = 0
        for i in range(3):
            if abs(vertex[i] - min_bounds[i]) < threshold or abs(vertex[i] - max_bounds[i]) < threshold:
                near_boundary_count += 1
        
        return near_boundary_count >= 2
    
    def _identify_structural_elements(self) -> np.ndarray:
        """
        Identify structural elements (rails, bars, brackets) by their gray colors.
        
        Returns:
            Boolean mask indicating which vertices belong to structural elements
        """
        if not hasattr(self.mesh.visual, 'vertex_colors') or self.mesh.visual.vertex_colors is None:
            return np.zeros(len(self.mesh.vertices), dtype=bool)
        
        colors = self.mesh.visual.vertex_colors
        
        # Structural elements typically have gray colors (RGB values 80-180)
        # Check if all RGB channels are in the gray range
        gray_mask = (
            (colors[:, 0] >= 80) & (colors[:, 0] <= 180) &  # Red channel
            (colors[:, 1] >= 80) & (colors[:, 1] <= 180) &  # Green channel
            (colors[:, 2] >= 80) & (colors[:, 2] <= 180) &  # Blue channel
            (np.abs(colors[:, 0] - colors[:, 1]) < 30) &     # Similar R and G (grayish)
            (np.abs(colors[:, 1] - colors[:, 2]) < 30)      # Similar G and B (grayish)
        )
        
        return gray_mask
    
    def _apply_structural_bending(self, dent_center: np.ndarray, dent_radius: float, 
                                  dent_depth: float, dent_normal: np.ndarray):
        """
        Apply bending deformation to structural elements near a dent.
        Structural elements (rails, bars, brackets) bend when impacted.
        
        Args:
            dent_center: Center of the dent
            dent_radius: Radius of the dent
            dent_depth: Depth of the dent
            dent_normal: Normal direction of the dent
        """
        if not np.any(self.structural_mask):
            return
        
        vertices = self.mesh.vertices
        structural_vertices = vertices[self.structural_mask]
        
        # Find structural vertices near the dent
        distances = np.linalg.norm(structural_vertices - dent_center, axis=1)
        # Extend influence radius for structural elements (they bend further)
        influence_radius = dent_radius * 1.5
        
        near_mask = distances < influence_radius
        if not np.any(near_mask):
            return
        
        # Get indices of affected structural vertices
        structural_indices = np.where(self.structural_mask)[0]
        affected_indices = structural_indices[near_mask]
        affected_vertices = vertices[affected_indices]
        affected_distances = distances[near_mask]
        
        # Calculate bending displacement
        # Structural elements bend along their length, following the dent direction
        normalized_dist = affected_distances / influence_radius
        
        # Bending intensity decreases with distance
        bend_factor = np.exp(-2.0 * normalized_dist ** 2)
        
        # Calculate bending direction (perpendicular to structural element direction)
        # For structural elements, we want them to bend toward the dent center
        to_dent = dent_center - affected_vertices
        to_dent_norm = to_dent / (np.linalg.norm(to_dent, axis=1, keepdims=True) + 1e-10)
        
        # Blend between dent normal and direction to dent center
        # This creates a bending effect
        bend_direction = 0.6 * dent_normal + 0.4 * to_dent_norm.mean(axis=0)
        bend_direction = bend_direction / (np.linalg.norm(bend_direction) + 1e-10)
        
        # Apply bending displacement (less than panel dents, but still visible)
        bend_depth = dent_depth * 0.4  # Structural elements bend less than panels
        displacements = -bend_direction * (bend_depth * bend_factor[:, np.newaxis])
        
        # OPTIMIZATION: Pre-filter structural vertices to only those near affected area
        # This avoids recalculating distances for ALL structural vertices in each loop
        max_propagation_radius = 0.15  # 15cm connection distance
        # Expand search area to include propagation radius
        search_radius = influence_radius + max_propagation_radius
        search_distances = np.linalg.norm(structural_vertices - dent_center, axis=1)
        search_mask = search_distances < search_radius
        nearby_structural_indices = structural_indices[search_mask]
        nearby_structural_vertices = vertices[nearby_structural_indices]
        
        # Pre-calculate distances from affected vertices to nearby structural vertices
        # This is much more efficient than recalculating in the loop
        if len(nearby_structural_indices) > 0:
            # Propagate bending along structural elements
            # Find connected vertices along structural elements
            for i, vertex_idx in enumerate(affected_indices):
                # Find nearby structural vertices (along the same element)
                # Use pre-filtered structural vertices instead of all structural vertices
                vertex_pos = vertices[vertex_idx]
                nearby_distances = np.linalg.norm(nearby_structural_vertices - vertex_pos, axis=1)
                nearby_mask = nearby_distances < max_propagation_radius
                
                if np.sum(nearby_mask) > 1:
                    # Propagate some displacement to connected vertices
                    nearby_indices = nearby_structural_indices[nearby_mask]
                    # Remove the current vertex from nearby indices
                    nearby_indices = nearby_indices[nearby_indices != vertex_idx]
                    
                    if len(nearby_indices) > 0:
                        # Reduced displacement for propagated vertices
                        prop_factor = 0.3
                        prop_displacement = displacements[i] * prop_factor
                        
                        # Apply to nearby structural vertices (with distance falloff)
                        # Limit to 5 nearest to avoid excessive computation
                        # Calculate distances for nearby indices
                        nearby_dists = np.array([
                            np.linalg.norm(vertices[idx] - vertex_pos) 
                            for idx in nearby_indices
                        ])
                        
                        if len(nearby_dists) > 0:
                            # Get top 5 nearest
                            top_n = min(5, len(nearby_indices))
                            # Sort by distance and get top N indices
                            sorted_idx = np.argsort(nearby_dists)[:top_n]
                            top_nearby_indices = nearby_indices[sorted_idx]
                            top_nearby_dists = nearby_dists[sorted_idx]
                            
                            for nearby_idx, nearby_dist in zip(top_nearby_indices, top_nearby_dists):
                                falloff = np.exp(-nearby_dist / 0.1)  # 10cm falloff
                                vertices[nearby_idx] += prop_displacement * falloff
        
        # Apply main displacement
        vertices[affected_indices] += displacements
        
        # Update mesh
        self.mesh.vertices = vertices
    
    def _apply_floor_deformation_from_side_impact(self, dent_center: np.ndarray, dent_radius: float, 
                                                  dent_depth: float, dent_normal: np.ndarray):
        """
        Apply VERY subtle deformation to floor panel due to pressure from side panel impacts.
        
        PHYSICS LOGIC:
        - Floor does NOT get direct dents (excluded from dent placement)
        - Floor CAN deform minimally when nearby side panels are impacted
        - This simulates structural stress propagation, NOT visible dents
        - Deformation is VERY subtle (only 5% of panel dent depth, max 1cm)
        - Floor deformation should NOT be visible as major dents
        
        Args:
            dent_center: Center of the dent on side panel
            dent_radius: Radius of the dent
            dent_depth: Depth of the dent
            dent_normal: Normal direction of the dent
        """
        vertices = self.mesh.vertices
        
        # Find floor vertices
        floor_mask = np.array([self._is_on_floor(v) for v in vertices])
        if not np.any(floor_mask):
            return
        
        floor_vertices = vertices[floor_mask]
        
        # Only apply deformation if dent is on side panels (not roof/doors)
        # Check if dent is on side by checking if it's near side walls
        # Side walls are at y = ±width/2
        side_wall_threshold = self.width * 0.4  # Within 40% of width from center
        dent_y_distance_from_side = min(
            abs(dent_center[1] - (-self.width / 2)),
            abs(dent_center[1] - (self.width / 2))
        )
        
        # Only apply if dent is close to side walls (side panel impact)
        if dent_y_distance_from_side > side_wall_threshold:
            return  # Dent is on roof/doors/back, not side panels
        
        # Find floor vertices near the dent (projected onto floor plane)
        # Project dent center onto floor plane (z = min_z)
        floor_z = self.bounds[0][2]
        projected_center = np.array([dent_center[0], dent_center[1], floor_z])
        
        # Calculate horizontal distances from projected center to floor vertices
        floor_2d_positions = floor_vertices[:, :2]  # X, Y coordinates
        projected_center_2d = projected_center[:2]
        horizontal_distances = np.linalg.norm(floor_2d_positions - projected_center_2d, axis=1)
        
        # Influence radius: floor deformation extends further than dent radius
        # But only affects floor near the side where impact occurred
        influence_radius = dent_radius * 2.0  # Floor deformation spreads wider
        
        # Only affect floor vertices on the same side as the impact
        impact_side = 1 if dent_center[1] > 0 else -1
        floor_side_mask = np.sign(floor_vertices[:, 1]) == impact_side
        
        # Combine distance and side filters
        near_mask = (horizontal_distances < influence_radius) & floor_side_mask
        
        if not np.any(near_mask):
            return
        
        # Get indices of affected floor vertices
        floor_indices = np.where(floor_mask)[0]
        affected_indices = floor_indices[near_mask]
        affected_vertices = vertices[affected_indices]
        affected_distances = horizontal_distances[near_mask]
        
        # Calculate deformation intensity
        # VERY subtle - only structural stress propagation, NOT visible dents
        normalized_dist = affected_distances / influence_radius
        
        # Smooth falloff - floor deformation is gradual and minimal
        deformation_factor = np.exp(-3.0 * normalized_dist ** 2)
        
        # Floor deformation depth is VERY small - only structural stress, not visible dents
        # Represents subtle compression from side panel impacts
        floor_deformation_depth = dent_depth * 0.05  # Only 5% of panel dent depth (reduced from 15%)
        
        # Deformation direction: floor pushes DOWN slightly (compression from side impact)
        # Also slight horizontal component toward the impact point
        floor_normal = np.array([0, 0, -1])  # Downward (negative Z)
        
        # Add slight horizontal component toward impact
        to_impact = projected_center - affected_vertices
        to_impact_horizontal = to_impact.copy()
        to_impact_horizontal[:, 2] = 0  # Remove vertical component
        to_impact_horizontal_norm = to_impact_horizontal / (np.linalg.norm(to_impact_horizontal, axis=1, keepdims=True) + 1e-10)
        
        # Blend: mostly downward, slight horizontal pull
        deformation_direction = 0.85 * floor_normal + 0.15 * to_impact_horizontal_norm.mean(axis=0)
        deformation_direction = deformation_direction / (np.linalg.norm(deformation_direction) + 1e-10)
        
        # Apply subtle deformation
        displacements = -deformation_direction * (floor_deformation_depth * deformation_factor[:, np.newaxis])
        
        # PHYSICS CONSTRAINT: Limit floor deformation to prevent unrealistic effects
        # Absolute maximum: never more than 0.01m (1cm) regardless of panel dent depth
        # Relative maximum: never more than 10% of panel dent depth
        max_floor_displacement_relative = dent_depth * 0.10  # Reduced from 25% to 10%
        max_floor_displacement_absolute = 0.01  # Absolute limit: 1cm maximum
        max_floor_displacement = min(max_floor_displacement_relative, max_floor_displacement_absolute)
        
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        excessive_mask = displacement_magnitudes > max_floor_displacement
        if np.any(excessive_mask):
            displacements[excessive_mask] *= (max_floor_displacement / displacement_magnitudes[excessive_mask])[:, np.newaxis]
        
        # Apply displacements
        vertices[affected_indices] += displacements
        
        # Update mesh
        self.mesh.vertices = vertices
    
    def _find_realistic_dent_location(self, dent_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find a realistic location for a dent based on type.
        EXCLUDES FLOOR PANEL - dents only on walls, roof, and doors.
        
        Args:
            dent_type: Type of dent ('circular', 'elliptical', 'crease', 'corner', 'surface')
            
        Returns:
            Tuple of (position, normal)
        """
        vertices = self.mesh.vertices
        
        # Filter out floor vertices from ALL dent types
        non_floor_mask = np.array([not self._is_on_floor(v) for v in vertices])
        non_floor_vertices = vertices[non_floor_mask]
        
        if len(non_floor_vertices) == 0:
            logger.warning("No non-floor vertices found for dent placement!")
            non_floor_vertices = vertices  # Fallback
        
        if dent_type == 'corner':
            # Prefer corner regions (excluding floor)
            corner_candidates = []
            for v in non_floor_vertices:
                if self._is_on_corner(v):
                    corner_candidates.append(v)
            
            if corner_candidates:
                position = np.array(random.choice(corner_candidates))
            else:
                # Fallback: pick near a corner (but not floor corner)
                # Upper corners only
                corner_idx = random.choice([4, 5, 6, 7])  # Only top corners (z=max)
                corner_pos = np.array([
                    self.bounds[corner_idx & 1][0],
                    self.bounds[(corner_idx >> 1) & 1][1],
                    self.bounds[1][2]  # Always use top z
                ])
                # Find nearest vertex
                distances = np.linalg.norm(non_floor_vertices - corner_pos, axis=1)
                position = non_floor_vertices[np.argmin(distances)]
            
            normal = self._get_surface_normal(position)
            
        elif dent_type == 'crease':
            # Prefer edge regions (excluding floor edges)
            edge_candidates = []
            for v in non_floor_vertices:
                if self._is_on_edge(v) and not self._is_on_corner(v):
                    edge_candidates.append(v)
            
            if edge_candidates:
                position = np.array(random.choice(edge_candidates))
            else:
                # Fallback: pick near an edge (vertical edges only)
                axis = random.choice([0, 1])  # Only x or y axis, not z
                side = random.choice([0, 1])
                target_pos = non_floor_vertices.copy()
                target_pos[:, axis] = self.bounds[side][axis]
                distances = np.linalg.norm(non_floor_vertices - target_pos, axis=1)
                position = non_floor_vertices[np.argmin(distances)]
            
            normal = self._get_surface_normal(position)
            
        elif dent_type == 'surface':
            # Avoid edges and corners - prefer middle regions (excluding floor)
            surface_candidates = []
            for v in non_floor_vertices:
                if not self._is_on_edge(v, threshold=0.1) and not self._is_on_corner(v, threshold=0.1):
                    surface_candidates.append(v)
            
            if surface_candidates:
                position = np.array(random.choice(surface_candidates))
            else:
                # Fallback: random non-floor vertex
                position = non_floor_vertices[random.randint(0, len(non_floor_vertices) - 1)]
            
            normal = self._get_surface_normal(position)
            
        else:  # 'circular' or 'elliptical'
            # Can occur anywhere (except floor), but slightly prefer edges
            if random.random() < 0.4:
                edge_candidates = []
                for v in non_floor_vertices:
                    if self._is_on_edge(v):
                        edge_candidates.append(v)
                
                if edge_candidates:
                    position = np.array(random.choice(edge_candidates))
                else:
                    position = non_floor_vertices[random.randint(0, len(non_floor_vertices) - 1)]
            else:
                position = non_floor_vertices[random.randint(0, len(non_floor_vertices) - 1)]
            
            normal = self._get_surface_normal(position)
        
        return position, normal
    
    def _apply_circular_dent(self, center: np.ndarray, normal: np.ndarray,
                            radius: float, depth: float, falloff: float = 3.0):
        """
        Apply a circular dent to the mesh with physics-based constraints.
        
        PHYSICS LOGIC:
        - Dents ALWAYS push inward (never create outward spikes)
        - Smooth Gaussian falloff for realistic deformation
        - Respects local surface geometry
        - Prevents mesh tearing and artifacts
        
        Args:
            center: Center point of the dent
            normal: Surface normal at center (MUST be outward-pointing)
            radius: Radius of the dent (meters)
            depth: Maximum depth of the dent (meters)
            falloff: Falloff exponent (higher = sharper edges, default 3.0 for smoothness)
        """
        vertices = self.mesh.vertices
        
        # PHYSICS CONSTRAINT: Ensure normal points outward
        to_center = self.center - center
        if np.dot(normal, to_center) > 0:
            normal = -normal  # Flip if pointing wrong way
        
        # Calculate 3D distances (not just 2D projection) for more accurate deformation
        to_vertices = vertices - center
        distances_3d = np.linalg.norm(to_vertices, axis=1)
        
        # Find affected vertices within radius
        mask = distances_3d < radius
        if not np.any(mask):
            return
        
        # OPTIMIZATION: Use center normal for most vertices, local normals only near center
        # This balances realism with performance (10-100x faster on large meshes)
        affected_vertices = vertices[mask]
        num_affected = np.sum(mask)
        
        # For small dents, use center normal (fast)
        # For large dents, blend center normal with local normals near center
        if num_affected < 100:
            # Small dent: calculate local normals for all (fast enough)
            local_normals = np.zeros_like(affected_vertices)
            for i, vertex in enumerate(affected_vertices):
                local_normals[i] = self._get_surface_normal(vertex, k=10)
        else:
            # Large dent: use center normal, blend with local normals in center region
            normalized_dist = distances_3d[mask] / radius
            
            # Use center normal for all vertices initially
            local_normals = np.tile(normal, (num_affected, 1))
            
            # Only calculate local normals for vertices very close to center (where curvature matters most)
            center_region_mask = normalized_dist < 0.3  # Inner 30% of dent
            if np.sum(center_region_mask) > 0 and np.sum(center_region_mask) < 50:
                # Calculate local normals only for center region
                center_vertices = affected_vertices[center_region_mask]
                for i, vertex_idx in enumerate(np.where(center_region_mask)[0]):
                    local_normals[vertex_idx] = self._get_surface_normal(center_vertices[i], k=10)
                
                # Blend: center region uses local normals, outer region uses center normal
                # Smooth transition between them
                blend_factor = np.clip((normalized_dist - 0.3) / 0.2, 0, 1)  # 0.3-0.5 transition zone
                blend_factor = blend_factor[:, np.newaxis]
                local_normals = local_normals * (1 - blend_factor) + np.tile(normal, (num_affected, 1)) * blend_factor
        
        # Smooth falloff using Gaussian function (physics-based)
        normalized_dist = distances_3d[mask] / radius
        depth_factor = np.exp(-falloff * normalized_dist ** 2)
        
        # PHYSICS: Apply displacement along normals (inward)
        # Negative sign = INWARD displacement (away from camera/viewer)
        displacements = -local_normals * (depth * depth_factor[:, np.newaxis])
        
        # PHYSICS CONSTRAINT: Limit maximum displacement to prevent artifacts
        max_displacement = radius * 0.5  # No vertex moves more than half the radius
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        excessive_mask = displacement_magnitudes > max_displacement
        if np.any(excessive_mask):
            displacements[excessive_mask] *= (max_displacement / displacement_magnitudes[excessive_mask])[:, np.newaxis]
        
        # Apply displacements
        vertices[mask] += displacements
        
        # Apply structural bending for nearby structural elements
        self._apply_structural_bending(center, radius, depth, normal)
        
        # Floor deformation disabled - no dents on floor panel
        # self._apply_floor_deformation_from_side_impact(center, radius, depth, normal)
        
        # Highlight dented area with color
        self._highlight_dented_vertices(mask, normalized_dist)
    
    def _apply_elliptical_dent(self, center: np.ndarray, normal: np.ndarray,
                              radius_x: float, radius_y: float, depth: float,
                              angle: float = 0.0, falloff: float = 3.0):
        """
        Apply an elliptical dent to the mesh with physics-based constraints.
        
        PHYSICS LOGIC:
        - Elongated deformation (sliding/scraping impacts)
        - Follows local surface curvature
        - Always pushes inward
        
        Args:
            center: Center point of the dent
            normal: Surface normal at center (outward-pointing)
            radius_x: Major radius (meters)
            radius_y: Minor radius (meters)
            depth: Maximum depth of the dent (meters)
            angle: Rotation angle around normal (radians)
            falloff: Falloff exponent
        """
        vertices = self.mesh.vertices
        
        # PHYSICS CONSTRAINT: Ensure normal points outward
        to_center = self.center - center
        if np.dot(normal, to_center) > 0:
            normal = -normal
        
        # Create two orthogonal vectors in the plane perpendicular to normal
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, [0, 0, 1])
        else:
            u = np.cross(normal, [1, 0, 0])
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(normal, u)
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # Rotate basis vectors
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        u_rot = u * cos_a + v * sin_a
        v_rot = -u * sin_a + v * cos_a
        
        # Project vertices onto plane
        to_vertices = vertices - center
        proj_u = np.dot(to_vertices, u_rot)
        proj_v = np.dot(to_vertices, v_rot)
        
        # Calculate elliptical distance
        distances_2d = np.sqrt((proj_u / radius_x) ** 2 + (proj_v / radius_y) ** 2)
        
        mask = distances_2d < 1.0
        if not np.any(mask):
            return
        
        # OPTIMIZATION: Use center normal for most vertices, local normals only near center
        affected_vertices = vertices[mask]
        num_affected = np.sum(mask)
        
        if num_affected < 100:
            # Small dent: calculate local normals for all
            local_normals = np.zeros_like(affected_vertices)
            for i, vertex in enumerate(affected_vertices):
                local_normals[i] = self._get_surface_normal(vertex, k=10)
        else:
            # Large dent: use center normal with local normals in center region
            normalized_dist = distances_2d[mask]
            local_normals = np.tile(normal, (num_affected, 1))
            
            # Calculate local normals only for center region
            center_region_mask = normalized_dist < 0.3
            if np.sum(center_region_mask) > 0 and np.sum(center_region_mask) < 50:
                center_vertices = affected_vertices[center_region_mask]
                for i, vertex_idx in enumerate(np.where(center_region_mask)[0]):
                    local_normals[vertex_idx] = self._get_surface_normal(center_vertices[i], k=10)
                
                # Smooth blend between center and outer regions
                blend_factor = np.clip((normalized_dist - 0.3) / 0.2, 0, 1)
                blend_factor = blend_factor[:, np.newaxis]
                local_normals = local_normals * (1 - blend_factor) + np.tile(normal, (num_affected, 1)) * blend_factor
        
        # Smooth falloff
        normalized_dist = distances_2d[mask]
        depth_factor = np.exp(-falloff * normalized_dist ** 2)
        
        # PHYSICS: Apply displacement along normals (inward)
        displacements = -local_normals * (depth * depth_factor[:, np.newaxis])
        
        # PHYSICS CONSTRAINT: Limit maximum displacement
        max_displacement = max(radius_x, radius_y) * 0.4
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        excessive_mask = displacement_magnitudes > max_displacement
        if np.any(excessive_mask):
            displacements[excessive_mask] *= (max_displacement / displacement_magnitudes[excessive_mask])[:, np.newaxis]
        
        # Apply displacement
        vertices[mask] += displacements
        
        # Apply structural bending for nearby structural elements
        avg_radius = (radius_x + radius_y) / 2
        self._apply_structural_bending(center, avg_radius, depth, normal)
        
        # Floor deformation disabled - no dents on floor panel
        # self._apply_floor_deformation_from_side_impact(center, avg_radius, depth, normal)
        
        # Highlight dented area with color
        self._highlight_dented_vertices(mask, normalized_dist)
    
    def _apply_crease_dent(self, center: np.ndarray, normal: np.ndarray,
                          length: float, width: float, depth: float,
                          direction: Optional[np.ndarray] = None, falloff: float = 3.0):
        """
        Apply a crease dent (elongated dent along an edge) with physics constraints.
        
        PHYSICS LOGIC:
        - Long, narrow deformation (scraping/sliding impacts)
        - Follows edge contours naturally
        - Smooth depth variation along length
        
        Args:
            center: Center point of the crease
            normal: Surface normal at center (outward-pointing)
            length: Length of the crease (meters)
            width: Width of the crease (meters)
            depth: Maximum depth (meters)
            direction: Direction vector along the crease (if None, auto-detect)
            falloff: Falloff exponent
        """
        vertices = self.mesh.vertices
        
        # PHYSICS CONSTRAINT: Ensure normal points outward
        to_center = self.center - center
        if np.dot(normal, to_center) > 0:
            normal = -normal
        
        # Auto-detect direction if not provided
        if direction is None:
            # Find edge direction by checking nearby vertices
            distances = np.linalg.norm(vertices - center, axis=1)
            nearby_indices = np.argsort(distances)[:20]
            nearby_vertices = vertices[nearby_indices]
            
            # Find principal direction
            if len(nearby_vertices) > 1:
                directions = nearby_vertices - center
                # Project onto plane perpendicular to normal
                directions_proj = directions - np.outer(np.dot(directions, normal), normal)
                # Use PCA or just take longest direction
                if len(directions_proj) > 0:
                    direction = directions_proj[np.argmax(np.linalg.norm(directions_proj, axis=1))]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                else:
                    direction = np.array([1, 0, 0])
            else:
                direction = np.array([1, 0, 0])
        else:
            direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Create perpendicular vector
        if abs(np.dot(direction, normal)) < 0.9:
            perp = np.cross(normal, direction)
        else:
            perp = np.cross(normal, [1, 0, 0])
        perp = perp / (np.linalg.norm(perp) + 1e-10)
        
        # Project vertices
        to_vertices = vertices - center
        along_crease = np.dot(to_vertices, direction)
        across_crease = np.dot(to_vertices, perp)
        
        # Calculate distance metric (elongated shape)
        along_factor = (along_crease / (length / 2)) ** 2
        across_factor = (across_crease / (width / 2)) ** 2
        distances_2d = np.sqrt(along_factor + across_factor)
        
        mask = distances_2d < 1.0
        if not np.any(mask):
            return
        
        # OPTIMIZATION: Use center normal for most vertices, local normals only near center
        affected_vertices = vertices[mask]
        num_affected = np.sum(mask)
        
        if num_affected < 100:
            # Small dent: calculate local normals for all
            local_normals = np.zeros_like(affected_vertices)
            for i, vertex in enumerate(affected_vertices):
                local_normals[i] = self._get_surface_normal(vertex, k=10)
        else:
            # Large dent: use center normal with local normals in center region
            normalized_dist = distances_2d[mask]
            local_normals = np.tile(normal, (num_affected, 1))
            
            # Calculate local normals only for center region
            center_region_mask = normalized_dist < 0.3
            if np.sum(center_region_mask) > 0 and np.sum(center_region_mask) < 50:
                center_vertices = affected_vertices[center_region_mask]
                for i, vertex_idx in enumerate(np.where(center_region_mask)[0]):
                    local_normals[vertex_idx] = self._get_surface_normal(center_vertices[i], k=10)
                
                # Smooth blend between center and outer regions
                blend_factor = np.clip((normalized_dist - 0.3) / 0.2, 0, 1)
                blend_factor = blend_factor[:, np.newaxis]
                local_normals = local_normals * (1 - blend_factor) + np.tile(normal, (num_affected, 1)) * blend_factor
        
        # Smooth falloff
        normalized_dist = distances_2d[mask]
        depth_factor = np.exp(-falloff * normalized_dist ** 2)
        
        # PHYSICS: Apply displacement along normals (inward)
        displacements = -local_normals * (depth * depth_factor[:, np.newaxis])
        
        # PHYSICS CONSTRAINT: Limit maximum displacement
        max_displacement = width * 0.8  # Creases can be relatively deep
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)
        excessive_mask = displacement_magnitudes > max_displacement
        if np.any(excessive_mask):
            displacements[excessive_mask] *= (max_displacement / displacement_magnitudes[excessive_mask])[:, np.newaxis]
        
        # Apply displacement
        vertices[mask] += displacements
        
        # Apply structural bending for nearby structural elements
        avg_size = (length + width) / 2
        self._apply_structural_bending(center, avg_size, depth, normal)
        
        # Floor deformation disabled - no dents on floor panel
        # self._apply_floor_deformation_from_side_impact(center, avg_size, depth, normal)
        
        # Highlight dented area with color
        self._highlight_dented_vertices(mask, normalized_dist)
    
    def add_dent(self, dent_type: str = 'random', 
                 size_range: Tuple[float, float] = (0.08, 0.50),
                 depth_range: Tuple[float, float] = (0.02, 0.15),
                 severity: str = 'normal',
                 **kwargs):
        """
        Add a single dent to the container.
        
        Args:
            dent_type: Type of dent ('circular', 'elliptical', 'crease', 'corner', 'surface', 'random')
            size_range: (min, max) radius/size in meters
            depth_range: (min, max) depth in meters
            severity: 'light', 'normal', 'heavy', or 'extreme' for dent intensity
            **kwargs: Additional parameters for specific dent types
        """
        if dent_type == 'random':
            dent_type = random.choice(['circular', 'elliptical', 'crease', 'corner', 'surface'])
        
        # Find realistic location (EXCLUDES FLOOR)
        position, normal = self._find_realistic_dent_location(dent_type)
        
        # Apply severity multipliers for dramatic damage variety
        # Aggressive multipliers for serious damage (physics fixes prevent artifacts)
        severity_multipliers = {
            'light': (0.9, 0.9),      # Slightly smaller
            'normal': (1.2, 1.3),    # Base level - serious damage
            'heavy': (1.8, 2.0),     # Large, deep impacts
            'extreme': (2.5, 2.8)    # Major collisions - very dramatic
        }
        size_mult, depth_mult = severity_multipliers.get(severity, (1.0, 1.0))
        
        # Random size and depth with severity scaling
        size = random.uniform(*size_range) * size_mult
        depth = random.uniform(*depth_range) * depth_mult
        
        logger.debug(f"Adding {severity} {dent_type} dent at {position}, size={size:.3f}m, depth={depth:.3f}m")
        
        # Store dent specification before applying
        dent_spec = {
            'type': dent_type,
            'severity': severity,
            'position': position.tolist(),
            'normal': normal.tolist(),
            'depth': depth,
            'depth_mm': depth * 1000,
        }
        
        # Apply dent based on type
        if dent_type == 'circular':
            dent_spec['radius'] = size
            self._apply_circular_dent(position, normal, size, depth)
        
        elif dent_type == 'elliptical':
            # Random aspect ratio and rotation - very elongated for serious damage
            aspect_ratio = random.uniform(2.0, 5.0)  # Very elongated
            radius_x = size * np.sqrt(aspect_ratio)
            radius_y = size / np.sqrt(aspect_ratio)
            angle = random.uniform(0, 2 * np.pi)
            dent_spec['radius_x'] = radius_x
            dent_spec['radius_y'] = radius_y
            dent_spec['angle'] = angle
            self._apply_elliptical_dent(position, normal, radius_x, radius_y, depth, angle)
        
        elif dent_type == 'crease':
            # Crease is elongated - very long creases for serious scraping damage
            length = size * random.uniform(3.0, 6.0)  # Very long creases
            width = size * random.uniform(0.2, 0.4)   # Narrow but deep
            dent_spec['length'] = length
            dent_spec['width'] = width
            self._apply_crease_dent(position, normal, length, width, depth)
        
        elif dent_type == 'corner':
            # Corner dents are typically circular but MUCH deeper
            depth = depth * random.uniform(1.8, 3.0)  # Very deep for corners
            dent_spec['radius'] = size
            dent_spec['depth'] = depth  # Update depth after modification
            dent_spec['depth_mm'] = depth * 1000
            self._apply_circular_dent(position, normal, size, depth)
        
        elif dent_type == 'surface':
            # Surface dents are wider and varied in depth
            size = size * random.uniform(1.8, 3.0)  # Much wider
            depth = depth * random.uniform(0.9, 1.3)  # Varied depth
            dent_spec['radius'] = size
            dent_spec['depth'] = depth  # Update depth after modification
            dent_spec['depth_mm'] = depth * 1000
            self._apply_circular_dent(position, normal, size, depth)
        
        # Store dent specification
        self.dent_specs_list.append(dent_spec)
    
    def add_multiple_dents(self, num_dents: int = 5,
                          dent_type_distribution: Dict[str, float] = None,
                          size_range: Tuple[float, float] = (0.08, 0.50),
                          depth_range: Tuple[float, float] = (0.02, 0.15),
                          varied_severity: bool = True):
        """
        Add multiple dents with realistic distribution.
        AUTOMATICALLY EXCLUDES FLOOR PANEL.
        
        Args:
            num_dents: Number of dents to add
            dent_type_distribution: Probability distribution for dent types
                                   (default: realistic distribution)
            size_range: (min, max) size in meters
            depth_range: (min, max) depth in meters
            varied_severity: If True, mix light and heavy damage for variety
        """
        if dent_type_distribution is None:
            # Realistic distribution: creases and corners are more common
            dent_type_distribution = {
                'crease': 0.35,      # Most common (edge impacts)
                'corner': 0.25,      # Common (stacking/handling)
                'circular': 0.20,    # Common (forklift, cargo)
                'elliptical': 0.15,  # Less common (sliding impacts)
                'surface': 0.05      # Least common (direct impacts)
            }
        
        # Normalize probabilities
        total = sum(dent_type_distribution.values())
        dent_type_distribution = {k: v / total for k, v in dent_type_distribution.items()}
        
        dent_types = list(dent_type_distribution.keys())
        probabilities = list(dent_type_distribution.values())
        
        # Severity distribution for variety (some small, some BIG)
        severity_options = ['light', 'normal', 'heavy', 'extreme']
        severity_probs = [0.15, 0.35, 0.35, 0.15]  # Mix of severities
        
        logger.info(f"Adding {num_dents} dents to container (varied severity, excluding floor)...")
        
        for i in range(num_dents):
            dent_type = np.random.choice(dent_types, p=probabilities)
            
            if varied_severity:
                severity = np.random.choice(severity_options, p=severity_probs)
            else:
                severity = 'normal'
            
            logger.info(f"  [{i+1}/{num_dents}] Adding {severity} {dent_type} dent...")
            self.add_dent(dent_type=dent_type, size_range=size_range, 
                         depth_range=depth_range, severity=severity)
        
        logger.info(f"✓ Added {num_dents} dents successfully (floor panel was excluded)")
    
    def get_mesh(self) -> trimesh.Trimesh:
        """Get the modified mesh with dents."""
        # Ensure mesh is still valid
        self.mesh.remove_duplicate_faces()
        self.mesh.remove_unreferenced_vertices()
        return self.mesh
    
    def get_dent_specs(self) -> List[Dict]:
        """Get list of dent specifications."""
        return self.dent_specs_list.copy()


def add_dents_to_container(input_path: str, output_path: str,
                          num_dents: int = 5,
                          dent_type_distribution: Dict[str, float] = None,
                          size_range: Tuple[float, float] = (0.08, 0.50),
                          depth_range: Tuple[float, float] = (0.02, 0.15),
                          varied_severity: bool = True,
                          save_specs: bool = True):
    """
    Add realistic dents to a container OBJ file with physics-based deformation.
    AUTOMATICALLY EXCLUDES FLOOR PANEL from dents.
    
    PHYSICS-BASED LOGIC:
    - Dents ALWAYS push inward (never create spikes/artifacts)
    - Local surface normals ensure proper deformation direction
    - Smooth Gaussian falloff for realistic material behavior
    - Displacement constraints prevent mesh tearing
    
    SERIOUS DAMAGE DEFAULTS:
    - Size: 8-50cm (serious impact zones)
    - Depth: 2-15cm (deep depressions)
    - Varied severity creates mix of minor to extreme damage
    
    Args:
        input_path: Path to input container OBJ file
        output_path: Path to save dented container OBJ file
        num_dents: Number of dents to add
        dent_type_distribution: Probability distribution for dent types
        size_range: (min, max) size in meters (default: 8-50cm for serious damage)
        depth_range: (min, max) depth in meters (default: 2-15cm for serious damage)
        varied_severity: If True, mix light and heavy damage for variety
    """
    logger.info(f"Loading container from {input_path}...")
    mesh = trimesh.load(input_path)
    
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, get the first geometry
        mesh = list(mesh.geometry.values())[0]
    
    logger.info(f"Container loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Ensure mesh has vertex colors
    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        # Try to get material color or use default
        if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'main_color'):
            base_color = mesh.visual.material.main_color[:3] * 255
            mesh.visual.vertex_colors = np.tile(
                np.append(base_color, 255).astype(np.uint8),
                (len(mesh.vertices), 1)
            )
        else:
            # Default gray color
            mesh.visual.vertex_colors = np.tile([128, 128, 128, 255], (len(mesh.vertices), 1))
    
    # Create dent generator
    generator = DentGenerator(mesh)
    
    # Add dents (floor is automatically excluded)
    generator.add_multiple_dents(
        num_dents=num_dents,
        dent_type_distribution=dent_type_distribution,
        size_range=size_range,
        depth_range=depth_range,
        varied_severity=varied_severity
    )
    
    # Get modified mesh
    dented_mesh = generator.get_mesh()
    
    # Count highlighted vertices
    num_dented_vertices = np.sum(generator.dented_vertices)
    total_vertices = len(dented_mesh.vertices)
    dent_percentage = (num_dented_vertices / total_vertices) * 100 if total_vertices > 0 else 0
    
    # Export
    logger.info(f"Exporting dented container to {output_path}...")
    dented_mesh.export(output_path)
    
    # Save dent specifications if requested
    if save_specs:
        specs_path = Path(output_path).with_suffix('.json')
        dent_specs = {
            'filename': str(output_path),
            'container_type': Path(input_path).stem.split('_')[1] if '_' in Path(input_path).stem else 'unknown',
            'num_dents': len(generator.dent_specs_list),
            'dented_vertices_count': int(num_dented_vertices),
            'total_vertices': int(total_vertices),
            'dent_percentage': float(dent_percentage),
            'dents': generator.get_dent_specs(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(specs_path, 'w') as f:
            json.dump(dent_specs, f, indent=2)
        logger.info(f"✓ Dent specifications saved to: {specs_path}")
    
    logger.info("=" * 60)
    logger.info("Dent generation complete!")
    logger.info(f"Dented vertices: {num_dented_vertices}/{total_vertices} ({dent_percentage:.1f}%)")
    logger.info("Dented areas are highlighted in dark red/orange color")
    logger.info("=" * 60)
    return dented_mesh, generator.get_dent_specs() if save_specs else None


def batch_process_containers(input_folder: str = "complete_containers",
                            output_folder: str = "complete_containers_dents",
                            num_dents: int = 5,
                            size_range: Tuple[float, float] = (0.08, 0.50),
                            depth_range: Tuple[float, float] = (0.02, 0.15),
                            varied_severity: bool = True):
    """
    Batch process all container OBJ files in a folder with physics-based dents.
    AUTOMATICALLY EXCLUDES FLOOR PANEL from dents.
    
    SERIOUS DAMAGE DEFAULTS:
    - Size: 8-50cm (serious impact zones)
    - Depth: 2-15cm (deep depressions)
    
    Args:
        input_folder: Folder containing input container OBJ files
        output_folder: Folder to save dented containers
        num_dents: Number of dents to add per container
        size_range: (min, max) size in meters (default: 8-50cm for serious damage)
        depth_range: (min, max) depth in meters (default: 2-15cm for serious damage)
        varied_severity: If True, mix light and heavy damage for variety
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Check if input folder exists
    if not input_path.exists():
        logger.error(f"Input folder '{input_folder}' does not exist!")
        return
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_path.absolute()}")
    
    # Find all OBJ files (skip _scene.obj files)
    obj_files = [f for f in input_path.glob("*.obj") if "_scene.obj" not in f.name]
    
    if not obj_files:
        logger.warning(f"No OBJ files found in '{input_folder}'")
        return
    
    logger.info(f"Found {len(obj_files)} container file(s) to process")
    logger.info("=" * 60)
    
    # Process each file
    all_specs = []
    for i, input_file in enumerate(obj_files, 1):
        logger.info(f"\n[{i}/{len(obj_files)}] Processing: {input_file.name}")
        
        # Create output filename (preserve original name)
        output_file = output_path / input_file.name
        
        try:
            result = add_dents_to_container(
                input_path=str(input_file),
                output_path=str(output_file),
                num_dents=num_dents,
                size_range=size_range,
                depth_range=depth_range,
                varied_severity=varied_severity,
                save_specs=True
            )
            if isinstance(result, tuple):
                dented_mesh, dent_specs = result
                if dent_specs:
                    # Load the saved specs file
                    specs_file = Path(output_file).with_suffix('.json')
                    if specs_file.exists():
                        with open(specs_file, 'r') as f:
                            container_spec = json.load(f)
                            all_specs.append(container_spec)
            logger.info(f"✓ Successfully processed: {output_file.name}")
        except Exception as e:
            logger.error(f"✗ Error processing {input_file.name}: {e}")
            continue
    
    # Save combined specifications file
    if all_specs:
        combined_specs_path = output_path / "dented_container_specifications.json"
        combined_specs = {
            'timestamp': datetime.now().isoformat(),
            'total_containers': len(all_specs),
            'containers': all_specs
        }
        with open(combined_specs_path, 'w') as f:
            json.dump(combined_specs, f, indent=2)
        logger.info(f"✓ Combined specifications saved to: {combined_specs_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Batch processing complete! Processed {len(obj_files)} file(s)")
    logger.info(f"Output saved to: {output_path.absolute()}")


def main():
    """Main function - batch processes containers by default."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Add realistic dents to container models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch process all containers in complete_containers folder (default)
  python add_dents.py
  
  # Process a single file
  python add_dents.py input.obj output.obj
  
  # Batch process with custom parameters
  python add_dents.py --num-dents 10 --min-size 0.08 --max-size 0.30
        """
    )
    parser.add_argument('input', type=str, nargs='?', default=None,
                       help='Input container OBJ file (optional - if not provided, batch processes complete_containers folder)')
    parser.add_argument('output', type=str, nargs='?', default=None,
                       help='Output dented container OBJ file (required if input is provided)')
    parser.add_argument('--num-dents', type=int, default=5,
                       help='Number of dents to add (default: 5)')
    parser.add_argument('--min-size', type=float, default=0.08,
                       help='Minimum dent size in meters (default: 0.08 = 8cm for serious damage)')
    parser.add_argument('--max-size', type=float, default=0.50,
                       help='Maximum dent size in meters (default: 0.50 = 50cm for serious damage)')
    parser.add_argument('--min-depth', type=float, default=0.02,
                       help='Minimum dent depth in meters (default: 0.02 = 2cm for serious damage)')
    parser.add_argument('--max-depth', type=float, default=0.15,
                       help='Maximum dent depth in meters (default: 0.15 = 15cm for serious damage)')
    parser.add_argument('--no-varied-severity', action='store_true',
                       help='Disable varied severity (all dents will be same intensity)')
    parser.add_argument('--input-folder', type=str, default='complete_containers',
                       help='Input folder for batch processing (default: complete_containers)')
    parser.add_argument('--output-folder', type=str, default='complete_containers_dents',
                       help='Output folder for batch processing (default: complete_containers_dents)')
    
    args = parser.parse_args()
    
    # If input/output are provided, process single file
    if args.input is not None:
        if args.output is None:
            parser.error("output file is required when input file is provided")
        
        add_dents_to_container(
            input_path=args.input,
            output_path=args.output,
            num_dents=args.num_dents,
            size_range=(args.min_size, args.max_size),
            depth_range=(args.min_depth, args.max_depth),
            varied_severity=not args.no_varied_severity
        )
    else:
        # Batch process all containers in the folder
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING MODE")
        logger.info("=" * 60)
        batch_process_containers(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            num_dents=args.num_dents,
            size_range=(args.min_size, args.max_size),
            depth_range=(args.min_depth, args.max_depth),
            varied_severity=not args.no_varied_severity
        )


if __name__ == "__main__":
    main()

