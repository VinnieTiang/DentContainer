#!/usr/bin/env python3
"""
Dent Generator
Handles the creation and application of realistic dents to corrugated panels.
"""

import numpy as np
import trimesh
import random
from enum import Enum

class DentType(Enum):
    """Different types of realistic container damage"""
    CIRCULAR_IMPACT = "circular_impact"      # Falling object impact
    DIAGONAL_SCRAPE = "diagonal_scrape"      # Forklift scrape
    ELONGATED_SCRATCH = "elongated_scratch"  # Dragging damage
    IRREGULAR_COLLISION = "irregular_collision"  # Accident damage
    MULTI_IMPACT = "multi_impact"            # Multiple small impacts
    CORNER_DAMAGE = "corner_damage"          # Corner post impact

class DentGenerator:
    def __init__(self):
        """Initialize with default dent specifications"""
        self.dent_enabled = True
        self.dent_type = DentType.CIRCULAR_IMPACT
        self.dent_specs = {}
        
    def randomize_dent_parameters(self, panel_width, panel_height):
        """
        Generate random dent parameters based on the panel dimensions.
        
        Args:
            panel_width: Width of the container back panel (meters)
            panel_height: Height of the container back panel (meters)
        """
        # Randomly select dent type
        dent_type = random.choice(list(DentType))
        
        if dent_type == DentType.CIRCULAR_IMPACT:
            self._generate_circular_impact(panel_width, panel_height)
        elif dent_type == DentType.DIAGONAL_SCRAPE:
            self._generate_diagonal_scrape(panel_width, panel_height)
        elif dent_type == DentType.ELONGATED_SCRATCH:
            self._generate_elongated_scratch(panel_width, panel_height)
        elif dent_type == DentType.IRREGULAR_COLLISION:
            self._generate_irregular_collision(panel_width, panel_height)
        elif dent_type == DentType.MULTI_IMPACT:
            self._generate_multi_impact(panel_width, panel_height)
        elif dent_type == DentType.CORNER_DAMAGE:
            self._generate_corner_damage(panel_width, panel_height)
    
    def _generate_circular_impact(self, panel_width, panel_height):
        """Generate circular impact dent (falling objects, machinery contact)"""
        # Larger impact areas can support deeper damage realistically
        radius = random.uniform(0.20, 0.60)  # 20-60cm radius (40cm-1.2m diameter) - LARGER
        
        # Generate center coordinates
        center_x = random.uniform(panel_width * 0.25, panel_width * 0.75)
        center_y = random.uniform(panel_height * 0.25, panel_height * 0.75)
        
        # Calculate realistic depth with enhanced visibility
        depth_factor = (radius / 0.60) ** 0.5  # Square root scaling for realistic physics
        min_depth = 0.015  # 15mm minimum (was 4mm)
        max_depth = 0.060  # 60mm maximum for very large impacts (was 25mm)
        
        actual_depth = min_depth + (max_depth - min_depth) * depth_factor
        
        # Ensure depth is within reasonable bounds and well above corrugation amplitude
        actual_depth = max(min_depth, min(actual_depth, max_depth))
        
        self.dent_specs = {
            'type': 'circular_impact',
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'depth': actual_depth,  # Now 15-60mm range
            'description': f"Circular impact at ({center_x:.3f}, {center_y:.3f}) with {radius:.3f}m radius and {actual_depth*1000:.1f}mm depth"
        }
        self.dent_enabled = True
        print(f"Generated circular impact dent: {radius*2:.2f}m diameter, {actual_depth*1000:.1f}mm deep")
    
    def _generate_diagonal_scrape(self, panel_width, panel_height):
        """Generate diagonal scrape dent (dragging, sliding damage)"""
        # Wider scrapes can be deeper due to distributed loading
        width = random.uniform(0.12, 0.35)  # 12-35cm width - WIDER
        
        # Enhanced depth calculation for better visibility
        depth_factor = (width / 0.35) ** 0.7  # Scrapes scale differently than impacts
        min_depth = 0.010  # 10mm minimum (was 3mm)
        max_depth = 0.035  # 35mm maximum for very wide scrapes (was 15mm)
        
        actual_depth = min_depth + (max_depth - min_depth) * depth_factor
        actual_depth = max(min_depth, min(actual_depth, max_depth))
        
        # Generate random start and end points for the scrape
        start_x = random.uniform(panel_width * 0.1, panel_width * 0.9)
        start_y = random.uniform(panel_height * 0.1, panel_height * 0.9)
        end_x = random.uniform(panel_width * 0.1, panel_width * 0.9)
        end_y = random.uniform(panel_height * 0.1, panel_height * 0.9)
        
        # Calculate length of the scrape
        length = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
        
        # Calculate angle of the scrape
        dx = end_x - start_x
        dy = end_y - start_y
        angle = np.arctan2(dy, dx)
        
        self.dent_specs = {
            'type': 'diagonal_scrape',
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'width': width,
            'length': length,
            'depth': actual_depth,  # Now 10-35mm range
            'angle': angle,
            'description': f"Diagonal scrape from ({start_x:.3f}, {start_y:.3f}) to ({end_x:.3f}, {end_y:.3f}) with {width:.3f}m width and {actual_depth*1000:.1f}mm depth"
        }
        self.dent_enabled = True
        print(f"Generated diagonal scrape: {width*100:.1f}cm wide, {actual_depth*1000:.1f}mm deep")
    
    def _generate_elongated_scratch(self, panel_width, panel_height):
        """Generate elongated scratch (key/tool scratches, sliding damage)"""
        # Generate mostly vertical or horizontal scratches
        orientation = random.choice(['vertical', 'horizontal', 'diagonal'])
        
        if orientation == 'vertical':
            start_x = random.uniform(panel_width * 0.2, panel_width * 0.8)
            start_y = random.uniform(panel_height * 0.1, panel_height * 0.3)
            end_x = start_x + random.uniform(-0.05, 0.05)  # Slight horizontal drift
            end_y = random.uniform(panel_height * 0.7, panel_height * 0.9)
        elif orientation == 'horizontal':
            start_x = random.uniform(panel_width * 0.1, panel_width * 0.3)
            start_y = random.uniform(panel_height * 0.2, panel_height * 0.8)
            end_x = random.uniform(panel_width * 0.7, panel_width * 0.9)
            end_y = start_y + random.uniform(-0.05, 0.05)  # Slight vertical drift
        else:  # diagonal
            start_x = random.uniform(panel_width * 0.1, panel_width * 0.4)
            start_y = random.uniform(panel_height * 0.1, panel_height * 0.4)
            end_x = random.uniform(panel_width * 0.6, panel_width * 0.9)
            end_y = random.uniform(panel_height * 0.6, panel_height * 0.9)
        
        # Wider scratches can be slightly deeper
        width = random.uniform(0.03, 0.10)   # 3-10cm width - WIDER
        
        # Enhanced depth calculation - scratches need to be visible above corrugations
        depth_factor = (width / 0.10) ** 0.8
        min_depth = 0.008  # 8mm minimum (was 1mm)
        max_depth = 0.020  # 20mm maximum for deep scratches (was 6mm)
        
        actual_depth = min_depth + (max_depth - min_depth) * depth_factor
        actual_depth = max(min_depth, min(actual_depth, max_depth))
        
        # Calculate length of the scratch
        length = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
        
        # Calculate angle of the scratch
        dx = end_x - start_x
        dy = end_y - start_y
        angle = np.arctan2(dy, dx)
        
        self.dent_specs = {
            'type': 'elongated_scratch',
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'width': width,
            'length': length,
            'depth': actual_depth,  # Now 8-20mm range
            'angle': angle,
            'description': f"Elongated scratch from ({start_x:.3f}, {start_y:.3f}) to ({end_x:.3f}, {end_y:.3f}) with {width:.3f}m width and {actual_depth*1000:.1f}mm depth"
        }
        self.dent_enabled = True
        print(f"Generated elongated scratch: {length:.2f}m long, {self.dent_specs['width']*100:.1f}cm wide, {self.dent_specs['depth']*1000:.1f}mm deep")
    
    def _generate_irregular_collision(self, panel_width, panel_height):
        """Generate irregular collision dent (complex impact damage)"""
        # Large irregular impacts from major collisions
        base_radius = random.uniform(0.30, 0.70)  # 30-70cm radius - MUCH LARGER
        
        # Enhanced depth calculation for irregular collisions
        depth_factor = (base_radius / 0.70) ** 0.6  # Complex impacts scale differently
        min_depth = 0.020  # 20mm minimum (was 6mm)
        max_depth = 0.050  # 50mm maximum for major collisions (was 20mm)
        
        actual_depth = min_depth + (max_depth - min_depth) * depth_factor
        actual_depth = max(min_depth, min(actual_depth, max_depth))
        
        # Generate random center position
        center_x = random.uniform(panel_width * 0.2, panel_width * 0.8)
        center_y = random.uniform(panel_height * 0.2, panel_height * 0.8)
        
        # Generate irregular factor and number of lobes
        irregular_factor = 1.0 + 0.3 * np.sin(8 * np.arctan2(center_y - panel_height/2, center_x - panel_width/2))
        num_lobes = random.randint(3, 8)
        
        self.dent_specs = {
            'type': 'irregular_collision',
            'center_x': center_x,
            'center_y': center_y,
            'base_radius': base_radius,
            'irregular_factor': irregular_factor,
            'num_lobes': num_lobes,
            'depth': actual_depth,  # Now 20-50mm range
            'description': f"Irregular collision at ({center_x:.3f}, {center_y:.3f}) with {base_radius:.3f}m base radius and {actual_depth*1000:.1f}mm depth"
        }
        self.dent_enabled = True
        print(f"Generated irregular collision: {base_radius*2:.2f}m diameter, {actual_depth*1000:.1f}mm deep")
    
    def _generate_multi_impact(self, panel_width, panel_height):
        """Generate multiple impact dent (hail damage, projectile impacts)"""
        # Generate cluster of impacts
        num_impacts = random.randint(5, 15)
        cluster_x = random.uniform(panel_width * 0.3, panel_width * 0.7)
        cluster_y = random.uniform(panel_height * 0.3, panel_height * 0.7)
        cluster_radius = random.uniform(0.35, 0.80)  # 35-80cm cluster spread - LARGER
        
        impacts = []
        for _ in range(num_impacts):
            # Generate random position within cluster
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, cluster_radius)
            
            x = cluster_x + distance * np.cos(angle)
            y = cluster_y + distance * np.sin(angle)
            
            # Ensure impact stays within panel bounds
            x = max(0.05, min(panel_width - 0.05, x))
            y = max(0.05, min(panel_height - 0.05, y))
            
            # Individual impact size - larger impacts can be deeper
            radius = random.uniform(0.04, 0.12)  # 4-12cm individual impacts - LARGER
            
            # Enhanced depth calculation for individual impacts
            depth_factor = (radius / 0.12) ** 0.6
            min_depth = 0.008  # 8mm minimum (was 3mm)
            max_depth = 0.030  # 30mm maximum for large individual impacts (was 12mm)
            
            impact_depth = min_depth + (max_depth - min_depth) * depth_factor
            impact_depth = max(min_depth, min(impact_depth, max_depth))
            
            impacts.append({
                'x': x,
                'y': y,
                'radius': radius,
                'depth': impact_depth  # Now 8-30mm range per impact
            })
        
        self.dent_specs = {
            'type': 'multi_impact',
            'impacts': impacts,
            'num_impacts': num_impacts,
            'description': f"Multi-impact with {num_impacts} impacts, depths {min([i['depth'] for i in impacts])*1000:.1f}-{max([i['depth'] for i in impacts])*1000:.1f}mm"
        }
        self.dent_enabled = True
        print(f"Generated multi-impact cluster: {len(impacts)} impacts in {cluster_radius*2:.2f}m area, max {max(i['depth']*1000 for i in impacts):.1f}mm deep")
    
    def _generate_corner_damage(self, panel_width, panel_height):
        """Generate corner damage (forklift hits, loading accidents)"""
        # Select random corner (0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left)
        corner = random.randint(0, 3)
        corner_names = ['bottom-left', 'bottom-right', 'top-right', 'top-left']
        corner_type = corner_names[corner]
        
        # Corner damage can be quite large from forklift/crane impacts
        corner_offset = random.uniform(0.05, 0.25)  # 5-25cm from corner
        corners = [
            (corner_offset, corner_offset),                                    # bottom-left
            (panel_width - corner_offset, corner_offset),                     # bottom-right
            (panel_width - corner_offset, panel_height - corner_offset),      # top-right
            (corner_offset, panel_height - corner_offset)                     # top-left
        ]
        
        center_x, center_y = corners[corner]
        
        # Larger corner damage from heavy equipment
        radius = random.uniform(0.20, 0.50)  # 20-50cm radius - MUCH LARGER
        
        # Enhanced depth calculation for corner damage  
        depth_factor = (radius / 0.50) ** 0.7
        min_depth = 0.015  # 15mm minimum (was 5mm)
        max_depth = 0.045  # 45mm maximum for severe corner damage (was 18mm)
        
        actual_depth = min_depth + (max_depth - min_depth) * depth_factor
        actual_depth = max(min_depth, min(actual_depth, max_depth))
        
        self.dent_specs = {
            'type': 'corner_damage',
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'depth': actual_depth,  # Now 15-45mm range
            'corner_type': corner_type,
            'description': f"Corner damage at {corner_type} corner ({center_x:.3f}, {center_y:.3f}) with {radius:.3f}m radius and {actual_depth*1000:.1f}mm depth"
        }
        self.dent_enabled = True
        print(f"Generated corner damage at {corner_type} corner: {radius*2:.2f}m diameter, {actual_depth*1000:.1f}mm deep")
    
    def get_dent_specs(self):
        """Get current dent specifications as a dictionary"""
        if not self.dent_enabled:
            return {'dent_enabled': False}
            
        return {
            'dent_enabled': self.dent_enabled,
            **self.dent_specs
        }
    
    def calculate_dent_effect(self, x_coords, y_coords, z_coords):
        """
        Calculate realistic corrugation flattening dent based on dent type.
        """
        if not self.dent_enabled:
            return np.zeros_like(x_coords)
        
        dent_type = self.dent_specs['type']
        
        if dent_type in ['circular_impact', 'circular']:
            return self._calculate_circular_dent(x_coords, y_coords, z_coords)
        elif dent_type == 'diagonal_scrape':
            return self._calculate_diagonal_scrape_dent(x_coords, y_coords, z_coords)
        elif dent_type == 'elongated_scratch':
            return self._calculate_elongated_scratch_dent(x_coords, y_coords, z_coords)
        elif dent_type == 'irregular_collision':
            return self._calculate_irregular_collision_dent(x_coords, y_coords, z_coords)
        elif dent_type == 'multi_impact':
            return self._calculate_multi_impact_dent(x_coords, y_coords, z_coords)
        elif dent_type == 'corner_damage':
            return self._calculate_corner_damage_dent(x_coords, y_coords, z_coords)
        
        return np.zeros_like(x_coords)
    
    def _calculate_circular_dent(self, x_coords, y_coords, z_coords):
        """Calculate circular impact dent"""
        specs = self.dent_specs
        dx = x_coords - specs['center_x']
        dy = y_coords - specs['center_y']
        distance = np.sqrt(dx**2 + dy**2)
        
        # Only affect area within radius
        in_impact_zone = distance <= specs['radius']
        
        if not np.any(in_impact_zone):
            return np.zeros_like(x_coords)
        
        # Gaussian falloff from center
        sigma = specs['radius'] / 3  # 3-sigma covers the radius
        impact_strength = np.exp(-(distance**2) / (2 * sigma**2))
        impact_strength *= specs.get('intensity', 1.0)  # Use specs instead of separate intensity
        
        return self._apply_corrugation_flattening(x_coords, y_coords, z_coords, 
                                                impact_strength, in_impact_zone, specs['depth'])
    
    def _calculate_diagonal_scrape_dent(self, x_coords, y_coords, z_coords):
        """Calculate diagonal scrape dent"""
        specs = self.dent_specs
        
        # Calculate distance from line segment
        line_length = np.sqrt((specs['end_x'] - specs['start_x'])**2 + 
                             (specs['end_y'] - specs['start_y'])**2)
        
        if line_length == 0:
            return np.zeros_like(x_coords)
        
        # Vector from start to end
        line_dx = specs['end_x'] - specs['start_x']
        line_dy = specs['end_y'] - specs['start_y']
        
        # Calculate distance from each point to the line segment
        distances_to_line = []
        for i in range(len(x_coords)):
            px, py = x_coords[i], y_coords[i]
            
            # Vector from start to point
            px_dx = px - specs['start_x']
            px_dy = py - specs['start_y']
            
            # Project point onto line
            t = max(0, min(1, (px_dx * line_dx + px_dy * line_dy) / (line_length**2)))
            
            # Closest point on line segment
            closest_x = specs['start_x'] + t * line_dx
            closest_y = specs['start_y'] + t * line_dy
            
            # Distance from point to closest point on line
            dist_to_line = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
            distances_to_line.append(dist_to_line)
        
        distances_to_line = np.array(distances_to_line)
        
        # Only affect area within scrape width
        in_scrape_zone = distances_to_line <= specs['width']
        
        if not np.any(in_scrape_zone):
            return np.zeros_like(x_coords)
        
        # Linear falloff from centerline
        impact_strength = np.maximum(0, 1 - distances_to_line / specs['width'])
        impact_strength *= specs.get('intensity', 1.0)
        
        return self._apply_corrugation_flattening(x_coords, y_coords, z_coords, 
                                                impact_strength, in_scrape_zone, specs['depth'])
    
    def _calculate_elongated_scratch_dent(self, x_coords, y_coords, z_coords):
        """Calculate elongated scratch dent (similar to diagonal scrape but shallower)"""
        return self._calculate_diagonal_scrape_dent(x_coords, y_coords, z_coords)
    
    def _calculate_irregular_collision_dent(self, x_coords, y_coords, z_coords):
        """Calculate irregular collision dent using circular approximation"""
        specs = self.dent_specs
        
        # Use circular approximation based on base_radius
        dx = x_coords - specs['center_x']
        dy = y_coords - specs['center_y']
        distance = np.sqrt(dx**2 + dy**2)
        
        # Only affect area within base radius (with some irregularity)
        # Add some randomness to make it irregular
        irregular_factor = 1.0 + 0.3 * np.sin(8 * np.arctan2(dy, dx))  # 8 lobes for irregularity
        effective_radius = specs['base_radius'] * irregular_factor
        in_collision_zone = distance <= effective_radius
        
        if not np.any(in_collision_zone):
            return np.zeros_like(x_coords)
        
        # Gaussian-like falloff from center with irregularity
        sigma = specs['base_radius'] / 3
        impact_strength = np.exp(-(distance**2) / (2 * sigma**2))
        
        # Add irregular variation to the impact strength
        angular_variation = 0.2 * np.sin(6 * np.arctan2(dy, dx))  # 6 lobes variation
        impact_strength *= (1.0 + angular_variation)
        impact_strength = np.clip(impact_strength, 0, 1)  # Keep in valid range
        
        return self._apply_corrugation_flattening(x_coords, y_coords, z_coords, 
                                                impact_strength, in_collision_zone, specs['depth'])
    
    def _calculate_multi_impact_dent(self, x_coords, y_coords, z_coords):
        """Calculate multiple impact dents"""
        specs = self.dent_specs
        total_displacement = np.zeros_like(x_coords)
        
        for impact in specs['impacts']:
            dx = x_coords - impact['x']
            dy = y_coords - impact['y']
            distance = np.sqrt(dx**2 + dy**2)
            
            # Only affect area within impact radius
            in_impact_zone = distance <= impact['radius']
            
            if np.any(in_impact_zone):
                # Gaussian falloff from impact center
                sigma = impact['radius'] / 3
                impact_strength = np.exp(-(distance**2) / (2 * sigma**2))
                impact_strength *= impact.get('intensity', 1.0)
                
                displacement = self._apply_corrugation_flattening(
                    x_coords, y_coords, z_coords, 
                    impact_strength, in_impact_zone, impact['depth'])
                
                # Accumulate effects (impacts can overlap)
                total_displacement += displacement
        
        return total_displacement
    
    def _calculate_corner_damage_dent(self, x_coords, y_coords, z_coords):
        """Calculate corner damage dent"""
        specs = self.dent_specs
        dx = x_coords - specs['center_x']
        dy = y_coords - specs['center_y']
        distance = np.sqrt(dx**2 + dy**2)
        
        # Only affect area within radius
        in_damage_zone = distance <= specs['radius']
        
        if not np.any(in_damage_zone):
            return np.zeros_like(x_coords)
        
        # Gaussian falloff from center
        sigma = specs['radius'] / 3
        impact_strength = np.exp(-(distance**2) / (2 * sigma**2))
        impact_strength *= specs.get('intensity', 1.0)
        
        return self._apply_corrugation_flattening(x_coords, y_coords, z_coords, 
                                                impact_strength, in_damage_zone, specs['depth'])
    
    def _apply_corrugation_flattening(self, x_coords, y_coords, z_coords, impact_strength, in_zone, depth):
        """Apply corrugation flattening physics"""
        # Calculate target flattened level in the impact area
        impacted_z_values = z_coords[in_zone]
        if len(impacted_z_values) > 0:
            min_z = np.min(impacted_z_values)  # Valley level
            target_level = min_z - depth  # Push inward from valley level
        else:
            target_level = np.mean(z_coords) - depth
        
        # Calculate displacement needed to reach target level
        target_displacement = target_level - z_coords
        
        # Apply impact strength falloff
        final_displacement = target_displacement * impact_strength * in_zone.astype(float)
        
        # Only apply negative (inward) displacements
        final_displacement = np.minimum(final_displacement, 0)
        
        return final_displacement
    
    def _points_in_polygon(self, points, polygon_vertices):
        """Simple point-in-polygon test using ray casting"""
        x, y = points[:, 0], points[:, 1]
        n = len(polygon_vertices)
        inside = np.zeros(len(points), dtype=bool)
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon_vertices[i]
            xj, yj = polygon_vertices[j]
            
            mask = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            inside = inside ^ mask  # XOR for ray casting algorithm
            
            j = i
        
        return inside

    def apply_dent_to_mesh(self, mesh, panel_width, panel_height):
        """
        Apply realistic dent to both surfaces of the panel.
        
        Args:
            mesh: The trimesh.Trimesh object to modify
            panel_width: Width of the container back panel (meters)
            panel_height: Height of the container back panel (meters)
            
        Returns:
            trimesh.Trimesh: The modified mesh with dent applied
        """
        if not self.dent_enabled:
            return mesh
            
        print(f"Applying {self.dent_specs['type'].replace('_', ' ')} dent to mesh...")
        
        # Get vertices and separate front/back surfaces
        vertices = mesh.vertices.copy()
        num_vertices = len(vertices)
        front_vertices = vertices[:num_vertices//2]  # Outer surface (impact side)
        back_vertices = vertices[num_vertices//2:]   # Inner surface
        
        # Apply dent to outer surface
        x_coords = front_vertices[:, 0]
        y_coords = front_vertices[:, 1]
        outer_dent_effect = self.calculate_dent_effect(x_coords, y_coords, front_vertices[:, 2])
        
        # Apply deformation to outer surface
        front_vertices[:, 2] += outer_dent_effect
        
        # Inner surface follows with reduced deformation
        inner_damping = 0.6
        x_coords_inner = back_vertices[:, 0]
        y_coords_inner = back_vertices[:, 1]
        inner_dent_effect = self.calculate_dent_effect(x_coords_inner, y_coords_inner, back_vertices[:, 2])
        
        # Apply dampened deformation to inner surface
        back_vertices[:, 2] += inner_dent_effect * inner_damping
        
        # Ensure minimum wall thickness
        min_wall_thickness = 0.001  # 1mm minimum
        wall_thickness = front_vertices[:, 2] - back_vertices[:, 2]
        too_thin_mask = wall_thickness < min_wall_thickness
        
        if np.any(too_thin_mask):
            back_vertices[too_thin_mask, 2] = front_vertices[too_thin_mask, 2] - min_wall_thickness
        
        # Update mesh vertices
        modified_vertices = np.vstack([front_vertices, back_vertices])
        modified_mesh = trimesh.Trimesh(vertices=modified_vertices, faces=mesh.faces.copy())
        
        # Preserve visual properties if they exist
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
            modified_mesh.visual.face_colors = mesh.visual.face_colors.copy()
        
        affected_vertices = np.sum(outer_dent_effect != 0)
        print(f"  ✓ Applied {self.dent_specs['type'].replace('_', ' ')} dent to {affected_vertices} vertices")
        
        return modified_mesh

    def create_dented_panel_from_base(self, base_mesh, panel_width, panel_height, container_color_rgb=None):
        """
        Create a dented version of a base panel mesh.
        
        Args:
            base_mesh: The base corrugated panel mesh
            panel_width: Width of the container back panel (meters)
            panel_height: Height of the container back panel (meters)
            container_color_rgb: RGB tuple (0-255) of the container's base color
            
        Returns:
            trimesh.Trimesh: A new mesh with dent applied
        """
        # Create a copy of the base mesh
        dented_mesh = base_mesh.copy()
        
        # Randomize dent parameters for this panel
        self.randomize_dent_parameters(panel_width, panel_height)
        
        # Apply dent to the mesh
        dented_mesh = self.apply_dent_to_mesh(dented_mesh, panel_width, panel_height)
        
        # Apply dent coloring with the correct container color
        self.apply_dent_coloring(dented_mesh, dented_mesh.faces, dented_mesh.vertices, container_color_rgb)
        
        return dented_mesh
    
    def apply_dent_coloring(self, mesh, faces, vertices, container_color_rgb=None):
        """
        Apply realistic damage coloring to impacted areas based on dent type.
        The dent maintains the same base color as the container, with darker shading to show deformation.
        
        Args:
            mesh: The mesh object to color
            faces: Mesh faces
            vertices: Mesh vertices
            container_color_rgb: RGB tuple (0-255) of the container's base color
        """
        if not self.dent_enabled:
            return
        
        # Get current face colors or extract base color from existing colors
        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            face_colors = mesh.visual.face_colors.copy()
            
            # If no container color provided, extract from existing face colors
            if container_color_rgb is None:
                # Get the most common color (mode) from existing face colors as base
                unique_colors, counts = np.unique(face_colors[:, :3], axis=0, return_counts=True)
                most_common_idx = np.argmax(counts)
                container_color_rgb = tuple(unique_colors[most_common_idx])
        else:
            # Default to gray if no color information available
            if container_color_rgb is None:
                container_color_rgb = (128, 128, 128)  # Gray fallback
            face_colors = np.full((len(faces), 4), [*container_color_rgb, 255], dtype=np.uint8)
        
        # Extract RGB values for calculations
        base_r, base_g, base_b = container_color_rgb
        base_color = [base_r, base_g, base_b, 255]
        
        # Create damage shading variations of the SAME color
        # Impact center: darkest (simulate deep shadows)
        impact_center_factor = 0.4  # 40% of original brightness
        impact_center_color = [int(base_r * impact_center_factor), 
                             int(base_g * impact_center_factor), 
                             int(base_b * impact_center_factor), 255]
        
        # Stress zones: medium darkness (simulate deformation shadows)  
        stress_factor = 0.6  # 60% of original brightness
        stress_color = [int(base_r * stress_factor), 
                       int(base_g * stress_factor), 
                       int(base_b * stress_factor), 255]
        
        # Edge zones: slight darkening (simulate edge shadows)
        edge_factor = 0.8  # 80% of original brightness
        edge_color = [int(base_r * edge_factor), 
                     int(base_g * edge_factor), 
                     int(base_b * edge_factor), 255]
        
        # Color faces based on dent type and proximity to damage
        for i in range(len(faces)):
            face_vertices = faces[i]
            # Get centroid of face
            face_center_x = np.mean([vertices[v][0] for v in face_vertices])
            face_center_y = np.mean([vertices[v][1] for v in face_vertices])
            
            # Calculate damage intensity at this face
            face_coords = np.array([[face_center_x, face_center_y]])
            dummy_z = np.array([0])  # Not used for coloring calculation
            
            damage_effect = self.calculate_dent_effect(face_coords[:, 0], face_coords[:, 1], dummy_z)
            damage_intensity = abs(damage_effect[0]) if len(damage_effect) > 0 else 0
            
            # Normalize damage intensity for coloring
            if damage_intensity > 0:
                # Scale intensity for color blending
                color_intensity = min(damage_intensity * 20, 1.0)  # Scale factor for visibility
                
                # Determine shading level based on damage intensity
                if color_intensity > 0.7:
                    target_color = impact_center_color
                    blend_factor = color_intensity
                elif color_intensity > 0.3:
                    target_color = stress_color
                    blend_factor = color_intensity * 0.8
                elif color_intensity > 0.1:
                    target_color = edge_color
                    blend_factor = color_intensity * 0.6
                else:
                    continue
                
                # Blend between normal and darkened color of the SAME base color
                blended_color = [
                    int(base_color[j] * (1-blend_factor) + target_color[j] * blend_factor)
                    for j in range(4)
                ]
                face_colors[i] = blended_color
        
        # Apply colors to mesh
        mesh.visual.face_colors = face_colors
        
        container_color_name = f"RGB{container_color_rgb}"
        print(f"  ✓ Applied realistic {self.dent_specs['type'].replace('_', ' ')} damage shading to {container_color_name} container") 