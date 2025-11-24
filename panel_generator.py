#!/usr/bin/env python3
"""
Corrugated Panel Generator
Handles the creation of corrugated shipping container panels with various realistic patterns.
"""

import numpy as np
import trimesh
import random
from enum import Enum

class CorrugationPattern(Enum):
    """Different types of realistic shipping container corrugation patterns"""
    STANDARD_VERTICAL = "standard_vertical"      # Classic vertical corrugations (sides)
    BACK_PANEL_VERTICAL = "back_panel_vertical"  # Back panel (end wall opposite door) - ISO standard
    DOOR_HORIZONTAL = "door_horizontal"          # Door panel horizontal corrugations (NOT for walls!)
    ROOF_PATTERN = "roof_pattern"               # Roof corrugations with drainage ridges
    DEEP_WAVE = "deep_wave"                     # Deep wave industrial containers
    MINI_WAVE = "mini_wave"                     # Fine corrugations for small containers
    SQUARE_FLUTE = "square_flute"               # Angular square-wave pattern
    TRAPEZOIDAL = "trapezoidal"                 # Trapezoidal corrugations (common in Europe)
    ASYMMETRIC_WAVE = "asymmetric_wave"         # Asymmetric corrugations for special use

class StandardContainerColors:
    """
    Standard shipping container colors based on ISO specifications and industry standards.
    Colors are defined in RGB format (0-255) and correspond to real container colors.
    """
    
    # Most common shipping container colors by shipping line
    MAERSK_BLUE = (65, 150, 215)      # Maersk Line light blue (RAL 5010 variant)
    MSC_YELLOW = (255, 204, 0)        # Mediterranean Shipping Company yellow
    CMA_CGM_BLUE = (0, 46, 116)       # CMA CGM dark blue
    COSCO_BLUE = (0, 90, 156)         # COSCO shipping blue
    HAPAG_ORANGE = (255, 102, 0)      # Hapag-Lloyd orange
    EVERGREEN_GREEN = (0, 122, 51)    # Evergreen Line green
    
    # Standard container colors by type
    REEFER_WHITE = (255, 255, 255)    # White for refrigerated containers (RAL 9003)
    CARGO_GRAY = (128, 128, 128)      # Gray for general cargo (RAL 7035)
    LEASING_MAROON = (128, 0, 32)     # Maroon/brown for leasing companies
    TRITON_BROWN = (101, 67, 33)      # Brown for Triton leasing
    
    # Additional standard colors
    CONTAINER_BEIGE = (245, 245, 220) # Light beige (RAL 1015)
    BOTTLE_GREEN = (49, 79, 79)       # Dark green (RAL 6007)
    TRAFFIC_GRAY = (143, 143, 143)    # Traffic gray (RAL 7042)
    PURE_WHITE = (255, 255, 255)      # Pure white (RAL 9010)
    SIGNAL_BLACK = (40, 40, 40)       # Near black (RAL 9005)
    GENTIAN_BLUE = (62, 108, 181)     # Gentian blue (RAL 5010)
    
    @classmethod
    def get_all_colors(cls):
        """Get all available container colors as a list of tuples (name, rgb_tuple)"""
        colors = []
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, tuple) and len(attr_value) == 3:
                    colors.append((attr_name, attr_value))
        return colors
    
    @classmethod
    def get_random_color(cls):
        """Get a random container color"""
        colors = cls.get_all_colors()
        color_name, rgb_color = random.choice(colors)
        return color_name, rgb_color
    
    @classmethod
    def get_weighted_random_color(cls):
        """
        Get a weighted random color based on real-world frequency.
        More common colors have higher probability.
        """
        # Define weights based on real shipping container color frequency
        color_weights = {
            'MAERSK_BLUE': 0.15,      # Very common - Maersk is largest shipping line
            'CARGO_GRAY': 0.12,       # Very common - standard cargo containers
            'REEFER_WHITE': 0.10,     # Common - many reefer containers
            'MSC_YELLOW': 0.08,       # Common - MSC is 2nd largest line
            'CMA_CGM_BLUE': 0.08,     # Common - CMA CGM is 3rd largest
            'EVERGREEN_GREEN': 0.07,  # Common - major shipping line
            'LEASING_MAROON': 0.06,   # Common - many leasing companies use dark colors
            'GENTIAN_BLUE': 0.06,     # Standard RAL color
            'COSCO_BLUE': 0.05,       # Major Chinese shipping line
            'HAPAG_ORANGE': 0.05,     # Distinctive orange containers
            'BOTTLE_GREEN': 0.04,     # Standard green containers
            'CONTAINER_BEIGE': 0.04,  # Light colored containers
            'TRAFFIC_GRAY': 0.04,     # Gray variants
            'TRITON_BROWN': 0.03,     # Leasing company colors
            'PURE_WHITE': 0.02,       # Less common pure white
            'SIGNAL_BLACK': 0.01      # Rare black containers
        }
        
        colors = list(color_weights.keys())
        weights = list(color_weights.values())
        
        chosen_color_name = random.choices(colors, weights=weights)[0]
        rgb_color = getattr(cls, chosen_color_name)
        
        return chosen_color_name, rgb_color

class CorrugatedPanelGenerator:
    def __init__(self):
        """Initialize with default shipping container corrugation specifications"""
        # Current pattern selection - default to back panel since that's what we're generating
        self.corrugation_pattern = CorrugationPattern.BACK_PANEL_VERTICAL
        
        # Base corrugation specifications (will be overridden by pattern selection)
        self.corrugation_depth = 0.036    # 36mm depth (ISO standard for back panels)
        self.corrugation_frequency = 3.6  # 3.6 corrugations per meter (278mm pitch)
        self.wall_thickness = 0.002      # 2mm steel thickness (ISO standard)
        
        # Standard shipping container back panel dimensions
        # Default to 20ft container back panel (using internal dimensions for accuracy)
        self.panel_width = 2.352   # Standard internal width (back panel width)
        self.panel_height = 2.390  # Standard internal height
        self.container_type = "20ft"  # Track container type
        
        # Container color (RGB format 0-255)
        self.container_color_name = "CARGO_GRAY"
        self.container_color_rgb = StandardContainerColors.CARGO_GRAY
        
        # Legacy support (keeping panel_length for backward compatibility)
        self.panel_length = self.panel_width  # For existing code compatibility
        
    def set_container_type(self, container_type="20ft"):
        """
        Set the container type and update panel dimensions accordingly.
        
        Args:
            container_type: "20ft" or "40ft" standard container
        """
        self.container_type = container_type
        
        if container_type == "20ft":
            # 20ft container back panel (internal dimensions)
            self.panel_width = 2.352   # Internal width
            self.panel_height = 2.390  # Internal height
        elif container_type == "40ft":
            # 40ft container back panel (internal dimensions)  
            self.panel_width = 2.352   # Same internal width as 20ft
            self.panel_height = 2.393  # Slightly different internal height
        else:
            print(f"Warning: Unknown container type '{container_type}', using 20ft dimensions")
            self.panel_width = 2.352
            self.panel_height = 2.390
            
        # Update legacy panel_length for backward compatibility
        self.panel_length = self.panel_width
        
        print(f"Set to {container_type} container back panel: {self.panel_width:.3f}m × {self.panel_height:.3f}m")
        
    def randomize_parameters(self):
        """Randomize corrugation parameters while keeping standard container dimensions"""
        # Get all patterns except horizontal (which is only for doors, not back panels)
        wall_patterns = [pattern for pattern in CorrugationPattern 
                        if pattern != CorrugationPattern.DOOR_HORIZONTAL]
        
        # Randomly select a corrugation pattern (excluding horizontal)
        self.corrugation_pattern = random.choice(wall_patterns)
        
        # Set pattern-specific parameters
        self._set_pattern_parameters()
        
        # Randomly choose container type (but keep standard dimensions)
        container_types = ["20ft", "40ft"]
        chosen_type = random.choice(container_types)
        self.set_container_type(chosen_type)
        
        # Randomly select container color using weighted distribution
        self.randomize_color()
        
        print(f"Selected {chosen_type} container back panel with {self.corrugation_pattern.value} corrugation")
        print(f"Container color: {self.container_color_name} {self.container_color_rgb}")
    
    def randomize_color(self, use_weighted=True):
        """
        Randomly select a container color.
        
        Args:
            use_weighted: If True, use weighted random selection based on real-world frequency
        """
        if use_weighted:
            self.container_color_name, self.container_color_rgb = StandardContainerColors.get_weighted_random_color()
        else:
            self.container_color_name, self.container_color_rgb = StandardContainerColors.get_random_color()
    
    def set_color(self, color_name):
        """
        Set container color by name.
        
        Args:
            color_name: Name of the color from StandardContainerColors class
        """
        if hasattr(StandardContainerColors, color_name):
            self.container_color_name = color_name
            self.container_color_rgb = getattr(StandardContainerColors, color_name)
            print(f"Set container color to: {color_name} {self.container_color_rgb}")
        else:
            print(f"Warning: Unknown color '{color_name}', keeping current color")
            available_colors = [name for name, _ in StandardContainerColors.get_all_colors()]
            print(f"Available colors: {available_colors}")
    
    def get_color_info(self):
        """Get current color information"""
        return {
            'color_name': self.container_color_name,
            'rgb_color': self.container_color_rgb,
            'normalized_rgb': tuple(c/255.0 for c in self.container_color_rgb)  # For trimesh compatibility
        }
    
    def _set_pattern_parameters(self):
        """Set parameters based on the selected corrugation pattern using official ISO standards"""
        if self.corrugation_pattern == CorrugationPattern.STANDARD_VERTICAL:
            # Official ISO container back panel/end wall pattern (ISO 1496-1)
            # Back panels use same corrugation as side walls: 36mm depth, trapezium section
            self.corrugation_depth = 0.036  # Exact ISO specification: 36mm depth
            self.corrugation_frequency = 3.6  # ~278mm pitch = 3.6 corrugations per meter
            self.wall_thickness = random.uniform(0.0016, 0.002)  # 1.6-2.0mm steel (ISO standard)
            
        elif self.corrugation_pattern == CorrugationPattern.BACK_PANEL_VERTICAL:
            # Official ISO back panel (end wall opposite door) - ISO 1496-1 specification
            # End wall corrugations: 45.6mm depth (front end) or 36mm depth (rear end/back panel)
            self.corrugation_depth = 0.036  # ISO back panel specification: 36mm depth
            self.corrugation_frequency = 3.6  # ~278mm pitch = 3.6 corrugations per meter  
            self.wall_thickness = 0.002  # 2.0mm steel (end wall standard thickness)
            
        elif self.corrugation_pattern == CorrugationPattern.DOOR_HORIZONTAL:
            # Door panel horizontal corrugations (3-5 per door) - NOT for wall panels!
            # Doors also use 36mm depth but horizontal orientation
            self.corrugation_depth = 0.036  # ISO specification: 36mm depth
            self.corrugation_frequency = random.uniform(1.2, 2.0) # 1.2-2 corrugations per meter
            self.wall_thickness = 0.002   # 2.0mm steel (doors are standard thickness)
            
        elif self.corrugation_pattern == CorrugationPattern.ROOF_PATTERN:
            # Roof corrugations - official ISO specification
            self.corrugation_depth = 0.020  # Exact ISO specification: 20mm depth
            self.corrugation_frequency = 4.8  # ~209mm pitch = 4.8 corrugations per meter
            self.wall_thickness = 0.002  # 2.0mm steel (roof standard)
            
        elif self.corrugation_pattern == CorrugationPattern.DEEP_WAVE:
            # Heavy-duty industrial containers (larger than standard)
            self.corrugation_depth = random.uniform(0.050, 0.070)  # 50-70mm depth (deeper than standard)
            self.corrugation_frequency = random.randint(6, 10)   # 6-10 corrugations per meter
            self.wall_thickness = random.uniform(0.003, 0.005)   # 3-5mm steel (heavy duty)
            
        elif self.corrugation_pattern == CorrugationPattern.MINI_WAVE:
            # Fine corrugations for specialized containers (smaller than standard)
            self.corrugation_depth = random.uniform(0.015, 0.025)  # 15-25mm depth (shallower than standard)
            self.corrugation_frequency = random.randint(16, 24)  # 16-24 corrugations per meter
            self.wall_thickness = random.uniform(0.001, 0.0015)   # 1-1.5mm steel (thin gauge)
            
        elif self.corrugation_pattern == CorrugationPattern.SQUARE_FLUTE:
            # Angular corrugations based on ISO standard but angular profile
            self.corrugation_depth = 0.036  # Same as standard but angular profile
            self.corrugation_frequency = 3.6   # Same pitch as standard
            self.wall_thickness = 0.002   # 2mm steel (standard)
            
        elif self.corrugation_pattern == CorrugationPattern.TRAPEZOIDAL:
            # ISO trapezoidal corrugations (European style but same dimensions)
            self.corrugation_depth = 0.036  # ISO standard: 36mm depth
            self.corrugation_frequency = 3.6   # ISO standard pitch
            self.wall_thickness = random.uniform(0.0016, 0.002) # 1.6-2.0mm steel (ISO range)
            
        elif self.corrugation_pattern == CorrugationPattern.ASYMMETRIC_WAVE:
            # Special asymmetric pattern based on ISO dimensions
            self.corrugation_depth = random.uniform(0.030, 0.042)  # Around ISO standard (30-42mm)
            self.corrugation_frequency = random.uniform(3.0, 4.2)   # Around ISO frequency
            self.wall_thickness = random.uniform(0.0016, 0.002)   # ISO steel thickness range
    
    def get_panel_specs(self):
        """Get current panel specifications as a dictionary"""
        return {
            'container_type': self.container_type,
            'corrugation_pattern': self.corrugation_pattern.value,
            'corrugation_depth': self.corrugation_depth,
            'corrugation_frequency': self.corrugation_frequency,
            'wall_thickness': self.wall_thickness,
            'panel_width': self.panel_width,
            'panel_height': self.panel_height,
            'panel_length': self.panel_length,  # For backward compatibility
            'container_color_name': self.container_color_name,
            'container_color_rgb': self.container_color_rgb,
            'container_color_normalized': tuple(c/255.0 for c in self.container_color_rgb)
        }
    
    def create_corrugated_panel(self, panel_width=None, panel_height=None):
        """
        Create a corrugated container back panel with the selected pattern.
        
        Args:
            panel_width: Width of the back panel (meters), uses self.panel_width if None
            panel_height: Height of the back panel (meters), uses self.panel_height if None
            
        Returns:
            trimesh.Trimesh: The corrugated back panel mesh
        """
        if panel_width is None:
            panel_width = self.panel_width
        if panel_height is None:
            panel_height = self.panel_height
            
        print(f"Creating {self.container_type} container back panel...")
        print(f"Pattern: {self.corrugation_pattern.value}")
        print(f"Color: {self.container_color_name} {self.container_color_rgb}")
        print(f"Dimensions: {panel_width:.3f}m × {panel_height:.3f}m")
        print(f"Corrugation: {self.corrugation_depth:.3f}m depth, {self.corrugation_frequency:.1f} freq")
        
        # Create base grid with appropriate resolution for the pattern
        resolution = self._get_resolution_for_pattern()
        u_points, v_points = resolution
        
        u_coords = np.linspace(0, panel_width, u_points)
        v_coords = np.linspace(0, panel_height, v_points)
        U, V = np.meshgrid(u_coords, v_coords)
        
        # Generate corrugation based on selected pattern
        Z = self._generate_corrugation_pattern(U, V, panel_width, panel_height)
        
        # Create front and back surface vertices
        front_vertices = np.column_stack([U.flatten(), V.flatten(), Z.flatten()])
        back_vertices = np.column_stack([U.flatten(), V.flatten(), Z.flatten() - self.wall_thickness])
        vertices = np.vstack([front_vertices, back_vertices])
        
        # Generate faces for the mesh
        faces = self._generate_faces(u_points, v_points, len(front_vertices))
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Validate and fix mesh orientation
        try:
            # Fix mesh normals and ensure consistent face winding
            mesh.fix_normals()
            
            # Remove any duplicate or degenerate faces
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            
            # Check volume and fix if negative
            if mesh.volume < 0:
                print(f"    ⚠ Negative volume detected ({mesh.volume:.6f}), fixing mesh orientation...")
                mesh.invert()  # Invert face normals
                if mesh.volume < 0:
                    print(f"    ⚠ Still negative volume, using absolute value")
            
            # Validate final mesh
            is_valid = not mesh.is_empty and len(mesh.vertices) > 0 and len(mesh.faces) > 0
            volume = abs(mesh.volume)  # Ensure positive volume for display
            
            if is_valid:
                print(f"✓ Valid {self.container_type} back panel: {len(vertices)} vertices, {len(faces)} faces, volume: {volume:.6f} m³")
            else:
                print("⚠ Warning: Mesh appears to be empty")
                
        except Exception as e:
            print(f"⚠ Warning: Mesh validation failed: {e}")
            # Basic cleanup on failure
            try:
                mesh.fix_normals()
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
            except:
                pass
        
        # Add realistic colors based on pattern
        self._apply_pattern_colors(mesh, faces, vertices, len(front_vertices))
        
        return mesh
    
    def _get_resolution_for_pattern(self):
        """Get appropriate mesh resolution based on corrugation pattern"""
        if self.corrugation_pattern == CorrugationPattern.MINI_WAVE:
            return (120, 60)  # High resolution for fine details
        elif self.corrugation_pattern == CorrugationPattern.DOOR_HORIZONTAL:
            return (40, 100)  # More resolution along corrugation direction
        elif self.corrugation_pattern == CorrugationPattern.DEEP_WAVE:
            return (60, 40)   # Moderate resolution for large features
        else:
            return (80, 40)   # Default resolution
    
    def _generate_corrugation_pattern(self, U, V, panel_width, panel_height):
        """Generate the specific corrugation pattern"""
        if self.corrugation_pattern == CorrugationPattern.STANDARD_VERTICAL:
            return self._standard_vertical_pattern(U, V, panel_width, panel_height)
        elif self.corrugation_pattern == CorrugationPattern.BACK_PANEL_VERTICAL:
            return self._back_panel_vertical_pattern(U, V, panel_width, panel_height)
        elif self.corrugation_pattern == CorrugationPattern.DOOR_HORIZONTAL:
            return self._door_horizontal_pattern(U, V, panel_width, panel_height)
        elif self.corrugation_pattern == CorrugationPattern.ROOF_PATTERN:
            return self._roof_pattern(U, V, panel_width, panel_height)
        elif self.corrugation_pattern == CorrugationPattern.DEEP_WAVE:
            return self._deep_wave_pattern(U, V, panel_width, panel_height)
        elif self.corrugation_pattern == CorrugationPattern.MINI_WAVE:
            return self._mini_wave_pattern(U, V, panel_width, panel_height)
        elif self.corrugation_pattern == CorrugationPattern.SQUARE_FLUTE:
            return self._square_flute_pattern(U, V, panel_width, panel_height)
        elif self.corrugation_pattern == CorrugationPattern.TRAPEZOIDAL:
            return self._trapezoidal_pattern(U, V, panel_width, panel_height)
        elif self.corrugation_pattern == CorrugationPattern.ASYMMETRIC_WAVE:
            return self._asymmetric_wave_pattern(U, V, panel_width, panel_height)
        else:
            return self._standard_vertical_pattern(U, V, panel_width, panel_height)
    
    def _standard_vertical_pattern(self, U, V, panel_width, panel_height):
        """Standard vertical sine wave corrugations (most common)"""
        wave_phase = 2 * np.pi * self.corrugation_frequency * U / panel_width
        primary_wave = self.corrugation_depth * np.sin(wave_phase)
        # Add subtle secondary harmonics
        secondary_wave = (self.corrugation_depth * 0.1) * np.sin(wave_phase * 3)
        return primary_wave + secondary_wave
    
    def _back_panel_vertical_pattern(self, U, V, panel_width, panel_height):
        """Official ISO back panel (end wall opposite door) corrugations - ISO 1496-1"""
        # Back panel uses precise ISO trapezium corrugation pattern
        # 36mm depth, 278mm pitch, vertical orientation
        wave_phase = 2 * np.pi * self.corrugation_frequency * U / panel_width
        
        # Create precise trapezium wave pattern (closer to actual container shape)
        # Using triangle wave as base for trapezium approximation
        triangle_wave = self.corrugation_depth * (2/np.pi) * np.arcsin(np.sin(wave_phase))
        
        # Add slight structural variations typical of back panels
        # Back panels often have subtle manufacturing variations
        structural_variation = (self.corrugation_depth * 0.05) * np.sin(wave_phase * 2)
        
        return triangle_wave + structural_variation
    
    def _roof_pattern(self, U, V, panel_width, panel_height):
        """Roof corrugations with slight camber for water drainage"""
        wave_phase = 2 * np.pi * self.corrugation_frequency * U / panel_width
        primary_wave = self.corrugation_depth * np.sin(wave_phase)
        
        # Add slight roof camber for drainage (highest in center)
        camber_height = 0.015  # 15mm camber
        camber = camber_height * (1 - 4 * (V / panel_height - 0.5)**2)
        
        return primary_wave + camber
    
    def _deep_wave_pattern(self, U, V, panel_width, panel_height):
        """Deep wave pattern for heavy-duty containers"""
        wave_phase = 2 * np.pi * self.corrugation_frequency * U / panel_width
        # Deeper, more pronounced waves
        primary_wave = self.corrugation_depth * np.sin(wave_phase)
        # Add structural reinforcement ridges
        reinforcement = (self.corrugation_depth * 0.2) * np.sin(wave_phase * 2)
        return primary_wave + reinforcement
    
    def _mini_wave_pattern(self, U, V, panel_width, panel_height):
        """Fine corrugations for specialized containers"""
        wave_phase = 2 * np.pi * self.corrugation_frequency * U / panel_width
        # Smooth, fine corrugations
        primary_wave = self.corrugation_depth * np.sin(wave_phase)
        # Add very subtle texture
        texture = (self.corrugation_depth * 0.05) * np.sin(wave_phase * 5)
        return primary_wave + texture
    
    def _square_flute_pattern(self, U, V, panel_width, panel_height):
        """Angular square-wave corrugations for modern containers"""
        wave_phase = 2 * np.pi * self.corrugation_frequency * U / panel_width
        # Create square wave pattern
        square_wave = self.corrugation_depth * np.sign(np.sin(wave_phase))
        # Smooth the edges slightly for manufacturability
        smoothing = (self.corrugation_depth * 0.1) * np.sin(wave_phase * 4)
        return square_wave + smoothing
    
    def _trapezoidal_pattern(self, U, V, panel_width, panel_height):
        """Trapezoidal corrugations common in European containers"""
        wave_phase = 2 * np.pi * self.corrugation_frequency * U / panel_width
        # Create trapezoidal pattern using triangle wave
        triangle_wave = self.corrugation_depth * (2/np.pi) * np.arcsin(np.sin(wave_phase))
        # Add slight rounding at peaks
        rounding = (self.corrugation_depth * 0.1) * np.sin(wave_phase * 2)
        return triangle_wave + rounding
    
    def _asymmetric_wave_pattern(self, U, V, panel_width, panel_height):
        """Asymmetric corrugations for special applications"""
        wave_phase = 2 * np.pi * self.corrugation_frequency * U / panel_width
        # Create asymmetric wave (steeper rise, gentler fall)
        asymmetric_wave = self.corrugation_depth * (np.sin(wave_phase) + 0.3 * np.sin(2 * wave_phase))
        # Add slight vertical variation
        vertical_mod = (self.corrugation_depth * 0.1) * np.sin(np.pi * V / panel_height)
        return asymmetric_wave + vertical_mod
    
    def _door_horizontal_pattern(self, U, V, panel_width, panel_height):
        """Horizontal corrugations for door panels ONLY (3-5 ridges per panel) - NOT for walls!"""
        wave_phase = 2 * np.pi * self.corrugation_frequency * V / panel_height
        # Create broader, more spaced corrugations
        primary_wave = self.corrugation_depth * np.sin(wave_phase)
        # Add slight curvature variation
        curve_variation = (self.corrugation_depth * 0.05) * np.sin(2 * np.pi * U / panel_width)
        return primary_wave + curve_variation
        
    def _generate_faces(self, u_points, v_points, vertex_offset):
        """Generate faces for the corrugated mesh with consistent winding order"""
        faces = []
        
        # Front surface faces (counter-clockwise when viewed from outside)
        front_faces = []
        for i in range(v_points - 1):
            for j in range(u_points - 1):
                v1 = i * u_points + j
                v2 = i * u_points + (j + 1)
                v3 = (i + 1) * u_points + (j + 1)
                v4 = (i + 1) * u_points + j
                
                # Counter-clockwise winding for outward-facing normals
                front_faces.extend([[v1, v2, v3], [v1, v3, v4]])
        
        # Back surface faces (clockwise when viewed from outside, since it's the inner surface)
        back_faces = []
        for i in range(v_points - 1):
            for j in range(u_points - 1):
                v1 = vertex_offset + i * u_points + j
                v2 = vertex_offset + i * u_points + (j + 1)
                v3 = vertex_offset + (i + 1) * u_points + (j + 1)
                v4 = vertex_offset + (i + 1) * u_points + j
                
                # Clockwise winding for inward-facing normals (since we're inside the mesh)
                back_faces.extend([[v1, v3, v2], [v1, v4, v3]])
        
        # Side faces to connect front and back
        side_faces = self._generate_side_faces(u_points, v_points, vertex_offset)
        
        all_faces = np.array(front_faces + back_faces + side_faces)
        
        return all_faces
    
    def _generate_side_faces(self, u_points, v_points, vertex_offset):
        """Generate side faces to connect front and back surfaces"""
        side_faces = []
        
        # Bottom edge
        for j in range(u_points - 1):
            v1_front, v2_front = j, j + 1
            v1_back, v2_back = vertex_offset + j, vertex_offset + j + 1
            side_faces.extend([[v1_front, v2_front, v2_back], [v1_front, v2_back, v1_back]])
        
        # Top edge
        top_offset = (v_points - 1) * u_points
        for j in range(u_points - 1):
            v1_front, v2_front = top_offset + j, top_offset + j + 1
            v1_back, v2_back = vertex_offset + top_offset + j, vertex_offset + top_offset + j + 1
            side_faces.extend([[v1_front, v2_back, v2_front], [v1_front, v1_back, v2_back]])
        
        # Left edge
        for i in range(v_points - 1):
            v1_front, v2_front = i * u_points, (i + 1) * u_points
            v1_back, v2_back = vertex_offset + i * u_points, vertex_offset + (i + 1) * u_points
            side_faces.extend([[v1_front, v1_back, v2_back], [v1_front, v2_back, v2_front]])
        
        # Right edge
        right_offset = u_points - 1
        for i in range(v_points - 1):
            v1_front = i * u_points + right_offset
            v2_front = (i + 1) * u_points + right_offset
            v1_back = vertex_offset + i * u_points + right_offset
            v2_back = vertex_offset + (i + 1) * u_points + right_offset
            side_faces.extend([[v1_front, v2_back, v1_back], [v1_front, v2_front, v2_back]])
        
        return side_faces
    
    def _apply_pattern_colors(self, mesh, faces, vertices, front_vertex_count):
        """Apply realistic container colors based on the selected color scheme"""
        num_faces = len(faces)
        face_colors = np.zeros((num_faces, 4), dtype=np.uint8)
        
        # Get the selected container color
        base_r, base_g, base_b = self.container_color_rgb
        
        # Create base color with alpha
        base_color = [base_r, base_g, base_b, 255]
        
        # Create highlight color (slightly lighter)
        highlight_factor = 1.2
        highlight_r = min(255, int(base_r * highlight_factor))
        highlight_g = min(255, int(base_g * highlight_factor))
        highlight_b = min(255, int(base_b * highlight_factor))
        highlight_color = [highlight_r, highlight_g, highlight_b, 255]
        
        # Create shadow/edge color (darker)
        shadow_factor = 0.7
        shadow_r = int(base_r * shadow_factor)
        shadow_g = int(base_g * shadow_factor)
        shadow_b = int(base_b * shadow_factor)
        edge_color = [shadow_r, shadow_g, shadow_b, 255]
        
        # Apply weathering effects for certain colors
        if self.container_color_name in ['LEASING_MAROON', 'TRITON_BROWN', 'CARGO_GRAY']:
            # Weathered containers use more muted highlights
            highlight_factor = 1.1
            highlight_color = [min(255, int(base_r * highlight_factor)),
                             min(255, int(base_g * highlight_factor)), 
                             min(255, int(base_b * highlight_factor)), 255]
        
        # Color front faces with base color
        front_face_count = (len(vertices) // 2 - front_vertex_count) * 2
        face_colors[:front_face_count] = base_color
        
        # Color back and side faces with edge color for depth
        face_colors[front_face_count:] = edge_color
        
        # Add pattern-specific highlights to simulate corrugation ridges
        if self.corrugation_pattern == CorrugationPattern.ROOF_PATTERN:
            # More pronounced highlights for roof corrugations
            highlight_interval = max(1, front_face_count // 15)
        elif self.corrugation_pattern in [CorrugationPattern.DEEP_WAVE, CorrugationPattern.SQUARE_FLUTE]:
            # Strong highlights for deep patterns
            highlight_interval = max(1, front_face_count // 12)
        elif self.corrugation_pattern == CorrugationPattern.MINI_WAVE:
            # Subtle highlights for fine patterns
            highlight_interval = max(1, front_face_count // 30)
        else:
            # Standard highlights
            highlight_interval = max(1, front_face_count // 20)
        
        # Apply highlights to simulate corrugation ridges catching light
        for i in range(0, min(front_face_count, len(face_colors)), highlight_interval):
            face_colors[i] = highlight_color
        
        # Add subtle color variation for realism (slight weathering/wear)
        if self.container_color_name not in ['REEFER_WHITE', 'PURE_WHITE']:
            # Add slight random variation to non-white containers
            variation_faces = random.sample(range(front_face_count), 
                                          min(front_face_count // 50, front_face_count))
            for face_idx in variation_faces:
                # Slight darkening for wear spots
                face_colors[face_idx] = [max(0, base_r - 10), 
                                       max(0, base_g - 10), 
                                       max(0, base_b - 10), 255]
        
        mesh.visual.face_colors = face_colors
        
        print(f"✓ Applied {self.container_color_name} color scheme to mesh")
    
    def export_to_obj(self, mesh, filename):
        """Export the mesh to an OBJ file with colors"""
        try:
            mesh.export(filename)
            print(f"✓ {self.corrugation_pattern.value} panel exported to: {filename}")
            return True
        except Exception as e:
            print(f"✗ Error exporting OBJ file: {str(e)}")
            return False 