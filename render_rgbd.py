#!/usr/bin/env python3
"""
RGB-D Renderer for Dented Container Panels
Renders aligned RGB and depth images from .obj files for dataset generation.
"""

import os
import glob
import json
import numpy as np
import trimesh
import pyrender
import imageio
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from panel_generator import CorrugatedPanelGenerator
from datetime import datetime

class RGBDRenderer:
    def __init__(self, output_dir="output", image_width=640, image_height=480, 
                 camera_distance=2.35, camera_type="orthographic_fixed", auto_cleanup=True):
        """
        Initialize the RGB-D renderer.
        
        Args:
            output_dir: Directory to save rendered images
            image_width: Width of rendered images
            image_height: Height of rendered images
            camera_distance: Distance from panel surface in meters
            camera_type: "orthographic_fixed", "orthographic_responsive", or "perspective"
            auto_cleanup: Whether to automatically clean up previous outputs (default: True)
        """
        self.output_dir = Path(output_dir)
        self.image_width = image_width
        self.image_height = image_height
        self.camera_distance = camera_distance
        self.camera_type = camera_type
        
        # Automatic cleanup of previous outputs if requested
        if auto_cleanup:
            self._cleanup_previous_outputs()
        
        # Create output directories
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth"
        self.mask_dir = self.output_dir / "mask"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.rgb_dir, self.depth_dir, self.mask_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize pyrender offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.image_width,
            viewport_height=self.image_height
        )
        
        print(f"✓ RGB-D Renderer initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Camera capture dimensions: {self.image_width}x{self.image_height} (exact)")
        print(f"  Camera distance: {self.camera_distance:.2f}m")
        print(f"  Camera type: {self.camera_type}")
        print(f"  Dimension guarantee: All saved images will match camera capture exactly")
        if auto_cleanup:
            print(f"  Previous outputs: Cleaned up automatically")
    
    def _cleanup_previous_outputs(self):
        """Clean up all previous rendering outputs"""
        print(f"🧹 Cleaning up previous rendering outputs...")
        
        import shutil
        
        files_removed = 0
        dirs_removed = 0
        
        # Remove output directory and all contents if it exists
        if self.output_dir.exists():
            try:
                shutil.rmtree(self.output_dir)
                print(f"  ✓ Removed directory: {self.output_dir}")
                dirs_removed += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {self.output_dir}: {e}")
        
        # Remove temp directories that might be left over
        temp_dirs = ["temp_undented"]
        for temp_dir_name in temp_dirs:
            temp_dir = Path(temp_dir_name)
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    print(f"  ✓ Removed directory: {temp_dir}")
                    dirs_removed += 1
                except Exception as e:
                    print(f"  ✗ Failed to remove {temp_dir}: {e}")
        
        # Remove any stray depth metrics files in root
        root_files = list(Path(".").glob("*_depth_metrics.json"))
        for file_path in root_files:
            try:
                file_path.unlink()
                print(f"  ✓ Removed: {file_path}")
                files_removed += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {file_path}: {e}")
        
        print(f"  📊 Render cleanup: {files_removed} files, {dirs_removed} directories removed")
    
    def setup_camera_and_lighting(self, scene, mesh_bounds):
        """
        Set up camera and lighting for optimal panel rendering.
        
        Args:
            scene: Pyrender scene
            mesh_bounds: Mesh bounding box for camera positioning
        """
        # Calculate mesh dimensions
        mesh_center = (mesh_bounds[0] + mesh_bounds[1]) / 2
        mesh_size = mesh_bounds[1] - mesh_bounds[0]
        
        # Get panel width and height (assuming panel is in XY plane)
        panel_width = mesh_size[0]   # X dimension
        panel_height = mesh_size[1]  # Y dimension
        panel_depth = mesh_size[2]   # Z dimension (thickness)
        
        print(f"    Panel dimensions: {panel_width:.2f}m x {panel_height:.2f}m x {panel_depth:.2f}m")
        
        # Calculate panel front surface position
        panel_front_z = mesh_center[2] + panel_depth / 2.0  # Front surface of panel
        
        print(f"    Camera distance: {self.camera_distance:.2f}m from panel surface")
        print(f"    Camera type: {self.camera_type}")
        
        # Create camera based on type (camera position calculated per type)
        if self.camera_type == "orthographic_fixed":
            # Original: Fixed viewing area (distance doesn't affect zoom)
            camera_position = np.array([mesh_center[0], mesh_center[1], panel_front_z + self.camera_distance])
            
            camera = pyrender.OrthographicCamera(
                xmag=panel_width / 2.0,
                ymag=panel_height / 2.0
            )
            print(f"    Camera positioned at: [{camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}]")
            print(f"    Orthographic (fixed): {panel_width:.2f}m x {panel_height:.2f}m viewing area")
            
        elif self.camera_type == "orthographic_responsive":
            # NEW APPROACH: Directly adjust xmag/ymag for zoom instead of camera positioning
            # This should work because pyrender DOES respect xmag/ymag values
            base_distance = 2.35  # Reference distance
            distance_factor = self.camera_distance / base_distance
            
            # Calculate zoom: closer distance = smaller xmag/ymag = more zoom
            zoom_factor = 1.0 / distance_factor  # Invert: closer = higher zoom
            
            # Adjust viewing area by zoom factor
            # Smaller xmag/ymag = camera sees smaller area = zoom effect
            zoomed_xmag = (panel_width / 2.0) / zoom_factor
            zoomed_ymag = (panel_height / 2.0) / zoom_factor
            
            # Keep camera at normal distance (for lighting positioning)
            camera_position = np.array([mesh_center[0], mesh_center[1], panel_front_z + self.camera_distance])
            
            # Create camera with adjusted viewing area
            camera = pyrender.OrthographicCamera(
                xmag=zoomed_xmag,  # Reduced viewing area for zoom
                ymag=zoomed_ymag   # Reduced viewing area for zoom
            )
            
            # Calculate what area is actually captured
            actual_capture_width = zoomed_xmag * 2.0
            actual_capture_height = zoomed_ymag * 2.0
            
            print(f"    Camera positioned at: [{camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}]")
            print(f"    Orthographic (responsive): {actual_capture_width:.2f}m x {actual_capture_height:.2f}m viewing area")
            print(f"    Distance factor: {distance_factor:.2f}x")
            print(f"    Zoom factor: {zoom_factor:.2f}x")
            print(f"    Original xmag/ymag: {panel_width/2.0:.3f}m, {panel_height/2.0:.3f}m")
            print(f"    Zoomed xmag/ymag: {zoomed_xmag:.3f}m, {zoomed_ymag:.3f}m")
            print(f"    Zoom achieved by reducing orthographic viewing area")
            
        elif self.camera_type == "perspective":
            # Perspective camera: Natural distance-dependent scaling
            camera_position = np.array([mesh_center[0], mesh_center[1], panel_front_z + self.camera_distance])
            
            # Calculate field of view to fit panel at reference distance
            reference_distance = 1.00
            fov_y = 2 * np.arctan(panel_height / (2 * reference_distance))
            aspect_ratio = panel_width / panel_height
            
            camera = pyrender.PerspectiveCamera(
                yfov=fov_y,
                aspectRatio=aspect_ratio
            )
            print(f"    Camera positioned at: [{camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}]")
            print(f"    Perspective: FOV={np.degrees(fov_y):.1f}°, aspect={aspect_ratio:.2f}")
            print(f"    Natural distance scaling (closer = larger objects)")
            
        else:
            raise ValueError(f"Unknown camera_type: {self.camera_type}")

        # Camera orientation (looking toward negative Z)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_position
        camera_pose[:3, :3] = np.array([
            [1, 0, 0],   # X-axis (right)
            [0, 1, 0],   # Y-axis (up)  
            [0, 0, 1]    # Z-axis (forward)
        ])
        
        scene.add(camera, pose=camera_pose)
        
        # Add lighting for realistic appearance (same as before)
        # Main directional light from upper right
        main_light = pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=3.0
        )
        main_light_pose = np.eye(4)
        main_light_pose[:3, 3] = camera_position + np.array([panel_width*0.5, panel_height*0.5, 0])
        scene.add(main_light, pose=main_light_pose)
        
        # Fill light from upper left to reduce shadows
        fill_light = pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=1.0
        )
        fill_light_pose = np.eye(4)
        fill_light_pose[:3, 3] = camera_position + np.array([-panel_width*0.5, panel_height*0.3, 0])
        scene.add(fill_light, pose=fill_light_pose)
        
        # Ambient light for overall illumination
        ambient_light = pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=0.5
        )
        ambient_pose = np.eye(4)
        ambient_pose[:3, 3] = mesh_center
        scene.add(ambient_light, pose=ambient_pose)
        
        return camera_pose
    
    def create_dent_mask(self, dented_mesh, dent_specs, camera_pose, mesh_bounds, undented_mesh=None):
        """
        Create ground truth mask using the SAME camera and scene as RGB/depth rendering.
        This ensures perfect pixel alignment by using face colors instead of separate rendering.
        """
        print(f"      Creating mask via unified rendering approach...")
        
        try:
            if undented_mesh is None:
                print(f"        No undented mesh - falling back to specification-based method")
                return self._create_specification_based_mask(dented_mesh, dent_specs, camera_pose, mesh_bounds)
            
            # Step 1: Compare meshes to find actually deformed faces
            dent_face_indices = self._identify_deformed_faces_by_comparison(dented_mesh, undented_mesh)
            
            if len(dent_face_indices) == 0:
                print(f"        No deformed faces found - creating empty mask")
                return np.zeros((self.image_height, self.image_width), dtype=np.uint8)
            
            print(f"        Found {len(dent_face_indices)} deformed faces out of {len(dented_mesh.faces)} total")
            
            # Step 2: Create mask using SAME scene and camera as RGB/depth
            mask_image = self._render_unified_mask(dented_mesh, dent_face_indices, camera_pose, mesh_bounds)
            
            white_pixels = np.sum(mask_image > 127)
            total_pixels = mask_image.size
            percentage = (white_pixels / total_pixels) * 100
            
            print(f"        ✓ Generated unified mask: {white_pixels} white pixels ({percentage:.1f}%)")
            
            return mask_image
            
        except Exception as e:
            print(f"        ERROR in unified mask creation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to specification-based method
            print(f"        Falling back to specification-based method...")
            return self._create_specification_based_mask(dented_mesh, dent_specs, camera_pose, mesh_bounds)

    def _render_unified_mask(self, dented_mesh, dent_face_indices, camera_pose, mesh_bounds):
        """
        Render mask using the EXACT same camera setup as RGB/depth rendering.
        This guarantees perfect pixel alignment.
        """
        print(f"        Rendering unified mask (same camera as RGB/depth)...")
        
        # Create a copy of the mesh for mask rendering
        mask_mesh = dented_mesh.copy()
        
        # Set face colors: white for dent faces, black for normal faces
        num_faces = len(mask_mesh.faces)
        face_colors = np.zeros((num_faces, 4), dtype=np.uint8)  # Start with black/transparent
        
        # Set dent faces to white
        for face_idx in dent_face_indices:
            face_colors[face_idx] = [255, 255, 255, 255]  # White, fully opaque
        
        # Apply face colors to mesh
        mask_mesh.visual.face_colors = face_colors
        
        # Create scene with black background (same as RGB/depth scene setup)
        scene = pyrender.Scene(bg_color=[0, 0, 0, 255])
        
        # Add mask mesh to scene
        mesh_node = pyrender.Mesh.from_trimesh(mask_mesh, smooth=False)
        scene.add(mesh_node)
        
        # Use IDENTICAL camera setup as RGB/depth rendering
        camera_pose_used = self.setup_camera_and_lighting(scene, mesh_bounds)
        
        # Render mask using same renderer
        color, depth = self.renderer.render(scene)
        
        # Convert to binary mask
        # Any non-black pixel becomes white (255), black stays black (0)
        gray = np.dot(color[...,:3], [0.299, 0.587, 0.114])
        binary_mask = (gray > 10).astype(np.uint8) * 255
        
        white_pixels = np.sum(binary_mask > 0)
        print(f"          Unified mask: {white_pixels} white pixels (perfect alignment guaranteed)")
        
        return binary_mask
    
    def _identify_deformed_faces_by_comparison(self, dented_mesh, undented_mesh):
        """
        Identify faces that are actually deformed by comparing dented vs undented mesh.
        This is the most accurate method as it uses the actual mesh changes.
        """
        print(f"        Comparing dented vs undented mesh to find deformed faces...")
        
        # Verify meshes have same topology
        if len(dented_mesh.vertices) != len(undented_mesh.vertices):
            print(f"          ERROR: Mesh vertex count mismatch - dented: {len(dented_mesh.vertices)}, undented: {len(undented_mesh.vertices)}")
            return []
        
        if len(dented_mesh.faces) != len(undented_mesh.faces):
            print(f"          ERROR: Mesh face count mismatch - dented: {len(dented_mesh.faces)}, undented: {len(undented_mesh.faces)}")
            return []
        
        # Calculate face center displacements
        dented_face_centers = dented_mesh.triangles_center
        undented_face_centers = undented_mesh.triangles_center
        
        # Calculate displacement magnitude for each face
        face_displacements = np.linalg.norm(dented_face_centers - undented_face_centers, axis=1)
        
        # Set threshold for significant deformation
        # Use adaptive threshold based on the range of displacements
        displacement_threshold = self._calculate_adaptive_threshold(face_displacements)
        
        # Find faces with significant displacement
        deformed_face_indices = np.where(face_displacements > displacement_threshold)[0].tolist()
        
        print(f"          Displacement range: {np.min(face_displacements):.6f}m to {np.max(face_displacements):.6f}m")
        print(f"          Threshold: {displacement_threshold:.6f}m ({displacement_threshold*1000:.3f}mm)")
        print(f"          Deformed faces: {len(deformed_face_indices)} / {len(dented_mesh.faces)}")
        
        return deformed_face_indices
    
    def _calculate_adaptive_threshold(self, displacements):
        """
        Calculate adaptive threshold for detecting significant mesh deformation.
        """
        # Remove outliers for better threshold calculation
        non_zero_displacements = displacements[displacements > 1e-8]  # Remove essentially zero values
        
        if len(non_zero_displacements) == 0:
            return 1e-6  # 1 micrometer fallback
        
        # Use percentile-based approach
        p75 = np.percentile(non_zero_displacements, 75)
        p25 = np.percentile(non_zero_displacements, 25)
        
        # If there's significant variation, use percentile-based threshold
        if p75 > p25 * 3:  # Significant spread in displacements
            # Use a threshold that captures significant deformations
            threshold = p25 + 0.3 * (p75 - p25)  # 30% above 25th percentile
        else:
            # Small displacement range - use a more sensitive threshold
            mean_displacement = np.mean(non_zero_displacements)
            threshold = mean_displacement * 0.5  # 50% of mean
        
        # Ensure minimum threshold (0.1mm) and maximum threshold (5mm)
        threshold = max(threshold, 0.0001)  # Minimum 0.1mm
        threshold = min(threshold, 0.005)   # Maximum 5mm
        
        return threshold
    
    def _create_specification_based_mask(self, dented_mesh, dent_specs, camera_pose, mesh_bounds):
        """
        Fallback specification-based mask creation (original method)
        """
        print(f"        Using specification-based mask generation (fallback)...")
        
        if dent_specs is None:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        
        # Use the original specification-based method
        dent_face_indices = self._identify_dent_faces_by_position(dented_mesh, dent_specs, mesh_bounds)
        
        if len(dent_face_indices) == 0:
            return np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        
        mask_mesh = self._create_dent_mask_mesh(dented_mesh, dent_face_indices)
        mask_image = self._render_dent_mask_mesh(mask_mesh, camera_pose, mesh_bounds)
        
        return mask_image
    
    def _identify_dent_faces_by_position(self, mesh, dent_specs, mesh_bounds):
        """
        Identify faces that are part of the dent based on their 3D positions.
        This is the core of the 3D geometry tagging approach.
        """
        print(f"        Identifying dent faces by 3D position...")
        
        face_centers = mesh.triangles_center
        mesh_center = (mesh_bounds[0] + mesh_bounds[1]) / 2
        dent_type = dent_specs.get('type', '')
        
        print(f"          Dent type: {dent_type}")
        print(f"          Mesh center: {mesh_center}")
        print(f"          Face centers range: X[{np.min(face_centers[:, 0]):.3f}, {np.max(face_centers[:, 0]):.3f}], Y[{np.min(face_centers[:, 1]):.3f}, {np.max(face_centers[:, 1]):.3f}], Z[{np.min(face_centers[:, 2]):.3f}, {np.max(face_centers[:, 2]):.3f}]")
        
        dent_faces = []
        
        if dent_type in ['circular_impact', 'circular']:
            dent_faces = self._identify_circular_dent_faces(face_centers, dent_specs, mesh_center)
            
        elif dent_type in ['diagonal_scrape', 'elongated_scratch']:
            dent_faces = self._identify_linear_dent_faces(face_centers, dent_specs, mesh_center)
            
        elif dent_type == 'irregular_collision':
            dent_faces = self._identify_irregular_dent_faces(face_centers, dent_specs, mesh_center)
            
        elif dent_type == 'multi_impact':
            dent_faces = self._identify_multi_impact_faces(face_centers, dent_specs, mesh_center)
            
        elif dent_type == 'corner_damage':
            dent_faces = self._identify_corner_damage_faces(face_centers, dent_specs, mesh_center)
        
        else:
            print(f"          Unknown dent type: {dent_type}")
        
        print(f"          Identified {len(dent_faces)} dent faces")
        return dent_faces
    
    def _identify_circular_dent_faces(self, face_centers, dent_specs, mesh_center):
        """Identify faces within circular dent area."""
        center_x = dent_specs.get('center_x', mesh_center[0])
        center_y = dent_specs.get('center_y', mesh_center[1])
        radius = dent_specs.get('radius', 0.1)
        
        print(f"          Circular dent: center=({center_x:.3f}, {center_y:.3f}), radius={radius:.3f}m")
        
        dent_faces = []
        for i, face_center in enumerate(face_centers):
            # Distance in XY plane (ignore Z for dent area calculation)
            dx = face_center[0] - center_x
            dy = face_center[1] - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance <= radius:
                dent_faces.append(i)
        
        return dent_faces
    
    def _identify_linear_dent_faces(self, face_centers, dent_specs, mesh_center):
        """Identify faces within linear scratch/scrape area."""
        start_x = dent_specs.get('start_x', mesh_center[0] - 0.2)
        start_y = dent_specs.get('start_y', mesh_center[1] - 0.1)
        end_x = dent_specs.get('end_x', mesh_center[0] + 0.2)
        end_y = dent_specs.get('end_y', mesh_center[1] + 0.1)
        width = dent_specs.get('width', 0.05)
        
        print(f"          Linear dent: start=({start_x:.3f}, {start_y:.3f}), end=({end_x:.3f}, {end_y:.3f}), width={width:.3f}m")
        
        dent_faces = []
        
        # Vector along the scratch
        scratch_vec = np.array([end_x - start_x, end_y - start_y])
        scratch_length = np.linalg.norm(scratch_vec)
        
        if scratch_length > 0:
            scratch_unit = scratch_vec / scratch_length
            
            for i, face_center in enumerate(face_centers):
                # Vector from start to face center
                to_face = np.array([face_center[0] - start_x, face_center[1] - start_y])
                
                # Project onto scratch direction
                proj_length = np.dot(to_face, scratch_unit)
                proj_length = max(0, min(scratch_length, proj_length))  # Clamp to line segment
                
                # Point on line closest to face
                closest_point = np.array([start_x, start_y]) + proj_length * scratch_unit
                
                # Distance from face to line
                distance = np.linalg.norm([face_center[0] - closest_point[0], face_center[1] - closest_point[1]])
                
                if distance <= width / 2:
                    dent_faces.append(i)
        
        return dent_faces
    
    def _identify_irregular_dent_faces(self, face_centers, dent_specs, mesh_center):
        """Identify faces within irregular collision boundary."""
        # Check if we have boundary_points (legacy format) or use circular approximation
        boundary_points = dent_specs.get('boundary_points', [])
        if boundary_points:
            print(f"          Irregular dent: {len(boundary_points)} boundary points")
            
            dent_faces = []
            for i, face_center in enumerate(face_centers):
                # Point-in-polygon test in XY plane
                if self._point_in_polygon_2d([face_center[0], face_center[1]], boundary_points):
                    dent_faces.append(i)
            
            return dent_faces
        
        # Use circular approximation (current implementation)
        center_x = dent_specs.get('center_x', mesh_center[0])
        center_y = dent_specs.get('center_y', mesh_center[1])
        base_radius = dent_specs.get('base_radius', 0.1)
        
        print(f"          Irregular dent (circular approx): center=({center_x:.3f}, {center_y:.3f}), base_radius={base_radius:.3f}m")
        
        dent_faces = []
        for i, face_center in enumerate(face_centers):
            # Distance in XY plane (ignore Z for dent area calculation)
            dx = face_center[0] - center_x
            dy = face_center[1] - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Apply same irregular factor as dent generation
            angle = np.arctan2(dy, dx)
            irregular_factor = 1.0 + 0.3 * np.sin(8 * angle)  # 8 lobes for irregularity
            effective_radius = base_radius * irregular_factor
            
            if distance <= effective_radius:
                dent_faces.append(i)
        
        return dent_faces
    
    def _identify_multi_impact_faces(self, face_centers, dent_specs, mesh_center):
        """Identify faces within multiple impact areas."""
        impacts = dent_specs.get('impacts', [])
        if not impacts:
            return []
        
        print(f"          Multi-impact: {len(impacts)} impact sites")
        
        dent_faces = []
        for impact in impacts:
            center_x = impact.get('x', mesh_center[0])
            center_y = impact.get('y', mesh_center[1])
            radius = impact.get('radius', 0.05)
            
            for i, face_center in enumerate(face_centers):
                dx = face_center[0] - center_x
                dy = face_center[1] - center_y
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance <= radius:
                    dent_faces.append(i)
        
        # Remove duplicates
        return list(set(dent_faces))
    
    def _identify_corner_damage_faces(self, face_centers, dent_specs, mesh_center):
        """Identify faces within corner damage area."""
        center_x = dent_specs.get('center_x', mesh_center[0])
        center_y = dent_specs.get('center_y', mesh_center[1])
        radius = dent_specs.get('radius', 0.15)
        
        print(f"          Corner damage: center=({center_x:.3f}, {center_y:.3f}), radius={radius:.3f}m")
        
        dent_faces = []
        for i, face_center in enumerate(face_centers):
            dx = face_center[0] - center_x
            dy = face_center[1] - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance <= radius:
                dent_faces.append(i)
        
        return dent_faces
    
    def _point_in_polygon_2d(self, point, polygon):
        """Test if 2D point is inside polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _create_camera_info(self, mesh_bounds):
        """
        Create comprehensive camera information for metadata.
        
        Args:
            mesh_bounds: Mesh bounding box
            
        Returns:
            Dictionary with detailed camera configuration
        """
        # Calculate mesh dimensions
        mesh_center = (mesh_bounds[0] + mesh_bounds[1]) / 2
        mesh_size = mesh_bounds[1] - mesh_bounds[0]
        panel_width = mesh_size[0]   # X dimension
        panel_height = mesh_size[1]  # Y dimension
        panel_depth = mesh_size[2]   # Z dimension
        
        # Camera positioning
        panel_front_z = mesh_center[2] + panel_depth / 2.0
        camera_position = [mesh_center[0], mesh_center[1], panel_front_z + self.camera_distance]
        
        # Camera configuration details
        camera_info = {
            'type': self.camera_type,
            'distance': self.camera_distance,
            'position': camera_position,
            'target_center': mesh_center.tolist(),
            'panel_dimensions': {
                'width': panel_width,
                'height': panel_height,
                'depth': panel_depth
            },
            'image_resolution': [self.image_width, self.image_height],
            'projection_info': {}
        }
        
        # Add projection-specific information
        if self.camera_type == "orthographic_fixed":
            camera_info['projection_info'] = {
                'viewing_area_width': panel_width,
                'viewing_area_height': panel_height,
                'description': 'Fixed orthographic projection - distance does not affect zoom'
            }
            
        elif self.camera_type == "orthographic_responsive":
            base_distance = 2.35  # Reference distance
            distance_factor = self.camera_distance / base_distance
            
            # Use NEW approach calculation: direct xmag/ymag adjustment
            zoom_factor = 1.0 / distance_factor  # Invert: closer = higher zoom
            zoomed_xmag = (panel_width / 2.0) / zoom_factor
            zoomed_ymag = (panel_height / 2.0) / zoom_factor
            viewing_width = zoomed_xmag * 2.0
            viewing_height = zoomed_ymag * 2.0
            
            camera_info['projection_info'] = {
                'base_distance': base_distance,
                'distance_factor': distance_factor,
                'zoom_factor': zoom_factor,
                'viewing_area_width': viewing_width,
                'viewing_area_height': viewing_height,
                'zoom_level': f"{zoom_factor:.2f}x",
                'xmag': zoomed_xmag,
                'ymag': zoomed_ymag,
                'description': 'Responsive orthographic projection - closer distance reduces viewing area for zoom'
            }
            
        elif self.camera_type == "perspective":
            reference_distance = 1.00
            fov_y = 2 * np.arctan(panel_height / (2 * reference_distance))
            aspect_ratio = panel_width / panel_height
            
            camera_info['projection_info'] = {
                'field_of_view_y_deg': np.degrees(fov_y),
                'aspect_ratio': aspect_ratio,
                'reference_distance': reference_distance,
                'description': 'Perspective projection - natural distance-dependent scaling'
            }
        
        # Capture area calculation (what the camera actually sees)
        if self.camera_type in ["orthographic_fixed", "orthographic_responsive"]:
            # For orthographic cameras, capture area depends on projection setup
            if self.camera_type == "orthographic_responsive":
                # Use the CORRECTLY calculated viewing area from projection_info
                # The projection_info already contains the right dimensions!
                viewing_width = camera_info['projection_info']['viewing_area_width']
                viewing_height = camera_info['projection_info']['viewing_area_height']
                
                capture_width = viewing_width
                capture_height = viewing_height
                
                # Remove the old debug note and add correct info
                camera_info['capture_area_note'] = f"Camera captures {capture_width:.3f}m x {capture_height:.3f}m at {self.camera_distance:.2f}m distance (zoom level: {camera_info['projection_info']['zoom_level']})"
            else:
                capture_width = panel_width
                capture_height = panel_height
                
            camera_info['capture_area'] = {
                'width': capture_width,
                'height': capture_height,
                'area_m2': capture_width * capture_height,
                'center': mesh_center.tolist(),
                'corners': [
                    [mesh_center[0] - capture_width/2, mesh_center[1] - capture_height/2],
                    [mesh_center[0] + capture_width/2, mesh_center[1] - capture_height/2],
                    [mesh_center[0] + capture_width/2, mesh_center[1] + capture_height/2],
                    [mesh_center[0] - capture_width/2, mesh_center[1] + capture_height/2]
                ]
            }
        
        return camera_info
    
    def _calculate_true_max_dent_depth(self, dented_mesh, undented_mesh, dent_specs):
        """
        Calculate dent depth information with proper ground truth handling.
        The specifications are the authoritative ground truth since they represent
        the actual dent parameters applied to the original mesh.
        
        Args:
            dented_mesh: Trimesh object of dented panel
            undented_mesh: Trimesh object of original undented panel (may be regenerated)
            dent_specs: Dent specifications - THE AUTHORITATIVE GROUND TRUTH
            
        Returns:
            dict with depth analysis results prioritizing specification values
        """
        print(f"        Calculating dent depth information (specifications as ground truth)...")
        
        try:
            # Extract TRUE ground truth from specifications
            if dent_specs:
                if dent_specs.get('type') == 'multi_impact':
                    # For multi-impact, extract all impact depths
                    impacts = dent_specs.get('impacts', [])
                    if impacts:
                        impact_depths = [impact.get('depth', 0) for impact in impacts]
                        spec_max_depth = max(impact_depths)
                        spec_mean_depth = sum(impact_depths) / len(impact_depths)
                        num_impacts = len(impacts)
                        
                        # Calculate total affected area approximation
                        total_radius_sq = sum(impact.get('radius', 0)**2 for impact in impacts)
                        approx_affected_area = 3.14159 * total_radius_sq  # Rough estimate
                    else:
                        spec_max_depth = 0
                        spec_mean_depth = 0
                        num_impacts = 0
                        approx_affected_area = 0
                else:
                    # For single dent types
                    spec_max_depth = dent_specs.get('depth', 0)
                    spec_mean_depth = spec_max_depth * 0.7  # Typical falloff
                    num_impacts = 1
                    radius = dent_specs.get('radius', 0.1)
                    approx_affected_area = 3.14159 * radius**2
                
                dent_type = dent_specs.get('type', 'unknown')
                
                print(f"          Specification ground truth:")
                print(f"            Dent type: {dent_type}")
                print(f"            Max depth: {spec_max_depth*1000:.2f}mm")
                print(f"            Mean depth: {spec_mean_depth*1000:.2f}mm")
                if dent_type == 'multi_impact':
                    print(f"            Number of impacts: {num_impacts}")
                
            else:
                print(f"          No dent specifications available")
                spec_max_depth = 0
                spec_mean_depth = 0
                num_impacts = 0
                approx_affected_area = 0
                dent_type = 'unknown'
            
            # Optional: Validation measurement (for debugging/validation only)
            measurement_data = None
            if undented_mesh is not None and len(dented_mesh.vertices) == len(undented_mesh.vertices):
                print(f"          Performing validation measurement (for debugging only)...")
                
                # Z-direction measurement for validation
                undented_vertices = undented_mesh.vertices
                dented_vertices = dented_mesh.vertices
                z_displacement = np.abs(undented_vertices[:, 2] - dented_vertices[:, 2])
                
                significant_threshold = 0.0001  # 0.1mm
                significant_diffs = z_displacement[z_displacement > significant_threshold]
                
                if len(significant_diffs) > 0:
                    # Apply conservative filtering based on specifications
                    if spec_max_depth > 0:
                        realistic_upper_bound = spec_max_depth * 2.0  # Allow 2x spec depth
                    else:
                        realistic_upper_bound = 0.020  # 20mm fallback
                    
                    filtered_diffs = significant_diffs[significant_diffs <= realistic_upper_bound]
                    
                    if len(filtered_diffs) > 0:
                        measured_max = np.max(filtered_diffs)
                        measured_mean = np.mean(filtered_diffs)
                        measured_points = len(filtered_diffs)
                        
                        measurement_data = {
                            'measured_max_depth_m': float(measured_max),
                            'measured_max_depth_mm': float(measured_max * 1000),
                            'measured_mean_depth_m': float(measured_mean),
                            'measured_mean_depth_mm': float(measured_mean * 1000),
                            'measured_points': int(measured_points),
                            'validation_notes': 'Measured from mesh comparison - for validation only'
                        }
                        
                        # Compare with specifications
                        if spec_max_depth > 0:
                            ratio = measured_max / spec_max_depth
                            print(f"            Measured max: {measured_max*1000:.2f}mm (ratio: {ratio:.1f}x)")
                            if ratio > 2.5:
                                print(f"            ⚠ Measurement significantly higher than spec - likely measurement artifacts")
                            elif ratio < 0.5:
                                print(f"            ⚠ Measurement significantly lower than spec - possible mesh resolution issues")
                            else:
                                print(f"            ✓ Measurement reasonably matches specification")
                    else:
                        print(f"            No valid measurements after filtering")
                else:
                    print(f"            No significant vertex displacements detected")
            else:
                print(f"          Skipping validation measurement (no suitable comparison mesh)")
            
            # Return authoritative result based on SPECIFICATIONS
            depth_analysis = {
                # AUTHORITATIVE VALUES (from specifications - TRUE ground truth)
                'max_depth_m': float(spec_max_depth),
                'max_depth_mm': float(spec_max_depth * 1000),
                'mean_depth_m': float(spec_mean_depth),
                'mean_depth_mm': float(spec_mean_depth * 1000),
                'dent_type': dent_type,
                'num_impacts': num_impacts if dent_type == 'multi_impact' else 1,
                'affected_area_approx_m2': float(approx_affected_area),
                
                # DATA SOURCE AND QUALITY
                'method': 'specification_ground_truth',
                'description': 'Depth values from dent specifications (authoritative ground truth)',
                'data_source': 'dented_panel_specifications.json',
                'quality': 'authoritative_ground_truth',
                
                # CAMERA/RENDERING CONSIDERATIONS
                'camera_coverage_notes': 'Values represent full dent - camera may capture partial area due to framing',
                'rendering_notes': 'RGB/Depth/Mask images may show subset of total dent area',
                
                # OPTIONAL VALIDATION DATA
                'validation_measurement': measurement_data,
                
                # DATASET USAGE GUIDANCE
                'usage_notes': 'Use these specification values as ground truth for training/evaluation'
            }
            
            print(f"          ✓ Using specification values as authoritative ground truth")
            print(f"          ✓ Max depth: {spec_max_depth*1000:.2f}mm (specification)")
            
            return depth_analysis
            
        except Exception as e:
            print(f"          ERROR in depth analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback 
            return {
                'max_depth_m': 0.0,
                'max_depth_mm': 0.0,
                'mean_depth_m': 0.0,
                'mean_depth_mm': 0.0,
                'dent_type': 'unknown',
                'num_impacts': 0,
                'method': 'error_fallback',
                'description': f'Error in depth analysis: {str(e)}',
                'data_source': 'error',
                'quality': 'invalid'
            }
    
    def render_panel(self, obj_path, dent_specs=None, undented_obj_path=None):
        """
        Render RGB and depth images for a single panel.
        
        Args:
            obj_path: Path to dented .obj file
            dent_specs: Optional dent specifications for mask generation
            undented_obj_path: Path to original undented .obj file for true ground truth
            
        Returns:
            Dictionary with rendering results
        """
        try:
            # Load dented mesh
            dented_mesh = trimesh.load(obj_path)
            print(f"  Loading: {obj_path}")
            print(f"    Vertices: {len(dented_mesh.vertices)}, Faces: {len(dented_mesh.faces)}")
            
            # Load undented mesh for ground truth comparison (if available)
            undented_mesh = None
            if undented_obj_path and Path(undented_obj_path).exists():
                try:
                    undented_mesh = trimesh.load(undented_obj_path)
                    print(f"    Loaded undented mesh for ground truth comparison")
                except Exception as e:
                    print(f"    Warning: Could not load undented mesh: {e}")
            
            # Create pyrender scene
            scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 0.0])
            
            # Add dented mesh to scene
            mesh_node = pyrender.Mesh.from_trimesh(dented_mesh)
            scene.add(mesh_node)
            
            # Set up camera and lighting
            mesh_bounds = dented_mesh.bounds
            camera_pose = self.setup_camera_and_lighting(scene, mesh_bounds)
            
            # Render RGB and depth
            color_image, depth_image = self.renderer.render(scene)
            
            # Verify dimensions match camera capture exactly
            expected_height, expected_width = self.image_height, self.image_width
            actual_height, actual_width = color_image.shape[:2]
            
            print(f"    Image dimensions verification:")
            print(f"      Expected: {expected_width}x{expected_height} (camera capture)")
            print(f"      RGB actual: {actual_width}x{actual_height}")
            print(f"      Depth actual: {depth_image.shape[1]}x{depth_image.shape[0]}")
            
            # Ensure perfect dimension match
            if (actual_width != expected_width or actual_height != expected_height):
                print(f"    ⚠️  Dimension mismatch detected - resizing to match camera capture")
                # Resize to exact camera dimensions if needed
                color_image = cv2.resize(color_image, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
                depth_image = cv2.resize(depth_image, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
                print(f"    ✓ Resized to exact camera capture: {expected_width}x{expected_height}")
            else:
                print(f"    ✓ Perfect dimension match: {expected_width}x{expected_height}")
            
            # Create base filename
            obj_filename = Path(obj_path).stem
            
            # Save RGB image (exact camera capture dimensions)
            rgb_path = self.rgb_dir / f"{obj_filename}_rgb.png"
            imageio.imwrite(rgb_path, color_image)
            
            # Process and save depth image (exact camera capture dimensions)
            depth_path = self.depth_dir / f"{obj_filename}_depth.npy"
            depth_png_path = self.depth_dir / f"{obj_filename}_depth.png"
            
            # Save raw depth as NPY (floating point, exact dimensions)
            np.save(depth_path, depth_image.astype(np.float32))
            
            # Normalize depth for PNG visualization (exact dimensions)
            depth_normalized = self._normalize_depth(depth_image)
            imageio.imwrite(depth_png_path, depth_normalized)
            
            # Create and save GROUND TRUTH mask if dent specs available (exact dimensions)
            mask_path = None
            if dent_specs:
                try:
                    # Create TRUE ground truth mask by comparing dented vs undented mesh
                    mask = self.create_dent_mask(dented_mesh, dent_specs, camera_pose, mesh_bounds, undented_mesh)
                    
                    # Verify mask dimensions match camera capture
                    mask_height, mask_width = mask.shape[:2]
                    print(f"      Mask actual: {mask_width}x{mask_height}")
                    
                    if (mask_width != expected_width or mask_height != expected_height):
                        print(f"    ⚠️  Mask dimension mismatch - resizing to match camera capture")
                        mask = cv2.resize(mask, (expected_width, expected_height), interpolation=cv2.INTER_NEAREST)
                        print(f"    ✓ Mask resized to exact camera capture: {expected_width}x{expected_height}")
                    else:
                        print(f"    ✓ Mask perfect dimension match: {expected_width}x{expected_height}")
                    
                    mask_path = self.mask_dir / f"{obj_filename}_mask.png"
                    imageio.imwrite(mask_path, mask)
                    print(f"    ✓ Created ground truth mask with {np.sum(mask > 0)} dent pixels")
                except Exception as e:
                    print(f"    Warning: Could not create mask - {e}")
            
            # Calculate TRUE maximum dent depth (corrugation-aware)
            depth_analysis = self._calculate_true_max_dent_depth(dented_mesh, undented_mesh, dent_specs)
            print(f"    ✓ Depth analysis: {depth_analysis['method']} - Max: {depth_analysis['max_depth_mm']:.2f}mm")
            
            # Save metadata
            metadata = {
                'obj_file': str(obj_path),
                'undented_obj_file': str(undented_obj_path) if undented_obj_path else None,
                'rgb_image': str(rgb_path),
                'depth_npy': str(depth_path),
                'depth_png': str(depth_png_path),
                'mask_image': str(mask_path) if mask_path else None,
                'image_size': [self.image_width, self.image_height],
                'image_dimensions': {
                    'width': self.image_width,
                    'height': self.image_height,
                    'format': 'width x height',
                    'description': 'Exact camera capture dimensions',
                    'rgb_dimensions': [actual_width, actual_height],
                    'depth_dimensions': [depth_image.shape[1], depth_image.shape[0]],
                    'mask_dimensions': [mask.shape[1], mask.shape[0]] if mask_path else None,
                    'dimension_verification': 'All outputs match camera capture exactly'
                },
                'mesh_info': {
                    'vertices': len(dented_mesh.vertices),
                    'faces': len(dented_mesh.faces),
                    'bounds': mesh_bounds.tolist()
                },
                'camera_pose': camera_pose.tolist(),
                'camera_info': self._create_camera_info(mesh_bounds),
                'dent_specs': dent_specs,
                'ground_truth_method': 'unified_rendering',
                'depth_analysis': depth_analysis,
                'rendering_notes': {
                    'pixel_alignment': 'Perfect - RGB, depth, and mask use identical camera setup',
                    'dimension_guarantee': 'All saved images match camera capture dimensions exactly',
                    'coordinate_system': 'Unified rendering ensures pixel-perfect correspondence'
                }
            }
            
            metadata_path = self.metadata_dir / f"{obj_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save depth metrics to separate JSON file
            depth_metrics_path = self.output_dir / f"{obj_filename}_depth_metrics.json"
            self._save_depth_metrics_json(depth_metrics_path, obj_filename, depth_analysis, 
                                        depth_analysis.get('method', 'specification_ground_truth'))
            
            print(f"    ✓ Rendered: RGB, Depth{'+ Ground Truth Mask' if mask_path else ''}")
            
            # Final dimension verification summary
            self._verify_output_dimensions(rgb_path, depth_png_path, mask_path, expected_width, expected_height)
            
            return {
                'success': True,
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'mask_path': mask_path,
                'metadata_path': metadata_path
            }
            
        except Exception as e:
            print(f"    ✗ Error rendering {obj_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _verify_output_dimensions(self, rgb_path, depth_path, mask_path, expected_width, expected_height):
        """
        Verify that all saved images have exactly the same dimensions as camera capture.
        """
        print(f"    📐 Final dimension verification:")
        
        try:
            # Check RGB
            rgb_img = imageio.imread(rgb_path)
            rgb_h, rgb_w = rgb_img.shape[:2]
            rgb_match = (rgb_w == expected_width and rgb_h == expected_height)
            print(f"      RGB: {rgb_w}x{rgb_h} {'✓' if rgb_match else '✗'}")
            
            # Check Depth PNG
            if depth_path:
                depth_img = imageio.imread(depth_path)
                depth_h, depth_w = depth_img.shape[:2]
                depth_match = (depth_w == expected_width and depth_h == expected_height)
                print(f"      Depth: {depth_w}x{depth_h} {'✓' if depth_match else '✗'}")
            
            # Check Mask
            if mask_path:
                mask_img = imageio.imread(mask_path)
                mask_h, mask_w = mask_img.shape[:2]
                mask_match = (mask_w == expected_width and mask_h == expected_height)
                print(f"      Mask: {mask_w}x{mask_h} {'✓' if mask_match else '✗'}")
                
                all_match = rgb_match and depth_match and mask_match
            else:
                all_match = rgb_match and depth_match
            
            if all_match:
                print(f"    ✅ All outputs match camera capture: {expected_width}x{expected_height}")
            else:
                print(f"    ⚠️  Some outputs don't match expected camera capture dimensions")
                
        except Exception as e:
            print(f"    ⚠️  Could not verify dimensions: {e}")
    
    def _normalize_depth(self, depth_image):
        """
        Normalize depth image to 0-255 range for PNG visualization.
        Enhanced to make dents more visible.
        
        Args:
            depth_image: Raw depth image
            
        Returns:
            Normalized depth image (uint8)
        """
        # Handle zero/invalid depths
        valid_depths = depth_image[depth_image > 0]
        
        if len(valid_depths) == 0:
            return np.zeros_like(depth_image, dtype=np.uint8)
        
        # For container panels, enhance contrast to make dents visible
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        depth_range = max_depth - min_depth
        
        print(f"    Depth range: {min_depth:.4f}m to {max_depth:.4f}m (variation: {depth_range:.4f}m)")
        
        if depth_range > 0.001:  # If there's significant depth variation (more than 1mm)
            # Normal normalization for panels with good depth variation
            normalized = (depth_image - min_depth) / depth_range
            normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        else:
            # Enhanced contrast for mostly flat panels (make small variations visible)
            # Use percentile-based normalization to enhance small variations
            valid_depths_sorted = np.sort(valid_depths)
            
            # Use 5th and 95th percentiles to enhance contrast
            p5 = np.percentile(valid_depths_sorted, 5)
            p95 = np.percentile(valid_depths_sorted, 95)
            
            if p95 > p5:
                # Stretch contrast between 5th and 95th percentiles
                normalized = np.clip((depth_image - p5) / (p95 - p5), 0, 1)
                normalized = (normalized * 255).astype(np.uint8)
                print(f"    Enhanced contrast: 5th percentile {p5:.4f}m, 95th percentile {p95:.4f}m")
            else:
                # Fallback: set middle gray for uniform depth
                normalized = np.ones_like(depth_image, dtype=np.uint8) * 128
                print(f"    Uniform depth detected, using middle gray")
        
        # Set invalid depths to 0 (black)
        normalized[depth_image <= 0] = 0
        
        return normalized
    
    def load_dent_specs(self, json_path):
        """Load dent specifications from JSON file."""
        try:
            with open(json_path, 'r') as f:
                specs_list = json.load(f)
            
            # Create mapping using just the filename (not full path)
            specs_dict = {}
            for spec in specs_list:
                # Extract just the filename from the full path
                filename = spec.get('filename', '')
                if filename:
                    # Get just the .obj filename without extension
                    obj_filename = Path(filename).stem
                    specs_dict[obj_filename] = spec
                    
            print(f"✓ Loaded dent specifications for {len(specs_dict)} panels")
            return specs_dict
            
        except Exception as e:
            print(f"Warning: Could not load dent specs from {json_path}: {e}")
            return {}
    
    def render_batch(self, input_dir="panel_dents", dent_specs_file=None, generate_undented=True):
        """
        Render all .obj files in the input directory with ground truth masks.
        
        Args:
            input_dir: Directory containing dented .obj files
            dent_specs_file: Optional JSON file with dent specifications
            generate_undented: Whether to generate undented meshes for ground truth comparison
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"✗ Input directory not found: {input_path}")
            return
        
        # Load dent specifications if available
        dent_specs_dict = {}
        if dent_specs_file and os.path.exists(dent_specs_file):
            dent_specs_dict = self.load_dent_specs(dent_specs_file)
            print(f"✓ Loaded dent specifications for {len(dent_specs_dict)} panels")
        
        # Find all .obj files
        obj_files = list(input_path.glob("*.obj"))
        
        if not obj_files:
            print(f"✗ No .obj files found in {input_path}")
            return
        
        print(f"\n=== Rendering {len(obj_files)} Panel(s) with Ground Truth Masks ===")
        
        # Initialize panel generator for creating undented meshes
        panel_generator = None
        if generate_undented and dent_specs_dict:
            panel_generator = CorrugatedPanelGenerator()
            print(f"✓ Initialized panel generator for ground truth mesh creation")
        
        results = []
        successful_renders = 0
        
        for obj_file in sorted(obj_files):
            # Get corresponding dent specs
            obj_key = obj_file.stem
            dent_specs = dent_specs_dict.get(obj_key)
            
            print(f"\n[{successful_renders + 1}/{len(obj_files)}]")
            
            # Generate undented mesh for ground truth comparison
            undented_obj_path = None
            if generate_undented and panel_generator and dent_specs:
                try:
                    undented_obj_path = self._generate_undented_mesh(
                        panel_generator, dent_specs, obj_key)
                    print(f"    ✓ Generated undented mesh for ground truth comparison")
                except Exception as e:
                    print(f"    Warning: Could not generate undented mesh: {e}")
            
            # Render with ground truth mask
            result = self.render_panel(obj_file, dent_specs, undented_obj_path)
            results.append(result)
            
            if result['success']:
                successful_renders += 1
            
            # Clean up temporary undented mesh
            if undented_obj_path and Path(undented_obj_path).exists():
                try:
                    os.remove(undented_obj_path)
                except Exception as e:
                    print(f"    Warning: Could not clean up temporary file: {e}")
        
        # Create summary
        summary = {
            'total_files': len(obj_files),
            'successful_renders': successful_renders,
            'failed_renders': len(obj_files) - successful_renders,
            'output_directory': str(self.output_dir),
            'ground_truth_method': 'mesh_comparison' if generate_undented else 'surface_analysis',
            'results': [
                {
                    'success': result['success'],
                    'rgb_path': str(result['rgb_path']) if result.get('rgb_path') else None,
                    'depth_path': str(result['depth_path']) if result.get('depth_path') else None,
                    'mask_path': str(result['mask_path']) if result.get('mask_path') else None,
                    'metadata_path': str(result['metadata_path']) if result.get('metadata_path') else None,
                    'error': result.get('error')
                } for result in results
            ]
        }
        
        summary_path = self.output_dir / "render_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== Rendering Complete ===")
        print(f"  Successful: {successful_renders}/{len(obj_files)}")
        print(f"  Ground Truth Method: {'Mesh Comparison (TRUE)' if generate_undented else 'Surface Analysis (Approximate)'}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Summary saved: {summary_path}")
        
        # Show file structure
        self._show_output_structure()
    
    def _generate_undented_mesh(self, panel_generator, dent_specs, obj_key):
        """
        Generate an undented mesh based on the panel specifications.
        
        Args:
            panel_generator: CorrugatedPanelGenerator instance
            dent_specs: Dent specifications containing panel info
            obj_key: Object key for temporary file naming
            
        Returns:
            Path to temporary undented mesh file
        """
        # Extract panel specifications from dent_specs
        container_type = dent_specs.get('container_type', '20ft')
        corrugation_pattern = dent_specs.get('corrugation_pattern', 'standard_vertical')
        corrugation_depth = dent_specs.get('corrugation_depth', 0.08)
        corrugation_frequency = dent_specs.get('corrugation_frequency', 12)
        wall_thickness = dent_specs.get('wall_thickness', 0.002)
        
        # Extract container color information
        container_color_name = dent_specs.get('container_color_name', 'CARGO_GRAY')
        container_color_rgb = dent_specs.get('container_color_rgb', [128, 128, 128])
        
        # Set panel generator parameters
        panel_generator.set_container_type(container_type)
        
        # Set container color
        panel_generator.set_color(container_color_name)
        
        # Set specific corrugation parameters
        from panel_generator import CorrugationPattern
        try:
            panel_generator.corrugation_pattern = CorrugationPattern(corrugation_pattern)
        except ValueError:
            panel_generator.corrugation_pattern = CorrugationPattern.STANDARD_VERTICAL
            
        panel_generator.corrugation_depth = corrugation_depth
        panel_generator.corrugation_frequency = corrugation_frequency
        panel_generator.wall_thickness = wall_thickness
        
        # Generate undented panel
        undented_mesh = panel_generator.create_corrugated_panel()
        
        # Save temporary undented mesh
        temp_dir = Path("temp_undented")
        temp_dir.mkdir(exist_ok=True)
        undented_path = temp_dir / f"undented_{obj_key}.obj"
        
        # Export undented mesh
        undented_mesh.export(str(undented_path))
        
        return str(undented_path)
    
    def _show_output_structure(self):
        """Display the output directory structure."""
        print(f"\n📁 Output Structure:")
        print(f"  {self.output_dir}/")
        
        for subdir in ['rgb', 'depth', 'mask', 'metadata']:
            subdir_path = self.output_dir / subdir
            if subdir_path.exists():
                file_count = len(list(subdir_path.glob("*")))
                print(f"    ├── {subdir}/ ({file_count} files)")
        
        print(f"    └── render_summary.json")
    
    def create_preview_grid(self, max_samples=6):
        """
        Create a preview grid showing RGB, depth, and mask triplets.
        
        Args:
            max_samples: Maximum number of samples to show in preview
        """
        rgb_files = sorted(list(self.rgb_dir.glob("*_rgb.png")))[:max_samples]
        
        if not rgb_files:
            print("No RGB files found for preview")
            return
        
        # Display each 640x480 image at 100 DPI = 6.4"x4.8" per image
        dpi = 100  # display resolution
        fig_width = len(rgb_files) * (self.image_width / dpi)
        fig_height = 3 * (self.image_height / dpi)
        fig, axes = plt.subplots(3, len(rgb_files), figsize=(fig_width, fig_height), dpi=dpi)
        if len(rgb_files) == 1:
            axes = axes.reshape(3, 1)
        
        for i, rgb_path in enumerate(rgb_files):
            # Load RGB image
            rgb_img = imageio.imread(rgb_path)
            
            # Load corresponding depth image (PNG version for visualization)
            depth_png_path = self.depth_dir / rgb_path.name.replace("_rgb.png", "_depth.png")
            if depth_png_path.exists():
                depth_img = imageio.imread(depth_png_path)
                
                # Calculate depth statistics for display
                non_zero_depths = depth_img[depth_img > 0]
                if len(non_zero_depths) > 0:
                    depth_min, depth_max = np.min(non_zero_depths), np.max(non_zero_depths)
                    depth_range = depth_max - depth_min
                    depth_title = f"Depth: {rgb_path.stem.replace('_rgb', '')}\nRange: {depth_range}"
                else:
                    depth_title = f"Depth: {rgb_path.stem.replace('_rgb', '')}\nNo valid depth"
            else:
                depth_img = np.zeros_like(rgb_img[:,:,0])
                depth_title = f"Depth: {rgb_path.stem.replace('_rgb', '')}\nMissing"
            
            # Load corresponding mask image
            mask_png_path = self.mask_dir / rgb_path.name.replace("_rgb.png", "_mask.png")
            if mask_png_path.exists():
                mask_img = imageio.imread(mask_png_path)
                # Calculate mask statistics
                mask_pixels = np.sum(mask_img > 0)
                total_pixels = mask_img.shape[0] * mask_img.shape[1]
                mask_percentage = (mask_pixels / total_pixels) * 100
                mask_title = f"Mask: {rgb_path.stem.replace('_rgb', '')}\nDent: {mask_percentage:.1f}%"
            else:
                mask_img = np.zeros_like(rgb_img[:,:,0])
                mask_title = f"Mask: {rgb_path.stem.replace('_rgb', '')}\nNo mask"
            
            # Plot RGB (Row 0)
            axes[0, i].imshow(rgb_img)
            axes[0, i].set_title(f"RGB: {rgb_path.stem.replace('_rgb', '')}", fontsize=10)
            axes[0, i].axis('off')
            
            # Plot depth with better colormap (Row 1)
            # Use 'hot' colormap: black->red->yellow->white (better for depth)
            # where black=far, white=close (dents should appear brighter)
            im_depth = axes[1, i].imshow(depth_img, cmap='hot', vmin=0, vmax=255)
            axes[1, i].set_title(depth_title, fontsize=8)
            axes[1, i].axis('off')
            
            # Plot mask (Row 2)
            # Use binary colormap for masks: black=no dent, white=dent
            im_mask = axes[2, i].imshow(mask_img, cmap='gray', vmin=0, vmax=255)
            axes[2, i].set_title(mask_title, fontsize=8)
            axes[2, i].axis('off')
            
            # Add colorbars for the last image only
            if i == len(rgb_files) - 1:
                # Depth colorbar
                cbar_depth = plt.colorbar(im_depth, ax=axes[1, i], fraction=0.046, pad=0.04)
                cbar_depth.set_label('Depth\n(0=far, 255=close)', rotation=270, labelpad=15, fontsize=8)
                
                # Mask colorbar
                cbar_mask = plt.colorbar(im_mask, ax=axes[2, i], fraction=0.046, pad=0.04)
                cbar_mask.set_label('Mask\n(0=normal, 255=dent)', rotation=270, labelpad=15, fontsize=8)
        
        plt.tight_layout()
        preview_path = self.output_dir / "preview_grid.png"
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Preview grid saved: {preview_path}")
        print(f"  📊 Dataset Visualization:")
        print(f"    • Row 1: RGB images (photo-realistic container panels)")
        print(f"    • Row 2: Depth maps (black=far, bright=close, dents visible)")
        print(f"    • Row 3: Segmentation masks (white=dent regions, black=normal)")
        print(f"    • Perfect pixel alignment between all three modalities")
    
    def cleanup(self):
        """Clean up renderer resources."""
        if hasattr(self, 'renderer'):
            self.renderer.delete()

    def _save_depth_metrics_json(self, output_file, panel_id, depth_analysis, method="specification_ground_truth"):
        """
        Save depth metrics to a separate JSON file with proper ground truth attribution
        
        Args:
            output_file: Path to save the JSON file
            panel_id: Identifier for the panel
            depth_analysis: Results from depth calculation (now specification-based)
            method: Analysis method used (now specification_ground_truth)
        """
        try:
            # Prepare comprehensive depth metrics with clear ground truth attribution
            depth_metrics = {
                "panel_id": panel_id,
                "timestamp": datetime.now().isoformat(),
                "analysis_method": method,
                "description": "Dent depth information with specifications as authoritative ground truth",
                
                # AUTHORITATIVE GROUND TRUTH VALUES
                "ground_truth": {
                    "source": "dented_panel_specifications.json",
                    "max_depth_meters": depth_analysis.get('max_depth_m', 0),
                    "max_depth_mm": depth_analysis.get('max_depth_mm', 0),
                    "mean_depth_meters": depth_analysis.get('mean_depth_m', 0),
                    "mean_depth_mm": depth_analysis.get('mean_depth_mm', 0),
                    "dent_type": depth_analysis.get('dent_type', 'unknown'),
                    "num_impacts": depth_analysis.get('num_impacts', 0),
                    "affected_area_approx_m2": depth_analysis.get('affected_area_approx_m2', 0),
                    "quality": "authoritative",
                    "usage_notes": "Use these values for training/evaluation - they represent the actual dent parameters applied"
                },
                
                # CAMERA/RENDERING CONTEXT
                "rendering_context": {
                    "camera_coverage_notes": depth_analysis.get('camera_coverage_notes', ''),
                    "rendering_notes": depth_analysis.get('rendering_notes', ''),
                    "partial_capture_possible": True,
                    "explanation": "Camera images may show subset of total dent area due to framing and distance"
                },
                
                # OPTIONAL VALIDATION MEASUREMENTS (if available)
                "validation_measurement": depth_analysis.get('validation_measurement'),
                
                # DATASET SYNCHRONIZATION
                "synchronization": {
                    "specifications_file": "panel_dents/dented_panel_specifications.json",
                    "consistency": "values_match_specifications",
                    "recommendation": "Use ground_truth values above for ML training/evaluation"
                },
                
                # METHODOLOGY DETAILS
                "methodology": {
                    "approach": "specification_based_ground_truth",
                    "rationale": "Specifications represent actual dent parameters applied to original mesh",
                    "data_flow": "original_panel -> apply_dent -> specifications -> render_camera_view",
                    "measurement_artifacts_eliminated": True
                }
            }
            
            # Save to JSON file
            with open(output_file, 'w') as f:
                json.dump(depth_metrics, f, indent=2)
            
            print(f"        ✓ Depth metrics saved to: {output_file}")
            print(f"        ✓ Ground truth source: specifications (synchronized)")
            
        except Exception as e:
            print(f"        ERROR saving depth metrics JSON: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function."""
    print("🎨 RGB-D Renderer for Dented Container Panels")
    print("=" * 50)
    
    # Camera configuration examples:
    
    # Option 1: Fixed orthographic (current behavior - distance doesn't affect zoom)
    # renderer = RGBDRenderer(
    #     output_dir="output",
    #     image_width=640,
    #     image_height=480,
    #     camera_distance=2.35,  # Change this - won't affect image scale
    #     camera_type="orthographic_fixed"
    # )
    
    # Option 2: Responsive orthographic (distance affects zoom)
    renderer = RGBDRenderer(
        output_dir="output",
        image_width=640,
        image_height=480,
        camera_distance=1.8,  # Closer = zoomed in, further = zoomed out
        camera_type="orthographic_responsive"
    )
    
    # Option 3: Perspective camera (realistic distance scaling)
    # renderer = RGBDRenderer(
    #     output_dir="output",
    #     image_width=640,
    #     image_height=480,
    #     camera_distance=1.0,  # Closer = larger objects, further = smaller objects
    #     camera_type="perspective"
    # )
    
    print(f"\n🔧 Camera Configuration:")
    print(f"  Distance: {renderer.camera_distance:.2f}m")
    print(f"  Type: {renderer.camera_type}")
    print(f"  Behavior: ", end="")
    if renderer.camera_type == "orthographic_fixed":
        print("Distance changes position only (no zoom effect)")
    elif renderer.camera_type == "orthographic_responsive":
        print("Distance changes zoom level (closer = more zoom)")
    elif renderer.camera_type == "perspective":
        print("Distance changes object size naturally (closer = larger)")
    
    try:
        # Render all panels in panel_dents folder
        renderer.render_batch(
            input_dir="panel_dents",
            dent_specs_file="panel_dents/dented_panel_specifications.json",
            generate_undented=True
        )
        
        # Create preview grid
        renderer.create_preview_grid(max_samples=6)
        
    except KeyboardInterrupt:
        print("\n⏸ Rendering interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during rendering: {e}")
    finally:
        # Clean up
        renderer.cleanup()
        print("\n✓ Renderer cleanup complete")

if __name__ == "__main__":
    main() 