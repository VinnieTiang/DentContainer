"""
Renders container scenes using PyRender (same library as RGB-D rendering).
This is an alternative to PyTorch3D that uses pyrender instead.
"""

import numpy as np
from pathlib import Path
import trimesh
import pyrender
from PIL import Image
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime

from config import ContainerConfig, RendererConfig
from camera_position import CameraPoseGenerator

logger = logging.getLogger(__name__)


def look_at_matrix(eye: np.ndarray, at: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build a pyrender camera pose (4x4 world_from_camera) from eye, at, up (numpy arrays).
    
    Args:
        eye: Camera position (3,)
        at: Target point (3,)
        up: Up vector (3,)
    
    Returns:
        4x4 transformation matrix (world_from_camera)
    """
    # eye, at, up are (3,)
    z = (eye - at).astype(np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    
    # 3x3 rotation where columns are camera axes in world coords (right, up, back)
    R = np.stack([x, y, z], axis=1)  # world_from_camera rotation
    T = eye.reshape(3, 1)
    
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R
    mat[:3, 3:] = T
    return mat


class SceneRendererPyRender:
    """Renders a 3D container mesh into a scene using PyRender."""

    def __init__(self, renderer_config: RendererConfig, container_config: ContainerConfig):
        self.r_config = renderer_config
        self.c_config = container_config
        
        # Initialize pyrender offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.r_config.IMAGE_SIZE,
            viewport_height=self.r_config.IMAGE_SIZE
        )
        
        self.pose_generator = CameraPoseGenerator(self.c_config, self.r_config)
        
        logger.info("SceneRendererPyRender initialized")

    def render_and_save(self, mesh_path: Path, container_type: str, output_dir: Path, 
                       sample_id: int, defects_info: Optional[List[Dict]] = None, 
                       shot_name: Optional[str] = None):
        """
        Loads a mesh, places it in a scene, and renders it from multiple camera views,
        saving the RGB, depth, point cloud, mask, metadata, and annotation outputs for each view.
        """
        logger.info(f"Rendering all camera shots for sample {sample_id}: {mesh_path.name}")
        try:
            # 1. Load mesh
            scene_mesh = trimesh.load_mesh(mesh_path, process=False)
            
            # Load dent specifications if available
            dent_specs = self._load_dent_specs(mesh_path)
            
            # 2. Create ground plane
            ground_plane = self._create_ground_plane()
            
            # 3. Generate all camera poses
            poses = self.pose_generator.generate_poses(container_type)
            
            # 4. If a specific shot is requested, filter the poses
            if shot_name:
                poses = [p for p in poses if p['name'] == shot_name]
                if not poses:
                    logger.warning(f"Shot name '{shot_name}' not found. No images will be rendered for this sample.")
                    return
            
            # 5. Loop through each pose and render
            for pose in poses:
                current_shot_name = pose['name']
                logger.info(f"  - Rendering shot: {current_shot_name}")
                
                try:
                    # Render RGB and depth
                    rgb_image, depth_map = self._render_view(scene_mesh, ground_plane, pose)
                    
                    # Generate point cloud from depth
                    point_cloud = self._depth_to_pointcloud(depth_map, pose)
                    
                    # Generate dent mask if dent specs available
                    dent_mask = None
                    if dent_specs:
                        dent_mask = self._create_dent_mask(scene_mesh, dent_specs, pose)
                    
                    # Save all outputs
                    self._save_outputs(
                        rgb_image,
                        depth_map,
                        point_cloud,
                        output_dir,
                        sample_id,
                        current_shot_name,
                        annotations=None,  # Annotations not implemented for pyrender version
                        dent_mask=dent_mask,
                        dent_specs=dent_specs,
                        pose=pose,
                        mesh_path=mesh_path
                    )
                except Exception as e:
                    logger.error(f"Failed to render shot {current_shot_name}: {e}", exc_info=True)
                    continue

            logger.info(f"✓ Successfully rendered and saved all shots for sample {sample_id}")

        except Exception as e:
            logger.error(f"Failed to render sample {sample_id}: {e}", exc_info=True)

    def _create_ground_plane(self) -> trimesh.Trimesh:
        """Creates a large, simple ground plane mesh."""
        # Create a simple ground plane using trimesh box
        plane = trimesh.creation.box(extents=(100.0, 0.01, 100.0))
        plane.apply_translation([0, -0.005, 0])  # tiny thickness and move slightly down
        # Dark gray color
        plane.visual.vertex_colors = np.tile([77, 77, 77, 255], (len(plane.vertices), 1))
        return plane

    def _render_view(self, container_mesh: trimesh.Trimesh, ground_plane: Optional[trimesh.Trimesh], 
                    pose: Dict) -> tuple[np.ndarray, np.ndarray]:
        """Renders a single view using pyrender."""
        # Convert pose tensors to numpy if needed
        eye = pose['eye'].cpu().numpy()[0] if hasattr(pose['eye'], 'cpu') else np.asarray(pose['eye'])
        at = pose['at'].cpu().numpy()[0] if hasattr(pose['at'], 'cpu') else np.asarray(pose['at'])
        up = pose['up'].cpu().numpy()[0] if hasattr(pose['up'], 'cpu') else np.asarray(pose['up'])
        
        # Ensure they're 1D arrays
        if eye.ndim > 1:
            eye = eye.flatten()
        if at.ndim > 1:
            at = at.flatten()
        if up.ndim > 1:
            up = up.flatten()
        
        # Ensure mesh has vertex normals (compute if missing)
        # compute_vertex_normals() is safe to call even if normals exist
        try:
            container_mesh.compute_vertex_normals()
        except:
            pass  # If normals can't be computed, pyrender will handle it
        
        # Create pyrender mesh from trimesh
        pr_mesh = pyrender.Mesh.from_trimesh(container_mesh, smooth=False)
        
        # Create pyrender scene with ambient light
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
        scene.add(pr_mesh, name="container")
        
        # Add ground plane if provided
        if ground_plane is not None:
            try:
                ground_plane.compute_vertex_normals()
            except:
                pass  # If normals can't be computed, pyrender will handle it
            pr_plane = pyrender.Mesh.from_trimesh(ground_plane, smooth=False)
            scene.add(pr_plane, name="ground")
        
        # Setup camera
        H = W = self.r_config.IMAGE_SIZE
        fov_y_rad = np.deg2rad(self.r_config.CAMERA_FOV)
        camera = pyrender.PerspectiveCamera(
            yfov=fov_y_rad, 
            znear=self.r_config.ZNEAR, 
            zfar=self.r_config.ZFAR
        )
        
        # Calculate camera pose matrix using look_at_matrix
        cam_world = look_at_matrix(eye, at, up)  # world_from_camera transform
        camera_node = scene.add(camera, pose=cam_world)
        
        # Setup lighting - point light at camera location
        light = pyrender.PointLight(color=np.ones(3), intensity=10.0)
        scene.add(light, pose=cam_world)
        
        # Render
        color, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        
        # Convert color to float [0, 1]
        rgb = color[..., :3].astype(np.float32) / 255.0
        
        return rgb, depth


    def _depth_to_pointcloud(self, depth_map: np.ndarray, pose: Dict) -> np.ndarray:
        """Converts a depth map to a 3D point cloud in world coordinates."""
        H, W = depth_map.shape
        
        # Get camera parameters
        eye = pose['eye'].cpu().numpy()[0] if hasattr(pose['eye'], 'cpu') else np.asarray(pose['eye'])
        at = pose['at'].cpu().numpy()[0] if hasattr(pose['at'], 'cpu') else np.asarray(pose['at'])
        up = pose['up'].cpu().numpy()[0] if hasattr(pose['up'], 'cpu') else np.asarray(pose['up'])
        
        # Ensure they're 1D arrays
        if eye.ndim > 1:
            eye = eye.flatten()
        if at.ndim > 1:
            at = at.flatten()
        if up.ndim > 1:
            up = up.flatten()
        
        # Calculate camera pose matrix
        cam_world = look_at_matrix(eye, at, up)  # world_from_camera transform
        
        # Compute intrinsics from vertical FOV
        fov_y_rad = np.deg2rad(self.r_config.CAMERA_FOV)
        fy = H / (2.0 * np.tan(fov_y_rad / 2.0))
        fx = fy  # aspect == 1.0
        cx = W / 2.0
        cy = H / 2.0
        
        # Create pixel grid
        ys, xs = np.mgrid[0:H, 0:W]
        
        # Select valid depth pixels
        valid = np.isfinite(depth_map) & (depth_map > 0) & (depth_map < self.r_config.ZFAR)
        
        if not np.any(valid):
            return np.array([]).reshape(0, 3)
        
        # Get valid pixels
        zs = depth_map[valid]
        u = xs[valid].astype(np.float32)
        v = ys[valid].astype(np.float32)
        
        # Convert to camera coordinates
        x_cam = (u - cx) * zs / fx
        y_cam = (v - cy) * zs / fy
        points_cam = np.stack([x_cam, y_cam, zs], axis=1)  # N x 3
        
        # Transform from camera coords to world coords using cam_world (world_from_camera)
        ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([points_cam, ones], axis=1)  # N x 4
        pts_world = (cam_world @ pts_h.T).T[:, :3]  # N x 3
        
        return pts_world
    
    def _load_dent_specs(self, mesh_path: Path) -> Optional[Dict]:
        """Load dent specifications from JSON file if available."""
        specs_path = mesh_path.with_suffix('.json')
        if specs_path.exists():
            try:
                with open(specs_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load dent specs from {specs_path}: {e}")
        return None
    
    def _create_dent_mask(self, mesh: trimesh.Trimesh, dent_specs: Dict, pose: Dict) -> np.ndarray:
        """Create dent mask by identifying dented faces and rendering them."""
        try:
            # Identify dented faces
            dent_face_indices = self._identify_dent_faces(mesh, dent_specs)
            
            if len(dent_face_indices) == 0:
                return np.zeros((self.r_config.IMAGE_SIZE, self.r_config.IMAGE_SIZE), dtype=np.uint8)
            
            # Create mask mesh with white faces for dents
            mask_mesh = mesh.copy()
            num_faces = len(mask_mesh.faces)
            face_colors = np.zeros((num_faces, 4), dtype=np.uint8)
            
            # Set dent faces to white
            for face_idx in dent_face_indices:
                if face_idx < num_faces:
                    face_colors[face_idx] = [255, 255, 255, 255]
            
            mask_mesh.visual.face_colors = face_colors
            
            # Render mask using same camera setup
            rgb_mask, _ = self._render_view(mask_mesh, None, pose)
            
            # Convert to binary mask
            gray = np.dot(rgb_mask[..., :3], [0.299, 0.587, 0.114])
            binary_mask = (gray > 0.1).astype(np.uint8) * 255
            
            return binary_mask
            
        except Exception as e:
            logger.warning(f"Could not create dent mask: {e}")
            return np.zeros((self.r_config.IMAGE_SIZE, self.r_config.IMAGE_SIZE), dtype=np.uint8)
    
    def _identify_dent_faces(self, mesh: trimesh.Trimesh, dent_specs: Dict) -> List[int]:
        """Identify faces that are part of dents based on specifications."""
        dent_faces = []
        
        if 'dents' not in dent_specs:
            return dent_faces
        
        face_centers = mesh.triangles_center
        
        for dent in dent_specs.get('dents', []):
            dent_type = dent.get('type', '')
            position = np.array(dent.get('position', [0, 0, 0]))
            
            if dent_type == 'circular':
                radius = dent.get('radius', 0.1)
                for i, face_center in enumerate(face_centers):
                    dist = np.linalg.norm(face_center - position)
                    if dist <= radius:
                        dent_faces.append(i)
            
            elif dent_type == 'elliptical':
                radius_x = dent.get('radius_x', 0.1)
                radius_y = dent.get('radius_y', 0.1)
                max_radius = max(radius_x, radius_y)
                for i, face_center in enumerate(face_centers):
                    dist = np.linalg.norm(face_center - position)
                    if dist <= max_radius:
                        dent_faces.append(i)
            
            elif dent_type == 'crease':
                length = dent.get('length', 0.5)
                width = dent.get('width', 0.1)
                max_dim = max(length, width)
                for i, face_center in enumerate(face_centers):
                    dist = np.linalg.norm(face_center - position)
                    if dist <= max_dim:
                        dent_faces.append(i)
            
            elif dent_type in ['corner', 'surface']:
                radius = dent.get('radius', 0.1)
                for i, face_center in enumerate(face_centers):
                    dist = np.linalg.norm(face_center - position)
                    if dist <= radius:
                        dent_faces.append(i)
        
        return list(set(dent_faces))
    
    def _create_metadata(self, mesh_path: Optional[Path], rgb_path: Path, depth_path: Path,
                        mask_path: Optional[Path], pcd_path: Path, dent_specs: Optional[Dict],
                        pose: Optional[Dict], sample_id: int, shot_name: str) -> Dict:
        """Create metadata dictionary similar to render_rgbd.py."""
        metadata = {
            'sample_id': sample_id,
            'shot_name': shot_name,
            'mesh_file': str(mesh_path) if mesh_path else None,
            'rgb_image': str(rgb_path),
            'depth_npy': str(depth_path),
            'pointcloud': str(pcd_path),
            'mask_image': str(mask_path) if mask_path else None,
            'image_size': [self.r_config.IMAGE_SIZE, self.r_config.IMAGE_SIZE],
            'timestamp': datetime.now().isoformat(),
        }
        
        # Camera info
        if pose:
            eye = pose['eye'].cpu().numpy()[0] if hasattr(pose['eye'], 'cpu') else np.asarray(pose['eye'])
            if eye.ndim > 1:
                eye = eye.flatten()
            metadata['camera_info'] = {
                'position': eye.tolist(),
                'fov': self.r_config.CAMERA_FOV,
                'znear': self.r_config.ZNEAR,
                'zfar': self.r_config.ZFAR
            }
        
        # Dent specifications
        if dent_specs:
            metadata['dent_specs'] = dent_specs
            # Calculate depth analysis
            if 'dents' in dent_specs:
                depths = [d.get('depth_mm', 0) for d in dent_specs['dents']]
                if depths:
                    metadata['depth_analysis'] = {
                        'max_depth_mm': max(depths),
                        'mean_depth_mm': sum(depths) / len(depths),
                        'num_dents': len(dent_specs['dents']),
                        'method': 'specification_ground_truth'
                    }
        
        return metadata

    def _save_outputs(self, rgb, depth, pcd, output_dir, sample_id, shot_name, 
                     annotations: Optional[Dict] = None, dent_mask: Optional[np.ndarray] = None,
                     dent_specs: Optional[Dict] = None, pose: Optional[Dict] = None,
                     mesh_path: Optional[Path] = None):
        """Saves the generated outputs to files with a descriptive name."""
        base_filename = f"container_{sample_id:03d}_{shot_name}"
        
        # RGB image - convert from float [0,1] to uint8 [0,255]
        rgb_path = output_dir / f"{base_filename}_rgb.png"
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_uint8).save(rgb_path)
        
        # Depth map
        depth_path = output_dir / f"{base_filename}_depth.npy"
        np.save(depth_path, depth)
        
        # Point cloud
        pcd_path = output_dir / f"{base_filename}_pointcloud.ply"
        if pcd.shape[0] > 0:
            trimesh.points.PointCloud(vertices=pcd).export(pcd_path)
        else:
            logger.warning(f"No points in point cloud for {shot_name}, not saving .ply file.")
        
        # Dent mask
        mask_path = None
        if dent_mask is not None:
            mask_path = output_dir / f"{base_filename}_mask.png"
            Image.fromarray(dent_mask).save(mask_path)
            logger.info(f"  ✓ Saved dent mask: {mask_path.name}")
        
        # Annotations (if provided)
        if annotations and annotations.get('annotations'):
            annot_path = output_dir / f"{base_filename}_annotations.json"
            with open(annot_path, 'w') as f:
                json.dump(annotations, f, indent=4)
        
        # Metadata
        metadata_path = output_dir / f"{base_filename}_metadata.json"
        metadata = self._create_metadata(
            mesh_path, rgb_path, depth_path, mask_path, pcd_path,
            dent_specs, pose, sample_id, shot_name
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✓ Saved metadata: {metadata_path.name}")

