"""
Renders the generated container using PyTorch3D to produce RGB, depth, and point cloud data.
"""

import torch
import numpy as np
from pathlib import Path
import trimesh
from PIL import Image
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime

from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)

from config import ContainerConfig, RendererConfig
from camera_position import CameraPoseGenerator

# Make annotation generator optional
try:
    from annotation_generator import AnnotationGenerator
    ANNOTATION_AVAILABLE = True
except ImportError:
    ANNOTATION_AVAILABLE = False
    # Create a stub class if annotation generator is not available
    class AnnotationGenerator:
        def __init__(self, *args, **kwargs):
            pass
        def generate_annotations(self, *args, **kwargs):
            return None

logger = logging.getLogger(__name__)

class SceneRenderer:
    """Renders a 3D container mesh into a scene and captures data."""

    def __init__(self, renderer_config: RendererConfig, container_config: ContainerConfig):
        self.r_config = renderer_config
        self.c_config = container_config
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU. Rendering will be slow.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
            
        self.pose_generator = CameraPoseGenerator(self.c_config, self.r_config)
        if ANNOTATION_AVAILABLE:
            self.annotation_gen = AnnotationGenerator(
                image_size=self.r_config.IMAGE_SIZE,
                min_vertices_threshold=5  # Much lower threshold: keeps defects with 5+ vertices, filters 0-4 vertex "failed" defects
            )
        else:
            self.annotation_gen = None

    def render_and_save(self, mesh_path: Path, container_type: str, output_dir: Path, sample_id: int, defects_info: Optional[List[Dict]] = None, shot_name: Optional[str] = None):
        """
        Loads a mesh, places it in a scene, and renders it from multiple camera views,
        saving the RGB, depth, point cloud, mask, metadata, and annotation outputs for each view.
        """
        logger.info(f"Rendering all camera shots for sample {sample_id}: {mesh_path.name}")
        try:
            # 1. Load mesh once
            trimesh_mesh = trimesh.load_mesh(mesh_path, process=False)
            verts, faces, textures = self._load_mesh_for_pytorch3d(mesh_path)
            container_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
            ground_plane = self._create_ground_plane()
            scene_mesh = join_meshes_as_batch([container_mesh, ground_plane])
            
            # Load dent specifications if available
            dent_specs = self._load_dent_specs(mesh_path)

            # 2. Generate all camera poses
            poses = self.pose_generator.generate_poses(container_type)

            # 3. If a specific shot is requested, filter the poses
            if shot_name:
                poses = [p for p in poses if p['name'] == shot_name]
                if not poses:
                    logger.warning(f"Shot name '{shot_name}' not found. No images will be rendered for this sample.")
                    return

            # 4. Loop through each pose and render
            for pose in poses:
                current_shot_name = pose['name']
                logger.info(f"  - Rendering shot: {current_shot_name}")
                
                # Setup camera for the current pose
                cameras = self._setup_camera_from_pose(pose)

                # Generate annotations if applicable
                annotations = None
                if (self.annotation_gen and defects_info and 
                    current_shot_name not in self.r_config.ANNOTATION_EXCLUSION_LIST):
                    # Create image_id and image_path for the simplified format
                    base_filename = f"container_{sample_id:03d}_{current_shot_name}"
                    image_id = base_filename
                    image_path = f"./{base_filename}.npy"
                    annotations = self.annotation_gen.generate_annotations(cameras, defects_info, verts, image_id, image_path)

                # Setup renderer for the current camera
                renderer = self._setup_renderer(cameras)
                
                # Render RGB image and get depth map
                fragments = renderer.rasterizer(scene_mesh)
                images = renderer.shader(fragments, scene_mesh, cameras=cameras)
                
                rgb_image = images[0, ..., :3].cpu().numpy()
                zbuf_ndc = fragments.zbuf[0, ..., 0]

                # Convert depth and generate point cloud
                depth_map = self._ndc_to_view_depth(zbuf_ndc, cameras)
                point_cloud = self._depth_to_pointcloud(depth_map, cameras)
                
                # Generate dent mask if dent specs available
                dent_mask = None
                if dent_specs:
                    dent_mask = self._create_dent_mask(
                        trimesh_mesh, dent_specs, cameras, verts, faces
                    )

                # Save all outputs with a descriptive name
                self._save_outputs(
                    rgb_image, 
                    depth_map.cpu().numpy(), 
                    point_cloud, 
                    output_dir, 
                    sample_id, 
                    current_shot_name, 
                    annotations,
                    dent_mask,
                    dent_specs,
                    cameras,
                    mesh_path,
                    trimesh_mesh
                )

            logger.info(f"✓ Successfully rendered and saved all shots for sample {sample_id}")

        except Exception as e:
            logger.error(f"Failed to render sample {sample_id}: {e}", exc_info=True)

    def _load_mesh_for_pytorch3d(self, mesh_path: Path):
        """Loads a mesh using trimesh and converts it to PyTorch3D format."""
        trimesh_mesh = trimesh.load_mesh(mesh_path, process=False)
        
        verts = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int64, device=self.device)
        
        if hasattr(trimesh_mesh.visual, 'vertex_colors') and len(trimesh_mesh.visual.vertex_colors):
            vertex_colors = torch.tensor(trimesh_mesh.visual.vertex_colors[:, :3], dtype=torch.float32, device=self.device) / 255.0
        else:
            logger.warning("Mesh has no vertex colors, using default gray.")
            vertex_colors = torch.ones_like(verts) * 0.5
            
        textures = TexturesVertex(verts_features=[vertex_colors])
        return verts, faces, textures

    def _create_ground_plane(self) -> Meshes:
        """Creates a large, simple ground plane mesh."""
        # A large plane at y=0, which is the floor level of the container world
        verts = torch.tensor([[-50, 0, -50], [50, 0, -50], [50, 0, 50], [-50, 0, 50]], dtype=torch.float32, device=self.device) * 2
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=self.device)
        colors = torch.ones((4, 3), dtype=torch.float32, device=self.device) * 0.3  # Dark gray
        textures = TexturesVertex(verts_features=[colors])
        return Meshes(verts=[verts], faces=[faces], textures=textures)

    def _setup_camera_from_pose(self, pose: Dict) -> FoVPerspectiveCameras:
        """Sets up a camera based on a pose dictionary."""
        eye = pose['eye'].to(self.device, dtype=torch.float32)
        at = pose['at'].to(self.device, dtype=torch.float32)
        up = pose['up'].to(self.device, dtype=torch.float32)
        
        R, T = look_at_view_transform(eye=eye, at=at, up=up)
        
        return FoVPerspectiveCameras(
            R=R, T=T,
            znear=self.r_config.ZNEAR,
            zfar=self.r_config.ZFAR,
            fov=self.r_config.CAMERA_FOV,
            aspect_ratio=1.0
        )

    def _setup_renderer(self, cameras: FoVPerspectiveCameras) -> MeshRenderer:
        """Initializes the Pytorch3D renderer."""
        # Lights: A point light source at the camera's position
        lights = PointLights(device=self.device, location=cameras.get_camera_center())
        
        raster_settings = RasterizationSettings(
            image_size=self.r_config.IMAGE_SIZE,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights)
        )

    def _ndc_to_view_depth(self, ndc_depth: torch.Tensor, cameras: FoVPerspectiveCameras) -> torch.Tensor:
        """Converts an NDC depth map from the rasterizer to view-space depth."""
        proj_transform = cameras.get_projection_transform()
        p_matrix = proj_transform.get_matrix()
        
        # These are the perspective projection matrix parameters
        A = p_matrix[..., 2, 2]
        B = p_matrix[..., 2, 3]
        
        # The z-buffer values are in NDC space, which is [-1, 1].
        # The conversion to view space is z_view = -B / (z_ndc + A)
        view_depth = -B / (ndc_depth + A)
        
        # Mask out background pixels, which have an NDC depth of -1
        view_depth[ndc_depth <= -1.0] = float('NaN')
        return view_depth

    def _depth_to_pointcloud(self, view_depth: torch.Tensor, cameras: FoVPerspectiveCameras) -> np.ndarray:
        """Converts a view-space depth map to a 3D point cloud in world coordinates."""
        H, W = view_depth.shape
        y, x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        
        # Select only valid depth pixels
        mask = ~torch.isnan(view_depth)
        
        # Create (x, y, depth) triplets for unprojection
        xy_depth = torch.stack([
            x[mask].to(torch.float32), 
            y[mask].to(torch.float32), 
            view_depth[mask]
        ], dim=1)
        
        # Unproject to world coordinates.
        # `unproject_points` expects an (N, 3) tensor of (x, y, z) screen coordinates
        # where z is the camera view depth.
        point_cloud_tensor = cameras.unproject_points(
            xy_depth.unsqueeze(0), 
            world_coordinates=True
        )
        
        return point_cloud_tensor.squeeze(0).cpu().numpy()
    
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
    
    def _create_dent_mask(self, trimesh_mesh: trimesh.Trimesh, dent_specs: Dict,
                         cameras: FoVPerspectiveCameras, verts: torch.Tensor, faces: torch.Tensor) -> np.ndarray:
        """Create dent mask by identifying dented faces and rendering them."""
        try:
            # Identify dented faces based on dent specifications
            dent_face_indices = self._identify_dent_faces(trimesh_mesh, dent_specs)
            
            if len(dent_face_indices) == 0:
                # Return empty mask
                return np.zeros((self.r_config.IMAGE_SIZE, self.r_config.IMAGE_SIZE), dtype=np.uint8)
            
            # Create mask mesh with white faces for dents
            mask_mesh = trimesh_mesh.copy()
            num_faces = len(mask_mesh.faces)
            face_colors = np.zeros((num_faces, 4), dtype=np.uint8)
            
            # Set dent faces to white
            for face_idx in dent_face_indices:
                if face_idx < num_faces:
                    face_colors[face_idx] = [255, 255, 255, 255]
            
            mask_mesh.visual.face_colors = face_colors
            
            # Convert to PyTorch3D format and render
            mask_verts = torch.tensor(mask_mesh.vertices, dtype=torch.float32, device=self.device)
            mask_faces = torch.tensor(mask_mesh.faces, dtype=torch.int64, device=self.device)
            
            # Create white vertex colors for dent faces
            mask_vertex_colors = torch.zeros((len(mask_verts), 3), dtype=torch.float32, device=self.device)
            # Set dent vertices to white
            dent_vertex_mask = np.zeros(len(mask_verts), dtype=bool)
            mask_faces_np = mask_faces.cpu().numpy()
            for face_idx in dent_face_indices:
                if face_idx < len(mask_faces_np):
                    face = mask_faces_np[face_idx]
                    dent_vertex_mask[face] = True
            mask_vertex_colors[torch.tensor(dent_vertex_mask, device=self.device)] = 1.0
            
            mask_textures = TexturesVertex(verts_features=[mask_vertex_colors])
            mask_mesh_pytorch = Meshes(verts=[mask_verts], faces=[mask_faces], textures=mask_textures)
            
            # Render mask
            renderer = self._setup_renderer(cameras)
            fragments = renderer.rasterizer(mask_mesh_pytorch)
            images = renderer.shader(fragments, mask_mesh_pytorch, cameras=cameras)
            
            # Convert to binary mask
            mask_image = images[0, ..., :3].cpu().numpy()
            gray = np.dot(mask_image, [0.299, 0.587, 0.114])
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
                # Simplified: use max radius for now
                max_radius = max(radius_x, radius_y)
                for i, face_center in enumerate(face_centers):
                    dist = np.linalg.norm(face_center - position)
                    if dist <= max_radius:
                        dent_faces.append(i)
            
            elif dent_type == 'crease':
                length = dent.get('length', 0.5)
                width = dent.get('width', 0.1)
                # Simplified: use max dimension
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
        
        # Remove duplicates
        return list(set(dent_faces))
    
    def _create_metadata(self, mesh_path: Optional[Path], rgb_path: Path, depth_path: Path,
                        mask_path: Optional[Path], pcd_path: Path, dent_specs: Optional[Dict],
                        cameras: Optional[FoVPerspectiveCameras], trimesh_mesh: Optional[trimesh.Trimesh],
                        sample_id: int, shot_name: str) -> Dict:
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
        
        # Mesh info
        if trimesh_mesh:
            metadata['mesh_info'] = {
                'vertices': len(trimesh_mesh.vertices),
                'faces': len(trimesh_mesh.faces),
                'bounds': trimesh_mesh.bounds.tolist()
            }
        
        # Camera info
        if cameras:
            camera_center = cameras.get_camera_center()[0].cpu().numpy()
            metadata['camera_info'] = {
                'position': camera_center.tolist(),
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
                      dent_specs: Optional[Dict] = None, cameras: Optional[FoVPerspectiveCameras] = None,
                      mesh_path: Optional[Path] = None, trimesh_mesh: Optional[trimesh.Trimesh] = None):
        """Saves the generated outputs to files with a descriptive name."""
        base_filename = f"container_{sample_id:03d}_{shot_name}"
        
        # RGB image
        rgb_path = output_dir / f"{base_filename}_rgb.png"
        img_array = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_array).save(rgb_path)

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

        # Annotations
        if annotations and annotations.get('annotations'):
            annot_path = output_dir / f"{base_filename}_annotations.json"
            with open(annot_path, 'w') as f:
                json.dump(annotations, f, indent=4)
        
        # Metadata
        metadata_path = output_dir / f"{base_filename}_metadata.json"
        metadata = self._create_metadata(
            mesh_path, rgb_path, depth_path, mask_path, pcd_path,
            dent_specs, cameras, trimesh_mesh, sample_id, shot_name
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✓ Saved metadata: {metadata_path.name}") 