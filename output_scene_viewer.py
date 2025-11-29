#!/usr/bin/env python3
"""
Output Scene Viewer - Comprehensive UI for visualizing container scene data
Visualizes all containers, shots, RGB-D images, depth differences, and statistics

Note: If you see a RuntimeError about torch.classes during startup, this is harmless.
It occurs because Streamlit's file watcher tries to inspect PyTorch modules.
The app will function normally despite this error.
"""

import streamlit as st
import numpy as np
import json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import trimesh

# Import camera pose generator
try:
    from config import ContainerConfig, RendererConfig
    from camera_position import CameraPoseGenerator
    CAMERA_POSE_GENERATOR_AVAILABLE = True
except ImportError:
    CAMERA_POSE_GENERATOR_AVAILABLE = False
    st.warning("⚠️ Could not import CameraPoseGenerator. Camera simulation may be limited.")

# Camera configuration
CAMERA_FOV = 75.0  # degrees
IMAGE_SIZE = 512  # Square images

# Page configuration
st.set_page_config(
    page_title="📊 Output Scene Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=60)
def load_output_scene_data():
    """Load all output_scene data"""
    output_scene_dir = Path("output_scene")
    
    if not output_scene_dir.exists():
        return {}
    
    containers = {}
    
    # Find all container directories
    for container_dir in sorted(output_scene_dir.iterdir()):
        if not container_dir.is_dir():
            continue
        
        container_name = container_dir.name
        
        # Load comparison summary if exists
        # Summary files are named like: {container_id}_comparison_summary.json
        # e.g., 20ft_0005_comparison_summary.json
        summary_files = list(container_dir.glob("*_comparison_summary.json"))
        comparison_summary = None
        if summary_files:
            try:
                with open(summary_files[0], 'r') as f:
                    comparison_summary = json.load(f)
            except Exception as e:
                st.warning(f"Error loading summary for {container_name}: {e}")
        
        # Find all shot directories
        shots = {}
        for shot_dir in sorted(container_dir.iterdir()):
            if not shot_dir.is_dir():
                continue
            
            shot_name = shot_dir.name
            
            # Find all files in shot directory
            shot_files = {
                'original_rgb': None,
                'dented_rgb': None,
                'original_depth_png': None,
                'dented_depth_png': None,
                'depth_diff_png': None,
                'dent_mask': None,
                'visual_output': None,
                'original_depth_npy': None,
                'dented_depth_npy': None,
                'depth_diff_npy': None,
                'original_pointcloud_ply': None,
                'dented_pointcloud_ply': None
            }
            
            # Pattern matching for files
            # Files are named like: {container_id}_{type}.png/npy
            # e.g., 20ft_0005_original_rgb.png or 40ft_0001_dented_depth.npy
            
            for file_path in shot_dir.iterdir():
                filename = file_path.name
                
                # Skip debug files - they are for debugging only, not for display
                if 'debug_' in filename:
                    continue
                
                if '_original_rgb.png' in filename:
                    shot_files['original_rgb'] = file_path
                elif '_dented_rgb.png' in filename:
                    shot_files['dented_rgb'] = file_path
                elif '_original_depth.png' in filename:
                    shot_files['original_depth_png'] = file_path
                elif '_dented_depth.png' in filename:
                    shot_files['dented_depth_png'] = file_path
                elif '_depth_diff.png' in filename:
                    shot_files['depth_diff_png'] = file_path
                elif '_dent_mask.png' in filename:
                    shot_files['dent_mask'] = file_path
                elif '_visualOutput.png' in filename:
                    shot_files['visual_output'] = file_path
                elif '_original_depth.npy' in filename:
                    shot_files['original_depth_npy'] = file_path
                elif '_dented_depth.npy' in filename:
                    shot_files['dented_depth_npy'] = file_path
                elif '_depth_diff.npy' in filename:
                    shot_files['depth_diff_npy'] = file_path
                elif '_original_pointcloud.ply' in filename:
                    shot_files['original_pointcloud_ply'] = file_path
                elif '_dented_pointcloud.ply' in filename:
                    shot_files['dented_pointcloud_ply'] = file_path
                elif filename.endswith('.ply'):
                    # Fallback: if no specific match, assign to original if not already set
                    if shot_files['original_pointcloud_ply'] is None:
                        shot_files['original_pointcloud_ply'] = file_path
            
            # Get shot statistics from comparison summary
            shot_stats = None
            if comparison_summary and 'shots' in comparison_summary:
                for shot_data in comparison_summary['shots']:
                    if shot_data.get('shot_name') == shot_name:
                        shot_stats = shot_data
                        break
            
            shots[shot_name] = {
                'files': shot_files,
                'stats': shot_stats,
                'path': shot_dir
            }
        
        containers[container_name] = {
            'name': container_name,
            'path': container_dir,
            'shots': shots,
            'comparison_summary': comparison_summary
        }
    
    return containers

def apply_colormap(image, colormap='viridis'):
    """Apply colormap to depth image for visualization"""
    if len(image.shape) == 3:
        image = image[:,:,0] if image.shape[2] == 1 else np.mean(image, axis=2)
    
    # Normalize to 0-1 range
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max - img_min < 1e-8:
        img_norm = np.zeros_like(image)
    else:
        img_norm = (image - img_min) / (img_max - img_min)
    
    # Apply colormap
    if colormap == 'viridis':
        colored = cm.viridis(img_norm)
    elif colormap == 'plasma':
        colored = cm.plasma(img_norm)
    elif colormap == 'inferno':
        colored = cm.inferno(img_norm)
    elif colormap == 'magma':
        colored = cm.magma(img_norm)
    elif colormap == 'jet':
        colored = cm.jet(img_norm)
    else:
        colored = cm.viridis(img_norm)
    
    return (colored[:,:,:3] * 255).astype(np.uint8)

def load_npy_depth(file_path):
    """Load depth data from NPY file"""
    try:
        return np.load(file_path)
    except Exception as e:
        st.error(f"Error loading NPY file {file_path}: {e}")
        return None

def load_ply_pointcloud(file_path):
    """Load PLY point cloud file"""
    try:
        mesh = trimesh.load(file_path)
        colors = None
        
        if isinstance(mesh, trimesh.PointCloud):
            vertices = mesh.vertices
            if hasattr(mesh, 'colors') and mesh.colors is not None and len(mesh.colors) > 0:
                colors = mesh.colors
        elif isinstance(mesh, trimesh.Trimesh):
            # If it's a mesh, extract vertices
            vertices = mesh.vertices
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                try:
                    vc = mesh.visual.vertex_colors
                    if vc is not None and len(vc.shape) > 1 and vc.shape[1] >= 3:
                        colors = vc[:, :3]
                except:
                    pass
        else:
            # Try to get vertices from any trimesh object
            if hasattr(mesh, 'vertices'):
                vertices = mesh.vertices
                if hasattr(mesh, 'colors') and mesh.colors is not None and len(mesh.colors) > 0:
                    colors = mesh.colors
                elif hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                    try:
                        vc = mesh.visual.vertex_colors
                        if vc is not None and len(vc.shape) > 1 and vc.shape[1] >= 3:
                            colors = vc[:, :3]
                    except:
                        pass
            else:
                return None, None
        
        if vertices is not None and len(vertices) > 0:
            return vertices, colors
        return None, None
    except Exception as e:
        st.error(f"Error loading PLY file {file_path}: {e}")
        return None, None

def transform_pointcloud_to_camera_view(vertices, camera_pose):
    """
    Transform point cloud from world coordinates to camera view coordinates.
    
    This reverses the transformation used in compare_dents_depth.py when creating PLY files.
    The camera view coordinate system matches image orientation:
    - X axis: Width (horizontal, right direction) - matches image u coordinate (left to right)
    - Y axis: Depth (forward, toward the panel from camera) - matches image depth
    - Z axis: Height (vertical, DOWNWARD direction) - matches image v coordinate (top to bottom, v increases downward)
    
    In compare_dents_depth.py:
    - Camera space: x_cam (horizontal), y_cam (vertical, increases downward), z_cam (depth forward)
    - Transformation: R_cam_to_world = [right, up_corrected, -forward]
    - World space: points_world = R_cam_to_world @ points_cam + eye
    
    To reverse this:
    - World space → Camera space: points_cam = R_cam_to_world^T @ (points_world - eye)
    - Then rearrange to match image: X=width, Y=depth, Z=height (downward)
    
    Args:
        vertices: Point cloud vertices in world coordinates (N, 3)
        camera_pose: Dictionary with 'eye', 'at', 'up' vectors
    
    Returns:
        Transformed vertices in camera view coordinates (N, 3)
    """
    eye = camera_pose['eye']
    at = camera_pose['at']
    up = camera_pose['up']
    
    # Calculate camera coordinate system (matching compare_dents_depth.py exactly)
    # Forward direction: from eye to at (points toward the panel)
    forward = at - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    
    # Right direction: cross product of forward and up (matching compare_dents_depth.py line 962)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    
    # Up direction: cross product of right and forward (matching compare_dents_depth.py line 964)
    up_corrected = np.cross(right, forward)
    up_corrected = up_corrected / (np.linalg.norm(up_corrected) + 1e-8)
    
    # Build camera-to-world rotation matrix (matching compare_dents_depth.py line 967)
    R_cam_to_world = np.stack([right, up_corrected, -forward], axis=1)
    
    # Transform vertices from world to camera space
    # Translate to camera position (eye)
    vertices_translated = vertices - eye
    
    # Rotate to camera coordinate system (inverse of R_cam_to_world is its transpose)
    R_world_to_cam = R_cam_to_world.T
    vertices_cam = (R_world_to_cam @ vertices_translated.T).T
    
    # Now rearrange to match image orientation:
    # In original camera space: [x_cam, y_cam, z_cam] where:
    #   x_cam = horizontal (width, left to right) → our X
    #   y_cam = vertical (height, TOP to BOTTOM, increases downward) → our Z (flipped)
    #   z_cam = depth (forward) → our Y
    # 
    # Our desired camera view: [X=width, Y=depth, Z=height]
    # So: X = x_cam, Y = z_cam, Z = -y_cam (flip because image Y increases downward)
    vertices_camera_view = np.column_stack([
        vertices_cam[:, 0],   # X = width (horizontal, left to right)
        vertices_cam[:, 2],   # Y = depth (forward, toward panel)
        -vertices_cam[:, 1]   # Z = height (vertical, TOP to BOTTOM, matches image v coordinate)
    ])
    
    return vertices_camera_view

def get_camera_view_eye_position(camera_pose, scene_bounds=None):
    """
    Calculate the initial camera eye position for plotly visualization.
    This positions the viewer behind the camera, looking at the target panel.
    
    In camera view coordinates (matching image orientation):
    - X = Width (horizontal, left to right)
    - Y = Depth (positive Y points toward panel from camera)
    - Z = Height (vertical, TOP to BOTTOM, increases downward like image v coordinate)
    
    Args:
        camera_pose: Dictionary with 'eye', 'at', 'up' vectors
        scene_bounds: Optional tuple (min_point, max_point) for scene bounds
    
    Returns:
        Dictionary with 'eye', 'center', 'up' for plotly camera
    """
    eye = camera_pose['eye']
    at = camera_pose['at']
    
    # Estimate scene size if bounds provided
    if scene_bounds:
        scene_size = np.max(scene_bounds[1] - scene_bounds[0])
    else:
        distance = np.linalg.norm(at - eye)
        scene_size = distance * 2
    
    # Position viewer behind the camera in camera view coordinates
    # Looking along positive Y axis (toward the panel)
    # Eye position: slightly back and elevated for good view
    # Note: Z increases downward (like image v), so negative Z is "up" (top of image)
    eye_x = 0  # Centered horizontally
    eye_y = -scene_size * 0.8  # Behind the camera (negative Y, looking toward positive Y)
    eye_z = -scene_size * 0.3  # Above the scene (negative Z = top of image, since Z increases downward)
    
    return {
        'eye': {'x': eye_x, 'y': eye_y, 'z': eye_z},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'up': {'x': 0, 'y': 0, 'z': -1}  # Negative Z is up (top of image, since Z increases downward)
    }

def visualize_pointcloud_3d(vertices, colors=None, title="3D Point Cloud"):
    """Create 3D visualization of point cloud using plotly"""
    if vertices is None or len(vertices) == 0:
        return None
    
    # Downsample if too many points for performance
    max_points = 50000
    if len(vertices) > max_points:
        indices = np.random.choice(len(vertices), max_points, replace=False)
        vertices = vertices[indices]
        if colors is not None:
            colors = colors[indices]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Determine color mapping
    if colors is not None and len(colors) > 0:
        # Use vertex colors if available
        if colors.shape[1] == 3:
            # RGB colors (0-255 or 0-1)
            if colors.max() > 1.0:
                colors = colors / 255.0
            color_str = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]
        else:
            color_str = 'blue'
    else:
        # Color by Z coordinate (depth)
        z_coords = vertices[:, 2]
        z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-8)
        color_str = z_normalized
    
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=color_str if isinstance(color_str, (list, np.ndarray)) else 'blue',
            colorscale='Viridis' if not isinstance(color_str, list) else None,
            showscale=not isinstance(color_str, list),
            colorbar=dict(title="Z Coordinate" if not isinstance(color_str, list) else None),
            opacity=0.8
        ),
        name="Point Cloud"
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (Width, meters)",
            yaxis_title="Y (Length, meters)",
            zaxis_title="Z (Height, meters)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def format_shot_name(shot_name):
    """Format shot name for display"""
    return shot_name.replace('_', ' ').title()

def get_shot_category(shot_name):
    """Categorize shot by type"""
    if 'external' in shot_name:
        return 'External'
    elif 'internal_overall' in shot_name:
        return 'Internal - Overview'
    elif 'internal_door' in shot_name:
        return 'Internal - Door'
    elif 'internal_back_wall' in shot_name:
        return 'Internal - Back Wall'
    elif 'corner' in shot_name:
        return 'Internal - Corners'
    elif 'roof' in shot_name:
        return 'Internal - Roof'
    elif 'left_wall' in shot_name:
        return 'Internal - Left Wall'
    elif 'right_wall' in shot_name:
        return 'Internal - Right Wall'
    else:
        return 'Internal - Other'

@st.cache_data
def get_camera_pose_for_shot(shot_name: str, container_type: str):
    """
    Get the exact camera pose for a shot using CameraPoseGenerator.
    
    Args:
        shot_name: Name of the shot (e.g., 'internal_door_left', 'internal_roof_1')
        container_type: Type of container ('20ft', '40ft', '40ft_hc')
    
    Returns:
        Dictionary with 'eye', 'at', 'up', 'target_panel', 'capture_section' keys
    """
    if not CAMERA_POSE_GENERATOR_AVAILABLE:
        return None
    
    try:
        # Initialize camera pose generator
        container_config = ContainerConfig()
        renderer_config = RendererConfig()
        renderer_config.IMAGE_SIZE = IMAGE_SIZE
        renderer_config.CAMERA_FOV = CAMERA_FOV
        pose_generator = CameraPoseGenerator(container_config, renderer_config)
        
        # Generate all poses for this container type
        poses = pose_generator.generate_poses(container_type)
        
        # Find the matching pose
        matching_pose = None
        for pose in poses:
            if pose['name'] == shot_name:
                matching_pose = pose
                break
        
        if matching_pose is None:
            return None
        
        # Convert torch tensors to numpy arrays
        eye = matching_pose['eye'].cpu().numpy()[0] if hasattr(matching_pose['eye'], 'cpu') else np.asarray(matching_pose['eye']).flatten()
        at = matching_pose['at'].cpu().numpy()[0] if hasattr(matching_pose['at'], 'cpu') else np.asarray(matching_pose['at']).flatten()
        up = matching_pose['up'].cpu().numpy()[0] if hasattr(matching_pose['up'], 'cpu') else np.asarray(matching_pose['up']).flatten()
        
        # Determine target panel and capture section
        target_panel, capture_section = _get_panel_info_from_shot(shot_name, container_type, eye, at)
        
        return {
            'eye': eye,
            'at': at,
            'up': up,
            'target_panel': target_panel,
            'capture_section': capture_section
        }
    except Exception as e:
        st.error(f"Error generating camera pose: {e}")
        return None

def _get_panel_info_from_shot(shot_name: str, container_type: str, eye: np.ndarray, at: np.ndarray):
    """Determine panel information and capture section from shot name and pose."""
    spec = ContainerConfig().CONTAINER_SPECS[container_type]
    length, height, width = spec["external_y_up"]
    
    if 'internal_door_left' in shot_name:
        target_panel = "Left Door Panel"
        # Calculate which section of the door is being captured
        # Coordinate system: X=Width, Y=Length, Z=Height
        # Original at[2] is width, which maps to X in new system
        capture_section = f"Left door panel (centered at X={at[2]:.2f}m)"
        
    elif 'internal_door_right' in shot_name:
        target_panel = "Right Door Panel"
        # Original at[2] is width, which maps to X in new system
        capture_section = f"Right door panel (centered at X={at[2]:.2f}m)"
        
    elif 'internal_back_wall' in shot_name:
        target_panel = "Back Wall"
        # Back wall spans full width and height
        # Original at[0] is length, which maps to Y in new system
        capture_section = f"Back wall (full wall at Y={at[0]:.2f}m)"
        
    elif 'internal_roof' in shot_name:
        try:
            roof_num = int(shot_name.split('_')[-1])
        except:
            roof_num = 1
        target_panel = "Roof"
        # Calculate roof section based on camera position
        # Original eye[0] is length, which maps to Y in new system
        section_start_y = eye[0] - length * 0.15  # Approximate section coverage
        section_end_y = eye[0] + length * 0.15
        capture_section = f"Roof section {roof_num} (Y: {section_start_y:.2f}m to {section_end_y:.2f}m)"
        
    elif 'internal_left_wall' in shot_name:
        try:
            wall_num = int(shot_name.split('_')[-1])
        except:
            wall_num = 1
        target_panel = "Left Side Wall"
        # Original eye[0] is length, which maps to Y in new system
        section_start_y = eye[0] - length * 0.15
        section_end_y = eye[0] + length * 0.15
        capture_section = f"Left wall section {wall_num} (Y: {section_start_y:.2f}m to {section_end_y:.2f}m)"
        
    elif 'internal_right_wall' in shot_name:
        try:
            wall_num = int(shot_name.split('_')[-1])
        except:
            wall_num = 1
        target_panel = "Right Side Wall"
        # Original eye[0] is length, which maps to Y in new system
        section_start_y = eye[0] - length * 0.15
        section_end_y = eye[0] + length * 0.15
        capture_section = f"Right wall section {wall_num} (Y: {section_start_y:.2f}m to {section_end_y:.2f}m)"
        
    else:
        target_panel = "Unknown Panel"
        capture_section = "Unknown section"
    
    return target_panel, capture_section

def create_camera_simulation_plot(camera_pose: dict, container_type: str):
    """
    Create 3D visualization of camera simulation showing camera position, direction, and target panel.
    
    Args:
        camera_pose: Dictionary with 'eye', 'at', 'up', 'target_panel', 'capture_section'
        container_type: Container type for dimensions
    
    Returns:
        Plotly figure
    """
    eye = camera_pose['eye']
    at = camera_pose['at']
    up = camera_pose['up']
    target_panel = camera_pose['target_panel']
    capture_section = camera_pose.get('capture_section', '')
    
    # Get container dimensions
    if CAMERA_POSE_GENERATOR_AVAILABLE:
        spec = ContainerConfig().CONTAINER_SPECS[container_type]
    else:
        # Fallback specs
        specs = {
            "20ft": (6.058, 2.591, 2.438),
            "40ft": (12.192, 2.591, 2.438),
            "40ft_hc": (12.192, 2.896, 2.438)
        }
        spec = specs.get(container_type, specs["20ft"])
    
    length, height, width = spec["external_y_up"]
    
    # Calculate camera direction vector
    direction = at - eye
    direction_normalized = direction / (np.linalg.norm(direction) + 1e-8)
    
    # Calculate camera view frustum
    fov_rad = np.deg2rad(CAMERA_FOV)
    distance = np.linalg.norm(direction)
    
    # Calculate view coverage at target distance
    view_size = 2 * distance * np.tan(fov_rad / 2)
    
    # Create container wireframe
    # Coordinate system: X=Width, Y=Length, Z=Height
    # Original: [length, height, width] -> New: [width, length, height] = [z, x, y]
    container_corners_original = np.array([
        [-length/2, 0, -width/2],      # Back-left-bottom
        [length/2, 0, -width/2],        # Front-left-bottom
        [length/2, 0, width/2],         # Front-right-bottom
        [-length/2, 0, width/2],        # Back-right-bottom
        [-length/2, height, -width/2],  # Back-left-top
        [length/2, height, -width/2],    # Front-left-top
        [length/2, height, width/2],     # Front-right-top
        [-length/2, height, width/2],    # Back-right-top
    ])
    # Swap coordinates: [width, length, height] = [original_z, original_x, original_y]
    container_corners = np.array([
        [corner[2], corner[0], corner[1]] for corner in container_corners_original
    ])
    
    fig = go.Figure()
    
    # Draw container wireframe
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    for edge in edges:
        p1, p2 = container_corners[edge[0]], container_corners[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color='gray', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add panel labels on the container skeleton
    # Coordinate system: X=Width, Y=Length, Z=Height
    # Roof label (at top center) - original: [0, height, 0] -> new: [0, 0, height]
    roof_center = np.array([0, 0, height])
    fig.add_trace(go.Scatter3d(
        x=[roof_center[0]],
        y=[roof_center[1]],
        z=[roof_center[2]],
        mode='text',
        text=['ROOF'],
        textfont=dict(size=16, color='blue', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Floor label (at bottom center) - original: [0, 0, 0] -> new: [0, 0, 0]
    floor_center = np.array([0, 0, 0])
    fig.add_trace(go.Scatter3d(
        x=[floor_center[0]],
        y=[floor_center[1]],
        z=[floor_center[2]],
        mode='text',
        text=['FLOOR'],
        textfont=dict(size=16, color='brown', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Left Panel label (on left side, centered) - original: [0, height/2, -width/2] -> new: [-width/2, 0, height/2]
    left_panel_center = np.array([-width/2, 0, height/2])
    fig.add_trace(go.Scatter3d(
        x=[left_panel_center[0]],
        y=[left_panel_center[1]],
        z=[left_panel_center[2]],
        mode='text',
        text=['LEFT PANEL'],
        textfont=dict(size=14, color='green', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Right Panel label (on right side, centered) - original: [0, height/2, width/2] -> new: [width/2, 0, height/2]
    right_panel_center = np.array([width/2, 0, height/2])
    fig.add_trace(go.Scatter3d(
        x=[right_panel_center[0]],
        y=[right_panel_center[1]],
        z=[right_panel_center[2]],
        mode='text',
        text=['RIGHT PANEL'],
        textfont=dict(size=14, color='green', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Door label (at front center) - original: [length/2, height/2, 0] -> new: [0, length/2, height/2]
    door_center = np.array([0, length/2, height/2])
    fig.add_trace(go.Scatter3d(
        x=[door_center[0]],
        y=[door_center[1]],
        z=[door_center[2]],
        mode='text',
        text=['DOOR'],
        textfont=dict(size=16, color='orange', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Back Wall label (at back center) - original: [-length/2, height/2, 0] -> new: [0, -length/2, height/2]
    back_wall_center = np.array([0, -length/2, height/2])
    fig.add_trace(go.Scatter3d(
        x=[back_wall_center[0]],
        y=[back_wall_center[1]],
        z=[back_wall_center[2]],
        mode='text',
        text=['BACK WALL'],
        textfont=dict(size=14, color='purple', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Draw camera position (eye)
    # Coordinate system: X=Width, Y=Length, Z=Height
    # Swap coordinates: [width, length, height] = [original_z, original_x, original_y]
    eye_swapped = np.array([eye[2], eye[0], eye[1]])
    fig.add_trace(go.Scatter3d(
        x=[eye_swapped[0]],
        y=[eye_swapped[1]],
        z=[eye_swapped[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond',
            line=dict(width=2, color='darkred')
        ),
        name="Camera Position",
        hovertemplate=f"Camera Position<br>X (Width): {eye_swapped[0]:.2f}m<br>Y (Length): {eye_swapped[1]:.2f}m<br>Z (Height): {eye_swapped[2]:.2f}m<extra></extra>"
    ))
    
    # Draw camera target (at)
    # Swap coordinates: [width, length, height] = [original_z, original_x, original_y]
    at_swapped = np.array([at[2], at[0], at[1]])
    fig.add_trace(go.Scatter3d(
        x=[at_swapped[0]],
        y=[at_swapped[1]],
        z=[at_swapped[2]],
        mode='markers',
        marker=dict(
            size=12,
            color='yellow',
            symbol='circle',
            line=dict(width=2, color='orange')
        ),
        name="Camera Target",
        hovertemplate=f"Target Point<br>X (Width): {at_swapped[0]:.2f}m<br>Y (Length): {at_swapped[1]:.2f}m<br>Z (Height): {at_swapped[2]:.2f}m<extra></extra>"
    ))
    
    # Draw camera direction line
    # Swap direction vector coordinates: [width, length, height] = [original_z, original_x, original_y]
    direction_swapped = np.array([direction[2], direction[0], direction[1]])
    direction_normalized_swapped = direction_swapped / (np.linalg.norm(direction_swapped) + 1e-8)
    direction_end_swapped = eye_swapped + direction_normalized_swapped * distance * 0.9
    fig.add_trace(go.Scatter3d(
        x=[eye_swapped[0], direction_end_swapped[0]],
        y=[eye_swapped[1], direction_end_swapped[1]],
        z=[eye_swapped[2], direction_end_swapped[2]],
        mode='lines',
        line=dict(color='red', width=4, dash='dash'),
        name="Camera Direction",
        hovertemplate=f"Camera Direction<br>Distance: {distance:.2f}m<extra></extra>"
    ))
    
    # Draw capture area on target panel
    # Calculate capture area based on FOV and distance
    # For roof shots, show circular area on roof
    # For wall shots, show rectangular area on wall
    is_roof = 'roof' in target_panel.lower()
    
    if is_roof:
        # Roof: circular capture area
        # Coordinate system: X=Width, Y=Length, Z=Height
        # Use swapped coordinates for visualization
        num_points = 32
        angles = np.linspace(0, 2*np.pi, num_points)
        capture_radius = view_size / 2
        # Calculate in original coordinates, then swap
        capture_circle_orig_x = at[0] + capture_radius * np.cos(angles)
        capture_circle_orig_y = np.full(num_points, at[1])  # At roof height
        capture_circle_orig_z = at[2] + capture_radius * np.sin(angles)
        # Swap: [width, length, height] = [original_z, original_x, original_y]
        capture_circle_x = capture_circle_orig_z
        capture_circle_y = capture_circle_orig_x
        capture_circle_z = capture_circle_orig_y
        
        fig.add_trace(go.Scatter3d(
            x=capture_circle_x,
            y=capture_circle_y,
            z=capture_circle_z,
            mode='lines',
            line=dict(color='lime', width=3),
            name="Capture Area",
            hovertemplate=f"Capture Area<br>Radius: {capture_radius:.2f}m<extra></extra>"
        ))
        
        # Draw view frustum lines from camera to circle points
        for i in range(0, num_points, 4):  # Sample every 4th point
            fig.add_trace(go.Scatter3d(
                x=[eye_swapped[0], capture_circle_x[i]],
                y=[eye_swapped[1], capture_circle_y[i]],
                z=[eye_swapped[2], capture_circle_z[i]],
                mode='lines',
                line=dict(color='cyan', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        # Wall shots: rectangular capture area
        # Coordinate system: X=Width, Y=Length, Z=Height
        # Calculate perpendicular vectors to camera direction (in original coordinates)
        right_vec = np.cross(up, direction_normalized)
        right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-8)
        up_vec = np.cross(direction_normalized, right_vec)
        up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)
        
        # Create rectangular capture area in original coordinates
        half_width = view_size / 2
        half_height = view_size / 2
        
        capture_corners_original = np.array([
            at - half_width * right_vec - half_height * up_vec,
            at + half_width * right_vec - half_height * up_vec,
            at + half_width * right_vec + half_height * up_vec,
            at - half_width * right_vec + half_height * up_vec,
            at - half_width * right_vec - half_height * up_vec,  # Close loop
        ])
        # Swap coordinates: [width, length, height] = [original_z, original_x, original_y]
        capture_corners = np.array([
            [corner[2], corner[0], corner[1]] for corner in capture_corners_original
        ])
        
        fig.add_trace(go.Scatter3d(
            x=capture_corners[:, 0],
            y=capture_corners[:, 1],
            z=capture_corners[:, 2],
            mode='lines',
            line=dict(color='lime', width=3),
            name="Capture Area",
            hovertemplate=f"Capture Area<br>Size: {view_size:.2f}m × {view_size:.2f}m<extra></extra>"
        ))
        
        # Draw view frustum lines from camera to rectangle corners
        for corner in capture_corners[:4]:
            fig.add_trace(go.Scatter3d(
                x=[eye_swapped[0], corner[0]],
                y=[eye_swapped[1], corner[1]],
                z=[eye_swapped[2], corner[2]],
                mode='lines',
                line=dict(color='cyan', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout
    title = f"Camera Simulation: {target_panel}"
    if capture_section:
        title += f"<br><sub>{capture_section}</sub>"
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (Width, meters)",
            yaxis_title="Y (Length, meters)",
            zaxis_title="Z (Height, meters)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    return fig

def main():
    st.title("📊 Output Scene Viewer")
    st.markdown("**Comprehensive visualization of container scene data with RGB-D images, depth analysis, and statistics**")
    
    # Load data
    containers = load_output_scene_data()
    
    if not containers:
        st.error("❌ No output_scene data found. Please ensure output_scene directory exists.")
        return
    
    # Sidebar - Container and shot selection
    with st.sidebar:
        st.header("🎛️ Navigation")
        
        # Container selection
        container_names = sorted(containers.keys())
        selected_container_name = st.selectbox(
            "Select Container:",
            container_names,
            format_func=lambda x: x.replace('container_', '').replace('_', ' ').title()
        )
        
        selected_container = containers[selected_container_name]
        
        # Container info
        st.divider()
        st.subheader("📦 Container Info")
        
        if selected_container['comparison_summary']:
            summary = selected_container['comparison_summary']
            st.write(f"**Type:** {summary.get('container_type', 'Unknown')}")
            if 'timestamp' in summary:
                try:
                    ts = datetime.fromisoformat(summary['timestamp'])
                    st.write(f"**Generated:** {ts.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.write(f"**Generated:** {summary['timestamp']}")
        
        # Shot selection
        st.divider()
        st.subheader("📸 Shot Selection")
        
        # Group shots by category
        shot_categories = {}
        for shot_name in sorted(selected_container['shots'].keys()):
            category = get_shot_category(shot_name)
            if category not in shot_categories:
                shot_categories[category] = []
            shot_categories[category].append(shot_name)
        
        # Create selectbox with grouped options
        all_shots = []
        for category in sorted(shot_categories.keys()):
            all_shots.extend(shot_categories[category])
        
        selected_shot_name = st.selectbox(
            "Select Shot:",
            all_shots,
            format_func=format_shot_name
        )
        
        selected_shot = selected_container['shots'][selected_shot_name]
        
        # Shot statistics
        if selected_shot['stats']:
            st.divider()
            st.subheader("📊 Shot Statistics")
            stats = selected_shot['stats']
            
            st.metric("Dent Pixels", f"{stats.get('dent_pixels', 0):,}")
            st.metric("Dent Percentage", f"{stats.get('dent_percentage', 0):.2f}%")
            st.metric("Max Depth Diff", f"{stats.get('max_depth_diff_mm', 0):.2f} mm")
            st.metric("Mean Depth Diff", f"{stats.get('mean_depth_diff_mm', 0):.2f} mm")
        
        # Refresh button
        st.divider()
        if st.button("🔄 Refresh Data"):
            load_output_scene_data.clear()
            st.rerun()
    
    # Main content area
    container_id = selected_container_name.replace('container_', '')
    
    # Header with container and shot info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"📦 {selected_container_name.replace('_', ' ').title()}")
        st.caption(f"📸 Shot: {format_shot_name(selected_shot_name)}")
    
    with col2:
        if selected_shot['stats']:
            stats = selected_shot['stats']
            severity_color = 'red' if stats.get('max_depth_diff_mm', 0) > 50 else 'orange' if stats.get('max_depth_diff_mm', 0) > 20 else 'green'
            st.metric(
                "Max Depth Difference",
                f"{stats.get('max_depth_diff_mm', 0):.2f} mm",
                delta=f"{stats.get('mean_depth_diff_mm', 0):.2f} mm avg"
            )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🖼️ Images", "📊 Statistics", "📈 Comparison", "🔍 Depth Analysis", "☁️ 3D Point Cloud", "📷 Camera Simulation"])
    
    with tab1:
        st.subheader("Image Visualization")
        
        shot_files = selected_shot['files']
        
        # RGB Images
        st.markdown("### RGB Images")
        rgb_col1, rgb_col2 = st.columns(2)
        
        with rgb_col1:
            if shot_files['original_rgb'] and shot_files['original_rgb'].exists():
                original_rgb = imageio.imread(shot_files['original_rgb'])
                st.image(original_rgb, caption="Original RGB", use_container_width=True)
            else:
                st.info("Original RGB image not found")
        
        with rgb_col2:
            if shot_files['dented_rgb'] and shot_files['dented_rgb'].exists():
                dented_rgb = imageio.imread(shot_files['dented_rgb'])
                st.image(dented_rgb, caption="Dented RGB", use_container_width=True)
            else:
                st.info("Dented RGB image not found")
        
        # Depth Images
        st.markdown("### Depth Images")
        
        depth_colormap = st.selectbox(
            "Depth Colormap:",
            ['viridis', 'plasma', 'inferno', 'magma', 'jet'],
            key='depth_colormap'
        )
        
        depth_col1, depth_col2, depth_col3 = st.columns(3)
        
        with depth_col1:
            if shot_files['original_depth_png'] and shot_files['original_depth_png'].exists():
                original_depth = imageio.imread(shot_files['original_depth_png'])
                original_depth_colored = apply_colormap(original_depth, depth_colormap)
                st.image(original_depth_colored, caption="Original Depth", use_container_width=True)
                
                # Load NPY for statistics
                if shot_files['original_depth_npy'] and shot_files['original_depth_npy'].exists():
                    original_depth_data = load_npy_depth(shot_files['original_depth_npy'])
                    if original_depth_data is not None:
                        st.caption(f"Range: {np.min(original_depth_data):.3f} - {np.max(original_depth_data):.3f} m")
            else:
                st.info("Original depth image not found")
        
        with depth_col2:
            if shot_files['dented_depth_png'] and shot_files['dented_depth_png'].exists():
                dented_depth = imageio.imread(shot_files['dented_depth_png'])
                dented_depth_colored = apply_colormap(dented_depth, depth_colormap)
                st.image(dented_depth_colored, caption="Dented Depth", use_container_width=True)
                
                # Load NPY for statistics
                if shot_files['dented_depth_npy'] and shot_files['dented_depth_npy'].exists():
                    dented_depth_data = load_npy_depth(shot_files['dented_depth_npy'])
                    if dented_depth_data is not None:
                        st.caption(f"Range: {np.min(dented_depth_data):.3f} - {np.max(dented_depth_data):.3f} m")
            else:
                st.info("Dented depth image not found")
        
        with depth_col3:
            if shot_files['depth_diff_png'] and shot_files['depth_diff_png'].exists():
                depth_diff = imageio.imread(shot_files['depth_diff_png'])
                depth_diff_colored = apply_colormap(depth_diff, depth_colormap)
                st.image(depth_diff_colored, caption="Depth Difference", use_container_width=True)
                
                # Load NPY for statistics
                if shot_files['depth_diff_npy'] and shot_files['depth_diff_npy'].exists():
                    depth_diff_data = load_npy_depth(shot_files['depth_diff_npy'])
                    if depth_diff_data is not None:
                        st.caption(f"Range: {np.min(depth_diff_data):.3f} - {np.max(depth_diff_data):.3f} m")
            else:
                st.info("Depth difference image not found")
        
        # Dent Mask and Visual Output
        st.markdown("### Dent Detection Visualization")
        mask_col1, mask_col2 = st.columns(2)
        
        with mask_col1:
            st.markdown("#### Dent Mask")
            if shot_files['dent_mask'] and shot_files['dent_mask'].exists():
                dent_mask = imageio.imread(shot_files['dent_mask'])
                
                # Calculate mask statistics
                if len(dent_mask.shape) == 3:
                    mask_gray = dent_mask[:,:,0] if dent_mask.shape[2] >= 1 else np.mean(dent_mask, axis=2)
                else:
                    mask_gray = dent_mask
                
                white_pixels = np.sum(mask_gray > 127)
                total_pixels = mask_gray.size
                dent_percentage = (white_pixels / total_pixels) * 100
                
                st.image(dent_mask, caption=f"Dent Mask - {white_pixels:,} pixels ({dent_percentage:.2f}%)", use_container_width=True)
            else:
                st.info("Dent mask image not found")
        
        with mask_col2:
            st.markdown("#### Visual Output (Dent Labels)")
            if shot_files['visual_output'] and shot_files['visual_output'].exists():
                visual_output = imageio.imread(shot_files['visual_output'])
                st.image(visual_output, caption="RGB Image with Dent Labels", use_container_width=True)
            else:
                st.info("Visual output image not found")
    
    with tab2:
        st.subheader("Shot Statistics")
        
        if selected_shot['stats']:
            stats = selected_shot['stats']
            
            # Key metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Dent Pixels", f"{stats.get('dent_pixels', 0):,}")
            with metric_col2:
                st.metric("Dent Percentage", f"{stats.get('dent_percentage', 0):.2f}%")
            with metric_col3:
                st.metric("Max Depth Diff", f"{stats.get('max_depth_diff_mm', 0):.2f} mm")
            
            st.divider()
            
            # Depth metrics
            st.markdown("### Depth Metrics")
            depth_metric_col1, depth_metric_col2 = st.columns(2)
            
            with depth_metric_col1:
                st.metric("Max Depth Difference", f"{stats.get('max_depth_diff_mm', 0):.2f} mm")
            with depth_metric_col2:
                st.metric("Mean Depth Difference", f"{stats.get('mean_depth_diff_mm', 0):.2f} mm")
            
            # Visual indicators
            st.divider()
            st.markdown("### Severity Assessment")
            
            max_depth_mm = stats.get('max_depth_diff_mm', 0)
            mean_depth_mm = stats.get('mean_depth_diff_mm', 0)
            dent_percentage = stats.get('dent_percentage', 0)
            
            if max_depth_mm > 50:
                st.error(f"🔴 **SEVERE** - Maximum depth difference: {max_depth_mm:.2f} mm")
            elif max_depth_mm > 20:
                st.warning(f"🟠 **MODERATE** - Maximum depth difference: {max_depth_mm:.2f} mm")
            elif max_depth_mm > 5:
                st.info(f"🟡 **MINOR** - Maximum depth difference: {max_depth_mm:.2f} mm")
            else:
                st.success(f"🟢 **SUPERFICIAL** - Maximum depth difference: {max_depth_mm:.2f} mm")
            
            # IICL-6 Compliance
            st.divider()
            st.markdown("### IICL-6 Compliance Check")
            if max_depth_mm > 5.0:
                st.error("❌ **FAIL** - Depth exceeds 5mm threshold (IICL-6 standard)")
            else:
                st.success("✅ **PASS** - Depth within 5mm threshold (IICL-6 standard)")
        else:
            st.warning("No statistics available for this shot")
    
    with tab3:
        st.subheader("Container Comparison")
        
        if selected_container['comparison_summary']:
            summary = selected_container['comparison_summary']
            
            # Overall container statistics
            st.markdown("### Container Overview")
            overview_col1, overview_col2, overview_col3 = st.columns(3)
            
            with overview_col1:
                st.metric("Container Type", summary.get('container_type', 'Unknown'))
            with overview_col2:
                st.metric("Threshold", f"{summary.get('threshold_mm', 0):.1f} mm")
            with overview_col3:
                if 'timestamp' in summary:
                    try:
                        ts = datetime.fromisoformat(summary['timestamp'])
                        st.metric("Generated", ts.strftime('%Y-%m-%d'))
                    except:
                        st.metric("Generated", summary['timestamp'])
            
            # Shot comparison table
            if 'shots' in summary and summary['shots']:
                st.divider()
                st.markdown("### All Shots Comparison")
                
                # Create DataFrame
                shots_data = []
                for shot_data in summary['shots']:
                    shots_data.append({
                        'Shot Name': format_shot_name(shot_data.get('shot_name', '')),
                        'Category': get_shot_category(shot_data.get('shot_name', '')),
                        'Dent Pixels': shot_data.get('dent_pixels', 0),
                        'Dent %': f"{shot_data.get('dent_percentage', 0):.2f}%",
                        'Max Depth (mm)': f"{shot_data.get('max_depth_diff_mm', 0):.2f}",
                        'Mean Depth (mm)': f"{shot_data.get('mean_depth_diff_mm', 0):.2f}"
                    })
                
                df = pd.DataFrame(shots_data)
                
                # Display table
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.markdown("#### Max Depth by Shot")
                    fig1 = px.bar(
                        df,
                        x='Shot Name',
                        y='Max Depth (mm)',
                        color='Category',
                        title="Maximum Depth Difference by Shot"
                    )
                    fig1.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with chart_col2:
                    st.markdown("#### Dent Coverage by Shot")
                    fig2 = px.bar(
                        df,
                        x='Shot Name',
                        y='Dent %',
                        color='Category',
                        title="Dent Coverage Percentage by Shot"
                    )
                    fig2.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Summary statistics
                st.divider()
                st.markdown("### Summary Statistics")
                
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                total_shots = len(summary['shots'])
                shots_with_dents = sum(1 for s in summary['shots'] if s.get('dent_pixels', 0) > 0)
                max_depth_all = max((s.get('max_depth_diff_mm', 0) for s in summary['shots']), default=0)
                avg_dent_percentage = np.mean([s.get('dent_percentage', 0) for s in summary['shots']])
                
                with summary_col1:
                    st.metric("Total Shots", total_shots)
                with summary_col2:
                    st.metric("Shots with Dents", shots_with_dents)
                with summary_col3:
                    st.metric("Max Depth (All Shots)", f"{max_depth_all:.2f} mm")
                with summary_col4:
                    st.metric("Avg Dent Coverage", f"{avg_dent_percentage:.2f}%")
        else:
            st.warning("No comparison summary available for this container")
    
    with tab4:
        st.subheader("Depth Analysis")
        
        # Load depth data
        depth_diff_data = None
        if shot_files['depth_diff_npy'] and shot_files['depth_diff_npy'].exists():
            depth_diff_data = load_npy_depth(shot_files['depth_diff_npy'])
        
        original_depth_data = None
        if shot_files['original_depth_npy'] and shot_files['original_depth_npy'].exists():
            original_depth_data = load_npy_depth(shot_files['original_depth_npy'])
        
        dented_depth_data = None
        if shot_files['dented_depth_npy'] and shot_files['dented_depth_npy'].exists():
            dented_depth_data = load_npy_depth(shot_files['dented_depth_npy'])
        
        if depth_diff_data is not None:
            # Depth histogram
            st.markdown("### Depth Difference Distribution")
            
            # Flatten and filter out zeros/near-zeros
            depth_flat = depth_diff_data.flatten()
            depth_filtered = depth_flat[depth_flat > 0.001]  # Filter out near-zero values
            
            if len(depth_filtered) > 0:
                hist_col1, hist_col2 = st.columns(2)
                
                with hist_col1:
                    fig_hist = px.histogram(
                        x=depth_filtered * 1000,  # Convert to mm
                        nbins=50,
                        title="Depth Difference Histogram (mm)",
                        labels={'x': 'Depth Difference (mm)', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with hist_col2:
                    # Statistics
                    st.markdown("#### Key Statistics")
                    st.metric("Non-zero Pixels", f"{len(depth_filtered):,}")
                    st.metric("Mean Depth Diff", f"{np.mean(depth_filtered) * 1000:.2f} mm")
                    st.metric("Median Depth Diff", f"{np.median(depth_filtered) * 1000:.2f} mm")
                    st.metric("Max Depth Diff", f"{np.max(depth_filtered) * 1000:.2f} mm")
                    st.metric("95th Percentile", f"{np.percentile(depth_filtered, 95) * 1000:.2f} mm")
            
            # 3D depth visualization
            st.markdown("### 3D Depth Visualization")
            
            # Sample data for 3D plot (downsample for performance)
            sample_rate = max(1, depth_diff_data.shape[0] // 100)
            depth_sampled = depth_diff_data[::sample_rate, ::sample_rate]
            
            y_coords, x_coords = np.meshgrid(
                np.arange(0, depth_sampled.shape[0]) * sample_rate,
                np.arange(0, depth_sampled.shape[1]) * sample_rate,
                indexing='ij'
            )
            
            fig_3d = go.Figure(data=[go.Surface(
                z=depth_sampled * 1000,  # Convert to mm
                x=x_coords,
                y=y_coords,
                colorscale='Viridis',
                colorbar=dict(title="Depth Difference (mm)")
            )])
            
            fig_3d.update_layout(
                title="3D Depth Difference Surface",
                scene=dict(
                    xaxis_title="X (pixels)",
                    yaxis_title="Y (pixels)",
                    zaxis_title="Depth Difference (mm)"
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
        else:
            st.warning("Depth difference data (NPY) not available for analysis")
    
    with tab5:
        st.subheader("3D Point Cloud Visualization")
        
        shot_files = selected_shot['files']
        
        # Check which PLY files are available
        available_plys = []
        if shot_files['original_pointcloud_ply'] and shot_files['original_pointcloud_ply'].exists():
            available_plys.append(('Original', shot_files['original_pointcloud_ply']))
        if shot_files['dented_pointcloud_ply'] and shot_files['dented_pointcloud_ply'].exists():
            available_plys.append(('Dented', shot_files['dented_pointcloud_ply']))
        
        if available_plys:
            # Point cloud selector
            if len(available_plys) > 1:
                selected_ply_name = st.selectbox(
                    "Select Point Cloud:",
                    [name for name, _ in available_plys],
                    index=1  # Default to dented if available
                )
                selected_ply_path = next(path for name, path in available_plys if name == selected_ply_name)
            else:
                selected_ply_name = available_plys[0][0]
                selected_ply_path = available_plys[0][1]
            
            if selected_ply_path and selected_ply_path.exists():
                # Load PLY file
                with st.spinner(f"Loading {selected_ply_name.lower()} point cloud..."):
                    vertices, colors = load_ply_pointcloud(selected_ply_path)
                
                if vertices is not None and len(vertices) > 0:
                    # Get camera pose for this shot to transform to camera view
                    container_type = "20ft"  # Default
                    if selected_container['comparison_summary']:
                        container_type = selected_container['comparison_summary'].get('container_type', '20ft')
                    
                    camera_pose = None
                    vertices_camera_view = vertices
                    camera_view_eye = None
                    
                    if CAMERA_POSE_GENERATOR_AVAILABLE:
                        camera_pose = get_camera_pose_for_shot(selected_shot_name, container_type)
                        if camera_pose:
                            # Transform point cloud to camera view coordinates
                            vertices_camera_view = transform_pointcloud_to_camera_view(vertices, camera_pose)
                            # Calculate scene bounds for camera positioning
                            scene_bounds = (vertices_camera_view.min(axis=0), vertices_camera_view.max(axis=0))
                            camera_view_eye = get_camera_view_eye_position(camera_pose, scene_bounds)
                    
                    # Point cloud statistics (use camera view vertices for stats)
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        st.metric("Total Points", f"{len(vertices):,}")
                    with stats_col2:
                        bounds = vertices_camera_view.max(axis=0) - vertices_camera_view.min(axis=0)
                        st.metric("Size (Width × Depth × Height)", f"{bounds[0]:.2f} × {bounds[1]:.2f} × {bounds[2]:.2f} m")
                    
                    st.divider()
                    
                    # Visualization options
                    viz_col1, viz_col2 = st.columns([3, 1])
                    
                    with viz_col2:
                        st.markdown("### Visualization Options")
                        
                        color_mode = st.selectbox(
                            "Color Mode:",
                            ["Y Coordinate (Depth)", "Z Coordinate (Height)", "Uniform Color"],
                            index=0
                        )
                        
                        point_size = st.slider("Point Size", 1, 5, 1)
                        
                        max_display_points = st.slider(
                            "Max Points to Display",
                            1000, 100000, 50000,
                            help="Reduce for better performance with large point clouds"
                        )
                        
                        if camera_pose:
                            st.info("📷 View aligned with camera capture")
                    
                    with viz_col1:
                        # Prepare visualization
                        display_vertices = vertices_camera_view
                        display_colors = colors
                        
                        # Downsample if needed
                        if len(display_vertices) > max_display_points:
                            indices = np.random.choice(len(display_vertices), max_display_points, replace=False)
                            display_vertices = display_vertices[indices]
                            if display_colors is not None:
                                display_colors = display_colors[indices]
                        
                        # Create visualization based on color mode
                        if color_mode == "Y Coordinate (Depth)":
                            y_coords = display_vertices[:, 1]  # Y axis = Depth
                            y_normalized = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min() + 1e-8)
                            color_data = y_normalized
                            color_title = "Y Coordinate (Depth)"
                            use_colorscale = True
                        elif color_mode == "Z Coordinate (Height)":
                            z_coords = display_vertices[:, 2]  # Z axis = Height
                            z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-8)
                            color_data = z_normalized
                            color_title = "Z Coordinate (Height)"
                            use_colorscale = True
                        else:  # Uniform Color
                            color_data = 'blue'
                            color_title = "Uniform"
                            use_colorscale = False
                        
                        # Create plotly figure
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter3d(
                            x=display_vertices[:, 0],  # X = Width
                            y=display_vertices[:, 1],  # Y = Depth
                            z=display_vertices[:, 2],  # Z = Height
                            mode='markers',
                            marker=dict(
                                size=point_size,
                                color=color_data,
                                colorscale='Viridis' if use_colorscale else None,
                                showscale=use_colorscale,
                                colorbar=dict(title=color_title) if use_colorscale else None,
                                opacity=0.8
                            ),
                            name="Point Cloud"
                        ))
                        
                        # Set camera view
                        if camera_view_eye:
                            camera_dict = camera_view_eye
                        else:
                            # Default view
                            camera_dict = dict(eye=dict(x=1.5, y=-1.5, z=1.5))
                        
                        fig.update_layout(
                            title=f"3D Point Cloud ({selected_ply_name}) - {format_shot_name(selected_shot_name)}",
                            scene=dict(
                                xaxis_title="X (Width, meters)",
                                yaxis_title="Y (Depth, meters)",
                                zaxis_title="Z (Height, meters)",
                                camera=camera_dict,
                                aspectmode='data'
                            ),
                            height=700,
                            margin=dict(l=0, r=0, b=0, t=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Failed to load point cloud data from PLY file")
        else:
            st.info("📁 No PLY point cloud files found for this shot.")
            st.caption("PLY files are generated during rendering and contain 3D point cloud data from the scene.")
            st.caption("💡 Tip: Re-run the comparison script to generate PLY files from depth maps.")
    
    with tab6:
        st.subheader("Camera Simulation: How the Dented Panel is Captured")
        st.markdown("**Visualization of camera position, view direction, and target panel for this shot**")
        
        # Get container type from summary
        container_type = "20ft"  # Default
        if selected_container['comparison_summary']:
            container_type = selected_container['comparison_summary'].get('container_type', '20ft')
        
        # Get camera pose using CameraPoseGenerator
        if not CAMERA_POSE_GENERATOR_AVAILABLE:
            st.error("❌ CameraPoseGenerator not available. Cannot simulate camera poses.")
            st.info("Please ensure `camera_position.py` and `config.py` are available.")
        else:
            camera_pose = get_camera_pose_for_shot(selected_shot_name, container_type)
            
            if camera_pose is None:
                st.warning(f"⚠️ Could not find camera pose for shot: {selected_shot_name}")
                st.info("This may happen if the shot name doesn't match any generated poses.")
            else:
        
                # Display camera information
                eye = camera_pose['eye']
                at = camera_pose['at']
                direction = at - eye
                distance = np.linalg.norm(direction)
                fov_rad = np.deg2rad(CAMERA_FOV)
                view_size = 2 * distance * np.tan(fov_rad / 2)
                
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.markdown("### 📍 Camera Info")
                    st.metric("Target Panel", camera_pose['target_panel'])
                    st.metric("Distance to Target", f"{distance:.2f} m")
                    st.metric("Field of View", f"{CAMERA_FOV}°")
                
                with info_col2:
                    st.markdown("### 📐 View Info")
                    st.metric("View Size", f"{view_size:.2f} m")
                    st.metric("Image Resolution", f"{IMAGE_SIZE}×{IMAGE_SIZE}")
                    st.write(f"**Capture Section:** {camera_pose.get('capture_section', 'N/A')}")
                
                with info_col3:
                    st.markdown("### 🎯 Panel Info")
                    if CAMERA_POSE_GENERATOR_AVAILABLE:
                        spec = ContainerConfig().CONTAINER_SPECS[container_type]
                    else:
                        specs = {
                            "20ft": (6.058, 2.591, 2.438),
                            "40ft": (12.192, 2.591, 2.438),
                            "40ft_hc": (12.192, 2.896, 2.438)
                        }
                        spec = specs.get(container_type, specs["20ft"])
                    length, height, width = spec["external_y_up"]
                    st.metric("Container Type", container_type)
                    st.write(f"**Dimensions:** {length:.2f}m × {width:.2f}m × {height:.2f}m")
                    st.write(f"**(Length × Width × Height)**")
                
                st.divider()
                
                # 3D Visualization
                st.markdown("### 🎨 3D Visualization")
                
                fig = create_camera_simulation_plot(camera_pose, container_type)
                st.plotly_chart(fig, use_container_width=True)
                
                # Legend
                st.caption("**Legend:** Red Diamond = Camera Position | Yellow Circle = Target Point | Green Outline = Capture Area | Cyan Lines = View Frustum")

if __name__ == "__main__":
    main()

