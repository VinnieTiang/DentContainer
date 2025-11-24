#!/usr/bin/env python3
"""
Output Scene Viewer - Comprehensive UI for visualizing container scene data
Visualizes all containers, shots, RGB-D images, depth differences, and statistics
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
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)",
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
            st.write(f"**Original:** {summary.get('original_file', 'Unknown')}")
            st.write(f"**Dented:** {summary.get('dented_file', 'Unknown')}")
            if 'timestamp' in summary:
                try:
                    ts = datetime.fromisoformat(summary['timestamp'])
                    st.write(f"**Generated:** {ts.strftime('%Y-%m-%d %H:%M:%S')}")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🖼️ Images", "📊 Statistics", "📈 Comparison", "🔍 Depth Analysis", "☁️ 3D Point Cloud"])
    
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
                st.caption(f"Resolution: {original_rgb.shape[1]}×{original_rgb.shape[0]} pixels")
            else:
                st.info("Original RGB image not found")
        
        with rgb_col2:
            if shot_files['dented_rgb'] and shot_files['dented_rgb'].exists():
                dented_rgb = imageio.imread(shot_files['dented_rgb'])
                st.image(dented_rgb, caption="Dented RGB", use_container_width=True)
                st.caption(f"Resolution: {dented_rgb.shape[1]}×{dented_rgb.shape[0]} pixels")
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
        
        # Dent Mask
        st.markdown("### Dent Mask")
        mask_col1, mask_col2 = st.columns([2, 1])
        
        # Initialize mask variables
        dent_mask = None
        white_pixels = 0
        total_pixels = 0
        dent_percentage = 0.0
        
        with mask_col1:
            if shot_files['dent_mask'] and shot_files['dent_mask'].exists():
                dent_mask = imageio.imread(shot_files['dent_mask'])
                st.image(dent_mask, caption="Dent Mask", use_container_width=True)
                
                # Calculate mask statistics
                if len(dent_mask.shape) == 3:
                    mask_gray = dent_mask[:,:,0] if dent_mask.shape[2] >= 1 else np.mean(dent_mask, axis=2)
                else:
                    mask_gray = dent_mask
                
                white_pixels = np.sum(mask_gray > 127)
                total_pixels = mask_gray.size
                dent_percentage = (white_pixels / total_pixels) * 100
                
                st.caption(f"Dent pixels: {white_pixels:,} ({dent_percentage:.2f}%)")
            else:
                st.info("Dent mask image not found")
        
        with mask_col2:
            if shot_files['dent_mask'] and shot_files['dent_mask'].exists() and dent_mask is not None:
                st.metric("Dent Pixels", f"{white_pixels:,}")
                st.metric("Total Pixels", f"{total_pixels:,}")
                st.metric("Dent Coverage", f"{dent_percentage:.2f}%")
            else:
                st.info("No mask data")
    
    with tab2:
        st.subheader("Shot Statistics")
        
        if selected_shot['stats']:
            stats = selected_shot['stats']
            
            # Key metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Dent Pixels", f"{stats.get('dent_pixels', 0):,}")
            with metric_col2:
                st.metric("Total Pixels", f"{stats.get('total_pixels', 0):,}")
            with metric_col3:
                st.metric("Dent Percentage", f"{stats.get('dent_percentage', 0):.2f}%")
            with metric_col4:
                st.metric("Output Directory", stats.get('output_dir', 'N/A'))
            
            st.divider()
            
            # Depth metrics
            st.markdown("### Depth Metrics")
            depth_metric_col1, depth_metric_col2, depth_metric_col3, depth_metric_col4 = st.columns(4)
            
            with depth_metric_col1:
                st.metric("Max Depth Diff (m)", f"{stats.get('max_depth_diff_m', 0):.4f}")
            with depth_metric_col2:
                st.metric("Max Depth Diff (mm)", f"{stats.get('max_depth_diff_mm', 0):.2f}")
            with depth_metric_col3:
                st.metric("Mean Depth Diff (m)", f"{stats.get('mean_depth_diff_m', 0):.4f}")
            with depth_metric_col4:
                st.metric("Mean Depth Diff (mm)", f"{stats.get('mean_depth_diff_mm', 0):.2f}")
            
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
                    st.markdown("#### Statistics")
                    st.write(f"**Non-zero pixels:** {len(depth_filtered):,}")
                    st.write(f"**Mean:** {np.mean(depth_filtered) * 1000:.2f} mm")
                    st.write(f"**Median:** {np.median(depth_filtered) * 1000:.2f} mm")
                    st.write(f"**Std Dev:** {np.std(depth_filtered) * 1000:.2f} mm")
                    st.write(f"**Min:** {np.min(depth_filtered) * 1000:.2f} mm")
                    st.write(f"**Max:** {np.max(depth_filtered) * 1000:.2f} mm")
                    st.write(f"**95th Percentile:** {np.percentile(depth_filtered, 95) * 1000:.2f} mm")
                    st.write(f"**99th Percentile:** {np.percentile(depth_filtered, 99) * 1000:.2f} mm")
            
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
            
            # Comparison with original and dented
            if original_depth_data is not None and dented_depth_data is not None:
                st.markdown("### Depth Comparison")
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown("#### Original Depth")
                    orig_flat = original_depth_data.flatten()
                    fig_orig = px.histogram(
                        x=orig_flat,
                        nbins=50,
                        title="Original Depth Distribution (m)",
                        labels={'x': 'Depth (m)', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig_orig, use_container_width=True)
                
                with comp_col2:
                    st.markdown("#### Dented Depth")
                    dented_flat = dented_depth_data.flatten()
                    fig_dented = px.histogram(
                        x=dented_flat,
                        nbins=50,
                        title="Dented Depth Distribution (m)",
                        labels={'x': 'Depth (m)', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig_dented, use_container_width=True)
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
                    # Point cloud statistics
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    with stats_col1:
                        st.metric("Total Points", f"{len(vertices):,}")
                    with stats_col2:
                        st.metric("Has Colors", "Yes" if colors is not None else "No")
                    with stats_col3:
                        bounds = vertices.max(axis=0) - vertices.min(axis=0)
                        st.metric("Size (X×Y×Z)", f"{bounds[0]:.2f}×{bounds[1]:.2f}×{bounds[2]:.2f} m")
                    with stats_col4:
                        center = vertices.mean(axis=0)
                        st.metric("Center", f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
                    
                    st.divider()
                    
                    # Visualization options
                    viz_col1, viz_col2 = st.columns([3, 1])
                    
                    with viz_col2:
                        st.markdown("### Visualization Options")
                        
                        color_mode = st.selectbox(
                            "Color Mode:",
                            ["Z Coordinate (Depth)", "RGB Colors", "Uniform Color"],
                            index=0 if colors is None else 1
                        )
                        
                        point_size = st.slider("Point Size", 1, 10, 2)
                        
                        max_display_points = st.slider(
                            "Max Points to Display",
                            1000, 100000, 50000,
                            help="Reduce for better performance with large point clouds"
                        )
                    
                    with viz_col1:
                        # Prepare visualization
                        display_vertices = vertices
                        display_colors = colors
                        
                        # Downsample if needed
                        if len(display_vertices) > max_display_points:
                            indices = np.random.choice(len(display_vertices), max_display_points, replace=False)
                            display_vertices = display_vertices[indices]
                            if display_colors is not None:
                                display_colors = display_colors[indices]
                        
                        # Create visualization based on color mode
                        if color_mode == "Z Coordinate (Depth)":
                            z_coords = display_vertices[:, 2]
                            z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-8)
                            color_data = z_normalized
                            color_title = "Z Coordinate (Depth)"
                            use_colorscale = True
                        elif color_mode == "RGB Colors" and display_colors is not None:
                            if display_colors.shape[1] == 3:
                                if display_colors.max() > 1.0:
                                    display_colors = display_colors / 255.0
                                color_data = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in display_colors]
                                color_title = "RGB Colors"
                                use_colorscale = False
                            else:
                                color_data = 'blue'
                                color_title = "Uniform"
                                use_colorscale = False
                        else:
                            color_data = 'blue'
                            color_title = "Uniform"
                            use_colorscale = False
                        
                        # Create plotly figure
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter3d(
                            x=display_vertices[:, 0],
                            y=display_vertices[:, 1],
                            z=display_vertices[:, 2],
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
                        
                        fig.update_layout(
                            title=f"3D Point Cloud ({selected_ply_name}) - {format_shot_name(selected_shot_name)}",
                            scene=dict(
                                xaxis_title="X (meters)",
                                yaxis_title="Y (meters)",
                                zaxis_title="Z (meters)",
                                camera=dict(
                                    eye=dict(x=1.5, y=1.5, z=1.5)
                                ),
                                aspectmode='data'
                            ),
                            height=700,
                            margin=dict(l=0, r=0, b=0, t=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional information
                    st.divider()
                    st.markdown("### Point Cloud Details")
                    
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        st.markdown("#### Bounding Box")
                        bbox_min = vertices.min(axis=0)
                        bbox_max = vertices.max(axis=0)
                        st.write(f"**Min:** ({bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}) m")
                        st.write(f"**Max:** ({bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}) m")
                        st.write(f"**Extent:** ({bbox_max[0]-bbox_min[0]:.3f}, {bbox_max[1]-bbox_min[1]:.3f}, {bbox_max[2]-bbox_min[2]:.3f}) m")
                    
                    with info_col2:
                        st.markdown("#### Statistics")
                        st.write(f"**Mean:** ({vertices[:, 0].mean():.3f}, {vertices[:, 1].mean():.3f}, {vertices[:, 2].mean():.3f}) m")
                        st.write(f"**Std Dev:** ({vertices[:, 0].std():.3f}, {vertices[:, 1].std():.3f}, {vertices[:, 2].std():.3f}) m")
                        st.write(f"**File:** {selected_ply_path.name}")
                else:
                    st.error("Failed to load point cloud data from PLY file")
        else:
            st.info("📁 No PLY point cloud files found for this shot.")
            st.caption("PLY files are generated during rendering and contain 3D point cloud data from the scene.")
            st.caption("💡 Tip: Re-run the comparison script to generate PLY files from depth maps.")

if __name__ == "__main__":
    main()

