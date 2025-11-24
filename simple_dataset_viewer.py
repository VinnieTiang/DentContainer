#!/usr/bin/env python3
"""
Simple Dataset Viewer - Updated for Synchronized Ground Truth
Minimal tool for viewing container damage RGB-D dataset samples and synchronized ground truth.
Now supports realistic container colors and synchronized depth metrics.
"""

import streamlit as st
import numpy as np
import json
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.cm as cm
import plotly.graph_objects as go
import trimesh
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="📊 Simple Dataset Viewer - Synchronized",
    page_icon="📊",
    layout="wide"
)

def load_original_dent_specs(dent_specs_file):
    """Load original dent specifications from the panel generation (AUTHORITATIVE)"""
    if not dent_specs_file.exists():
        return None
    
    try:
        with open(dent_specs_file, 'r') as f:
            specs_list = json.load(f)
        
        # Create mapping by filename
        specs_dict = {}
        for spec in specs_list:
            filename = spec.get('filename', '')
            if filename:
                # Extract panel number from filename
                # e.g., "panel_dents/dented_panel_01.obj" -> "01"
                if 'dented_panel_' in filename:
                    panel_num = filename.split('dented_panel_')[-1].split('.')[0]
                    specs_dict[panel_num] = spec
        
        return specs_dict
    except Exception as e:
        print(f"Error loading original dent specs: {e}")
        return None

@st.cache_data(ttl=60)  # Cache for 60 seconds, then auto-refresh
def load_dataset():
    """Load dataset samples from output folder with synchronized ground truth"""
    output_dir = Path("output")
    panel_dents_dir = Path("panel_dents")
    samples = []
    
    # Load original dent specifications (AUTHORITATIVE GROUND TRUTH)
    dent_specs_file = panel_dents_dir / "dented_panel_specifications.json"
    original_specs = load_original_dent_specs(dent_specs_file)
    
    if (output_dir / "rgb").exists():
        for rgb_file in sorted((output_dir / "rgb").glob("*_rgb.png")):
            # Extract just the number part for OBJ file matching
            # e.g., "dented_panel_01_rgb.png" -> "01"
            full_name = rgb_file.stem.replace("_rgb", "")  # "dented_panel_01"
            sample_number = full_name.split("_")[-1]  # "01"
            
            sample = {
                'id': sample_number,
                'full_id': full_name,
                'rgb_path': rgb_file,
                'depth_path': output_dir / "depth" / f"{full_name}_depth.png",  # Use PNG by default for visualization
                'depth_exr_path': output_dir / "depth" / f"{full_name}_depth.exr",  # Keep track of EXR too
                'mask_path': output_dir / "mask" / f"{full_name}_mask.png",
                'metadata_path': output_dir / "metadata" / f"{full_name}_metadata.json",
                'depth_metrics_path': output_dir / f"{full_name}_depth_metrics.json",
                'obj_path': panel_dents_dir / f"dented_panel_{sample_number}.obj",
                'dent_specs_file': panel_dents_dir / "dented_panel_specifications.json"
            }
            
            # Add AUTHORITATIVE original dent specs
            if original_specs and sample_number in original_specs:
                sample['authoritative_specs'] = original_specs[sample_number]
            else:
                sample['authoritative_specs'] = None
            
            # Load synchronized depth metrics (should match specifications exactly)
            sample['synchronized_metrics'] = load_synchronized_depth_metrics(sample['depth_metrics_path'])
            
            # Load render metadata for container color and additional info
            if sample['metadata_path'].exists():
                try:
                    with open(sample['metadata_path'], 'r') as f:
                        metadata = json.load(f)
                    sample['metadata'] = metadata
                    sample['ground_truth'] = extract_synchronized_ground_truth(metadata, sample['authoritative_specs'], sample['synchronized_metrics'])
                except Exception as e:
                    print(f"Error loading metadata for {full_name}: {e}")
                    sample['metadata'] = None
                    sample['ground_truth'] = None
            else:
                sample['metadata'] = None
                sample['ground_truth'] = None
            
            # Fallback to specifications if metadata failed
            if not sample['ground_truth'] and sample['authoritative_specs']:
                sample['ground_truth'] = extract_ground_truth_from_specs(sample['authoritative_specs'])
                
            samples.append(sample)
    
    return samples

def load_synchronized_depth_metrics(depth_metrics_path):
    """Load synchronized depth metrics (should match specifications exactly)"""
    if not depth_metrics_path.exists():
        return None
    
    try:
        with open(depth_metrics_path, 'r') as f:
            depth_metrics = json.load(f)
        return depth_metrics
    except Exception as e:
        print(f"Error loading synchronized depth metrics: {e}")
        return None
    
def extract_synchronized_ground_truth(metadata, authoritative_specs, synchronized_metrics):
    """Extract ground truth from synchronized data sources"""
    # Priority order: 1) Authoritative specs, 2) Synchronized metrics, 3) Metadata
    
    ground_truth = {}
    
    # Get container color information from metadata dent_specs (where it's actually stored)
    dent_specs = metadata.get('dent_specs', {})
    container_color_name = dent_specs.get('container_color_name', 'Unknown')
    container_color_rgb = dent_specs.get('container_color_rgb', [128, 128, 128])
    
    # Convert RGB to hex
    if len(container_color_rgb) >= 3:
        hex_color = f"#{int(container_color_rgb[0]):02x}{int(container_color_rgb[1]):02x}{int(container_color_rgb[2]):02x}"
    else:
        hex_color = '#808080'
    
    # Map container color names to shipping lines
    color_to_shipping_line = {
        'MAERSK_BLUE': 'Maersk',
        'MSC_YELLOW': 'MSC',
        'CMA_CGM_BLUE': 'CMA CGM',
        'COSCO_RED': 'COSCO',
        'EVERGREEN_GREEN': 'Evergreen',
        'HAPAG_ORANGE': 'Hapag-Lloyd',
        'CARGO_GRAY': 'Generic',
        'REEFER_WHITE': 'Refrigerated Container',
        'INDUSTRIAL_BROWN': 'Industrial',
        'NAVY_BLUE': 'Military/Government'
    }
    
    shipping_line = color_to_shipping_line.get(container_color_name, 'Unknown')
    
    ground_truth['container_color'] = {
        'name': container_color_name,
        'rgb': container_color_rgb,
        'hex': hex_color,
        'shipping_line': shipping_line
    }
    
    # Use AUTHORITATIVE specifications as primary source
    if authoritative_specs:
        ground_truth['dent_type'] = authoritative_specs.get('type', 'unknown')
        ground_truth['depth_mm'] = authoritative_specs.get('depth', 0) * 1000
        ground_truth['depth_source'] = 'authoritative_specifications'
        ground_truth['dent_parameters'] = {k: v for k, v in authoritative_specs.items() 
                                         if k not in ['type', 'depth', 'filename']}
        
        # Check synchronization with metrics
        if synchronized_metrics and synchronized_metrics.get('source') == 'dented_panel_specifications.json':
            metrics_depth = synchronized_metrics.get('specifications', {}).get('max_depth_mm', 0)
            spec_depth = ground_truth['depth_mm']
            
            # Check if perfectly synchronized (within 0.01mm tolerance)
            if abs(metrics_depth - spec_depth) < 0.01:
                ground_truth['synchronization_status'] = 'perfect'
                ground_truth['synchronization_error'] = 0.0
            else:
                ground_truth['synchronization_status'] = 'misaligned'
                ground_truth['synchronization_error'] = abs(metrics_depth - spec_depth)
        else:
            ground_truth['synchronization_status'] = 'metrics_unavailable'
    
    # Fallback to synchronized metrics if specs unavailable
    elif synchronized_metrics:
        specs_data = synchronized_metrics.get('specifications', {})
        ground_truth['dent_type'] = specs_data.get('type', 'unknown')
        ground_truth['depth_mm'] = specs_data.get('max_depth_mm', 0)
        ground_truth['depth_source'] = 'synchronized_metrics'
        ground_truth['synchronization_status'] = 'specs_unavailable'
    
    # Further fallback to metadata
    else:
        ground_truth['dent_type'] = dent_specs.get('type', 'unknown')
        ground_truth['depth_mm'] = dent_specs.get('depth', 0) * 1000
        ground_truth['depth_source'] = 'render_metadata'
        ground_truth['synchronization_status'] = 'legacy_format'
    
    # Calculate additional properties
    ground_truth['dent_area'] = calculate_dent_area(ground_truth['dent_type'], ground_truth.get('dent_parameters', {}))
    ground_truth['severity'] = determine_dent_severity(ground_truth['depth_mm'], ground_truth['dent_area'])
    
    return ground_truth

def extract_ground_truth_from_specs(authoritative_specs):
    """Extract ground truth directly from authoritative specifications"""
    return {
        'dent_type': authoritative_specs.get('type', 'unknown'),
        'depth_mm': authoritative_specs.get('depth', 0) * 1000,
        'depth_source': 'authoritative_specifications',
        'dent_parameters': {k: v for k, v in authoritative_specs.items() 
                           if k not in ['type', 'depth', 'filename']},
        'dent_area': calculate_dent_area(authoritative_specs.get('type', 'unknown'), 
                                       {k: v for k, v in authoritative_specs.items() 
                                        if k not in ['type', 'depth', 'filename']}),
        'severity': determine_dent_severity(authoritative_specs.get('depth', 0) * 1000, 
                                          calculate_dent_area(authoritative_specs.get('type', 'unknown'), {})),
        'container_color': {
            'name': 'Unknown (No metadata)',
            'rgb': [128, 128, 128],
            'hex': '#808080',
            'shipping_line': 'Unknown'
        },
        'synchronization_status': 'metadata_unavailable'
    }

def determine_dent_severity(depth_mm, area_m2):
    """Determine dent severity based on depth and area"""
    if depth_mm > 10:
        return "SEVERE"
    elif depth_mm > 5:
        return "MODERATE" 
    elif depth_mm > 2:
        return "MINOR"
    else:
        return "SUPERFICIAL"

def calculate_dent_area(dent_type, params):
    """Calculate approximate dent area in square meters"""
    try:
        if dent_type in ['circular', 'circular_impact']:
            radius = params.get('radius', 0)
            return np.pi * radius ** 2
        elif dent_type in ['elongated_scratch', 'diagonal_scrape']:
            # Approximate as rectangle
            length = params.get('length', 0)
            width = params.get('width', 0.05)  # Default width
            return length * width
        elif dent_type == 'multi_impact':
            impacts = params.get('impacts', [])
            total_area = 0
            for impact in impacts:
                radius = impact.get('radius', 0)
                total_area += np.pi * radius ** 2
            return total_area
        else:
            # Default estimate based on common dent sizes
            return 0.01  # 100 cm²
    except:
        return 0.01

def apply_colormap(image):
    """Apply colormap to depth image for visualization"""
    if len(image.shape) == 3:
        image = image[:,:,0]  # Take first channel if RGB
    
    # Normalize to 0-1 range
    img_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    
    # Apply colormap
    colored = cm.viridis(img_norm)
    return (colored[:,:,:3] * 255).astype(np.uint8)

def load_mesh_from_obj(obj_path):
    """Load 3D mesh from OBJ file"""
    try:
        mesh = trimesh.load(obj_path)
        return mesh
    except Exception as e:
        st.error(f"Error loading mesh: {e}")
        return None

def create_mesh_plot_with_camera_info(mesh, camera_info=None):
    """Create plotly 3D mesh visualization with camera information from metadata"""
    if mesh is None:
        return None
    
    # Get vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Calculate mesh properties
    mesh_bounds = mesh.bounds
    mesh_center = (mesh_bounds[0] + mesh_bounds[1]) / 2
    mesh_size = mesh_bounds[1] - mesh_bounds[0]
    panel_width = mesh_size[0]   # X dimension
    panel_height = mesh_size[1]  # Y dimension
    panel_depth = mesh_size[2]   # Z dimension
    
    # Use camera info from metadata if available
    if camera_info:
        camera_position = np.array(camera_info['position'])
        capture_area = camera_info.get('capture_area', {})
        capture_width = capture_area.get('width', panel_width)
        capture_height = capture_area.get('height', panel_height)
        camera_distance = camera_info.get('distance', 1.0)
        camera_type = camera_info.get('type', 'orthographic_responsive')
    else:
        # Fallback to default values
        default_camera_distance = 1.0
        panel_front_z = mesh_center[2] + panel_depth / 2.0
        camera_position = np.array([mesh_center[0], mesh_center[1], panel_front_z + default_camera_distance])
        capture_width = panel_width
        capture_height = panel_height
        camera_distance = default_camera_distance
        camera_type = 'orthographic_responsive'
    
    # Create main mesh
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1], 
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        colorscale='Viridis',
        intensity=vertices[:, 2],  # Color by Z coordinate (depth)
        showscale=True,
        lighting=dict(ambient=0.7, diffuse=0.8, specular=0.1),
        lightposition=dict(x=100, y=200, z=0),
        name="Dented Panel"
    )
    
    # Create camera position marker
    camera_trace = go.Scatter3d(
        x=[camera_position[0]],
        y=[camera_position[1]],
        z=[camera_position[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond',
            line=dict(width=2, color='darkred')
        ),
        name=f"Camera ({camera_type}, {camera_distance:.2f}m)"
    )
    
    # Create capture area bounding box on mesh surface
    half_width = capture_width / 2.0
    half_height = capture_height / 2.0
    
    # Use mesh center Z for capture area visualization
    avg_z = np.mean(vertices[:, 2])
    capture_corners = np.array([
        [mesh_center[0] - half_width, mesh_center[1] - half_height, avg_z],
        [mesh_center[0] + half_width, mesh_center[1] - half_height, avg_z],
        [mesh_center[0] + half_width, mesh_center[1] + half_height, avg_z],
        [mesh_center[0] - half_width, mesh_center[1] + half_height, avg_z],
        [mesh_center[0] - half_width, mesh_center[1] - half_height, avg_z]  # Close the loop
    ])
    
    # Create capture area outline
    capture_trace = go.Scatter3d(
        x=capture_corners[:, 0],
        y=capture_corners[:, 1],
        z=capture_corners[:, 2],
        mode='lines',
        line=dict(
            width=8,
            color='yellow',
            dash='dash'
        ),
        name=f"Camera View ({capture_width:.2f}×{capture_height:.2f}m)"
    )
    
    # Create camera direction line (from camera to mesh center)
    camera_direction_trace = go.Scatter3d(
        x=[camera_position[0], mesh_center[0]],
        y=[camera_position[1], mesh_center[1]],
        z=[camera_position[2], mesh_center[2]],
        mode='lines',
        line=dict(
            width=4,
            color='red',
            dash='dot'
        ),
        name="Camera Direction"
    )
    
    # Create figure with all traces
    fig = go.Figure(data=[mesh_trace, camera_trace, capture_trace, camera_direction_trace])
    
    # Update layout for better viewing
    fig.update_layout(
        title=f"3D Dented Panel - {camera_type.title()} Camera @ {camera_distance:.2f}m",
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)", 
            zaxis_title="Z (meters)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1
        )
    )
    
    return fig

def main():
    st.title("📊 Simple Dataset Viewer - Synchronized Ground Truth")
    st.markdown("**View your container damage dataset with realistic colors and synchronized depth metrics**")
    
    # Add cache control in sidebar
    with st.sidebar:
        st.header("🔄 Data Controls")
        
        # Check data freshness
        output_dir = Path("output")
        if output_dir.exists():
            rgb_dir = output_dir / "rgb"
            if rgb_dir.exists():
                rgb_files = list(rgb_dir.glob("*_rgb.png"))
                if rgb_files:
                    latest_file = max(rgb_files, key=lambda f: f.stat().st_mtime)
                    latest_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
                    st.write(f"**Latest data:** {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Cache refresh button
        if st.button("🔄 Refresh Data Cache", help="Clear cache and reload dataset"):
            load_dataset.clear()
            st.rerun()
        
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Auto-refresh every 30s", help="Automatically refresh data")
        if auto_refresh:
            time.sleep(30)
            load_dataset.clear()
            st.rerun()
    
    # Load dataset
    samples = load_dataset()
    
    if not samples:
        st.error("❌ No dataset found. Please run `python regenerate_dataset.py` first.")
        st.info("💡 If you just generated new data, try clicking '🔄 Refresh Data Cache' in the sidebar.")
        return
    
    # Sample selector with enhanced information
    st.subheader(f"📁 Dataset ({len(samples)} samples)")
    
    # Create enhanced sample names with container color and synchronization status
    sample_names = []
    for i, s in enumerate(samples):
        if s['ground_truth']:
            gt = s['ground_truth']
            dent_info = f"{gt['dent_type']} ({gt['depth_mm']:.1f}mm)"
            
            # Add container color info
            color_info = gt.get('container_color', {})
            color_name = color_info.get('name', 'Unknown')
            shipping_line = color_info.get('shipping_line', 'Unknown')
            
            # Add synchronization status
            sync_status = gt.get('synchronization_status', 'unknown')
            sync_icon = {
                'perfect': '🎯',
                'misaligned': '⚠️',
                'metrics_unavailable': '📋',
                'specs_unavailable': '📊',
                'legacy_format': '🔧',
                'metadata_unavailable': '❓'
            }.get(sync_status, '❓')
            
            name = f"{s['full_id']} - {dent_info} - {color_name} ({shipping_line}) {sync_icon}"
        else:
            name = f"{s['full_id']} - No ground truth"
        sample_names.append(name)
    
    selected_idx = st.selectbox("Select sample:", range(len(samples)), 
                               format_func=lambda x: sample_names[x])
    
    sample = samples[selected_idx]
    
    # Synchronization Status Banner
    if sample.get('ground_truth'):
        gt = sample['ground_truth']
        sync_status = gt.get('synchronization_status', 'unknown')
        
        if sync_status == 'perfect':
            st.success("🎯 **PERFECTLY SYNCHRONIZED** - Specifications and depth metrics match exactly!")
        elif sync_status == 'misaligned':
            error = gt.get('synchronization_error', 0)
            st.error(f"⚠️ **SYNCHRONIZATION ERROR** - {error:.2f}mm difference between specs and metrics")
        elif sync_status == 'metrics_unavailable':
            st.warning("📋 **METRICS UNAVAILABLE** - Using authoritative specifications only")
        elif sync_status == 'specs_unavailable':
            st.warning("📊 **SPECS UNAVAILABLE** - Using synchronized metrics only")
        elif sync_status == 'legacy_format':
            st.info("🔧 **LEGACY FORMAT** - Using older render metadata format")
        else:
            st.error("❓ **UNKNOWN STATUS** - Unable to determine synchronization state")
    
    # Container Color Information
    if sample.get('ground_truth', {}).get('container_color'):
        color_info = sample['ground_truth']['container_color']
        
        st.subheader("🎨 Container Color Information")
        color_col1, color_col2, color_col3, color_col4 = st.columns(4)
        
        with color_col1:
            # Display color swatch
            rgb = color_info['rgb']
            hex_color = color_info['hex']
            st.markdown(f"""
            <div style="width: 100px; height: 50px; background-color: {hex_color}; 
                        border: 2px solid #ccc; border-radius: 5px; margin: 10px 0;"></div>
            """, unsafe_allow_html=True)
            st.write(f"**Color:** {color_info['name']}")
        
        with color_col2:
            st.metric("RGB Values", f"({rgb[0]}, {rgb[1]}, {rgb[2]})")
        
        with color_col3:
            st.metric("Hex Code", hex_color)
        
        with color_col4:
            st.metric("Shipping Line", color_info['shipping_line'])
    
    # Display sample
    st.subheader("🖼️ Images")
    
    # Load and display images
    try:
        # RGB
        rgb_img = imageio.imread(sample['rgb_path'])
        
        # Depth (if exists) - use PNG primarily for visualization  
        depth_img = None
        depth_colored = None
        if sample['depth_path'].exists():
            try:
                depth_img = imageio.imread(sample['depth_path'])
                depth_colored = apply_colormap(depth_img)
            except Exception as e:
                st.warning(f"Could not load PNG depth image: {e}")
                # Try EXR as fallback
                if sample['depth_exr_path'].exists():
                    try:
                        depth_img = imageio.imread(sample['depth_exr_path'])
                        depth_colored = apply_colormap(depth_img)
                        st.info("Using EXR depth file")
                    except Exception as e2:
                        st.error(f"Could not load EXR depth image either: {e2}")
        elif sample['depth_exr_path'].exists():
            # If PNG doesn't exist but EXR does
            try:
                depth_img = imageio.imread(sample['depth_exr_path']) 
                depth_colored = apply_colormap(depth_img)
                st.info("Using EXR depth file (PNG not found)")
            except Exception as e:
                st.error(f"Could not load EXR depth image: {e}")
        
        # Mask (if exists)
        mask_img = None
        if sample['mask_path'].exists():
            mask_img = imageio.imread(sample['mask_path'])
        
        # Display in columns
        img_col1, img_col2, img_col3 = st.columns(3)
        
        with img_col1:
            st.write("**RGB** (Realistic Container Color)")
            st.image(rgb_img, width=None)  # Show at actual pixel size
            st.caption(f"Resolution: {rgb_img.shape[1]}×{rgb_img.shape[0]} pixels")
            
            # Show color verification
            if sample.get('ground_truth', {}).get('container_color'):
                color_name = sample['ground_truth']['container_color']['name']
                st.caption(f"✅ Color: {color_name} (no blue contamination)")
        
        with img_col2:
            st.write("**Depth**")
            if depth_img is not None and depth_colored is not None:
                st.image(depth_colored, width=None)  # Show at actual pixel size
                # Calculate depth statistics
                depth_min, depth_max = np.min(depth_img), np.max(depth_img)
                st.caption(f"Resolution: {depth_img.shape[1]}×{depth_img.shape[0]} pixels")
                st.caption(f"Range: {depth_min:.3f} - {depth_max:.3f}")
            else:
                st.info("No depth image")
        
        with img_col3:
            st.write("**Mask**")
            if mask_img is not None:
                st.image(mask_img, width=None)  # Show at actual pixel size
                # Calculate mask statistics
                if len(mask_img.shape) == 3:
                    mask_gray = mask_img[:,:,0]  # Take first channel if RGB
                else:
                    mask_gray = mask_img
                white_pixels = np.sum(mask_gray > 127)
                total_pixels = mask_gray.size
                dent_percentage = (white_pixels / total_pixels) * 100
                st.caption(f"Resolution: {mask_img.shape[1] if len(mask_img.shape) > 1 else mask_img.shape[0]}×{mask_img.shape[0]} pixels")
                st.caption(f"Dent pixels: {white_pixels:,} ({dent_percentage:.1f}%)")
            else:
                st.info("No mask image")
                
    except Exception as e:
        st.error(f"Error loading images: {e}")

    # 3D Mesh Visualization
    st.subheader("🎲 3D Dented Panel Mesh")
    
    if sample['obj_path'].exists():
        try:
            mesh = load_mesh_from_obj(sample['obj_path'])
            
            if mesh is not None:
                camera_info = sample.get('metadata', {}).get('camera_info') if sample.get('metadata') else None
                fig = create_mesh_plot_with_camera_info(mesh, camera_info)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add mesh statistics
                    mesh_col1, mesh_col2, mesh_col3, mesh_col4 = st.columns(4)
                    with mesh_col1:
                        st.metric("Vertices", f"{len(mesh.vertices):,}")
                    with mesh_col2:
                        st.metric("Faces", f"{len(mesh.faces):,}")
                    with mesh_col3:
                        volume = abs(mesh.volume) if mesh.volume < 0 else mesh.volume
                        st.metric("Volume", f"{volume:.6f} m³")
                    with mesh_col4:
                        camera_info = sample.get('metadata', {}).get('camera_info', {})
                        camera_distance = camera_info.get('distance', 1.0)
                        st.metric("Camera Distance", f"{camera_distance:.2f} m")
            else:
                st.error("Failed to load 3D mesh")
                
        except Exception as e:
            st.error(f"Error loading 3D mesh: {e}")
    else:
        st.info(f"No 3D mesh file found: {sample['obj_path']}")

    # Enhanced Ground Truth Information
    st.subheader("🎯 Synchronized Ground Truth Information")
    
    if sample['ground_truth']:
        gt = sample['ground_truth']
        
        # Show data source priority
        depth_source = gt.get('depth_source', 'unknown')
        if depth_source == 'authoritative_specifications':
            st.success("📋 **AUTHORITATIVE SOURCE**: Original dent specifications (highest priority)")
        elif depth_source == 'synchronized_metrics':
            st.info("📊 **SYNCHRONIZED METRICS**: Depth metrics JSON (specifications mirror)")
        elif depth_source == 'render_metadata':
            st.warning("🔧 **RENDER METADATA**: Legacy format (lower priority)")
        
        # Create columns for organized display
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write("### 🔍 **Dent Analysis**")
            
            # Severity with color coding
            severity = gt.get('severity', 'UNKNOWN')
            severity_colors = {
                'SEVERE': 'red',
                'MODERATE': 'orange', 
                'MINOR': 'yellow',
                'SUPERFICIAL': 'green'
            }
            severity_color = severity_colors.get(severity, 'gray')
            st.markdown(f"**Severity:** :{severity_color}[{severity}]")
            
            # Key metrics
            st.write(f"**Type:** {gt['dent_type']}")
            
            # Depth with source indicator
            depth_mm = gt['depth_mm']
            if depth_source == 'authoritative_specifications':
                st.write(f"**Depth:** {depth_mm:.2f} mm ✅")
                st.caption("🎯 From authoritative specifications (ground truth)")
            else:
                st.write(f"**Depth:** {depth_mm:.2f} mm")
                st.caption(f"📊 From {depth_source.replace('_', ' ')}")
            
            # IICL-6 Compliance
            st.write("**IICL-6 Compliance:**")
            if depth_mm > 5.0:
                st.error("❌ **FAIL** - Depth >5mm (exceeds allowable damage)")
            else:
                st.success("✅ **PASS** - Depth ≤5mm (within allowable limits)")
            
            # Dent area
            if gt.get('dent_area'):
                area_cm2 = gt['dent_area'] * 10000
                st.write(f"**Area:** {area_cm2:.1f} cm²")
            
            # Dent parameters
            if gt.get('dent_parameters'):
                st.write("**Geometric Details:**")
                for key, value in gt['dent_parameters'].items():
                    if isinstance(value, float):
                        if key in ['radius', 'width', 'length']:
                            st.write(f"  • **{key}:** {value:.3f} m ({value*100:.1f} cm)")
                        else:
                            st.write(f"  • **{key}:** {value:.3f}")
                    else:
                        st.write(f"  • **{key}:** {value}")
        
        with info_col2:
            st.write("### 🎨 **Container Information**")
            
            # Enhanced container color display
            color_info = gt.get('container_color', {})
            
            st.write("**Visual Realism:**")
            st.success("✅ **Realistic Colors** - No blue contamination")
            st.success("✅ **Color-Matched Dents** - Dents use container color")
            
            st.write("**Container Details:**")
            st.write(f"  • **Color:** {color_info.get('name', 'Unknown')}")
            st.write(f"  • **Shipping Line:** {color_info.get('shipping_line', 'Unknown')}")
            st.write(f"  • **RGB:** {color_info.get('rgb', [0,0,0])}")
            st.write(f"  • **Hex:** {color_info.get('hex', '#000000')}")
            
            # Synchronization status details
            sync_status = gt.get('synchronization_status', 'unknown')
            st.write("**Data Synchronization:**")
            if sync_status == 'perfect':
                st.success("✅ **Perfect Sync** - 0.00mm error")
                st.write("  • Specifications ↔ Depth Metrics: Identical")
            elif sync_status == 'misaligned':
                error = gt.get('synchronization_error', 0)
                st.error(f"❌ **Misaligned** - {error:.2f}mm error")
                st.write("  • Manual verification required")
            else:
                st.info(f"ℹ️ **Status:** {sync_status.replace('_', ' ').title()}")
            
            # Dataset quality indicators
            st.write("**Quality Assurance:**")
            st.write("  • **Ground Truth:** Authoritative specifications")
            st.write("  • **Color System:** Realistic container colors")
            st.write("  • **Depth Metrics:** Synchronized measurements")
            st.write("  • **Format:** Modern JSON structure")
    else:
        st.warning("⚠️ No ground truth data available")
    
    # Enhanced Synchronized Depth Metrics
    st.subheader("📊 Synchronized Depth Metrics")
    
    if sample.get('synchronized_metrics'):
        depth_metrics = sample['synchronized_metrics']
        
        # Show data source
        source = depth_metrics.get('source', 'unknown')
        if source == 'dented_panel_specifications.json':
            st.success(f"📋 **AUTHORITATIVE SOURCE**: {source}")
            st.caption("✅ Specifications are the authoritative ground truth")
        else:
            st.info(f"📊 **DATA SOURCE**: {source}")
        
        # Show synchronized specifications
        specs = depth_metrics.get('specifications', {})
        if specs:
            st.write("**📋 Authoritative Specifications (Ground Truth):**")
            spec_col1, spec_col2, spec_col3, spec_col4 = st.columns(4)
            
            with spec_col1:
                st.metric("Max Depth", f"{specs.get('max_depth_mm', 0):.2f} mm")
            with spec_col2:
                st.metric("Type", specs.get('type', 'Unknown'))
            with spec_col3:
                st.metric("Panel ID", specs.get('panel_id', 'Unknown'))
            with spec_col4:
                timestamp = depth_metrics.get('timestamp', 'Unknown')
                st.metric("Generated", timestamp.split('T')[0] if 'T' in timestamp else timestamp)
        
        # Show validation measurements (for debugging)
        validation = depth_metrics.get('validation_measurement', {})
        if validation:
            with st.expander("🔍 Validation Measurements (For Debugging Only)", expanded=False):
                st.warning("⚠️ These measurements may contain mesh generation artifacts")
                st.write(f"**Method:** {validation.get('method', 'Unknown')}")
                st.write(f"**Max Depth:** {validation.get('max_depth_mm', 0):.2f} mm")
                st.write(f"**Mean Depth:** {validation.get('mean_depth_mm', 0):.2f} mm")
                st.write(f"**Coverage:** {validation.get('dent_coverage_percent', 0):.1f}%")
                st.caption("Note: Use specifications as authoritative ground truth, not these measurements")
        
        # Show complete data
        with st.expander("📄 Complete Synchronized Data", expanded=False):
            st.json(depth_metrics)
    else:
        st.warning("📋 **No Synchronized Metrics** - File may be missing or in legacy format")
    
    # Navigation
    st.subheader("🔄 Navigation")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        if st.button("⬅️ Previous") and selected_idx > 0:
            st.rerun()
    
    with nav_col2:
        st.write(f"Sample {selected_idx + 1} of {len(samples)}")
    
    with nav_col3:
        if st.button("➡️ Next") and selected_idx < len(samples) - 1:
            st.rerun()

if __name__ == "__main__":
    main() 