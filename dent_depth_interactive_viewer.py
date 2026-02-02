#!/usr/bin/env python3
"""
Interactive Dent Depth Viewer
Visualizes dent segmentation masks with interactive depth information.
Hover over pixels to see depth values for each dent segment.
"""

import streamlit as st
import numpy as np
import json
from pathlib import Path
import imageio.v2 as imageio
import plotly.graph_objects as go
import cv2

st.set_page_config(
    page_title="🔍 Interactive Dent Depth Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_depth_map(depth_path: Path) -> np.ndarray:
    """Load depth map from .npy file"""
    if depth_path.suffix == '.npy':
        return np.load(depth_path)
    elif depth_path.suffix == '.png':
        depth_img = imageio.imread(depth_path)
        # Convert PNG to depth (assuming normalized 0-1 range)
        if len(depth_img.shape) == 3:
            depth_img = depth_img[:, :, 0]
        return depth_img.astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth file format: {depth_path.suffix}")

def load_dent_mask(mask_path: Path) -> np.ndarray:
    """Load dent mask from image file"""
    mask_img = imageio.imread(mask_path)
    if len(mask_img.shape) == 3:
        mask_img = mask_img[:, :, 0]  # Take first channel
    return (mask_img > 127).astype(np.uint8) * 255

def create_depth_visualization(depth_map: np.ndarray, dent_mask: np.ndarray, 
                               dent_segments: list = None) -> go.Figure:
    """
    Create interactive Plotly visualization with hover depth information
    
    Args:
        depth_map: Depth map in meters (H, W)
        dent_mask: Binary dent mask (H, W) where 255 = dent
        dent_segments: Optional list of segment dictionaries with metadata
    """

    # --- ADD THESE LINES TO HANDLE THE 3-CHANNEL FORMAT ---
    depth_map = np.asarray(depth_map)
    
    # If the map is (Channels, Height, Width), take the first channel
    if depth_map.ndim == 3:
        # If shape is (3, 320, 384), we want (320, 384)
        if depth_map.shape[0] == 3:
            depth_map = depth_map[0]
        # If shape is (320, 384, 3), we want (320, 384)
        elif depth_map.shape[2] == 3:
            depth_map = depth_map[:, :, 0]

    H, W = depth_map.shape

    # 3. --- NEW FIX: Force mask to match depth map size ---
    # This prevents the IndexError by ensuring indices i and j are always valid
    if dent_mask.shape[0] != H or dent_mask.shape[1] != W:
        print(f"DEBUG: Resizing mask from {dent_mask.shape} to ({H}, {W})")
        # Use INTER_NEAREST to keep the mask values (0 or 1) sharp
        dent_mask = cv2.resize(dent_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Normalize depth for visualization (0-1 range)
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) == 0:
        st.error("No valid depth values found!")
        return None
    
    depth_min = np.min(valid_depths)
    depth_max = np.max(valid_depths)
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-10)
    
    # Convert depth to millimeters for display
    depth_map_mm = depth_map * 1000.0
    
    # Create segment info map
    segment_info_map = {}
    if dent_segments:
        for seg in dent_segments:
            seg_id = seg.get('segment_id', 0)
            segment_info_map[seg_id] = seg
    
    # Create segment info map for quick lookup
    segment_info_map = {}
    if dent_segments:
        for seg in dent_segments:
            seg_id = seg.get('segment_id', 0)
            segment_info_map[seg_id] = seg
    
    # Create segment overlay map (which segment each pixel belongs to)
    segment_map = np.zeros((H, W), dtype=np.int32)
    if dent_segments:
        for seg in dent_segments:
            seg_id = seg.get('segment_id', 0)
            bbox = seg.get('bbox', [])
            if len(bbox) >= 4:
                seg_x, seg_y, seg_w, seg_h = bbox
                x_end = min(seg_x + seg_w, W)
                y_end = min(seg_y + seg_h, H)
                segment_map[seg_y:y_end, seg_x:x_end] = seg_id
    
    # Create custom colorscale (blue gradient: dark blue to light cyan)
    # Reverse so dark = deep, light = shallow
    colorscale = [
        [0.0, 'rgb(50, 100, 150)'],    # Dark blue (deep)
        [0.33, 'rgb(75, 125, 175)'],   # Medium-dark blue
        [0.66, 'rgb(100, 175, 225)'],  # Medium blue
        [1.0, 'rgb(150, 255, 255)']    # Light cyan (shallow)
    ]
    
    # Create hover text matrix
    hover_text = np.empty((H, W), dtype=object)
    for i in range(H):
        for j in range(W):
            depth_val = depth_map_mm[i, j]
            if depth_val > 0:
                seg_id = segment_map[i, j]
                is_dent = dent_mask[i, j] > 127
                
                hover_text[i, j] = (
                    f"<b>Pixel:</b> ({i}, {j})<br>"
                    f"<b>Depth:</b> {depth_val:.2f} mm<br>"
                    f"<b>Status:</b> {'Dent' if is_dent else 'Healthy'}"
                )
                
                if seg_id > 0 and seg_id in segment_info_map:
                    seg_info = segment_info_map[seg_id]
                    hover_text[i, j] += (
                        f"<br><b>Segment ID:</b> {seg_id}<br>"
                        f"<b>Max Depth:</b> {seg_info.get('max_depth_diff_mm', 0):.2f} mm<br>"
                        f"<b>Area:</b> {seg_info.get('area_cm2', 0):.2f} cm²<br>"
                        f"<b>Wave Location:</b> {seg_info.get('wave_location', 'N/A')}"
                    )
            else:
                hover_text[i, j] = f"<b>Pixel:</b> ({i}, {j})<br><b>Depth:</b> Invalid"
    
    # Create figure
    fig = go.Figure()
    
    # Add depth heatmap with custom hover
    fig.add_trace(go.Heatmap(
        z=depth_map_mm,
        x=list(range(W)),
        y=list(range(H)),
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title="Depth (mm)"
        ),
        zmin=depth_min * 1000,
        zmax=depth_max * 1000,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name="Depth"
    ))
    
    # Add dent mask overlay as contour
    if np.any(dent_mask > 127):
        # Create contour lines for dent regions
        mask_uint8 = (dent_mask > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) > 2:
                contour_2d = contour.reshape(-1, 2)
                fig.add_trace(go.Scatter(
                    x=contour_2d[:, 0],
                    y=contour_2d[:, 1],
                    mode='lines',
                    line=dict(color='red', width=2),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    name="Dent Region",
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Update layout
    fig.update_layout(
        title="Interactive Dent Depth Visualization (Hover to see depth values)",
        xaxis_title="X (pixels)",
        yaxis_title="Y (pixels)",
        height=700,
        hovermode='closest',
        showlegend=False
    )
    
    # Set axis to image mode
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(autorange="reversed")  # Image coordinates (top to bottom)
    
    return fig

def create_depth_difference_visualization(depth_diff: np.ndarray, dent_mask: np.ndarray, 
                                         dent_segments: list = None) -> go.Figure:
    """
    Create interactive Plotly visualization for depth difference with hover information
    
    Args:
        depth_diff: Depth difference map in meters (H, W) - difference from reference
        dent_mask: Binary dent mask (H, W) where 255 = dent
        dent_segments: Optional list of segment dictionaries with metadata
    """
    H, W = depth_diff.shape
    
    # Convert to millimeters
    depth_diff_mm = depth_diff * 1000.0
    
    # Normalize for visualization
    valid_diffs = depth_diff_mm[depth_diff_mm > 0]
    if len(valid_diffs) == 0:
        st.warning("No valid depth difference values found!")
        return None
    
    diff_min = np.min(valid_diffs)
    diff_max = np.max(valid_diffs)
    
    # Create segment info map
    segment_info_map = {}
    if dent_segments:
        for seg in dent_segments:
            seg_id = seg.get('segment_id', 0)
            segment_info_map[seg_id] = seg
    
    # Create segment overlay map
    segment_map = np.zeros((H, W), dtype=np.int32)
    if dent_segments:
        for seg in dent_segments:
            seg_id = seg.get('segment_id', 0)
            bbox = seg.get('bbox', [])
            if len(bbox) >= 4:
                seg_x, seg_y, seg_w, seg_h = bbox
                x_end = min(seg_x + seg_w, W)
                y_end = min(seg_y + seg_h, H)
                segment_map[seg_y:y_end, seg_x:x_end] = seg_id
    
    # Create colorscale for depth difference (red gradient: light = small diff, dark = large diff)
    colorscale = [
        [0.0, 'rgb(255, 255, 255)'],    # White (no difference)
        [0.2, 'rgb(200, 220, 255)'],     # Light blue (small difference)
        [0.4, 'rgb(150, 180, 255)'],    # Medium blue
        [0.6, 'rgb(100, 120, 255)'],    # Blue
        [0.8, 'rgb(50, 60, 200)'],      # Dark blue
        [1.0, 'rgb(0, 0, 150)']         # Very dark blue (large difference)
    ]
    
    # Create hover text matrix
    hover_text = np.empty((H, W), dtype=object)
    for i in range(H):
        for j in range(W):
            diff_val = depth_diff_mm[i, j]
            if diff_val > 0:
                seg_id = segment_map[i, j]
                is_dent = dent_mask[i, j] > 127
                
                hover_text[i, j] = (
                    f"<b>Pixel:</b> ({i}, {j})<br>"
                    f"<b>Depth Difference:</b> {diff_val:.2f} mm<br>"
                    f"<b>Status:</b> {'Dent' if is_dent else 'Healthy'}"
                )
                
                if seg_id > 0 and seg_id in segment_info_map:
                    seg_info = segment_info_map[seg_id]
                    hover_text[i, j] += (
                        f"<br><b>Segment ID:</b> {seg_id}<br>"
                        f"<b>Max Depth Diff:</b> {seg_info.get('max_depth_diff_mm', 0):.2f} mm<br>"
                        f"<b>Area:</b> {seg_info.get('area_cm2', 0):.2f} cm²<br>"
                        f"<b>Wave Location:</b> {seg_info.get('wave_location', 'N/A')}"
                    )
            else:
                hover_text[i, j] = f"<b>Pixel:</b> ({i}, {j})<br><b>Depth Difference:</b> 0.00 mm"
    
    # Create figure
    fig = go.Figure()
    
    # Add depth difference heatmap
    fig.add_trace(go.Heatmap(
        z=depth_diff_mm,
        x=list(range(W)),
        y=list(range(H)),
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title="Depth Difference (mm)"
        ),
        zmin=diff_min,
        zmax=diff_max,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name="Depth Difference"
    ))
    
    # Add dent mask overlay as contour
    if np.any(dent_mask > 127):
        mask_uint8 = (dent_mask > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) > 2:
                contour_2d = contour.reshape(-1, 2)
                fig.add_trace(go.Scatter(
                    x=contour_2d[:, 0],
                    y=contour_2d[:, 1],
                    mode='lines',
                    line=dict(color='red', width=2),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    name="Dent Region",
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Update layout
    fig.update_layout(
        title="Interactive Depth Difference Visualization (Hover to see difference values)",
        xaxis_title="X (pixels)",
        yaxis_title="Y (pixels)",
        height=700,
        hovermode='closest',
        showlegend=False
    )
    
    # Set axis to image mode
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(autorange="reversed")
    
    return fig

def main():
    st.title("🔍 Interactive Dent Depth Viewer")
    st.markdown("""
    **Visualize dent segmentation masks with interactive depth information.**
    - Hover over pixels to see depth values
    - Blue gradient shows depth (dark = deep, light = shallow)
    - Red overlay shows dent regions
    """)
    
    # Sidebar for file selection
    st.sidebar.header("📁 File Selection")
    
    # File upload option
    st.sidebar.markdown("### 📤 Upload Files")
    uploaded_depth_file = st.sidebar.file_uploader(
        "Drag & Drop Depth Map (.npy)",
        type=['npy'],
        help="Upload a depth map file (.npy) to visualize"
    )
    
    uploaded_mask_file = st.sidebar.file_uploader(
        "Drag & Drop Dent Mask (.png)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a dent mask image file (optional)"
    )
    
    uploaded_segments_file = st.sidebar.file_uploader(
        "Drag & Drop Segments (.json)",
        type=['json'],
        help="Upload a segments JSON file (optional)"
    )
    
    uploaded_depth_diff_file = st.sidebar.file_uploader(
        "Drag & Drop Depth Difference (.npy)",
        type=['npy'],
        help="Upload a depth difference file for comparison (optional)"
    )
    
    uploaded_original_depth_file = st.sidebar.file_uploader(
        "Drag & Drop Original Depth (.npy)",
        type=['npy'],
        help="Upload original depth map to calculate difference (optional)"
    )
    
    # Use uploaded files if available, otherwise use directory selection
    uploaded_depth_diff = None
    uploaded_original_depth = None
    
    if uploaded_depth_file is not None:
        # Use uploaded files
        try:
            import tempfile
            
            with st.spinner("Loading uploaded depth map..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
                    tmp_file.write(uploaded_depth_file.getvalue())
                    tmp_path = Path(tmp_file.name)
                    depth_map = load_depth_map(tmp_path)
                    tmp_path.unlink()  # Clean up temp file
            
            dent_mask = None
            if uploaded_mask_file is not None:
                with st.spinner("Loading uploaded dent mask..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(uploaded_mask_file.getvalue())
                        tmp_path = Path(tmp_file.name)
                        dent_mask = load_dent_mask(tmp_path)
                        tmp_path.unlink()
            else:
                # Create empty mask if none provided
                dent_mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
                st.sidebar.warning("No mask file uploaded. Using empty mask.")
            
            dent_segments = None
            if uploaded_segments_file is not None:
                with st.spinner("Loading uploaded segment data..."):
                    segments_data = json.loads(uploaded_segments_file.read())
                    if isinstance(segments_data, dict) and 'segments' in segments_data:
                        dent_segments = segments_data['segments']
                    else:
                        dent_segments = segments_data
            
            # Load depth difference file if uploaded
            if uploaded_depth_diff_file is not None:
                with st.spinner("Loading uploaded depth difference map..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
                        tmp_file.write(uploaded_depth_diff_file.getvalue())
                        tmp_path = Path(tmp_file.name)
                        uploaded_depth_diff = load_depth_map(tmp_path)
                        tmp_path.unlink()
            
            # Load original depth file if uploaded
            if uploaded_original_depth_file is not None:
                with st.spinner("Loading uploaded original depth map..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
                        tmp_file.write(uploaded_original_depth_file.getvalue())
                        tmp_path = Path(tmp_file.name)
                        uploaded_original_depth = load_depth_map(tmp_path)
                        tmp_path.unlink()
            
            # Display file info
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 File Info")
            st.sidebar.text(f"Depth: {uploaded_depth_file.name}")
            if uploaded_mask_file:
                st.sidebar.text(f"Mask: {uploaded_mask_file.name}")
            if uploaded_segments_file:
                st.sidebar.text(f"Segments: {uploaded_segments_file.name}")
                st.sidebar.text(f"Num Segments: {len(dent_segments) if dent_segments else 0}")
            if uploaded_depth_diff_file:
                st.sidebar.text(f"Depth Diff: {uploaded_depth_diff_file.name}")
            if uploaded_original_depth_file:
                st.sidebar.text(f"Original Depth: {uploaded_original_depth_file.name}")
            
        except Exception as e:
            st.error(f"Error loading uploaded files: {e}")
            import traceback
            st.code(traceback.format_exc())
            return
    else:
        # Use directory-based selection
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📂 Or Select from Directory")
        
        # Get base directory
        base_dir = st.sidebar.text_input(
            "Base Directory",
            value="output_scene",
            help="Base directory containing container scenes"
        )
        base_path = Path(base_dir)
        
        if not base_path.exists():
            st.error(f"Directory not found: {base_dir}")
            return
        
        # Find all container directories
        container_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
        if not container_dirs:
            st.error(f"No container directories found in {base_dir}")
            return
        
        container_names = [d.name for d in container_dirs]
        selected_container = st.sidebar.selectbox(
            "Select Container",
            container_names
        )
        
        container_path = base_path / selected_container
        
        # Find all shot directories
        shot_dirs = sorted([d for d in container_path.iterdir() if d.is_dir()])
        if not shot_dirs:
            st.error(f"No shot directories found in {container_path}")
            return
        
        shot_names = [d.name for d in shot_dirs]
        selected_shot = st.sidebar.selectbox(
            "Select Shot",
            shot_names
        )
        
        shot_path = container_path / selected_shot
        
        # Find files
        files = list(shot_path.glob("*"))
        
        # Find depth file
        depth_files = [f for f in files if "dented_depth" in f.name and f.suffix == ".npy"]
        if not depth_files:
            st.error(f"No depth file found in {shot_path}")
            return
        
        depth_file = depth_files[0]
        
        # Find mask file
        mask_files = [f for f in files if "dent_mask" in f.name and f.suffix == ".png"]
        if not mask_files:
            st.error(f"No mask file found in {shot_path}")
            return
        
        mask_file = mask_files[0]
        
        # Find segments JSON
        segments_file = None
        segments_files = [f for f in files if "dent_segments" in f.name and f.suffix == ".json"]
        if segments_files:
            segments_file = segments_files[0]
        
        # Load data
        try:
            with st.spinner("Loading depth map..."):
                depth_map = load_depth_map(depth_file)
            
            with st.spinner("Loading dent mask..."):
                dent_mask = load_dent_mask(mask_file)
            
            dent_segments = None
            if segments_file:
                with st.spinner("Loading segment data..."):
                    with open(segments_file, 'r') as f:
                        dent_segments = json.load(f)
                        if isinstance(dent_segments, dict) and 'segments' in dent_segments:
                            dent_segments = dent_segments['segments']
        except Exception as e:
            st.error(f"Error loading files: {e}")
            import traceback
            st.code(traceback.format_exc())
            return
        
        # Display file info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 File Info")
        st.sidebar.text(f"Depth: {depth_file.name}")
        st.sidebar.text(f"Mask: {mask_file.name}")
        if segments_file:
            st.sidebar.text(f"Segments: {segments_file.name}")
            st.sidebar.text(f"Num Segments: {len(dent_segments) if dent_segments else 0}")
    
    # Check dimensions match
    if depth_map.shape[:2] != dent_mask.shape[:2]:
        st.warning(f"Dimension mismatch: Depth {depth_map.shape} vs Mask {dent_mask.shape}")
        # Resize mask to match depth
        dent_mask = cv2.resize(dent_mask, (depth_map.shape[1], depth_map.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Statistics
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) > 0:
        st.sidebar.markdown("### 📈 Statistics")
        st.sidebar.metric("Min Depth", f"{np.min(valid_depths)*1000:.2f} mm")
        st.sidebar.metric("Max Depth", f"{np.max(valid_depths)*1000:.2f} mm")
        st.sidebar.metric("Mean Depth", f"{np.mean(valid_depths)*1000:.2f} mm")
        
        dent_pixels = np.sum(dent_mask > 127)
        total_pixels = dent_mask.size
        dent_percentage = (dent_pixels / total_pixels) * 100
        st.sidebar.metric("Dent Coverage", f"{dent_percentage:.1f}%")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["📊 Depth Map", "📉 Depth Difference"])
    
    with tab1:
        st.markdown("### 🖼️ Interactive Depth Visualization")
        st.markdown("**Hover over the image to see depth values for each pixel**")
        
        fig = create_depth_visualization(depth_map, dent_mask, dent_segments)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📉 Interactive Depth Difference Visualization")
        st.markdown("**Hover over the image to see depth difference values for each pixel**")
        
        depth_diff = None
        
        # Check if we have uploaded depth difference or original depth
        if uploaded_depth_file is not None:
            # Using uploaded files
            if uploaded_depth_diff is not None:
                # Use uploaded depth difference file
                depth_diff = uploaded_depth_diff
            elif uploaded_original_depth is not None:
                # Calculate from uploaded original depth
                st.info("Calculating depth difference from uploaded original and dented depth maps...")
                try:
                    if uploaded_original_depth.shape == depth_map.shape:
                        depth_diff = np.abs(depth_map - uploaded_original_depth)
                    else:
                        st.error(f"Shape mismatch: Original {uploaded_original_depth.shape} vs Dented {depth_map.shape}")
                except Exception as e:
                    st.error(f"Error calculating depth difference: {e}")
            else:
                st.info("Upload a depth difference file (.npy) or original depth file (.npy) to visualize depth differences.")
        else:
            # Using directory-based files
            files = list(shot_path.glob("*"))
            
            # Find depth difference file
            depth_diff_files = [f for f in files if "depth_diff" in f.name and f.suffix == ".npy"]
            if not depth_diff_files:
                st.warning("No depth difference file found. Looking for files with 'depth_diff' in name.")
                # Try to calculate from original and dented depth
                original_depth_files = [f for f in files if "original_depth" in f.name and f.suffix == ".npy"]
                if original_depth_files:
                    st.info("Attempting to calculate depth difference from original and dented depth maps...")
                    try:
                        original_depth = load_depth_map(original_depth_files[0])
                        if original_depth.shape == depth_map.shape:
                            depth_diff = np.abs(depth_map - original_depth)
                        else:
                            st.error(f"Shape mismatch: Original {original_depth.shape} vs Dented {depth_map.shape}")
                    except Exception as e:
                        st.error(f"Error calculating depth difference: {e}")
                else:
                    st.error("Cannot visualize depth difference: No depth_diff.npy or original_depth.npy file found")
            else:
                depth_diff_file = depth_diff_files[0]
                try:
                    with st.spinner("Loading depth difference map..."):
                        depth_diff = load_depth_map(depth_diff_file)
                except Exception as e:
                    st.error(f"Error loading depth difference: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Visualize depth difference if available
        if depth_diff is not None:
            # Check dimensions match
            if depth_diff.shape[:2] != dent_mask.shape[:2]:
                st.warning(f"Dimension mismatch: Depth Diff {depth_diff.shape} vs Mask {dent_mask.shape}")
                depth_diff = cv2.resize(depth_diff, (dent_mask.shape[1], dent_mask.shape[0]), 
                                      interpolation=cv2.INTER_LINEAR)
            
            fig_diff = create_depth_difference_visualization(
                depth_diff, dent_mask, dent_segments
            )
            if fig_diff:
                st.plotly_chart(fig_diff, use_container_width=True)
    
    # Display segment information if available
    if dent_segments:
        st.markdown("---")
        st.markdown("### 📋 Segment Information")
        
        for seg in dent_segments:
            with st.expander(f"Segment {seg.get('segment_id', 'N/A')} - {seg.get('area_cm2', 0):.2f} cm²"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Max Depth", f"{seg.get('max_depth_diff_mm', 0):.2f} mm")
                    st.metric("Area", f"{seg.get('area_cm2', 0):.2f} cm²")
                
                with col2:
                    st.metric("Width", f"{seg.get('width_cm', 0):.2f} cm")
                    st.metric("Length", f"{seg.get('length_cm', 0):.2f} cm")
                
                with col3:
                    st.metric("Wave Location", seg.get('wave_location', 'N/A'))
                    st.metric("Direction", seg.get('direction', 'N/A'))

if __name__ == "__main__":
    main()
