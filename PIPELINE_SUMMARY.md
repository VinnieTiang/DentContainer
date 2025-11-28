# Full Container Generation Pipeline Summary

## Overview

This pipeline generates complete shipping containers, adds realistic dents, and produces depth maps with binary masks for dent detection.

---

## Step-by-Step Process

### **Step 1: Generate Container Sides**

- **Purpose**: Create individual panel components (floor, walls, roof, doors)
- **Input**: Container specifications (20ft, 40ft, 40ft_hc)
- **Output**: Individual OBJ files for each panel component
- **Location**: `container_sides/` folder
- **Details**:
  - Panels include corrugation patterns
  - Follows ISO 668:1995 standards
  - Includes structural details (frames, hinges, etc.)

### **Step 2: Assemble Complete Containers**

- **Purpose**: Combine all panel components into a complete 3D container mesh
- **Input**: Panel OBJ files from Step 1
- **Output**: Complete container OBJ files (e.g., `container_20ft_0001.obj`)
- **Location**: `complete_containers/` folder
- **Details**:
  - Combines side walls, back wall, roof, floor, doors
  - Adds corner castings, rails, cross members
  - Includes identification plates, forklift pockets, tie-downs
  - Applies realistic container colors (blue, red, green, etc.)

### **Step 3: Add Dents to Complete Containers**

- **Purpose**: Apply realistic damage/dents to the complete container mesh
- **Input**: Original container OBJ files from Step 2
- **Output**: Dented container OBJ files (e.g., `container_20ft_0001_dented.obj`)
- **Location**: `complete_containers_dented/` folder
- **Details**:
  - Uses physics-based deformation (pushes inward, never creates spikes)
  - Multiple dent types: circular impacts, diagonal scrapes, irregular collisions
  - Automatically excludes floor panel from dents
  - Size range: 8-50cm, Depth range: 8-60mm
  - Saves dent metadata (position, size, depth) to JSON

### **Step 4: Generate Camera Poses**

- **Purpose**: Define multiple camera viewpoints for rendering
- **Input**: Container type and configuration
- **Output**: List of camera pose dictionaries (eye, at, up vectors)
- **Details**:
  - Multiple shot types: internal door views, back wall, side walls, roof, corners
  - Camera positioned at realistic inspection distances (1.0-2.0m)
  - Fixed camera height: 1.5m (internal shots)
  - Multiple shots per panel type (e.g., 5 roof shots, 5 side wall shots)

### **Step 5: Render Depth Maps**

- **Purpose**: Generate depth maps from both original and dented containers
- **Input**:
  - Original container mesh (Step 2)
  - Dented container mesh (Step 3)
  - Camera poses (Step 4)
- **Output**:
  - Original depth maps: `*_original_depth.npy`
  - Dented depth maps: `*_dented_depth.npy`
- **Location**: `output_scene/container_*/shot_name/` folders
- **Details**:
  - Uses PyRender for offscreen rendering
  - Image size: 512×512 pixels
  - Camera FOV: 75 degrees
  - Depth values in meters

### **Step 6: Extract Panel Regions (RANSAC Plane Fitting)**

- **Purpose**: Identify main panel surfaces, excluding structural elements (frames, corners, etc.)
- **Input**: Original depth map (smoothed)
- **Output**: Panel mask (binary: 1 = panel region, 0 = excluded)
- **Details**:
  - Applies Gaussian smoothing to flatten corrugation pattern
  - Uses RANSAC to fit plane to main panel surface
  - Adaptive threshold based on corrugation depth
  - Creates mask to filter out non-panel regions
  - Saves panel mask: `*_panel_mask.png` and `*_panel_mask.npy`

### **Step 7: Compare Depths and Generate Dent Mask**

- **Purpose**: Identify dented areas by comparing original vs dented depth maps
- **Input**:
  - Masked original depth map (from Step 6)
  - Masked dented depth map (from Step 6)
  - Depth difference threshold (default: 0.01m = 10mm)
- **Output**:
  - Depth difference map: `*_depth_diff.npy`
  - Binary dent mask: `*_dent_mask.png` (WHITE=255 = dented, BLACK=0 = normal)
- **Details**:
  - Pixel-by-pixel depth comparison
  - Applies threshold to identify significant depth differences
  - Filters outliers using robust statistics
  - Saves both to `output_scene/` and `output_scene_dataset/` folders

### **Step 8: Generate Additional Outputs**

- **Purpose**: Create visualization and training data
- **Outputs**:
  - **RGB Images**: `*_original_rgb.png`, `*_dented_rgb.png`
  - **Normalized Depth Images**: `*_original_depth.png`, `*_dented_depth.png`, `*_depth_diff.png`
  - **Point Clouds**: `*_original_pointcloud.ply`, `*_dented_pointcloud.ply`
  - **Debug Visualizations**: `*_debug_*.png` (smoothed depth, panel mask, etc.)
  - **Statistics**: JSON files with dent metrics (pixel count, max depth, etc.)

---

## Final Output Structure

```
output_scene/
└── container_20ft_0001/
    ├── internal_door_right/
    │   ├── 20ft_0001_original_rgb.png
    │   ├── 20ft_0001_dented_rgb.png
    │   ├── 20ft_0001_original_depth.npy
    │   ├── 20ft_0001_dented_depth.npy
    │   ├── 20ft_0001_depth_diff.npy
    │   ├── 20ft_0001_panel_mask.png
    │   ├── 20ft_0001_dent_mask.png
    │   └── ...
    ├── internal_back_wall/
    │   └── ...
    └── ...

output_scene_dataset/
├── 20ft_0001_internal_door_right_dented_depth.npy
├── 20ft_0001_internal_door_right_dent_mask.png
└── ...
```

---

## Key Features

1. **Realistic Container Generation**: Follows ISO standards with accurate dimensions
2. **Physics-Based Dents**: Realistic deformation that pushes inward
3. **Multi-View Rendering**: Multiple camera angles per container
4. **Panel Extraction**: RANSAC-based filtering to focus on panel regions
5. **Depth-Based Detection**: Uses depth comparison for accurate dent identification
6. **Training-Ready Output**: Organized dataset format for machine learning
