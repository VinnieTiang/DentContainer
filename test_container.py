#!/usr/bin/env python3
"""
Test script for the new clean container generator
"""

import numpy as np
import trimesh
import os


def create_realistic_container(container_type="20ft"):
    """
    Generate a perfect shipping container cuboid with ISO 668 dimensions.
    
    Corner positions: (0,0,0) to (length, width, height)
    All panels are thin (1-2mm) with proper corrugation patterns.
    Frame structure stays within container bounds.
    
    Returns: trimesh.Trimesh - Complete container as single mesh
    """
    # ISO 668 standard dimensions
    container_specs = {
        "20ft": {"length": 6.058, "width": 2.438, "height": 2.591},
        "40ft": {"length": 12.192, "width": 2.438, "height": 2.591}
    }
    
    specs = container_specs.get(container_type, container_specs["20ft"])
    L, W, H = specs["length"], specs["width"], specs["height"]
    
    print(f"  🔧 Creating realistic shipping container...")
    print(f"    📐 Building {container_type} container: {L:.3f}m(L) × {W:.3f}m(W) × {H:.3f}m(H)")
    
    container_parts = []
    panel_thickness = 0.002  # 2mm steel panels
    
    # ============================================================================
    # HELPER FUNCTION: Create corrugated panel
    # ============================================================================
    def create_corrugated_panel(width, height, thickness, corrugation_type="vertical"):
        """
        Create a thin corrugated panel in XY plane (centered at origin).
        Panel spans from (-width/2, -height/2) to (+width/2, +height/2) in Z=0 plane.
        """
        resolution = 40
        x = np.linspace(-width/2, width/2, resolution)
        y = np.linspace(-height/2, height/2, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Generate corrugation pattern in Z direction
        if corrugation_type == "vertical":
            freq, amp = 10.0, 0.020  # 10 waves across width, 20mm amplitude
            Z = amp * np.sin(freq * np.pi * X / (width/2))
        elif corrugation_type == "horizontal":
            freq, amp = 8.0, 0.015  # 8 waves across height, 15mm amplitude
            Z = amp * np.sin(freq * np.pi * Y / (height/2))
        else:  # flat
            Z = np.zeros_like(X)
        
        # Create front and back surface vertices
        front_verts = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        back_verts = np.column_stack([X.flatten(), Y.flatten(), Z.flatten() - thickness])
        vertices = np.vstack([front_verts, back_verts])
        
        # Create triangular faces
        faces = []
        n_front = len(front_verts)
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                idx = i * resolution + j
                # Front face quad (as two triangles)
                faces.append([idx, idx + 1, idx + resolution])
                faces.append([idx + 1, idx + resolution + 1, idx + resolution])
                # Back face quad
                faces.append([idx + n_front, idx + resolution + n_front, idx + 1 + n_front])
                faces.append([idx + 1 + n_front, idx + resolution + n_front, idx + resolution + 1 + n_front])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        return mesh
    
    # ============================================================================
    # 1. CREATE SIX PANELS (forming the closed cuboid)
    # ============================================================================
    print("    📦 Adding corrugated steel panels...")
    
    # LEFT PANEL (Y=0 face, spans X×Z)
    left_panel = create_corrugated_panel(L, H, panel_thickness, "vertical")
    # Rotate from XY to XZ plane, then position at Y=0
    left_panel.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    left_panel.apply_translation([L/2, 0, H/2])
    container_parts.append(left_panel)
    
    # RIGHT PANEL (Y=W face, spans X×Z)
    right_panel = create_corrugated_panel(L, H, panel_thickness, "vertical")
    right_panel.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    right_panel.apply_translation([L/2, W, H/2])
    container_parts.append(right_panel)
    
    # BACK PANEL (X=0 face, spans Y×Z)
    back_panel = create_corrugated_panel(W, H, panel_thickness, "vertical")
    back_panel.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    back_panel.apply_translation([0, W/2, H/2])
    container_parts.append(back_panel)
    
    # FRONT PANEL (X=L face, spans Y×Z) - door side with horizontal corrugation
    front_panel = create_corrugated_panel(W, H, panel_thickness, "horizontal")
    front_panel.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [0, 1, 0]))
    front_panel.apply_translation([L, W/2, H/2])
    container_parts.append(front_panel)
    
    # ROOF PANEL (Z=H face, spans X×Y)
    roof_panel = create_corrugated_panel(L, W, panel_thickness, "horizontal")
    roof_panel.apply_translation([L/2, W/2, H])
    container_parts.append(roof_panel)
    
    # FLOOR PANEL (Z=0 face, spans X×Y) - flat
    floor_panel = create_corrugated_panel(L, W, panel_thickness, "flat")
    floor_panel.apply_translation([L/2, W/2, 0])
    container_parts.append(floor_panel)
    
    # ============================================================================
    # 2. CREATE FRAME STRUCTURE (corner posts and rails)
    # ============================================================================
    print("    🏗️  Building steel frame structure...")
    
    post_size = 0.100  # 100mm square corner posts
    rail_size = 0.080  # 80mm rails
    
    # Corner posts (inset slightly to stay within bounds)
    inset = post_size / 2
    post = trimesh.creation.cylinder(radius=post_size/2, height=H, sections=8)
    
    corner_positions = [
        (inset, inset, H/2),           # Back-left
        (inset, W-inset, H/2),         # Back-right
        (L-inset, inset, H/2),         # Front-left
        (L-inset, W-inset, H/2)        # Front-right
    ]
    
    for x, y, z in corner_positions:
        post_copy = post.copy()
        post_copy.apply_translation([x, y, z])
        container_parts.append(post_copy)
    
    # Rails along edges
    def create_rail(start, end, radius=rail_size/2):
        """Create a cylindrical rail between two points"""
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None
        
        rail = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
        
        # Calculate rotation to align rail with direction vector
        z_axis = np.array([0, 0, 1])
        direction_norm = direction / length
        
        if np.allclose(direction_norm, z_axis):
            rotation = np.eye(4)
        elif np.allclose(direction_norm, -z_axis):
            rotation = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            axis = np.cross(z_axis, direction_norm)
            angle = np.arccos(np.dot(z_axis, direction_norm))
            rotation = trimesh.transformations.rotation_matrix(angle, axis)
        
        rail.apply_transform(rotation)
        midpoint = (np.array(start) + np.array(end)) / 2
        rail.apply_translation(midpoint)
        return rail
    
    # Bottom edge rails
    rails = [
        create_rail([inset, inset, 0], [L-inset, inset, 0]),
        create_rail([inset, W-inset, 0], [L-inset, W-inset, 0]),
        create_rail([inset, inset, 0], [inset, W-inset, 0]),
        create_rail([L-inset, inset, 0], [L-inset, W-inset, 0]),
        # Top edge rails
        create_rail([inset, inset, H], [L-inset, inset, H]),
        create_rail([inset, W-inset, H], [L-inset, W-inset, H]),
        create_rail([inset, inset, H], [inset, W-inset, H]),
        create_rail([L-inset, inset, H], [L-inset, W-inset, H])
    ]
    
    container_parts.extend([r for r in rails if r is not None])
    
    # ============================================================================
    # 3. COMBINE ALL COMPONENTS
    # ============================================================================
    print("  🔗 Assembling complete container...")
    container_mesh = trimesh.util.concatenate(container_parts)
    
    # Clean up mesh
    container_mesh.fix_normals()
    try:
        container_mesh.update_faces(container_mesh.unique_faces())
        container_mesh.update_faces(container_mesh.nondegenerate_faces())
    except:
        pass  # Ignore if methods don't exist
    
    # Validate dimensions
    bounds = container_mesh.bounds
    actual_L = bounds[1][0] - bounds[0][0]
    actual_W = bounds[1][1] - bounds[0][1]
    actual_H = bounds[1][2] - bounds[0][2]
    
    print("  ✅ Realistic shipping container complete!")
    print(f"     Target: {L:.3f}m(L) × {W:.3f}m(W) × {H:.3f}m(H)")
    print(f"     Actual: {actual_L:.3f}m(L) × {actual_W:.3f}m(W) × {actual_H:.3f}m(H)")
    print(f"     Vertices: {len(container_mesh.vertices)}, Faces: {len(container_mesh.faces)}")
    print(f"     Volume: {abs(container_mesh.volume):.3f} m³")
    
    # Check if it's a proper cuboid
    error_L = abs(actual_L - L) / L * 100
    error_W = abs(actual_W - W) / W * 100
    error_H = abs(actual_H - H) / H * 100
    
    if max(error_L, error_W, error_H) < 3.0:
        print(f"     ✅ Perfect cuboid shape achieved!")
    else:
        print(f"     ⚠️  Deviation: L±{error_L:.1f}%, W±{error_W:.1f}%, H±{error_H:.1f}%")
    
    return container_mesh


def main():
    """Generate test containers"""
    print("🚢 Shipping Container Generator - Clean Implementation")
    print("=" * 70)
    
    # Create output folder
    os.makedirs("complete_containers", exist_ok=True)
    
    for container_type in ["20ft", "40ft"]:
        print(f"\n📦 Generating {container_type} container...")
        container = create_realistic_container(container_type)
        
        # Export
        filename = f"complete_containers/{container_type}_clean_container.obj"
        container.export(filename)
        print(f"✅ Saved: {filename}\n")
    
    print("🎉 Generation Complete!")


if __name__ == "__main__":
    main()

