#!/usr/bin/env python3
"""
Container Sides Generator
Generates the 5 remaining panels for a complete shipping container:
- Front panel (door panel with horizontal corrugations)
- Left side panel (long corrugated wall)
- Right side panel (long corrugated wall)  
- Top panel (roof with drainage ridges)
- Bottom panel (floor)
"""

import os
import json
import random
import sys
from pathlib import Path
from panel_generator import CorrugatedPanelGenerator, CorrugationPattern

def create_output_folder():
    """Create output folder for container sides"""
    folder = 'container_sides'
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")
    return folder

def generate_front_door_panel(container_type="20ft"):
    """Generate front door panel with horizontal corrugations"""
    print(f"\n=== Generating {container_type} Front Door Panel ===")
    
    generator = CorrugatedPanelGenerator()
    generator.set_container_type(container_type)
    
    # Set door-specific pattern
    generator.corrugation_pattern = CorrugationPattern.DOOR_HORIZONTAL
    generator._set_pattern_parameters()
    
    # Randomize color
    generator.randomize_color()
    
    # Create door panel with same dimensions as back panel
    panel_mesh = generator.create_corrugated_panel()
    
    # Save
    filename = f"container_sides/{container_type}_front_door_panel.obj"
    success = generator.export_to_obj(panel_mesh, filename)
    
    if success:
        specs = generator.get_panel_specs()
        specs['filename'] = filename
        specs['panel_type'] = 'front_door'
        print(f"  ✓ {container_type} Front Door Panel saved: {filename}")
        return specs
    return None

def generate_side_panel(container_type="20ft", side="left"):
    """Generate left or right side panel (long corrugated walls)"""
    print(f"\n=== Generating {container_type} {side.title()} Side Panel ===")
    
    generator = CorrugatedPanelGenerator()
    
    # Set side panel dimensions (much longer than back panel)
    if container_type == "20ft":
        panel_length = 5.90   # 20ft container internal length (approx 6m)
        panel_height = 2.390  # Same height as back panel
    elif container_type == "40ft":
        panel_length = 12.03  # 40ft container internal length (approx 12m)
        panel_height = 2.393  # Same height as back panel
    else:
        panel_length = 5.90   # Default to 20ft
        panel_height = 2.390
    
    generator.panel_width = panel_length  # Side panel width is the container length
    generator.panel_height = panel_height
    generator.container_type = container_type
    
    # Use standard vertical corrugations for side walls
    generator.corrugation_pattern = CorrugationPattern.STANDARD_VERTICAL
    generator._set_pattern_parameters()
    
    # Randomize color
    generator.randomize_color()
    
    # Create side panel
    panel_mesh = generator.create_corrugated_panel()
    
    # Save
    filename = f"container_sides/{container_type}_{side}_side_panel.obj"
    success = generator.export_to_obj(panel_mesh, filename)
    
    if success:
        specs = generator.get_panel_specs()
        specs['filename'] = filename
        specs['panel_type'] = f'{side}_side'
        specs['actual_panel_length'] = panel_length  # Store the actual length used
        print(f"  ✓ {container_type} {side.title()} Side Panel saved: {filename}")
        print(f"    Dimensions: {panel_length:.3f}m × {panel_height:.3f}m")
        return specs
    return None

def generate_roof_panel(container_type="20ft"):
    """Generate roof panel with drainage corrugations"""
    print(f"\n=== Generating {container_type} Roof Panel ===")
    
    generator = CorrugatedPanelGenerator()
    
    # Set roof panel dimensions
    if container_type == "20ft":
        panel_length = 5.90   # 20ft container length
        panel_width = 2.352   # Same width as back panel
    elif container_type == "40ft":
        panel_length = 12.03  # 40ft container length
        panel_width = 2.352   # Same width as back panel
    else:
        panel_length = 5.90   # Default to 20ft
        panel_width = 2.352
    
    generator.panel_width = panel_length  # Roof width is the container length
    generator.panel_height = panel_width  # Roof height is the container width
    generator.container_type = container_type
    
    # Use roof pattern with drainage ridges
    generator.corrugation_pattern = CorrugationPattern.ROOF_PATTERN
    generator._set_pattern_parameters()
    
    # Randomize color
    generator.randomize_color()
    
    # Create roof panel
    panel_mesh = generator.create_corrugated_panel()
    
    # Save
    filename = f"container_sides/{container_type}_roof_panel.obj"
    success = generator.export_to_obj(panel_mesh, filename)
    
    if success:
        specs = generator.get_panel_specs()
        specs['filename'] = filename
        specs['panel_type'] = 'roof'
        specs['actual_panel_length'] = panel_length
        print(f"  ✓ {container_type} Roof Panel saved: {filename}")
        print(f"    Dimensions: {panel_length:.3f}m × {panel_width:.3f}m")
        return specs
    return None

def generate_floor_panel(container_type="20ft"):
    """Generate floor panel (typically flat or minimal corrugation)"""
    print(f"\n=== Generating {container_type} Floor Panel ===")
    
    generator = CorrugatedPanelGenerator()
    
    # Set floor panel dimensions (same as roof)
    if container_type == "20ft":
        panel_length = 5.90   # 20ft container length
        panel_width = 2.352   # Same width as back panel
    elif container_type == "40ft":
        panel_length = 12.03  # 40ft container length
        panel_width = 2.352   # Same width as back panel
    else:
        panel_length = 5.90   # Default to 20ft
        panel_width = 2.352
    
    generator.panel_width = panel_length  # Floor width is the container length
    generator.panel_height = panel_width  # Floor height is the container width
    generator.container_type = container_type
    
    # Use minimal corrugation for floor (mini wave for texture but shallow)
    generator.corrugation_pattern = CorrugationPattern.MINI_WAVE
    generator._set_pattern_parameters()
    
    # Override with very shallow corrugations for floor
    generator.corrugation_depth = 0.005  # 5mm - very shallow for floor
    generator.corrugation_frequency = 8.0  # Higher frequency for floor texture
    
    # Randomize color
    generator.randomize_color()
    
    # Create floor panel
    panel_mesh = generator.create_corrugated_panel()
    
    # Save
    filename = f"container_sides/{container_type}_floor_panel.obj"
    success = generator.export_to_obj(panel_mesh, filename)
    
    if success:
        specs = generator.get_panel_specs()
        specs['filename'] = filename
        specs['panel_type'] = 'floor'
        specs['actual_panel_length'] = panel_length
        print(f"  ✓ {container_type} Floor Panel saved: {filename}")
        print(f"    Dimensions: {panel_length:.3f}m × {panel_width:.3f}m")
        return specs
    return None

def generate_complete_container_sides(container_type="20ft"):
    """Generate all 5 remaining container sides"""
    print(f"\n🚢 Generating Complete {container_type} Container Sides")
    print("=" * 60)
    
    create_output_folder()
    all_specs = []
    
    # Generate all 5 panels
    panels = [
        ("Front Door", lambda: generate_front_door_panel(container_type)),
        ("Left Side", lambda: generate_side_panel(container_type, "left")),
        ("Right Side", lambda: generate_side_panel(container_type, "right")),
        ("Roof", lambda: generate_roof_panel(container_type)),
        ("Floor", lambda: generate_floor_panel(container_type))
    ]
    
    for panel_name, generate_func in panels:
        try:
            specs = generate_func()
            if specs:
                all_specs.append(specs)
                print(f"  ✅ {panel_name} panel generated successfully")
            else:
                print(f"  ❌ Failed to generate {panel_name} panel")
        except Exception as e:
            print(f"  ❌ Error generating {panel_name} panel: {e}")
    
    # Save specifications
    if all_specs:
        specs_file = f"container_sides/{container_type}_container_sides_specifications.json"
        with open(specs_file, 'w') as f:
            json.dump(all_specs, f, indent=2)
        print(f"\n📄 Specifications saved: {specs_file}")
    
    print(f"\n✅ Container sides generation complete!")
    print(f"Generated {len(all_specs)}/5 panels for {container_type} container")
    
    return all_specs

def generate_all_container_types():
    """Generate container sides for both 20ft and 40ft containers non-interactively"""
    print("🚢 Generating Complete Container Sides for Both Container Types")
    print("=" * 70)
    
    # Generate both 20ft and 40ft containers
    for container_type in ["20ft", "40ft"]:
        print(f"\n{'='*20} {container_type} Container {'='*20}")
        generate_complete_container_sides(container_type)
        print(f"{'='*50}")

def main():
    """Main function"""
    print("🚢 Shipping Container Sides Generator")
    print("=====================================")
    print()
    print("This will generate the 5 remaining panels for a complete shipping container:")
    print("1. Front Door Panel (horizontal corrugations)")
    print("2. Left Side Panel (long vertical corrugations)")  
    print("3. Right Side Panel (long vertical corrugations)")
    print("4. Roof Panel (drainage pattern corrugations)")
    print("5. Floor Panel (minimal texture corrugations)")
    print()
    
    # Check if running with argument for non-interactive mode
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        # Ask user for container type
        print("Select container type:")
        print("1. 20ft container")
        print("2. 40ft container")
        print("3. Both (20ft and 40ft)")
        
        choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        generate_complete_container_sides("20ft")
    elif choice == "2":
        generate_complete_container_sides("40ft")
    elif choice == "3":
        generate_all_container_types()
    else:
        print("Invalid choice. Generating both container types by default.")
        generate_all_container_types()

if __name__ == "__main__":
    main() 