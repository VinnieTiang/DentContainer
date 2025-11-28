#!/usr/bin/env python3
"""
Main Application
Orchestrates the generation of corrugated shipping container back panels and dents.

This application generates realistic shipping container back panels (end panels) with:
- Standard 20ft and 40ft container dimensions based on ISO 668:1995
- Various corrugation patterns (vertical, roof, deep wave, etc.)
- Realistic dent types from industrial damage scenarios
- Fixed camera distance of 2.35m for consistent inspection setup
"""

import os
# Fix OpenMP library initialization error (multiple OpenMP runtimes)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import random
import shutil
from pathlib import Path
from panel_generator import CorrugatedPanelGenerator
from dent_generator import DentGenerator
total_panels = 500

def cleanup_panel_outputs():
    """Clean up all panel generation outputs before creating new ones"""
    print("\n🧹 Cleaning up previous panel outputs...")
    
    # Directories to clean
    panel_dir = Path("panel")
    
    files_removed = 0
    dirs_removed = 0
    
    # Remove panel directory and all contents
    if panel_dir.exists():
        try:
            shutil.rmtree(panel_dir)
            print(f"  ✓ Removed directory: {panel_dir}")
            dirs_removed += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {panel_dir}: {e}")
    
    # Remove any stray .obj files in root directory
    stray_files = [
        'corrugated_container_panel.obj',
        'shipping_container_20ft.obj',
        'shipping_container.obj'
    ]
    
    for filename in stray_files:
        if Path(filename).exists():
            try:
                os.remove(filename)
                print(f"  ✓ Removed: {filename}")
                files_removed += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {filename}: {e}")
    
    print(f"  📊 Panel cleanup: {files_removed} files, {dirs_removed} directories removed")
    return files_removed + dirs_removed > 0

def cleanup_dent_outputs():
    """Clean up all dent generation outputs before creating new ones"""
    print("\n🧹 Cleaning up previous dent outputs...")
    
    # Directories to clean
    panel_dents_dir = Path("panel_dents")
    temp_undented_dir = Path("temp_undented")
    
    files_removed = 0
    dirs_removed = 0
    
    # Remove panel_dents directory and all contents
    if panel_dents_dir.exists():
        try:
            shutil.rmtree(panel_dents_dir)
            print(f"  ✓ Removed directory: {panel_dents_dir}")
            dirs_removed += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {panel_dents_dir}: {e}")
    
    # Remove temp_undented directory (from previous renders)
    if temp_undented_dir.exists():
        try:
            shutil.rmtree(temp_undented_dir)
            print(f"  ✓ Removed directory: {temp_undented_dir}")
            dirs_removed += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {temp_undented_dir}: {e}")
    
    print(f"  📊 Dent cleanup: {files_removed} files, {dirs_removed} directories removed")
    return files_removed + dirs_removed > 0

def cleanup_render_outputs():
    """Clean up all rendering outputs before creating new ones"""
    print("\n🧹 Cleaning up previous rendering outputs...")
    
    # Directories to clean
    output_dir = Path("output")
    temp_undented_dir = Path("temp_undented")  # Additional cleanup
    
    files_removed = 0
    dirs_removed = 0
    
    # Remove output directory and all contents
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
            print(f"  ✓ Removed directory: {output_dir}")
            dirs_removed += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {output_dir}: {e}")
    
    # Remove temp directories that might be left over
    if temp_undented_dir.exists():
        try:
            shutil.rmtree(temp_undented_dir)
            print(f"  ✓ Removed directory: {temp_undented_dir}")
            dirs_removed += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {temp_undented_dir}: {e}")
    
    # Remove any stray depth metrics files in root
    root_files = Path(".").glob("*_depth_metrics.json")
    for file_path in root_files:
        try:
            os.remove(file_path)
            print(f"  ✓ Removed: {file_path}")
            files_removed += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {file_path}: {e}")
    
    print(f"  📊 Render cleanup: {files_removed} files, {dirs_removed} directories removed")
    return files_removed + dirs_removed > 0

def create_output_folders():
    """Create output folders for panels"""
    folders = ['panel', 'panel_dents']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")
    return folders

def cleanup_unwanted_files():
    """Clean up unwanted files from the workspace"""
    unwanted_files = [
        'shipping_container_20ft.obj',
        'shipping_container.obj', 
        'generate_shipping_container.py',
        'corrugated_container_panel.obj'  # This might be from previous runs
    ]
    
    print("\n=== Cleaning up unwanted files ===")
    
    for filename in unwanted_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"✓ Removed: {filename}")
            except Exception as e:
                print(f"✗ Failed to remove {filename}: {e}")
        else:
            print(f"  (File not found: {filename})")
    
    print("✓ Cleanup completed!")

def generate_random_panels(num_panels=20):
    """Generate random corrugated panels without dents"""
    print(f"\n=== Generating {num_panels} Random Container Back Panels ===")
    
    # Clean up previous panel outputs first
    cleanup_panel_outputs()
    
    create_output_folders()
    panel_generator = CorrugatedPanelGenerator()
    panel_specs = []
    
    for i in range(num_panels):
        print(f"\nGenerating Back Panel {i+1}/{num_panels}...")
        
        # Randomize parameters
        panel_generator.randomize_parameters()
        
        # Create panel
        panel_mesh = panel_generator.create_corrugated_panel()
        
        # Generate filename and save
        filename = f"panel/panel_{i+1:02d}.obj"
        success = panel_generator.export_to_obj(panel_mesh, filename)
        
        if success:
            # Save specifications
            specs = panel_generator.get_panel_specs()
            specs['filename'] = filename
            panel_specs.append(specs)
            
            print(f"  ✓ {specs['container_type']} Back Panel {i+1} saved: {filename}")
            print(f"    Dimensions: {specs['panel_width']:.3f}m × {specs['panel_height']:.3f}m")
            print(f"    Corrugation: {specs['corrugation_depth']:.3f}m depth, {specs['corrugation_frequency']} freq")
        else:
            print(f"  ✗ Failed to save Back Panel {i+1}")
    
    # Save panel specifications to JSON
    if panel_specs:
        with open('panel/panel_specifications.json', 'w') as f:
            json.dump(panel_specs, f, indent=2)
    
    print(f"\n✓ Generated {len(panel_specs)} random container back panels in 'panel' folder")
    return panel_specs

def generate_dented_panels(num_panels=total_panels):
    """Generate random corrugated container back panels with dents"""
    print(f"\n=== Generating {num_panels} Dented Container Back Panels ===")
    
    # Clean up previous dent outputs first
    cleanup_dent_outputs()
    
    create_output_folders()
    panel_generator = CorrugatedPanelGenerator()
    dent_generator = DentGenerator()
    dented_specs = []
    
    for i in range(num_panels):
        print(f"\nGenerating Dented Back Panel {i+1}/{num_panels}...")
        
        # Randomize panel parameters
        panel_generator.randomize_parameters()
        
        # Create base panel
        base_mesh = panel_generator.create_corrugated_panel()
        
        # Create dented version (panel_length is maintained for backward compatibility)
        dented_mesh = dent_generator.create_dented_panel_from_base(
            base_mesh, 
            panel_generator.panel_length,  # This is now panel_width (back panel width)
            panel_generator.panel_height,
            panel_generator.container_color_rgb  # Pass the container color
        )
        
        # Generate filename and save
        filename = f"panel_dents/dented_panel_{i+1:02d}.obj"
        success = panel_generator.export_to_obj(dented_mesh, filename)
        
        if success:
            # Save specifications (combine panel and dent specs)
            specs = panel_generator.get_panel_specs()
            specs.update(dent_generator.get_dent_specs())
            specs['filename'] = filename
            dented_specs.append(specs)
            
            print(f"  ✓ {specs['container_type']} Dented Back Panel {i+1} saved: {filename}")
            print(f"    Dimensions: {specs['panel_width']:.3f}m × {specs['panel_height']:.3f}m")
            print(f"    Corrugation: {specs['corrugation_depth']:.3f}m depth, {specs['corrugation_frequency']} freq")
            
            # Print dent info based on type
            dent_type = specs.get('dent_type', 'unknown')
            if dent_type == 'circular':
                radius = specs.get('radius', 0)
                depth = specs.get('depth', 0)
                print(f"    Dent: {dent_type} impact, {radius*2:.2f}m diameter, {depth:.3f}m deep")
            elif dent_type in ['diagonal_scrape', 'elongated_scratch']:
                if 'start_x' in specs and 'end_x' in specs and 'start_y' in specs and 'end_y' in specs:
                    length = ((specs['end_x'] - specs['start_x'])**2 + (specs['end_y'] - specs['start_y'])**2)**0.5
                    width = specs.get('width', 0)
                    depth = specs.get('depth', 0)
                    print(f"    Dent: {dent_type.replace('_', ' ')}, {length:.2f}m x {width:.2f}m, {depth:.3f}m deep")
                else:
                    print(f"    Dent: {dent_type.replace('_', ' ')}")
            elif dent_type == 'irregular_collision':
                depth = specs.get('depth', 0)
                print(f"    Dent: {dent_type.replace('_', ' ')}, {depth:.3f}m deep")
            elif dent_type == 'multi_impact':
                impacts = specs.get('impacts', [])
                print(f"    Dent: {dent_type.replace('_', ' ')}, {len(impacts)} impacts")
            elif dent_type == 'corner_damage':
                corner = specs.get('corner', 0)
                radius = specs.get('radius', 0)
                depth = specs.get('depth', 0)
                corner_names = ['bottom-left', 'bottom-right', 'top-left', 'top-right']
                corner_name = corner_names[corner] if corner < len(corner_names) else 'unknown'
                print(f"    Dent: {dent_type.replace('_', ' ')}, {corner_name} corner, {radius:.2f}m radius, {depth:.3f}m deep")
            else:
                print(f"    Dent: {dent_type.replace('_', ' ')}")
        else:
            print(f"  ✗ Failed to save Dented Back Panel {i+1}")
    
    # Save dented panel specifications to JSON
    if dented_specs:
        with open('panel_dents/dented_panel_specifications.json', 'w') as f:
            json.dump(dented_specs, f, indent=2)
    
    print(f"\n✓ Generated {len(dented_specs)} dented container back panels in 'panel_dents' folder")
    return dented_specs

def generate_single_panel():
    """Generate a single container back panel with default parameters"""
    print("\nGenerating single container back panel with dent...")
    
    # Clean up any previous single panel outputs
    stray_files = ['corrugated_container_panel.obj']
    for filename in stray_files:
        if Path(filename).exists():
            try:
                os.remove(filename)
                print(f"  ✓ Cleaned up previous: {filename}")
            except Exception as e:
                print(f"  ✗ Failed to remove {filename}: {e}")
    
    # Create generators
    panel_generator = CorrugatedPanelGenerator()
    dent_generator = DentGenerator()
    
    # Set to 20ft container back panel by default
    panel_generator.set_container_type("20ft")
    
    # Create base panel with default parameters
    base_mesh = panel_generator.create_corrugated_panel()
    
    # Apply dent (using panel_length for backward compatibility)
    dented_mesh = dent_generator.apply_dent_to_mesh(
        base_mesh.copy(), 
        panel_generator.panel_length,  # This is panel_width for back panel
        panel_generator.panel_height
    )
    dent_generator.apply_dent_coloring(dented_mesh, dented_mesh.faces, dented_mesh.vertices, panel_generator.container_color_rgb)
    
    # Export
    output_filename = "corrugated_container_panel.obj"
    success = panel_generator.export_to_obj(dented_mesh, output_filename)
    
    if success:
        print(f"\n📊 Container Back Panel Statistics:")
        print(f"  Container Type: {panel_generator.container_type}")
        print(f"  Panel Type: Back Panel (end panel opposite to doors)")
        print(f"  Dimensions: {panel_generator.panel_width:.3f}m × {panel_generator.panel_height:.3f}m")
        print(f"  Vertices: {len(dented_mesh.vertices)}")
        print(f"  Faces: {len(dented_mesh.faces)}")
        print(f"  Surface Area: {dented_mesh.area:.3f} m²")
        print(f"  Volume: {dented_mesh.volume:.6f} m³")
        print(f"  Bounding Box: {dented_mesh.bounds}")
        print(f"  Colors: Realistic shipping container blue with damage highlights")
        print(f"  File: {output_filename}")
        print(f"\n🎯 This back panel represents the area typically inspected by:")
        print(f"     • Automated damage detection systems")
        print(f"     • Container inspection cameras at 2.35m distance")
        print(f"     • Quality control processes in shipping yards")

def cleanup_dented_containers():
    """Clean up all dented container outputs before creating new ones"""
    print("\n🧹 Cleaning up previous dented container outputs...")
    
    # Directory to clean
    dented_dir = Path("complete_containers_dented")
    
    files_removed = 0
    dirs_removed = 0
    
    # Count files before removing
    if dented_dir.exists():
        # Count OBJ files
        obj_files = list(dented_dir.glob("*.obj"))
        files_removed = len(obj_files)
        
        # Remove complete_containers_dented directory and all contents
        try:
            shutil.rmtree(dented_dir)
            print(f"  ✓ Removed directory: {dented_dir}")
            if files_removed > 0:
                print(f"  ✓ Removed {files_removed} file(s)")
            dirs_removed += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {dented_dir}: {e}")
    
    print(f"  📊 Dented container cleanup: {files_removed} files, {dirs_removed} directories removed")
    return files_removed + dirs_removed > 0

def cleanup_scene_outputs():
    """Clean up all scene rendering outputs before creating new ones"""
    print("\n🧹 Cleaning up previous scene rendering outputs...")
    
    # Directories to clean
    output_dir = Path("output_scene")
    dataset_dir = Path("output_scene_dataset")
    
    files_removed = 0
    dirs_removed = 0
    
    # Count and remove output_scene directory
    if output_dir.exists():
        # Count all files
        all_files = list(output_dir.glob("*"))
        files_removed = len(all_files)
        
        # Remove output_scene directory and all contents
        try:
            shutil.rmtree(output_dir)
            print(f"  ✓ Removed directory: {output_dir}")
            if files_removed > 0:
                print(f"  ✓ Removed {files_removed} file(s)")
            dirs_removed += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {output_dir}: {e}")
    
    # Count and remove output_scene_dataset directory
    if dataset_dir.exists():
        # Count all files
        dataset_files = list(dataset_dir.glob("*"))
        dataset_files_count = len(dataset_files)
        
        # Remove output_scene_dataset directory and all contents
        try:
            shutil.rmtree(dataset_dir)
            print(f"  ✓ Removed directory: {dataset_dir}")
            if dataset_files_count > 0:
                print(f"  ✓ Removed {dataset_files_count} file(s) from dataset folder")
            dirs_removed += 1
            files_removed += dataset_files_count
        except Exception as e:
            print(f"  ✗ Failed to remove {dataset_dir}: {e}")
    
    print(f"  📊 Scene output cleanup: {files_removed} files, {dirs_removed} directories removed")
    return files_removed + dirs_removed > 0

def generate_dented_complete_containers():
    """Add dents to complete containers from complete_containers folder"""
    print("\n=== Adding Dents to Complete Containers ===")
    
    # Clean up previous dented container outputs first
    cleanup_dented_containers()
    
    # Check if complete_containers folder exists
    input_folder = Path("complete_containers")
    if not input_folder.exists():
        print("❌ No containers found! Please generate complete containers first (option 5)")
        return
    
    # Find all OBJ files (skip _scene.obj files)
    obj_files = [f for f in input_folder.glob("*.obj") if "_scene.obj" not in f.name]
    
    if not obj_files:
        print("❌ No container files found in 'complete_containers' folder!")
        return
    
    print(f"Found {len(obj_files)} container file(s) to process")
    
    # Get user input for number of dents
    while True:
        try:
            num_dents = int(input("\nEnter number of dents per container (1-20): ").strip())
            if 1 <= num_dents <= 20:
                break
            else:
                print("Please enter a number between 1 and 20")
        except ValueError:
            print("Please enter a valid number")
    
    # Create output folder
    output_folder = Path("complete_containers_dented")
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Created folder: {output_folder}")
    
    # Import the dent generation function
    try:
        from generate_dents_complete import add_dents_to_container
    except ImportError as e:
        print(f"❌ Error importing generate_dents_complete module: {e}")
        return
    
    print("\n" + "=" * 60)
    print(f"Processing {len(obj_files)} container(s)...")
    print("=" * 60)
    
    processed_count = 0
    
    # Process each file
    for i, input_file in enumerate(obj_files, 1):
        print(f"\n[{i}/{len(obj_files)}] Processing: {input_file.name}")
        
        # Create output filename with "dented" added before .obj
        # e.g., container_20ft_0001.obj -> container_20ft_0001_dented.obj
        stem = input_file.stem  # filename without extension
        output_filename = f"{stem}_dented.obj"
        output_file = output_folder / output_filename
        
        try:
            add_dents_to_container(
                input_path=str(input_file),
                output_path=str(output_file),
                num_dents=num_dents,
                size_range=(0.08, 0.50),
                depth_range=(0.02, 0.15),
                varied_severity=True
            )
            processed_count += 1
            print(f"  ✓ Successfully processed: {output_filename}")
        except Exception as e:
            print(f"  ✗ Error processing {input_file.name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("DENT GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Successfully processed {processed_count} out of {len(obj_files)} container(s)")
    print(f"All dented containers saved to: {output_folder.absolute()}")

def render_scenes():
    """Render container scenes by comparing original and dented containers to generate dent segmentation"""
    print("\n=== Rendering Container Scenes with Dent Segmentation ===")
    
    # Setup paths
    original_dir = Path("complete_containers")
    dented_dir = Path("complete_containers_dented")
    output_dir = Path("output_scene")
    
    # Check if directories exist
    if not original_dir.exists():
        print("❌ No original containers found!")
        print("   Please generate complete containers first (option 5)")
        return
    
    if not dented_dir.exists():
        print("❌ No dented containers found!")
        print("   Please add dents to containers first (option 6)")
        return
    
    # Find all original container mesh files
    original_files = sorted([f for f in original_dir.glob("*.obj") if "_scene.obj" not in f.name])
    
    if not original_files:
        print(f"❌ No container mesh files found in {original_dir}")
        print("   Please generate complete containers first (option 5)")
        return
    
    print(f"Found {len(original_files)} original container file(s)")
    
    # Ask user if they want to clean up previous outputs
    cleanup_choice = input("\nClean up previous scene outputs? (y/n): ").strip().lower()
    if cleanup_choice == 'y':
        cleanup_scene_outputs()
    
    # Ask for depth threshold
    try:
        threshold_input = input("\nEnter depth difference threshold in meters (default 0.01 = 10mm): ").strip()
        threshold = float(threshold_input) if threshold_input else 0.01
    except ValueError:
        threshold = 0.01
        print(f"Invalid input, using default threshold: {threshold}m")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Depth threshold: {threshold}m ({threshold*1000:.1f}mm)")
    
    # Import required modules
    try:
        from config import ContainerConfig, RendererConfig
        from camera_position import CameraPoseGenerator
        from compare_dents_depth import DentComparisonRenderer
        import re
        import logging
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("   Please ensure config.py, camera_position.py, and compare_dents_depth.py exist")
        import traceback
        traceback.print_exc()
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize comparison renderer
    try:
        renderer_config = RendererConfig()
        comparison_renderer = DentComparisonRenderer(
            image_size=renderer_config.IMAGE_SIZE,
            camera_fov=renderer_config.CAMERA_FOV
        )
        print("✓ DentComparisonRenderer initialized")
    except Exception as e:
        print(f"❌ Error initializing DentComparisonRenderer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Extract container info helper function
    def extract_container_info(filename: str):
        """Extract container type and sample ID from filename."""
        pattern = r'container_(20ft|40ft|40ft_hc)_(\d+)\.obj'
        match = re.match(pattern, filename)
        if match:
            container_type = match.group(1)
            sample_id = int(match.group(2))
            return container_type, sample_id
        return None, None
    
    # Process each container pair
    successful = 0
    failed = 0
    
    print("\n" + "=" * 60)
    print(f"Processing {len(original_files)} container pair(s)...")
    print("=" * 60)
    
    for i, original_path in enumerate(original_files, 1):
        container_type, sample_id = extract_container_info(original_path.name)
        
        if container_type is None or sample_id is None:
            print(f"[{i}/{len(original_files)}] ⚠️  Could not parse container info from: {original_path.name}")
            failed += 1
            continue
        
        # Find corresponding dented file
        dented_path = dented_dir / f"container_{container_type}_{sample_id:04d}_dented.obj"
        
        if not dented_path.exists():
            print(f"[{i}/{len(original_files)}] ⚠️  Dented file not found: {dented_path.name}")
            failed += 1
            continue
        
        print(f"\n[{i}/{len(original_files)}] Processing:")
        print(f"   Original: {original_path.name}")
        print(f"   Dented: {dented_path.name}")
        print(f"   Container Type: {container_type}, Sample ID: {sample_id}")
        
        try:
            # Process container pair - this will render both, compare depths, and generate segmentation
            container_output_dir = output_dir / f"container_{container_type}_{sample_id:04d}"
            comparison_renderer.process_container_pair(
                original_path=original_path,
                dented_path=dented_path,
                output_dir=container_output_dir,
                container_type=container_type,
                threshold=threshold
            )
            successful += 1
            print(f"   ✓ Successfully processed container {sample_id}")
            
        except Exception as e:
            print(f"   ✗ Failed to process {original_path.name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Cleanup
    try:
        comparison_renderer.cleanup()
    except:
        pass
    
    # Summary
    print("\n" + "=" * 60)
    print("SCENE RENDERING WITH DENT SEGMENTATION COMPLETE!")
    print("=" * 60)
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(original_files)}")
    print(f"Output directory: {output_dir.absolute()}")
    print("\nGenerated files per container:")
    print("  • RGB images: *_original_rgb.png, *_dented_rgb.png")
    print("  • Depth maps: *_original_depth.npy, *_dented_depth.npy")
    print("  • Depth difference: *_depth_diff.npy")
    print("  • Dent segmentation mask: *_dent_mask.png (WHITE = dented areas)")
    print("  • Point clouds: *_original_pointcloud.ply, *_dented_pointcloud.ply")
    print("=" * 60)

def print_menu():
    """Print the main menu options"""
    print("\n" + "="*50)
    print("🚢 SHIPPING CONTAINER GENERATOR")
    print("="*50)
    print("1. Generate Random Back Panels (No Dents)")
    print("2. Generate Dented Back Panels") 
    print("3. Generate Single Back Panel")
    print("4. Render RGB-D Images")
    print("5. Generate Complete Shipping Container")
    print("6. Add Dents to Complete Containers")
    print("7. Render Container Scenes (PyRender)")
    print("0. Exit")
    print("="*50)

def main():
    """Main application entry point"""
    print("🚢 Welcome to the Shipping Container Generator!")
    print("This tool generates realistic corrugated shipping container panels with dents.")
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == '1':
            try:
                num = int(input(f"Enter number of panels to generate (1-{total_panels}): "))
                if 1 <= num <= total_panels:
                    generate_random_panels(num)
                else:
                    print(f"Please enter a number between 1 and {total_panels}")
            except ValueError:
                print("Please enter a valid number")
                
        elif choice == '2':
            try:
                num = int(input(f"Enter number of dented panels to generate (1-{total_panels}): "))
                if 1 <= num <= total_panels:
                    generate_dented_panels(num)
                else:
                    print(f"Please enter a number between 1 and {total_panels}")
            except ValueError:
                print("Please enter a valid number")
                
        elif choice == '3':
            generate_single_panel()
            
        elif choice == '4':
            # Check if panels exist before rendering
            panel_files = list(Path("panel").glob("*.obj")) if Path("panel").exists() else []
            dented_files = list(Path("panel_dents").glob("*.obj")) if Path("panel_dents").exists() else []
            
            if not panel_files and not dented_files:
                print("❌ No panels found! Please generate panels first (option 1 or 2)")
                continue
                
            try:
                from render_rgbd import main as render_main
                render_main()
            except ImportError as e:
                print(f"❌ Error importing render module: {e}")
            except Exception as e:
                print(f"❌ Error during rendering: {e}")
        
        elif choice == '5':
            try:
                from generate_complete_container import main as generate_complete_main
                generate_complete_main()
            except ImportError as e:
                print(f"❌ Error importing generate_complete_container module: {e}")
            except Exception as e:
                print(f"❌ Error during container generation: {e}")
        
        elif choice == '6':
            try:
                generate_dented_complete_containers()
            except ImportError as e:
                print(f"❌ Error importing generate_dents_complete module: {e}")
            except Exception as e:
                print(f"❌ Error during dent generation: {e}")
        
        elif choice == '7':
            try:
                render_scenes()
            except Exception as e:
                print(f"❌ Error during scene rendering: {e}")
                import traceback
                traceback.print_exc()
                
        elif choice == '0':
            print("👋 Thank you for using the Shipping Container Generator!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 0-7.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 