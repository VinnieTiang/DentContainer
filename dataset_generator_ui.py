#!/usr/bin/env python3
"""
Simple Dataset Generator UI
A streamlined interface for generating 3D container datasets
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Fix OpenMP library initialization error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def print_header():
    """Print a nice header"""
    print("\n" + "=" * 70)
    print(" " * 15 + "🚢 3D Container Dataset Generator")
    print("=" * 70)


def print_menu():
    """Print the main menu"""
    print("\n📋 Main Menu:")
    print("  [1] Generate Complete Containers")
    print("  [2] Add Dents to Containers")
    print("  [3] Render Container Scenes (RGB-D + Segmentation)")
    print("  [4] Run Full Pipeline (All Steps)")
    print("  [5] Check Status")
    print("  [0] Exit")
    print("-" * 70)


def count_rendered_images():
    """Count total number of scenes (shots) captured across all containers"""
    scenes_dir = Path("output_scene")
    if not scenes_dir.exists():
        return 0
    
    total_scenes = 0
    
    # Iterate through all container directories
    for container_dir in scenes_dir.glob("container_*"):
        if not container_dir.is_dir():
            continue
        
        # Count shot directories (each shot is one captured scene)
        for shot_dir in container_dir.iterdir():
            if shot_dir.is_dir():
                # Check if this shot directory has any image files (to ensure it's a valid shot)
                has_images = False
                for file_path in shot_dir.iterdir():
                    if file_path.is_file():
                        filename = file_path.name
                        # Skip debug files and JSON files
                        if 'debug_' in filename or filename.endswith('.json'):
                            continue
                        # Check if it's an image/data file
                        if file_path.suffix.lower() in {'.png', '.npy', '.ply'}:
                            has_images = True
                            break
                
                if has_images:
                    total_scenes += 1
    
    return total_scenes

def check_status():
    """Check the status of generated files"""
    print("\n📊 Current Status:")
    print("-" * 70)
    
    # Check containers
    containers_dir = Path("complete_containers")
    if containers_dir.exists():
        containers = list(containers_dir.glob("*.obj"))
        containers = [c for c in containers if "_scene.obj" not in c.name]
        print(f"  ✓ Complete Containers: {len(containers)} found")
        if containers:
            print(f"    Examples: {containers[0].name}, {containers[-1].name if len(containers) > 1 else ''}")
    else:
        print("  ✗ Complete Containers: None")
    
    # Check dented containers
    dented_dir = Path("complete_containers_dented")
    if dented_dir.exists():
        dented = list(dented_dir.glob("*.obj"))
        print(f"  ✓ Dented Containers: {len(dented)} found")
        if dented:
            print(f"    Examples: {dented[0].name}, {dented[-1].name if len(dented) > 1 else ''}")
    else:
        print("  ✗ Dented Containers: None")
    
    # Check rendered scenes - count total scenes (shots)
    total_scenes = count_rendered_images()
    if total_scenes > 0:
        print(f"  ✓ Rendered Scenes: {total_scenes} scenes found")
    else:
        print("  ✗ Rendered Scenes: None")
    
    print("-" * 70)


def generate_containers():
    """Generate complete containers"""
    print("\n🚀 Step 1: Generate Complete Containers")
    print("-" * 70)
    
    try:
        from generate_complete_container import ShippingContainerGenerator
        import shutil
    except ImportError as e:
        print(f"❌ Error: Could not import required modules: {e}")
        return False
    
    # Get user input
    while True:
        try:
            num_containers = int(input("\nEnter number of containers to generate (1-500): ").strip())
            if 1 <= num_containers <= 500:
                break
            else:
                print("Please enter a number between 1 and 500")
        except ValueError:
            print("Please enter a valid number")
    
    print("\nSelect container type:")
    print("  [1] 20ft container")
    print("  [2] 40ft container")
    print("  [3] 40ft High Cube container")
    print("  [4] Random (mix of all types)")
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        else:
            print("Please enter a valid choice (1-4)")
    
    # Map choice to container types
    container_types_map = {
        '1': ['20ft'],
        '2': ['40ft'],
        '3': ['40ft_hc'],
        '4': ['20ft', '40ft', '40ft_hc']
    }
    container_types = container_types_map[choice]
    
    # Clean up previous outputs
    output_dir = Path("complete_containers")
    if output_dir.exists():
        cleanup = input("\nClean up previous containers? (y/n): ").strip().lower()
        if cleanup == 'y':
            shutil.rmtree(output_dir)
            print("  ✓ Cleaned up previous containers")
    
    output_dir.mkdir(exist_ok=True)
    
    # Generate containers
    generator = ShippingContainerGenerator()
    print(f"\n📦 Generating {num_containers} container(s)...")
    print("-" * 70)
    
    successful = 0
    for i in range(num_containers):
        # Select container type
        if len(container_types) > 1:
            container_type = container_types[i % len(container_types)]
        else:
            container_type = container_types[0]
        
        output_filename = f"container_{container_type}_{i+1:04d}.obj"
        output_path = output_dir / output_filename
        
        try:
            generator.generate(
                output_path=str(output_path),
                container_type=container_type
            )
            successful += 1
            print(f"  [{i+1}/{num_containers}] ✓ Generated: {output_filename}")
        except Exception as e:
            print(f"  [{i+1}/{num_containers}] ✗ Failed: {e}")
    
    print("-" * 70)
    print(f"✓ Successfully generated {successful}/{num_containers} container(s)")
    print(f"  Output: {output_dir.absolute()}")
    return successful > 0


def add_dents():
    """Add dents to containers"""
    print("\n🔨 Step 2: Add Dents to Containers")
    print("-" * 70)
    
    try:
        from generate_dents_complete import add_dents_to_container
        import shutil
    except ImportError as e:
        print(f"❌ Error: Could not import required modules: {e}")
        return False
    
    input_folder = Path("complete_containers")
    if not input_folder.exists():
        print("❌ Error: No containers found!")
        print("   Please generate containers first (Step 1)")
        return False
    
    # Find all container files
    obj_files = sorted([f for f in input_folder.glob("*.obj") if "_scene.obj" not in f.name])
    if not obj_files:
        print("❌ Error: No container files found!")
        return False
    
    print(f"Found {len(obj_files)} container file(s)")
    
    # Get user input
    while True:
        try:
            num_dents = int(input("\nEnter number of dents per container (1-20): ").strip())
            if 1 <= num_dents <= 20:
                break
            else:
                print("Please enter a number between 1 and 20")
        except ValueError:
            print("Please enter a valid number")
    
    # Clean up previous outputs
    output_folder = Path("complete_containers_dented")
    if output_folder.exists():
        cleanup = input("\nClean up previous dented containers? (y/n): ").strip().lower()
        if cleanup == 'y':
            shutil.rmtree(output_folder)
            print("  ✓ Cleaned up previous dented containers")
    
    output_folder.mkdir(exist_ok=True)
    
    # Process containers
    print(f"\n🔨 Processing {len(obj_files)} container(s)...")
    print("-" * 70)
    
    successful = 0
    for i, input_file in enumerate(obj_files, 1):
        stem = input_file.stem
        output_filename = f"{stem}_dented.obj"
        output_file = output_folder / output_filename
        
        try:
            add_dents_to_container(
                input_path=str(input_file),
                output_path=str(output_file),
                num_dents=num_dents,
                size_range=(0.08, 0.50),
                depth_range=(0.02, 0.07), 
                varied_severity=True
            )
            successful += 1
            print(f"  [{i}/{len(obj_files)}] ✓ Processed: {output_filename}")
        except Exception as e:
            print(f"  [{i}/{len(obj_files)}] ✗ Failed {input_file.name}: {e}")
    
    print("-" * 70)
    print(f"✓ Successfully processed {successful}/{len(obj_files)} container(s)")
    print(f"  Output: {output_folder.absolute()}")
    return successful > 0


def render_scenes():
    """Render container scenes"""
    print("\n🎬 Step 3: Render Container Scenes")
    print("-" * 70)
    
    try:
        from config import ContainerConfig, RendererConfig
        from camera_position import CameraPoseGenerator
        from compare_dents_depth import DentComparisonRenderer
        import re
        import logging
        import shutil
    except ImportError as e:
        print(f"❌ Error: Could not import required modules: {e}")
        return False
    
    # Setup paths
    original_dir = Path("complete_containers")
    dented_dir = Path("complete_containers_dented")
    output_dir = Path("output_scene")
    
    # Check prerequisites
    if not original_dir.exists():
        print("❌ Error: No original containers found!")
        print("   Please generate containers first (Step 1)")
        return False
    
    if not dented_dir.exists():
        print("❌ Error: No dented containers found!")
        print("   Please add dents first (Step 2)")
        return False
    
    # Find all original container files
    original_files = sorted([f for f in original_dir.glob("*.obj") if "_scene.obj" not in f.name])
    if not original_files:
        print("❌ Error: No container files found!")
        return False
    
    print(f"Found {len(original_files)} container file(s)")
    
    # Get user input
    cleanup_choice = input("\nClean up previous scene outputs? (y/n): ").strip().lower()
    if cleanup_choice == 'y':
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print("  ✓ Cleaned up previous scene outputs")
    
    while True:
        try:
            threshold_input = input("\nEnter depth difference threshold in meters (default 0.035 = 35mm): ").strip()
            threshold = float(threshold_input) if threshold_input else 0.035
            if 0.001 <= threshold <= 0.1:
                break
            else:
                print("Please enter a value between 0.001 and 0.1")
        except ValueError:
            threshold = 0.035
            print(f"Invalid input, using default threshold: {threshold}m")
            break
    
    # Ask for minimum area threshold
    while True:
        try:
            min_area_input = input("\nEnter minimum dent area threshold in cm² (default 1.0): ").strip()
            min_area_cm2 = float(min_area_input) if min_area_input else 1.0
            if 0.0 <= min_area_cm2 <= 1000.0:
                break
            else:
                print("Please enter a value between 0.0 and 1000.0")
        except ValueError:
            min_area_cm2 = 1.0
            print(f"Invalid input, using default minimum area: {min_area_cm2} cm²")
            break
    
    output_dir.mkdir(exist_ok=True)
    print(f"\n📐 Configuration:")
    print(f"  Depth threshold: {threshold}m ({threshold*1000:.1f}mm)")
    print(f"  Minimum area threshold: {min_area_cm2} cm²")
    print(f"  Output directory: {output_dir.absolute()}")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize renderer
    print("\n🔧 Initializing renderer...")
    renderer_config = RendererConfig()
    comparison_renderer = DentComparisonRenderer(
        image_size=renderer_config.IMAGE_SIZE,
        camera_fov=renderer_config.CAMERA_FOV
    )
    print("  ✓ Renderer initialized")
    
    # Extract container info helper
    def extract_container_info(filename: str):
        pattern = r'container_(20ft|40ft|40ft_hc)_(\d+)\.obj'
        match = re.match(pattern, filename)
        if match:
            return match.group(1), int(match.group(2))
        return None, None
    
    # Process containers
    print(f"\n🎬 Processing {len(original_files)} container pair(s)...")
    print("-" * 70)
    
    successful = 0
    failed = 0
    
    for i, original_path in enumerate(original_files, 1):
        container_type, sample_id = extract_container_info(original_path.name)
        
        if container_type is None or sample_id is None:
            print(f"  [{i}/{len(original_files)}] ⚠️  Could not parse container info from: {original_path.name}")
            failed += 1
            continue
        
        # Find corresponding dented file
        dented_path = dented_dir / f"container_{container_type}_{sample_id:04d}_dented.obj"
        
        if not dented_path.exists():
            print(f"  [{i}/{len(original_files)}] ⚠️  Dented file not found: {dented_path.name}")
            failed += 1
            continue
        
        print(f"  [{i}/{len(original_files)}] Processing: {original_path.name}")
        
        try:
            container_output_dir = output_dir / f"container_{container_type}_{sample_id:04d}"
            comparison_renderer.process_container_pair(
                original_path=original_path,
                dented_path=dented_path,
                output_dir=container_output_dir,
                container_type=container_type,
                threshold=threshold,
                min_area_cm2=min_area_cm2
            )
            successful += 1
            print(f"    ✓ Successfully processed container {sample_id}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            failed += 1
    
    # Cleanup
    try:
        comparison_renderer.cleanup()
    except:
        pass
    
    print("-" * 70)
    print(f"✓ Successfully processed: {successful}")
    if failed > 0:
        print(f"✗ Failed: {failed}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print("\nGenerated files per container:")
    print("  • RGB images: *_original_rgb.png, *_dented_rgb.png")
    print("  • Depth maps: *_original_depth.npy, *_dented_depth.npy")
    print("  • Depth difference: *_depth_diff.npy")
    print("  • Dent segmentation mask: *_dent_mask.png")
    print("  • Point clouds: *_original_pointcloud.ply, *_dented_pointcloud.ply")
    
    return successful > 0


def run_full_pipeline():
    """Run the complete pipeline"""
    print("\n🚀 Running Full Pipeline")
    print("=" * 70)
    
    # Step 1: Generate containers
    if not generate_containers():
        print("\n❌ Pipeline stopped: Failed to generate containers")
        return
    
    # Step 2: Add dents
    if not add_dents():
        print("\n❌ Pipeline stopped: Failed to add dents")
        return
    
    # Step 3: Render scenes
    if not render_scenes():
        print("\n❌ Pipeline stopped: Failed to render scenes")
        return
    
    print("\n" + "=" * 70)
    print("✅ Full Pipeline Completed Successfully!")
    print("=" * 70)
    print("\n📊 Next Steps:")
    print("  • View results: streamlit run output_scene_viewer.py")
    print("  • Check output_scene/ directory for generated data")


def main():
    """Main application entry point"""
    print_header()
    
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            generate_containers()
        elif choice == '2':
            add_dents()
        elif choice == '3':
            render_scenes()
        elif choice == '4':
            run_full_pipeline()
        elif choice == '5':
            check_status()
        elif choice == '0':
            print("\n👋 Thank you for using the Dataset Generator!")
            break
        else:
            print("❌ Invalid choice. Please enter 0-5.")
        
        if choice != '0':
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

