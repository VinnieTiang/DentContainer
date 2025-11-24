#!/usr/bin/env python3
"""
Main script to render container scenes using PyTorch3D.
Renders all containers from complete_containers_dented/ folder and saves outputs to output_scene/.
"""

import logging
from pathlib import Path
import re
from typing import Optional, Tuple

from config import ContainerConfig, RendererConfig
from scene_renderer import SceneRenderer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_container_info(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract container type and sample ID from filename.
    
    Examples:
        container_20ft_0003_dented.obj -> ("20ft", 3)
        container_40ft_hc_0001_dented.obj -> ("40ft_hc", 1)
    """
    # Pattern: container_{type}_{id}_dented.obj
    pattern = r'container_(20ft|40ft|40ft_hc)_(\d+)_dented\.obj'
    match = re.match(pattern, filename)
    
    if match:
        container_type = match.group(1)
        sample_id = int(match.group(2))
        return container_type, sample_id
    return None, None


def main():
    """Main function to render all container scenes."""
    
    # Setup paths
    input_dir = Path("complete_containers_dented")
    output_dir = Path("output_scene")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Check if input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.info("Please ensure containers are generated in complete_containers_dented/")
        return
    
    # Initialize configurations
    container_config = ContainerConfig()
    renderer_config = RendererConfig()
    
    # Initialize renderer
    renderer = SceneRenderer(renderer_config, container_config)
    logger.info("SceneRenderer initialized")
    
    # Find all container mesh files
    mesh_files = sorted(input_dir.glob("container_*_dented.obj"))
    
    if not mesh_files:
        logger.warning(f"No container mesh files found in {input_dir}")
        return
    
    logger.info(f"Found {len(mesh_files)} container mesh files")
    
    # Process each container
    successful = 0
    failed = 0
    
    for mesh_path in mesh_files:
        container_type, sample_id = extract_container_info(mesh_path.name)
        
        if container_type is None or sample_id is None:
            logger.warning(f"Could not parse container info from: {mesh_path.name}")
            failed += 1
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {mesh_path.name}")
        logger.info(f"Container Type: {container_type}, Sample ID: {sample_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Render all camera shots for this container
            # Note: defects_info is None since we're just rendering scenes
            renderer.render_and_save(
                mesh_path=mesh_path,
                container_type=container_type,
                output_dir=output_dir,
                sample_id=sample_id,
                defects_info=None,  # No defect annotations for now
                shot_name=None  # Render all shots
            )
            successful += 1
            logger.info(f"✓ Successfully rendered container {sample_id}")
            
        except Exception as e:
            logger.error(f"✗ Failed to render {mesh_path.name}: {e}", exc_info=True)
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Rendering Summary:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total: {len(mesh_files)}")
    logger.info(f"  Output directory: {output_dir.absolute()}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

