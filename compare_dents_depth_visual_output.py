#!/usr/bin/env python3
"""
Generate Visual Overlay Outputs from Depth Comparison Summary

This script:
1. Loads comparison summary JSON files produced by compare_dents_depth.py
2. Loads rendered RGB images and depth maps
3. Creates visual overlays showing dent regions with labels
4. Saves visual output images

This script must be run AFTER compare_dents_depth.py has generated the summary JSON.
"""

import numpy as np
import json
import imageio
import cv2
from pathlib import Path
from typing import Optional, Dict
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class VisualOutputGenerator:
    """Generates visual overlay outputs from depth comparison summary JSON."""
    
    def __init__(self, camera_fov: float = 75.0):
        """
        Initialize the visual output generator.
        
        Args:
            camera_fov: Camera field of view in degrees (must match compare_dents_depth.py)
        """
        self.camera_fov = camera_fov
        
        # Pre-compute camera intrinsics from FOV (matching compare_dents_depth.py)
        # These intrinsics are used for area calculations when fallback is needed
        fov_y_rad = np.deg2rad(camera_fov)
        self.fov_x_rad = fov_y_rad  # Square images: fov_x = fov_y
        
        logger.info(f"VisualOutputGenerator initialized (fov={camera_fov}°)")
    
    def load_dent_metadata(self, dented_path: Path) -> Optional[Dict]:
        """
        Load dent metadata from JSON file corresponding to the dented container OBJ file.
        
        Args:
            dented_path: Path to dented container OBJ file
            
        Returns:
            Dictionary with dent metadata or None if file not found
        """
        # Try to find corresponding JSON file
        json_path = dented_path.with_suffix('.json')
        
        if not json_path.exists():
            logger.warning(f"Dent metadata JSON not found: {json_path}")
            return None
        
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded dent metadata from: {json_path}")
            return metadata
        except Exception as e:
            logger.warning(f"Failed to load dent metadata from {json_path}: {e}")
            return None
    
    def extract_dent_area_from_metadata(self, dent_metadata: Optional[Dict]) -> float:
        """
        Extract total dent area in cm² from metadata.
        
        Args:
            dent_metadata: Dictionary with dent metadata
            
        Returns:
            Total dent area in cm²
        """
        if not dent_metadata or 'dents' not in dent_metadata:
            return 0.0
        
        total_area_m2 = 0.0
        
        for dent in dent_metadata['dents']:
            dent_type = dent.get('type', '')
            
            if dent_type == 'elliptical':
                # Area = π * radius_x * radius_y
                radius_x = dent.get('radius_x', 0)
                radius_y = dent.get('radius_y', 0)
                area = np.pi * radius_x * radius_y
                total_area_m2 += area
            elif dent_type == 'corner':
                # Area = π * radius²
                radius = dent.get('radius', 0)
                area = np.pi * radius * radius
                total_area_m2 += area
            elif dent_type == 'crease':
                # Area = length * width
                length = dent.get('length', 0)
                width = dent.get('width', 0)
                area = length * width
                total_area_m2 += area
            elif dent_type == 'circular':
                # Area = π * radius²
                radius = dent.get('radius', 0)
                area = np.pi * radius * radius
                total_area_m2 += area
        
        # Convert to cm²
        total_area_cm2 = total_area_m2 * 10000.0
        return total_area_cm2
    
    def extract_dent_depth_from_metadata(self, dent_metadata: Optional[Dict]) -> float:
        """
        Extract maximum dent depth in mm from metadata.
        
        Args:
            dent_metadata: Dictionary with dent metadata
            
        Returns:
            Maximum dent depth in mm
        """
        if not dent_metadata or 'dents' not in dent_metadata:
            return 0.0
        
        max_depth_mm = 0.0
        
        for dent in dent_metadata['dents']:
            # Prefer depth_mm if available, otherwise convert depth (meters) to mm
            depth_mm = dent.get('depth_mm', 0)
            if depth_mm == 0:
                depth_m = dent.get('depth', 0)
                depth_mm = depth_m * 1000.0
            
            max_depth_mm = max(max_depth_mm, depth_mm)
        
        return max_depth_mm
    
    def calculate_dent_area(self, dent_mask: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Calculate dent area in cm² by converting pixel area to real-world area.
        
        This method matches the calculation in compare_dents_depth.py for consistency.
        Uses camera intrinsics derived from FOV (matching pyrender PerspectiveCamera).
        
        Note: This calculates the VISIBLE dent area in this particular shot.
        The same dent may appear in multiple shots with different visible areas.
        
        Args:
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            depth_map: Depth map (H, W) in meters
            
        Returns:
            Dent area in cm² (visible area in this shot)
        """
        h, w = dent_mask.shape
        dent_pixels = (dent_mask > 127)
        
        if not np.any(dent_pixels):
            return 0.0
        
        # Use pre-computed camera intrinsics from FOV (matching compare_dents_depth.py)
        fov_y_rad = np.deg2rad(self.camera_fov)
        
        # Get depths for dent pixels
        dent_depths = depth_map[dent_pixels]
        valid_depths = dent_depths[dent_depths > 0]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Calculate average depth for dent region
        # Note: This matches compare_dents_depth.py's current implementation
        avg_depth = np.mean(valid_depths)
        
        # Calculate pixel dimensions in meters at average depth using camera intrinsics
        # Using the same FOV that was used to create the pyrender PerspectiveCamera
        pixel_width_m = 2 * avg_depth * np.tan(self.fov_x_rad / 2.0) / w
        pixel_height_m = 2 * avg_depth * np.tan(fov_y_rad / 2.0) / h
        pixel_area_m2 = pixel_width_m * pixel_height_m
        
        # Calculate total area
        num_dent_pixels = np.sum(dent_pixels)
        total_area_m2 = num_dent_pixels * pixel_area_m2
        
        # Convert to cm²
        total_area_cm2 = total_area_m2 * 10000.0
        
        return total_area_cm2
    
    def calculate_dent_depth(self, depth_diff: np.ndarray, dent_mask: np.ndarray) -> float:
        """
        Calculate maximum dent depth in mm.
        
        Args:
            depth_diff: Depth difference map (H, W) in meters
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas
            
        Returns:
            Maximum dent depth in mm
        """
        dent_pixels = (dent_mask > 127)
        
        if not np.any(dent_pixels):
            return 0.0
        
        # Get depth differences for dent pixels
        dent_depths = depth_diff[dent_pixels]
        valid_depths = dent_depths[np.isfinite(dent_depths) & (dent_depths > 0)]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Calculate maximum depth difference
        max_depth_m = np.max(valid_depths)
        
        # Convert to mm
        max_depth_mm = max_depth_m * 1000.0
        
        return max_depth_mm
    
    def create_visual_overlay(self, background_rgb: np.ndarray, dent_mask: np.ndarray,
                              depth_diff: np.ndarray, dented_depth: np.ndarray,
                              dent_metadata: Optional[Dict] = None,
                              shot_statistics: Optional[Dict] = None,
                              dent_segments: Optional[list] = None,
                              overlay_alpha: float = 0.2, outline_thickness: int = 2) -> np.ndarray:
        """
        Create a visual overlay showing dent regions on the background image with labels.
        
        Args:
            background_rgb: Background RGB image (H, W, 3) - the dented container image
            dent_mask: Binary mask (H, W) where WHITE (255) = dented areas, BLACK (0) = normal areas
            depth_diff: Depth difference map (H, W) in meters
            dented_depth: Depth map of dented container (H, W) in meters
            dent_metadata: Optional dictionary with dent metadata from JSON file
            shot_statistics: Optional dictionary with shot statistics from summary JSON
            overlay_alpha: Transparency of the overlay fill (0.0-1.0, default: 0.2)
            outline_thickness: Thickness of the red outline in pixels (default: 2)
            
        Returns:
            RGB image with dent overlay visualization and labels (H, W, 3)
        """
        # Ensure mask and background have matching dimensions
        if background_rgb.shape[:2] != dent_mask.shape[:2]:
            logger.warning(f"Shape mismatch: background {background_rgb.shape[:2]} vs mask {dent_mask.shape[:2]}")
            h, w = background_rgb.shape[:2]
            dent_mask = cv2.resize(dent_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Ensure depth maps match dimensions
        h, w = background_rgb.shape[:2]
        if depth_diff.shape[:2] != (h, w):
            depth_diff = cv2.resize(depth_diff, (w, h), interpolation=cv2.INTER_LINEAR)
        if dented_depth.shape[:2] != (h, w):
            dented_depth = cv2.resize(dented_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert background to float for blending
        background = background_rgb.astype(np.float32) / 255.0
        
        # Create red overlay color (RGB: red = [1, 0, 0])
        overlay_color = np.array([1.0, 0.0, 0.0])  # Red color
        
        # Create binary mask (0 or 1) from dent_mask
        mask_binary = (dent_mask > 127).astype(np.float32)
        
        # Create overlay image with red color
        overlay = np.zeros_like(background)
        overlay[mask_binary > 0] = overlay_color
        
        # Blend overlay with background using alpha transparency
        # result = background * (1 - alpha * mask) + overlay * (alpha * mask)
        alpha_mask = mask_binary * overlay_alpha
        result = background * (1.0 - alpha_mask[..., np.newaxis]) + overlay * alpha_mask[..., np.newaxis]
        
        # Add red outline for better visibility
        # Find contours of the dent mask
        mask_uint8 = (mask_binary * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw red outline on the result
        result_uint8 = (result * 255).astype(np.uint8)
        cv2.drawContours(result_uint8, contours, -1, (255, 0, 0), outline_thickness)
        
        # Get dent area and depth from shot_statistics (preferred), fallback to metadata or calculation
        if shot_statistics:
            # Use values from summary JSON if available (even if 0.0, as that means no dents detected)
            if 'dent_area_cm2' in shot_statistics:
                dent_area_cm2 = shot_statistics['dent_area_cm2']
            else:
                # Key not in statistics, try metadata
                dent_area_cm2 = self.extract_dent_area_from_metadata(dent_metadata)
                if dent_area_cm2 == 0.0:
                    # Final fallback to calculation
                    dent_area_cm2 = self.calculate_dent_area(dent_mask, dented_depth)
            
            if 'dent_depth_mm' in shot_statistics:
                dent_depth_mm = shot_statistics['dent_depth_mm']
            else:
                # Key not in statistics, try metadata
                dent_depth_mm = self.extract_dent_depth_from_metadata(dent_metadata)
                if dent_depth_mm == 0.0:
                    # Final fallback to calculation
                    dent_depth_mm = self.calculate_dent_depth(depth_diff, dent_mask)
        else:
            # Try metadata first, then calculation
            dent_area_cm2 = self.extract_dent_area_from_metadata(dent_metadata)
            if dent_area_cm2 == 0.0:
                dent_area_cm2 = self.calculate_dent_area(dent_mask, dented_depth)
            
            dent_depth_mm = self.extract_dent_depth_from_metadata(dent_metadata)
            if dent_depth_mm == 0.0:
                dent_depth_mm = self.calculate_dent_depth(depth_diff, dent_mask)
        
        # Add text labels near the dent region
        result_uint8 = self.add_dent_labels(
            result_uint8, dent_mask, dent_area_cm2, dent_depth_mm, contours,
            shot_statistics=shot_statistics,
            dent_segments=dent_segments
        )
        
        return result_uint8
    
    def add_dent_labels(self, image: np.ndarray, dent_mask: np.ndarray,
                       dent_area_cm2: float, dent_depth_mm: float,
                       contours: list, shot_statistics: Optional[Dict] = None,
                       dent_segments: Optional[list] = None) -> np.ndarray:
        """
        Add text labels showing dent area and depth to the image.
        If multiple segments are present, adds a label for each segment.
        
        Args:
            image: RGB image (H, W, 3)
            dent_mask: Binary mask (H, W)
            dent_area_cm2: Dent area in cm² (fallback if segments not available)
            dent_depth_mm: Dent depth in mm (fallback if segments not available)
            contours: List of contours from dent mask
            shot_statistics: Optional dictionary with shot statistics from summary JSON
            dent_segments: Optional list of dent segment dictionaries with segment information
            
        Returns:
            Image with text labels added
        """
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 20
        
        # Text color (white with black outline for visibility)
        text_color = (255, 255, 255)  # White
        outline_color = (0, 0, 0)  # Black
        
        # If we have segment information, label each segment separately
        if dent_segments and len(dent_segments) > 0:
            # Sort segments by area (largest first) for better label placement
            sorted_segments = sorted(dent_segments, key=lambda x: x.get('area_cm2', 0), reverse=True)
            
            for seg_idx, segment in enumerate(sorted_segments):
                # Get segment properties
                centroid = segment.get('centroid', [0, 0])
                area_cm2 = segment.get('area_cm2', 0.0)
                width_cm = segment.get('width_cm', 0.0)
                length_cm = segment.get('length_cm', 0.0)
                max_depth_mm = segment.get('max_depth_diff_mm', 0.0)
                
                # Use centroid from segment data
                cx = int(centroid[0])
                cy = int(centroid[1])
                
                # Get direction information
                direction = segment.get('direction', 'unknown')
                direction_label = f" ({direction})" if direction != 'unknown' else ""
                
                # Prepare text labels for this segment
                segment_label = f"Dent {seg_idx + 1}{direction_label}"
                area_text = f"Area: {area_cm2:.2f} cm2"
                dimensions_text = f"Dimension: {width_cm:.1f}cm x {length_cm:.1f}cm"
                depth_text = f"Depth: {max_depth_mm:.2f} mm"
                
                # Calculate text size to position labels
                (label_w, label_h), _ = cv2.getTextSize(segment_label, font, font_scale, font_thickness)
                (area_w, area_h), _ = cv2.getTextSize(area_text, font, font_scale, font_thickness)
                (dimensions_w, dimensions_h), _ = cv2.getTextSize(dimensions_text, font, font_scale, font_thickness)
                (depth_w, depth_h), _ = cv2.getTextSize(depth_text, font, font_scale, font_thickness)
                
                max_text_w = max(label_w, area_w, dimensions_w, depth_w)
                
                # Position labels near the segment centroid (above and to the right)
                # Offset to avoid overlapping with the dent
                label_x = min(cx + 30, image.shape[1] - max_text_w - 10)
                label_y = max(cy - 40, (seg_idx + 1) * (line_height * 4) + 10)
                
                # Ensure labels don't go off-screen
                if label_y + (line_height * 4) > image.shape[0]:
                    label_y = max(10, image.shape[0] - (line_height * 4) - 10)
                
                # Draw segment number label
                cv2.putText(image, segment_label, (label_x, label_y),
                           font, font_scale, outline_color, font_thickness + 2, cv2.LINE_AA)
                cv2.putText(image, segment_label, (label_x, label_y),
                           font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
                # Draw area label
                cv2.putText(image, area_text, (label_x, label_y + line_height),
                           font, font_scale, outline_color, font_thickness + 2, cv2.LINE_AA)
                cv2.putText(image, area_text, (label_x, label_y + line_height),
                           font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
                # Draw dimensions label (width x length)
                cv2.putText(image, dimensions_text, (label_x, label_y + line_height * 2),
                           font, font_scale, outline_color, font_thickness + 2, cv2.LINE_AA)
                cv2.putText(image, dimensions_text, (label_x, label_y + line_height * 2),
                           font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
                # Draw depth label
                cv2.putText(image, depth_text, (label_x, label_y + line_height * 3),
                           font, font_scale, outline_color, font_thickness + 2, cv2.LINE_AA)
                cv2.putText(image, depth_text, (label_x, label_y + line_height * 3),
                           font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        else:
            # Fallback: single label for entire dent mask (original behavior)
            # Find the centroid of the largest dent region for label placement
            if len(contours) > 0:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    # Fallback to center of image
                    h, w = image.shape[:2]
                    cx, cy = w // 2, h // 2
            else:
                # Fallback to center of image
                h, w = image.shape[:2]
                cx, cy = w // 2, h // 2
            
            # Prepare text labels - use statistics if available, otherwise use calculated values
            if shot_statistics:
                # Prefer max_depth_diff_mm from statistics (filtered value)
                max_depth_mm = shot_statistics.get('max_depth_diff_mm', dent_depth_mm)
                
                # Use the exact area from shot_statistics (calculated using camera intrinsics)
                area_cm2 = shot_statistics.get('dent_area_cm2', dent_area_cm2)
                
                area_text = f"Dent Area: {area_cm2:.2f} cm2"
                depth_text = f"Dent Max Depth: {max_depth_mm:.2f} mm"
            else:
                # Fallback: use calculated or metadata values
                area_text = f"Dent Area: {dent_area_cm2:.2f} cm2"
                depth_text = f"Dent Depth: {dent_depth_mm:.2f} mm"
            
            # Calculate text size to position labels
            (area_w, area_h), _ = cv2.getTextSize(area_text, font, font_scale * 1.2, font_thickness + 1)
            (depth_w, depth_h), _ = cv2.getTextSize(depth_text, font, font_scale * 1.2, font_thickness + 1)
            
            # Position labels near the dent region (above and to the right)
            label_x = min(cx + 30, image.shape[1] - max(area_w, depth_w) - 10)
            label_y = max(cy - 30, line_height * 2 + 10)
            
            # Draw text with black outline for better visibility
            # Area label
            cv2.putText(image, area_text, (label_x, label_y),
                       font, font_scale * 1.2, outline_color, font_thickness + 3, cv2.LINE_AA)
            cv2.putText(image, area_text, (label_x, label_y),
                       font, font_scale * 1.2, text_color, font_thickness + 1, cv2.LINE_AA)
            
            # Depth label
            cv2.putText(image, depth_text, (label_x, label_y + line_height * 1.5),
                       font, font_scale * 1.2, outline_color, font_thickness + 3, cv2.LINE_AA)
            cv2.putText(image, depth_text, (label_x, label_y + line_height * 1.5),
                       font, font_scale * 1.2, text_color, font_thickness + 1, cv2.LINE_AA)
        
        return image
    
    def process_summary_json(self, summary_json_path: Path, 
                            overlay_alpha: float = 0.2, 
                            outline_thickness: int = 2) -> None:
        """
        Process a comparison summary JSON file and generate visual overlays.
        
        Args:
            summary_json_path: Path to comparison summary JSON file
            overlay_alpha: Transparency of the overlay fill (0.0-1.0)
            outline_thickness: Thickness of the red outline in pixels
        """
        logger.info(f"Processing summary JSON: {summary_json_path}")
        
        if not summary_json_path.exists():
            logger.error(f"Summary JSON not found: {summary_json_path}")
            return
        
        # Load summary JSON
        with open(summary_json_path, 'r') as f:
            summary_data = json.load(f)
        
        # Extract base information
        dented_file_path = Path(summary_data['dented_file'])
        base_name = dented_file_path.stem.replace("_dented", "").replace("container_", "")
        
        # Load dent metadata
        dent_metadata = self.load_dent_metadata(dented_file_path)
        
        # Process each shot
        shots = summary_data.get('shots', [])
        logger.info(f"Found {len(shots)} shots to process")
        
        for shot_idx, shot_stat in enumerate(shots):
            shot_name = shot_stat['shot_name']
            shot_output_dir = Path(shot_stat['output_dir'])
            
            logger.info(f"  [{shot_idx+1}/{len(shots)}] Processing shot: {shot_name}")
            
            try:
                # Load rendered images and depth maps
                dented_rgb_path = shot_output_dir / f"{base_name}_dented_rgb.png"
                dent_mask_path = shot_output_dir / f"{base_name}_dent_mask.png"
                depth_diff_npy_path = shot_output_dir / f"{base_name}_depth_diff.npy"
                dented_depth_npy_path = shot_output_dir / f"{base_name}_dented_depth.npy"
                segment_json_path = shot_output_dir / f"{base_name}_dent_segments.json"
                
                # Check if all required files exist
                missing_files = []
                if not dented_rgb_path.exists():
                    missing_files.append(dented_rgb_path.name)
                if not dent_mask_path.exists():
                    missing_files.append(dent_mask_path.name)
                if not depth_diff_npy_path.exists():
                    missing_files.append(depth_diff_npy_path.name)
                if not dented_depth_npy_path.exists():
                    missing_files.append(dented_depth_npy_path.name)
                
                if missing_files:
                    logger.warning(f"    ⚠️  Missing files for {shot_name}: {', '.join(missing_files)}")
                    continue
                
                # Load images
                dented_rgb = imageio.imread(dented_rgb_path)
                dent_mask = imageio.imread(dent_mask_path)
                
                # Ensure dent_mask is grayscale (single channel)
                if len(dent_mask.shape) == 3:
                    # Convert RGB to grayscale if needed
                    dent_mask = cv2.cvtColor(dent_mask, cv2.COLOR_RGB2GRAY)
                
                # Load depth maps
                depth_diff = np.load(depth_diff_npy_path)
                dented_depth = np.load(dented_depth_npy_path)
                
                # Load dent segment information if available
                dent_segments = None
                if segment_json_path.exists():
                    try:
                        with open(segment_json_path, 'r') as f:
                            segment_data = json.load(f)
                            dent_segments = segment_data.get('segments', [])
                            if dent_segments:
                                logger.info(f"    Loaded {len(dent_segments)} dent segment(s) from JSON")
                    except Exception as e:
                        logger.warning(f"    Failed to load segment JSON: {e}")
                
                # Create visual overlay
                visual_overlay = self.create_visual_overlay(
                    dented_rgb,
                    dent_mask,
                    depth_diff,
                    dented_depth,
                    dent_metadata=dent_metadata,
                    shot_statistics=shot_stat,
                    dent_segments=dent_segments,
                    overlay_alpha=overlay_alpha,
                    outline_thickness=outline_thickness
                )
                
                # Save visual output
                visual_output_path = shot_output_dir / f"{base_name}_visualOutput.png"
                imageio.imwrite(visual_output_path, visual_overlay)
                logger.info(f"    ✓ Saved visual overlay: {visual_output_path.name}")
                
            except Exception as e:
                logger.error(f"    ✗ Error processing shot {shot_name}: {e}", exc_info=True)
                continue
        
        logger.info(f"✓ Visual output generation complete for: {summary_json_path}")


def main():
    """Main function to generate visual outputs from summary JSON files."""
    parser = argparse.ArgumentParser(
        description='Generate visual overlay outputs from depth comparison summary JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single summary JSON file
  python compare_dents_depth_visual_output.py \\
    --summary comparison_output/container_20ft_0005/20ft_0005_comparison_summary.json
  
  # Process all summary JSON files in a directory
  python compare_dents_depth_visual_output.py \\
    --summary-dir comparison_output
  
  # Process with custom overlay settings
  python compare_dents_depth_visual_output.py \\
    --summary comparison_output/container_20ft_0005/20ft_0005_comparison_summary.json \\
    --overlay-alpha 0.3 \\
    --outline-thickness 3
        """
    )
    
    parser.add_argument('--summary', type=str, help='Path to comparison summary JSON file')
    parser.add_argument('--summary-dir', type=str, help='Directory containing summary JSON files (for batch processing)')
    parser.add_argument('--overlay-alpha', type=float, default=0.2,
                       help='Transparency of overlay fill (0.0-1.0, default: 0.2)')
    parser.add_argument('--outline-thickness', type=int, default=2,
                       help='Thickness of red outline in pixels (default: 2)')
    parser.add_argument('--camera-fov', type=float, default=75.0,
                       help='Camera field of view in degrees (must match compare_dents_depth.py, default: 75.0)')
    
    args = parser.parse_args()
    
    if not args.summary and not args.summary_dir:
        parser.error("Either --summary or --summary-dir must be provided")
    
    # Initialize generator
    generator = VisualOutputGenerator(camera_fov=args.camera_fov)
    
    try:
        if args.summary_dir:
            # Batch processing
            summary_dir = Path(args.summary_dir)
            
            if not summary_dir.exists():
                logger.error(f"Summary directory not found: {summary_dir}")
                return
            
            # Find all summary JSON files
            summary_files = sorted(summary_dir.rglob("*_comparison_summary.json"))
            
            if not summary_files:
                logger.warning(f"No summary JSON files found in: {summary_dir}")
                return
            
            logger.info(f"Found {len(summary_files)} summary JSON files")
            logger.info("=" * 60)
            
            for summary_file in summary_files:
                logger.info(f"\nProcessing: {summary_file.name}")
                generator.process_summary_json(
                    summary_file,
                    overlay_alpha=args.overlay_alpha,
                    outline_thickness=args.outline_thickness
                )
        
        else:
            # Single file processing
            summary_path = Path(args.summary)
            
            if not summary_path.exists():
                logger.error(f"Summary JSON not found: {summary_path}")
                return
            
            generator.process_summary_json(
                summary_path,
                overlay_alpha=args.overlay_alpha,
                outline_thickness=args.outline_thickness
            )
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return
    
    logger.info("✓ All visual outputs generated successfully")


if __name__ == "__main__":
    main()

