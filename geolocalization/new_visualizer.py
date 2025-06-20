import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV (BGR) format or grayscale.
    
    Args:
        pil_image: The PIL Image to convert.
        
    Returns:
        A NumPy array representing the image in OpenCV's BGR format or grayscale.
    """
    np_image = np.array(pil_image)
    
    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
        # RGB to BGR
        return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    elif len(np_image.shape) == 2:
        # Grayscale to BGR (for consistency in display if needed, otherwise keep 2D)
        # For SuperPoint, grayscale is often preferred for processing
        return np_image
    else:
        raise ValueError("Unsupported image format for conversion.")

def _draw_superpoint_matches_on_panel(img1_pil: Image.Image, img2_pil: Image.Image, kpts1, kpts2, matches, avg_desc_dist, panel_width: int):
    """Draw SuperPoint matches between two images for the visualization panel.
    
    Args:
        img1_pil: The first PIL image (camera view).
        img2_pil: The second PIL image (map patch).
        kpts1: Keypoints detected in the first image.
        kpts2: Keypoints detected in the second image.
        matches: List of tuples (idx1, idx2, distance) representing matches.
        avg_desc_dist: Average descriptor distance of good matches.
        panel_width: The width available for the panel display.
        
    Returns:
        An OpenCV image (NumPy array) with SuperPoint matches drawn.
    """
    # Convert PIL images to grayscale OpenCV format for SuperPoint visualization
    img1 = _pil_to_cv2(img1_pil.convert('L'))
    img2 = _pil_to_cv2(img2_pil.convert('L'))

    # Ensure images are 3-channel for drawing lines/circles (OpenCV expects BGR for colored drawings)
    if len(img1.shape) == 2: img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Create side-by-side image for visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Scale images to fit panel while maintaining aspect ratio
    target_h = 100 # Maximum height for each image in the SP visualization panel
    scale1 = target_h / h1
    scale2 = target_h / h2
    
    # Use the smaller scale to fit both images consistently
    display_scale = min(scale1, scale2)

    # Do not scale up images if they are already smaller than the target height
    if h1 < target_h and h2 < target_h:
        display_scale = 1.0

    w1_resized, h1_resized = int(w1 * display_scale), int(h1 * display_scale)
    w2_resized, h2_resized = int(w2 * display_scale), int(h2 * display_scale)

    img1_resized = cv2.resize(img1, (w1_resized, h1_resized), interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2, (w2_resized, h2_resized), interpolation=cv2.INTER_AREA)

    # Adjust panel_width if the combined image width exceeds it, scaling down further if necessary
    if (w1_resized + w2_resized) > panel_width:
        overall_scale = panel_width / (w1_resized + w2_resized)
        img1_resized = cv2.resize(img1_resized, (int(w1_resized * overall_scale), int(h1_resized * overall_scale)), interpolation=cv2.INTER_AREA)
        img2_resized = cv2.resize(img2_resized, (int(w2_resized * overall_scale), int(h2_resized * overall_scale)), interpolation=cv2.INTER_AREA)
        w1_resized, h1_resized = img1_resized.shape[1], img1_resized.shape[0]
        w2_resized, h2_resized = img2_resized.shape[1], img2_resized.shape[0]

    max_h_combined = max(h1_resized, h2_resized)
    vis_w = w1_resized + w2_resized
    # Create a blank canvas with space for text below images
    vis = np.zeros((max_h_combined + 50, vis_w, 3), dtype=np.uint8)
    # Place resized images onto the canvas
    vis[:h1_resized, :w1_resized] = img1_resized
    vis[:h2_resized, w1_resized:w1_resized+w2_resized] = img2_resized

    # Scale keypoints to match resized images
    kpts1_scaled = kpts1 * display_scale
    kpts2_scaled = kpts2 * display_scale
    
    # Define colors for drawing
    good_match_color = (0, 255, 0)  # Green for good matches
    
    # Filter for good matches based on distance threshold (e.g., < 0.5)
    good_matches = [m for m in matches if m[2] < 0.5]

    # Extract indices of matched keypoints for coloring
    matched_kpts1_indices = {match[0] for match in good_matches}
    matched_kpts2_indices = {match[1] for match in good_matches}

    # Draw keypoints (only matched ones) and lines for good matches
    for i, kpt in enumerate(kpts1_scaled):
        if i in matched_kpts1_indices:
            cv2.circle(vis, (int(kpt[0]), int(kpt[1])), 3, good_match_color, -1) # Draw filled circle
    
    for i, kpt in enumerate(kpts2_scaled):
        if i in matched_kpts2_indices:
            cv2.circle(vis, (int(kpt[0] + w1_resized), int(kpt[1])), 3, good_match_color, -1)

    for (idx1, idx2, dist) in good_matches:
        pt1 = (int(kpts1_scaled[idx1][0]), int(kpts1_scaled[idx1][1]))
        pt2 = (int(kpts2_scaled[idx2][0] + w1_resized), int(kpts2_scaled[idx2][1]))
        cv2.line(vis, pt1, pt2, good_match_color, 1) # Draw green line for good matches

    # Add text info: number of good matches and average descriptor distance
    info_text = f"SP Matches: {len(good_matches)} | Avg Dist: {avg_desc_dist:.3f}"
    text_pos_y = max_h_combined + 20
    cv2.putText(vis, info_text, (5, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # White text

    return vis

class NewSimulationVisualizer:
    """
    Visualizes the new global localization algorithm simulation.
    This class is responsible for rendering the drone's true and VIO trajectories,
    the confidence circle, probability heatmap of active patches, and a retrieval panel
    showing SuperPoint matches.
    """
    def __init__(self, config, localizer, output_video_path: str):
        """
        Initializes the visualizer with configuration and localizer.

        Args:
            config: The configuration object containing visualization parameters.
            localizer: The NewGlobalLocalizer instance to visualize data from.
            output_video_path: Path to save the simulation video.
        """
        self.config = config
        self.localizer = localizer
        self.output_video_path = output_video_path
        
        # Visualization parameters
        self.viz_scale = 0.3  # Scale down map for display to fit screen
        self.map_image = localizer.map_image # The full satellite map image

        # Calculate display dimensions dynamically to accommodate map and retrieval panel
        self.panel_width = 320  # Fixed width for the right panel for SuperPoint viz
        self.padding = 20       # Padding between map and panel, and overall frame
        
        self.map_display_w = int(localizer.map_w * self.viz_scale)
        self.map_display_h = int(localizer.map_h * self.viz_scale)

        self.display_w = self.map_display_w + self.panel_width + self.padding # Total width
        self.display_h = max(self.map_display_h, 700) # Ensure enough height for panel content
        
        # Video writer setup using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec for AVI file
        self.video_writer = cv2.VideoWriter(
            output_video_path, fourcc, config.VISUALIZATION_FPS, (self.display_w, self.display_h)
        )
        
        # Trajectory history for drawing paths over time
        self.true_trajectory = []     # Stores true drone positions
        self.vio_trajectory = []      # Stores noisy VIO estimated positions
        self.circle_centers = []      # Stores historical centers of the confidence circle
        
        print(f"Visualizer initialized. Output video: {output_video_path}")
        print(f"Display size: {self.display_w}x{self.display_h}")

    def _world_to_display(self, x_m: float, y_m: float) -> tuple:
        """Convert world coordinates (meters) to display pixel coordinates.
        
        Args:
            x_m: X-coordinate in meters (world frame).
            y_m: Y-coordinate in meters (world frame).
            
        Returns:
            A tuple (display_x, display_y) in pixel coordinates for the visualization frame.
        """
        # Convert to full-scale pixel coordinates on the original map image
        x_px_full = x_m / self.localizer.m_per_pixel + self.localizer.map_w / 2.0
        y_px_full = y_m / self.localizer.m_per_pixel + self.localizer.map_h / 2.0
        
        # Scale down for display on the visualization frame
        display_x = int(x_px_full * self.viz_scale)
        display_y = int(y_px_full * self.viz_scale)
        
        return (display_x, display_y)

    def draw_frame(self, drone, step: int):
        """Draw a single frame of the simulation, including the map, drone, and retrieval panel.
        
        Args:
            drone: The NewDrone instance providing position information.
            step: The current simulation step number.
        """
        # Start with a blank canvas (black background) for the entire display
        frame = np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)

        # Place scaled map image on the left side of the frame
        base_image = self.map_image.resize((self.map_display_w, self.map_display_h), Image.Resampling.LANCZOS)
        frame[:self.map_display_h, :self.map_display_w] = np.array(base_image)
        
        # Get current state information from the localizer and drone
        localizer_state = self.localizer.get_state_info()
        true_pos = drone.get_true_position_world()
        vio_pos = drone.get_vio_position_world()
        
        # Update trajectory history for drawing paths
        self.true_trajectory.append(true_pos)
        self.vio_trajectory.append(vio_pos)
        self.circle_centers.append(localizer_state['center_world_m']) # The center of the confidence circle
        
        # Keep trajectory history manageable to prevent excessive memory usage
        max_history = 500 # Keep last 500 positions for trajectory drawing
        if len(self.true_trajectory) > max_history:
            self.true_trajectory = self.true_trajectory[-max_history:]
            self.vio_trajectory = self.vio_trajectory[-max_history:]
            self.circle_centers = self.circle_centers[-max_history:] # Keep history for circle center as well
        
        # Draw various elements on the map area
        self._draw_probability_heatmap(frame, localizer_state['patch_probabilities'])
        self._draw_confidence_circle(frame, localizer_state)
        self._draw_trajectories(frame)
        self._draw_positions(frame, true_pos, vio_pos, localizer_state['center_world_m'])
        self._draw_info_text(frame, step, drone, localizer_state)
        
        # Draw the top 5 candidate patches on the map
        self._draw_top5_patches(frame, localizer_state['last_top5_patches'])
        
        # Draw the retrieval panel on the right side of the frame
        self._draw_retrieval_panel(frame, localizer_state)
        
        # Convert frame to BGR (OpenCV format) and write to video file
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame_bgr)

    def _draw_probability_heatmap(self, frame: np.ndarray, patch_probabilities: dict):
        """Draw probability heatmap for active patches within the confidence circle.
        Higher probabilities are shown in red, lower in blue, and medium in green.
        
        Args:
            frame: The current video frame (NumPy array) to draw on.
            patch_probabilities: A dictionary mapping (grid_row, grid_col) to probability values.
        """
        if not patch_probabilities:
            return

        # Create an overlay specific to the map area for transparent drawing
        overlay_map_area = frame[:self.map_display_h, :self.map_display_w].copy()

        # Get probability statistics for normalization and color mapping
        probs = list(patch_probabilities.values())
        min_prob = min(probs) if probs else 0.0
        max_prob = max(probs) if probs else 1.0

        # Avoid division by zero if all probabilities are the same (e.g., initial uniform state)
        if max_prob == min_prob: 
            max_prob = min_prob + 1e-9 # Add a small epsilon to prevent division by zero

        # Font settings for displaying probability text on patches
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_thickness = 1

        for (grid_row, grid_col), prob in patch_probabilities.items():
            # Convert grid coordinates of the patch center to world coordinates, then to display coordinates
            patch_center_world = self.localizer._grid_to_world(grid_row, grid_col)

            # Convert world coordinates to pixel coordinates on the full map, then scale and shift for display
            x_px_full = patch_center_world[0] / self.localizer.m_per_pixel + self.localizer.map_w / 2.0
            y_px_full = patch_center_world[1] / self.localizer.m_per_pixel + self.localizer.map_h / 2.0

            display_center_x = int(x_px_full * self.viz_scale)
            display_center_y = int(y_px_full * self.viz_scale)

            # Calculate patch size in display coordinates
            patch_size_display = int(self.localizer.patch_size_px * self.viz_scale)
            
            # Normalize probability to a 0-1 range for color mapping
            normalized_prob = (prob - min_prob) / (max_prob - min_prob)
            
            # Color mapping: blue (low) -> green (medium) -> red (high)
            if normalized_prob < 0.5:
                # Transition from blue (0,0,255) to green (0,255,0)
                blue = int(255 * (1 - 2 * normalized_prob))
                green = int(255 * 2 * normalized_prob)
                red = 0
            else:
                # Transition from green (0,255,0) to red (255,0,0)
                blue = 0
                green = int(255 * (2 - 2 * normalized_prob))
                red = int(255 * (2 * normalized_prob - 1))

            color = (red, green, blue)  # RGB format for drawing

            # Calculate rectangle corners for the patch on the overlay
            top_left_x = display_center_x - patch_size_display // 2
            top_left_y = display_center_y - patch_size_display // 2
            bottom_right_x = display_center_x + patch_size_display // 2
            bottom_right_y = display_center_y + patch_size_display // 2

            # Draw patch rectangle with transparency (alpha)
            alpha = self.config.PROBABILITY_ALPHA
            cv2.rectangle(overlay_map_area, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, -1) # Filled rectangle
            cv2.addWeighted(overlay_map_area, alpha, frame[:self.map_display_h, :self.map_display_w], 1 - alpha, 0, frame[:self.map_display_h, :self.map_display_w])
            
            # Draw probability text on the patch
            prob_text = f"{prob:.2e}" # Scientific notation for small probabilities
            text_size = cv2.getTextSize(prob_text, font, font_scale, text_thickness)[0]
            text_x = display_center_x - text_size[0] // 2
            text_y = display_center_y + text_size[1] // 2
            
            # Draw white text with a black outline for readability
            cv2.putText(frame, prob_text, (text_x, text_y), font, font_scale, (0, 0, 0), text_thickness + 1, cv2.LINE_AA) # Black outline
            cv2.putText(frame, prob_text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA) # White text

    def _draw_confidence_circle(self, frame: np.ndarray, localizer_state: dict):
        """Draw the confidence circle around the estimated drone position.
        
        Args:
            frame: The current video frame (NumPy array) to draw on.
            localizer_state: Dictionary containing localization state info.
        """
        center_m = localizer_state['center_world_m']
        radius_m = localizer_state['radius_m']
        
        # Convert center and radius to display coordinates
        display_center = self._world_to_display(center_m[0], center_m[1])
        display_radius = int(radius_m / self.localizer.m_per_pixel * self.viz_scale)
        
        # Draw the circle on the map area of the frame
        cv2.circle(frame, display_center, display_radius, self.config.CIRCLE_COLOR, 2) # Green circle, 2 pixels thick

    def _draw_trajectories(self, frame: np.ndarray):
        """Draw the historical true and VIO trajectories of the drone.
        
        Args:
            frame: The current video frame (NumPy array) to draw on.
        """
        # Draw true trajectory (red line)
        if len(self.true_trajectory) > 1:
            for i in range(1, len(self.true_trajectory)):
                pt1_true = self._world_to_display(self.true_trajectory[i-1][0], self.true_trajectory[i-1][1])
                pt2_true = self._world_to_display(self.true_trajectory[i][0], self.true_trajectory[i][1])
                cv2.line(frame, pt1_true, pt2_true, self.config.TRUE_POS_COLOR, 2) # Red line

        # Draw VIO trajectory (blue line)
        if len(self.vio_trajectory) > 1:
            for i in range(1, len(self.vio_trajectory)):
                pt1_vio = self._world_to_display(self.vio_trajectory[i-1][0], self.vio_trajectory[i-1][1])
                pt2_vio = self._world_to_display(self.vio_trajectory[i][0], self.vio_trajectory[i][1])
                cv2.line(frame, pt1_vio, pt2_vio, self.config.VIO_POS_COLOR, 1) # Blue line
        
        # Draw confidence circle center trajectory (gray dashed line) - for debugging/analysis
        if len(self.circle_centers) > 1:
            for i in range(1, len(self.circle_centers)):
                pt1_circle = self._world_to_display(self.circle_centers[i-1][0], self.circle_centers[i-1][1])
                pt2_circle = self._world_to_display(self.circle_centers[i][0], self.circle_centers[i][1])
                # Simple dashed line: draw line, then skip a few pixels
                # This is a basic way to represent a dashed line in OpenCV
                if i % 5 != 0: # Draw for most steps
                    cv2.line(frame, pt1_circle, pt2_circle, self.config.TRAJECTORY_COLOR, 1) # Gray line
                
    def _draw_positions(self, frame: np.ndarray, true_pos: tuple, vio_pos: tuple, circle_center_pos: tuple):
        """Draw current positions of true drone, VIO drone, and confidence circle center.
        
        Args:
            frame: The current video frame (NumPy array) to draw on.
            true_pos: Current true position of the drone in world coordinates.
            vio_pos: Current VIO estimated position of the drone in world coordinates.
            circle_center_pos: Current center of the confidence circle in world coordinates.
        """
        # Convert world coordinates to display coordinates
        display_true_pos = self._world_to_display(true_pos[0], true_pos[1])
        display_vio_pos = self._world_to_display(vio_pos[0], vio_pos[1])
        display_circle_center_pos = self._world_to_display(circle_center_pos[0], circle_center_pos[1])

        # Draw true position (filled red circle)
        cv2.circle(frame, display_true_pos, 5, self.config.TRUE_POS_COLOR, -1) # Filled circle

        # Draw VIO position (filled blue circle)
        cv2.circle(frame, display_vio_pos, 4, self.config.VIO_POS_COLOR, -1) # Filled circle
        
        # Draw confidence circle center (filled green circle, slightly smaller than true pos)
        cv2.circle(frame, display_circle_center_pos, 3, self.config.CIRCLE_COLOR, -1) # Filled circle

    def _draw_info_text(self, frame: np.ndarray, step: int, drone, localizer_state: dict):
        """Draw informative text overlays on the map visualization area.
        
        Args:
            frame: The current video frame (NumPy array) to draw on.
            step: Current simulation step.
            drone: The NewDrone instance.
            localizer_state: Dictionary with current localization state.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        text_color = (255, 255, 255) # White color for text
        outline_color = (0, 0, 0) # Black color for outline
        
        def draw_text_with_outline(text, position, target_frame):
            # Draw black outline
            cv2.putText(target_frame, text, position, font, font_scale, outline_color, text_thickness + 1, cv2.LINE_AA)
            # Draw white text
            cv2.putText(target_frame, text, position, font, font_scale, text_color, text_thickness, cv2.LINE_AA)

        y_offset = 20
        # Display simulation step
        draw_text_with_outline(f"Step: {step}", (self.padding, y_offset), frame)
        y_offset += 20
        
        # Display true position
        true_pos = drone.get_true_position_world()
        draw_text_with_outline(f"True Pos: ({true_pos[0]:.1f}, {true_pos[1]:.1f})m", (self.padding, y_offset), frame)
        y_offset += 20

        # Display VIO position
        vio_pos = drone.get_vio_position_world()
        draw_text_with_outline(f"VIO Pos: ({vio_pos[0]:.1f}, {vio_pos[1]:.1f})m", (self.padding, y_offset), frame)
        y_offset += 20

        # Display position error
        pos_error = drone.get_position_error_m()
        draw_text_with_outline(f"Pos Error: {pos_error:.1f}m", (self.padding, y_offset), frame)
        y_offset += 20

        # Display confidence circle radius
        radius = localizer_state['radius_m']
        draw_text_with_outline(f"Radius: {radius:.1f}m", (self.padding, y_offset), frame)
        y_offset += 20

        # Display number of active patches
        num_patches = len(localizer_state['patch_probabilities'])
        draw_text_with_outline(f"Active Patches: {num_patches}", (self.padding, y_offset), frame)
        y_offset += 20

        # Display accumulated VIO error (epsilon)
        accumulated_epsilon = drone.get_accumulated_vio_data()[1] # Second element is epsilon
        draw_text_with_outline(f"Acc. VIO Error: {accumulated_epsilon:.1f}m", (self.padding, y_offset), frame)
        y_offset += 20

        # Display localization update status
        update_status = "Updated" if drone.distance_traveled_since_last_update >= self.config.UPDATE_INTERVAL_M else "Waiting"
        draw_text_with_outline(f"Localization: {update_status} ({drone.distance_traveled_since_last_update:.1f}/{self.config.UPDATE_INTERVAL_M:.1f}m)", (self.padding, y_offset), frame)
        y_offset += 20

        # Display correction status
        correction_status = "Active" if drone.correction_active else "Idle"
        draw_text_with_outline(f"Correction: {correction_status}", (self.padding, y_offset), frame)
        y_offset += 20

    def _draw_retrieval_panel(self, frame: np.ndarray, localizer_state: dict):
        """Draw the right-side panel showing top-5 retrieval patches and SuperPoint matches.
        
        Args:
            frame: The current video frame (NumPy array) to draw on.
            localizer_state: Dictionary with current localization state.
        """
        panel_x_start = self.map_display_w + self.padding
        panel_y_start = 0
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_thickness = 1
        text_color = (255, 255, 255) # White
        
        def draw_panel_text(text, y_pos, target_frame=frame):
            cv2.putText(target_frame, text, (panel_x_start + 5, y_pos), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

        y_offset = panel_y_start + 20
        draw_panel_text("Retrieval Panel", y_offset)
        y_offset += 30

        # Display SuperPoint matches (if available from the best candidate)
        sp_vis_data = localizer_state.get('last_superpoint_vis_data')
        if sp_vis_data and sp_vis_data.get('best_match_image') is not None:
            # The _draw_superpoint_matches_on_panel function returns an OpenCV image
            sp_match_vis = _draw_superpoint_matches_on_panel(
                localizer_state['last_camera_view'],
                sp_vis_data['best_match_image'],
                sp_vis_data['kpts_camera'],
                sp_vis_data['kpts_patch'],
                sp_vis_data['matches'],
                sp_vis_data['avg_desc_dist'],
                self.panel_width - 10 # Provide some margin
            )
            
            # Place the SuperPoint match visualization onto the main frame
            # Calculate center position for the SP vis in the panel
            sp_vis_h, sp_vis_w = sp_match_vis.shape[:2]
            # Center horizontally within the panel
            sp_vis_x = panel_x_start + (self.panel_width - sp_vis_w) // 2
            sp_vis_y = y_offset # Place it below the title
            
            frame[sp_vis_y : sp_vis_y + sp_vis_h, sp_vis_x : sp_vis_x + sp_vis_w] = sp_match_vis
            y_offset += sp_vis_h + 10 # Move y_offset past the SP viz
        else:
            draw_panel_text("No SuperPoint matches to display.", y_offset)
            y_offset += 20

        y_offset += 10 # Small gap before top 5 patches
        draw_panel_text("Top 5 Candidates:", y_offset)
        y_offset += 20

        # Display top 5 patches with their probabilities and images
        top5_patches = localizer_state.get('last_top5_patches', [])
        
        # Define thumbnail size for patches
        thumbnail_size = 50 # 50x50 pixels
        patch_display_margin = 5 # Margin between patches
        
        for i, patch_info in enumerate(top5_patches):
            patch_coord = patch_info['coord']
            prob = patch_info['prob']
            patch_image_pil = patch_info['image']
            
            if patch_image_pil:
                # Resize patch image for thumbnail display
                patch_thumbnail = patch_image_pil.resize((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
                patch_thumbnail_cv2 = _pil_to_cv2(patch_thumbnail)
                
                # Calculate position for thumbnail in the panel
                thumb_x = panel_x_start + 5 # Start slightly in from panel edge
                thumb_y = y_offset 

                # Ensure thumbnail fits within frame bounds
                if thumb_y + thumbnail_size <= self.display_h and thumb_x + thumbnail_size <= self.display_w:
                    frame[thumb_y : thumb_y + thumbnail_size, thumb_x : thumb_x + thumbnail_size] = patch_thumbnail_cv2

                # Display patch info next to thumbnail
                text_x = thumb_x + thumbnail_size + 5 # Text starts after thumbnail with margin
                text_y = thumb_y + 15 # Vertically align text with thumbnail

                draw_panel_text(f"Grid: {patch_coord}", text_y)
                draw_panel_text(f"Prob: {prob:.2e}", text_y + 15)

                y_offset += thumbnail_size + patch_display_margin # Move y_offset for next patch
            else:
                draw_panel_text(f"Patch {patch_coord}: No image", y_offset)
                y_offset += 30

    def _draw_top5_patches(self, frame: np.ndarray, top5_patches: list):
        """
        Draws outlines around the top N most probable patches on the map to highlight them.

        Args:
            frame: The current video frame (NumPy array) to draw on.
            top5_patches: A list of dictionaries, each containing information about a top patch.
                          Expected to have 'center_world_m' (tuple) and 'coord' (tuple).
        """
        outline_color = (0, 255, 255)  # Yellow color for the outline (BGR format)
        outline_thickness = 2          # Thickness of the outline

        for patch_info in top5_patches:
            grid_row, grid_col = patch_info['coord']
            
            # Calculate the top-left corner of the patch in world coordinates
            x_world_tl, y_world_tl = self.localizer._grid_to_world(grid_row, grid_col)
            
            # Convert patch size from meters to pixels on the original map
            patch_size_px_orig = self.localizer.patch_size_px

            # Calculate top-left and bottom-right corners in world coordinates
            # Adjusting to true top-left for the patch boundary
            x_world_start = x_world_tl - (self.localizer.patch_size_m / 2.0)
            y_world_start = y_world_tl - (self.localizer.patch_size_m / 2.0)
            x_world_end = x_world_start + self.localizer.patch_size_m
            y_world_end = y_world_start + self.localizer.patch_size_m

            # Convert world coordinates of corners to display coordinates
            top_left_display = self._world_to_display(x_world_start, y_world_start)
            bottom_right_display = self._world_to_display(x_world_end, y_world_end)

            # Ensure coordinates are integers and in the correct order for rectangle drawing
            x1, y1 = top_left_display
            x2, y2 = bottom_right_display

            # Draw the rectangle on the map area of the frame
            cv2.rectangle(
                img=frame[:self.map_display_h, :self.map_display_w], # Draw only on the map section
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=outline_color,
                thickness=outline_thickness,
                lineType=cv2.LINE_AA # For anti-aliased lines
            )

    def close(self):
        """
        Release the video writer and close any open visualization windows.
        """
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows() 