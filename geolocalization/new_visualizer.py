import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class NewSimulationVisualizer:
    """
    Visualizes the new global localization algorithm simulation.
    Shows confidence circle, probability map, true/VIO positions, and trajectories.
    """
    def __init__(self, config, localizer, output_video_path: str):
        self.config = config
        self.localizer = localizer
        self.output_video_path = output_video_path
        
        # Visualization parameters
        self.viz_scale = 0.3  # Scale down map for display
        self.map_image = localizer.map_image
        self.display_w = int(localizer.map_w * self.viz_scale)
        self.display_h = int(localizer.map_h * self.viz_scale)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            output_video_path, fourcc, 10.0, (self.display_w, self.display_h)
        )
        
        # Trajectory history
        self.true_trajectory = []
        self.vio_trajectory = []
        self.circle_centers = []
        
        print(f"Visualizer initialized. Output video: {output_video_path}")
        print(f"Display size: {self.display_w}x{self.display_h}")

    def _world_to_display(self, x_m: float, y_m: float) -> tuple:
        """Convert world coordinates to display pixel coordinates."""
        # Convert to full-scale pixel coordinates first
        x_px = x_m / self.localizer.m_per_pixel + self.localizer.map_w // 2
        y_px = y_m / self.localizer.m_per_pixel + self.localizer.map_h // 2
        
        # Scale down for display
        display_x = int(x_px * self.viz_scale)
        display_y = int(y_px * self.viz_scale)
        
        return (display_x, display_y)

    def draw_frame(self, drone, step: int):
        """Draw a single frame of the simulation, including retrieval panel."""
        # Start with scaled map image
        base_image = self.map_image.resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        frame = np.array(base_image)
        
        # Get current state
        localizer_state = self.localizer.get_state_info()
        true_pos = drone.get_true_position_world()
        vio_pos = drone.get_vio_position_world()
        
        # Update trajectory history
        self.true_trajectory.append(true_pos)
        self.vio_trajectory.append(vio_pos)
        self.circle_centers.append(localizer_state['center_world_m'])
        
        # Keep trajectory history manageable
        max_history = 500
        if len(self.true_trajectory) > max_history:
            self.true_trajectory = self.true_trajectory[-max_history:]
            self.vio_trajectory = self.vio_trajectory[-max_history:]
            self.circle_centers = self.circle_centers[-max_history:]
        
        # Draw probability heatmap
        self._draw_probability_heatmap(frame, localizer_state['patch_probabilities'])
        
        # Draw confidence circle
        self._draw_confidence_circle(frame, localizer_state)
        
        # Draw trajectories
        self._draw_trajectories(frame)
        
        # Draw current positions
        self._draw_positions(frame, true_pos, vio_pos)
        
        # Draw info text
        self._draw_info_text(frame, step, drone, localizer_state)
        
        # Draw retrieval panel (top right)
        self._draw_retrieval_panel(frame)
        
        # Write frame to video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame_bgr)

    def _draw_probability_heatmap(self, frame: np.ndarray, patch_probabilities: dict):
        """Draw probability heatmap for active patches."""
        if not patch_probabilities:
            return
        
        # Create overlay
        overlay = frame.copy()
        
        # Get probability statistics
        probs = list(patch_probabilities.values())
        min_prob = min(probs) if probs else 0.0
        max_prob = max(probs) if probs else 1.0
        avg_prob = np.mean(probs) if probs else 0.0
        
        print(f"Prob stats: min={min_prob:.6f}, max={max_prob:.6f}, avg={avg_prob:.6f}")
        
        # Font for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_thickness = 1
        
        for (grid_row, grid_col), prob in patch_probabilities.items():
            # Convert grid coordinates to world coordinates
            patch_center = self.localizer._grid_to_world(grid_row, grid_col)
            display_center = self._world_to_display(patch_center[0], patch_center[1])
            
            # Calculate patch size in display coordinates
            patch_size_display = int(self.localizer.patch_size_px * self.viz_scale)
            
            # Normalize probability (0-1)
            if max_prob > min_prob:
                normalized_prob = (prob - min_prob) / (max_prob - min_prob)
            else:
                normalized_prob = 0.5
            
            # Color mapping: blue (low) -> green (medium) -> red (high)
            if normalized_prob < 0.5:
                # Blue to green
                blue = int(255 * (1 - 2 * normalized_prob))
                green = int(255 * 2 * normalized_prob)
                red = 0
            else:
                # Green to red
                blue = 0
                green = int(255 * (2 - 2 * normalized_prob))
                red = int(255 * (2 * normalized_prob - 1))
            
            color = (red, green, blue)  # RGB
            
            # Draw patch rectangle
            top_left = (display_center[0] - patch_size_display // 2,
                       display_center[1] - patch_size_display // 2)
            bottom_right = (display_center[0] + patch_size_display // 2,
                           display_center[1] + patch_size_display // 2)
            
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
            
            # Draw probability value as text (scientific notation)
            prob_text = f"{prob:.2e}"
            text_size = cv2.getTextSize(prob_text, font, font_scale, text_thickness)[0]
            text_pos = (display_center[0] - text_size[0] // 2,
                       display_center[1] + text_size[1] // 2)
            
            # Draw text with white background for readability
            cv2.rectangle(overlay, 
                         (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                         (text_pos[0] + text_size[0] + 2, text_pos[1] + 2),
                         (255, 255, 255), -1)
            cv2.putText(overlay, prob_text, text_pos, font, font_scale, (0, 0, 0), text_thickness)
        
        # Blend overlay with frame
        cv2.addWeighted(frame, 0.6, overlay, 0.4, 0, frame)

    def _draw_confidence_circle(self, frame: np.ndarray, localizer_state: dict):
        """Draw the confidence circle."""
        center_display = self._world_to_display(
            localizer_state['center_world_m'][0],
            localizer_state['center_world_m'][1]
        )
        
        radius_display = int(localizer_state['radius_m'] / self.localizer.m_per_pixel * self.viz_scale)
        
        # Draw circle outline
        cv2.circle(frame, center_display, radius_display, (0, 255, 0), 2)  # Green circle

    def _draw_trajectories(self, frame: np.ndarray):
        """Draw trajectory lines."""
        # Draw true trajectory (red)
        if len(self.true_trajectory) > 1:
            points = [self._world_to_display(pos[0], pos[1]) for pos in self.true_trajectory]
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)
        
        # Draw VIO trajectory (blue)
        if len(self.vio_trajectory) > 1:
            points = [self._world_to_display(pos[0], pos[1]) for pos in self.vio_trajectory]
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 0, 255), 2)
        
        # Draw circle center trajectory (gray)
        if len(self.circle_centers) > 1:
            points = [self._world_to_display(pos[0], pos[1]) for pos in self.circle_centers]
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (128, 128, 128), 1)

    def _draw_positions(self, frame: np.ndarray, true_pos: tuple, vio_pos: tuple):
        """Draw current drone positions."""
        # True position (red circle)
        true_display = self._world_to_display(true_pos[0], true_pos[1])
        cv2.circle(frame, true_display, 8, (255, 0, 0), -1)
        cv2.circle(frame, true_display, 8, (255, 255, 255), 2)
        
        # VIO position (blue circle)
        vio_display = self._world_to_display(vio_pos[0], vio_pos[1])
        cv2.circle(frame, vio_display, 6, (0, 0, 255), -1)
        cv2.circle(frame, vio_display, 6, (255, 255, 255), 2)

    def _draw_info_text(self, frame: np.ndarray, step: int, drone, localizer_state: dict):
        """Draw information text overlay."""
        y_offset = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # Text with black outline for better visibility
        def draw_text_with_outline(text, position):
            # Black outline
            cv2.putText(frame, text, position, font, font_scale, (0, 0, 0), thickness + 1)
            # White text
            cv2.putText(frame, text, position, font, font_scale, color, thickness)
        
        info_lines = [
            f"Step: {step}",
            f"Circle Radius: {localizer_state['radius_m']:.1f}m",
            f"Active Patches: {localizer_state['num_patches']}",
            f"Max Probability: {localizer_state['max_probability']:.6f}",
            f"Position Error: {drone.get_position_error_m():.1f}m",
            f"Distance Traveled: {drone.total_distance_traveled:.1f}m"
        ]
        
        for i, line in enumerate(info_lines):
            position = (10, y_offset + i * 25)
            draw_text_with_outline(line, position)
        
        # Legend
        legend_y = frame.shape[0] - 100
        draw_text_with_outline("Legend:", (10, legend_y))
        draw_text_with_outline("Red: True Position", (10, legend_y + 20))
        draw_text_with_outline("Blue: VIO Position", (10, legend_y + 40))
        draw_text_with_outline("Green: Confidence Circle", (10, legend_y + 60))
        draw_text_with_outline("Yellow-Red: Probability", (10, legend_y + 80))

    def _draw_retrieval_panel(self, frame: np.ndarray):
        """Draw the current camera view and top-5 retrievals in a panel on the frame."""
        # Panel parameters
        panel_w = 300
        panel_h = 400
        margin = 10
        thumb_size = 64
        spacing = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        text_thickness = 1
        
        # Top right corner
        x0 = frame.shape[1] - panel_w - margin
        y0 = margin
        
        # Draw panel background
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        
        y_cursor = y0 + 10
        x_cursor = x0 + 10
        
        # Draw camera view
        if hasattr(self.localizer, 'last_camera_view') and self.localizer.last_camera_view is not None:
            cam_img = self.localizer.last_camera_view.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            cam_img_np = np.array(cam_img)
            frame[y_cursor:y_cursor+thumb_size, x_cursor:x_cursor+thumb_size, :] = cam_img_np
            cv2.putText(frame, 'Drone View', (x_cursor + thumb_size + 8, y_cursor + 32), font, font_scale, (255,255,255), text_thickness)
        
        y_cursor += thumb_size + spacing
        
        # Draw top-5 retrievals
        if hasattr(self.localizer, 'last_top5_patches') and self.localizer.last_top5_patches:
            for i, patch in enumerate(self.localizer.last_top5_patches):
                patch_img = patch['image'].resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                patch_img_np = np.array(patch_img)
                frame[y_cursor:y_cursor+thumb_size, x_cursor:x_cursor+thumb_size, :] = patch_img_np
                label = f"#{i+1} ({patch['coord'][0]},{patch['coord'][1]})"
                prob = patch['prob']
                cv2.putText(frame, label, (x_cursor + thumb_size + 8, y_cursor + 24), font, font_scale, (200,255,200), text_thickness)
                cv2.putText(frame, f"p={prob:.2e}", (x_cursor + thumb_size + 8, y_cursor + 48), font, font_scale, (255,255,0), text_thickness)
                y_cursor += thumb_size + spacing
        
        # Panel border
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (200, 200, 200), 2)

    def close(self):
        """Close the video writer."""
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved to: {self.output_video_path}") 