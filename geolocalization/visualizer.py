import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class SimulationVisualizer:
    """
    Handles the creation of dynamic, zoomed-in visualization frames for the 
    probabilistic localization simulation.
    """
    def __init__(self, database, output_resolution=(1920, 1080), max_fps=20):
        self.database = database
        self.output_resolution = output_resolution
        self.max_fps = max_fps
        
        # Font setup for information overlay
        font_size = 20
        try:
            self.small_font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            self.small_font = ImageFont.load_default()
        
        # Viewport tracking (never shrink, only grow to show full circle)
        self.min_viewport_size = 1000  # Minimum viewport size in pixels
        self.max_viewport_size = max(4000, self.database.map_w, self.database.map_h)  # Never exceed map size
        self.current_viewport_size = self.min_viewport_size
        
        # Video writing setup
        self.frames = []
        self.is_recording = True
        
        # Path history for trajectory visualization
        self.true_path_history = []
        self.vio_path_history = []
        self.confidence_centers_history = []
        self.corrections_history = []
        
        # Video writer setup
        self.is_open = False
        self.video_writer = None
        
        # Color maps for probability visualization
        self.prob_colormap = LinearSegmentedColormap.from_list(
            'probability', ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000'], N=256
        )
        
        print(f"Visualizer initialized for {output_resolution}")

    def _calculate_viewport(self, center_m, radius_m):
        """Calculate viewport to show the full confidence circle, never shrinking."""
        # Calculate required viewport to show full circle
        required_diameter_m = 2 * radius_m + 200  # Add 200m padding
        required_diameter_px = int(required_diameter_m / self.database.m_per_pixel)
        
        # Update viewport size (never shrink)
        self.current_viewport_size = max(self.current_viewport_size, required_diameter_px, self.min_viewport_size)
        self.current_viewport_size = min(self.current_viewport_size, self.max_viewport_size)
        
        # Convert center to pixel coordinates
        center_px = (
            int(center_m[0] / self.database.m_per_pixel + self.database.map_w // 2),
            int(-center_m[1] / self.database.m_per_pixel + self.database.map_h // 2)
        )
        
        # Calculate viewport bounds
        half_size = self.current_viewport_size // 2
        l = max(0, center_px[0] - half_size)
        t = max(0, center_px[1] - half_size)
        r = min(self.database.map_w, center_px[0] + half_size)
        b = min(self.database.map_h, center_px[1] + half_size)
        
        # Fix coordinate inversion: ensure l < r and t < b
        if l >= r:
            mid_x = (l + r) // 2
            l = max(0, mid_x - self.current_viewport_size // 2)
            r = min(self.database.map_w, l + self.current_viewport_size)
        if t >= b:
            mid_y = (t + b) // 2
            t = max(0, mid_y - self.current_viewport_size // 2)
            b = min(self.database.map_h, t + self.current_viewport_size)
        
        # Final safety check to ensure valid coordinates
        l = max(0, min(l, self.database.map_w - 1))
        t = max(0, min(t, self.database.map_h - 1))
        r = max(l + 1, min(r, self.database.map_w))
        b = max(t + 1, min(b, self.database.map_h))
        
        return l, t, r, b

    def draw_frame(self, localization_state, drone, step, correction_triggered=False):
        """Draw a single frame of the simulation."""
        try:
            # Calculate viewport bounds
            l, t, r, b = self._calculate_viewport(localization_state.center_world_m, localization_state.radius_m)
            
            # Ensure coordinates are valid for PIL crop
            if l >= r or t >= b:
                print(f"Warning: Invalid viewport coordinates ({l},{t},{r},{b}) at step {step}")
                # Return a black frame on error
                error_frame = Image.new("RGB", self.output_resolution, (0, 0, 0))
                self.frames.append(error_frame)
                return error_frame
            
            # Extract viewport from map
            viewport_frame = self.database.map_image.crop((l, t, r, b))
            
            # Convert to RGB and resize to output resolution
            viewport_frame = viewport_frame.convert("RGB")
            viewport_frame = viewport_frame.resize(self.output_resolution, Image.LANCZOS)
            
            # Create draw context
            draw = ImageDraw.Draw(viewport_frame)
            
            # Calculate scaling factors for drawing overlays
            scale_x = self.output_resolution[0] / (r - l)
            scale_y = self.output_resolution[1] / (b - t)
            
            # Draw overlays
            self._draw_confidence_circle(draw, localization_state, l, t, scale_x, scale_y)
            self._draw_drone_positions(draw, drone, l, t, scale_x, scale_y)
            self._draw_patches(draw, localization_state, l, t, scale_x, scale_y)
            
            # Add information overlay
            self._add_information_overlay(draw, localization_state, drone, step)
            
            # Store frame for video
            self.frames.append(viewport_frame.copy())
            
            return viewport_frame
            
        except Exception as e:
            print(f"Error drawing frame {step}: {e}")
            # Return a black frame on error
            error_frame = Image.new("RGB", self.output_resolution, (0, 0, 0))
            self.frames.append(error_frame)
            return error_frame

    def _draw_probability_heatmap(self, draw, localization_state, viewport_left, viewport_top):
        """Draw probability heatmap for active patches."""
        coords, probs, _, _ = localization_state.get_active_patches_for_visualization()
        
        if len(coords) == 0:
            return
        
        # Normalize probabilities for color mapping
        max_prob = np.max(probs) if len(probs) > 0 else 1.0
        normalized_probs = probs / max_prob if max_prob > 0 else probs
        
        # Draw each probability patch
        for i, coord in enumerate(coords):
            # Ensure coord is a tuple/list of two values
            if isinstance(coord, np.ndarray):
                coord = tuple(coord)
            
            grid_row, grid_col = coord[0], coord[1]
            
            # Calculate patch bounds directly from grid coordinates
            # Convert grid coordinates to pixel coordinates
            center_x_px = (grid_col + 0.5) * self.database.patch_size_px
            center_y_px = (grid_row + 0.5) * self.database.patch_size_px
            
            half_patch = self.database.patch_size_px // 2
            
            # Calculate viewport coordinates
            left_vp = center_x_px - half_patch - viewport_left
            top_vp = center_y_px - half_patch - viewport_top
            right_vp = center_x_px + half_patch - viewport_left
            bottom_vp = center_y_px + half_patch - viewport_top
            
            # Get color from colormap
            color_rgba = self.prob_colormap(normalized_probs[i])
            color = tuple(int(c * 255) for c in color_rgba[:3])
            alpha = int(normalized_probs[i] * 150 + 50)  # Semi-transparent
            
            # Draw probability patch
            try:
                draw.rectangle([left_vp, top_vp, right_vp, bottom_vp], 
                             fill=color + (alpha,), outline=color, width=1)
            except Exception as e:
                # Skip patches that cause drawing errors
                continue

    def _draw_confidence_circle(self, draw, localization_state, viewport_left, viewport_top, scale_x, scale_y):
        """Draw the confidence circle."""
        center_world = localization_state.center_world_m
        radius_m = localization_state.radius_m
        
        # Convert center to pixel coordinates within viewport
        center_px_map = self.database.world_to_pixel(center_world[0], center_world[1])
        center_x = (center_px_map[0] - viewport_left) * scale_x
        center_y = (center_px_map[1] - viewport_top) * scale_y
        
        # Convert radius to pixels
        radius_px = radius_m / self.database.m_per_pixel * scale_x
        
        # Draw circle
        draw.ellipse([
            center_x - radius_px, center_y - radius_px,
            center_x + radius_px, center_y + radius_px
        ], outline="yellow", width=3)

    def _draw_drone_positions(self, draw, drone, viewport_left, viewport_top, scale_x, scale_y):
        """Draw drone positions (true and VIO)."""
        # True position
        true_pos = drone.get_true_position_world()
        true_px = self.database.world_to_pixel(true_pos[0], true_pos[1])
        true_x = (true_px[0] - viewport_left) * scale_x
        true_y = (true_px[1] - viewport_top) * scale_y
        
        # VIO position
        vio_pos = drone.get_vio_position_world()
        vio_px = self.database.world_to_pixel(vio_pos[0], vio_pos[1])
        vio_x = (vio_px[0] - viewport_left) * scale_x
        vio_y = (vio_px[1] - viewport_top) * scale_y
        
        # Draw positions
        marker_size = 8
        # True position (green)
        draw.ellipse([
            true_x - marker_size, true_y - marker_size,
            true_x + marker_size, true_y + marker_size
        ], fill="green", outline="darkgreen")
        
        # VIO position (red)
        draw.ellipse([
            vio_x - marker_size, vio_y - marker_size,
            vio_x + marker_size, vio_y + marker_size
        ], fill="red", outline="darkred")

    def _draw_patches(self, draw, localization_state, viewport_left, viewport_top, scale_x, scale_y):
        """Draw probability patches with colors."""
        if not localization_state.patch_probabilities:
            return
            
        # Find max probability for normalization
        max_prob = max(localization_state.patch_probabilities.values()) if localization_state.patch_probabilities else 1.0
        
        patches_drawn = 0
        for patch_coord, probability in localization_state.patch_probabilities.items():
            if probability < 0.001:  # Skip very low probability patches
                continue
                
            # Get patch center using the database method
            patch_center_world = self.database.get_patch_center_world(patch_coord)
            if patch_center_world is None:
                continue
                
            # Convert to viewport coordinates
            patch_px = self.database.world_to_pixel(patch_center_world[0], patch_center_world[1])
            
            # Check if patch is within viewport
            if (patch_px[0] < viewport_left or patch_px[0] > viewport_left + (self.output_resolution[0] / scale_x) or
                patch_px[1] < viewport_top or patch_px[1] > viewport_top + (self.output_resolution[1] / scale_y)):
                continue
                
            patch_x = (patch_px[0] - viewport_left) * scale_x
            patch_y = (patch_px[1] - viewport_top) * scale_y
            
            # Color based on probability (normalized)
            normalized_prob = probability / max_prob
            if normalized_prob > 0.7:
                color = "lime"  # Bright green for high probability
                size = 8
            elif normalized_prob > 0.3:
                color = "yellow"
                size = 6
            elif normalized_prob > 0.1:
                color = "orange"
                size = 4
            else:
                color = "red"
                size = 3
            
            # Draw patch with size based on probability
            try:
                draw.ellipse([
                    patch_x - size, patch_y - size,
                    patch_x + size, patch_y + size
                ], fill=color, outline="black", width=1)
                patches_drawn += 1
            except Exception as e:
                # Skip patches that cause drawing errors
                continue
        
        # Debug info
        if patches_drawn == 0:
            print(f"Warning: No patches drawn in viewport (total patches: {len(localization_state.patch_probabilities)})")

    def _draw_trajectory_paths(self, draw, viewport_left, viewport_top):
        """Draw the trajectory paths for true and VIO positions."""
        def world_to_viewport(world_pos):
            px_pos = self.database.world_to_pixel(world_pos[0], world_pos[1])
            return (px_pos[0] - viewport_left, px_pos[1] - viewport_top)
        
        # Draw true path
        if len(self.true_path_history) > 1:
            true_points = [world_to_viewport(pos) for pos in self.true_path_history[-50:]]  # Last 50 points
            if len(true_points) > 1:
                for i in range(len(true_points) - 1):
                    draw.line([true_points[i], true_points[i + 1]], fill='lime', width=3)
        
        # Draw VIO path
        if len(self.vio_path_history) > 1:
            vio_points = [world_to_viewport(pos) for pos in self.vio_path_history[-50:]]  # Last 50 points
            if len(vio_points) > 1:
                for i in range(len(vio_points) - 1):
                    draw.line([vio_points[i], vio_points[i + 1]], fill='cyan', width=3)
        
        # Draw confidence center path
        if len(self.confidence_centers_history) > 1:
            center_points = [world_to_viewport(pos) for pos in self.confidence_centers_history[-50:]]
            if len(center_points) > 1:
                for i in range(len(center_points) - 1):
                    draw.line([center_points[i], center_points[i + 1]], fill='orange', width=2)

    def _draw_current_positions(self, draw, drone, localization_state, viewport_left, viewport_top):
        """Draw current positions of drone and estimated position."""
        def world_to_viewport(world_pos):
            px_pos = self.database.world_to_pixel(world_pos[0], world_pos[1])
            return (px_pos[0] - viewport_left, px_pos[1] - viewport_top)
        
        # True position (green)
        true_pos_vp = world_to_viewport(drone.get_true_position_world())
        draw.ellipse([true_pos_vp[0] - 8, true_pos_vp[1] - 8, 
                     true_pos_vp[0] + 8, true_pos_vp[1] + 8], 
                    fill='lime', outline='black', width=2)
        
        # VIO position (cyan)
        vio_pos_vp = world_to_viewport(drone.get_vio_position_world())
        draw.ellipse([vio_pos_vp[0] - 8, vio_pos_vp[1] - 8, 
                     vio_pos_vp[0] + 8, vio_pos_vp[1] + 8], 
                    fill='cyan', outline='black', width=2)
        
        # Most confident position (magenta)
        confident_pos = localization_state.get_most_confident_position()
        confident_pos_vp = world_to_viewport(confident_pos)
        draw.ellipse([confident_pos_vp[0] - 10, confident_pos_vp[1] - 10, 
                     confident_pos_vp[0] + 10, confident_pos_vp[1] + 10], 
                    fill='magenta', outline='white', width=2)

    def _draw_correction_indicator(self, draw, drone, localization_state, viewport_left, viewport_top):
        """Draw visual indicator when correction is applied."""
        def world_to_viewport(world_pos):
            px_pos = self.database.world_to_pixel(world_pos[0], world_pos[1])
            return (px_pos[0] - viewport_left, px_pos[1] - viewport_top)
        
        # Draw arrow from VIO position to confident position
        vio_pos_vp = world_to_viewport(drone.get_vio_position_world())
        confident_pos_vp = world_to_viewport(localization_state.get_most_confident_position())
        
        # Draw correction arrow
        draw.line([vio_pos_vp, confident_pos_vp], fill='red', width=4)
        
        # Draw arrowhead
        dx = confident_pos_vp[0] - vio_pos_vp[0]
        dy = confident_pos_vp[1] - vio_pos_vp[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            unit_x, unit_y = dx / length, dy / length
            arrow_size = 15
            arrow_p1 = (confident_pos_vp[0] - arrow_size * unit_x + arrow_size * unit_y / 2,
                       confident_pos_vp[1] - arrow_size * unit_y - arrow_size * unit_x / 2)
            arrow_p2 = (confident_pos_vp[0] - arrow_size * unit_x - arrow_size * unit_y / 2,
                       confident_pos_vp[1] - arrow_size * unit_y + arrow_size * unit_x / 2)
            draw.polygon([confident_pos_vp, arrow_p1, arrow_p2], fill='red')

    def _add_information_overlay(self, draw, localization_state, drone, step):
        """Add text information overlay to the frame."""
        # Get metrics
        metrics = localization_state.get_confidence_metrics()
        position_error = drone.get_position_error_m()
        
        # Get radius from either metrics or localization state directly
        current_radius = metrics.get('radius_m', localization_state.radius_m)
        
        # Get VIO drift information
        vio_bias = drone.get_vio_bias()
        exploration_stats = drone.get_exploration_stats()
        
        # Prepare information lines
        info_lines = [
            f"Step: {step}",
            f"Distance: {exploration_stats['total_distance']:.0f}m",
            f"Radius: {current_radius:.1f}m",
            f"Position Error: {position_error:.1f}m",
            f"VIO Bias: ({vio_bias[0]:.2f}, {vio_bias[1]:.2f})",
            f"Row: {exploration_stats['current_row']} {'→' if exploration_stats['going_right'] else '←'}",
            f"Active Patches: {len([p for p in localization_state.patch_probabilities.values() if p > 0.001])}"
        ]
        
        # Draw information box
        y_offset = 20
        for line in info_lines:
            draw.text((20, y_offset), line, fill="white", font=self.small_font)
            y_offset += 25

    def close(self):
        """Close the visualization and finalize outputs."""
        print(f"Visualization complete with {len(self.frames)} frames")
        
        # Generate summary statistics
        if self.true_path_history and self.vio_path_history:
            self._generate_summary_plot()
        
        print("Visualization closed successfully.")

    def _generate_summary_plot(self):
        """Generate a summary plot of the entire simulation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Full trajectory overview
        true_x = [pos[0] for pos in self.true_path_history]
        true_y = [pos[1] for pos in self.true_path_history]
        vio_x = [pos[0] for pos in self.vio_path_history]
        vio_y = [pos[1] for pos in self.vio_path_history]
        
        ax1.plot(true_x, true_y, 'g-', label='True Path', linewidth=2)
        ax1.plot(vio_x, vio_y, 'c-', label='VIO Path', linewidth=2)
        
        # Mark corrections
        for correction_idx in self.corrections_history:
            if correction_idx < len(self.true_path_history):
                ax1.plot(true_x[correction_idx], true_y[correction_idx], 'ro', markersize=8, label='Correction' if correction_idx == self.corrections_history[0] else "")
        
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('Complete Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Position error over time
        errors = []
        for i in range(len(self.true_path_history)):
            if i < len(self.vio_path_history):
                true_pos = np.array(self.true_path_history[i])
                vio_pos = np.array(self.vio_path_history[i])
                error = np.linalg.norm(true_pos - vio_pos)
                errors.append(error)
        
        ax2.plot(errors, 'b-', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position Error (meters)')
        ax2.set_title('Position Error Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Mark corrections on error plot
        for correction_idx in self.corrections_history:
            if correction_idx < len(errors):
                ax2.axvline(x=correction_idx, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('simulation_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Summary plot saved as simulation_summary.png")

    def save_video(self, filename: str = "geolocalization_simulation.mp4"):
        """Save the visualization as a video with 2-minute target duration."""
        if not self.frames:
            print("No frames to save!")
            return
            
        # Calculate frame rate for approximately 2-minute video
        total_frames = len(self.frames)
        target_duration_seconds = 120  # 2 minutes
        fps = max(10, total_frames // target_duration_seconds)  # At least 10 FPS for smooth playback
        actual_duration = total_frames / fps
        
        print(f"Saving video: {total_frames} frames at {fps} FPS for {actual_duration:.1f}s duration")
        
        # Get frame dimensions from PIL image
        frame = self.frames[0]
        width, height = frame.size
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for {filename}")
            return
        
        # Write frames
        print(f"Writing {total_frames} frames...")
        for i, frame in enumerate(self.frames):
            # Convert PIL Image to numpy array
            frame_array = np.array(frame)
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{total_frames} frames...")
        
        out.release()
        print(f"Video saved as {filename}")
        print(f"Final video: {total_frames} frames, {fps} FPS, {actual_duration:.1f}s duration") 