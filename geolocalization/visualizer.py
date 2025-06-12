import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt

class SimulationVisualizer:
    """
    Handles the creation of dynamic, zoomed-in visualization frames for the localization simulation.
    """
    def __init__(self, map_image: Image.Image, m_per_pixel: float, patch_size_px: int, output_path: str, config):
        self.base_map = map_image.copy()
        self.m_per_pixel = m_per_pixel
        self.patch_size_px = patch_size_px
        self.config = config
        self.output_resolution = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, self.output_resolution)
        self.is_open = self.video_writer.isOpened()
        if not self.is_open: print("\n--- WARNING: Could not open video writer. No video will be saved. ---\n")

        self.true_path_history = []
        self.vio_path_history = []
        self.estimated_path_history = []
        
        try:
            self.font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            self.font = ImageFont.load_default()

    def _calculate_dynamic_viewport(self, state) -> tuple:
        """Calculates a tight viewport centered on the estimated position."""
        # The camera will be centered on the filter's most likely position
        center_x, center_y = state.get_most_likely_position_px()
        
        # The zoom level is determined by the confidence radius
        # We'll make the view 20x the radius, which gives a wide zoom with lots of context
        view_width_m = state.radius_m * 20
        view_height_m = view_width_m * (self.output_resolution[1] / self.output_resolution[0]) # Maintain aspect ratio
        
        view_w_px = view_width_m / self.m_per_pixel
        view_h_px = view_height_m / self.m_per_pixel

        # Define the crop box, clamping to the map boundaries
        left = max(0, center_x - view_w_px / 2)
        top = max(0, center_y - view_h_px / 2)
        right = min(self.base_map.width, left + view_w_px)
        bottom = min(self.base_map.height, top + view_h_px)
        
        return int(left), int(top), int(right), int(bottom)

    def draw_frame(self, state, drone, step: int):
        """Draws a single, dynamically zoomed frame of the simulation."""
        if not self.is_open: return

        self.true_path_history.append(drone.get_true_position_px())
        self.vio_path_history.append(drone.get_vio_position_px())
        self.estimated_path_history.append(state.get_most_likely_position_px())
        
        l, t, r, b = self._calculate_dynamic_viewport(state)
        viewport_frame = self.base_map.crop((l, t, r, b))
        draw = ImageDraw.Draw(viewport_frame, 'RGBA')

        def to_viewport_coords(p):
            return (p[0] - l, p[1] - t)

        # Draw Heatmap (only non-zero probabilities)
        coords = state.get_active_patches()
        if coords.size > 0:
            probs = state.prob_grid[coords[:, 0], coords[:, 1]]
            max_prob = probs.max()
            if max_prob > 0:
                normalized_probs = probs / max_prob
                cmap = plt.get_cmap('viridis')
                colors = cmap(normalized_probs)
                
                for i, (gy, gx) in enumerate(coords):
                    px_x, px_y = gx * self.patch_size_px, gy * self.patch_size_px
                    tl = to_viewport_coords((px_x, px_y))
                    br = to_viewport_coords((px_x + self.patch_size_px, px_y + self.patch_size_px))
                    
                    color = tuple(int(c * 255) for c in colors[i])
                    draw.rectangle([tl, br], fill=color[:3] + (int(normalized_probs[i] * 200),))

        # Draw Confidence Circle
        radius_px = state.radius_m / self.m_per_pixel
        cx, cy = to_viewport_coords((state.center_x_px, state.center_y_px))
        draw.ellipse([cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px], outline='yellow', width=3)

        # Draw Paths
        if len(self.true_path_history) > 1:
            draw.line([to_viewport_coords(p) for p in self.true_path_history], fill='lime', width=3)
            draw.line([to_viewport_coords(p) for p in self.vio_path_history], fill='cyan', width=3)
            draw.line([to_viewport_coords(p) for p in self.estimated_path_history], fill='magenta', width=4, joint='curve')
            
        # Draw current positions
        true_pos, vio_pos, est_pos = [to_viewport_coords(p[-1]) for p in [self.true_path_history, self.vio_path_history, self.estimated_path_history]]
        draw.ellipse([true_pos[0]-8, true_pos[1]-8, true_pos[0]+8, true_pos[1]+8], fill='lime', outline='black')
        draw.ellipse([vio_pos[0]-8, vio_pos[1]-8, vio_pos[0]+8, vio_pos[1]+8], fill='cyan', outline='black')
        draw.ellipse([est_pos[0]-8, est_pos[1]-8, est_pos[0]+8, est_pos[1]+8], fill='magenta', outline='black')
        
        final_frame = viewport_frame.resize(self.output_resolution, Image.Resampling.LANCZOS)
        self.video_writer.write(cv2.cvtColor(np.array(final_frame), cv2.COLOR_RGB2BGR))

    def close(self):
        if self.is_open:
            self.video_writer.release()
            print("Visualization video saved.") 