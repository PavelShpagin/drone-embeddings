import torch
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from PIL import Image
import numpy as np

class SuperPoint:
    def __init__(self, device='cuda'):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint").to(self.device)

    def detect(self, image: np.ndarray):
        """
        Detect keypoints and descriptors in a single image (HWC, uint8, RGB or grayscale).
        Returns: keypoints (N,2), scores (N,), descriptors (N,256)
        """
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        pil_img = Image.fromarray(image)
        inputs = self.processor(pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Post-process to get keypoints, scores, descriptors
        image_size = [(image.shape[0], image.shape[1])]
        results = self.processor.post_process_keypoint_detection(outputs, image_size)[0]
        return results["keypoints"], results["scores"], results["descriptors"] 