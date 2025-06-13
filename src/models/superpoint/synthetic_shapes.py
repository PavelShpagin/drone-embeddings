import numpy as np
import cv2
import os
from tqdm import tqdm

SHAPE_TYPES = ['triangle', 'rectangle', 'circle', 'line', 'ellipse']


def draw_shape(img, shape_type):
    h, w = img.shape
    color = 255
    thickness = np.random.randint(1, 3)
    if shape_type == 'triangle':
        pts = np.random.randint(0, min(h, w), (3, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    elif shape_type == 'rectangle':
        pt1 = tuple(np.random.randint(0, min(h, w), 2))
        pt2 = tuple(np.random.randint(0, min(h, w), 2))
        cv2.rectangle(img, pt1, pt2, color, thickness)
    elif shape_type == 'circle':
        center = tuple(np.random.randint(0, min(h, w), 2))
        radius = np.random.randint(5, min(h, w)//4)
        cv2.circle(img, center, radius, color, thickness)
    elif shape_type == 'line':
        pt1 = tuple(np.random.randint(0, min(h, w), 2))
        pt2 = tuple(np.random.randint(0, min(h, w), 2))
        cv2.line(img, pt1, pt2, color, thickness)
    elif shape_type == 'ellipse':
        center = tuple(np.random.randint(0, min(h, w), 2))
        axes = (np.random.randint(5, min(h, w)//4), np.random.randint(5, min(h, w)//4))
        angle = np.random.randint(0, 180)
        cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness)
    return img

def generate_synthetic_shapes(output_dir, n_images=10000, img_size=128, min_shapes=1, max_shapes=5):
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(n_images)):
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        n_shapes = np.random.randint(min_shapes, max_shapes+1)
        for _ in range(n_shapes):
            shape_type = np.random.choice(SHAPE_TYPES)
            img = draw_shape(img, shape_type)
        cv2.imwrite(os.path.join(output_dir, f"synthetic_{i:06d}.png"), img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='synthetic_shapes')
    parser.add_argument('--n_images', type=int, default=10000)
    parser.add_argument('--img_size', type=int, default=128)
    args = parser.parse_args()
    generate_synthetic_shapes(args.output_dir, args.n_images, args.img_size) 