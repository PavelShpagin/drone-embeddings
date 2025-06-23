import cv2
import numpy as np

def fastbox3x2(image: np.ndarray) -> np.ndarray:
    """
    Applies a custom 3x3 weighted blur twice to the input 8-bit grayscale image.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("fastbox3x2 requires a single-channel 8-bit image")

    # Define the 3x3 kernel: [[1,2,1],[2,2,2],[1,2,1]] / 14
    kernel = np.array([
        [1, 2, 1],
        [2, 2, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 14.0

    # Apply the filter twice
    blurred = cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_DEFAULT)
    blurred = cv2.filter2D(blurred, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_DEFAULT)
    return blurred


def normalize(image: np.ndarray, contrasting: int = 1, blur_map_iters: int = 1) -> np.ndarray:
    """
    Normalizes contrast of an 8-bit grayscale image with local contrast adjustment.

    Parameters:
    - image: single-channel 8-bit input image
    - contrasting: if <= 0, returns a copy of the original
    - blur_map_iters: number of small-blur iterations at the end
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("normalize requires a single-channel 8-bit image")

    if contrasting <= 0:
        return image.copy()

    # Expand precision to 16-bit (shift left by 8)
    initial = (image.astype(np.uint16) << 8)

    # Local blur with box filter for initial image
    blurred = cv2.boxFilter(
        initial, ddepth=-1, ksize=(31, 31), normalize=True,
        anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT
    )

    # Compute absolute deviation
    deviation = np.abs(initial.astype(np.int32) - blurred.astype(np.int32))
    deviation = np.clip(deviation, 0, 65535).astype(np.uint16)

    # Blur the deviation map locally
    blurred_dev = cv2.boxFilter(
        deviation, ddepth=-1, ksize=(31, 31), normalize=True,
        anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT
    )

    # Local contrast adjustment parameters
    max_contrast = 8
    dest_dev = 40

    # Avoid division by zero by ensuring minimum deviation
    dd = blurred_dev.astype(np.int32)
    dd[dd < 256] = 256

    # Compute per-pixel scaling factor k
    # k = dest_dev * 256 * 256 / dd, clipped to max_contrast * 256
    k = (dest_dev * 256 * 256) / dd
    k = np.minimum(k, max_contrast * 256).astype(np.int32)

    # Compute new pixel values in 16-bit
    d = initial.astype(np.int32) - blurred.astype(np.int32)
    result_16 = (d * k // 256 + 128 * 256).clip(0, 65535).astype(np.uint16)

    # Convert back to 8-bit
    result = (result_16 >> 8).astype(np.uint8)

    # Optional small blur to reduce noise
    out = result
    for _ in range(blur_map_iters):
        out = fastbox3x2(out)

    return out 