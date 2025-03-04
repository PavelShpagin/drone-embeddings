from PIL import Image

def crop_and_resize_azure_image(image_path: str, output_path: str, crop_box: tuple, size: tuple):
    """
    Crop and resize an Azure Maps image to remove watermark.
    
    :param image_path: Path to the input image.
    :param output_path: Path to save the processed image.
    :param crop_box: A tuple (left, upper, right, lower) to define the cropping box.
    :param size: A tuple (width, height) to define the new size.
    """
    with Image.open(image_path) as img:
        cropped_img = img.crop(crop_box)
        resized_img = cropped_img.resize(size, Image.LANCZOS)
        resized_img.save(output_path)

def crop_and_resize_google_image(image_path: str, output_path: str, crop_box: tuple, size: tuple):
    """
    Crop and resize a Google Maps image to remove watermark.
    
    :param image_path: Path to the input image.
    :param output_path: Path to save the processed image.
    :param crop_box: A tuple (left, upper, right, lower) to define the cropping box.
    :param size: A tuple (width, height) to define the new size.
    """
    with Image.open(image_path) as img:
        cropped_img = img.crop(crop_box)
        resized_img = cropped_img.resize(size, Image.LANCZOS)
        resized_img.save(output_path) 