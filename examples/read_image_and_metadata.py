import json
import os
from PIL import Image
from PIL.ExifTags import TAGS

def get_first_image_in_folder(folder_path="real_data"):
    """Get the first image file in the specified folder."""
    # Get all files in the folder
    files = os.listdir(folder_path)
    
    # Filter for image files (common extensions)
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_files = [f for f in files if f.lower().endswith(image_extensions)]
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {folder_path}")
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    
    # Return the first image file
    return os.path.join(folder_path, image_files[0])

# Get the first image in the real_data folder
image_path = get_first_image_in_folder()

print(f"Attempting to read image from: {os.path.abspath(image_path)}")

try:
    with open(image_path, 'rb') as f:
        full_content = f.read()

    all_metadata = {
        "file_size_bytes": len(full_content),
        "basic_image_info": {},
        "exif_metadata": {},
        "iptc_xmp_metadata": {},
        "appended_json_metadata": None,
        "image_binary_size_excluding_appended_json": len(full_content) # Default to full size if no JSON found
    }

    # --- Extract basic image info and EXIF/other data using Pillow ---
    try:
        with Image.open(image_path) as img:
            all_metadata["basic_image_info"] = {
                "filename": img.filename,
                "image_size": img.size,
                "image_height": img.height,
                "image_width": img.width,
                "image_format": img.format,
                "image_mode": img.mode,
                "image_is_animated": getattr(img, "is_animated", False),
                "frames_in_image": getattr(img, "n_frames", 1)
            }

            # Extract EXIF data
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except UnicodeDecodeError:
                            value = str(value) # Fallback to string representation if decoding fails
                    all_metadata["exif_metadata"][tag_name] = value

            # Attempt to extract IPTC/XMP data (often in img.info)
            if "icc_profile" in img.info:
                all_metadata["iptc_xmp_metadata"]["icc_profile"] = str(img.info["icc_profile"])
            if "photoshop" in img.info:
                # This can be complex; for now, represent as its string value
                all_metadata["iptc_xmp_metadata"]["photoshop"] = str(img.info["photoshop"])
            # More robust IPTC/XMP parsing would require dedicated libraries like piexif or pyxmp if Pillow's info isn't enough

    except Exception as e:
        print(f"Could not extract basic image info or EXIF/IPTC/XMP data with Pillow: {e}")

    # --- Extract JSON from end of file (our custom logic) ---
    metadata = None
    # We expect the JSON to be at the very end. Let's try to decode the last few KB
    # If the JSON is larger than a few KB, this might need adjustment.
    search_tail_size = min(len(full_content), 8192) # Look at the last 8KB
    tail_content_bytes = full_content[-search_tail_size:]

    decoded_tail = None
    try:
        decoded_tail = tail_content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            decoded_tail = tail_content_bytes.decode('latin-1')
        except UnicodeDecodeError:
            pass # Cannot decode as string, likely no text JSON at end

    if decoded_tail:
        # Find the last ']' as the end of the JSON array
        json_end_pos_in_tail = decoded_tail.rfind(']')
        
        if json_end_pos_in_tail != -1:
            # Search backwards from json_end_pos_in_tail for the first '['
            json_start_pos_in_tail = -1
            open_brackets = 0
            for idx in range(json_end_pos_in_tail, -1, -1):
                if decoded_tail[idx] == ']':
                    open_brackets += 1
                elif decoded_tail[idx] == '[':
                    open_brackets -= 1
                
                if open_brackets == 0:
                    json_start_pos_in_tail = idx
                    break

            if json_start_pos_in_tail != -1:
                potential_json_string = decoded_tail[json_start_pos_in_tail : json_end_pos_in_tail + 1]
                try:
                    metadata = json.loads(potential_json_string)
                    all_metadata["appended_json_metadata"] = metadata
                    
                    # Calculate the actual start of the JSON in the full_content
                    # It's the start of the tail + start of JSON in tail.
                    actual_json_start_in_full = len(full_content) - search_tail_size + json_start_pos_in_tail
                    all_metadata["image_binary_size_excluding_appended_json"] = actual_json_start_in_full
                    
                except json.JSONDecodeError:
                    pass # Not valid JSON, it might be partial or malformed

    print("\nFull Extracted Metadata (JSON):")
    print(json.dumps(all_metadata, indent=2))

except FileNotFoundError:
    print(f"Error: File not found at {image_path}. Please ensure the path is correct and the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 