import os
import json
import base64
import uuid
import re


# decode_and_save_base64_image remains the same as it correctly uses absolute paths for saving
def decode_and_save_base64_image(b64_string, output_dir):
    """Decodes a base64 string, saves it as an image with a UUID name, and returns the *absolute* path."""
    try:
        # Check for data URI prefix (e.g., data:image/png;base64,)
        match = re.match(r"data:image/(?P<format>\w+);base64,(?P<data>.*)", b64_string)
        if match:
            img_format = match.group("format").lower()
            img_data_b64 = match.group("data")
            # Handle common format variations if necessary
            if img_format == "jpeg":
                img_format = "jpg"
        else:
            # Assume PNG if no format specified, and string is pure base64
            img_format = "png"
            img_data_b64 = b64_string

        # Basic validation: Check if it looks like base64
        if not re.match(r"^[A-Za-z0-9+/=]+$", img_data_b64):
            # Handle potential padding issues if needed, or raise error
            # For simplicity, we'll try decoding directly
            pass

        img_data = base64.b64decode(img_data_b64)
        img_uuid = uuid.uuid4()
        img_filename = f"{img_uuid}.{img_format}"
        # Keep using absolute path for saving the file
        img_path_absolute = os.path.join(output_dir, img_filename)

        os.makedirs(output_dir, exist_ok=True)
        with open(img_path_absolute, "wb") as f:
            f.write(img_data)

        # Return the absolute path
        return img_path_absolute

    except (base64.binascii.Error, ValueError, TypeError) as e:
        print(
            f"Error decoding or saving base64 string: {e}. String (first 50 chars): {b64_string[:50]}..."
        )
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# Add project_root as an argument
def process_json_files(data_dir, imagery_output_dir, jsonl_output_file, project_root):
    """
    Processes JSON files in data_dir, converts base64 images to files in
    imagery_output_dir, and creates a jsonl file with *relative* image paths.
    """
    processed_records = []

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    print(f"Processing JSON files from: {data_dir}")
    print(f"Saving images to: {imagery_output_dir}")
    print(f"Saving JSONL records to: {jsonl_output_file}")

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".json"):
            json_path = os.path.join(data_dir, filename)
            print(f"Processing {filename}...")
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                # Check if 'images' key exists
                if "images" not in data or not isinstance(data["images"], dict):
                    print(
                        f"Warning: 'images' key missing or not a dictionary in {filename}. Skipping."
                    )
                    continue

                images_data = data["images"]
                record = {}

                # Process query image (expects a list with one item)
                if (
                    "query" in images_data
                    and isinstance(images_data["query"], list)
                    and len(images_data["query"]) == 1
                    and isinstance(images_data["query"][0], str)
                ):
                    query_path_absolute = decode_and_save_base64_image(
                        images_data["query"][0], imagery_output_dir
                    )
                    if query_path_absolute:
                        # Convert to relative path before storing
                        record["query_path"] = os.path.relpath(
                            query_path_absolute, project_root
                        ).replace(
                            "\\", "/"
                        )  # Ensure forward slashes
                else:
                    print(
                        f"Warning: 'images.query' field missing, not a list, empty, has multiple items, or item is not a string in {filename}. Skipping query."
                    )
                    continue  # Skip record if query is essential and missing/invalid

                # Process positive images
                positive_paths = []
                if "positive" in images_data and isinstance(
                    images_data["positive"], list
                ):
                    for b64_img in images_data["positive"]:
                        if isinstance(b64_img, str):
                            pos_path_absolute = decode_and_save_base64_image(
                                b64_img, imagery_output_dir
                            )
                            if pos_path_absolute:
                                # Convert to relative path before storing
                                positive_paths.append(
                                    os.path.relpath(
                                        pos_path_absolute, project_root
                                    ).replace("\\", "/")
                                )  # Ensure forward slashes
                        else:
                            print(
                                f"Warning: Non-string item found in 'images.positive' list in {filename}. Skipping item."
                            )
                record["positive_paths"] = positive_paths

                # Process negative images
                negative_paths = []
                if "negative" in images_data and isinstance(
                    images_data["negative"], list
                ):
                    for b64_img in images_data["negative"]:
                        if isinstance(b64_img, str):
                            neg_path_absolute = decode_and_save_base64_image(
                                b64_img, imagery_output_dir
                            )
                            if neg_path_absolute:
                                # Convert to relative path before storing
                                negative_paths.append(
                                    os.path.relpath(
                                        neg_path_absolute, project_root
                                    ).replace("\\", "/")
                                )  # Ensure forward slashes
                        else:
                            print(
                                f"Warning: Non-string item found in 'images.negative' list in {filename}. Skipping item."
                            )
                record["negative_paths"] = negative_paths

                if record.get(
                    "query_path"
                ):  # Only add record if query was processed successfully
                    processed_records.append(record)
                else:
                    print(
                        f"Skipping record for {filename} due to missing/failed query image processing."
                    )

            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {filename}. Skipping.")
            except KeyError as e:
                print(f"Error: Missing key {e} in {filename}. Skipping.")
            except IndexError as e:
                print(
                    f"Error: Index issue, likely accessing empty list for query image in {filename}: {e}. Skipping."
                )
            except Exception as e:
                print(
                    f"An unexpected error occurred processing {filename}: {e}. Skipping."
                )

    # Write the JSONL file
    try:
        # Ensure the output directory for the JSONL file exists
        os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)
        with open(jsonl_output_file, "w") as f:
            for record in processed_records:
                f.write(json.dumps(record) + "\n")
        print(f"\nSuccessfully processed {len(processed_records)} records.")
        print(
            f"Output JSONL file created at: {jsonl_output_file}"
        )  # Still shows absolute path here for clarity
    except IOError as e:
        print(f"Error writing JSONL file '{jsonl_output_file}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing JSONL file: {e}")


if __name__ == "__main__":
    # Define relative paths (adjust if your script is elsewhere)
    # Assuming script is in src/scripts/ and data is in data_samples/ at the root
    # and imagery should be created at the root.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels

    data_samples_directory = os.path.join(project_root, "data_samples")
    imagery_directory = os.path.join(project_root, "imagery")
    # Use absolute path for the output file itself, but store relative paths *inside* it
    output_file = os.path.join(project_root, "output.jsonl")

    # Pass project_root to the processing function
    process_json_files(
        data_samples_directory, imagery_directory, output_file, project_root
    )
