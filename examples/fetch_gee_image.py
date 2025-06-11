import ee
import os
from dotenv import load_dotenv
import requests
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from pathlib import Path
import time
import uuid
from google.cloud import storage
import sys
from tqdm import tqdm
from datetime import timedelta

def initialize_earth_engine():
    """Initializes the Earth Engine API with service account credentials."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Check if credentials path is set
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set in your .env file.")
            print("Please add it like this: GOOGLE_APPLICATION_CREDENTIALS=credentials/your-key-file.json")
            return False

        # Check if the credentials file exists
        if not Path(credentials_path).exists():
            print(f"Error: Credentials file not found at: {credentials_path}")
            print("Please make sure you've placed your service account JSON key file in the correct location.")
            return False

        # Initialize with service account
        credentials = ee.ServiceAccountCredentials(None, credentials_path)
        ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
        
        print("Google Earth Engine initialized successfully with service account credentials.")
        return True
    except Exception as e:
        print(f"Error initializing Earth Engine: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Ensure you've created a service account and downloaded its JSON key")
        print("2. Place the JSON key in the 'credentials' directory")
        print("3. Update .env with GOOGLE_APPLICATION_CREDENTIALS pointing to the key file")
        print("4. Make sure the service account has the 'Earth Engine Resource Writer' role")
        print("5. Verify that you've enabled the Earth Engine API in your Google Cloud Console")
        return False

def download_with_progress(url, output_path, total_size=None):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total size if not provided
    total_size = total_size or int(response.headers.get('content-length', 0))
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress bar
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def wait_for_task(task_id, output_dir):
    """Monitor task progress and download the result when complete."""
    try:
        # Initialize progress bar
        pbar = tqdm(total=100, desc="Exporting image", unit="%")
        last_progress = 0
        
        while True:
            # Get the task status
            status = ee.data.getTaskStatus(task_id)[0]
            state = status['state']
            
            if state == 'COMPLETED':
                pbar.update(100 - last_progress)
                pbar.close()
                print("\nExport completed successfully!")
                
                # Get the file from Google Cloud Storage
                storage_client = storage.Client()
                bucket = storage_client.bucket(os.getenv('GCS_BUCKET_NAME'))
                
                # List files with our prefix
                prefix = status['description']
                blobs = list(bucket.list_blobs(prefix=prefix))
                
                if not blobs:
                    print("Error: Could not find exported file in GCS bucket")
                    return False
                
                # Get a signed URL for the blob
                blob = blobs[0]
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(minutes=15),
                    method="GET"
                )
                
                # Download using the signed URL
                output_path = output_dir / blob.name
                print(f"\nDownloading image to {output_path}...")
                
                try:
                    download_with_progress(url, output_path, blob.size)
                    print(f"\nImage successfully saved to {output_path}")
                    
                    # Convert TIF to high quality JPG
                    jpg_path = output_path.with_suffix('.jpg')
                    print(f"Converting to {jpg_path}...")
                    with Image.open(output_path) as img:
                        # Enhance the image
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(1.2)  # Slightly increase contrast
                        enhancer = ImageEnhance.Sharpness(img)
                        img = enhancer.enhance(1.1)  # Slightly increase sharpness
                        
                        # Save with maximum quality
                        img.save(jpg_path, 'JPEG', quality=100, optimize=True)
                    print("Successfully converted.")

                    # Remove the original TIF file
                    output_path.unlink()
                    print(f"Removed original TIF: {output_path}")

                    # Clean up the GCS file
                    blob.delete()
                    return True
                    
                except Exception as e:
                    print(f"Error during download or conversion: {str(e)}")
                    return False
                
            elif state == 'FAILED':
                pbar.close()
                print(f"\nExport failed: {status.get('error_message', 'Unknown error')}")
                return False
                
            elif state == 'RUNNING':
                # Update progress if available
                if 'progress' in status:
                    progress = min(int(status['progress'] * 100), 100)
                    pbar.update(progress - last_progress)
                    last_progress = progress
            
            # Wait before checking again
            time.sleep(5)
            
    except Exception as e:
        print(f"\nError monitoring task: {str(e)}")
        return False

def get_gee_image(lat, lng, height_m=8000):
    """
    Fetches a high-quality, cloud-free satellite image from Google Earth Engine.
    Focuses on summer months for clear imagery.
    """
    try:
        # Define the Area of Interest (AOI)
        point = ee.Geometry.Point(lng, lat)
        
        # Calculate the region as a square around the point
        half_width = height_m / 2
        region = ee.Geometry.Rectangle([
            lng - half_width/111319,  # ~111,319 meters per degree at equator
            lat - half_width/111319,
            lng + half_width/111319,
            lat + half_width/111319
        ])

        # Search for Sentinel-2 imagery during summer months
        image_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .select(['B4', 'B3', 'B2'])  # Select RGB bands
            .map(lambda img: img.divide(10000)))  # Scale pixel values to [0,1]

        # Create summer date filters for the last few years
        summer_filters = [
            ee.Filter.date('2023-06-01', '2023-08-31'),  # Last summer
            ee.Filter.date('2022-06-01', '2022-08-31'),  # Summer before
            ee.Filter.date('2021-06-01', '2021-08-31')   # Summer before that
        ]
        
        # Combine filters using Or
        combined_filter = ee.Filter.Or(summer_filters)

        # Apply filters for summer collection
        summer_collection = image_collection.filter(combined_filter)
        summer_collection = summer_collection.filterBounds(region)
        summer_collection = summer_collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1))  # Even stricter cloud filter
        summer_collection = summer_collection.sort('CLOUDY_PIXEL_PERCENTAGE')

        # Get the collection size
        collection_size = summer_collection.size().getInfo()
        if collection_size == 0:
            print("No suitable summer images found. Trying with all seasons...")
            # Fallback to all seasons if no summer images found
            all_seasons = image_collection.filterDate('2021-01-01', '2024-02-29')
            all_seasons = all_seasons.filterBounds(region)
            all_seasons = all_seasons.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
            all_seasons = all_seasons.sort('CLOUDY_PIXEL_PERCENTAGE')
            
            collection_size = all_seasons.size().getInfo()
            if collection_size == 0:
                print("No suitable images found at all.")
                return None, None
            image_collection = all_seasons
        else:
            image_collection = summer_collection

        # Select the best image
        best_image = image_collection.first()
        
        # Get the image ID and date for logging
        image_id = best_image.get('system:index').getInfo()
        image_date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        print(f"Found image ID: {image_id}")
        print(f"Image date: {image_date}")
        
        # Get cloud cover percentage for logging
        cloud_cover = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        print(f"Cloud cover percentage: {cloud_cover:.1f}%")

        # Calculate dimensions to maintain perfect aspect ratio
        bounds = region.bounds().getInfo()['coordinates'][0]
        x_diff = abs(bounds[1][0] - bounds[0][0])
        y_diff = abs(bounds[2][1] - bounds[1][1])
        aspect_ratio = x_diff / y_diff
        
        # Calculate dimensions maintaining aspect ratio with max width of 8192
        width = 8192
        height = int(width / aspect_ratio)
        print(f"Calculated dimensions: {width}x{height} (maintaining aspect ratio)")

        # Enhance image quality
        enhanced_image = (best_image
            .focal_median(1)  # Slight smoothing to reduce noise
            .multiply(255)    # Scale to 8-bit range
            .toUint8())      # Convert to 8-bit

        # Generate a unique filename
        filename = f"satellite_{lat}_{lng}_{image_date}_ultrahd"
        
        # Export the image to Google Cloud Storage with maximum resolution
        task = ee.batch.Export.image.toCloudStorage(**{
            'image': enhanced_image,
            'description': filename,
            'bucket': os.getenv('GCS_BUCKET_NAME'),
            'fileNamePrefix': filename,
            'region': region,
            'maxPixels': 1e11,
            'fileFormat': 'GeoTIFF',
            'dimensions': f'{width}x{height}',
            'formatOptions': {
                'cloudOptimized': True,
                'fileDimensions': f'{width}x{height}'
            }
        })

        # Start the export task
        task.start()
        print("\nStarted ultra-high-resolution image export to Google Cloud Storage...")
        
        return task.id, filename

    except Exception as e:
        print(f"Error fetching image: {str(e)}")
        print("This might be due to:")
        print("- No imagery available for the specified location")
        print("- Invalid coordinates")
        print("- Network connectivity issues")
        print("- Earth Engine service issues")
        return None, None

def main():
    """Main function to run the GEE image fetcher."""
    load_dotenv()
    
    # Check for required environment variables
    if 'GCS_BUCKET_NAME' not in os.environ:
        print("Error: GCS_BUCKET_NAME not set in .env file")
        print("Please add GCS_BUCKET_NAME=your-bucket-name to your .env file")
        return

    if not initialize_earth_engine():
        return

    lat = 50.4162
    lng = 30.8906
    height = 8000  # in meters

    print(f"Fetching GEE image for Lat: {lat}, Lng: {lng}...")
    task_id, filename = get_gee_image(lat, lng, height)
    
    if task_id:
        # Define output directory
        output_dir = Path("data/earth_imagery")
        
        # Wait for task completion and download the result
        success = wait_for_task(task_id, output_dir)
        
        if success:
            print("\nProcess completed successfully!")
        else:
            print("\nFailed to complete the process.")

if __name__ == "__main__":
    main()

