# Google Earth Engine Setup Guide

This guide will help you set up Google Earth Engine (GEE) API access for high-resolution satellite image sampling.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud account with billing enabled
2. **Earth Engine Access**: Sign up for Google Earth Engine at [earthengine.google.com](https://earthengine.google.com/)

## Step 1: Enable Google Earth Engine API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Earth Engine API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Earth Engine API"
   - Click "Enable"

## Step 2: Create Service Account

1. In Google Cloud Console, go to "IAM & Admin" > "Service Accounts"
2. Click "Create Service Account"
3. Fill in the details:
   - **Name**: `earth-engine-sampler`
   - **Description**: `Service account for Earth Engine image sampling`
4. Click "Create and Continue"
5. Grant the following roles:
   - **Earth Engine Resource Writer**
   - **Storage Object Admin** (if using Cloud Storage)
6. Click "Done"

## Step 3: Generate and Download Key

1. Find your newly created service account in the list
2. Click on it, then go to the "Keys" tab
3. Click "Add Key" > "Create new key"
4. Choose "JSON" format
5. Download the JSON key file

## Step 4: Set Up Project Structure

1. Create a `secrets/` directory in your project root:
   ```bash
   mkdir -p secrets/
   ```

2. Move the downloaded JSON key to `secrets/earth-engine-key.json`:
   ```bash
   mv ~/Downloads/your-service-account-key.json secrets/earth-engine-key.json
   ```

3. Update your `.gitignore` to exclude secrets:
   ```bash
   echo "secrets/" >> .gitignore
   ```

## Step 5: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The key dependencies are:
- `earthengine-api`: Official Earth Engine Python API
- `geemap`: Simplified interface for Earth Engine
- `google-cloud-storage`: For cloud storage operations
- `retry`: For robust error handling

## Step 6: Test Your Setup

Run the test script to verify everything works:

```bash
cd examples/
python gee_image_sampler.py
```

This will:
1. ✅ Authenticate with Google Earth Engine
2. 🛰️ Sample high-resolution satellite imagery (1km coverage)
3. 💾 Save images to `data/gee_api/` directory
4. 🎯 Test multiple locations with different satellites

## Expected Output

The script will download satellite images with:
- **Resolution**: 10m (Sentinel-2) or 30m (Landsat)
- **Coverage**: 1km × 1km area
- **Format**: PNG with enhanced versions
- **Sources**: Sentinel-2, Landsat 8/9 (automatic best selection)

## Troubleshooting

### Authentication Errors
- ❌ `Service account key not found`: Check file path `secrets/earth-engine-key.json`
- ❌ `Permission denied`: Ensure service account has "Earth Engine Resource Writer" role
- ❌ `API not enabled`: Enable Earth Engine API in Google Cloud Console

### Image Download Errors
- ❌ `No suitable images found`: Try different date ranges or locations
- ❌ `Download failed`: Check network connection and API quotas
- ❌ `High cloud cover`: Script automatically filters for <10% cloud cover

### API Quotas
Google Earth Engine has usage quotas:
- **Free tier**: Limited requests per day
- **Paid usage**: Higher quotas available
- Monitor usage in Google Cloud Console

## Advanced Configuration

### Custom Resolution
Modify the resolution in `gee_image_sampler.py`:
```python
# Change this line for different resolutions
resolution_m = 10  # 10m, 20m, 30m, etc.
```

### Different Satellite Data
Add more satellite datasets in the `datasets` list:
```python
{
    "name": "Your Custom Satellite",
    "collection": "COLLECTION/NAME", 
    "bands": ['B4', 'B3', 'B2'],
    "scale_factor": 0.0001,
    "cloud_property": "CLOUD_COVER"
}
```

### Batch Processing
For large-scale sampling, consider:
- Implementing parallel processing
- Using Earth Engine's batch export features
- Setting up Cloud Storage for large datasets

## File Structure

After setup, your project should look like:
```
your-project/
├── secrets/
│   └── earth-engine-key.json     # 🔐 Your service account key
├── data/
│   └── gee_api/                   # 📁 Downloaded satellite images
├── examples/
│   └── gee_image_sampler.py       # 🧪 Test script
└── requirements.txt               # 📦 Dependencies
```

## Security Notes

- **Never commit** the `secrets/` directory to version control
- **Rotate keys** regularly for production use
- **Use IAM policies** to limit service account permissions
- **Monitor usage** to prevent unexpected charges

## Getting Help

If you encounter issues:
1. Check the [Earth Engine documentation](https://developers.google.com/earth-engine/)
2. Review [geemap examples](https://geemap.org/notebooks/)
3. Check your Google Cloud Console for API quotas and billing
4. Ensure your Earth Engine account is properly registered

Happy satellite image sampling! 🛰️🌍 