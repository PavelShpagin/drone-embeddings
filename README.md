# Drone Embeddings

A Python package for working with Google Maps coordinates and aerial imagery.

## Setup

1. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file with your Google Maps API key:**

   ```text
   GOOGLE_MAPS_API_KEY=your_api_key_here
   ```

4. **Enable required Google APIs:**

   - Maps Static API
   - Elevation API

## Usage

Run the example script to see the functionality in action:

```bash
python -m examples.snapshot_example
```

## Features

- Convert between latitude/longitude and XYZ coordinates
- Fetch ground elevation data using Google Maps Elevation API
- Generate aerial snapshots using Google Maps Static API
- Utilities for Mercator projection

## Project Structure

- `src/`: Contains the main modules
  - `config.py`: Loads configuration from `.env`
  - `coordinates/`: Handles coordinate conversions
  - `api/`: Interfaces with Google Maps APIs
  - `utils/`: Contains constants and utility functions
- `examples/`: Contains example scripts
- `requirements.txt`: Lists Python dependencies
- `.gitignore`: Specifies files to ignore in version control

## Notes

- Ensure your `.env` file is not committed to version control.
- Keep your virtual environment activated while working on the project.
- Adjust API key restrictions in Google Cloud Console as needed.