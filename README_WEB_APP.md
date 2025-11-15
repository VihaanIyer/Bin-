# Bin Layout Configuration Web App

A web interface for uploading bin layout images and configuring location-specific bin metadata.

## Features

- 📷 Upload bin layout images (PNG, JPG, JPEG, GIF, WEBP)
- 🌍 Select from predefined locations:
  - Atlanta, GA, USA
  - Budapest, Hungary
  - Hong Kong, Hong Kong
  - Singapore, Singapore
- 🤖 AI-powered bin layout analysis using Gemini Vision
- 💾 Automatic saving of location-specific bin configurations

## Usage

1. **Start the web server:**
   ```bash
   source venv/bin/activate
   python web_app.py
   ```

2. **Open your browser:**
   Navigate to the URL shown (usually `http://localhost:8080`)

3. **Configure bin layout:**
   - Select a location from the dropdown
   - Click "Start Camera" to use your device camera
   - Position your device to capture the bin layout
   - Click "📸 Capture Photo" to take the picture
   - Click "Analyze Bin Layout" to process the image
   - The system will automatically detect bins and save the configuration

4. **Use in main software:**
   After configuring, set the location in your `.env` file:
   ```
   BIN_LOCATION=atlanta_ga_usa
   ```
   Or export it before running:
   ```bash
   export BIN_LOCATION=atlanta_ga_usa
   python main.py
   ```

## File Structure

- `web_app.py` - Flask web application
- `templates/index.html` - Web interface
- `uploads/` - Temporary storage for uploaded images
- `bin_layout_<location>.json` - Location-specific bin configurations
- `bin_layout_metadata.json` - Main bin layout file (updated on each upload)

## API Endpoints

- `GET /` - Main page with upload form
- `POST /upload` - Upload image and analyze bin layout
- `GET /locations` - Get list of available locations
- `GET /layout/<location>` - Get bin layout for a specific location

## Requirements

- Flask >= 3.0.0
- Gemini API key (set in `.env` file)
- All other dependencies from `requirements.txt`

## Notes

- Uploaded images are temporarily stored in the `uploads/` directory
- Each location gets its own `bin_layout_<location>.json` file
- The main `bin_layout_metadata.json` is updated with the most recent upload
- Maximum file size: 16MB

