"""
Web application for uploading bin layout images and selecting locations
"""
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from gemini_classifier import TrashClassifier
from bin_layout_analyzer import BinLayoutAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create uploads directory if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Location options
LOCATIONS = {
    'atlanta_ga_usa': 'Atlanta, GA, USA',
    'budapest_hungary': 'Budapest, Hungary',
    'hong_kong': 'Hong Kong, Hong Kong',
    'singapore': 'Singapore, Singapore'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html', locations=LOCATIONS)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and location selection"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        location = request.form.get('location')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not location or location not in LOCATIONS:
            return jsonify({'error': 'Invalid location selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, WEBP'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize classifier and analyzer
        print(f"Initializing classifier for location: {LOCATIONS[location]}")
        classifier = TrashClassifier()
        
        if not getattr(classifier, 'supports_vision', False):
            return jsonify({'error': 'Gemini vision model not available'}), 500
        
        analyzer = BinLayoutAnalyzer(classifier)
        
        # Analyze the uploaded image
        print(f"Analyzing bin layout from uploaded image...")
        image = Image.open(filepath)
        result = analyzer._analyze_image(image)
        
        # Add location metadata
        result['location'] = location
        result['location_name'] = LOCATIONS[location]
        
        # Save location-specific bin layout
        location_file = f"bin_layout_{location}.json"
        location_path = Path(location_file)
        location_path.write_text(json.dumps(result, indent=2))
        
        # Also update the main bin_layout_metadata.json (for backward compatibility)
        main_path = Path("bin_layout_metadata.json")
        main_path.write_text(json.dumps(result, indent=2))
        
        print(f"✅ Bin layout saved to {location_file} and bin_layout_metadata.json")
        print(f"📝 To use this configuration in main.py, set environment variable:")
        print(f"   export BIN_LOCATION={location}")
        print(f"   Or update .env file with: BIN_LOCATION={location}")
        
        return jsonify({
            'success': True,
            'message': f'Bin layout analyzed and saved for {LOCATIONS[location]}',
            'bins': result.get('bins', []),
            'scene': result.get('scene', ''),
            'location': location,
            'location_name': LOCATIONS[location],
            'location_file': location_file
        })
        
    except Exception as e:
        print(f"Error processing upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/locations', methods=['GET'])
def get_locations():
    """Get list of available locations"""
    return jsonify({'locations': LOCATIONS})

@app.route('/layout/<location>', methods=['GET'])
def get_layout(location):
    """Get bin layout for a specific location"""
    if location not in LOCATIONS:
        return jsonify({'error': 'Invalid location'}), 404
    
    location_file = f"bin_layout_{location}.json"
    location_path = Path(location_file)
    
    if not location_path.exists():
        return jsonify({'error': 'Layout not found for this location'}), 404
    
    try:
        with open(location_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Error reading layout: {str(e)}'}), 500

if __name__ == '__main__':
    import socket
    
    def find_free_port(start_port=8080):
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port  # Fallback
    
    print("Starting bin layout web app...")
    print("Available locations:", list(LOCATIONS.values()))
    port = int(os.environ.get('PORT', find_free_port(8080)))
    print(f"Server starting on http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)

