from flask import Flask, request, send_from_directory, jsonify
import urllib.parse
import webbrowser
import os
import json
from datetime import datetime

app = Flask(__name__)

# Search templates - key-value pairs for search types and their query templates
SEARCH_TEMPLATES = {
    "Music": 'intitle:"index of" "parent directory" <keyword> "mp3" -html -htm -php -asp -jsp',
    "Images": 'intitle:"index of" "parent directory" <keyword> "jpg" "png" "gif" -html -htm -php -asp -jsp',
    "Archives": 'intitle:"index of" "parent directory" <keyword> "zip" "rar" "7z" "tar" "gz" -html -htm -php -asp -jsp',
    "PDFs": 'intitle:"index of" "parent directory" <keyword> "pdf" -html -htm -php -asp -jsp'
}

def get_search_query(search_type, user_input):
    """Generate search query from template and user input"""
    template = SEARCH_TEMPLATES.get(search_type)
    if template is None:
        return user_input
    return template.replace("<keyword>", user_input)

def open_google_search(query):
    """Open Google search in default browser"""
    try:
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://www.google.com/search?q={encoded_query}"
        webbrowser.open(search_url)
        return True
    except Exception as e:
        print(f"Error opening browser: {e}")
        return False

def log_eula_acceptance(ip_address):
    """Log EULA acceptance with timestamp and IP address"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "ip_address": ip_address,
            "accepted": True,
            "user_agent": request.headers.get('User-Agent', 'Unknown')
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Append to EULA acceptance log
        with open('logs/eula_acceptances.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        print(f"EULA accepted by {ip_address} at {log_entry['timestamp']}")
        return True
    except Exception as e:
        print(f"Error logging EULA acceptance: {e}")
        return False

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files (CSS, images, etc.)"""
    return send_from_directory('.', filename)

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    try:
        # Get form data
        search_type = request.form.get('searchType')
        user_input = request.form.get('userInput')
        
        if not search_type or not user_input:
            return jsonify({"error": "Missing search type or user input"}), 400
        
        # Generate search query
        query = get_search_query(search_type, user_input.strip())
        
        # Open Google search
        success = open_google_search(query)
        
        if success:
            return jsonify({"message": "Search opened in browser"}), 200
        else:
            return jsonify({"error": "Failed to open browser"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/eula-accept', methods=['POST'])
def eula_accept():
    """Handle EULA acceptance and log user data"""
    try:
        # Get client IP address
        ip_address = request.remote_addr
        
        # Log the acceptance
        log_success = log_eula_acceptance(ip_address)
        
        if log_success:
            return jsonify({
                "message": "EULA accepted",
                "timestamp": datetime.now().isoformat(),
                "ip_address": ip_address
            }), 200
        else:
            return jsonify({"error": "Failed to log acceptance"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Get local IP for network access
    import socket
    try:
        # Get local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"Server running on:")
        print(f"  Local:   http://localhost:5000")
        print(f"  Network: http://{local_ip}:5000")
    except:
        print("Server running on http://localhost:5000")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 