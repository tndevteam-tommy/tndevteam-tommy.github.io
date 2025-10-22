import torch
import PIL
import os
import threading
import queue
import uuid
import logging
import rembg
import gc
import requests
import io
import trimesh
import numpy as np
import urllib.parse

from PIL import Image
from flask import Flask, request, render_template, jsonify, send_from_directory, send_file
from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground
from tqdm import tqdm
from contextlib import nullcontext

app = Flask(__name__)
UPLOAD_FOLDER = 'c:\\generated\\images'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler()])

request_queue = queue.Queue()
model_loaded = threading.Event()
results_dict = {}
results_lock = threading.Lock()

char_limit = 128

torch.backends.cuda.enable_flash_sdp(True)


def garbage():
    torch.cuda.empty_cache()
    gc.collect()

def load_model(model_id="stabilityai/stable-fast-3d"):
    try:
        model = SF3D.from_pretrained(model_id, config_name="config.yaml", weight_name="model.safetensors")
        model.to(get_device())
        model.eval()

        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def model_loader():
    global model
    try:
        model = load_model()
        model_loaded.set()
        logging.info("Service Ready")
    except Exception as e:
        logging.error(f"Error in model_loader: {e}")

def process_images(image_paths, target_dir, args):
    rembg_session = rembg.new_session()
    images = []
    idx = 0

    def handle_image(image_path, idx):
        image = remove_background(Image.open(image_path).convert("RGBA"), rembg_session)
        image = resize_foreground(image, args['foreground_ratio'])
        os.makedirs(os.path.join(target_dir, str(idx)), exist_ok=True)
        image.save(os.path.join(target_dir, str(idx), "input.png"))
        images.append(image)

    for image_path in image_paths:
        if os.path.isdir(image_path):
            for img_path in [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith((".png", ".jpg", ".jpeg"))]:
                handle_image(img_path, idx)
                idx += 1
        else:
            handle_image(image_path, idx)
            idx += 1

    results = []
    for i in tqdm(range(0, len(images), args['batch_size'])):
        image_batch = images[i: i + args['batch_size']]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        garbage()
        with torch.no_grad():
            with torch.autocast(device_type=args['device'], dtype=torch.float16) if "cuda" in args['device'] else nullcontext():
                mesh, glob_dict = model.run_image(
                    image_batch,
                    bake_resolution=args['texture_resolution'],
                    remesh=args['remesh_option'],
                    vertex_count=args['target_vertex_count'],
                )
        if torch.cuda.is_available():
            logging.info("Peak Memory: %f MB", torch.cuda.max_memory_allocated() / 1024 / 1024)
        elif torch.backends.mps.is_available():
            logging.info("Peak Memory: %f MB", torch.mps.driver_allocated_memory() / 1024 / 1024)
        
        def get_unique_filename(directory, filename):
            base, ext = os.path.splitext(filename)
            counter = 1
            unique_filename = filename
            while os.path.exists(os.path.join(directory, unique_filename)):
                unique_filename = f"{base}_{counter}{ext}"
                counter += 1
            return unique_filename
        
        if len(image_batch) == 1:
            original_filename = os.path.basename(image_paths[i])
            base, _ = os.path.splitext(original_filename)
            original_filename_glb = f"{base[:char_limit]}.glb"
            unique_filename = get_unique_filename(target_dir, original_filename_glb)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
            out_mesh_path = os.path.join(target_dir, unique_filename)
            mesh.export(out_mesh_path, include_normals=True)
            results.append(out_mesh_path)
        else:
            for j in range(len(mesh)):
                original_filename = os.path.basename(image_paths[i + j])
                base, _ = os.path.splitext(original_filename)
                original_filename_glb = f"{base[:char_limit]}.glb"
                unique_filename = get_unique_filename(target_dir, original_filename_glb)
                mesh[j].apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
                out_mesh_path = os.path.join(target_dir, unique_filename)
                mesh[j].export(out_mesh_path, include_normals=True)
                results.append(out_mesh_path)
    return results

def log_input(input_text):
    try:
        if len(input_text) > 0:
            with open(os.path.join(UPLOAD_FOLDER, "prompt_log.txt"), "a") as file:
                file.write(input_text + "\n\n")
    except Exception as ex:
        logging.info(f"Prompt Log Error: {ex}")

# Renamed and modified to serve from UPLOAD_FOLDER directly using a new route prefix
@app.route('/images/<path:filename>')
def serve_generated_image(filename):
    logging.info(f"Attempting to serve file: {filename} from directory: {UPLOAD_FOLDER}")
    try:
        if os.path.isabs(filename):
            logging.warning(f"Attempt to serve absolute path rejected: {filename}")
            return jsonify({"status": "error", "message": "Invalid filename"}), 400
        
        full_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.isfile(full_path):
            logging.warning(f"File not found at {full_path} before calling send_from_directory.")

        response = send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)
        logging.info(f"Successfully prepared to serve {filename} from {UPLOAD_FOLDER}.")
        return response
    except FileNotFoundError: 
        logging.error(f"File not found by send_from_directory: {filename} in {UPLOAD_FOLDER}")
        return jsonify({"status": "error", "message": "Image file not found on server."}), 404
    except Exception as e:
        logging.error(f"Unexpected error serving {filename} from {UPLOAD_FOLDER}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Server error while trying to serve image."}), 500

# New route for serving GLB models from UPLOAD_FOLDER
@app.route('/models/<path:filename>')
def serve_model_file(filename):
    logging.info(f"Attempting to serve model: {filename} from {UPLOAD_FOLDER}")
    try:
        if not filename.lower().endswith('.glb'):
            logging.warning(f"Attempt to serve non-GLB file rejected: {filename}")
            return jsonify({"status": "error", "message": "Invalid file type. Only GLB files are served."}), 400
        
        if os.path.isabs(filename):
            logging.warning(f"Attempt to serve absolute path rejected for model: {filename}")
            return jsonify({"status": "error", "message": "Invalid model filename"}), 400

        # Check if file exists to provide a more specific log
        full_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.isfile(full_path):
            logging.warning(f"Model file not found at {full_path} before calling send_from_directory.")
            # Let send_from_directory handle the actual 404 response

        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)
    except FileNotFoundError: 
        logging.error(f"Model file not found by send_from_directory: {filename} in {UPLOAD_FOLDER}")
        return jsonify({"status": "error", "message": "Model file not found on server."}), 404
    except Exception as e:
        logging.error(f"Unexpected error serving model {filename} from {UPLOAD_FOLDER}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Server error while trying to serve model."}), 500

@app.route('/generate_for_approval', methods=['POST'])
def generate_for_approval_route():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"status": "error", "message": "Prompt is required."}), 400

    # Extract all parameters with defaults
    model = data.get('model', 'flux')
    seed = data.get('seed', 42)
    width = data.get('width', 1024)
    height = data.get('height', 1024)
    private = data.get('private', True)
    enhance = data.get('enhance', False)
    transparent = data.get('transparent', True)

    log_input(f"Prompt: {prompt} (Model: {model}, Seed: {seed}, Size: {width}x{height}, Private: {private}, Enhance: {enhance}, Transparent: {transparent})")
    logging.info(f"Generating image for approval. Prompt: {prompt}, Model: {model}, Seed: {seed}")

    logging.info(f"Debug: UPLOAD_FOLDER is {UPLOAD_FOLDER}") 
    # Save directly to UPLOAD_FOLDER
    image_filename = generate_image_and_save(prompt, UPLOAD_FOLDER, model=model, seed=seed, width=width, height=height, private=private, enhance=enhance, transparent=transparent)

    if image_filename is None:
        return jsonify({"status": "error", "message": "Image generation failed. Check server logs for details."}), 500

    image_url = f'/images/{image_filename}' # Updated URL prefix

    return jsonify({"status": "success", "image_url": image_url, "image_filename": image_filename})

@app.route('/process_approved_image', methods=['POST'])
def process_approved_image_route():
    data = request.json
    image_filename = data.get('image_filename')
    if not image_filename:
        return jsonify({"status": "error", "message": "Image filename is required."}), 400

    if '..' in image_filename or os.path.isabs(image_filename):
        return jsonify({"status": "error", "message": "Invalid image filename."}), 400

    # Image is now directly in UPLOAD_FOLDER
    approved_image_path = os.path.join(UPLOAD_FOLDER, image_filename)

    if not os.path.exists(approved_image_path):
        logging.error(f"Approved image not found at: {approved_image_path}")
        return jsonify({"status": "error", "message": f"Approved image not found: {image_filename}"}), 404

    logging.info(f"Queueing approved image for SF3D processing: {approved_image_path}")
    
    # Default arguments for SF3D processing
    # These could be passed from the client in the future if needed
    sf3d_args = {
        "device": "cuda",
        "pretrained_model": "stabilityai/stable-fast-3d",
        "foreground_ratio": 0.85, 
        "output_dir": UPLOAD_FOLDER, # Final models go to the main UPLOAD_FOLDER
        "texture_resolution": 1024,
        "remesh_option": "triangle",
        "target_vertex_count": -1,
        "batch_size": 1,
    }

    request_id = str(uuid.uuid4())
    # The process_queue -> process_images chain expects a list of paths
    request_queue.put((request_id, [approved_image_path], UPLOAD_FOLDER, sf3d_args))
    
    return jsonify({"status": "queued", "request_id": request_id})

@app.route('/status/<request_id>', methods=['GET'])
def status(request_id):
    with results_lock:
        if request_id in results_dict:
            result_paths = results_dict.pop(request_id)
            model_urls = []
            if isinstance(result_paths, list):
                for path in result_paths:
                    if isinstance(path, str) and path.lower().endswith('.glb'):
                        filename = os.path.basename(path)
                        model_urls.append(f'/models/{filename}')
                    else:
                        logging.warning(f"Result for request {request_id} is not a GLB path: {path}")
            
            return jsonify({"status": "completed", "model_urls": model_urls})
        else:
            return jsonify({"status": "pending"})

def process_queue():
    while True:
        request_id, image_paths, target_dir, args = request_queue.get()
        model_loaded.wait()
        try:
            results = process_images(image_paths, target_dir, args)
            with results_lock:
                results_dict[request_id] = results
        except Exception as e:
            logging.error(f"Error in process_queue: {e}")
            with results_lock:
                results_dict[request_id] = []

threading.Thread(target=process_queue, daemon=True).start()

def generate_image_and_save(input_prompt, save_directory, model='flux', seed=42, width=1024, height=1024, private=True, enhance=False, transparent=True):
    encoded_prompt = urllib.parse.quote(input_prompt)
    
    # Build the API URL with all parameters
    params = []
    params.append(f"model={model}")
    params.append(f"seed={seed}")
    params.append(f"width={width}")
    params.append(f"height={height}")
    params.append(f"private={'true' if private else 'false'}")
    params.append(f"enhance={'true' if enhance else 'false'}")
    params.append(f"transparent={'true' if transparent else 'false'}")
    
    query_string = "&".join(params)
    API_URL = f"https://pollinations.ai/p/{encoded_prompt}?{query_string}"

    headers = {
        "Accept": "image/jpeg" 
    }
    
    logging.info(f"Calling Pollinations.ai API with URL: {API_URL}")

    try:
        logging.info(f"Calling Pollinations.ai API: {API_URL}")
        response = requests.get(API_URL, headers=headers, timeout=120) 
        response.raise_for_status()
        
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))

    except requests.exceptions.Timeout:
        logging.error(f"Timeout calling Pollinations.ai API: {API_URL}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Pollinations.ai API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Pollinations.ai API response status: {e.response.status_code}")
            logging.error(f"Pollinations.ai API response content: {e.response.text[:500]}") 
        return None
    except PIL.UnidentifiedImageError:
        logging.error("Cannot identify image file from Pollinations.ai response. The response might not be an image.")
        if 'response' in locals() and response is not None:
            logging.error(f"Pollinations.ai Response content type: {response.headers.get('Content-Type')}")
            logging.error(f"Pollinations.ai Response content sample: {response.content[:200]}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during image generation with Pollinations.ai: {e}")
        return None

    # Ensure the save_directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    sanitized_prompt = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in input_prompt)
    sanitized_prompt = sanitized_prompt.replace(' ', '_')
    
    base_filename = f"{sanitized_prompt[:char_limit]}.png"
    # Use get_unique_filename to avoid overwriting in the save_directory
    image_filename = get_unique_filename(save_directory, base_filename)
    image_path = os.path.join(save_directory, image_filename)

    try:
        image.save(image_path)
        logging.info(f"Image saved to {image_path}")
        return image_filename
    except Exception as e:
        logging.error(f"Error saving image to {image_path}: {e}")
        return None

def show_image(image):
    threading.Thread(target=lambda: image.show()).start()

@app.route('/')
def index():
    return render_template('image_approval.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})
    file = request.files['file']
    foreground_ratio = request.form.get('foreground_ratio', type=float)
    texture_resolution = request.form.get('texture_resolution', type=int)
    remesh_option = request.form.get('remesh_option', type=str)
    target_vertex_count = request.form.get('target_vertex_count', type=int)
    batch_size = request.form.get('batch_size', type=int)
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, get_unique_filename(UPLOAD_FOLDER, file.filename))
        file.save(file_path)
        
        try:
            # Automatically queue the image for processing
            request_id = str(uuid.uuid4())
            args = {
                "device": "cuda",
                "pretrained_model": "stabilityai/stable-fast-3d",
                "foreground_ratio": foreground_ratio if foreground_ratio is not None else 0.85,
                "output_dir": UPLOAD_FOLDER,
                "texture_resolution": texture_resolution if texture_resolution is not None else 1024,
                "remesh_option": remesh_option if remesh_option is not None else "triangle",
                "target_vertex_count": target_vertex_count if target_vertex_count is not None else -1,
                "batch_size": batch_size if batch_size is not None else 1
            }
            request_queue.put((request_id, [file_path], UPLOAD_FOLDER, args))
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
        
        return jsonify({"status": "success", "file_path": file_path, "request_id": request_id})

def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename
    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1
    return unique_filename

def run_flask_app():
    app.run(port=5000)

@app.route('/modify_image', methods=['POST'])
def modify_image():
    data = request.json
    prompt = data.get('prompt')
    model = data.get('model', 'flux')
    seed = data.get('seed', 42)
    width = data.get('width', 1024)
    height = data.get('height', 1024)
    private = data.get('private', True)
    enhance = data.get('enhance', False)
    transparent = data.get('transparent', True)
    image_url = data.get('image_url')
    if not prompt or not image_url:
        return jsonify({'status': 'error', 'message': 'Prompt and image_url are required.'}), 400
    # Download the image to a temp file
    try:
        img_response = requests.get(image_url, timeout=60)
        img_response.raise_for_status()
        image_bytes = img_response.content
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to download image: {e}'}), 400
    # Save to a temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    # Prepare Pollinations.AI API call
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    params = []
    params.append(f"model={model}")
    params.append(f"seed={seed}")
    params.append(f"width={width}")
    params.append(f"height={height}")
    params.append(f"private={'true' if private else 'false'}")
    params.append(f"enhance={'true' if enhance else 'false'}")
    params.append(f"transparent={'true' if transparent else 'false'}")
    params.append(f"image={image_url}")
    query_string = "&".join(params)
    API_URL = f"https://image.pollinations.ai/prompt/{encoded_prompt}?{query_string}"
    headers = {"Accept": "image/jpeg"}
    try:
        response = requests.get(API_URL, headers=headers, timeout=120)
        response.raise_for_status()
        result_image = Image.open(io.BytesIO(response.content))
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Pollinations.AI API error: {e}'}), 500
    # Save result
    import os
    sanitized_prompt = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in prompt)
    sanitized_prompt = sanitized_prompt.replace(' ', '_')
    base_filename = f"modified_{sanitized_prompt[:char_limit]}.png"
    image_filename = get_unique_filename(UPLOAD_FOLDER, base_filename)
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    try:
        result_image.save(image_path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to save modified image: {e}'}), 500
    image_url_out = f'/images/{image_filename}'
    return jsonify({'status': 'success', 'image_url': image_url_out, 'image_filename': image_filename})

if __name__ == '__main__':
    # Start the model_loader thread first
    model_loader_thread = threading.Thread(target=model_loader)
    model_loader_thread.start()

    # Wait for the model to load
    model_loaded.wait()

    # Start the Flask app and request processing threads
    threading.Thread(target=run_flask_app, daemon=True).start()
    threading.Thread(target=process_queue, daemon=True).start()

    # Keep the main thread alive to allow other threads to run
    while True:
        pass 