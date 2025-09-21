# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont # Keep ImageFont here for fallback
import base64
import glob
import numpy as np
import time
import threading
import sys # Import sys for better error logging

# Import db and models from the new models.py file
from models import db, User, Patient, Slide 

# Try importing cv2, but allow the app to run without it if system libs are missing
try:
    import cv2
    print("DEBUG: OpenCV (cv2) imported successfully.")
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: OpenCV (cv2) could not be imported: {e}. Camera and advanced image processing features will be disabled.", file=sys.stderr)
    CV2_AVAILABLE = False
except Exception as e:
    print(f"WARNING: An unexpected error occurred during OpenCV import: {e}. Camera and advanced image processing features will be disabled.", file=sys.stderr)
    CV2_AVAILABLE = False


app = Flask(__name__)

# --- Configuration ---
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24)) 

# Set up SQLite database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pathovision.db'
print("DEBUG: Using SQLite database for local development and Render (non-persistent).")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Configuration for file uploads ---
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PROCESSED_FOLDER = 'static/processed'
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

BIOPY_CAPTURE_FOLDER = 'static/biopsy_captures'
app.config['BIOPY_CAPTURE_FOLDER'] = BIOPY_CAPTURE_FOLDER
os.makedirs(BIOPY_CAPTURE_FOLDER, exist_ok=True)

# Bind db to the app instance
db.init_app(app) 

# --- Add this block to create tables if they don't exist (for local AND Render) ---
# This will run when the app first starts.
with app.app_context():
    db.create_all()
    print("DEBUG: Database tables created or already exist.")

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Global Variables for Camera Capture and Real-Time Stitching ---
camera = None 

GRID_ROWS, GRID_COLS = 7, 7
current_cell = [0, 0]
cell_h, cell_w = None, None
stitched_canvas = None
last_update_ts = 0

is_capturing_continuously = False
capture_thread = None

current_focus = 50
current_zoom = 50

captured_tiles_count = 0
total_tiles_needed = GRID_ROWS * GRID_COLS

# --- Default dimensions for initial canvas if no image has been processed yet ---
DEFAULT_CANVAS_WIDTH = 640
DEFAULT_CANVAS_HEIGHT = 480

# ─── HELPERS FOR REAL-TIME STITCHING & GUIDANCE ───────────────────────────────────────────
def set_cell_dims(h, w):
    global cell_h, cell_w, stitched_canvas
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    
    # Ensure dimensions are at least 1 to prevent division by zero or invalid sizes
    if cell_h == 0: cell_h = 1
    if cell_w == 0: cell_w = 1

    # Initialize stitched_canvas with black pixels or default if cv2 not available
    if CV2_AVAILABLE:
        stitched_canvas = np.zeros((cell_h * GRID_ROWS,
                                    cell_w * GRID_COLS, 3), dtype="uint8")
        print(f"DEBUG: Initialized stitched_canvas with dimensions: {stitched_canvas.shape[:2]}")
    else:
        # Fallback for stitched_canvas if cv2 is not available
        stitched_canvas = np.full((cell_h * GRID_ROWS,
                                   cell_w * GRID_COLS, 3), 128, dtype="uint8") # Grey placeholder
        print(f"DEBUG: Initialized placeholder stitched_canvas with dimensions: {stitched_canvas.shape[:2]}")


def draw_overlay(frame):
    global cell_h, cell_w
    h, w = frame.shape[:2]

    if cell_h is None or cell_w is None: # Re-evaluate if dims changed or not set
        set_cell_dims(h, w)

    if CV2_AVAILABLE:
        # Draw grid lines
        for r_line in range(1, GRID_ROWS):
            cv2.line(frame, (0, r_line * cell_h), (w, r_line * cell_h), (255, 255, 255), 1)

        for c_line in range(1, GRID_COLS):
            cv2.line(frame, (c_line * cell_w, 0), (c_line * cell_w, h), (255, 255, 255), 1)

        # Draw rectangle for current cell
        r, c = current_cell
        cv2.rectangle(frame, (c * cell_w, r * cell_h),
                      (c * cell_w + cell_w, r * cell_h + cell_h),
                      (0, 255, 0), 2)

        # Add guidance text
        guidance_text = f"Capture {r + 1},{c + 1}"
        cv2.putText(frame, guidance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw focus center
        focus_center_x = c * cell_w + cell_w // 2
        focus_center_y = r * cell_h + cell_h // 2
        cv2.circle(frame, (focus_center_x, focus_center_y), 5, (0, 0, 255), -1)

    return frame

def place_on_canvas(img, r, c):
    global stitched_canvas, cell_h, cell_w
    
    if img is None or img.size == 0:
        print(f"Warning: Attempted to place empty or None image for cell {r},{c} on real-time canvas.", file=sys.stderr)
        return
    
    # If cell dimensions are not set or stitched_canvas is None, set them based on the first image
    if cell_w is None or cell_h is None or stitched_canvas is None:
        h, w = img.shape[:2]
        set_cell_dims(h, w)
        print(f"DEBUG: set_cell_dims called from place_on_canvas with img shape {img.shape[:2]}")
    
    # Ensure img has 3 channels (convert grayscale to BGR if necessary)
    if len(img.shape) == 2: # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print(f"DEBUG: Converted grayscale image to BGR for cell {r},{c}.")
    elif img.shape[2] == 4: # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        print(f"DEBUG: Converted RGBA image to BGR for cell {r},{c}.")

    if CV2_AVAILABLE:
        try:
            # Resize the captured tile to fit exactly into one cell of the stitched canvas
            resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            
            # Place the resized tile onto the stitched_canvas
            # Ensure the slice dimensions match the resized image dimensions
            stitched_canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
            print(f"DEBUG: Successfully placed tile {r},{c} on stitched_canvas. Canvas shape: {stitched_canvas.shape[:2]}")
        except Exception as e:
            print(f"ERROR: Failed to resize or place image for cell {r},{c} on real-time canvas: {e}", file=sys.stderr)
            # Draw a red square to indicate an error for this cell
            if stitched_canvas is not None:
                cv2.rectangle(stitched_canvas, (c * cell_w, r * cell_h),
                              (c * cell_w + cell_w, r * cell_h + cell_h),
                              (0, 0, 255), -1) # Red filled rectangle
    else:
        # Simple placeholder for stitched_canvas if cv2 not available
        # This will draw a colored square based on cell coordinates
        if stitched_canvas is not None:
            stitched_canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = np.full((cell_h, cell_w, 3), (r*30 % 255, c*30 % 255, 100), dtype="uint8")
            print(f"DEBUG: Placed placeholder tile {r},{c} on canvas (CV2 not available).")


@app.route('/live_video_feed')
def live_video_feed():
    placeholder_text = "Live Camera Feed (Client-Side)"
    img = np.zeros((480, 640, 3), dtype="uint8")
    if CV2_AVAILABLE:
        cv2.putText(img, placeholder_text, (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode(".jpg", img)
        frame_bytes = buffer.tobytes()
    else:
        from io import BytesIO
        img_pil = Image.new('RGB', (640, 480), color = (73, 109, 137))
        d = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
        d.text((100,240), "OpenCV Not Available", fill=(255,255,255), font=font)
        byte_io = BytesIO()
        img_pil.save(byte_io, 'jpeg')
        frame_bytes = byte_io.getvalue()

    while True:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\r\n" +
               frame_bytes + b"\r\n")
        time.sleep(1) # Send placeholder every second

@app.route('/stitched_biopsy_feed')
def stitched_biopsy_feed():
    global last_update_ts, stitched_canvas
    while True:
        # Check if stitched_canvas is initialized. If not, provide a placeholder.
        if stitched_canvas is None:
            placeholder_text = "Stitched View Not Ready / Initializing..."
            img_h, img_w = DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH
            if CV2_AVAILABLE:
                img = np.zeros((img_h, img_w, 3), dtype="uint8")
                cv2.putText(img, placeholder_text, (img_w//2 - 200, img_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode(".jpg", img)
                frame_bytes = buffer.tobytes()
            else:
                from io import BytesIO
                img_pil = Image.new('RGB', (img_w, img_h), color = (100, 100, 100))
                d = ImageDraw.Draw(img_pil)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                d.text((img_w//2 - 150, img_h//2), "OpenCV Not Available for Stitching", fill=(255,255,255), font=font)
                byte_io = BytesIO()
                img_pil.save(byte_io, 'jpeg')
                frame_bytes = byte_io.getvalue()

            yield (b"--frame\r\nContent-Type: image/jpeg\r\r\n" +
                   frame_bytes + b"\r\n")
            time.sleep(1) # Send placeholder every second
            continue

        # Only update if enough time has passed since last update
        if time.time() - last_update_ts < 0.1: # Limit updates to 10 FPS
            time.sleep(0.05)
            continue
        last_update_ts = time.time()

        frame_bytes = None
        if CV2_AVAILABLE:
            try:
                # Ensure stitched_canvas is not empty or malformed before encoding
                if stitched_canvas is not None and stitched_canvas.size > 0:
                    ret, buffer = cv2.imencode(".jpg", stitched_canvas)
                    if ret:
                        frame_bytes = buffer.tobytes()
                    else:
                        print("ERROR: cv2.imencode failed to encode stitched_canvas to JPEG.", file=sys.stderr)
                        # Fallback to a red error image if encoding fails
                        error_img = np.full((stitched_canvas.shape[0], stitched_canvas.shape[1], 3), (0, 0, 255), dtype="uint8")
                        cv2.putText(error_img, "Encoding Error!", (50, stitched_canvas.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        _, buffer = cv2.imencode(".jpg", error_img)
                        frame_bytes = buffer.tobytes()
                else:
                    print("WARNING: stitched_canvas is None or empty. Sending placeholder.", file=sys.stderr)
                    # Use a placeholder if canvas is unexpectedly empty
                    placeholder_img = np.zeros((DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, 3), dtype="uint8")
                    cv2.putText(placeholder_img, "Canvas Empty!", (50, DEFAULT_CANVAS_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    _, buffer = cv2.imencode(".jpg", placeholder_img)
                    frame_bytes = buffer.tobytes()

            except Exception as e:
                print(f"CRITICAL ERROR: Exception during stitched_biopsy_feed encoding: {e}", file=sys.stderr)
                # Generate a solid red image for critical errors
                critical_error_img = np.full((DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, 3), (0, 0, 255), dtype="uint8")
                cv2.putText(critical_error_img, "CRITICAL ERROR!", (50, DEFAULT_CANVAS_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                _, buffer = cv2.imencode(".jpg", critical_error_img)
                frame_bytes = buffer.tobytes()
        else:
            # Generate a solid color placeholder if cv2 not available
            placeholder_img = np.full((DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, 3), 128, dtype="uint8")
            try:
                from io import BytesIO
                img_pil = Image.fromarray(placeholder_img)
                d = ImageDraw.Draw(img_pil)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                d.text((50, DEFAULT_CANVAS_HEIGHT//2), "OpenCV Not Available", fill=(255,255,255), font=font)
                byte_io = BytesIO()
                img_pil.save(byte_io, 'jpeg')
                frame_bytes = byte_io.getvalue()
            except ImportError:
                # Fallback if PIL also fails (highly unlikely)
                print("WARNING: PIL not available for placeholder. Cannot generate image.", file=sys.stderr)
                frame_bytes = b'' # Send empty bytes if no image lib available

        if frame_bytes:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\r\n" +
                   frame_bytes + b"\r\n")
        else:
            # If for some reason frame_bytes is None/empty, yield a minimal placeholder
            yield (b"--frame\r\nContent-Type: image/jpeg\r\r\n\r\n")
            print("WARNING: No frame bytes generated for stitched_biopsy_feed. Sending minimal placeholder.", file=sys.stderr)


def continuous_capture_worker():
    global is_capturing_continuously, captured_tiles_count, total_tiles_needed
    print("DEBUG: Continuous capture worker started (backend will process client-sent frames).")
    while is_capturing_continuously:
        time.sleep(1) 
    print("DEBUG: Continuous capture worker stopped.")

def capture_biopsy_tile_internal(frame_data):
    global current_cell, stitched_canvas, captured_tiles_count, cell_h, cell_w

    if not CV2_AVAILABLE:
        print("ERROR: OpenCV not available. Cannot process image data.", file=sys.stderr)
        return False

    try:
        header, encoded = frame_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("ERROR: Failed to decode image data from client (cv2.imdecode returned None).", file=sys.stderr)
            return False
        
        # Ensure frame has 3 channels if it's grayscale or RGBA from client
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            print("DEBUG: Converted client grayscale frame to BGR.")
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            print("DEBUG: Converted client RGBA frame to BGR.")

        print(f"DEBUG: Client frame decoded successfully. Shape: {frame.shape}, Dtype: {frame.dtype}")

    except Exception as e:
        print(f"ERROR: Failed to decode client-sent image data: {e}", file=sys.stderr)
        return False

    patient_id = session.get('current_patient_id')
    if not patient_id:
        print("ERROR: No patient ID in session, cannot save tile.", file=sys.stderr)
        return False

    if captured_tiles_count >= total_tiles_needed:
        print("WARNING: Attempted to capture tile beyond grid boundaries. Capture already complete.", file=sys.stderr)
        return False

    r, c = current_cell
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    fname = f"patient_{patient_id}tile{r}{c}{timestamp}.jpg"
    path = os.path.join(app.config['BIOPY_CAPTURE_FOLDER'], fname)
    
    try:
        cv2.imwrite(path, frame)
        print(f"DEBUG: Captured tile {r},{c} for patient {patient_id} and saved to {path}")
    except Exception as e:
        print(f"ERROR: Failed to save tile image to disk at {path}: {e}", file=sys.stderr)
        return False

    if 'current_patient_captured_tiles' not in session:
        session['current_patient_captured_tiles'] = []
    if path not in session['current_patient_captured_tiles']:
        session['current_patient_captured_tiles'].append(path)
        session.modified = True
        print(f"DEBUG: Added tile path to session: {path}")

    # Initialize cell dimensions and stitched_canvas if not already done
    if cell_h is None or cell_w is None or stitched_canvas is None:
        set_cell_dims(frame.shape[0], frame.shape[1]) # Use first frame's dimensions

    place_on_canvas(frame, r, c)

    # Move to next cell
    if c < GRID_COLS - 1:
        current_cell[1] += 1
    else:
        current_cell[1] = 0
        current_cell[0] += 1
    
    print(f"DEBUG: Next cell target: Row {current_cell[0]}, Col {current_cell[1]}")
    return True

def perform_advanced_stitching(patient_id):
    if not CV2_AVAILABLE:
        print("ERROR: OpenCV not available. Cannot perform advanced stitching.", file=sys.stderr)
        return None

    if 'current_patient_captured_tiles' not in session:
        print("No captured tiles found in session for advanced stitching.", file=sys.stderr)
        return None

    tile_paths = [
        path for path in session['current_patient_captured_tiles']
        if f"patient_{patient_id}" in os.path.basename(path)
    ]

    if not tile_paths:
        print("No tile paths available for stitching.", file=sys.stderr)
        return None

    tile_paths.sort() 

    images = []
    for path in tile_paths:
        img = cv2.imread(path)
        if img is not None:
            # Ensure images are BGR for stitching
            if len(img.shape) == 2: # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            images.append(img)
        else:
            print(f"Warning: Could not read image from path: {path}", file=sys.stderr)

    if not images:
        print("No valid images loaded for stitching.", file=sys.stderr)
        return None

    # Use try-except for stitcher creation as well
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    except Exception as e:
        print(f"ERROR: Failed to create OpenCV Stitcher: {e}", file=sys.stderr)
        return None

    status, stitched_image = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("DEBUG: Advanced stitching successful!")
        return stitched_image
    else:
        print(f"ERROR: Advanced stitching failed with status: {status}", file=sys.stderr)
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("Stitching error: Need more images or insufficient overlap.", file=sys.stderr)
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Stitching error: Homography estimation failed (not enough matching features).", file=sys.stderr)
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("Stitching error: Camera parameters adjustment failed.", file=sys.stderr)
        return None


# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if 'username' in session:
        print(f"DEBUG: Login page - User already in session: {session.get('username')}. Redirecting to dashboard.")
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        print(f"DEBUG: Login attempt for email: {email}")
        user = User.query.filter_by(email=email).first()
        
        if user:
            print(f"DEBUG: User found: {user.email}")
            if user.check_password(password):
                session['username'] = user.email
                session['user_id'] = user.id
                session.modified = True
                print(f"DEBUG: Login successful for {user.email}. Session after setting: {session}")
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password.', 'error')
                print(f"DEBUG: Password mismatch for {user.email}")
                return render_template('login.html')
        else:
            flash('Invalid email or password.', 'error')
            print(f"DEBUG: User not found for email: {email}")
            return render_template('login.html')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if 'username' in session:
        print(f"DEBUG: Register page - User already in session: {session.get('username')}. Redirecting to dashboard.")
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('An account with this email already exists. Please login.', 'error')
            return render_template('register.html')
        new_user = User(email=email)
        new_user.set_password(password)
        try:
            db.session.add(new_user)
            db.session.commit()
            print(f"DEBUG: New user registered: {new_user.email}")
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login_page'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred during registration. Please try again. ({e})', 'error')
            print(f"DEBUG: Error during registration: {e}")
            return render_template('register.html')
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    print(f"DEBUG: Dashboard route accessed. Current session: {session}")
    if 'username' not in session:
        flash('Please login to access the dashboard.', 'error')
        print("DEBUG: User not in session. Redirecting to login page.")
        return redirect(url_for('login_page'))

    # Clear transient session data on dashboard load if not part of active workflow
    session.pop('processed_image_url', None)
    session.pop('biopsy_coordinates', None)

    user_id = session.get('user_id')
    current_patient_id = session.get('current_patient_id')
    current_patient = None
    if current_patient_id:
        current_patient = Patient.query.get(current_patient_id)
        if not current_patient:
            # If patient ID is in session but patient not found in DB, clear session
            session.pop('current_patient_id', None)
            session.pop('current_patient_name', None)
            session.pop('slide_uploaded_for_current_patient_flag', None)
            session.pop('biopsy_region_selected_flag', None)
            session.pop('uploaded_slide_url', None)
            session.pop('current_patient_captured_tiles', None)
            session.modified = True
            current_patient_name = 'N/A'
            patient_active = False
            print("DEBUG: Current patient ID in session not found in DB. Session cleared.")
        else:
            current_patient_name = current_patient.name
            patient_active = True
            print(f"DEBUG: Active patient: {current_patient_name}")
    else:
        current_patient_name = 'N/A'
        patient_active = False
        print("DEBUG: No active patient in session.")

    slide_uploaded_for_current_patient = session.get('slide_uploaded_for_current_patient_flag', False)
    biopsy_region_selected = session.get('biopsy_region_selected_flag', False)
    uploaded_slide_url = session.get('uploaded_slide_url', None)
    camera_mode_active = session.get('camera_mode_active', False)

    # Determine if capture is complete
    capture_complete = (captured_tiles_count >= total_tiles_needed)

    global is_capturing_continuously, capture_thread
    if capture_complete and is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5)
            print("DEBUG: Continuous capture thread stopped as all tiles are captured.")

    final_stitched_display_url = None
    if capture_complete and patient_active:
        print("DEBUG: Attempting to generate final stitched image for dashboard display...")
        stitched_img_np = perform_advanced_stitching(current_patient_id)
        if stitched_img_np is not None:
            display_filename = f"final_stitched_display_{current_patient_id}.jpg"
            display_path = os.path.join(app.config['PROCESSED_FOLDER'], display_filename)
            if CV2_AVAILABLE: # Only save if cv2 is available
                try:
                    cv2.imwrite(display_path, stitched_img_np)
                    final_stitched_display_url = url_for('static', filename=f'processed/{display_filename}')
                    print(f"DEBUG: Final stitched image saved for display at: {final_stitched_display_url}")
                except Exception as e:
                    print(f"ERROR: Failed to save final stitched image for display: {e}", file=sys.stderr)
                    flash("Failed to save final stitched image for display on server.", "warning")
            else:
                print("DEBUG: OpenCV not available, cannot save final stitched image for display.")
                flash("OpenCV not available on server, cannot generate high-quality stitched image for display.", "warning")
        else:
            print("DEBUG: Failed to generate final stitched image for display (perform_advanced_stitching returned None).")
            flash("Could not generate high-quality stitched image for display. Please check captured tiles.", "warning")

    return render_template('dashboard.html',
                           username=session['username'],
                           rows=GRID_ROWS,
                           cols=GRID_COLS,
                           current_patient_name=current_patient_name,
                           patient_active=patient_active,
                           slide_uploaded_for_current_patient=slide_uploaded_for_current_patient,
                           biopsy_region_selected=biopsy_region_selected,
                           uploaded_slide_url=uploaded_slide_url,
                           is_capturing_continuously=is_capturing_continuously,
                           current_focus=current_focus,
                           current_zoom=current_zoom,
                           capture_complete=capture_complete,
                           current_patient=current_patient,
                           camera_mode_active=camera_mode_active,
                           final_stitched_display_url=final_stitched_display_url,
                           datetime=datetime
                           )

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('current_patient_id', None)
    session.pop('current_patient_name', None)
    session.pop('slide_uploaded_for_current_patient_flag', None)
    session.pop('biopsy_region_selected_flag', None)
    session.pop('uploaded_slide_url', None)
    session.pop('processed_image_url', None)
    session.pop('biopsy_coordinates', None)
    session.pop('camera_mode_active', None)
    session.modified = True
    
    user_id = session.get('user_id')
    if user_id:
        cleanup_tile_images_for_user(user_id)
    session.pop('current_patient_captured_tiles', None)
    session.modified = True

    global is_capturing_continuously, capture_thread, current_cell, stitched_canvas, captured_tiles_count, cell_h, cell_w
    if is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5)
            print("DEBUG: Continuous capture thread stopped on logout.")

    current_cell = [0, 0]
    captured_tiles_count = 0
    stitched_canvas = None # Explicitly reset stitched_canvas
    cell_h, cell_w = None, None # Reset cell dimensions

    # Re-initialize stitched_canvas with default dimensions on logout
    set_cell_dims(DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH)
    print("DEBUG: Stitched canvas re-initialized with default dimensions on logout.")

    flash('You have been logged out.', 'success')
    print("DEBUG: User logged out. Session cleared.")
    return redirect(url_for('home'))

@app.route('/save_patient_details', methods=['POST'])
def save_patient_details():
    if request.method == 'POST':
        if 'user_id' not in session:
            flash('Please log in to save patient details.', 'error')
            return redirect(url_for('login_page'))
        try:
            patient_name = request.form['patientName']
            patient_age = int(request.form['patientAge'])
            patient_sex = request.form['patientSex']
            patient_date_str = request.form['patientDate']
            patient_date = datetime.strptime(patient_date_str, '%Y-%m-%d').date()
            user_id = session['user_id']

            new_patient = Patient(
                name=patient_name,
                age=patient_age,
                sex=patient_sex,
                date=patient_date,
                user_id=user_id
            )
            db.session.add(new_patient)
            db.session.commit()

            session['current_patient_id'] = new_patient.id
            session['current_patient_name'] = new_patient.name
            session['slide_uploaded_for_current_patient_flag'] = False
            session['biopsy_region_selected_flag'] = False
            session.pop('uploaded_slide_url', None)
            session['camera_mode_active'] = False
            session['current_patient_captured_tiles'] = []
            session.modified = True

            global current_cell, stitched_canvas, captured_tiles_count, cell_h, cell_w
            current_cell = [0, 0]
            captured_tiles_count = 0
            stitched_canvas = None # Explicitly reset stitched_canvas
            cell_h, cell_w = None, None # Reset cell dimensions
            set_cell_dims(DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH) # Initialize with default dims
            print("DEBUG: Stitched canvas re-initialized with default dimensions on new patient session.")


            flash('Patient details saved successfully! Now, please upload the biopsy slide image.', 'success')
            print(f"DEBUG: Patient details saved. Session: {session}")
            return redirect(url_for('dashboard'))

        except ValueError:
            flash('Invalid age or date format. Please check your input.', 'error')
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred while saving patient details: {e}', 'error')
            print(f"DEBUG: Error saving patient details: {e}", file=sys.stderr)
    return redirect(url_for('dashboard'))

@app.route('/clear_patient_session')
def clear_patient_session():
    patient_id_to_clear = session.get('current_patient_id')
    
    session.pop('current_patient_id', None)
    session.pop('current_patient_name', None)
    session.pop('slide_uploaded_for_current_patient_flag', None)
    session.pop('biopsy_region_selected_flag', None)
    session.pop('uploaded_slide_url', None)
    session.pop('camera_mode_active', None)
    session.modified = True
    
    if patient_id_to_clear:
        cleanup_tile_images_for_patient(patient_id_to_clear)
    session.pop('current_patient_captured_tiles', None)
    session.modified = True

    global is_capturing_continuously, capture_thread, current_cell, stitched_canvas, captured_tiles_count, cell_h, cell_w
    if is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5)
            print("DEBUG: Continuous capture thread stopped on clear patient session.")

    current_cell = [0, 0]
    captured_tiles_count = 0
    stitched_canvas = None # Explicitly reset stitched_canvas
    cell_h, cell_w = None, None # Reset cell dimensions

    set_cell_dims(DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH) # Initialize with default dims
    print("DEBUG: Stitched canvas re-initialized with default dimensions during reset.")


    print(f"DEBUG: Patient session cleared. Session: {session}")
    return redirect(url_for('dashboard'))

def cleanup_tile_images_for_user(user_id):
    with app.app_context():
        patient_ids = [p.id for p in Patient.query.filter_by(user_id=user_id).all()]
    total_deleted = 0
    for p_id in patient_ids:
        search_pattern = os.path.join(app.config['BIOPY_CAPTURE_FOLDER'], f"patient_{p_id}tile*.jpg")
        files_to_delete = glob.glob(search_pattern)
        for f in files_to_delete:
            try:
                os.remove(f)
                total_deleted += 1
            except OSError as e:
                print(f"Error deleting file {f} for user {user_id}: {e}", file=sys.stderr)
    print(f"DEBUG: Cleaned up {total_deleted} individual tile images for user {user_id} across all their patients.")

def cleanup_tile_images_for_patient(patient_id):
    search_pattern = os.path.join(app.config['BIOPY_CAPTURE_FOLDER'], f"patient_{patient_id}tile*.jpg")
    files_to_delete = glob.glob(search_pattern)
    total_deleted = 0
    for f in files_to_delete:
        try:
            os.remove(f)
            total_deleted += 1
        except OSError as e:
            print(f"Error deleting file {f} for patient {patient_id}: {e}", file=sys.stderr)
    print(f"DEBUG: Cleaned up {total_deleted} individual tile images for patient {patient_id}.")


@app.route('/upload_slide', methods=['POST'])
def upload_slide():
    if 'user_id' not in session:
        flash('Please log in to upload slides.', 'error')
        return redirect(url_for('login_page'))

    if not session.get('current_patient_id'):
        flash('Please save patient details first before uploading a slide.', 'error')
        return redirect(url_for('dashboard'))

    if 'slideImage' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('dashboard'))

    file = request.files['slideImage']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('dashboard'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        session['uploaded_slide_url'] = url_for('static', filename=f'uploads/{filename}')
        session['slide_uploaded_for_current_patient_flag'] = True
        session['biopsy_region_selected_flag'] = False
        session['camera_mode_active'] = False
        session['current_patient_captured_tiles'] = []
        session.modified = True

        # Re-initialize globals for a new slide upload
        global current_cell, stitched_canvas, captured_tiles_count, cell_h, cell_w
        current_cell = [0, 0]
        captured_tiles_count = 0
        stitched_canvas = None
        cell_h, cell_w = None, None
        set_cell_dims(DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH) # Initialize with default dims
        print("DEBUG: Stitched canvas re-initialized with default dimensions on new slide upload.")


        flash('Slide uploaded successfully! Now, please select the region of interest on the slide.', 'success')
        print(f"DEBUG: Slide uploaded. Session: {session}")
        return redirect(url_for('dashboard'))

    else:
        flash('Allowed image types are png, jpg, jpeg, gif', 'error')
        return redirect(url_for('dashboard'))

@app.route('/upload_camera_image', methods=['POST'])
def upload_camera_image():
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    if not session.get('current_patient_id'):
        return jsonify(status='error', message='Please save patient details first.'), 400

    data = request.json
    image_data_url = data.get('imageData')

    if not image_data_url:
        return jsonify(status='error', message='No image data provided.'), 400

    if not CV2_AVAILABLE:
        return jsonify(status='error', message='Server-side image processing (OpenCV) not available.'), 500

    try:
        header, encoded = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        
        # Determine file extension from mime type
        mime_type = header.split(':')[1].split(';')[0]
        ext = 'jpg' # Default to jpg
        if 'png' in mime_type:
            ext = 'png'
        elif 'jpeg' in mime_type:
            ext = 'jpg'
        else:
            print(f"WARNING: Unsupported MIME type received: {mime_type}. Defaulting to jpg.", file=sys.stderr)

        filename = f"camera_capture_initial_{session['current_patient_id']}{datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        session['uploaded_slide_url'] = url_for('static', filename=f'uploads/{filename}')
        session['slide_uploaded_for_current_patient_flag'] = True
        session['biopsy_region_selected_flag'] = False
        session['camera_mode_active'] = True
        session['current_patient_captured_tiles'] = []
        session.modified = True

        # Re-initialize globals for a new camera slide upload
        global current_cell, stitched_canvas, captured_tiles_count, cell_h, cell_w
        current_cell = [0, 0]
        captured_tiles_count = 0
        stitched_canvas = None
        cell_h, cell_w = None, None
        set_cell_dims(DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH) # Initialize with default dims
        print("DEBUG: Stitched canvas re-initialized with default dimensions on new camera slide upload.")


        print(f"DEBUG: Camera initial slide image uploaded. Session: {session}")
        return jsonify(status='success', message='Camera initial slide image uploaded successfully.')

    except Exception as e:
        print(f"ERROR: Error processing camera initial slide image upload: {e}", file=sys.stderr)
        return jsonify(status='error', message=f'Failed to process image: {e}'), 500


@app.route('/set_camera_mode_active', methods=['POST'])
def set_camera_mode_active():
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    
    data = request.json
    is_active = data.get('active', False)

    session['camera_mode_active'] = is_active
    session.modified = True
    print(f"DEBUG: Camera mode set to {is_active}. Session: {session}")
    return jsonify(status='success', camera_mode_active=is_active)


@app.route('/confirm_biopsy_region', methods=['POST'])
def confirm_biopsy_region():
    if 'user_id' not in session:
        return ('Access Denied', 403)
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag')):
        return ('Workflow step incomplete. Please upload a slide first.', 400)

    data = request.json
    selected_coords = {
        'x': data.get('x'),
        'y': data.get('y'),
        'width': data.get('width'),
        'height': data.get('height')
    }
    session['biopsy_region_coords'] = selected_coords
    session['biopsy_region_selected_flag'] = True
    session.modified = True

    flash('Biopsy region selected successfully! You can now proceed with live biopsy capture.', 'success')
    print(f"DEBUG: Biopsy region confirmed. Session: {session}")
    return ('', 204)

@app.route("/capture_biopsy_tile_from_client", methods=["POST"])
def capture_biopsy_tile_from_client():
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        return jsonify(status='error', message='Workflow step incomplete. Please complete previous workflow steps.'), 400

    global captured_tiles_count, total_tiles_needed
    if captured_tiles_count >= total_tiles_needed:
        return jsonify(status='capture_complete', message='All tiles already captured.'), 200

    data = request.json
    image_data_url = data.get('imageData')

    if not image_data_url:
        return jsonify(status='error', message='No image data provided.'), 400

    if capture_biopsy_tile_internal(image_data_url):
        captured_tiles_count += 1
        if captured_tiles_count >= total_tiles_needed:
            global is_capturing_continuously, capture_thread
            if is_capturing_continuously:
                is_capturing_continuously = False
                if capture_thread and capture_thread.is_alive():
                    capture_thread.join(timeout=5)
                    print("DEBUG: Continuous capture thread stopped as all tiles are captured.")
            with app.test_request_context(): # Use test_request_context for flash outside request
                flash("All biopsy sections captured! You can now review and generate the report.", 'success')
            print(f"DEBUG: Tile captured. All tiles complete. Current captured count: {captured_tiles_count}. Session: {session}")
            return jsonify(status='capture_complete', message='Tile captured. All tiles complete.')
        print(f"DEBUG: Tile captured. Current captured count: {captured_tiles_count}. Session: {session}")
        return jsonify(status='success', message='Tile captured.')
    else:
        print("DEBUG: Failed to capture tile internally from client data.")
        return jsonify(status='error', message='Failed to process client image.'), 500


@app.route('/start_continuous_capture', methods=['POST'])
def start_continuous_capture():
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        return jsonify(status='error', message='Workflow step incomplete. Please complete previous workflow steps.'), 400

    global is_capturing_continuously, capture_thread, captured_tiles_count, total_tiles_needed
    
    if captured_tiles_count >= total_tiles_needed:
        flash("All tiles already captured for this session. Please reset if you want to capture again.", 'info')
        return jsonify(status='capture_complete')

    if not is_capturing_continuously:
        is_capturing_continuously = True
        # The continuous_capture_worker is mostly a placeholder now, as client sends frames
        capture_thread = threading.Thread(target=continuous_capture_worker)
        capture_thread.daemon = True
        capture_thread.start()
        flash('Continuous video capture started. Please ensure your browser camera is active.', 'success')
        print("DEBUG: Continuous capture started (client-side expected).")
        return jsonify(status='started')
    print("DEBUG: Continuous capture already started.")
    return jsonify(status='already_started')

@app.route('/stop_continuous_capture', methods=['POST'])
def stop_continuous_capture():
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    global is_capturing_continuously, capture_thread
    if is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5)
            print("DEBUG: Continuous capture thread stopped via API.")
        flash('Continuous video capture stopped.', 'info')
        return jsonify(status='stopped')
    print("DEBUG: Continuous capture already stopped.")
    return jsonify(status='already_stopped')

@app.route('/get_camera_guidance')
def get_camera_guidance():
    r, c = current_cell
    
    # Calculate next cell for guidance display purposes
    next_row = r
    next_col = c + 1
    if next_col >= GRID_COLS:
        next_row += 1
        next_col = 0

    guidance_message = ""
    if captured_tiles_count >= total_tiles_needed:
        guidance_message = "All sections captured! Proceed to final review and report generation."
    elif next_row >= GRID_ROWS: # This case should ideally be caught by captured_tiles_count check
        guidance_message = "All sections captured! Proceed to final review and report generation."
    elif c < GRID_COLS - 1:
        guidance_message = f"Next section: Row {r + 1}, Column {c + 2} (move camera right)."
    else: # c is last column, move to next row
        guidance_message = f"Move to next row: Row {r + 2}, Column 1 (move camera down and reset left)."
    
    # Adjust guidance if we are on the very last cell
    if captured_tiles_count == total_tiles_needed - 1:
        guidance_message = "Capture final section. Then proceed to final review and report generation."


    return jsonify({
        'guidance': guidance_message,
        'current_cell': {'row': r, 'col': c},
        'next_cell': {'row': next_row, 'col': next_col},
        'focus_level': current_focus,
        'zoom_level': current_zoom,
        'capture_complete': (captured_tiles_count >= total_tiles_needed),
        'captured_tiles_count': captured_tiles_count,
        'total_tiles_needed': total_tiles_needed
    })

@app.route('/set_camera_focus', methods=['POST'])
def set_camera_focus():
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    data = request.json
    focus_level = data.get('focus_level')
    if focus_level is not None and 0 <= focus_level <= 100:
        global current_focus
        current_focus = int(focus_level)
        print(f"DEBUG: Focus set to {current_focus}")
        return jsonify(status='success', focus_level=current_focus)
    print(f"ERROR: Invalid focus level received: {focus_level}", file=sys.stderr)
    return jsonify(status='error', message='Invalid focus level'), 400

@app.route('/set_camera_zoom', methods=['POST'])
def set_camera_zoom():
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    data = request.json
    zoom_level = data.get('zoom_level')
    if zoom_level is not None and 0 <= zoom_level <= 100:
        global current_zoom
        current_zoom = int(zoom_level)
        print(f"DEBUG: Zoom set to {current_zoom}")
        return jsonify(status='success', zoom_level=current_zoom)
    print(f"ERROR: Invalid zoom level received: {zoom_level}", file=sys.stderr)
    return jsonify(status='error', message='Invalid zoom level'), 400


@app.route("/reset_biopsy_capture_session")
def reset_biopsy_capture_session():
    if 'user_id' not in session:
        flash('Please log in to reset.', 'error')
        return redirect(url_for('login_page'))
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        flash('Cannot reset biopsy capture. Please ensure a patient is selected, a slide is uploaded, and a region is selected.', 'error')
        return redirect(url_for('dashboard'))

    global is_capturing_continuously, capture_thread, current_cell, stitched_canvas, captured_tiles_count, cell_h, cell_w
    if is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5)
            print("DEBUG: Continuous capture thread stopped on reset.")

    patient_id = session.get('current_patient_id')
    if patient_id:
        cleanup_tile_images_for_patient(patient_id)
    session.pop('current_patient_captured_tiles', None)
    session.modified = True

    current_cell = [0, 0]
    captured_tiles_count = 0
    stitched_canvas = None # Explicitly reset stitched_canvas
    cell_h, cell_w = None, None # Reset cell dimensions

    set_cell_dims(DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH) # Initialize with default dims
    print("DEBUG: Stitched canvas re-initialized with default dimensions during reset.")


    print(f"DEBUG: Biopsy capture session reset. Session: {session}")
    return redirect(url_for('dashboard'))


@app.route("/download_stitched_biopsy")
def download_stitched_biopsy():
    if 'user_id' not in session:
        flash('Access Denied. Please log in.', 'error')
        return redirect(url_for('login_page'))
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        flash('Workflow step incomplete. Please complete previous workflow steps before downloading.', 'error')
        return redirect(url_for('dashboard'))

    patient_id = session['current_patient_id']
    
    stitched_img_np = perform_advanced_stitching(patient_id)

    if stitched_img_np is None:
        flash("Failed to generate high-quality stitched image for download. Please ensure enough tiles were captured and have sufficient overlap.", 'error')
        return redirect(url_for('dashboard'))
    
    stitched_filename = f"patient_{patient_id}seamless_biopsy{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    stitched_path = os.path.join(app.config['PROCESSED_FOLDER'], stitched_filename)
    
    try:
        if CV2_AVAILABLE: # Only save if cv2 is available
            cv2.imwrite(stitched_path, stitched_img_np)
            print(f"DEBUG: Seamless stitched image saved to: {stitched_path}")
        else:
            print("WARNING: OpenCV not available, cannot save stitched image to disk.", file=sys.stderr)
            flash("OpenCV not available on server, cannot save high-quality stitched image to disk.", "warning")
            return redirect(url_for('dashboard')) # Redirect if save fails due to missing cv2

        selected_region = session.get('biopsy_region_coords', {})
        biopsy_coords_str = f"Stitched from {GRID_ROWS}x{GRID_COLS} grid captures. Selected region on original slide: {selected_region}"

        new_slide = Slide(
            filename=stitched_filename,
            processed_filename=stitched_filename,
            biopsy_coords=biopsy_coords_str,
            patient_id=patient_id
        )
        db.session.add(new_slide)
        db.session.commit()

        flash('Seamless stitched biopsy image downloaded and saved to patient records!', 'success')
        return send_file(stitched_path, as_attachment=True, download_name=stitched_filename)
    except Exception as e:
        db.session.rollback()
        flash(f'Error downloading or saving stitched image: {e}', 'error')
        print(f"DEBUG: Error in download_stitched_biopsy: {e}", file=sys.stderr)
        return redirect(url_for('dashboard'))


@app.route('/save_diagnostic_report', methods=['POST'])
def save_diagnostic_report():
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication or patient session missing.'), 403
    
    patient_id = session['current_patient_id']
    report_text = request.json.get('report_text')

    patient = Patient.query.get(patient_id)
    if patient:
        patient.diagnostic_report = report_text
        try:
            db.session.commit()
            flash('Diagnostic report saved successfully!', 'success')
            print(f"DEBUG: Diagnostic report saved for patient {patient_id}.")
            return jsonify(status='success')
        except Exception as e:
            db.session.rollback()
            print(f"DEBUG: Error saving diagnostic report: {e}", file=sys.stderr)
            return jsonify(status='error', message=f'Failed to save report: {e}'), 500
    print(f"ERROR: Patient {patient_id} not found for saving diagnostic report.", file=sys.stderr)
    return jsonify(status='error', message='Patient not found.'), 404

@app.route('/archive_case', methods=['POST'])
def archive_case():
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication or patient session missing.'), 403

    patient_id = session['current_patient_id']
    patient = Patient.query.get(patient_id)
    if patient:
        patient.is_archived = True
        try:
            db.session.commit()
            session.pop('current_patient_id', None)
            session.pop('current_patient_name', None)
            session.pop('slide_uploaded_for_current_patient_flag', None)
            session.pop('biopsy_region_selected_flag', None)
            session.pop('uploaded_slide_url', None)
            session.pop('camera_mode_active', None)
            session.modified = True
            
            cleanup_tile_images_for_patient(patient_id)
            session.pop('current_patient_captured_tiles', None)
            session.modified = True

            global current_cell, stitched_canvas, captured_tiles_count, cell_h, cell_w
            current_cell = [0, 0]
            captured_tiles_count = 0
            stitched_canvas = None # Explicitly reset stitched_canvas
            cell_h, cell_w = None, None # Reset cell dimensions

            flash('Case archived successfully! You can now start a new patient workflow.', 'success')
            print(f"DEBUG: Case archived for patient {patient_id}. Session: {session}")
            return jsonify(status='success')
        except Exception as e:
            db.session.rollback()
            print(f"DEBUG: Error archiving case: {e}", file=sys.stderr)
            return jsonify(status='error', message=f'Failed to archive case: {e}'), 500
    print(f"ERROR: Patient {patient_id} not found for archiving.", file=sys.stderr)
    return jsonify(status='error', message='Patient not found.'), 404


if __name__ == '__main__':
    app.run(debug=True)
