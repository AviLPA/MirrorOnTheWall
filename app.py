from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, make_response
import os
import logging
import cv2
import time
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Temporary compatibility shim for httpx>=0.28 where Client/AsyncClient removed 'proxies' kwarg
# Some OpenAI SDK versions still pass 'proxies', which crashes on Render if httpx is new.
try:
    import httpx  # type: ignore

    _orig_httpx_client_init = httpx.Client.__init__
    def _shim_httpx_client_init(self, *args, **kwargs):  # noqa: D401
        # Drop unsupported kwargs to avoid TypeError on newer httpx
        kwargs.pop('proxies', None)
        return _orig_httpx_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _shim_httpx_client_init  # type: ignore

    if hasattr(httpx, 'AsyncClient'):
        _orig_httpx_asyncclient_init = httpx.AsyncClient.__init__  # type: ignore
        def _shim_httpx_asyncclient_init(self, *args, **kwargs):  # noqa: D401
            kwargs.pop('proxies', None)
            return _orig_httpx_asyncclient_init(self, *args, **kwargs)
        httpx.AsyncClient.__init__ = _shim_httpx_asyncclient_init  # type: ignore
except Exception:
    # If anything goes wrong, proceed without shim
    pass
import speech_recognition as sr
import numpy as np
import tempfile
from datetime import datetime
from prompts import MentalHealthPrompts
from firebase_admin import firestore

# Load environment variables first
load_dotenv()

# Initialize variables for OpenAI client
client = None

# Debug: show if OPENAI_API_KEY is visible in environment (without printing it)
print(f"Env OPENAI_API_KEY present: {bool(os.environ.get('OPENAI_API_KEY'))}")

# Try to import config - add error handling
try:
    from config import OPENAI_API_KEY
    # Initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Ensure env var is set so downstream libs (e.g., LangChain) use the same key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("OpenAI client initialized with API key from config.py")
except ImportError:
    print("""
    Error: config.py not found or OPENAI_API_KEY not defined!
    1. Make sure config.py exists
    2. Make sure it contains: OPENAI_API_KEY = "your-api-key-here"
    """)
    # Try to get from environment variable as fallback
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key") or os.environ.get("OpenAI_API_KEY")
    if api_key:
        print("Using OPENAI_API_KEY from environment variables instead")
        client = OpenAI(api_key=api_key)
        # Normalize env var for all libraries
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        print("WARNING: No OPENAI_API_KEY found in environment. Set it in Render service env vars.")

# Move the configure_logging function definition to the top
def configure_logging():
    """Configure logging to show only essential information"""
    # Set root logger to ERROR to suppress most messages
    logging.getLogger().setLevel(logging.ERROR)
    
    # Set our app logger to INFO for important app status messages
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Configure handler with a cleaner format
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    
    # Replace any existing handlers
    if logger.handlers:
        logger.handlers = []
    logger.addHandler(handler)
    
    # Suppress specific noisy loggers
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    
    return logger

# Then use the function
logger = configure_logging()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Global variables
mirror = None
camera = None

# Try to import Firebase
try:
    from firebase_config import FirebaseManager, SystemMemoryManager
    firebase_manager = FirebaseManager()
    system_memory = SystemMemoryManager(firebase_manager)
    firebase_available = True
    logger.info("Firebase and SystemMemory initialized successfully")
except Exception as e:
    logger.warning(f"Firebase/SystemMemory initialization error: {e}")
    firebase_available = False
    firebase_manager = None
    system_memory = None

def start_mirror(user_id=None):
    global mirror, camera
    
    # If mirror is already initialized, just update the user_id
    if mirror is not None:
        logger.info(f"Mirror already initialized, updating user_id to: {user_id}")
        mirror.user_id = user_id
        mirror.current_user = user_id
        return True
        
    try:
        # Import the WebMirror class
        from web_mirror import WebMirror
        
        # Use provided user_id or default to 'guest'
        if user_id is None:
            # Try to get from session only if in a request context
            try:
                user_id = session.get('user_id', 'guest')
            except RuntimeError:
                user_id = 'guest'
                
        logger.info(f"Initializing WebMirror with user_id: {user_id}")
        
        # Create the WebMirror instance
        mirror = WebMirror(user_id=user_id)
        
        # Add recording attributes
        mirror.recording = False
        mirror.should_stop_recording = False
        mirror.recording_start_time = None
        mirror.selected_mic_index = None
        
        # Initialize the camera for web streaming if not already initialized
        # Allow disabling camera on platforms without hardware (e.g., Render) via env flag
        if os.environ.get('DISABLE_CAMERA', '0') != '1':
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    logger.error("Could not open camera")
                else:
                    logger.info("Camera initialized successfully")
        
        logger.info("Mirror initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error starting mirror: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_frames():
    """Generate camera frames for streaming"""
    global camera
    
    # If camera is not initialized, try to initialize it
    if camera is None or not camera.isOpened():
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                logger.error("Could not open camera")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
                return
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            return
    
    while True:
        success, frame = camera.read()
        if not success:
            logger.error("Failed to read from camera")
            # Try to reinitialize camera
            camera.release()
            camera = cv2.VideoCapture(0)
            time.sleep(1)
            continue
        
        # Flip the frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Add a small delay to control frame rate
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    # Initialize mirror and camera on first page load
    global mirror, camera
    if mirror is None:
        # When called from a route, we can safely pass the session user_id
        user_id = session.get('user_id')
        start_mirror(user_id)
    
    # Use the existing mirror.html template instead of looking for index.html/login.html
    # Pass user_id if logged in
    return render_template('mirror.html', logged_in='user_id' in session, user_id=session.get('user_id', ''))

@app.route('/video_feed')
def video_feed():
    """Route for streaming video from the camera"""
    # Make sure camera is initialized
    global camera
    if camera is None or not camera.isOpened():
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(0)
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Helper function to create standardized JSON responses
def json_response(data, status_code=200):
    """Create a JSON response with proper headers"""
    response = make_response(jsonify(data), status_code)
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    """Handle user login with improved error handling"""
    # Add CORS headers
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
        return ('', 204, headers)
    
    try:
        # Try to get user_id from JSON first (for fetch API requests)
        data = request.get_json(silent=True)
        if data and 'user_id' in data:
            user_id = data['user_id']
        # Fall back to form data (for traditional form submissions)
        else:
            user_id = request.form.get('user_id')
            
        logger.info(f"Login attempt with user_id: '{user_id}'")
        
        if not user_id:
            logger.warning("Login failed: Empty user ID")
            return json_response({
                "success": False, 
                "error": "User ID cannot be empty"
            }, 400)
        
        # Check if user exists
        if firebase_available and firebase_manager.check_user_exists(user_id):
            # Update last login time
            firebase_manager.update_last_login(user_id)
            session['user_id'] = user_id
            
            # If mirror is already initialized, update its user
            if mirror:
                mirror.user_id = user_id
                mirror.current_user = user_id
                logger.info(f"Updated mirror user to: {user_id}")
            
            return json_response({"success": True})
        elif not firebase_available:
            # If Firebase is not available, just accept any user ID
            session['user_id'] = user_id
            logger.info(f"Firebase not available, accepting user ID: {user_id}")
            
            # If mirror is already initialized, update its user
            if mirror:
                mirror.user_id = user_id
                mirror.current_user = user_id
                
            return json_response({"success": True})
        else:
            logger.warning(f"Login failed: User '{user_id}' not found")
            return json_response({
                "success": False, 
                "error": f"User '{user_id}' not found"
            }, 404)
    except Exception as e:
        logger.error(f"Login error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return json_response({
            "success": False,
            "error": f"Error during login: {str(e)}"
        }, 500)

@app.route('/create_user', methods=['POST'])
def create_user():
    """Sign up endpoint. Accepts JSON {user_id} or form data."""
    # Try JSON first
    data = request.get_json(silent=True)
    user_id = None
    if data and 'user_id' in data:
        user_id = data['user_id']
    else:
        # Fallback to form
        user_id = request.form.get('user_id')

    if not user_id or not str(user_id).strip():
        return jsonify({"success": False, "error": "User ID cannot be empty"}), 400

    # Create new user
    if firebase_available and firebase_manager.create_user(user_id):
        session['user_id'] = user_id
        if mirror:
            mirror.user_id = user_id
            mirror.current_user = user_id
        return jsonify({"success": True})
    elif not firebase_available:
        # If Firebase is not available, accept any user ID
        session['user_id'] = user_id
        logger.info(f"Firebase not available, accepting user ID: {user_id}")
        if mirror:
            mirror.user_id = user_id
            mirror.current_user = user_id
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Failed to create user"}), 500

@app.route('/start', methods=['POST'])
def start():
    # Make sure user is logged in
    if 'user_id' not in session:
        return json_response({'error': 'User not logged in'}, 401)
    
    user_id = session['user_id']
    
    # Generate session start summary and proactive questions
    session_summary = None
    proactive_questions = []
    progress_insights = ""
    
    if system_memory:
        try:
            # Generate comprehensive session summary
            session_summary = system_memory.generate_session_start_summary(user_id)
            
            # Generate proactive questions
            proactive_questions = system_memory.generate_proactive_questions(user_id)
            
            # Get progress insights
            progress_insights = system_memory.get_progress_insights(user_id)
            
            # Log session start with memory context
            logger.info(f"Session started for {user_id} with system memory context")
            
        except Exception as e:
            logger.error(f"Error generating session memory: {e}")
    
    # Start the mirror with enhanced context
    if start_mirror(user_id):
        # Store session context in mirror instance
        if mirror and session_summary:
            mirror.session_memory = session_summary
            mirror.proactive_questions = proactive_questions
            mirror.progress_insights = progress_insights
        
        return json_response({
            'success': True, 
            'message': 'Mirror started successfully',
            'session_memory': session_summary,
            'proactive_questions': proactive_questions,
            'progress_insights': progress_insights
        })
    else:
        return json_response({'error': 'Failed to start mirror'}, 500)

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return ('', 204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        })
        
    global mirror
    
    # Use guest user if not logged in
    user_id = session.get('user_id', 'guest')
    
    if mirror is None:
        success = start_mirror(user_id)
        if not success:
            logger.error("Failed to start mirror in analyze route")
            return json_response({
                "success": False,
                "error": "Failed to start mirror",
                "emotion": "neutral",
                "posture": ["relaxed"]
            }, 500)
    
    try:
        # Use the web-specific analyze method
        logger.info(f"Starting web analysis for user: {user_id}")
        # Add more detailed logging
        logger.info(f"Mirror object state: recording={getattr(mirror, 'recording', None)}, user_id={getattr(mirror, 'user_id', None)}")
        
        # Check if the web_analyze_state method exists
        if not hasattr(mirror, 'web_analyze_state'):
            logger.error("web_analyze_state method not found on mirror object")
            return json_response({
                "success": False,
                "error": "Analysis method not available", 
                "details": "web_analyze_state not found",
                "emotion": "neutral",
                "posture": ["relaxed"]
            }, 500)
            
        # Prefer client-provided frame; if none, try camera only if available
        data = request.get_json(silent=True) or {}
        provided_image_b64 = data.get('image')
        frame = None
        if provided_image_b64:
            try:
                # Support data URLs or raw base64
                if 'base64,' in provided_image_b64:
                    provided_image_b64 = provided_image_b64.split('base64,', 1)[1]
                decoded = base64.b64decode(provided_image_b64)
                np_arr = np.frombuffer(decoded, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as decode_err:
                logger.error(f"Error decoding provided image: {decode_err}")

        if frame is None:
            # Fallback to server camera if not disabled
            if os.environ.get('DISABLE_CAMERA', '0') == '1':
                return json_response({
                    "success": False,
                    "error": "No image provided and camera disabled",
                    "emotion": "neutral",
                    "posture": ["relaxed"]
                }, 400)

            ret, frame = mirror.cap.read() if hasattr(mirror, 'cap') else (False, None)
            if not ret or frame is None:
                logger.error("Failed to capture frame from camera")
                return json_response({
                    "success": False,
                    "error": "Failed to capture frame",
                    "emotion": "neutral",
                    "posture": ["relaxed"]
                }, 500)
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Call web_analyze_state with the frame and duration
        result = mirror.web_analyze_state(frame, duration=2)
        
        # Extract emotion and posture from the result
        if isinstance(result, dict) and result.get('status') == 'success':
            emotion = result.get('emotion', 'neutral')
            posture = result.get('posture', ["relaxed"])
        else:
            logger.error(f"Analysis failed: {result}")
            emotion = 'neutral'
            posture = ["relaxed"]
            
        logger.info(f"Web analysis complete: {emotion}, {posture}")
        
        # Ensure posture is always a list
        if not isinstance(posture, list):
            posture = [posture] if posture else ["relaxed"]
            
        return json_response({
            "success": True,
            "emotion": emotion or "neutral",
            "posture": posture
        })
    except Exception as e:
        logger.error(f"Error in web analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        try:
            # Fall back to simplified analysis
            logger.info("Falling back to simplified analysis...")
            
            # Check if the simplified_analyze method exists
            if not hasattr(mirror, 'simplified_analyze'):
                logger.error("simplified_analyze method not found on mirror object")
                return json_response({
                    "success": False,
                    "emotion": "neutral",
                    "posture": ["relaxed"],
                    "error": "Analysis methods not available",
                    "details": str(e)
                }, 500)
                
            emotion, posture = mirror.simplified_analyze()
            logger.info(f"Simplified analysis complete: {emotion}, {posture}")
            
            # Ensure posture is always a list
            if not isinstance(posture, list):
                posture = [posture] if posture else ["relaxed"]
                
            return json_response({
                "success": True,
                "emotion": emotion or "neutral",
                "posture": posture,
                "note": "Used simplified analysis due to error"
            })
        except Exception as e2:
            logger.error(f"Error in simplified analysis: {e2}")
            logger.error(traceback.format_exc())
            return json_response({
                "success": False,
                "emotion": "neutral",
                "posture": ["relaxed"],
                "error": "Analysis failed completely",
                "details": f"{str(e)} / {str(e2)}"
            }, 500)

@app.route('/listen', methods=['POST'])
def listen():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"})
        
    global mirror
    if mirror is None:
        success = start_mirror(session.get('user_id'))
        if not success:
            return jsonify({"error": "Failed to start mirror"})
    
    # Use the web-specific listen method
    transcript = mirror.web_listen()
    if transcript:
        # Use the web-specific generate response method
        response_text = mirror.web_generate_response(transcript)
        
        # Get risk data for UI display
        risk_level = getattr(mirror, 'last_risk_level', 0)
        risk_indicators = getattr(mirror, 'last_risk_indicators', [])
        is_emergency = getattr(mirror, 'last_is_emergency', False)
        
        mirror.speak_message(response_text)
        return jsonify({
            "user_said": transcript,
            "mirror_response": response_text,
            "risk_level": risk_level,
            "risk_indicators": risk_indicators,
            "is_emergency": is_emergency
        })
    return jsonify({"error": "No speech detected"})

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('index'))
        
    user_id = session['user_id']
    
    if firebase_available and firebase_manager:
        # Get formatted interaction data
        interactions = firebase_manager.get_formatted_interactions(user_id)
        # Get interaction statistics
        stats = firebase_manager.get_interaction_stats(user_id)
    else:
        interactions = []
        stats = {
            'total_interactions': 0,
            'average_risk': 0,
            'common_emotions': [],
            'latest_interaction': None
        }
    
    return render_template('history.html', 
                         interactions=interactions,
                         stats=stats,
                         user_id=user_id)

@app.route('/debug')
def debug_page():
    """Debug page for emotion detection"""
    return render_template('debug.html')

@app.route('/debug_emotion', methods=['POST'])
def debug_emotion():
    """Debug route to test emotion detection"""
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"})
        
    global mirror, camera
    if mirror is None:
        success = start_mirror(session.get('user_id'))
        if not success:
            return jsonify({"error": "Failed to start mirror"})
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    # Capture a frame
    ret, frame = camera.read()
    if not ret:
        return jsonify({"error": "Failed to capture frame"})
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect emotion using the improved web analyzer
    result = mirror.web_analyze_state(frame, duration=1)
    if isinstance(result, dict) and result.get('status') == 'success':
        emotion = result.get('emotion', 'neutral')
    else:
        emotion = 'neutral'
    
    # Convert frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "emotion": emotion,
        "frame": frame_b64
    })

@app.route('/list_microphones', methods=['GET'])
def list_microphones():
    """List available microphones for the frontend to select from"""
    try:
        from speech_recognition import Microphone
        mic_list = Microphone.list_microphone_names()
        logger.info(f"Available microphones: {mic_list}")
        return jsonify({"success": True, "mics": mic_list})
    except Exception as e:
        logger.error(f"Error listing microphones: {e}")
        return jsonify({"error": str(e)})

@app.route('/start_listening', methods=['POST'])
def start_listening():
    """Start the listening process with guest support"""
    global mirror
    
    # Use guest user if not logged in
    user_id = session.get('user_id', 'guest')
    
    if not mirror:
        success = start_mirror(user_id)
        if not success:
            logger.error("Failed to initialize mirror in start_listening")
            return jsonify({"success": False, "error": "Failed to initialize mirror"})
    
    # Log the action
    logger.info(f"Starting listening for user: {user_id}")
    
    # Simply return success for now
    return jsonify({"success": True, "message": "Listening started"})

@app.route('/process_web_speech', methods=['POST'])
def process_web_speech():
    """Handle speech recognition results from Web Speech API"""
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"})
    
    global mirror
    if not mirror:
        return jsonify({"error": "Mirror not initialized"})
    
    try:
        data = request.json
        transcript = data.get('transcript')
        
        if not transcript:
            return jsonify({
                "success": False,
                "error": "No transcript provided"
            })
        
        logger.info(f"Processing Web Speech API transcript: {transcript}")
        
        # Generate response using the mirror
        emotion = mirror.locked_emotion or "neutral"
        posture = mirror.locked_body_language or []
        
        # Generate response (this now returns just the text)
        if hasattr(mirror, 'web_generate_response'):
            response_text = mirror.web_generate_response(transcript)
        else:
            response_text = mirror.generate_response(transcript, emotion, posture)
        
        # Get risk data for UI display
        risk_level = getattr(mirror, 'last_risk_level', 0)
        risk_indicators = getattr(mirror, 'last_risk_indicators', [])
        is_emergency = getattr(mirror, 'last_is_emergency', False)
        
        # Add to current session's interactions
        if hasattr(mirror, 'current_session'):
            interaction = {
                'timestamp': datetime.now(),
                'emotion': emotion,
                'posture': posture,
                'user_input': transcript,
                'ai_output': response_text,
                'risk_level': risk_level,
                'risk_indicators': risk_indicators,
                'is_emergency': is_emergency
            }
            mirror.current_session['interactions'].append(interaction)
        
        # Generate audio response
        try:
            # Use OpenAI's TTS API directly
            audio_data = None
            audio_format = "mp3"
            
            # Generate audio using OpenAI's TTS API
            audio_response = client.audio.speech.create(
                model="tts-1",
                voice="sage",
                input=response_text
            )
            
            # Save to temporary file
            temp_audio_path = "temp_audio.mp3"
            audio_response.stream_to_file(temp_audio_path)
            
            # Read the audio data
            with open(temp_audio_path, "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
            
            # Clean up temporary file
            os.remove(temp_audio_path)
            
            logger.info("Audio generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            audio_data = None
            audio_format = None
        
        # Include audio data in the response if available
        result = {
            "success": True,
            "transcript": transcript,
            "response": response_text,
            "risk_level": risk_level,
            "risk_indicators": risk_indicators,
            "is_emergency": is_emergency
        }
        
        # Add audio data to response if available
        if audio_data:
            result["audio_data"] = audio_data
            result["audio_format"] = audio_format
            logger.info("Audio data included in response")
        else:
            logger.warning("No audio data available for response")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing web speech: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/stop_listening', methods=['POST'])
def stop_listening():
    """Process recorded audio with improved speech recognition"""
    global mirror
    
    # Use guest user if not logged in
    user_id = session.get('user_id', 'guest')
    
    if not mirror:
        success = start_mirror(user_id)
        if not success:
            return jsonify({"success": False, "error": "Failed to initialize mirror"})
    
    try:
        # Check if we received an audio file
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file received"})
        
        audio_file = request.files['audio']
        
        # Save to a temporary file
        temp_dir = tempfile.gettempdir()
        webm_path = os.path.join(temp_dir, 'recording.webm')
        audio_file.save(webm_path)
        
        file_size = os.path.getsize(webm_path)
        logger.info(f"Audio saved to {webm_path}, size: {file_size} bytes")
        
        # Check if file is too small (likely empty or corrupt)
        if file_size < 1000:  # Less than 1KB
            logger.warning("Audio file too small, likely no speech")
            return jsonify({
                "success": False,
                "error": "No speech detected (audio file too small)",
                "transcript": "",
                "response": "I didn't hear anything. Could you try speaking louder?"
            })
        
        # Convert webm to wav
        try:
            import subprocess
            wav_path = os.path.join(temp_dir, 'recording.wav')
            
            # Debug: List the file to make sure it exists
            if not os.path.exists(webm_path):
                logger.error(f"WebM file doesn't exist at: {webm_path}")
                return jsonify({"success": False, "error": "WebM file not saved correctly"})
                
            logger.info(f"About to convert: {webm_path} to {wav_path}")
            
            # Use ffmpeg to convert webm to wav
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', webm_path, '-ar', '16000', '-ac', '1', wav_path],
                check=True, capture_output=True
            )
            
            logger.info(f"Conversion stdout: {result.stdout}")
            logger.info(f"Conversion stderr: {result.stderr}")
            
            if not os.path.exists(wav_path):
                logger.error("WAV file wasn't created by ffmpeg")
                return jsonify({"success": False, "error": "Failed to convert audio format"})
                
            logger.info(f"Converted to WAV: {wav_path}, size: {os.path.getsize(wav_path)} bytes")
            
            # Use our standalone speech recognition function
            transcript = recognize_speech_from_file(wav_path)
            logger.info(f"Initial transcript attempt: {transcript}")
            
            if not transcript:
                # If our function couldn't recognize speech, try the mirror methods
                logger.info("Initial transcript failed, trying mirror methods")
                if hasattr(mirror, 'process_audio_file'):
                    transcript = mirror.process_audio_file(wav_path)
                    logger.info(f"Transcript from mirror.process_audio_file: {transcript}")
                elif hasattr(mirror, 'whisper_process'):
                    transcript = mirror.whisper_process(wav_path)
                    logger.info(f"Transcript from mirror.whisper_process: {transcript}")
                elif hasattr(mirror, 'whisper_transcribe'):
                    # Try direct whisper transcription as a last resort
                    try:
                        with open(wav_path, "rb") as audio_file:
                            transcription = client.audio.transcribe("whisper-1", audio_file)
                            transcript = transcription.text
                            logger.info(f"Direct Whisper transcript: {transcript}")
                    except Exception as e:
                        logger.error(f"Direct Whisper transcription failed: {e}")
            
            # If we still don't have a transcript, use fallback
            if not transcript:
                logger.warning("All speech recognition methods failed")
                return jsonify({
                    "success": False,
                    "error": "Could not understand speech",
                    "transcript": "",
                    "response": "I couldn't understand what you said. Could you try speaking more clearly?"
                })
            
            # Generate response using the transcript
            if hasattr(mirror, 'web_generate_response'):
                response_text = mirror.web_generate_response(transcript)
            elif hasattr(mirror, 'generate_response'):
                emotion = getattr(mirror, 'locked_emotion', 'neutral')
                posture = getattr(mirror, 'locked_body_language', [])
                response_text = mirror.generate_response(transcript, emotion, posture)
            else:
                response_text = f"I heard you say: '{transcript}', but I'm not sure how to respond right now."
            
            # Extract risk data for UI display
            risk_level = getattr(mirror, 'last_risk_level', 0)
            risk_indicators = getattr(mirror, 'last_risk_indicators', [])
            is_emergency = getattr(mirror, 'last_is_emergency', False)
            
            # Generate audio for the response
            audio_data = None
            audio_format = "mp3"
            
            try:
                # Generate audio with OpenAI's TTS API
                audio_response = client.audio.speech.create(
                    model="tts-1",
                    voice="sage", 
                    input=response_text
                )
                
                # Save to temporary file
                temp_audio_path = "temp_fallback_audio.mp3"
                audio_response.stream_to_file(temp_audio_path)
                
                # Read the data as base64 for sending to client
                with open(temp_audio_path, "rb") as audio_file:
                    audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
                
                logger.info(f"Generated audio response, size: {os.path.getsize(temp_audio_path)} bytes")
            
            except Exception as audio_error:
                logger.error(f"Error generating audio response: {audio_error}")
                # Continue without audio
            
            # Clean up temporary files
            try:
                for path in [webm_path, wav_path]:
                    if os.path.exists(path):
                        os.remove(path)
            except Exception as e:
                logger.error(f"Error removing temporary files: {e}")
            
            # Return success with transcript, response, and audio data
            result = {
                "success": True,
                "transcript": transcript,
                "response": response_text,
                "risk_level": risk_level,
                "risk_indicators": risk_indicators,
                "is_emergency": is_emergency
            }
            
            if audio_data:
                result["audio_data"] = audio_data
                result["audio_format"] = audio_format
            
            return jsonify(result)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode() if hasattr(e, 'stderr') else 'No error output'}")
            return jsonify({"success": False, "error": "Failed to convert audio format"})
            
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)})

@app.route('/text_input', methods=['POST'])
def text_input():
    """Process text input when speech recognition fails"""
    global mirror
    
    # Use guest user if not logged in
    user_id = session.get('user_id', 'guest')
    
    if not mirror:
        success = start_mirror(user_id)
        if not success:
            return jsonify({"success": False, "error": "Failed to initialize mirror"})
    
    try:
        # Get text from request
        text = request.json.get('text', '')
        if not text:
            return jsonify({"success": False, "error": "No text provided"})
            
        logger.info(f"Processing TEXT INPUT for user {user_id}: {text}")
        
        # Get current state
        emotion = mirror.locked_emotion or "neutral"
        posture = mirror.locked_body_language or []
        if not isinstance(posture, list):
            posture = [posture] if posture else []
        
        # Generate response (this now returns just the text)
        if hasattr(mirror, 'web_generate_response'):
            response_text = mirror.web_generate_response(text)
        else:
            response_text = mirror.generate_response(text, emotion, posture)
        
        # Get risk data for UI display
        risk_level = getattr(mirror, 'last_risk_level', 0)
        risk_indicators = getattr(mirror, 'last_risk_indicators', [])
        is_emergency = getattr(mirror, 'last_is_emergency', False)
        
        # Add to current session's interactions
        if hasattr(mirror, 'current_session'):
            interaction = {
                'timestamp': datetime.now(),
                'emotion': emotion,
                'posture': posture,
                'user_input': text,
                'ai_output': response_text,
                'risk_level': risk_level,
                'risk_indicators': risk_indicators,
                'is_emergency': is_emergency
            }
            mirror.current_session['interactions'].append(interaction)
        
        # Optionally generate TTS for text-only input
        audio_data = None
        audio_format = "mp3"
        try:
            audio_response = client.audio.speech.create(
                model="tts-1",
                voice="sage",
                input=response_text
            )
            temp_audio_path = "temp_text_input_audio.mp3"
            audio_response.stream_to_file(temp_audio_path)
            with open(temp_audio_path, "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
            os.remove(temp_audio_path)
        except Exception as _tts_err:
            audio_data = None
            audio_format = None

        return jsonify({
            "success": True,
            "response": response_text,
            "emotion": emotion,
            "risk_level": risk_level,
            "risk_indicators": risk_indicators,
            "is_emergency": is_emergency,
            **({"audio_data": audio_data, "audio_format": audio_format} if audio_data else {})
        })
        
    except Exception as e:
        logger.error(f"Error processing text input: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/get_audio_level', methods=['GET'])
def get_audio_level():
    """Get the current audio input level"""
    global mirror
    if not mirror or not mirror.recording:
        return jsonify({"level": 0})
    
    # Check if we should be monitoring audio
    if not getattr(mirror, 'monitoring_audio', False):
        return jsonify({"level": 0})
    
    try:
        # Get the current audio level from the mirror
        level = mirror.get_audio_level()
        return jsonify({"level": level})
    except Exception as e:
        logger.error(f"Error getting audio level: {e}")
        return jsonify({"level": 0})

@app.route('/check_audio_level', methods=['POST'])
def check_audio_level():
    """Check current audio input level with increased sensitivity"""
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"})
    
    global mirror
    if not mirror or not mirror.recording:
        return jsonify({"error": "Not recording"})
    
    try:
        with sr.Microphone(device_index=mirror.selected_mic_index) as source:
            # Shorter duration for quicker response
            audio_data = mirror.recognizer.listen(source, timeout=0.1, phrase_time_limit=0.1)
            if audio_data:
                # Get raw audio data
                audio_array = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
                
                # Calculate RMS with increased sensitivity
                rms = np.sqrt(np.mean(np.square(audio_array)))
                audio_level = min(100, int((rms / 16384) * 100))  # More sensitive scaling
                
                # Adjust energy threshold if needed
                if audio_level > 10:
                    mirror.recognizer.energy_threshold = max(
                        100,  # Minimum threshold
                        min(mirror.recognizer.energy_threshold, rms * 1.2)  # Gradual adjustment
                    )
                
                return jsonify({
                    "level": audio_level,
                    "threshold": mirror.recognizer.energy_threshold,
                    "raw_rms": float(rms)
                })
    except Exception as e:
        logger.error(f"Error checking audio level: {e}")
    
    return jsonify({"level": 0})

@app.route('/test_microphone', methods=['POST'])
def test_microphone():
    """Test basic microphone functionality"""
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"})
    
    global mirror
    if not mirror:
        success = start_mirror(session.get('user_id'))
        if not success:
            return jsonify({"error": "Failed to initialize mirror"})
    
    try:
        # Get microphone index from request
        data = request.json or {}
        mic_index = data.get('mic_index')
        
        logger.info(f"Testing microphone with index: {mic_index}")
        
        # Try to record a short audio sample
        with sr.Microphone(device_index=mic_index) as source:
            logger.info("Microphone opened successfully")
            mirror.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info(f"Ambient noise level: {mirror.recognizer.energy_threshold}")
            
            return jsonify({
                "success": True,
                "message": "Microphone test successful",
                "energy_threshold": mirror.recognizer.energy_threshold
            })
            
    except Exception as e:
        logger.error(f"Microphone test failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })

def recognize_speech_from_file(audio_file_path):
    """Standalone speech recognition function using Whisper only"""
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        # Check if file exists and has content
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return None
            
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:  # Less than 1KB
            logger.error(f"Audio file too small ({file_size} bytes), likely no speech")
            return None
            
        logger.info(f"Processing audio file: {audio_file_path}, size: {file_size} bytes")
        
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise and record
            audio_data = recognizer.record(source)
            
            # Use Whisper API for recognition
            try:
                logger.info("Attempting Whisper API speech recognition")
                with open(audio_file_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
                    )
                    logger.info(f"Whisper recognition successful: {transcript.text}")
                    return transcript.text
            except Exception as e:
                logger.error(f"Whisper recognition failed: {str(e)}")
                return None
    except Exception as e:
        logger.error(f"Error in recognize_speech_from_file: {str(e)}")
        return None

def cleanup():
    """Cleanup resources before shutdown"""
    global mirror, camera
    
    # Release camera
    if camera is not None:
        camera.release()
        
    # Cleanup mirror instance
    if mirror is not None:
        if hasattr(mirror, 'cleanup'):
            mirror.cleanup()
        mirror = None

def log_interaction_summary(user_id, emotion, posture, transcript, response, risk_level=None, rag_articles=None):
    """Print a clean, formatted summary of the interaction to the terminal"""
    print("\n" + "-"*50)
    
    # Display current state
    print(f"CURRENT STATE:")
    print(f"Emotion: {emotion if emotion else 'Not detected'}")
    print(f"Body Language: {', '.join(posture) if isinstance(posture, list) and posture else 'Not detected'}")
    if risk_level is not None:
        print(f"Risk Level: {risk_level}%")
    else:
        print("Risk Level: Not assessed")
    
    print("\n" + "-"*50)
    
    # Display user input and AI response
    print("INTERACTION:")
    print(f"User: {transcript}")
    print(f"AI: {response}")
    
    print("\n" + "-"*50)
    
    # Display RAG articles if available
    print("RAG ARTICLES USED:")
    if rag_articles and len(rag_articles) > 0:
        for article in rag_articles:
            print(f"- {article}")
    else:
        print("No articles used")
    
    print("\n" + "-"*50)
    
    # Get session count from Firebase
    try:
        firebase_manager = FirebaseManager()
        sessions = firebase_manager.get_user_session_history(user_id)
        print(f"Session #{len(sessions)} saved successfully")
        
        # Get progress analysis
        progress = firebase_manager.analyze_user_progress(user_id)
        if progress and progress['trend'] != 'insufficient_data':
            print("\nPROGRESS SUMMARY:")
            print(f"Total Sessions: {progress['session_count']}")
            print(f"Average Risk Level: {progress['avg_risk_level']:.1f}%")
            print(f"Common Emotions: {', '.join(progress['common_emotions'])}")
            
            # Add trend analysis with more detail
            if progress['trend'] == 'improving':
                print("\nEmotional Progress:")
                print("- User shows positive emotional progress over recent sessions")
                print("- Risk levels have decreased significantly")
                print("- Emotional stability has improved")
            elif progress['trend'] == 'declining':
                print("\nAreas Needing Attention:")
                print("- User may need additional support based on recent sessions")
                print("- Risk levels have increased")
                print("- Emotional patterns indicate increased stress")
            else:
                print("\nStability Assessment:")
                print("- User's emotional state has remained stable across sessions")
                print("- Risk levels are consistent")
                print("- Emotional patterns show good consistency")
    except Exception as e:
        print("Error accessing session history")
    
    print("-"*50 + "\n")

# Let's add a function to generate the mental health assessment prompt
def generate_mental_health_prompt(text, emotion, body_language):
    """Generate a prompt for mental health risk assessment"""
    try:
        return MentalHealthPrompts.get_main_prompt(text, emotion, body_language)
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        raise

@app.route('/end_session', methods=['POST'])
def end_session():
    try:
        # Save current session to Firebase if available
        if hasattr(app, 'mirror') and app.mirror:
            app.mirror.save_session()
            
            # Initialize new session
            app.mirror.current_session = {
                'start_time': datetime.now().isoformat(),
                'interactions': [],
                'last_input': None,
                'last_response': None
            }
            
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error ending session: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logout', methods=['POST'])
def logout():
    try:
        # Save current session to Firebase if available
        if hasattr(app, 'mirror') and app.mirror:
            app.mirror.save_session()
            
        # Clear session
        session.clear()
        
        # Reset mirror to guest state
        app.mirror = None
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if not data or 'feedback' not in data:
            return jsonify({"success": False, "error": "Feedback field is required"}), 400
            
        feedback = data['feedback']
        if not feedback or not feedback.strip():
            return jsonify({"success": False, "error": "Feedback cannot be empty"}), 400

        # Just store the feedback - no prompt adjustments
        if firebase_available and firebase_manager:
            try:
                firebase_manager.save_feedback(feedback)
            except Exception as e:
                print(f"Error saving feedback to Firebase: {e}")
        
        return jsonify({
            "success": True, 
            "message": "Feedback received"
        })
        
    except Exception as e:
        print(f"Error processing feedback: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/get_session_summary', methods=['GET'])
def get_session_summary():
    """Get the current session summary and memory context"""
    if 'user_id' not in session:
        return json_response({'error': 'User not logged in'}, 401)
    
    user_id = session['user_id']
    
    if not system_memory:
        return json_response({'error': 'System memory not available'}, 503)
    
    try:
        # Get comprehensive session summary
        session_summary = system_memory.generate_session_start_summary(user_id)
        proactive_questions = system_memory.generate_proactive_questions(user_id)
        progress_insights = system_memory.get_progress_insights(user_id)
        
        return json_response({
            'session_summary': session_summary,
            'proactive_questions': proactive_questions,
            'progress_insights': progress_insights,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        return json_response({'error': 'Failed to generate session summary'}, 500)

@app.route('/get_progress_report', methods=['GET'])
def get_progress_report():
    """Get a detailed progress report for the user"""
    if 'user_id' not in session:
        return json_response({'error': 'User not logged in'}, 401)
    
    user_id = session['user_id']
    
    try:
        # Get interaction statistics and progress analysis
        if firebase_manager:
            stats = firebase_manager.get_interaction_stats(user_id)
            progress_analysis = firebase_manager.analyze_user_progress(user_id)
        else:
            stats = None
            progress_analysis = None
        
        # Get insights from system memory
        insights = ""
        if system_memory:
            insights = system_memory.get_progress_insights(user_id)
        
        return json_response({
            'stats': stats,
            'progress_analysis': progress_analysis,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting progress report: {e}")
        return json_response({'error': 'Failed to generate progress report'}, 500)

@app.route('/update_session_feedback', methods=['POST'])
def update_session_feedback():
    """Update session with user feedback about interaction quality"""
    if 'user_id' not in session:
        return json_response({'error': 'User not logged in'}, 401)
    
    try:
        data = request.get_json()
        feedback_rating = data.get('rating')  # 1-5 scale
        feedback_text = data.get('feedback', '')
        interaction_id = data.get('interaction_id')
        
        user_id = session['user_id']
        
        # Store feedback for system memory improvement
        if firebase_manager:
            feedback_data = {
                'user_id': user_id,
                'interaction_id': interaction_id,
                'rating': feedback_rating,
                'feedback_text': feedback_text,
                'timestamp': firestore.SERVER_TIMESTAMP
            }
            
            firebase_manager.db.collection('feedback').add(feedback_data)
        
        return json_response({'success': True, 'message': 'Feedback recorded'})
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return json_response({'error': 'Failed to record feedback'}, 500)

if __name__ == '__main__':
    try:
        # Initialize mirror only once at startup
        if mirror is None:
            start_mirror('guest')  # Start with guest user
        
        # Register cleanup handler
        import atexit
        atexit.register(cleanup)
        
        # Start Flask app
        port = int(os.environ.get('PORT', 8080))
        app.run(debug=True, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        cleanup()
    print(f"Server running on http://localhost:{port}") 