import cv2
import numpy as np
import time
from openai import OpenAI
from datetime import datetime
# Optional TTS engine (fallback)
try:
    import pyttsx3  # type: ignore
    _PYTTSX3_AVAILABLE = True
except Exception:
    pyttsx3 = None  # type: ignore
    _PYTTSX3_AVAILABLE = False
import os
_USE_DEEPFACE = os.environ.get("USE_DEEPFACE", "0") == "1"
try:
    if _USE_DEEPFACE:
        from deepface import DeepFace
    else:
        DeepFace = None  # type: ignore
except Exception:
    DeepFace = None  # type: ignore
from firebase_config import FirebaseManager
# Optional PyAudio for local microphone capture (not required in web mode)
try:
    import pyaudio  # type: ignore
    _PYAUDIO_AVAILABLE = True
except Exception:
    pyaudio = None  # type: ignore
    _PYAUDIO_AVAILABLE = False
import wave
import os
import threading
from queue import Queue
from speech_recognition import Recognizer, Microphone
import speech_recognition
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
import firebase_admin
from firebase_admin import firestore
import re
import os
from flask import send_file, Flask, render_template, jsonify, request, Response
import io
from pydub import AudioSegment
from pydantic_compat import PydanticCompatBase as BaseModel, compat_validator
import sys
import traceback
import tempfile
import pkg_resources
import base64
import json
from prompts import MentalHealthPrompts
import ast

# Resolve OpenAI configuration with safe fallbacks
# Try config.py first, otherwise use environment variables
OPENAI_API_KEY = None
OPENAI_TTS_VOICE = None
OPENAI_TTS_MODEL = None
try:
    from config import OPENAI_API_KEY as _CONF_KEY, OPENAI_TTS_VOICE as _CONF_VOICE, OPENAI_TTS_MODEL as _CONF_MODEL  # type: ignore
    OPENAI_API_KEY = _CONF_KEY
    OPENAI_TTS_VOICE = _CONF_VOICE
    OPENAI_TTS_MODEL = _CONF_MODEL
except Exception:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key") or os.environ.get("OpenAI_API_KEY")
    OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "sage")
    OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")

# Initialize OpenAI client once globally and normalize env var if key is available
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)
    client.timeout = 15  # 15 seconds timeout
else:
    # Create a client that will likely fail on network calls, but avoid crashing import
    client = OpenAI()
    client.timeout = 15

# Check OpenAI library version
openai_version = pkg_resources.get_distribution("openai").version
print(f"OpenAI Python library version: {openai_version}")
USING_OPENAI_V1 = int(openai_version.split('.')[0]) >= 1

# Set global timeout for all HTTP requests to prevent freezing
requests.adapters.DEFAULT_TIMEOUT = 15  # 15 seconds timeout

# Do not hard-exit when config.py is missing; we already handled env fallbacks above

class SmartMirror:
    def __init__(self, web_mode=None):
        """Initialize the Smart Mirror with all required components"""
        # First prompt for mode if not specified
        if web_mode is None:
            self.web_mode = self.prompt_mode_selection()
        else:
            self.web_mode = web_mode

        self.speech_active = False
        self.analysis_complete = False

        # Initialize OpenAI first

        # Initialize OpenAI TTS (skip heavy startup test unless explicitly enabled)
        self.use_openai_tts = False
        try:
            if OPENAI_API_KEY:
                if os.environ.get("ENABLE_TTS_STARTUP_TEST", "0") == "1":
                    print(f"Testing OpenAI TTS with voice: {OPENAI_TTS_VOICE}, model: {OPENAI_TTS_MODEL}")
                    test_response_content = client.audio.speech.create(
                        model=OPENAI_TTS_MODEL,
                        voice=OPENAI_TTS_VOICE,
                        input="Hello, I'm testing OpenAI text to speech."
                    ).content
                    test_file = "test_audio.mp3"
                    with open(test_file, "wb") as f:
                        f.write(test_response_content)
                    if os.path.getsize(test_file) > 0:
                        self.use_openai_tts = True
                        print(f"OpenAI TTS configured successfully with voice: {OPENAI_TTS_VOICE}")
                else:
                    # Assume available; actual calls handle errors gracefully
                    self.use_openai_tts = True
        except Exception as e:
            self.use_openai_tts = False
            print(f"OpenAI TTS not configured, using fallback TTS: {e}")

        # Initialize text-to-speech engine (fallback) if available
        self.engine = None
        if _PYTTSX3_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
            except Exception:
                self.engine = None

        # Initialize Firebase
        try:
            self.firebase_manager = FirebaseManager()
            self.user_id = None
            self.current_user = None
            print("Firebase initialized successfully")
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
            self.firebase_manager = None
            self.user_id = "guest"
            self.current_user = "guest"

        # Initialize camera
        print("\nInitializing camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

        # Initialize audio only if not in web mode
        if not self.web_mode:
            self._initialize_audio()
        else:
            # For web mode, just initialize basic audio parameters
            self.audio_format = pyaudio.paInt16
            self.channels = 1
            self.sample_rate = 16000
            self.chunk = 1024
            self.audio = pyaudio.PyAudio()
            self.stream = None
            self.audio_frames = []
            self.recording = False

        # Initialize face detection and other components
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_frequency = 10
        self.frame_count = 0

        # Initialize speech queue
        self.speech_queue = Queue()
        self.speech_thread = threading.Thread(target=self._process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        # Initialize speech recognition
        self.recognizer = speech_recognition.Recognizer()
        if not self.web_mode:  # Only set up local mic for non-web mode
            try:
                with speech_recognition.Microphone(device_index=self.input_device_index) as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    print(f"Speech recognition configured with energy_threshold={self.recognizer.energy_threshold}")
            except Exception as e:
                print(f"Error configuring speech recognition: {e}")

        # Initialize DeepFace
        print("Loading emotion detection model...")
        try:
            # Initialize variables for emotion tracking
            self.emotion_detection_frequency = 60  # Only detect emotions every X frames (increased from 30)
            self.frame_count = 0
            self.last_emotion = None
            self.debug_emotions = {}
        except Exception as e:
            print(f"Warning: Error initializing emotion detection: {e}")
            print("Continuing with basic functionality...")

        # State variables
        self.current_message = "Press SPACE to analyze posture and get feedback"
        self.last_analysis_time = 0
        self.analysis_cooldown = 3.0
        self.last_emotions = []

        # Emotion mapping for more supportive language
        self.emotion_mapping = {
            'angry': 'frustrated',
            'disgust': 'uncomfortable',
            'fear': 'anxious',
            'happy': 'happy',
            'sad': 'down',
            'surprise': 'surprised',
            'neutral': 'calm'
        }

        # Add new state variables for locking
        self.locked_emotion = None
        self.locked_body_language = None
        self.ready_to_send = False
        self.analyzing = False
        self.last_api_call = time.time()
        self.api_cooldown = 3.0

        # Initialize Firebase
        self.firebase = FirebaseManager()

        # Add user login prompt
        self.prompt_user_login()

        # Initialize audio recording settings with lower sample rate
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.audio_frames = []
        self.last_recording = None
        self.stream = None
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000  # Reduced from 44100
        self.chunk = 512  # Reduced from 1024
        self.max_record_seconds = 10  # Maximum recording time
        self.recording_start_time = None

        # Add speech queue for async TTS
        self.speech_queue = Queue()
        self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self.speech_thread.start()

        # Add speech recognition with improved settings for better detection
        print("Initializing speech recognition...")
        self.recognizer = Recognizer()
        self.mic = Microphone(sample_rate=16000)  # Use lower sample rate for better performance

        # Configure speech recognition for better speech detection
        self.recognizer.energy_threshold = 300  # Lower threshold to detect quieter speech
        self.recognizer.dynamic_energy_threshold = True  # Adjust for ambient noise
        self.recognizer.pause_threshold = 2.0  # Longer pause threshold (2 seconds)
        self.recognizer.phrase_threshold = 0.3  # More sensitive phrase detection
        self.recognizer.non_speaking_duration = 1.0  # Shorter non-speaking duration

        # Print microphone list to help with debugging
        print("\nAvailable microphones:")
        mic_list = self.mic.list_microphone_names()
        for i, mic_name in enumerate(mic_list):
            print(f"{i}: {mic_name}")
        print(f"Using default microphone: {self.mic.device_index}")

        # Conversation state
        self.in_conversation = False
        self.last_response = None
        self.conversation_history = []
        self.current_session = {
            'start_time': datetime.now(),
            'interactions': []
        }

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for more accurate but slower model
            min_detection_confidence=0.5
        )

        # Initialize MediaPipe Face Mesh for detailed facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Enhanced UI Elements
        self.overlay_alpha = 0.3
        self.colors = {
            'happy': (46, 204, 113),     # Emerald Green
            'sad': (231, 76, 60),        # Pomegranate Red
            'angry': (192, 57, 43),      # Dark Red
            'neutral': (241, 196, 15),   # Sun Yellow
            'surprised': (230, 126, 34),  # Carrot Orange
            'fear': (142, 68, 173),      # Wisteria Purple
            'disgust': (211, 84, 0)      # Pumpkin Orange
        }

        # Add animation states
        self.animation_state = {
            'progress': 0,
            'target': 0,
            'speed': 5
        }

        # Initialize buffers for temporal posture analysis
        self.landmark_history_size = 10  # Store last 10 frames of landmarks
        self.nose_history = []
        self.left_shoulder_history = []
        self.right_shoulder_history = []
        self.left_hip_history = []
        self.right_hip_history = []

    def _initialize_audio(self):
        """Initialize audio components for local mode"""
        print("\nInitializing audio system...")

        # Initialize audio parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()

        # Print available microphones
        print("\nAvailable microphones:")
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"{i}: {device_info['name']}")
            except Exception as e:
                print(f"Error getting device info for index {i}: {e}")

        # Initialize microphone
        try:
            default_info = self.audio.get_default_input_device_info()
            if default_info and default_info['maxInputChannels'] > 0:
                self.input_device_index = default_info['index']
                print(f"\nUsing default microphone: {default_info['name']}")
                print(f"Device index: {self.input_device_index}")

                # Test microphone access
                test_stream = self.audio.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.chunk
                )
                test_stream.close()
                print("Default microphone initialized successfully")
            else:
                raise Exception("Default device has no input channels")

        except Exception as e:
            print(f"\nError initializing default microphone: {e}")
            print("Attempting to find working microphone...")
            self.input_device_index = self._find_working_microphone()
            if self.input_device_index is None:
                print("WARNING: No working microphone found!")

        # Initialize speech recognition
        self.recognizer = speech_recognition.Recognizer()
        try:
            with speech_recognition.Microphone(device_index=self.input_device_index) as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print(f"Speech recognition configured with energy_threshold={self.recognizer.energy_threshold}")
        except Exception as e:
            print(f"Error configuring speech recognition: {e}")

        # Initialize speech queue
        self.speech_queue = Queue()
        self.speech_thread = threading.Thread(target=self._process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        # Initialize audio recording settings
        self.recording = False
        self.audio_frames = []
        self.last_recording = None
        self.stream = None

        # Initialize basic audio parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()

        # Initialize stream
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.input_device_index
        )
        self.recording = True
        self.audio_frames = []
        self.recording_start_time = time.time()

    def create_modern_overlay(self, frame, emotion=None, confidence=None):
        """Create a modern, animated overlay with emotion data"""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Create gradient background for top bar
        gradient = np.zeros((100, width, 3), dtype=np.uint8)
        for i in range(100):
            alpha = 1 - (i / 100)
            gradient[i] = [int(40 * alpha)] * 3

        # Blend gradient with top of frame
        overlay[0:100, 0:width] = cv2.addWeighted(
            overlay[0:100, 0:width], 1-self.overlay_alpha,
            gradient, self.overlay_alpha, 0
        )

        if emotion and confidence:
            # Animate progress bar
            self.animation_state['target'] = confidence
            self.animation_state['progress'] += (
                self.animation_state['target'] - self.animation_state['progress']
            ) * 0.1

            # Create animated emotion bar
            bar_width = int(width * 0.8)
            bar_height = 30
            bar_x = int((width - bar_width) / 2)
            bar_y = 50

            # Draw background bar with rounded corners
            cv2.rectangle(overlay, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (40, 40, 40), 
                         cv2.FILLED,
                         lineType=cv2.LINE_AA)

            # Draw progress with animated fill
            progress_width = int(bar_width * (self.animation_state['progress'] / 100))
            color = self.colors.get(emotion.lower(), (255, 255, 255))

            if progress_width > 0:
                cv2.rectangle(overlay,
                             (bar_x, bar_y),
                             (bar_x + progress_width, bar_y + bar_height),
                             color,
                             cv2.FILLED,
                             lineType=cv2.LINE_AA)

            # Add glowing text effect
            text = f"{emotion.title()} ({self.animation_state['progress']:.1f}%)"
            font_scale = 0.7
            thickness = 2

            # Draw text glow
            for i in range(3):
                cv2.putText(overlay, text,
                           (bar_x, bar_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale,
                           (color[0]//2, color[1]//2, color[2]//2),
                           thickness + i*2,
                           lineType=cv2.LINE_AA)

            # Draw main text
            cv2.putText(overlay, text,
                       (bar_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale,
                       (255, 255, 255),
                       thickness,
                       lineType=cv2.LINE_AA)

        return cv2.addWeighted(frame, 1-self.overlay_alpha, overlay, self.overlay_alpha, 0)

    def analyze_face_and_posture(self, frame):
        """Analyze facial expression and body posture in a frame with improved smoothing and confidence threshold"""
        try:
            # Initialize emotion buffer if not exists
            if not hasattr(self, 'emotion_buffer'):
                self.emotion_buffer = []
            EMOTION_BUFFER_SIZE = 15  # Increased from 5 to 15 for better smoothing
            EMOTION_CONFIDENCE_THRESHOLD = 0.6  # Only report if average confidence > 60%
            
            # Get face analysis
            results = self.face_detector.process(frame)
            emotion_result = None
            
            if results.detections:
                for detection in results.detections:
                    # Get face landmarks
                    face_landmarks = detection.location_data.relative_keypoints
                    
                    # Convert landmarks to pixel coordinates
                    h, w = frame.shape[:2]
                    face_points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks]
                    
                    # Extract face ROI
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    face_roi = frame[y:y+height, x:x+width]
                    if face_roi.size > 0:  # Check if ROI is valid
                        # Analyze emotion
                        emotion = self.emotion_classifier.predict(face_roi)
                        if emotion:
                            # Add to buffer
                            self.emotion_buffer.append(emotion)
                            # Keep buffer size fixed
                            if len(self.emotion_buffer) > EMOTION_BUFFER_SIZE:
                                self.emotion_buffer.pop(0)
                            # Only process if buffer is full
                            if len(self.emotion_buffer) == EMOTION_BUFFER_SIZE:
                                # Count occurrences and sum confidences
                                emotion_counts = {}
                                emotion_confidences = {}
                                for e in self.emotion_buffer:
                                    e_label, e_conf = e
                                    if e_label in emotion_counts:
                                        emotion_counts[e_label] += 1
                                        emotion_confidences[e_label] += e_conf
                                    else:
                                        emotion_counts[e_label] = 1
                                        emotion_confidences[e_label] = e_conf
                                # Get most frequent emotion and its average confidence
                                most_frequent = max(emotion_counts.items(), key=lambda x: x[1])[0]
                                avg_confidence = emotion_confidences[most_frequent] / emotion_counts[most_frequent]
                                if avg_confidence >= EMOTION_CONFIDENCE_THRESHOLD:
                                    emotion_result = (most_frequent, avg_confidence)
                                else:
                                    emotion_result = ("uncertain", avg_confidence)
                            elif len(self.emotion_buffer) == 1:  # If first frame, use it
                                e_label, e_conf = emotion
                                if e_conf >= EMOTION_CONFIDENCE_THRESHOLD:
                                    emotion_result = emotion
                                else:
                                    emotion_result = ("uncertain", e_conf)
            # Improved posture smoothing
            if not hasattr(self, 'posture_buffer'):
                self.posture_buffer = []
            POSTURE_BUFFER_SIZE = 10
            posture_result, _ = self.analyze_posture(frame)
            if posture_result:
                self.posture_buffer.append(posture_result)
                if len(self.posture_buffer) > POSTURE_BUFFER_SIZE:
                    self.posture_buffer.pop(0)
                # Use the most common posture cue in the buffer
                from collections import Counter
                flat_postures = [item for sublist in self.posture_buffer for item in sublist]
                if flat_postures:
                    most_common_posture = Counter(flat_postures).most_common(1)[0][0]
                    posture_result = [most_common_posture]
            return emotion_result, posture_result
        except Exception as e:
            print(f"Error in face/posture analysis: {e}")
            return None, None

    def visualize_emotions(self, frame, emotions):
        """Create a simplified visual representation of detected emotions"""
        # Only show top 3 emotions to reduce processing
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]

        height, width = frame.shape[:2]
        chart_width = 200  # Reduced from 250
        chart_height = 120  # Reduced from 180
        margin = 20
        bar_height = 25
        spacing = 8

        # Create chart area with rounded corners
        chart_x = width - chart_width - margin
        chart_y = height - chart_height - margin

        # Draw simplified background
        cv2.rectangle(frame,
                     (chart_x-10, chart_y-10),
                     (chart_x + chart_width+10, chart_y + chart_height+10),
                     (0, 0, 0),
                     cv2.FILLED)

        # Draw emotion bars with simplified rendering
        y_offset = chart_y + 15
        max_width = chart_width - 80

        for emotion, value in top_emotions:
            # Draw label
            cv2.putText(frame,
                       f"{emotion.title()}",
                       (chart_x + 5, y_offset + bar_height - 8),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 255, 255),
                       1,
                       lineType=cv2.LINE_AA)

            # Draw bar
            bar_width = int(max_width * (value / 100))
            color = self.colors.get(emotion.lower(), (255, 255, 255))

            # Draw background bar
            cv2.rectangle(frame,
                         (chart_x + 70, y_offset),
                         (chart_x + 70 + max_width, y_offset + bar_height - 8),
                         (40, 40, 40),
                         cv2.FILLED)

            # Draw value bar (simplified)
            if bar_width > 0:
                cv2.rectangle(frame,
                             (chart_x + 70, y_offset),
                             (chart_x + 70 + bar_width, y_offset + bar_height - 8),
                             color,
                             cv2.FILLED)

            # Draw percentage text (simplified)
            percentage_text = f"{value:.1f}%"
            cv2.putText(frame,
                       percentage_text,
                       (chart_x + 75 + bar_width, y_offset + bar_height - 8),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.4,
                       (255, 255, 255),
                       1,
                       lineType=cv2.LINE_AA)

            y_offset += bar_height + spacing

    def get_feedback(self, posture_cues, emotion):
        """Get AI feedback based on verbal input and emotion"""
        import json # Ensure json is imported
        import ast  # For ast.literal_eval

        # Helper function to robustly extract the actual response string
        def _extract_conversational_text_robust(content_val):
            current_content = content_val
            max_loops = 3 # Safety break for recursion depth

            for _ in range(max_loops):
                if not isinstance(current_content, str):
                    # If it's already a dict, try to get 'response'
                    if isinstance(current_content, dict) and 'response' in current_content:
                        current_content = current_content['response'] # Continue loop with this new content
                        continue
                    return str(current_content) # Not a string, or a dict without 'response', convert to string

                stripped_content = current_content.strip()
                if not (stripped_content.startswith('{') and stripped_content.endswith('}')):
                    return current_content # Not a JSON-like or dict-like string, return as is

                parsed_successfully = False
                # Try json.loads first
                try:
                    parsed_data = json.loads(stripped_content)
                    parsed_successfully = True
                except json.JSONDecodeError:
                    # If json.loads fails, try ast.literal_eval for Python dict-like strings
                    try:
                        parsed_data = ast.literal_eval(stripped_content)
                        if not isinstance(parsed_data, dict): # Ensure ast.literal_eval resulted in a dict
                            return current_content # ast.literal_eval didn't return a dict, return original
                        parsed_successfully = True
                    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                        return current_content # Both parsing attempts failed

                if parsed_successfully:
                    if 'response' in parsed_data:
                        current_content = parsed_data['response'] # Update and loop for further unwrapping
                    else:
                        # Parsed, but no 'response' key. Assume this string itself is the intended message.
                        return current_content
                else:
                    # Should not be reached if logic is correct, but as a fallback:
                    return current_content
            # If loops finished (e.g., max_loops hit), return current state of content
            # Ensure final output is a string
            if isinstance(current_content, dict) and 'response' in current_content:
                 return str(current_content['response'])
            return str(current_content)

        try:
            prompt = MentalHealthPrompts.get_main_prompt(
                self.current_input,
                emotion,
                posture_cues
            )

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            if response is None:
                raise Exception("API call returned None")

            raw_llm_content = response.choices[0].message.content
            
            # Extract risk assessment data from the response
            initial_risk_level = 0
            initial_risk_indicators = []
            initial_is_emergency = False
            initial_conversational_candidate = raw_llm_content # Default if parsing fails

            try:
                response_data = json.loads(raw_risk_response)
                initial_risk_level = response_data.get('risk_level', 0)
                initial_risk_indicators = response_data.get('risk_indicators', [])
                initial_is_emergency = response_data.get('is_emergency', False)
                if 'response' in response_data:
                    initial_conversational_candidate = response_data['response']
                else:
                    initial_conversational_candidate = raw_llm_content
            except json.JSONDecodeError:
                # If primary parsing fails, try ast.literal_eval
                try:
                    response_data = ast.literal_eval(raw_llm_content)
                    if isinstance(response_data, dict):
                        initial_risk_level = response_data.get('risk_level', 0)
                        initial_risk_indicators = response_data.get('risk_indicators', [])
                        initial_is_emergency = response_data.get('is_emergency', False)
                        if 'response' in response_data:
                            initial_conversational_candidate = response_data['response']
                        else:
                            initial_conversational_candidate = raw_llm_content
                    # If not a dict, initial_conversational_candidate remains raw_llm_content, risk defaults to 0
                except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                    # Both parsing failed for the main response, use raw content and default risks
                    pass # Defaults are already set

            self.last_risk_level = initial_risk_level
            self.last_risk_indicators = initial_risk_indicators
            self.last_is_emergency = initial_is_emergency

            final_conversational_text = ""

            # Only use crisis protocol with risk level >= 90
            if initial_risk_level >= 90: # Crisis
                crisis_prompt_content = MentalHealthPrompts.get_crisis_prompt(
                    self.current_input, emotion, posture_cues
                )
                crisis_response_raw = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": crisis_prompt_content}],
                    temperature=0.3
                ).choices[0].message.content
                final_conversational_text = _extract_conversational_text_robust(crisis_response_raw)
                return final_conversational_text
            # Only use high risk protocol with risk level >= 80
            elif initial_risk_level >= 80: # High Risk
                high_risk_prompt_content = MentalHealthPrompts.get_high_risk_prompt(
                    self.current_input, emotion, posture_cues
                )
                high_risk_response_raw = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": high_risk_prompt_content}],
                    temperature=0.3
                ).choices[0].message.content
                final_conversational_text = _extract_conversational_text_robust(high_risk_response_raw)
                return final_conversational_text
            # For all other risk levels (0-79), use standard response
            else: # Low/Moderate Risk
                final_conversational_text = _extract_conversational_text_robust(initial_conversational_candidate)
                return final_conversational_text

        except Exception as e:
            print(f"Error in get_feedback: {e}")
            traceback.print_exc() # Print full traceback for debugging
            self.last_risk_level = 0
            self.last_risk_indicators = []
            self.last_is_emergency = False
            return "I'm having trouble processing that right now. Could you try rephrasing?"

    def _process_speech_queue(self):
        """Process speech queue in background thread with improved OpenAI TTS handling"""
        while True:
            try:
                # Get message from queue (blocking)
                message = self.speech_queue.get()

                if self.use_openai_tts:
                    try:
                        print("[TTS] Using OpenAI TTS for speech...")

                        # Split long messages to avoid OpenAI TTS limits (4096 characters)
                        max_chars = 4000
                        message_parts = []

                        if len(message) > max_chars:
                            # Split by sentences
                            sentences = re.split(r'(?<=[.!?])\s+', message)
                            current_part = ""

                            for sentence in sentences:
                                if len(current_part) + len(sentence) < max_chars:
                                    current_part += sentence + " "
                                else:
                                    message_parts.append(current_part.strip())
                                    current_part = sentence + " "

                            if current_part:
                                message_parts.append(current_part.strip())
                        else:
                            message_parts = [message]

                        # Process each part
                        for part in message_parts:
                            try:
                                # Generate audio with OpenAI TTS
                                audio = generate_speech(
                                    text=part, 
                                    voice=OPENAI_TTS_VOICE,
                                    model=OPENAI_TTS_MODEL
                                )

                                # Add a small pause between parts if there are multiple
                                if len(message_parts) > 1:
                                    time.sleep(0.5)
                            except Exception as inner_e:
                                print(f"OpenAI TTS error in part processing: {inner_e}")
                                # Try fallback for this part
                                self._speak_with_fallback(part)

                    except Exception as e:
                        print(f"OpenAI TTS error: {e}")
                        print("Falling back to pyttsx3...")
                        self._speak_with_fallback(message)
                else:
                    self._speak_with_fallback(message)

                # Mark task as done
                self.speech_queue.task_done()

            except Exception as e:
                print(f"Error in speech queue processing: {e}")
                time.sleep(0.1)  # Prevent CPU spinning on errors

    def _speak_with_fallback(self, message):
        """Use pyttsx3 as fallback TTS"""
        try:
            if self.engine is None:
                print("Fallback TTS unavailable (pyttsx3 not installed)")
                return
            # Configure voice safely
            try:
                voices = self.engine.getProperty('voices')
                if voices and len(voices) > 1:
                    self.engine.setProperty('voice', voices[1].id)
                self.engine.setProperty('rate', 175)
            except Exception:
                pass

            # Speak message
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Fallback TTS error: {e}")

    def speak_message(self, message):
        """Speak a message using OpenAI TTS or fallback to pyttsx3"""
        print(f"\n[TTS] Speaking: {message}")

        if self.use_openai_tts:
            try:
                # Only try OpenAI TTS if message is a string and not empty
                if isinstance(message, str) and message.strip():
                    print("[TTS] Using OpenAI TTS for speech...")

                    # Try to generate and play audio
                    audio = generate_speech(
                        text=message,
                        voice=OPENAI_TTS_VOICE,
                        model=OPENAI_TTS_MODEL
                    )
                    return
                else:
                    raise ValueError("Invalid message format")
            except Exception as e:
                print(f"OpenAI TTS error: {e}")
                print("Falling back to pyttsx3...")
                self.use_openai_tts = False  # Disable OpenAI TTS for this session

        # Fallback to pyttsx3
        try:
            if self.engine is None:
                print("Fallback TTS unavailable (pyttsx3 not installed)")
                return
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

    def wrap_text(self, text, max_width, font_scale, thickness):
        """Wrap text to fit screen width"""
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            # Add word to current line temporarily
            test_line = ' '.join(current_line + [word])
            # Get size of text with current font settings
            (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            if w <= max_width:
                current_line.append(word)
            else:
                # Line is full, start a new line
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, split it
                    lines.append(word)
                    current_line = []

        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def start_conversation(self):
        """Initiate conversation with user"""
        # Start new session
        self.current_session = {
            'start_time': datetime.now(),
            'interactions': []
        }

        print("Analyzing visual cues...")
        emotion, posture = self.analyze_initial_state(duration=3)

        # Get history from Firebase
        if self.current_user:
            try:
                firebase_history = self.firebase.get_user_history(self.current_user)
                if firebase_history:
                    # Update local history with Firebase data
                    self.conversation_history = firebase_history
                    last_interaction = firebase_history[-1]
                    time_diff = datetime.now() - last_interaction['timestamp'].replace(tzinfo=None)

                    if time_diff.total_seconds() < 3600:  # Within last hour
                        greeting = "Thanks for sharing more. How are you feeling now?"
                    else:
                        greeting = "Welcome back! How are you doing today?"
                else:
                    greeting = "Hey, how are you doing today?"
            except Exception as e:
                print(f"Error getting Firebase history: {e}")
                greeting = "Hey, how are you doing today?"
        else:
            greeting = "Hey, how are you doing today?"

        self.speak_message(greeting)

        # Listen for response
        response = self.listen_for_response()
        if response:
            # Combine verbal and visual data
            follow_up = self.generate_response(
                verbal_response=response,
                locked_emotion=emotion,
                locked_posture=posture
            )
            self.speak_message(follow_up)

        # End conversation instead of maintaining it
        self.in_conversation = False

    def analyze_initial_state(self, duration=3):
        """Analyze initial emotional state without blocking video feed"""
        print("\nStarting initial state analysis...")
        emotions = []
        postures = []

        start_time = time.time()
        frames_analyzed = 0

        analysis_settings = {
            'actions': ['emotion'],
            'enforce_detection': False,
            'detector_backend': 'opencv',
            'align': True
        }

        target_frames = 5
        frames_between_analysis = duration / target_frames
        last_analysis_time = 0

        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            current_time = time.time()

            if not ret:
                print("Error: Could not read from camera")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)

            if current_time - last_analysis_time >= frames_between_analysis and frames_analyzed < target_frames:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    print(f"\nAnalyzing frame {frames_analyzed + 1}/{target_frames}...")

                    analysis = DeepFace.analyze(
                        img_path=rgb_frame,
                        **analysis_settings
                    )

                    if analysis:
                        current_emotions = analysis[0]['emotion']
                        sorted_emotions = sorted(current_emotions.items(), key=lambda x: x[1], reverse=True)

                        if sorted_emotions[0][1] > 30:
                            emotions.append(sorted_emotions[0][0])
                            frames_analyzed += 1
                            print(f"Detected emotion: {sorted_emotions[0][0]} ({sorted_emotions[0][1]:.1f}%)")
                            last_analysis_time = current_time

                except Exception as e:
                    print(f"Frame analysis error: {str(e)}")
                    if frames_analyzed < 2:
                        emotions.append("neutral")
                        frames_analyzed += 1
                    time.sleep(0.1)

            # Analyze posture
            if int(current_time * 2) % 2 == 0:
                posture_cues = self.analyze_posture(frame)
                if posture_cues:
                    postures.extend(posture_cues)

        # Process results
        if emotions:
            weighted_emotions = emotions[-2:] + emotions
            try:
                from statistics import mode
                dominant_emotion = mode(weighted_emotions)
                confidence = (weighted_emotions.count(dominant_emotion) / len(weighted_emotions)) * 100
                print(f"\nAnalysis Results:")
                print(f"Dominant emotion: {dominant_emotion} (Confidence: {confidence:.1f}%)")
            except:
                dominant_emotion = emotions[-1]
                print(f"No clear dominant emotion, using most recent: {dominant_emotion}")
        else:
            dominant_emotion = "neutral"
            print("No emotions detected, defaulting to neutral")

        unique_postures = list(set(postures))
        print(f"Detected posture cues: {unique_postures}")

        return dominant_emotion, unique_postures

    def analyze_posture(self, frame):
        """Analyze posture using MediaPipe Pose with detailed metrics
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            tuple: (posture_cues, visualization_frame)
        """
        try:
            # Initialize MediaPipe Pose if not already done
            if not hasattr(self, 'mp_pose'):
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=True,  # Use static mode for single frame
                    model_complexity=2,  # Use highest complexity model for maximum accuracy
                    min_detection_confidence=0.5,  # Higher detection confidence for better accuracy
                    min_tracking_confidence=0.5,  # Higher tracking confidence for better accuracy
                    smooth_landmarks=True  # Enable smoothing for more stable detection
                )
                print("MediaPipe Pose initialized for accurate posture analysis")

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            # Default posture cues
            posture_cues = ["neutral"]
            viz_frame = frame.copy()
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Update landmark history
                self._update_landmark_history(landmarks)
                
                # Check visibility of key landmarks with higher threshold for accuracy
                key_landmarks = [
                    self.mp_pose.PoseLandmark.NOSE,
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                    self.mp_pose.PoseLandmark.RIGHT_KNEE
                ]
                
                # Calculate visibility score with higher threshold
                visibility = sum(landmarks[lm].visibility for lm in key_landmarks) / len(key_landmarks)
                
                # Only proceed if visibility is sufficient
                if visibility > 0.5:  # Increased from 0.1 to 0.5 for better accuracy
                    # Extract key landmarks
                    nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                    left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
                    right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
                    left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                    left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                    right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                    
                    # Detect posture mode (standing or sitting)
                    hip_y = (left_hip.y + right_hip.y) / 2
                    knee_y = (left_knee.y + right_knee.y) / 2
                    
                    if knee_y > hip_y + 0.05:  # Knees below hips = standing
                        posture_mode = "standing"
                    else:
                        posture_mode = "sitting"
                    
                    # Calculate angles for detailed posture analysis
                    # Neck angle (head tilt)
                    neck_vector = np.array([(left_ear.x + right_ear.x)/2 - nose.x, 
                                            (left_ear.y + right_ear.y)/2 - nose.y])
                    neck_angle = np.degrees(np.arctan2(neck_vector[1], neck_vector[0]))
                    
                    # Shoulder angle (shoulder tilt)
                    shoulder_vector = np.array([right_shoulder.x - left_shoulder.x, 
                                               right_shoulder.y - left_shoulder.y])
                    shoulder_angle = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
                    
                    # Spine angle (forward lean)
                    spine_vector = np.array([(left_hip.x + right_hip.x)/2 - (left_shoulder.x + right_shoulder.x)/2,
                                             (left_hip.y + right_hip.y)/2 - (left_shoulder.y + right_shoulder.y)/2])
                    spine_angle = np.degrees(np.arctan2(spine_vector[1], spine_vector[0]))
                    
                    # Hip angle (hip tilt)
                    hip_vector = np.array([right_hip.x - left_hip.x, 
                                          right_hip.y - left_hip.y])
                    hip_angle = np.degrees(np.arctan2(hip_vector[1], hip_vector[0]))
                    
                    # Analyze posture based on angles with more precise thresholds
                    posture_cues = []
                    
                    # Check for confident posture
                    if abs(shoulder_angle) < 10 and abs(hip_angle) < 10 and abs(spine_angle) < 20:
                        posture_cues.append("confident")
                    
                    # Check for slouched posture
                    if spine_angle > 25 or (posture_mode == "sitting" and spine_angle > 20):
                        posture_cues.append("slouched")
                    
                    # Check for good posture
                    if abs(neck_angle) < 15 and abs(shoulder_angle) < 10 and abs(spine_angle) < 20:
                        posture_cues.append("good posture")
                    
                    # Check for tense posture
                    if abs(shoulder_angle) > 20 or abs(hip_angle) > 20:
                        posture_cues.append("tense")
                    
                    # Check for balanced posture
                    if abs(shoulder_angle) < 15 and abs(hip_angle) < 15:
                        posture_cues.append("balanced")
                    
                    # Add posture mode
                    posture_cues.append(posture_mode)
                    
                    # If no specific cues detected, use neutral
                    if not posture_cues:
                        posture_cues = ["neutral"]
                    
                    # --- Start: Temporal Analysis for Advanced Cues ---
                    if len(self.nose_history) == self.landmark_history_size: # Ensure buffers are full
                        # Calculate stability/fidgeting (example for shoulders)
                        left_shoulder_y_positions = [pos.y for pos in self.left_shoulder_history]
                        right_shoulder_y_positions = [pos.y for pos in self.right_shoulder_history]
                        left_shoulder_fidget = np.std(left_shoulder_y_positions)
                        right_shoulder_fidget = np.std(right_shoulder_y_positions)
                        
                        # Placeholder for confidence, scared, nervous logic
                        # This will be expanded based on these temporal metrics and angles
                        if left_shoulder_fidget > 0.01 or right_shoulder_fidget > 0.01: # Example threshold
                            # If there's noticeable shoulder movement, it might lean towards nervous
                            # We need more rules to confirm
                            pass 
                        
                        # Example: Check for "confident" - upright and stable
                        is_upright = abs(spine_angle) < 15 and abs(neck_angle) < 15 # Example thresholds
                        is_stable_shoulders = left_shoulder_fidget < 0.005 and right_shoulder_fidget < 0.005 # Example thresholds
                        
                        if is_upright and is_stable_shoulders:
                            if "confident" not in posture_cues: # Avoid duplicates if already there
                                posture_cues.append("confident")
                            if "neutral" in posture_cues: # Remove neutral if confident
                                posture_cues.remove("neutral")
                        
                        # TODO: Add rules for "scared" and "nervous" using temporal data + angles
                        
                    # --- End: Temporal Analysis ---
                    
                    # Draw landmarks on visualization frame
                    self.mp_drawing = mp.solutions.drawing_utils
                    self.mp_drawing.draw_landmarks(
                        viz_frame, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                else:
                    # Even with low visibility, try to detect basic posture
                    # Check if at least shoulders are visible
                    shoulder_visibility = (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].visibility + 
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility) / 2
                    
                    if shoulder_visibility > 0.3:  # Increased from 0.05 to 0.3 for better accuracy
                        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        
                        # Calculate shoulder angle
                        shoulder_vector = np.array([right_shoulder.x - left_shoulder.x, 
                                                   right_shoulder.y - left_shoulder.y])
                        shoulder_angle = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
                        
                        if abs(shoulder_angle) < 15:
                            posture_cues = ["neutral"]
                        else:
                            posture_cues = ["slight tilt"]
                    else:
                        posture_cues = ["Insufficient visibility"]
            else:
                posture_cues = ["No pose detected"]
                
            return posture_cues, viz_frame
            
        except Exception as e:
            print(f"Posture analysis error: {e}")
            return ["neutral"], frame

    def _get_user_posture_history(self):
        """Retrieve user's posture history from Firebase if signed in"""
        try:
            if not self.user_id or not self.firebase_manager:
                return None
                
            # Get user's posture history from Firebase
            posture_history = self.firebase_manager.get_user_posture_history(self.user_id)
            
            # Also get emotional history to correlate with posture
            emotion_history = self.firebase_manager.get_user_emotion_history(self.user_id)
            
            return {
                'posture': posture_history,
                'emotions': emotion_history
            }
        except Exception as e:
            print(f"Error retrieving user posture history: {e}")
            return None

    def _store_user_posture_data(self, metrics, cues):
        """Store current posture data in Firebase for the signed-in user"""
        try:
            if not self.user_id or not self.firebase_manager:
                return
                
            # Prepare data to store
            posture_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                'cues': cues,
                'mode': metrics.get('posture_mode', 'unknown'),
                'emotion': self.last_detected_emotion if hasattr(self, 'last_detected_emotion') else 'unknown'
            }
            
            # Store in Firebase
            self.firebase_manager.store_user_posture_data(self.user_id, posture_data)
            
            # Update local history
            self.user_posture_history.append(posture_data)
            if len(self.user_posture_history) > 50:  # Keep last 50 entries
                self.user_posture_history.pop(0)
                
        except Exception as e:
            print(f"Error storing user posture data: {e}")

    def _personalize_posture_feedback(self, current_cues, user_history):
        """Generate personalized posture feedback based on user history"""
        try:
            if not user_history:
                return current_cues
                
            personalized_cues = current_cues.copy()
            posture_history = user_history.get('posture', [])
            emotion_history = user_history.get('emotions', [])
            
            if not posture_history:
                return current_cues
                
            # Analyze common posture issues
            common_issues = {}
            for entry in posture_history:
                for cue in entry.get('cues', []):
                    if "excellent" not in cue.lower() and "great" not in cue.lower():
                        common_issues[cue] = common_issues.get(cue, 0) + 1
            
            # Find most common issues
            if common_issues:
                sorted_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)
                most_common_issue = sorted_issues[0][0]
                
                # Check if current posture has the most common issue
                for cue in current_cues:
                    if most_common_issue in cue:
                        # Add personalized feedback about recurring issue
                        personalized_cues.append(f"This is a recurring issue for you. Try focusing on improving your {most_common_issue.split('-')[0].strip().lower()}.")
                        break
            
            # Correlate emotions with posture
            if emotion_history and hasattr(self, 'last_detected_emotion'):
                current_emotion = self.last_detected_emotion
                emotion_posture_map = {}
                
                # Map emotions to posture issues
                for p_entry in posture_history:
                    p_time = p_entry.get('timestamp')
                    p_cues = p_entry.get('cues', [])
                    p_emotion = p_entry.get('emotion', 'unknown')
                    
                    if p_emotion != 'unknown' and p_cues:
                        if p_emotion not in emotion_posture_map:
                            emotion_posture_map[p_emotion] = []
                        
                        for cue in p_cues:
                            if "excellent" not in cue.lower() and "great" not in cue.lower():
                                emotion_posture_map[p_emotion].append(cue)
                
                # If current emotion correlates with specific posture issues
                if current_emotion in emotion_posture_map and emotion_posture_map[current_emotion]:
                    # Count occurrences of each issue
                    emotion_issues = {}
                    for issue in emotion_posture_map[current_emotion]:
                        emotion_issues[issue] = emotion_issues.get(issue, 0) + 1
                    
                    # Get most common issue for this emotion
                    sorted_emotion_issues = sorted(emotion_issues.items(), key=lambda x: x[1], reverse=True)
                    if sorted_emotion_issues:
                        top_issue = sorted_emotion_issues[0][0]
                        
                        # Check if any current cue matches the emotion-related issue
                        for cue in current_cues:
                            if top_issue in cue:
                                personalized_cues.append(f"I notice your {current_emotion} mood often coincides with {top_issue.split('-')[0].strip().lower()} issues.")
                                break
            
            # Add improvement tracking if we have enough history
            if len(posture_history) >= 5:
                # Compare recent posture with past posture
                recent_metrics = [entry.get('metrics', {}) for entry in posture_history[-5:]]
                
                # Calculate improvement in key metrics
                improvements = {}
                for metric in ['neck_angle', 'spine_angle', 'shoulder_angle', 'head_forward_ratio']:
                    values = [entry.get(metric, None) for entry in recent_metrics if metric in entry]
                    if len(values) >= 5:
                        # Calculate trend (positive = improving, negative = worsening)
                        if metric in ['neck_angle', 'spine_angle', 'shoulder_angle']:
                            # For angles, improvement means getting closer to 0
                            trend = abs(values[0]) - abs(values[-1])
                        else:
                            # For ratios, improvement means decreasing
                            trend = values[0] - values[-1]
                        
                        if abs(trend) > 0.05:  # Only note significant changes
                            improvements[metric] = trend
                
                # Add feedback about improvements
                if improvements:
                    improved = [k for k, v in improvements.items() if v > 0]
                    worsened = [k for k, v in improvements.items() if v < 0]
                    
                    if improved:
                        metric_name = improved[0].replace('_', ' ').title()
                        personalized_cues.append(f"Your {metric_name} has improved since last time!")
                    elif worsened:
                        metric_name = worsened[0].replace('_', ' ').title()
                        personalized_cues.append(f"Your {metric_name} needs more attention than before.")
            
            return personalized_cues
            
        except Exception as e:
            print(f"Error personalizing posture feedback: {e}")
            return current_cues

    def _angle_between_vectors(self, v1, v2):
        """Calculate angle between two 2D vectors in degrees"""
        # Normalize vectors
        v1_norm = np.sqrt(v1[0]**2 + v1[1]**2)
        v2_norm = np.sqrt(v2[0]**2 + v2[1]**2)
        
        # Avoid division by zero
        if v1_norm == 0 or v2_norm == 0:
            return 0
        
        v1 = [v1[0]/v1_norm, v1[1]/v1_norm]
        v2 = [v2[0]/v2_norm, v2[1]/v2_norm]
        
        # Calculate dot product
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        
        # Clamp to avoid numerical errors
        dot_product = max(-1.0, min(1.0, dot_product))
        
        # Calculate angle in degrees
        angle = np.degrees(np.arccos(dot_product))
        
        # Determine sign (positive = clockwise, negative = counterclockwise)
        cross_product = v1[0]*v2[1] - v1[1]*v2[0]
        if cross_product < 0:
            angle = -angle
        
        return angle

    def _draw_advanced_posture_visualization(self, frame, landmarks, metrics, cues, problem_landmarks):
        """Draw advanced posture visualization with metrics, feedback and user history"""
        height, width = frame.shape[:2]
        
        # Create connection specs that highlight problem areas
        connection_spec = {}
        landmark_spec = {}
        
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start, end = connection
            
            if start in problem_landmarks and end in problem_landmarks:
                # Problem connection
                connection_spec[connection] = self.problem_drawing_spec
            else:
                # Normal connection
                connection_spec[connection] = self.custom_drawing_spec
        
        for i in range(len(landmarks.landmark)):
            if i in problem_landmarks:
                landmark_spec[i] = self.problem_drawing_spec
            else:
                landmark_spec[i] = self.joint_drawing_spec
        
        # Draw pose with custom styling
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_spec,
            connection_drawing_spec=connection_spec
        )
        
        # Create semi-transparent overlay for metrics panel
        metrics_panel_width = 300
        metrics_panel_height = 200
        metrics_x = width - metrics_panel_width - 10
        metrics_y = 10
        
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (metrics_x, metrics_y),
            (metrics_x + metrics_panel_width, metrics_y + metrics_panel_height),
            (20, 20, 20),
            cv2.FILLED
        )
        
        # Apply overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
        # Add title with posture mode
        mode_text = f"Posture Analysis ({metrics.get('posture_mode', 'unknown').title()})"
        cv2.putText(
            frame,
            mode_text,
            (metrics_x + 10, metrics_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )
        
        # Add horizontal line
        cv2.line(
            frame,
            (metrics_x + 5, metrics_y + 35),
            (metrics_x + metrics_panel_width - 5, metrics_y + 35),
            (200, 200, 200),
            1
        )
        
        # Add key metrics
        y_offset = 55
        key_metrics = [
            ('Neck Angle', 'neck_angle', '°'),
            ('Shoulder Alignment', 'shoulder_angle', '°'),
            ('Spine Alignment', 'spine_angle', '°'),
            ('Head Forward', 'head_forward_ratio', '')
        ]
        
        for label, key, unit in key_metrics:
            if key in metrics:
                value = metrics[key]
                
                # Determine color based on thresholds
                if key == 'neck_angle' and abs(value) > 15:
                    color = (50, 50, 255)  # Red
                elif key == 'shoulder_angle' and abs(value) > 5:
                    color = (50, 50, 255)  # Red
                elif key == 'spine_angle' and abs(value) > 8:
                    color = (50, 50, 255)  # Red
                elif key == 'head_forward_ratio' and value > 0.15:
                    color = (50, 50, 255)  # Red
                else:
                    color = (50, 255, 50)  # Green
                
                # Draw label
                cv2.putText(
                    frame,
                    f"{label}:",
                    (metrics_x + 10, metrics_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    lineType=cv2.LINE_AA
                )
                
                # Draw value with color coding
                value_text = f"{value:.1f}{unit}"
                cv2.putText(
                    frame,
                    value_text,
                    (metrics_x + 200, metrics_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    lineType=cv2.LINE_AA
                )
                
                y_offset += 25
        
        # Add horizontal line
        cv2.line(
            frame,
            (metrics_x + 5, metrics_y + y_offset),
            (metrics_x + metrics_panel_width - 5, metrics_y + y_offset),
            (200, 200, 200),
            1
        )
        
        # Add posture feedback
        y_offset += 20
        cv2.putText(
            frame,
            "Feedback:",
            (metrics_x + 10, metrics_y + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            lineType=cv2.LINE_AA
        )
        
        y_offset += 25
        for i, cue in enumerate(cues[:2]):  # Show only top 2 cues to avoid clutter
            # Determine color based on content
            if "excellent" in cue.lower() or "great" in cue.lower():
                color = (50, 255, 50)  # Green
            else:
                color = (50, 200, 255)  # Yellow-orange
            
            cv2.putText(
                frame,
                f"• {cue}",
                (metrics_x + 15, metrics_y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                lineType=cv2.LINE_AA
            )
            y_offset += 20
        
        # After drawing the standard visualization, add user history if available
        if self.user_id and hasattr(self, 'user_posture_history') and self.user_posture_history:
            # Add user history panel
            history_panel_width = 300
            history_panel_height = 150
            history_x = 10
            history_y = height - history_panel_height - 10
            
            # Create semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (history_x, history_y),
                (history_x + history_panel_width, history_y + history_panel_height),
                (20, 20, 20),
                cv2.FILLED
            )
            
            # Apply overlay with transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            
            # Add title
            cv2.putText(
                frame,
                f"Posture History for {self.current_user.get('name', 'User')}",
                (history_x + 10, history_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA
            )
            
            # Add horizontal line
            cv2.line(
                frame,
                (history_x + 5, history_y + 35),
                (history_x + history_panel_width - 5, history_y + 35),
                (200, 200, 200),
                1
            )
            
            # Show trend data
            y_offset = 55
            
            # Calculate trends from history
            if len(self.user_posture_history) >= 2:
                # Get key metrics from history
                history_metrics = {}
                for metric in ['neck_angle', 'spine_angle', 'shoulder_angle', 'head_forward_ratio']:
                    values = []
                    for entry in self.user_posture_history:
                        if 'metrics' in entry and metric in entry['metrics']:
                            values.append(abs(entry['metrics'][metric]) if metric in ['neck_angle', 'spine_angle', 'shoulder_angle'] else entry['metrics'][metric])
                    
                    if values:
                        history_metrics[metric] = {
                            'current': values[-1],
                            'average': sum(values) / len(values),
                            'trend': values[-1] - values[0] if len(values) > 1 else 0
                        }
                
                # Display trends
                for metric, data in history_metrics.items():
                    label = metric.replace('_', ' ').title()
                    trend = data['trend']
                    
                    # Determine if trend is good or bad
                    if metric in ['neck_angle', 'spine_angle', 'shoulder_angle', 'head_forward_ratio']:
                        # For these metrics, negative trend is good (decreasing)
                        trend_good = trend < 0
                    else:
                        trend_good = trend > 0
                    
                    # Format trend text
                    if abs(trend) < 0.01:
                        trend_text = "stable"
                        trend_color = (200, 200, 200)  # Gray
                    else:
                        trend_text = "improving" if trend_good else "worsening"
                        trend_color = (50, 255, 50) if trend_good else (50, 50, 255)  # Green or red
                    
                    # Draw metric and trend
                    cv2.putText(
                        frame,
                        f"{label}:",
                        (history_x + 10, history_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                        lineType=cv2.LINE_AA
                    )
                    
                    cv2.putText(
                        frame,
                        trend_text,
                        (history_x + 180, history_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        trend_color,
                        1,
                        lineType=cv2.LINE_AA
                    )
                    
                    y_offset += 20
            else:
                # Not enough history data
                cv2.putText(
                    frame,
                    "Collecting posture history data...",
                    (history_x + 10, history_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    lineType=cv2.LINE_AA)
        
        return frame

    def _format_history_context(self, history):
        """Format user history into useful context"""
        if not history or history.get('trend') == 'insufficient_data':
            return "No previous session history available."
        
        context = []
        context.append(f"Session Count: {history['session_count']}")
        if history['avg_risk_level']:
            context.append(f"Average Risk Level: {history['avg_risk_level']:.1f}%")
        context.append(f"Emotional Trend: {history['trend']}")
        context.append(f"Common Emotions: {', '.join(history['common_emotions'])}")
        
        return "\n".join(context)

    def _get_conversation_context(self, max_turns: int = 6) -> str:
        """Build a concise recent conversation context from current session interactions.

        Args:
            max_turns: Maximum number of last user/AI exchanges to include
        Returns:
            String representing recent conversation history for prompting
        """
        try:
            interactions = []
            if hasattr(self, 'current_session') and self.current_session and 'interactions' in self.current_session:
                interactions = self.current_session['interactions']

            if not interactions:
                return "No prior conversation in this session."

            # Take the most recent interactions up to max_turns
            recent = interactions[-max_turns:]
            lines = []
            for item in recent:
                user_text = item.get('user_input') or ''
                ai_text = item.get('ai_output') or ''
                if user_text:
                    lines.append(f"User: {user_text}")
                if ai_text:
                    lines.append(f"AI: {ai_text}")

            return "\n".join(lines) if lines else "No prior conversation in this session."
        except Exception:
            return "No prior conversation in this session."

    def generate_response(self, text, emotion, body_language=[]):
        """Generate a response based on user input with mental health risk assessment using RAG"""
        try:
            # Import the prompt generator function
            from app import generate_mental_health_prompt
            from prompts import MentalHealthPrompts
            
            # Get system memory context if available
            memory_context = ""
            proactive_elements = []
            conversation_context = self._get_conversation_context(max_turns=6)
            
            # Check if we have session memory and should include proactive elements
            if hasattr(self, 'session_memory') and self.session_memory:
                try:
                    # Extract relevant context from session memory
                    user_profile = self.session_memory.get('user_profile', {})
                    recent_context = self.session_memory.get('recent_context', {})
                    system_notes = self.session_memory.get('system_notes', {})
                    
                    # Build memory context for AI
                    memory_context = f"""
SYSTEM MEMORY CONTEXT:
User Profile: {user_profile.get('name', 'Unknown')} - {user_profile.get('emotional_patterns', 'Monitoring patterns')}
Primary Concerns: {', '.join(user_profile.get('primary_concerns', []))}
Recent Context: {recent_context.get('last_session_summary', 'No recent context')}
Mood Trend: {recent_context.get('mood_trend', 'Monitoring')}
Interaction Style: {system_notes.get('interaction_style', 'Supportive')}
Effective Strategies: {', '.join(system_notes.get('effective_strategies', []))}
Topics to Avoid: {', '.join(system_notes.get('triggers_to_avoid', []))}

"""
                    
                    # Get proactive questions if this is early in the conversation
                    if hasattr(self, 'proactive_questions') and self.proactive_questions:
                        # Include 1-2 proactive elements contextually
                        for question in self.proactive_questions[:2]:
                            if len(question) < 100:  # Only include concise questions
                                proactive_elements.append(question)
                    
                except Exception as e:
                    print(f"Error processing session memory: {e}")
                    memory_context = ""
            
            # Use RAG system to gather relevant mental health information
            rag_context = self._get_rag_context(text, emotion, body_language)
            
            # Enhanced risk assessment prompt with memory context
            risk_assessment_prompt = (
                f"{memory_context}"
                f"Recent Conversation (most recent last):\n{conversation_context}\n\n"
                f"Current Context:\n"
                f"User Input: {text}\n"
                f"Detected Emotion: {emotion}\n"
                f"Body Language: {body_language}\n\n"
                "Assess the risk level and return ONLY a JSON object with:\n"
                "{\n"
                "    \"risk_level\": <number 0-100>,\n"
                "    \"risk_indicators\": [<list of indicators>],\n"
                "    \"is_emergency\": <true if risk_level >= 90>\n"
                "}\n\n"
                "IMPORTANT GUIDELINES FOR RISK ASSESSMENT:\n"
                "1. For common negative expressions like \"I feel god-awful today\", \"I'm having a bad day\", \"today was shitty\", \"I feel like crap\", etc., risk levels should be BELOW 30%.\n"
                "2. Only rate 50-70% for explicit mental health concerns WITHOUT crisis indicators.\n"
                "3. Only rate 70-80% for sustained negative thoughts or moderate depression symptoms.\n"
                "4. Only rate 80-90% for specific mentions of self-harm, harm to others, or suicidal thoughts.\n"
                "5. Only rate 90%+ (emergency) for explicit, imminent suicide risk or harm.\n"
                "6. DO NOT OVERREACT to common expressions of having a bad day or negative feelings - these are normal human experiences, not mental health crises.\n"
                "7. Consider the user's history and patterns from the system memory context above."
            )
            
            print(f"\n[DEBUG] RISK ASSESSMENT INPUT:")
            print(f"[DEBUG] Text: {text}")
            print(f"[DEBUG] Emotion: {emotion}")
            print(f"[DEBUG] Body Language: {body_language}")
            print(f"[DEBUG] Memory Context Available: {len(memory_context) > 0}")
            
            # Call OpenAI API to analyze the risk
            risk_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": risk_assessment_prompt}],
                temperature=0.3,  # Lower temperature for more consistent risk assessment
                max_tokens=200
            )
            
            # Log the raw risk assessment
            raw_risk_response = risk_response.choices[0].message.content
            print(f"\n[DEBUG] RAW RISK ASSESSMENT:")
            print(f"{raw_risk_response}")
            
            # Parse the risk assessment
            import json
            import re
            
            try:
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', raw_risk_response, re.DOTALL)
                if json_match:
                    assessment_json = json.loads(json_match.group())
                    risk_level = assessment_json.get('risk_level', 0)
                    risk_indicators = assessment_json.get('risk_indicators', [])
                    is_emergency = assessment_json.get('is_emergency', False)
                    
                    # Store the risk level for logging
                    self.last_risk_level = risk_level
                    self.last_risk_indicators = risk_indicators
                    self.last_is_emergency = is_emergency
                    
                    print(f"\n[DEBUG] RISK ASSESSMENT RESULTS:")
                    print(f"[DEBUG] Risk Level: {risk_level}%")
                    print(f"[DEBUG] Risk Indicators: {risk_indicators}")
                    print(f"[DEBUG] Is Emergency: {is_emergency}")
                    
                    # Determine the appropriate response method based on risk level
                    if risk_level >= 90:
                        print("[DEBUG] Using CRISIS response (90%+)")
                        final_response = self._generate_crisis_response(text, emotion, risk_level, body_language)
                    elif risk_level >= 80:
                        print("[DEBUG] Using HIGH RISK response (80-89%)")
                        final_response = self._generate_high_risk_response(text, emotion, risk_level, body_language)
                    else:
                        print("[DEBUG] Using SUPPORTIVE response (0-79%)")
                        final_response = self._generate_supportive_response(text, emotion, risk_level, body_language, memory_context, proactive_elements)
                    
                    # Check for prohibited phrases in low risk responses
                    if risk_level < 80 and any(phrase in final_response.lower() for phrase in [
                        "i'm unable to provide the help",
                        "i can't help you",
                        "i am not able to provide",
                        "i cannot provide",
                        "i'm not qualified"
                    ]):
                        print("[DEBUG] DETECTED PROHIBITED PHRASES IN LOW RISK RESPONSE - USING FALLBACK")
                        final_response = "I'm here to listen and support you. Would you like to talk more about what's on your mind?"
                    
                    # Update system memory after interaction
                    self._update_system_memory_after_interaction(text, emotion, risk_level, body_language, final_response)
                    
                    return final_response
                else:
                    print("[DEBUG] Could not extract JSON from risk assessment")
                    return self._generate_fallback_response(text, emotion)
            
            except Exception as e:
                print(f"[DEBUG] Error in risk assessment: {e}")
                return self._generate_fallback_response(text, emotion)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing that right now. Could you please try rephrasing or wait a moment and try again?"

    def _get_rag_context(self, text, emotion, body_language):
        """Get relevant context from RAG system for mental health assessment"""
        try:
            # Get user history context first
            user_id = self.current_user  # Changed from current_user_id
            user_history = self.firebase_manager.analyze_user_progress(user_id)
            
            # Add historical context to the query
            history_context = self._format_history_context(user_history)
            enhanced_query = f"{text}\n\nUser History Context: {history_context}"
            
            # Import necessary components from RAG setup
            import sys
            from pathlib import Path
            
            # Add the directory containing setup_rag.py to the path
            rag_dir = Path("/Users/avicomputer/AI mirror take 2")
            if rag_dir.exists() and str(rag_dir) not in sys.path:
                sys.path.append(str(rag_dir))
            
            try:
                from setup_rag import get_relevant_documents
                
                # Create search query combining all context
                search_query = f"""
                User text: {enhanced_query}
                Emotional state: {emotion}
                Body language: {body_language if body_language else 'Not available'}
                
                Find relevant mental health guidance for:
                1. The specific emotions and feelings expressed
                2. Appropriate support and response strategies
                3. Coping mechanisms and wellness activities
                4. Building emotional resilience
                """
                
                # Get relevant documents with increased k value
                docs = get_relevant_documents(search_query, k=10)
                
                # Store document sources for logging
                self.last_rag_articles = []
                
                # Format documents as context
                if docs:
                    context_parts = []
                    seen_sources = set()  # Track unique sources
                    
                    # Add user history first
                    context_parts.append(f"User History:\n{history_context}")
                    
                    # Add RAG documents
                    for doc in docs:
                        # Extract source info
                        source = doc.metadata.get('source', 'Unknown Source')
                        filename = doc.metadata.get('filename', 'Unknown File')
                        
                        # Only include unique sources
                        if source not in seen_sources:
                            seen_sources.add(source)
                            self.last_rag_articles.append(f"{source} ({filename})")
                        
                        # Add content to context
                        context_parts.append(f"From {source}:\n{doc.page_content}")
                    
                    # Join all context parts with clear separation
                    return "\n\n---\n\n".join(context_parts)
                else:
                    self.last_rag_articles = ["No relevant documents found"]
                    return "No relevant documents found."
                
            except ImportError:
                print("Could not import RAG functions from setup_rag.py")
                self.last_rag_articles = ["RAG system not available"]
                return "RAG system not available."
                
        except Exception as e:
            print(f"Error getting RAG context: {e}")
            self.last_rag_articles = [f"Error retrieving mental health context: {str(e)}"]
            return "Error retrieving mental health context."

    def _generate_crisis_response(self, text, emotion, risk_level, body_language=[]):
        """Generate response for severe risk situations (90-100%)"""
        from prompts import MentalHealthPrompts
        
        try:
            conversation_context = self._get_conversation_context(max_turns=6)
            prompt_content = (
                f"CONVERSATION HISTORY (most recent last):\n{conversation_context}\n\n" +
                MentalHealthPrompts.get_crisis_prompt(text, emotion, body_language)
            )

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.5,
                max_tokens=150
            )
            
            # Return the raw response text directly
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating crisis response: {e}")
            return "I'm deeply concerned about what you've shared. Please reach out to a crisis counselor right away by texting HOME to 741741 or calling the National Suicide Prevention Lifeline at 988. They're available 24/7 to help, and talking with someone can make a real difference right now."

    def _generate_high_risk_response(self, text, emotion, risk_level, body_language=[]):
        """Generate response for high risk situations (80-89%)"""
        from prompts import MentalHealthPrompts
        
        try:
            conversation_context = self._get_conversation_context(max_turns=6)
            prompt_content = (
                f"CONVERSATION HISTORY (most recent last):\n{conversation_context}\n\n" +
                MentalHealthPrompts.get_high_risk_prompt(text, emotion, body_language)
            )

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0.5,
                max_tokens=150
            )
            
            # Return the raw response text directly
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating high risk response: {e}")
            return "What you're feeling sounds really challenging, and I appreciate you sharing that with me. It might be helpful to talk with a mental health professional who can provide more personalized support. In the meantime, would it help to practice some deep breathing together or discuss some grounding techniques that might help in this moment?"

    def _generate_supportive_response(self, text, emotion, risk_level, body_language=[], memory_context="", proactive_elements=[]):
        """Generate supportive response for lower risk situations (0-79%) with memory integration"""
        from prompts import MentalHealthPrompts
        
        try:
            # Include recent conversation context for multi-turn continuity
            conversation_context = self._get_conversation_context(max_turns=6)
            # Build enhanced prompt with memory context and proactive elements
            supportive_prompt = f"""
{memory_context}

CONVERSATION HISTORY (most recent last):
{conversation_context}

CURRENT INTERACTION:
User Input: {text}
Emotion: {emotion}
Body Language: {body_language}
Risk Level: {risk_level}%

RAG Context: {self._get_rag_context(text, emotion, body_language)}

PROACTIVE OPPORTUNITIES:
{chr(10).join(f"- {element}" for element in proactive_elements) if proactive_elements else "No specific proactive elements for this interaction"}

Generate a supportive, empathetic response that:
1. Directly addresses what the user shared
2. Shows understanding of their emotional state
3. Incorporates relevant context from their history (if available)
4. Naturally weaves in 1 proactive element if appropriate and relevant
5. Offers practical support or coping strategies
6. Encourages continued dialogue
7. Maintains a warm, non-clinical tone
8. Avoids being overly formal or therapeutic-sounding

Guidelines:
- Keep the response conversational and natural
- Don't mention risk levels or clinical terms
- If including proactive elements, make them feel organic to the conversation
- Focus on validation, support, and gentle guidance
- Be specific to their situation, not generic
- Use "I" statements to show personal engagement

Response length: 2-4 sentences maximum.
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": supportive_prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating supportive response: {e}")
            return "I hear you, and I want you to know that I'm here to listen and support you. What's the most important thing on your mind right now?"

    def _update_system_memory_after_interaction(self, text, emotion, risk_level, body_language, response):
        """Update system memory with the completed interaction"""
        try:
            # Import system memory if available
            from app import system_memory
            
            if system_memory and hasattr(self, 'current_user'):
                interaction_data = {
                    'user_input': text,
                    'emotion': emotion,
                    'risk_level': risk_level,
                    'body_language': body_language,
                    'ai_response': response,
                    'timestamp': datetime.now()
                }
                
                system_memory.update_interaction_memory(self.current_user, interaction_data)
                print(f"[SystemMemory] Updated interaction memory for {self.current_user}")
        
        except Exception as e:
            print(f"Error updating system memory: {e}")

    def _generate_fallback_response(self, text, emotion):
        """Generate a fallback response when assessment fails"""
        return f"I hear you, and I appreciate you sharing that with me. Would you like to tell me more about how you're feeling today?"

    def listen_for_response(self):
        """Listen for user's verbal response with improved stop functionality"""
        print("\n[ACTION] Listening for response...")

        # Initialize recording variables
        self.recording = False
        self.audio_frames = []
        recording_thread = None

        def record_audio():
            """Record audio in a separate thread"""
            try:
                self.stream = self.audio.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk
                )

                self.recording = True
                start_time = time.time()

                while self.recording and (time.time() - start_time) < 30:  # Max 30 seconds
                    try:
                        data = self.stream.read(self.chunk)
                        self.audio_frames.append(data)
                    except Exception as e:
                        print(f"Error reading audio chunk: {e}")
                        break

                self.stream.stop_stream()
                self.stream.close()

            except Exception as e:
                print(f"Error in recording thread: {e}")
            finally:
                self.recording = False

        # Start recording in a separate thread
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()

        try:
            while True:
                # Get frame for UI
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]

                    # Create recording UI
                    gradient = np.zeros((60, width, 3), dtype=np.uint8)
                    for i in range(60):
                        alpha = 0.7 - (i / 60) * 0.5
                        gradient[i, :] = (0, 100, 50)

                    cv2.addWeighted(gradient, 0.6, frame[:60], 0.4, 0, frame[:60])

                    # Add recording indicator
                    if self.recording:
                        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red dot when recording
                        cv2.putText(frame, "Recording... Press 'L' to stop", 
                                  (50, 35), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, 
                                  (255, 255, 255), 
                                  1, 
                                  cv2.LINE_AA)

                    cv2.imshow('Smart Mirror', frame)

                # Check for 'L' key to stop recording
                key = cv2.waitKey(1) & 0xFF
                if key == ord('l'):
                    print("Stop key pressed, ending recording...")
                    self.recording = False
                    break

        finally:
            # Ensure recording is stopped
            self.recording = False
            if recording_thread:
                recording_thread.join()

        # Process recorded audio
        if self.audio_frames:
            try:
                # Save to temporary file
                filename = f"recording_{int(time.time())}.wav"
                wf = wave.open(filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()

                # Transcribe audio
                with speech_recognition.AudioFile(filename) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio)
                    print(f"Transcribed: {text}")

                    # Clean up
                    os.remove(filename)
                    return text

            except Exception as e:
                print(f"Error processing audio: {e}")

        return None

    def record_audio_direct(self, seconds=10):
        """Record audio directly using PyAudio for more reliable capture"""
        print(f"Recording audio directly for {seconds} seconds...")

        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()

            # Open stream with better parameters
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=1024,
                            input_device_index=None)  # Use default device

            print("Recording started...")
            frames = []

            # Record for specified duration
            for i in range(0, int(16000 / 1024 * seconds)):
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)

            print("Recording finished")

            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Convert to AudioData for recognition
            audio_data = speech_recognition.AudioData(
                b''.join(frames),
                sample_rate=16000,
                sample_width=2  # 16-bit audio
            )

            return audio_data

        except Exception as e:
            print(f"Error in direct audio recording: {e}")
            return None

    def maintain_conversation(self):
        """Maintain natural conversation flow"""
        # Only process one response and then end
        if self.in_conversation:
            response = self.listen_for_response()
            if response:
                emotion, posture = self.analyze_initial_state(duration=3)
                follow_up = self.generate_follow_up(response, emotion, posture)
                self.speak_message(follow_up)

            # End conversation after one interaction
            self.in_conversation = False

    def test_microphone(self):
        """Test microphone functionality to ensure speech recognition works"""
        print("\n[TEST] Testing microphone functionality...")

        # Create a visual indicator
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            # Create test UI
            gradient = np.zeros((100, width, 3), dtype=np.uint8)
            for i in range(100):
                alpha = 0.7 - (i / 100) * 0.5
                gradient[i, :] = (0, 0, 100)  # Blue background

            # Apply gradient overlay
            cv2.addWeighted(gradient, 0.7, frame[:100], 0.3, 0, frame[:100])

            cv2.putText(frame, "MICROPHONE TEST MODE", 
                       (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, 
                       (255, 255, 255), 
                       2, 
                       cv2.LINE_AA)

            cv2.putText(frame, "Please speak to test your microphone", 
                       (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

            cv2.putText(frame, "Press any key to exit test mode", 
                       (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

            cv2.imshow('Smart Mirror', frame)
            cv2.waitKey(1)

        # Print available microphones
        print("\nAvailable microphones:")
        mic_list = speech_recognition.Microphone.list_microphone_names()
        for i, mic in enumerate(mic_list):
            print(f"{i}: {mic}")

        # Test with different microphone settings
        try:
            # First, check if we can access the microphone at all
            print("\nTesting microphone access...")
            try:
                with self.mic as source:
                    print("Microphone access successful!")
            except Exception as e:
                print(f"ERROR: Could not access microphone: {e}")
                self.show_error_screen(frame, f"Could not access microphone: {e}")
                return False

            # Now test recording audio
            print("\nTesting audio recording...")
            with self.mic as source:
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)

                # Show current energy threshold
                print(f"Energy threshold: {self.recognizer.energy_threshold}")

                # Lower the threshold for better detection
                self.recognizer.energy_threshold = 300
                print(f"Lowered energy threshold to: {self.recognizer.energy_threshold}")

                # Listen for audio with visual feedback
                print("Listening for test audio...")

                # Create a simple animation loop
                start_time = time.time()
                listening = True

                while listening and (time.time() - start_time < 10):  # 10 second test
                    # Update visual feedback
                    if ret:
                        new_ret, new_frame = self.cap.read()
                        if new_ret and new_frame is not None:
                            frame = cv2.flip(new_frame, 1)

                            # Create test UI with animation
                            gradient = np.zeros((100, width, 3), dtype=np.uint8)
                            for i in range(100):
                                alpha = 0.7 - (i / 100) * 0.5
                                gradient[i, :] = (0, 0, 100)  # Blue background

                            # Apply gradient overlay
                            cv2.addWeighted(gradient, 0.7, frame[:100], 0.3, 0, frame[:100])

                            # Animated circle
                            circle_x = 30
                            circle_y = 50
                            pulse_size = 10 + int(abs(np.sin(time.time() * 3)) * 5)
                            cv2.circle(frame, (circle_x, circle_y), pulse_size, (0, 255, 255), -1)

                            cv2.putText(frame, "MICROPHONE TEST MODE", 
                                       (60, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, 
                                       (255, 255, 255), 
                                       2, 
                                       cv2.LINE_AA)

                            cv2.putText(frame, f"Listening... ({int(time.time() - start_time)}s)", 
                                       (60, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, 
                                       (255, 255, 255), 
                                       1, 
                                       cv2.LINE_AA)

                            # Show energy level
                            energy_level = min(int(self.recognizer.energy_threshold), 1000)
                            energy_bar_width = int((energy_level / 1000) * 200)
                            cv2.rectangle(frame, (60, 80), (60 + energy_bar_width, 90), (0, 255, 255), -1)

                            cv2.putText(frame, f"Energy: {energy_level}", 
                                       (270, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, 
                                       (255, 255, 255), 
                                       1, 
                                       cv2.LINE_AA)

                            cv2.imshow('Smart Mirror', frame)

                    # Check for key press to exit test
                    key = cv2.waitKey(100) & 0xFF
                    if key != 255:  # Any key pressed
                        listening = False
                        break

                # Try to capture audio
                try:
                    print("Now capturing test audio...")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)

                    # Show processing
                    if ret:
                        new_ret, new_frame = self.cap.read()
                        if new_ret and new_frame is not None:
                            frame = cv2.flip(new_frame, 1)

                            # Create test UI
                            gradient = np.zeros((100, width, 3), dtype=np.uint8)
                            for i in range(100):
                                alpha = 0.7 - (i / 100) * 0.5
                                gradient[i, :] = (0, 0, 100)  # Blue background

                            # Apply gradient overlay
                            cv2.addWeighted(gradient, 0.7, frame[:100], 0.3, 0, frame[:100])

                            cv2.putText(frame, "MICROPHONE TEST MODE", 
                                       (20, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, 
                                       (255, 255, 255), 
                                       2, 
                                       cv2.LINE_AA)

                            cv2.putText(frame, "Processing test audio...", 
                                       (20, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, 
                                       (255, 255, 255), 
                                       1, 
                                       cv2.LINE_AA)

                            cv2.imshow('Smart Mirror', frame)
                            cv2.waitKey(1)

                    # First try Google recognition
                    success = False
                    error_message = ""

                    try:
                        print("Trying Google recognition...")
                        text = self.recognizer.recognize_google(audio, language='en-US')
                        print(f"TEST SUCCESSFUL! Heard: {text}")
                        success = True
                    except speech_recognition.UnknownValueError:
                        error_message = "Google could not understand audio"
                        print(f"Google recognition error: {error_message}")
                    except speech_recognition.RequestError as e:
                        error_message = f"Google request error: {e}"
                        print(error_message)

                    # If Google fails, try Sphinx as fallback (offline recognition)
                    if not success:
                        try:
                            print("Trying Sphinx recognition (fallback)...")
                            # Check if we have pocketsphinx installed
                            try:
                                import pocketsphinx
                                text = self.recognizer.recognize_sphinx(audio)
                                print(f"SPHINX TEST SUCCESSFUL! Heard: {text}")
                                success = True
                            except ImportError:
                                print("Pocketsphinx not installed. Cannot use offline recognition.")
                                error_message += " | Pocketsphinx not installed for offline fallback"
                        except Exception as sphinx_error:
                            error_message += f" | Sphinx error: {sphinx_error}"
                            print(f"Sphinx recognition error: {sphinx_error}")

                    # Show success or failure
                    if success:
                        # Show success
                        if ret:
                            new_ret, new_frame = self.cap.read()
                            if new_ret and new_frame is not None:
                                frame = cv2.flip(new_frame, 1)

                                # Create test UI with green for success
                                gradient = np.zeros((100, width, 3), dtype=np.uint8)
                                for i in range(100):
                                    alpha = 0.7 - (i / 100) * 0.5
                                    gradient[i, :] = (0, 100, 0)  # Green background

                                # Apply gradient overlay
                                cv2.addWeighted(gradient, 0.7, frame[:100], 0.3, 0, frame[:100])

                                cv2.putText(frame, "TEST SUCCESSFUL!", 
                                           (20, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.7, 
                                           (255, 255, 255), 
                                           2, 
                                           cv2.LINE_AA)

                                cv2.putText(frame, f"Heard: {text}", 
                                           (20, 60), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.6, 
                                           (255, 255, 255), 
                                           1, 
                                           cv2.LINE_AA)

                                cv2.putText(frame, "Press any key to continue", 
                                           (20, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.5, 
                                           (255, 255, 255), 
                                           1, 
                                           cv2.LINE_AA)

                                cv2.imshow('Smart Mirror', frame)
                                cv2.waitKey(0)  # Wait for key press

                        return True
                    else:
                        # Show failure with detailed error
                        self.show_error_screen(frame, f"Recognition failed: {error_message}")
                        return False

                except speech_recognition.WaitTimeoutError:
                    print("TEST FAILED: No speech detected within timeout")
                    self.show_error_screen(frame, "No speech detected within timeout")
                    return False
        except Exception as e:
            print(f"TEST FAILED: Critical error in microphone test: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_screen(frame, f"Critical error: {e}")
            return False

    def show_error_screen(self, frame, error_message):
        """Display an error screen with detailed information"""
        if frame is None:
            # Try to get a new frame
            ret, frame = self.cap.read()
            if not ret:
                print("Could not get frame for error display")
                return
            frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # Create error UI with red for failure
        gradient = np.zeros((150, width, 3), dtype=np.uint8)
        for i in range(150):
            alpha = 0.7 - (i / 150) * 0.5
            gradient[i, :] = (0, 0, 100)  # Red background

        # Apply gradient overlay
        cv2.addWeighted(gradient, 0.7, frame[:150], 0.3, 0, frame[:150])

        cv2.putText(frame, "TEST FAILED", 
                   (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, 
                   (255, 255, 255), 
                   2, 
                   cv2.LINE_AA)

        # Split error message into multiple lines if needed
        max_chars = 60
        error_lines = []
        for i in range(0, len(error_message), max_chars):
            error_lines.append(error_message[i:i+max_chars])

        # Display error message lines
        for i, line in enumerate(error_lines):
            cv2.putText(frame, line, 
                       (20, 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

        # Add troubleshooting tips
        tips_y = 60 + len(error_lines)*20 + 10
        cv2.putText(frame, "Troubleshooting tips:", 
                   (20, tips_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (255, 255, 0), 
                   1, 
                   cv2.LINE_AA)

        tips = [
            "1. Check your internet connection for Google recognition",
            "2. Make sure your microphone is not muted",
            "3. Try installing pocketsphinx for offline recognition",
            "4. Check system audio settings and permissions"
        ]

        for i, tip in enumerate(tips):
            cv2.putText(frame, tip, 
                       (20, tips_y + 20 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

        cv2.putText(frame, "Press any key to continue", 
                   (20, tips_y + 20 + len(tips)*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (255, 255, 255), 
                   1, 
                   cv2.LINE_AA)

        cv2.imshow('Smart Mirror', frame)
        cv2.waitKey(0)  # Wait for key press

    def run(self):
        """Enhanced main loop for local mode"""
        if self.web_mode:
            return  # Web mode is handled by web routes

        print("Starting Smart Mirror in Local Mode...")
        last_frame_time = time.time()
        target_fps = 30

        analyzing = False
        currently_listening = False

        while True:
            # Calculate time since last frame
            current_time = time.time()
            elapsed = current_time - last_frame_time

            # Capture and process frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                time.sleep(0.1)
                continue

            # Update frame time
            last_frame_time = current_time

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)

            # Add subtle UI elements
            height, width = frame.shape[:2]

            # Create subtle gradient for top bar
            gradient = np.zeros((40, width, 3), dtype=np.uint8)
            for i in range(40):
                gradient[i, :] = (40, 40, 40)  # Dark gray background

            # Apply subtle gradient overlay
            cv2.addWeighted(gradient, 0.6, frame[:40], 0.4, 0, frame[:40])

            # Add small button-like UI elements
            button_size = 20
            button_spacing = 30
            button_y = 20

            # Draw R button (analyze)
            cv2.circle(frame, (button_spacing, button_y), 10, (0, 120, 255), 1)
            cv2.putText(frame, "R", 
                       (button_spacing - 3, button_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

            # Draw L button (listen only) - highlight if currently listening
            l_button_color = (0, 255, 120) if not currently_listening else (0, 255, 255)
            l_button_thickness = 1 if not currently_listening else 2
            cv2.circle(frame, (button_spacing*2, button_y), 10, l_button_color, l_button_thickness)
            cv2.putText(frame, "L", 
                       (button_spacing*2 - 3, button_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

            # Draw S button (switch user)
            cv2.circle(frame, (button_spacing*3, button_y), 10, (255, 120, 0), 1)
            cv2.putText(frame, "S", 
                       (button_spacing*3 - 3, button_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

            # Draw T button (test microphone)
            cv2.circle(frame, (button_spacing*4, button_y), 10, (255, 255, 0), 1)
            cv2.putText(frame, "T", 
                       (button_spacing*4 - 3, button_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

            # Draw Q button (quit)
            cv2.circle(frame, (button_spacing*5, button_y), 10, (0, 0, 120), 1)
            cv2.putText(frame, "Q", 
                       (button_spacing*5 - 3, button_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

            # Add subtle instructions
            l_text = "Start Listening" if not currently_listening else "Stop Listening"
            cv2.putText(frame, f"R:Analyze  L:{l_text}  S:Switch  T:Test Mic  Q:Quit",
                       (button_spacing*6, button_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)

            # Display frame
            cv2.imshow('Smart Mirror', frame)

            # Handle key presses (non-blocking)
            key = cv2.waitKey(1) & 0xFF

            # Process key presses
            if key == ord('r') and not analyzing and not currently_listening:
                analyzing = True
                print("\n[ACTION] Starting new analysis...")
                emotion, posture = self.analyze_initial_state(duration=3)
                self.locked_emotion = emotion
                self.locked_body_language = posture
                self.analysis_complete = True
                analyzing = False

            elif key == ord('l') and not analyzing:
                if not currently_listening and self.analysis_complete:
                    currently_listening = True
                    response = self.listen_for_response()
                    if response:
                        follow_up = self.generate_response(response, self.locked_emotion, self.locked_body_language)
                        self.speak_message(follow_up)
                    currently_listening = False

            elif key == ord('s') and not analyzing and not currently_listening:
                print("\n[ACTION] Switching user...")
                self.prompt_user_login()

            elif key == ord('t') and not analyzing and not currently_listening:
                print("\n[ACTION] Testing microphone...")
                self.test_microphone()

            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Smart Mirror stopped.")

    def can_make_api_call(self):
        """Check if enough time has passed since last API call"""
        return time.time() - self.last_api_call >= self.api_cooldown

    def lock_emotion(self):
        """Lock in the current emotion"""
        print(f"\n[Emotion Lock] Current emotion: {self.last_emotion}")
        if self.last_emotion:
            self.locked_emotion = self.last_emotion
            self.current_message = f"Emotion locked: {self.locked_emotion}. Press SPACE for feedback"
            print(f"Successfully locked emotion: {self.locked_emotion}")
            return True
        print("Failed to lock emotion: No emotion detected")
        return False

    def prompt_user_login(self):
        """Handle user login based on mode"""
        if not self.web_mode:
            # Existing terminal login logic
            print("\n" + "="*50)
            print("USER LOGIN")
            print("="*50)

            while True:
                user_id = input("\nEnter your user ID (or 'new' to create a new user): ").strip()

                if not user_id:
                    print("User ID cannot be empty. Please try again.")
                    continue

                if user_id.lower() == 'new':
                    user_id = input("Enter a new user ID: ").strip()
                    if not user_id:
                        print("User ID cannot be empty. Please try again.")
                        continue

                    self.firebase_manager.create_user(user_id)
                    print(f"New user '{user_id}' created successfully!")
                    self.login_user(user_id)
                    break
                else:
                    if self.firebase_manager.check_user_exists(user_id):
                        print(f"User '{user_id}' found. Logging in...")
                        self.login_user(user_id)
                        break
                    else:
                        print(f"User '{user_id}' not found. Please try again or create a new user.")

            print("\nLogin successful! Press any key to continue...")
            input()
        # Web login will be handled by the web interface

    def web_analyze_state(self, frame, duration=3):
        """Analyze state for web interface with emotion averaging
        
        Args:
            frame: The video frame to analyze
            duration: Duration in seconds to analyze (default: 3)
        """
        print("\nStarting web analysis...")
        try:
            if frame is None:
                return {'status': 'error', 'message': 'No frame provided'}

            # Initialize emotion accumulator with larger buffer
            emotion_counts = {
                'angry': 0, 'disgust': 0, 'fear': 0, 
                'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0
            }

            # Analyze 5 frames
            for _ in range(5):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    if DeepFace is not None:
                        analysis = DeepFace.analyze(
                            img_path=rgb_frame,
                            actions=['emotion'],
                            enforce_detection=False,
                            detector_backend='opencv'
                        )

                        if isinstance(analysis, list):
                            analysis = analysis[0]

                        # Add emotion probabilities
                        for emotion, score in analysis['emotion'].items():
                            emotion_counts[emotion] += score
                    else:
                        # If DeepFace disabled, assume neutral
                        emotion_counts['neutral'] += 100
                except Exception as e:
                    print(f"Frame analysis error: {e}")
                    continue

            # Average the emotions
            for emotion in emotion_counts:
                emotion_counts[emotion] /= 5

            # Get dominant emotion
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
            emotion = dominant_emotion[0]
            confidence = dominant_emotion[1]

            # Store the emotion for response generation
            self.locked_emotion = emotion

            # Analyze posture and ignore visualization frame for JSON safety
            posture_result = self.analyze_posture(frame)
            if isinstance(posture_result, tuple):
                posture_cues, _viz = posture_result
            else:
                posture_cues = posture_result or []
            self.locked_body_language = posture_cues

            return {
                'status': 'success',
                'emotion': emotion,
                'confidence': confidence,
                'posture': posture_cues
            }

        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'emotion': 'neutral',
                'posture': []
            }

    def web_toggle_speech(self):
        """Toggle speech recording for web interface"""
        try:
            if not self.speech_active:
                # Starting recording
                self.speech_active = True
                self.audio_frames = []  # Reset frames
                print("Starting web recording...")
                return {
                    'status': 'started',
                    'message': 'Recording started'
                }

            else:
                # Stopping recording
                self.speech_active = False
                print("Stopping web recording...")

                if not self.audio_frames:
                    return {
                        'status': 'error',
                        'message': 'No audio data recorded'
                    }

                try:
                    # Save the WebM file first
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    webm_filename = f"recording_{timestamp}.webm"
                    wav_filename = f"recording_{timestamp}.wav"

                    # Combine all chunks into WebM file
                    with open(webm_filename, 'wb') as f:
                        webm_data = b''.join(self.audio_frames)
                        f.write(webm_data)

                    try:
                        # Convert WebM to WAV using ffmpeg through pydub
                        audio = AudioSegment.from_file(webm_filename, format="webm")
                        audio.export(wav_filename, format="wav")

                        # Now process the WAV file
                        with speech_recognition.AudioFile(wav_filename) as source:
                            audio = self.recognizer.record(source)
                            text = self.recognizer.recognize_google(audio)
                            print(f"Recognized text: {text}")

                            # Generate response
                            response = self.generate_response(text, self.locked_emotion, self.locked_body_language)
                            print(f"Generated response: {response}")

                            # Speak the response using ElevenLabs
                            if self.use_elevenlabs:
                                try:
                                    print("Converting to speech with ElevenLabs...")
                                    os.environ["ELEVEN_API_KEY"] = ELEVENLABS_API_KEY
                                    audio = self.generate_and_play_audio(
                                        text=response,
                                        voice="Bella",
                                        model="eleven_monolingual_v1"
                                    )
                                except Exception as e:
                                    print(f"ElevenLabs error: {e}")
                                    print("Falling back to pyttsx3...")
                                    self._speak_with_fallback(response)
                            else:
                                self._speak_with_fallback(response)

                            # Add to conversation history
                            self.add_to_history({
                                'verbal': text,
                                'response': response
                            })

                            return {
                                'status': 'success',
                                'text': text,
                                'response': response
                            }

                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        return {
                            'status': 'error',
                            'message': str(e)
                        }

                except Exception as e:
                    print(f"Error handling audio: {e}")
                    return {
                        'status': 'error',
                        'message': str(e)
                    }
                finally:
                    # Cleanup temporary files
                    try:
                        if os.path.exists(webm_filename):
                            os.remove(webm_filename)
                        if os.path.exists(wav_filename):
                            os.remove(wav_filename)
                    except Exception as e:
                        print(f"Error cleaning up files: {e}")
                    self.audio_frames = []

        except Exception as e:
            print(f"Web toggle speech error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def login_user(self, user_id):
        """Login user and set up user-specific data"""
        self.user_id = user_id
        self.current_user = user_id  # Set both user tracking variables

        # Update last login time
        self.firebase_manager.update_last_login(user_id)

        # Load user history if available
        try:
            user_history = self.firebase_manager.get_user_history(user_id)
            if user_history:
                self.conversation_history = user_history
                print(f"Loaded {len(user_history)} previous interactions")
            else:
                self.conversation_history = []
                print("No previous interaction history found")
        except Exception as e:
            print(f"Error loading user history: {e}")
            self.conversation_history = []
            print("Continuing without history data")

        print(f"Logged in as: {user_id}")
        return True

    def start_recording(self):
        """Start recording audio with error handling"""
        print("\n[Audio] Starting recording...")
        try:
            # Close any existing stream
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()

            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=self.input_device_index,  # Use the selected device
                stream_callback=None
            )
            self.recording = True
            self.audio_frames = []
            self.recording_start_time = time.time()
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.recording = False
            return False

    def stop_recording(self):
        """Stop recording with safety checks"""
        if not self.recording:
            return None

        print("[Audio] Stopping recording...")
        try:
            self.recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            if not self.audio_frames:
                print("No audio data recorded")
                return None

            # Save recording to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"

            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_frames))

            print(f"Recording saved: {filename}")
            return filename
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None

    def add_to_history(self, interaction):
        """Add an interaction to conversation history and Firebase"""
        timestamp = datetime.now()

        # Create interaction data
        interaction_data = {
            'timestamp': timestamp,
            'emotion': self.locked_emotion,
            'verbal': interaction.get('verbal', ''),
            'response': interaction.get('response', ''),
            'session_id': self.current_session['start_time']
        }

        # Add to local history
        self.conversation_history.append(interaction_data)
        self.current_session['interactions'].append(interaction_data)

        # Save to Firebase
        if self.current_user:
            try:
                self.firebase.save_interaction(
                    self.current_user,
                    interaction_data
                )
            except Exception as e:
                print(f"Error saving to Firebase: {e}")

    def prompt_mode_selection(self):
        """Prompt user to select between web and local mode"""
        print("\n" + "="*50)
        print("MODE SELECTION")
        print("="*50)
        print("\nPlease select how you want to run the Smart Mirror:")
        print("1. Local Mode (terminal-based)")
        print("2. Web Mode (access through web browser)")

        while True:
            choice = input("\nEnter your choice (1/2): ").strip()
            if choice == "1":
                print("\nStarting in Local Mode...")
                return False
            elif choice == "2":
                print("\nStarting in Web Mode...")
                print("Please access through your web browser")
                return True
            else:
                print("Invalid choice. Please enter 1 or 2.")

    def start_web_server(self):
        """Start Flask server for web interface"""
        app = Flask(__name__)

        @app.route('/favicon.ico')
        def favicon():
            return send_file('static/favicon.ico', mimetype='image/vnd.microsoft.icon')

        @app.route('/')
        def home():
            return render_template('mirror.html')

        @app.route('/analyze', methods=['POST'])
        def analyze():
            """Analyze the image sent from the browser"""
            try:
                # Get image data from request
                image_data = request.files.get('image')
                if image_data:
                    # Convert to OpenCV format
                    nparr = np.frombuffer(image_data.read(), np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # Run analysis
                    self.analyzing = True
                    result = self.web_analyze_state(frame, duration=3)
                    self.analyzing = False
                    return jsonify(result)
                return jsonify({'status': 'error', 'message': 'No image data received'})
            except Exception as e:
                print(f"Analysis error: {e}")
                return jsonify({'status': 'error', 'message': str(e)})

        @app.route('/toggle_speech', methods=['POST'])
        def toggle_speech():
            try:
                result = self.web_toggle_speech()
                return jsonify(result)  # Return the result directly
            except Exception as e:
                print(f"Toggle speech route error: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                })

        @app.route('/stop_listening', methods=['POST'])
        def stop_listening():
            """Process recorded audio and return response with audio"""
            try:
                # Check if we received an audio file
                if 'audio' not in request.files:
                    return jsonify({
                        "success": False, 
                        "error": "No audio file received"
                    })

                audio_file = request.files['audio']

                # Save to a temporary file
                temp_dir = tempfile.gettempdir()
                webm_filename = os.path.join(temp_dir, f"recording_{int(time.time())}.webm")
                wav_filename = os.path.join(temp_dir, f"recording_{int(time.time())}.wav")

                audio_file.save(webm_filename)
                print(f"Audio saved to {webm_filename}, size: {os.path.getsize(webm_filename)} bytes")

                # Check if file is too small
                if os.path.getsize(webm_filename) < 1000:  # Less than 1KB
                    return jsonify({
                        "success": False,
                        "error": "No speech detected (audio file too small)",
                        "transcript": "",
                        "response": "I didn't hear anything. Could you try speaking louder?"
                    })

                try:
                    # Convert WebM to WAV using pydub
                    audio = AudioSegment.from_file(webm_filename, format="webm")
                    audio.export(wav_filename, format="wav")

                    # Use speech recognition on the WAV file
                    with speech_recognition.AudioFile(wav_filename) as source:
                        audio_data = self.recognizer.record(source)
                        transcript = self.recognizer.recognize_google(audio_data)
                        print(f"Recognized text: {transcript}")

                        # Generate response
                        response = self.generate_response(
                            transcript, 
                            self.locked_emotion if hasattr(self, 'locked_emotion') else 'neutral',
                            self.locked_body_language if hasattr(self, 'locked_body_language') else []
                        )
                        print(f"Generated response: {response}")

                        # Generate audio for the response and encode it as base64
                        audio_data = None
                        audio_format = "mp3"

                        try:
                            if self.use_openai_tts:
                                # Generate the audio but DON'T play it (we want to send it to client)
                                voice = OPENAI_TTS_VOICE if 'OPENAI_TTS_VOICE' in globals() else "alloy"
                                model = OPENAI_TTS_MODEL if 'OPENAI_TTS_MODEL' in globals() else "tts-1"

                                # Generate the audio
                                print(f"Generating audio with voice: {voice}, model: {model}")

                                # Create audio using OpenAI API
                                audio_response = client.audio.speech.create(
                                    model=model,
                                    voice=voice,
                                    input=response
                                )

                                # Convert to base64 for sending to client
                                audio_data = base64.b64encode(audio_response.content).decode('utf-8')
                                print(f"Generated audio response, size: {len(audio_response.content)} bytes")
                        except Exception as audio_error:
                            print(f"Error generating audio response: {audio_error}")
                            # Continue without audio if there's an error

                        # Clean up temporary files
                        try:
                            if os.path.exists(webm_filename):
                                os.remove(webm_filename)
                            if os.path.exists(wav_filename):
                                os.remove(wav_filename)
                        except Exception as e:
                            print(f"Error cleaning up files: {e}")

                        # Add to conversation history
                        self.add_to_history({
                            'verbal': transcript,
                            'response': response
                        })

                        # Build response object
                        result = {
                            "success": True,
                            "transcript": transcript,
                            "response": response
                        }

                        # Add audio if we generated it
                        if audio_data:
                            result["audio"] = audio_data
                            result["audio_format"] = audio_format

                        return jsonify(result)

                except Exception as e:
                    print(f"Error processing audio: {e}")
                    return jsonify({
                        "success": False,
                        "error": str(e),
                        "transcript": "",
                        "response": "I had trouble understanding that. Could you try again?"
                    })

            except Exception as e:
                print(f"Stop listening error: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                })

        @app.route('/login', methods=['POST'])
        def login():
            data = request.get_json()
            user_id = data.get('user_id')
            if user_id:
                success = self.login_user(user_id)
                return jsonify({'success': success})
            return jsonify({'success': False})

        @app.route('/audio', methods=['POST'])
        def receive_audio():
            """Handle incoming audio data"""
            if request.method == 'POST':
                try:
                    if not self.speech_active:
                        return jsonify({
                            'status': 'error',
                            'message': 'Recording not active'
                        })

                    audio_data = request.files.get('audio')
                    if audio_data:
                        # Simply store the raw data
                        audio_bytes = audio_data.read()
                        if audio_bytes:
                            self.audio_frames.append(audio_bytes)
                            return jsonify({'status': 'success'})

                    return jsonify({
                        'status': 'error',
                        'message': 'No audio data received'
                    })

                except Exception as e:
                    print(f"Error receiving audio: {e}")
                    return jsonify({
                        'status': 'error',
                        'message': str(e)
                    })
            return jsonify({'status': 'error', 'message': 'Invalid request'})

        @app.route('/test_mic', methods=['POST'])
        def test_mic():
            """Handle microphone test request"""
            try:
                result = self.test_microphone_web()
                return jsonify(result)
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                })

        if self.web_mode:
            app.run(host='0.0.0.0', port=8080, threaded=True)

    def _find_working_microphone(self):
        """Find first working microphone"""
        print("\nLooking for working microphone...")
        print("Available audio devices:")

        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"Index {i}: {device_info['name']}")
                    # Test if microphone works
                    test_stream = self.audio.open(
                        format=self.audio_format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=self.chunk
                    )
                    test_stream.close()
                    print(f"Found working microphone: {device_info['name']}")
                    return i
            except:
                continue

        print("No working microphone found!")
        return None

    def test_microphone_web(self):
        """Test microphone for web interface"""
        try:
            # Start a short test recording
            self.speech_active = True
            self.audio_frames = []

            # Wait for 2 seconds to collect audio
            time.sleep(2)

            # Stop recording
            self.speech_active = False

            if self.audio_frames:
                # Save test audio
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_{timestamp}.wav"

                try:
                    # Convert audio chunks to WAV
                    audio_segment = AudioSegment.empty()
                    for chunk in self.audio_frames:
                        segment = AudioSegment(
                            data=chunk,
                            sample_width=2,
                            frame_rate=16000,
                            channels=1
                        )
                        audio_segment += segment

                    # Export as WAV
                    audio_segment.export(filename, format="wav")

                    # Check audio levels
                    peak_amplitude = audio_segment.max_dBFS

                    # Cleanup
                    os.remove(filename)

                    if peak_amplitude > -float('inf'):
                        return {
                            'status': 'success',
                            'message': f'Microphone is working. Audio level: {peak_amplitude:.1f} dB'
                        }
                    else:
                        return {
                            'status': 'error',
                            'message': 'No audio detected. Please check your microphone.'
                        }

                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Error processing audio: {str(e)}'
                    }
            else:
                return {
                    'status': 'error',
                    'message': 'No audio data received. Please check your microphone permissions.'
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Microphone test failed: {str(e)}'
            }

    def generate_and_play_audio(self, text, voice=None, model=None):
        """Generate and play audio using OpenAI TTS API"""
        try:
            if not voice:
                voice = OPENAI_TTS_VOICE  # Default to the configured voice (sage)
            if not model:
                model = OPENAI_TTS_MODEL  # Default to the configured model

            print(f"[TTS] Generating audio with OpenAI voice: {voice}, model: {model}")

            # Create a temp file path for audio
            temp_file = "temp_audio.mp3"

            # Use the new client directly
            audio_response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            audio_content = audio_response.content

            if not audio_content:
                raise Exception("Failed to generate audio content")

            # Save to temporary file then play
            with open(temp_file, "wb") as audio_file:
                audio_file.write(audio_content)

            print(f"Audio file saved to {temp_file}, size: {os.path.getsize(temp_file)} bytes")

            # Play the audio file based on platform
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {temp_file}")
                print("Played audio with afplay")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {temp_file}")
                print("Played audio with start command")
            else:  # Linux or other
                os.system(f"mpg123 {temp_file}")
                print("Played audio with mpg123")

            # Return the audio data for potential further use
            return audio_content

        except Exception as e:
            print(f"Error generating audio: {e}")
            traceback.print_exc()  # Print detailed traceback
            return None

    def web_generate_response(self, text):
        # ... other code ...
        
        # Store the risk level and indicators for the admin panel
        risk_level = getattr(self, 'last_risk_level', None)
        risk_indicators = getattr(self, 'last_risk_indicators', [])
        is_emergency = getattr(self, 'last_is_emergency', False)
        
        # Log interaction details
        print(f"\n[WebMirror] RESPONSE VALUES:")
        print(f"[WebMirror] Emotion: {emotion}")
        print(f"[WebMirror] Body Language: {body_language}")
        print(f"[WebMirror] Risk Level: {risk_level}%")
        print(f"[WebMirror] Risk Indicators: {', '.join(risk_indicators)}")
        print(f"[WebMirror] Emergency Status: {is_emergency}")

    def _update_landmark_history(self, landmarks):
        """Helper function to update landmark history buffers."""
        self.nose_history.append(landmarks[self.mp_pose.PoseLandmark.NOSE])
        self.left_shoulder_history.append(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        self.right_shoulder_history.append(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        self.left_hip_history.append(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP])
        self.right_hip_history.append(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP])

        if len(self.nose_history) > self.landmark_history_size:
            self.nose_history.pop(0)
            self.left_shoulder_history.pop(0)
            self.right_shoulder_history.pop(0)
            self.left_hip_history.pop(0)
            self.right_hip_history.pop(0)

# Add this near the top after checking OpenAI version
# Comprehensive OpenAI API compatibility layer
class OpenAICompat:
    """Client-based OpenAI API implementation"""

    @staticmethod
    def chat_completion(model, messages, temperature=0.7, max_tokens=None):
        """Create a chat completion"""
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in chat completion: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def generate_speech(text, voice, model):
        """Generate speech from text"""
        try:
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            return response.content
        except Exception as e:
            print(f"Error generating audio: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def transcribe_audio(audio_file, model="whisper-1"):
        """Transcribe audio to text"""
        try:
            with open(audio_file, "rb") as file:
                transcript = client.audio.transcriptions.create(
                    model=model,
                    file=file
                )
            return transcript.text
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            traceback.print_exc()
            return None

# Wrapper functions for backward compatibility
def generate_speech(text, voice, model):
    """Wrapper for backward compatibility"""
    return OpenAICompat.generate_speech(text, voice, model)

def transcribe_audio(audio_file, model="whisper-1"):
    """Wrapper for backward compatibility"""
    return OpenAICompat.transcribe_audio(audio_file, model)

def main():
    """Main entry point"""
    mirror = SmartMirror()  # This will now prompt for mode first

    if mirror.web_mode:
        mirror.start_web_server()  # Start web server if in web mode
    else:
        mirror.run()  # Run in local mode

if __name__ == "__main__":
    main()