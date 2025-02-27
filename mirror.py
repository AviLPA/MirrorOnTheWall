import cv2
import numpy as np
import time
import openai
from datetime import datetime
import pyttsx3
from deepface import DeepFace
from firebase_config import FirebaseManager
import pyaudio
import wave
from elevenlabs import generate, play
import elevenlabs
from config import OPENAI_API_KEY, ELEVENLABS_API_KEY
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

# Set global timeout for all HTTP requests to prevent freezing
requests.adapters.DEFAULT_TIMEOUT = 15  # 15 seconds timeout
openai.timeout = 15  # 15 seconds timeout for OpenAI API calls

try:
    from config import OPENAI_API_KEY
except ImportError:
    print("""
    Error: config.py not found!
    1. Create a file named config.py
    2. Add your OpenAI API key like this:
       OPENAI_API_KEY = "your-api-key-here"
    """)
    exit(1)

class SmartMirror:
    def __init__(self):
        """Initialize the Smart Mirror with all required components"""
        # Initialize OpenAI
        openai.api_key = OPENAI_API_KEY
        
        # Initialize ElevenLabs
        try:
            elevenlabs.set_api_key(ELEVENLABS_API_KEY)
            self.use_elevenlabs = True
            print("ElevenLabs configured successfully")
        except Exception as e:
            self.use_elevenlabs = False
            print(f"ElevenLabs not configured, using fallback TTS: {e}")
        
        # Initialize text-to-speech engine (fallback)
        self.engine = pyttsx3.init()
        
        # Initialize Firebase and user login - MOVED HERE
        try:
            self.firebase_manager = FirebaseManager()
            self.user_id = None
            self.current_user = None  # Added to track current user
            self.prompt_user_login()  # Single login prompt
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
            print("Continuing without Firebase functionality")
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
        
        # Initialize face detection and emotion tracking
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_frequency = 10  # Only analyze emotions every 10 frames
        self.frame_count = 0
        
        # Map emotions to supportive language
        self.emotion_responses = {
            "angry": ["I notice you seem frustrated. Let's work through this together.", 
                     "It's okay to feel angry sometimes. How can I help?"],
            "disgust": ["I sense you're uncomfortable. Let's change direction.", 
                       "Your feelings are valid. What would help right now?"],
            "fear": ["I'm here with you. What's causing concern?", 
                    "It's okay to feel anxious. Let's take a deep breath together."],
            "happy": ["Your positive energy is wonderful!", 
                     "I'm glad you're feeling good today!"],
            "sad": ["I'm here for you if you want to talk about what's troubling you.", 
                   "It's okay to feel down sometimes. What might help lift your spirits?"],
            "surprise": ["You seem surprised! What's on your mind?", 
                        "I notice your reaction. What are you thinking about?"],
            "neutral": ["How are you feeling today?", 
                       "Is there something specific you'd like to discuss?"]
        }
        
        # Initialize audio recording
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Lower sample rate for better performance
        self.chunk = 1024  # Smaller chunk size
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.recording = False
        
        # Initialize speech queue for asynchronous TTS
        self.speech_queue = Queue()
        self.speech_thread = threading.Thread(target=self._process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        # Initialize speech recognition with improved parameters
        print("\nInitializing speech recognition system...")
        self.recognizer = Recognizer()
        
        # Print available microphones for debugging
        print("\nAvailable microphones:")
        mic_list = speech_recognition.Microphone.list_microphone_names()
        for i, mic in enumerate(mic_list):
            print(f"{i}: {mic}")
        
        # Set default microphone
        try:
            print("\nTrying to initialize default microphone...")
            self.mic = Microphone(sample_rate=16000, chunk_size=1024)
            print("Default microphone initialized successfully")
            
            # Test microphone access
            try:
                with self.mic as source:
                    print("Microphone access test successful!")
                    # Configure recognizer parameters for better pause handling
                    self.recognizer.energy_threshold = 300  # Lower threshold for quieter speech
                    self.recognizer.dynamic_energy_threshold = True  # Adjust for ambient noise
                    self.recognizer.pause_threshold = 2.0  # Allow 2 second pauses
                    self.recognizer.phrase_threshold = 0.3  # More sensitive phrase detection
                    self.recognizer.non_speaking_duration = 1.0  # Shorter non-speaking detection
                    print(f"Speech recognition configured with energy_threshold={self.recognizer.energy_threshold}")
            except Exception as e:
                print(f"ERROR: Could not access microphone: {e}")
                print("Speech recognition may not work properly!")
        except Exception as e:
            print(f"ERROR: Failed to initialize microphone: {e}")
            print("Attempting to use system default microphone...")
            try:
                # Try with default system settings
                self.mic = Microphone()
                print("System default microphone initialized")
            except Exception as e2:
                print(f"CRITICAL ERROR: Could not initialize any microphone: {e2}")
                print("Speech recognition will not work!")
        
        # Initialize conversation state
        self.locked_emotion = None
        self.locked_body_language = []
        self.ready_to_send = False
        self.in_conversation = False
        
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
        """Enhanced face and posture analysis using MediaPipe"""
        # Skip frames to reduce CPU usage
        self.frame_count += 1
        if self.frame_count % 5 != 0:  # Only process every 5th frame (increased from 3)
            if hasattr(self, 'last_annotated_frame'):
                return self.last_posture_cues, self.last_detected_emotion, self.last_annotated_frame
            else:
                # Return original frame if we don't have a previous result
                return [], "neutral", frame
        
        # Use a smaller frame for processing to reduce CPU usage
        small_frame = cv2.resize(frame, (320, 240))
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        small_height, small_width = small_frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        face_results = self.face_mesh.process(rgb_frame)
        face_detected = False
        posture_cues = []
        
        if face_results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face mesh (simplified to reduce CPU usage)
                # Only draw every 5th landmark to reduce CPU usage
                landmarks_to_draw = []
                for i, landmark in enumerate(face_landmarks.landmark):
                    if i % 5 == 0:  # Only include every 5th landmark
                        landmarks_to_draw.append((
                            int(landmark.x * width),
                            int(landmark.y * height)
                        ))
                
                # Draw simplified face mesh
                for point in landmarks_to_draw:
                    cv2.circle(annotated_frame, point, 1, (0, 255, 0), -1)
                
                # Analyze head pose
                if len(face_landmarks.landmark) > 4:  # Make sure we have enough landmarks
                    nose_tip = face_landmarks.landmark[4]
                    nose_x = int(nose_tip.x * small_width * (width / small_width))
                    nose_y = int(nose_tip.y * small_height * (height / small_height))
                    
                    # Head position analysis
                    if nose_x < width * 0.4:
                        posture_cues.append("head tilted left")
                    elif nose_x > width * 0.6:
                        posture_cues.append("head tilted right")
                    
                    if nose_y < height * 0.3:
                        posture_cues.append("head tilted back")
                    elif nose_y > height * 0.5:
                        posture_cues.append("head tilted forward")

        # Emotion analysis using DeepFace - only run every emotion_detection_frequency frames
        dominant_emotion = "neutral"
        if self.frame_count % self.emotion_detection_frequency == 0:
            try:
                analysis = DeepFace.analyze(
                    small_frame,  # Use smaller frame for analysis
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',  # Use opencv instead of mediapipe for better performance
                    prog_bar=False  # Disable progress bar to reduce console output
                )
                
                if analysis:
                    emotions = analysis[0]['emotion']
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    dominant_emotion = sorted_emotions[0][0]
                    
                    # Create emotion visualization (simplified)
                    cv2.rectangle(annotated_frame, (10, 10), (200, 40), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, 
                               f"Emotion: {dominant_emotion} ({sorted_emotions[0][1]:.1f}%)",
                               (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, 
                               (0, 255, 0), 
                               1)
                    
                    # Store the last detected emotion
                    self.last_detected_emotion = dominant_emotion
            except Exception as e:
                print(f"Emotion detection error: {e}")
                # Use the last detected emotion if available
                if hasattr(self, 'last_detected_emotion'):
                    dominant_emotion = self.last_detected_emotion
        elif hasattr(self, 'last_detected_emotion'):
            # Use the last detected emotion between detection frames
            dominant_emotion = self.last_detected_emotion
        
        # Store results for frame skipping
        self.last_posture_cues = posture_cues
        self.last_detected_emotion = dominant_emotion
        self.last_annotated_frame = annotated_frame
        
        return posture_cues, dominant_emotion, annotated_frame

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
        """Get AI feedback based primarily on verbal input and emotion"""
        print("\n[API Call] Preparing feedback request...")
        print(f"Emotion: {emotion}")
        
        verbal_context = ""
        history_context = ""
        
        # Get transcribed speech if available
        if self.last_recording:
            try:
                print("Transcribing audio message...")
                
                # Use a separate thread with timeout for audio transcription
                from requests.exceptions import Timeout
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                
                def transcribe_audio():
                    try:
                        with open(self.last_recording, "rb") as audio_file:
                            return openai.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                timeout=20  # 20 second timeout for the HTTP request
                            )
                    except Exception as e:
                        print(f"Transcription API call error: {e}")
                        return None
                
                # Execute API call with timeout
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(transcribe_audio)
                    try:
                        transcript = future.result(timeout=25)  # 25 second overall timeout
                        
                        if transcript is None:
                            raise Exception("Transcription API call returned None")
                            
                        verbal_context = f"They expressed verbally: {transcript.text}"
                        print(f"Transcribed: {transcript.text}")
                    except FutureTimeoutError:
                        print("Transcription API call timed out after 25 seconds")
                        verbal_context = "They expressed something verbally, but I couldn't transcribe it in time."
            except Exception as e:
                print(f"Error transcribing audio: {e}")
                verbal_context = "They expressed something verbally, but I couldn't transcribe it."
        
        # Add history context if available
        if self.current_user:
            progress = self.firebase.analyze_progress(self.current_user)
            if progress:
                recent = progress['recent_pattern']
                history_context = f"Recent emotional states: {', '.join(recent)}"
        
        prompt = f"""The GPT is designed to function as a supportive friend for young adults, providing self-esteem boosting compliments, affirmations, and advice to promote emotional well-being. It uses psychology-based techniques grounded in credible research to foster positivity, self-awareness, and confidence without replacing professional therapy or delving into mental/physical health diagnoses. It begins each interaction with a friendly and encouraging greeting, addressing users by their name for enhanced connection and rapport. The GPT employs the psychological technique of 'mirroring,' thoughtfully integrating users' language and emotions into responses to create empathy and understanding, without simply repeating their statements. Compliments and feedback are always contextual, supportive, and aimed at improving the user's mood while avoiding condescension, insensitivity, or harm. It proactively offers insights and suggestions, always ensuring that information is backed by psychological works and avoids unverified or speculative advice. Complex psychological terms are replaced with simple, friendly language to maintain a warm and empathetic tone. It suggests wellness activities tailored to the user's emotional state, such as relaxation techniques or mood-boosting actions, while maintaining a friendly and proactive demeanor. The GPT avoids engaging in sensitive or controversial topics, including self-harm, mental or physical health issues, violence, and political or religious debates. It steers clear of medical advice, diagnoses, or marginalizing comments. Instead, it focuses on fostering trust, positivity, and self-reflection, ensuring users feel valued, supported, and empowered. It cannot assist in discussions related to harmful actions or inappropriate content and redirects users to appropriate resources when necessary. The answers should not be more than 4 sentences and not use bullet points. This should be around 5-10 seconds in speech.

Based on the following observations:
1. Verbal Expression: {verbal_context if verbal_context else "No verbal input detected"}
2. Current Emotion: {emotion if emotion else 'neutral'}
3. History: {history_context}

Please provide a supportive response that primarily addresses what they've expressed verbally, while acknowledging their emotional state."""

        print("Sending request to OpenAI...")
        try:
            # Use a separate thread with timeout for OpenAI API call
            from requests.exceptions import Timeout
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            
            def make_api_call():
                try:
                    return openai.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.7,
                        timeout=15  # 15 second timeout for the HTTP request
                    )
                except Exception as e:
                    print(f"API call error: {e}")
                    return None
            
            # Execute API call with timeout
            with ThreadPoolExecutor() as executor:
                future = executor.submit(make_api_call)
                try:
                    response = future.result(timeout=20)  # 20 second overall timeout
                    
                    if response is None:
                        raise Exception("API call returned None")
                        
                    response_text = response.choices[0].message.content
                    print("\n[AI Response]")
                    print("=" * 50)
                    print(response_text)
                    print("=" * 50)
                    
                    # Update last API call time
                    self.last_api_call = time.time()
                    
                    # Save to Firebase if user is logged in
                    if self.current_user:
                        self.firebase.save_session(
                            self.current_user,
                            emotion,
                            verbal_context,  # Store verbal input instead of posture
                            response_text
                        )
                    
                    return response_text
                except FutureTimeoutError:
                    print("API call timed out after 20 seconds")
                    return "I notice how you're feeling. Let me think about that for a moment."
        except Exception as e:
            print(f"ERROR in API call: {e}")
            return "I hear what you're saying. Let's talk more about that."

    def _process_speech_queue(self):
        """Process speech queue in background"""
        while True:
            if not self.speech_queue.empty():
                message = self.speech_queue.get()
                try:
                    if self.use_elevenlabs and len(message) > 0:
                        print("Attempting to use ElevenLabs...")
                        try:
                            # Check if message is too long for ElevenLabs
                            if len(message) > 300:
                                print("Message too long for ElevenLabs, truncating...")
                                message = message[:300] + "..."
                            
                            # Generate and play audio directly (no timeout)
                            audio = generate(
                                text=message,
                                voice="Rachel",
                                model="eleven_monolingual_v1",
                                api_key=ELEVENLABS_API_KEY
                            )
                            play(audio)
                                
                        except Exception as e:
                            print(f"Error with ElevenLabs: {e}")
                            # After first ElevenLabs failure, disable it for the rest of the session
                            if "API key invalid" in str(e) or "quota exceeded" in str(e).lower():
                                print("Disabling ElevenLabs for the rest of the session due to API key or quota issues")
                                self.use_elevenlabs = False
                            # Fall back to local TTS
                            self.tts_engine.say(message)
                            self.tts_engine.runAndWait()
                    else:
                        # Use local TTS
                        self.tts_engine.say(message)
                        self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"Error speaking message: {e}")
                    print("Falling back to default TTS...")
                    try:
                        self.tts_engine.say(message)
                        self.tts_engine.runAndWait()
                    except Exception as fallback_error:
                        print(f"Even fallback TTS failed: {fallback_error}")
                        # Last resort - just continue without speaking
            else:
                # Sleep to prevent CPU spinning
                time.sleep(0.1)

    def speak_message(self, message):
        """Speak message with visual feedback"""
        # Show speaking indicator with subtle UI for mirror effect
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Create a subtle gradient overlay
            height, width = frame.shape[:2]
            gradient = np.zeros((60, width, 3), dtype=np.uint8)
            for i in range(60):
                alpha = 0.6 - (i / 60) * 0.5  # Less opaque gradient
                gradient[i, :] = (60, 30, 0)  # Dark orange background
            
            # Apply gradient overlay
            cv2.addWeighted(gradient, 0.7, frame[:60], 0.3, 0, frame[:60])
            
            # Add a small speaking animation
            speak_x = 30
            speak_y = 30
            cv2.circle(frame, (speak_x, speak_y), 10, (0, 165, 255), -1)  # Orange circle
            cv2.circle(frame, (speak_x, speak_y), 13, (0, 140, 220), 1)   # Outer ring
            
            # Add text with smaller font
            cv2.putText(frame, "Speaking...", 
                       (speak_x + 20, speak_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)
            
            # Display a preview of the message
            if len(message) > 50:
                preview = message[:47] + "..."
            else:
                preview = message
                
            cv2.putText(frame, preview, 
                       (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)
            
            cv2.imshow('Smart Mirror', frame)
            cv2.waitKey(1)
        
        # Add message to speech queue
        self.speech_queue.put(message)

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
        """Analyze initial emotional state with performance optimizations"""
        print("\nStarting initial state analysis...")
        emotions = []
        postures = []
        
        start_time = time.time()
        frames_analyzed = 0
        
        # Initialize DeepFace parameters with better settings
        analysis_settings = {
            'actions': ['emotion'],
            'enforce_detection': True,  # Changed to True to ensure face detection
            'detector_backend': 'opencv',  # More reliable than SSD
            'align': True  # Enable face alignment for better accuracy
        }
        
        # Analyze fewer frames but more accurately
        target_frames = 5  # Increased from 3 to 5 frames for better accuracy
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
            
            # Show analyzing status (keeping UI the same)
            height, width = frame.shape[:2]
            gradient = np.zeros((60, width, 3), dtype=np.uint8)
            for i in range(60):
                alpha = 0.7 - (i / 60) * 0.5
                gradient[i, :] = (0, 80, 0)
            
            cv2.addWeighted(gradient, 0.6, frame[:60], 0.4, 0, frame[:60])
            
            progress = int((current_time - start_time) / duration * 100)
            cv2.putText(frame, f"Analyzing... {progress}%", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Smart Mirror', frame)
            cv2.waitKey(1)
            
            # Only analyze at specific intervals
            if current_time - last_analysis_time >= frames_between_analysis and frames_analyzed < target_frames:
                try:
                    # Prepare frame for analysis
                    # Convert to RGB (DeepFace expects RGB)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect face first to ensure good quality frame for analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(100, 100)  # Increased minimum face size
                    )
                    
                    if len(faces) > 0:
                        # Get the largest face
                        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                        
                        # Add padding around the face
                        padding = 40
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(width - x, w + 2*padding)
                        h = min(height - y, h + 2*padding)
                        
                        # Extract face region with padding
                        face_region = rgb_frame[y:y+h, x:x+w]
                        
                        # Ensure face region is large enough
                        if face_region.shape[0] > 0 and face_region.shape[1] > 0:
                            print(f"\nAnalyzing frame {frames_analyzed + 1}/{target_frames}...")
                            
                            # Analyze the face region
                            analysis = DeepFace.analyze(
                                img_path=face_region,
                                **analysis_settings
                            )
                            
                            if analysis:
                                current_emotions = analysis[0]['emotion']
                                sorted_emotions = sorted(current_emotions.items(), key=lambda x: x[1], reverse=True)
                                
                                if sorted_emotions[0][1] > 40:  # Increased confidence threshold
                                    emotions.append(sorted_emotions[0][0])
                                    frames_analyzed += 1
                                    print(f"Detected emotion: {sorted_emotions[0][0]} ({sorted_emotions[0][1]:.1f}%)")
                                    
                                    # Show detection rectangle and emotion
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(frame, 
                                              f"{sorted_emotions[0][0]}: {sorted_emotions[0][1]:.1f}%",
                                              (x, y-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.6, 
                                              (0, 255, 0), 
                                              2)
                                    cv2.imshow('Smart Mirror', frame)
                                    cv2.waitKey(1)
            
                    last_analysis_time = current_time
                    
                except Exception as e:
                    print(f"Frame analysis error: {str(e)}")
                    time.sleep(0.1)
            
            # Analyze posture less frequently
            if int(current_time * 2) % 2 == 0:
                posture_cues = self.analyze_posture(frame)
                if posture_cues:
                    postures.extend(posture_cues)
        
        # Determine dominant emotion with weighted recent bias
        if emotions:
            # Give more weight to recent emotions
            weighted_emotions = emotions[-2:] + emotions  # Duplicate recent emotions
            try:
                from statistics import mode
                dominant_emotion = mode(weighted_emotions)
                confidence = (weighted_emotions.count(dominant_emotion) / len(weighted_emotions)) * 100
                print(f"\nAnalysis Results:")
                print(f"Dominant emotion: {dominant_emotion} (Confidence: {confidence:.1f}%)")
            except statistics.StatisticsError:
                dominant_emotion = emotions[-1]  # Use most recent emotion
                print(f"No clear dominant emotion, using most recent: {dominant_emotion}")
        else:
            dominant_emotion = "neutral"
            print("No emotions detected, defaulting to neutral")
        
        unique_postures = list(set(postures))
        print(f"Detected posture cues: {unique_postures}")
        
        return dominant_emotion, unique_postures

    def analyze_posture(self, frame):
        """Separate method for posture analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        posture_cues = []
        for (x, y, w, h) in faces:
            # More precise posture analysis
            frame_center = frame.shape[1] / 2
            face_center = x + (w / 2)
            face_top = y
            face_height_ratio = y / frame.shape[0]
            
            if abs(face_center - frame_center) > frame.shape[1] * 0.15:
                posture_cues.append("head tilted to side")
            if face_height_ratio < 0.2:
                posture_cues.append("head tilted back")
            elif face_height_ratio > 0.4:
                posture_cues.append("forward head posture")
        
        return posture_cues

    def generate_response(self, verbal_response, locked_emotion, locked_posture):
        """Generate response using conversation history"""
        print("\nGenerating response with:")
        print(f"Verbal: {verbal_response}")
        print(f"Current Emotion: {locked_emotion}")
        
        # Update last API call time
        self.last_api_call = time.time()
        
        # Get recent interactions
        recent_interactions = self.conversation_history[-3:] if self.conversation_history else []
        conversation_context = ""
        
        if recent_interactions:
            conversation_context = "Previous interactions:\n"
            for interaction in recent_interactions:
                conversation_context += f"- They felt {interaction['emotion']} and said: '{interaction['verbal']}'\n"
        
        prompt = f"""The GPT is designed to function as a supportive friend...
        
Previous Context:
{conversation_context}

Current Observation:
1. They are currently feeling: {locked_emotion}
2. They just said: "{verbal_response}"
3. Their body language shows: {', '.join(locked_posture)}

Please provide a supportive response that:
1. Acknowledges their current state
2. Shows understanding of how their feelings have evolved from previous interactions
3. Maintains a natural conversation flow
4. Keeps the response warm and encouraging"""

        try:
            # Use a separate thread with timeout for OpenAI API call
            from requests.exceptions import Timeout
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            
            def make_api_call():
                try:
                    return openai.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.7,
                        timeout=15  # 15 second timeout for the HTTP request
                    )
                except Exception as e:
                    print(f"API call error: {e}")
                    return None
            
            # Execute API call with timeout
            with ThreadPoolExecutor() as executor:
                future = executor.submit(make_api_call)
                try:
                    response = future.result(timeout=20)  # 20 second overall timeout
                    
                    if response is None:
                        raise Exception("API call returned None")
                        
                    response_text = response.choices[0].message.content
                    
                    # Save this interaction
                    self.add_to_history({
                        'verbal': verbal_response,
                        'response': response_text
                    })
                    
                    return response_text
                except FutureTimeoutError:
                    print("API call timed out after 20 seconds")
                    return "I'm thinking about what you said. Can you tell me more?"
                    
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I understand. Would you like to tell me more about that?"

    def listen_for_response(self):
        """Listen for user's verbal response with improved pause handling and error recovery"""
        print("\n[ACTION] Listening for response...")
        
        # Get frame for UI
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
        
        # Create a subtle gradient overlay for listening
        if ret:
            gradient = np.zeros((60, width, 3), dtype=np.uint8)
            for i in range(60):
                alpha = 0.7 - (i / 60) * 0.5  # Less opaque gradient
                gradient[i, :] = (0, 100, 50)  # Dark teal background for listening
            
            # Apply gradient overlay
            cv2.addWeighted(gradient, 0.6, frame[:60], 0.4, 0, frame[:60])
            
            # Add a small pulsing circle animation to indicate listening
            listen_x = 30
            listen_y = 30
            cv2.circle(frame, (listen_x, listen_y), 10, (0, 255, 128), -1)
            cv2.circle(frame, (listen_x, listen_y), 13, (0, 200, 100), 1)
            
            cv2.putText(frame, "Listening... (press L to stop)", 
                       (listen_x + 20, listen_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 255, 255), 
                       1, 
                       cv2.LINE_AA)
            cv2.imshow('Smart Mirror', frame)
            cv2.waitKey(1)
        
        # Initialize variables for listening
        response_text = ""
        start_time = time.time()
        max_listen_time = 30  # Maximum time to listen (seconds)
        
        try:
            with self.mic as source:
                # Adjust for ambient noise
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print(f"Energy threshold: {self.recognizer.energy_threshold}")
                
                # Lower threshold for better detection
                self.recognizer.energy_threshold = 300
                print(f"Lowered energy threshold to: {self.recognizer.energy_threshold}")
                
                # Main listening loop
                while time.time() - start_time < max_listen_time:
                    elapsed_time = time.time() - start_time
                    
                    # Update UI with elapsed time
                    if ret:
                        new_ret, new_frame = self.cap.read()
                        if new_ret and new_frame is not None:
                            frame = cv2.flip(new_frame, 1)
                            
                            # Create listening UI with animation
                            gradient = np.zeros((60, width, 3), dtype=np.uint8)
                            for i in range(60):
                                alpha = 0.7 - (i / 60) * 0.5
                                gradient[i, :] = (0, 100, 50)  # Dark teal background for listening
                            
                            # Apply gradient overlay
                            cv2.addWeighted(gradient, 0.6, frame[:60], 0.4, 0, frame[:60])
                            
                            # Animated circle - pulse based on time
                            circle_x = 30
                            circle_y = 30
                            pulse_size = 10 + int(abs(np.sin(time.time() * 3)) * 5)
                            cv2.circle(frame, (circle_x, circle_y), pulse_size, (0, 255, 128), -1)
                            
                            cv2.putText(frame, f"Listening... ({int(elapsed_time)}s) (press L to stop)", 
                                       (circle_x + 20, circle_y + 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, 
                                       (255, 255, 255), 
                                       1, 
                                       cv2.LINE_AA)
                            
                            # Show energy level
                            energy_level = min(int(self.recognizer.energy_threshold), 1000)
                            energy_bar_width = int((energy_level / 1000) * 200)
                            cv2.rectangle(frame, (60, 50), (60 + energy_bar_width, 55), (0, 255, 128), -1)
                            
                            cv2.imshow('Smart Mirror', frame)
                    
                    # Check for key press to stop listening
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('l'):  # L pressed to stop listening
                        print("Listening stopped by user")
                        break
                    
                    try:
                        print("Listening for speech...")
                        # Use a shorter timeout to keep the UI responsive
                        audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=10)
                        
                        # Show processing UI
                        if ret:
                            new_ret, new_frame = self.cap.read()
                            if new_ret and new_frame is not None:
                                frame = cv2.flip(new_frame, 1)
                                
                                # Create processing UI
                                gradient = np.zeros((60, width, 3), dtype=np.uint8)
                                for i in range(60):
                                    alpha = 0.7 - (i / 60) * 0.5
                                    gradient[i, :] = (50, 0, 100)  # Dark blue for processing
                                
                                # Apply gradient overlay
                                cv2.addWeighted(gradient, 0.6, frame[:60], 0.4, 0, frame[:60])
                                
                                cv2.putText(frame, "Processing speech...", 
                                           (30, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.5, 
                                           (255, 255, 255), 
                                           1, 
                                           cv2.LINE_AA)
                                
                                cv2.imshow('Smart Mirror', frame)
                                cv2.waitKey(1)
                        
                        # Try multiple recognition methods
                        try:
                            # First try Google (online)
                            try:
                                print("Trying Google recognition...")
                                text = self.recognizer.recognize_google(audio, language='en-US')
                                print(f"Google recognized: {text}")
                                response_text = text
                                break  # Successfully got text, exit the loop
                            except speech_recognition.UnknownValueError:
                                print("Google could not understand audio")
                            except speech_recognition.RequestError as e:
                                print(f"Google request error: {e}")
                                
                                # Try Sphinx as fallback (offline)
                                try:
                                    print("Trying Sphinx recognition (fallback)...")
                                    # Check if we have pocketsphinx installed
                                    try:
                                        import pocketsphinx
                                        text = self.recognizer.recognize_sphinx(audio)
                                        print(f"Sphinx recognized: {text}")
                                        response_text = text
                                        break  # Successfully got text, exit the loop
                                    except ImportError:
                                        print("Pocketsphinx not installed. Cannot use offline recognition.")
                                except Exception as sphinx_error:
                                    print(f"Sphinx recognition error: {sphinx_error}")
                        except Exception as recog_error:
                            print(f"Recognition error: {recog_error}")
                            # Continue listening despite errors
                    
                    except speech_recognition.WaitTimeoutError:
                        print("No speech detected in this segment, continuing to listen...")
                        # Continue listening
            
            # After the loop, check if we got any response
            if response_text:
                print(f"Final response: {response_text}")
                return response_text
            else:
                print("No speech was recognized")
                
                # Show no speech recognized message
                if ret:
                    new_ret, new_frame = self.cap.read()
                    if new_ret and new_frame is not None:
                        frame = cv2.flip(new_frame, 1)
                        
                        # Create UI for no speech
                        gradient = np.zeros((60, width, 3), dtype=np.uint8)
                        for i in range(60):
                            alpha = 0.7 - (i / 60) * 0.5
                            gradient[i, :] = (0, 0, 100)  # Red background
                        
                        # Apply gradient overlay
                        cv2.addWeighted(gradient, 0.6, frame[:60], 0.4, 0, frame[:60])
                        
                        cv2.putText(frame, "No speech was recognized", 
                                   (30, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, 
                                   (255, 255, 255), 
                                   1, 
                                   cv2.LINE_AA)
                        
                        cv2.imshow('Smart Mirror', frame)
                        cv2.waitKey(2000)  # Show for 2 seconds
                
                return None
        
        except Exception as e:
            print(f"Error in listen_for_response: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error message
            if ret:
                self.show_error_screen(frame, f"Listening error: {e}")
            
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
        """Enhanced main loop with subtle UI for mirror effect and additional options"""
        print("Starting Smart Mirror...")
        last_frame_time = time.time()
        target_fps = 10  # Reduced FPS to reduce CPU usage
        
        # Flag to control when to process frames
        process_frame = True
        
        # Flag to track if we're currently listening
        currently_listening = False
        
        while True:
            # Calculate time since last frame to maintain target FPS
            current_time = time.time()
            elapsed = current_time - last_frame_time
            sleep_time = max(0, (1.0/target_fps) - elapsed)
            
            # Sleep to maintain target FPS
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            # Only process every other frame to reduce CPU usage
            process_frame = not process_frame
            if not process_frame:
                # Just check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in [ord('r'), ord('l'), ord('s'), ord('t')]:
                    process_frame = True  # Force processing on key press
                else:
                    continue
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera")
                time.sleep(0.5)  # Wait before trying again
                continue
                
            # Update frame time
            last_frame_time = time.time()
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Create a subtle status overlay for mirror effect
            height, width = frame.shape[:2]
            
            # Create a semi-transparent gradient overlay at the top - more subtle
            gradient = np.zeros((40, width, 3), dtype=np.uint8)
            for i in range(40):
                alpha = 0.7 - (i / 40) * 0.5  # Less opaque gradient
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
            
            # Handle key presses (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # R pressed - start new analysis
                # Analysis Phase - Subtle UI
                # Create a subtle gradient overlay
                gradient = np.zeros((60, width, 3), dtype=np.uint8)
                for i in range(60):
                    alpha = 0.7 - (i / 60) * 0.5  # Less opaque gradient
                    gradient[i, :] = (0, 80, 0)  # Dark green background
                
                # Apply gradient overlay
                cv2.addWeighted(gradient, 0.6, frame[:60], 0.4, 0, frame[:60])
                
                # Add a small loading animation
                loading_x = 30
                loading_y = 30
                cv2.circle(frame, (loading_x, loading_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (loading_x, loading_y), 13, (0, 200, 0), 1)
                
                cv2.putText(frame, "Analyzing...", 
                           (loading_x + 20, loading_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           (255, 255, 255), 
                           1, 
                           cv2.LINE_AA)
                cv2.imshow('Smart Mirror', frame)
                cv2.waitKey(1)
                
                print("\n[ACTION] Starting new analysis...")
                emotion, posture = self.analyze_initial_state(duration=3)
                
                # Results Phase - Subtle UI
                # Create a subtle gradient overlay
                gradient = np.zeros((80, width, 3), dtype=np.uint8)
                for i in range(80):
                    alpha = 0.7 - (i / 80) * 0.5  # Less opaque gradient
                    gradient[i, :] = (50, 0, 80)  # Purple background
                
                # Apply gradient overlay
                cv2.addWeighted(gradient, 0.6, frame[:80], 0.4, 0, frame[:80])
                
                # Add emotion icon based on detected emotion
                icon_x = 30
                icon_y = 30
                emotion_color = (100, 100, 255)  # Default blue
                
                if emotion == "happy":
                    emotion_color = (0, 255, 255)  # Yellow
                elif emotion == "sad":
                    emotion_color = (255, 0, 0)    # Blue
                elif emotion == "angry":
                    emotion_color = (0, 0, 255)    # Red
                elif emotion == "fear":
                    emotion_color = (255, 0, 255)  # Purple
                
                # Draw emotion icon
                cv2.circle(frame, (icon_x, icon_y), 10, emotion_color, -1)
                
                # Add results with smaller font
                cv2.putText(frame, f"Emotion: {emotion}", 
                           (icon_x + 20, icon_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           (255, 255, 255), 
                           1, 
                           cv2.LINE_AA)
                
                # Add posture icon
                posture_icon_y = icon_y + 25
                cv2.circle(frame, (icon_x, posture_icon_y), 10, (0, 200, 200), -1)
                
                cv2.putText(frame, f"Posture: {', '.join(posture)}", 
                           (icon_x + 20, posture_icon_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           (255, 255, 255), 
                           1, 
                           cv2.LINE_AA)
                
                # Add instruction with smaller font
                cv2.putText(frame, "Please speak when prompted...", 
                           (icon_x, posture_icon_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           (255, 255, 255), 
                           1, 
                           cv2.LINE_AA)
                cv2.imshow('Smart Mirror', frame)
                cv2.waitKey(1500)  # Show results for 1.5 seconds
                
                # Lock in results
                self.locked_emotion = emotion
                self.locked_body_language = posture
                
                # Get verbal response
                currently_listening = True
                response = self.listen_for_response()
                currently_listening = False
                
                if response:
                    # Generate and speak response
                    follow_up = self.generate_response(response, self.locked_emotion, self.locked_body_language)
                    self.speak_message(follow_up)
                
                self.ready_to_send = True
                
            elif key == ord('l'):  # L pressed - toggle listening
                if not currently_listening:
                    # Start listening without analysis
                    print("\n[ACTION] Listening without analysis...")
                    
                    # Listening Phase - Subtle UI
                    # Create a subtle gradient overlay
                    gradient = np.zeros((60, width, 3), dtype=np.uint8)
                    for i in range(60):
                        alpha = 0.7 - (i / 60) * 0.5  # Less opaque gradient
                        gradient[i, :] = (0, 100, 50)  # Teal background
                    
                    # Apply gradient overlay
                    cv2.addWeighted(gradient, 0.6, frame[:60], 0.4, 0, frame[:60])
                    
                    # Add a small pulsing circle animation to indicate listening
                    listen_x = 30
                    listen_y = 30
                    cv2.circle(frame, (listen_x, listen_y), 10, (0, 255, 128), -1)
                    cv2.circle(frame, (listen_x, listen_y), 13, (0, 200, 100), 1)
                    
                    cv2.putText(frame, "Starting to listen... (press L to stop)", 
                               (listen_x + 20, listen_y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, 
                               (255, 255, 255), 
                               1, 
                               cv2.LINE_AA)
                    cv2.imshow('Smart Mirror', frame)
                    
                    # Set listening flag
                    currently_listening = True
                    
                    # Use neutral emotion and empty posture for direct response
                    self.locked_emotion = "neutral"
                    self.locked_body_language = []
                    
                    # Get verbal response without analysis
                    response = self.listen_for_response()
                    currently_listening = False
                    
                    if response:
                        # Generate and speak response
                        follow_up = self.generate_response(response, self.locked_emotion, self.locked_body_language)
                        self.speak_message(follow_up)
                    
                    self.ready_to_send = True
                else:
                    # L pressed while already listening - this will be handled by the listen_for_response method
                    # The method will detect the key press and stop listening
                    pass
                
            elif key == ord('s'):  # S pressed - switch user
                print("\n[ACTION] Switching user...")
                
                # Create a subtle gradient overlay for user switching
                gradient = np.zeros((60, width, 3), dtype=np.uint8)
                for i in range(60):
                    alpha = 0.7 - (i / 60) * 0.5  # Less opaque gradient
                    gradient[i, :] = (80, 40, 0)  # Orange background
                
                # Apply gradient overlay
                cv2.addWeighted(gradient, 0.6, frame[:60], 0.4, 0, frame[:60])
                
                cv2.putText(frame, "Switching user...", 
                           (30, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           (255, 255, 255), 
                           1, 
                           cv2.LINE_AA)
                cv2.imshow('Smart Mirror', frame)
                cv2.waitKey(1000)
                
                # Temporarily release camera for user switching
                self.cap.release()
                cv2.destroyAllWindows()
                
                # Prompt for new user
                self.prompt_user_login()
                
                # Reinitialize camera
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                
                if not self.cap.isOpened():
                    print("Error: Could not reopen camera after user switch!")
                    break
            
            elif key == ord('t'):  # T pressed - test microphone
                print("\n[ACTION] Testing microphone...")
                self.test_microphone()
            
            # Display frame
            cv2.imshow('Smart Mirror', frame)
            
            if key == ord('q'):
                break
        
        # Clean up resources
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
        """Prompt user to login or create a new user"""
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
                    
                # Create new user
                self.firebase_manager.create_user(user_id)
                print(f"New user '{user_id}' created successfully!")
                self.login_user(user_id)
                break
            else:
                # Check if user exists
                if self.firebase_manager.check_user_exists(user_id):
                    print(f"User '{user_id}' found. Logging in...")
                    self.login_user(user_id)
                    break
                else:
                    print(f"User '{user_id}' not found. Please try again or create a new user.")
        
        print("\nLogin successful! Press any key to continue...")
        input()

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
                input_device_index=None,  # Use default input device
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

if __name__ == "__main__":
    mirror = SmartMirror()
    mirror.run()
