import cv2
import numpy as np
import time
from datetime import datetime
import threading
from queue import Queue
try:
    import speech_recognition as sr
except Exception:
    print("speech_recognition not available. Speech features will be limited.")
    sr = None
try:
    import mediapipe as mp
except Exception:
    print("mediapipe not available. Face detection will be disabled.")
    mp = None
from mirror import SmartMirror, client  # Import the client from mirror.py
from firebase_config import FirebaseManager
import os
import io
import base64
from PIL import Image
try:
    from deepface import DeepFace
except Exception:
    print("deepface not available. Some emotion features may be limited.")
    DeepFace = None
import wave
from typing import Optional, Tuple, Dict
import os

# Optional high-accuracy emotion model (PyTorch)
try:
    import torch
    # Try primary import path
    try:
        from hsemotion.facial_emotions import HSEmotionRecognizer  # primary path
        HSE_AVAILABLE = True
    except Exception:
        # Fallback to older import path
        try:
            from hsemotion.face_emotion import HSEmotionRecognizer  # older path
            HSE_AVAILABLE = True
        except Exception:
            HSE_AVAILABLE = False
except Exception:
    # Torch not available; disable HSEmotion usage
    HSE_AVAILABLE = False

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    print("PyAudio not available. Audio recording features will be disabled.")
    AUDIO_AVAILABLE = False

class WebMirror(SmartMirror):
    """Optimized version of SmartMirror for web interface with lower latency"""

    def __init__(self, user_id="guest", audio_disabled=False):
        """Initialize with web-specific optimizations"""
        # Call parent init with web mode
        super().__init__(web_mode=True)
        
        # Override user_id from parent
        self.user_id = user_id
        self.current_user = user_id
        
        # Initialize emotion and cognitive state attributes
        self.locked_emotion = "neutral"
        self.locked_body_language = ["relaxed"]
        self.last_risk_level = None
        self.last_risk_indicators = []
        self.last_is_emergency = False
        
        # API cooldown to prevent excessive calls
        self.last_api_call = 0
        self.api_cooldown = 1.0  # seconds
        
        # Caching attributes
        self.emotion_cache = {"neutral": 100}
        self.emotion_cache_time = 0
        self.emotion_cache_expiry = 3.0  # seconds
        
        self.posture_cache = "relaxed"
        self.posture_cache_time = 0
        self.posture_cache_expiry = 3.0  # seconds
        
        # Background processing queue
        self.bg_queue = Queue()
        
        # Initialize Firebase
        try:
            self.firebase_manager = FirebaseManager()
            print(f"[WebMirror] Firebase initialized for user: {user_id}")
        except Exception as e:
            print(f"[WebMirror] Firebase initialization error: {e}")
            self.firebase_manager = None
        
        # Initialize audio components if not disabled
        if not audio_disabled:
            self._initialize_audio()

        # Initialize fast face detector (MediaPipe) for robust ROI extraction
        try:
            self.mp_face_detection = mp.solutions.face_detection
            # model_selection=1 for better distance range (selfie/webcam)
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.6
            )
            print("[WebMirror] MediaPipe FaceDetection initialized")
        except Exception as face_init_err:
            print(f"[WebMirror] Failed to init MediaPipe FaceDetection: {face_init_err}")
            self.face_detector = None

        # Temporal emotion smoothing (exponential moving average)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.smoothed_emotion_scores: Dict[str, float] = {k: 0.0 for k in self.emotion_labels}
        self.emotion_smoothing_alpha: float = 0.5  # moderate smoothing
        
        # Start background processor thread
        self.bg_thread = threading.Thread(target=self._background_processor)
        self.bg_thread.daemon = True
        self.bg_thread.start()

        print("[WebMirror] Initialized with optimized settings for lower latency")

        # Initialize session management
        self.current_session = {
            'start_time': datetime.now(),
            'interactions': [],
            'last_input': '',
            'last_response': ''
        }

        # High-accuracy emotion model toggle
        self.high_accuracy_emotions: bool = True  # enable as requested
        self.hse_model = None
        if self.high_accuracy_emotions and HSE_AVAILABLE:
            try:
                device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
                self.hse_device = device
                # Models: 'enet_b0_8_best_vgaf', 'enet_b2_8', etc.
                self.hse_model = HSEmotionRecognizer(model_name='enet_b2_8', device=device)
                print(f"[WebMirror] HSEmotion loaded on {device}")
            except Exception as e:
                print(f"[WebMirror] Failed to load HSEmotion: {e}")
                self.hse_model = None

    def ensure_camera_initialized(self):
        """Make sure the camera is properly initialized"""
        try:
            if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
                print("[WebMirror] Initializing camera...")
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("[WebMirror] WARNING: Failed to open camera")
                else:
                    print("[WebMirror] Camera initialized successfully")
            return self.cap.isOpened()
        except Exception as e:
            print(f"[WebMirror] Error initializing camera: {e}")
            return False

    def _background_processor(self):
        """Process background tasks to avoid blocking the main thread"""
        while True:
            try:
                # Check if we should stop (queue is None during cleanup)
                if self.bg_queue is None:
                    print("[WebMirror] Background processor stopping")
                    break

                # Get task with timeout to avoid blocking indefinitely
                try:
                    task, args, kwargs, callback = self.bg_queue.get(timeout=0.5)
                except Exception as queue_error:
                    # Just a timeout, continue the loop
                    continue

                # Execute the task with comprehensive error handling
                try:
                    result = task(*args, **kwargs)
                    if callback:
                        callback(result)
                except Exception as task_error:
                    import traceback
                    error_msg = traceback.format_exc()
                    print(f"[WebMirror] Background task execution error: {task_error}")
                    print(f"[WebMirror] Task details: {task.__name__ if hasattr(task, '__name__') else 'unknown'}")
                    print(f"[WebMirror] Error traceback: {error_msg}")

                    # Don't let errors in one task stop the background processor
                    continue

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"[WebMirror] Critical background processor error: {e}")
                print(f"[WebMirror] Error traceback: {error_msg}")

                # Sleep briefly to avoid tight error loops consuming CPU
                time.sleep(1)

        print("[WebMirror] Background processor exited")

    def add_bg_task(self, task, args=None, kwargs=None, callback=None):
        """Add a task to run in background"""
        self.bg_queue.put((task, args or (), kwargs or {}, callback))

    def _extract_face_roi(self, rgb_frame: np.ndarray, expand: float = 0.25) -> Optional[np.ndarray]:
        """Detect the most confident face and return a cropped RGB ROI.

        Args:
            rgb_frame: Frame in RGB color space
            expand: Fractional expansion around the detected bounding box to include context

        Returns:
            Cropped face ROI in RGB, or None if no face is detected
        """
        try:
            if self.face_detector is None:
                return None

            results = self.face_detector.process(rgb_frame)
            if not results or not results.detections:
                return None

            # Choose detection with highest score
            best_det = max(results.detections, key=lambda d: (d.score[0] if d.score else 0))
            rel_box = best_det.location_data.relative_bounding_box
            h, w, _ = rgb_frame.shape

            # Compute expanded absolute box
            x_min = max(0, int((rel_box.xmin - expand) * w))
            y_min = max(0, int((rel_box.ymin - expand) * h))
            x_max = min(w, int((rel_box.xmin + rel_box.width + expand) * w))
            y_max = min(h, int((rel_box.ymin + rel_box.height + expand) * h))

            if x_max <= x_min or y_max <= y_min:
                return None

            face_roi = rgb_frame[y_min:y_max, x_min:x_max]
            if face_roi.size == 0:
                return None

            # Optionally resize to a reasonable size for DeepFace
            target_size = 256
            fh, fw = face_roi.shape[:2]
            scale = target_size / max(fh, fw)
            if scale != 1.0:
                new_w, new_h = int(fw * scale), int(fh * scale)
                face_roi = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

            return face_roi
        except Exception:
            return None

    def _smooth_emotions(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Apply EMA smoothing to emotion scores (0-100 scale)."""
        alpha = self.emotion_smoothing_alpha
        # Ensure all labels present
        for label in self.emotion_labels:
            latest = float(raw_scores.get(label, 0.0))
            prev = float(self.smoothed_emotion_scores.get(label, 0.0))
            self.smoothed_emotion_scores[label] = alpha * latest + (1 - alpha) * prev
        return dict(self.smoothed_emotion_scores)

    def _normalize_face_roi(self, roi_rgb: np.ndarray) -> np.ndarray:
        """Stabilize lighting using CLAHE on L-channel and mild gamma correction."""
        try:
            # Convert to LAB and apply CLAHE on L channel
            lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge((l_eq, a, b))
            rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

            # Gamma correction (slight boost)
            gamma = 1.1
            inv_gamma = 1.0 / gamma
            table = (np.linspace(0, 1, 256) ** inv_gamma * 255).astype(np.uint8)
            rgb_gamma = cv2.LUT(rgb_eq, table)
            return rgb_gamma
        except Exception:
            return roi_rgb

    def web_analyze_state(self, frame, duration=2):
        """Analyze state for web interface with emotion averaging
        
        Args:
            frame: The video frame to analyze
            duration: Duration in seconds to analyze (default: 2)
        """
        print("\nStarting web analysis...")
        try:
            if frame is None:
                return {'status': 'error', 'message': 'No frame provided'}

            # HSEmotion-only multi-frame analysis for accuracy
            if not (self.high_accuracy_emotions and self.hse_model is not None):
                # Graceful fallback to base implementation (DeepFace-based)
                return super().web_analyze_state(frame, duration=duration)

            # Collect a short window of frames (0.8s) for stability
            num_frames = 8
            sleep_between = 0.08
            accum: Dict[str, float] = {k: 0.0 for k in self.emotion_labels}

            last_rgb = None
            for i in range(num_frames):
                if i == 0:
                    bgr = frame
                else:
                    # Try to read a fresh frame from the camera if possible
                    if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                        ok, bgr = self.cap.read()
                        if not ok:
                            bgr = frame
                    else:
                        bgr = frame

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                last_rgb = rgb

                # Detect and crop face
                roi = self._extract_face_roi(rgb)
                if roi is None:
                    # Fallback to centered crop from last rgb
                    h, w, _ = rgb.shape
                    cw, ch = int(w * 0.5), int(h * 0.5)
                    x0, y0 = (w - cw) // 2, (h - ch) // 2
                    roi = rgb[y0:y0+ch, x0:x0+cw]

                roi = self._normalize_face_roi(roi)

                # HSEmotion predict (returns probabilities summing to 1)
                try:
                    pred = self.hse_model.predict_emotions(roi)
                except Exception:
                    pred = {k: 0.0 for k in self.emotion_labels}
                    pred['neutral'] = 1.0

                # Accumulate as 0-100 scale
                for k in self.emotion_labels:
                    accum[k] += float(pred.get(k, 0.0)) * 100.0

                if i < num_frames - 1:
                    time.sleep(sleep_between)

            # Average
            for k in accum:
                accum[k] /= float(num_frames)

            # Smooth over calls
            smoothed = self._smooth_emotions(accum)
            emotion, confidence = max(smoothed.items(), key=lambda x: x[1])

            # Low-confidence fallback
            if confidence < 30:
                emotion = 'neutral'
                confidence = 100.0

            # Store the emotion for response generation
            self.locked_emotion = emotion

            # Analyze posture with improved error handling (ignore viz frame)
            try:
                posture_result = self.analyze_posture(frame)
                if isinstance(posture_result, tuple):
                    posture_cues, _viz = posture_result
                else:
                    posture_cues = posture_result
                if not posture_cues:
                    posture_cues = ["relaxed"]
                self.locked_body_language = posture_cues
            except Exception as e:
                print(f"Posture analysis error: {e}")
                posture_cues = ["relaxed"]
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
                'posture': ["relaxed"]
            }

    def _update_posture_cache(self, result):
        """Update posture cache with background processing result"""
        try:
            if result:
                self.posture_cache = result
                self.posture_cache_time = time.time()
        except Exception as e:
            print(f"[WebMirror] Error updating posture cache: {e}")

    def _get_current_frame(self):
        """Get current frame from camera without blocking"""
        try:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Resize for faster processing
                    frame = cv2.resize(frame, (320, 240))
                    return frame
            return None
        except Exception as e:
            print(f"[WebMirror] Error getting current frame: {e}")
            return None

    def _detailed_posture_analysis(self, frame):
        """More detailed but slower posture analysis run in background"""
        try:
            if frame is None:
                return "unknown"

            # This would be a more detailed analysis
            # But for now we'll just return a placeholder
            return "relaxed"
        except Exception as e:
            print(f"[WebMirror] Error in detailed posture analysis: {e}")
            return "relaxed"  # Safe default

    def _initialize_audio(self):
        """Initialize audio components with PyAudio"""
        if not AUDIO_AVAILABLE:
            print("[WebMirror] PyAudio not available. Some features will be disabled.")
            return

        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            self.audio_format = pyaudio.paInt16
            self.channels = 1
            self.sample_rate = 16000
            self.chunk = 1024
            self.stream = None
            self.recording = False
            self.should_stop_recording = False
            self.audio_frames = []

            # Initialize speech recognizer 
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300  # Default value, will be adjusted

            # Find a working microphone
            self.input_device_index = self._find_working_microphone()

            print(f"[WebMirror] Audio initialized with device index: {self.input_device_index}")
        except Exception as e:
            print(f"[WebMirror] Error initializing audio: {e}")

    def _find_working_microphone(self):
        """Find the first working microphone"""
        if not hasattr(self, 'audio') or self.audio is None:
            return None

        try:
            # List all audio devices
            info = self.audio.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')

            for i in range(num_devices):
                device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    print(f"[WebMirror] Found input device: {device_info.get('name')}")
                    return i

            print("[WebMirror] No working microphone found")
            return None
        except Exception as e:
            print(f"[WebMirror] Error finding microphone: {e}")
            return None

    def start_recording(self):
        """Start recording audio with PyAudio"""
        if not AUDIO_AVAILABLE:
            print("[WebMirror] PyAudio not available, can't start recording")
            return False

        try:
            # Close any existing stream
            if self.stream is not None and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()

            # Create a new stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=self.input_device_index
            )

            # Reset recording state
            self.recording = True
            self.should_stop_recording = False
            self.audio_frames = []
            self.recording_start_time = time.time()

            print("[WebMirror] Started recording")
            return True
        except Exception as e:
            print(f"[WebMirror] Error starting recording: {e}")
            self.recording = False
            return False

    def stop_recording(self):
        """Stop recording and process the audio"""
        if not self.recording:
            print("[WebMirror] Not recording, nothing to stop")
            return None

        try:
            # Set flags
            self.recording = False
            self.should_stop_recording = True

            # Stop the stream
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            # Calculate duration
            duration = 0
            if self.recording_start_time:
                duration = time.time() - self.recording_start_time
                self.recording_start_time = None

            print(f"[WebMirror] Stopped recording after {duration:.2f} seconds")

            # Process the recorded audio if we have frames
            if self.audio_frames:
                # Save recorded audio to a temporary WAV file
                temp_file = "temp_recording.wav"
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(self.audio_frames))

                # Process with Whisper (from parent class)
                try:
                    transcript = self.whisper_transcribe(temp_file)
                    if transcript:
                        print(f"[WebMirror] Transcribed: {transcript}")
                        return transcript
                    else:
                        print("[WebMirror] No speech detected")
                except Exception as e:
                    print(f"[WebMirror] Error transcribing: {e}")

            return None
        except Exception as e:
            print(f"[WebMirror] Error stopping recording: {e}")
            return None

    def record_frames(self, seconds=10):
        """Record audio frames for the specified duration"""
        if not self.recording:
            print("[WebMirror] Not recording, call start_recording first")
            return []

        try:
            start_time = time.time()
            frames = []

            print(f"[WebMirror] Recording for {seconds} seconds...")
            while time.time() - start_time < seconds and not self.should_stop_recording:
                try:
                    data = self.stream.read(self.chunk)
                    frames.append(data)
                except Exception as e:
                    print(f"[WebMirror] Error reading audio data: {e}")
                    break

            print(f"[WebMirror] Recorded {len(frames)} frames")
            return frames
        except Exception as e:
            print(f"[WebMirror] Error recording frames: {e}")
            return []

    def whisper_listen(self):
        """Record audio and transcribe it using Whisper"""
        if not self.start_recording():
            print("[WebMirror] Failed to start recording")
            return None

        # Record for up to 10 seconds
        self.audio_frames = self.record_frames(seconds=10)

        # Stop recording and get transcript
        return self.stop_recording()

    def get_audio_level(self):
        """Get the current audio level (0-100)"""
        if not self.recording or not self.stream:
            return 0

        try:
            # Read a chunk of audio
            data = self.stream.read(self.chunk, exception_on_overflow=False)

            # Convert to numpy array
            audio_array = np.frombuffer(data, dtype=np.int16)

            # Calculate RMS
            rms = np.sqrt(np.mean(np.square(audio_array)))

            # Convert to percentage (0-100)
            audio_level = min(100, int((rms / 32767) * 100))

            return audio_level
        except Exception as e:
            print(f"[WebMirror] Error getting audio level: {e}")
            return 0

    def cleanup(self):
        """Properly cleanup resources"""
        try:
            print("[WebMirror] Cleaning up resources...")

            # Signal the background thread to stop
            self.bg_queue = None  # This will cause the thread to exit when it tries to use the queue

            # Release camera if it exists
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    self.cap.release()
                    print("[WebMirror] Camera released")
                except Exception as cam_err:
                    print(f"[WebMirror] Error releasing camera: {cam_err}")

            # Clean up audio resources if needed
            if hasattr(self, 'stream') and self.stream is not None:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                    print("[WebMirror] Audio stream closed")
                except Exception as audio_err:
                    print(f"[WebMirror] Error closing audio stream: {audio_err}")

            if hasattr(self, 'audio') and self.audio is not None:
                try:
                    self.audio.terminate()
                    print("[WebMirror] PyAudio terminated")
                except Exception as audio_err:
                    print(f"[WebMirror] Error terminating PyAudio: {audio_err}")

            print("[WebMirror] Cleanup completed")
        except Exception as e:
            print(f"[WebMirror] Error during cleanup: {e}")

        # Call parent cleanup if it exists
        try:
            super().cleanup()
        except:
            pass 

    def whisper_transcribe(self, audio_file):
        """Transcribe audio to text using Whisper"""
        with open(audio_file, "rb") as audio_data:
            transcription = client.audio.transcribe("whisper-1",
            audio_data)
        return transcription.text  # Return the text property directly

    def generate_response(self, text, emotion=None, posture=None):
        """Generate a response based on user input and current state"""
        try:
            # Update session's last input
            self.current_session['last_input'] = text
            
            # Generate response using existing logic
            response_text = super().generate_response(text, emotion, posture)
            
            # Update session's last response
            self.current_session['last_response'] = response_text
            
            # Add to session interactions
            interaction = {
                'timestamp': datetime.now(),
                'emotion': emotion or self.locked_emotion or "neutral",
                'posture': posture or self.locked_body_language or [],
                'user_input': text,
                'ai_output': response_text,
                'risk_level': self.last_risk_level,
                'risk_indicators': self.last_risk_indicators,
                'is_emergency': self.last_is_emergency
            }
            self.current_session['interactions'].append(interaction)
            
            return response_text
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your input."

    def web_generate_response(self, text):
        """Generate a response for web interface with proper risk assessment"""
        print("\n[WebMirror] Starting web_generate_response")
        
        # Get current emotion and posture from class attributes
        emotion = getattr(self, 'locked_emotion', 'neutral')
        body_language = getattr(self, 'locked_body_language', [])
        
        print(f"[WebMirror] Current emotion: {emotion}")
        print(f"[WebMirror] Current body language: {body_language}")
        
        # Call the parent SmartMirror's generate_response method to ensure proper risk assessment
        # This now returns just the conversational response text
        response_text = super().generate_response(text, emotion, body_language)
        
        # Store the risk level and indicators for the admin panel and logging
        risk_level = getattr(self, 'last_risk_level', 0)
        risk_indicators = getattr(self, 'last_risk_indicators', [])
        is_emergency = getattr(self, 'last_is_emergency', False)
        
        # Log interaction details for debugging
        print(f"\n[WebMirror] RESPONSE VALUES:")
        print(f"[WebMirror] Emotion: {emotion}")
        print(f"[WebMirror] Body Language: {body_language}")
        print(f"[WebMirror] Risk Level: {risk_level}%")
        print(f"[WebMirror] Risk Indicators: {', '.join(risk_indicators) if risk_indicators else 'None'}")
        print(f"[WebMirror] Emergency Status: {is_emergency}")
        print(f"[WebMirror] Response: {response_text[:100]}...")
        
        # Only return the response text (not the metadata) for user-facing interactions
        return response_text

    def web_listen(self):
        """Listen for speech in web mode and return the transcript"""
        try:
            # Start recording
            if not self.start_recording():
                print("[WebMirror] Failed to start recording")
                return None
                
            # Record for up to 10 seconds
            self.audio_frames = self.record_frames(seconds=10)
            
            # Stop recording and get transcript
            transcript = self.stop_recording()
            
            if transcript:
                print(f"[WebMirror] Successfully transcribed: {transcript}")
                return transcript
            else:
                print("[WebMirror] No speech detected")
                return None
                
        except Exception as e:
            print(f"[WebMirror] Error in web_listen: {e}")
            return None 