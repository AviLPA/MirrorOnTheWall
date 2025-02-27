import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

class FirebaseManager:
    def __init__(self):
        # Initialize Firebase with your service account
        try:
            # Try to use the app if already initialized
            try:
                firebase_admin.get_app()
                print("Firebase already initialized, using existing app")
                self.db = firestore.client()
            except ValueError:
                # Not initialized yet, so look for credentials file
                try:
                    # Try different possible paths for the credentials file
                    possible_paths = [
                        'serviceAccountKey.json',
                        'firebase-key.json',
                        '/Users/avicomputer/AI mirror take 2/serviceAccountKey.json',
                        '/Users/avicomputer/AI mirror take 2/firebase-key.json'
                    ]
                    
                    cred = None
                    for path in possible_paths:
                        try:
                            print(f"Trying credentials path: {path}")
                            cred = credentials.Certificate(path)
                            print(f"Found credentials at: {path}")
                            break
                        except Exception:
                            continue
                    
                    if cred is None:
                        # If no file found, try to continue without Firebase
                        print("No credentials file found. Continuing without Firebase.")
                        self.db = None
                        return
                        
                    # Initialize with found credentials
                    firebase_admin.initialize_app(cred)
                    print("Firebase initialized successfully")
                    self.db = firestore.client()
                except Exception as e:
                    print(f"Error initializing Firebase with credentials: {e}")
                    self.db = None
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
            self.db = None

    def save_session(self, user_id, emotion, posture_cues, feedback):
        """Save a session to Firebase"""
        if not self.db:
            return False
            
        try:
            session_data = {
                'timestamp': datetime.now(),
                'emotion': emotion,
                'posture': posture_cues,
                'feedback': feedback
            }
            
            # Add to user's sessions collection
            self.db.collection('users').document(user_id).collection('sessions').add(session_data)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    def get_user_history(self, user_id):
        """Get user's conversation history"""
        if not self.db:
            print("Firebase not initialized, cannot get user history")
            return []
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                return user_data.get('history', [])
            else:
                return []
        except Exception as e:
            print(f"Error getting user history: {e}")
            return []

    def analyze_progress(self, user_id):
        """Analyze user's emotional progress over time"""
        if not self.db:
            print("Firebase not initialized, cannot analyze progress")
            return None
        
        try:
            history = self.get_user_history(user_id)
            if not history:
                return None
            
            # Extract emotions from history
            emotions = [entry.get('emotion', 'neutral') for entry in history if 'emotion' in entry]
            
            # Get recent pattern (last 5 emotions)
            recent = emotions[-5:] if len(emotions) >= 5 else emotions
            
            return {
                'recent_pattern': recent,
                'total_interactions': len(history)
            }
        except Exception as e:
            print(f"Error analyzing progress: {e}")
            return None

    def save_interaction(self, user_id, interaction_data):
        """Save an interaction to the user's history"""
        if not self.db:
            print("Firebase not initialized, cannot save interaction")
            return False
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            
            # Convert datetime objects to strings for Firestore
            interaction_copy = interaction_data.copy()
            if 'timestamp' in interaction_copy:
                interaction_copy['timestamp'] = interaction_copy['timestamp'].isoformat()
            if 'session_id' in interaction_copy and isinstance(interaction_copy['session_id'], datetime):
                interaction_copy['session_id'] = interaction_copy['session_id'].isoformat()
            
            # Add to history array
            user_ref.update({
                'history': firestore.ArrayUnion([interaction_copy])
            })
            return True
        except Exception as e:
            print(f"Error saving interaction: {e}")
            return False

    def check_user_exists(self, user_id):
        """Check if a user exists in the database"""
        if not self.db:
            print("Firebase not initialized, cannot check user")
            return False
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            user_doc = user_ref.get()
            return user_doc.exists
        except Exception as e:
            print(f"Error checking if user exists: {e}")
            return False

    def create_user(self, user_id):
        """Create a new user in the database"""
        if not self.db:
            print("Firebase not initialized, cannot create user")
            return False
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            user_ref.set({
                'created_at': datetime.now(),
                'last_login': datetime.now(),
                'history': []
            })
            return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

    def update_last_login(self, user_id):
        """Update user's last login time"""
        if not self.db:
            print("Firebase not initialized, cannot update login time")
            return False
        
        try:
            user_ref = self.db.collection('users').document(user_id)
            user_ref.update({
                'last_login': datetime.now()
            })
            return True
        except Exception as e:
            print(f"Error updating last login: {e}")
            return False 