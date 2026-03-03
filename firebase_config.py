import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
import json

class FirebaseManager:
    def __init__(self):
        # Path to your service account key
        service_account_path = "/Users/avicomputer/AI mirror take 2/mirror-8ea7e-firebase-adminsdk-gpx8d-4029383394.json"
        
        # Check if the file exists
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(f"Firebase service account key not found at {service_account_path}")
        
        # Check if Firebase app is already initialized
        try:
            # Try to get the default app - if it exists, we'll use it
            self.app = firebase_admin.get_app()
            print("Using existing Firebase app")
        except ValueError:
            # If no default app exists, initialize a new one
            print("Initializing new Firebase app")
            cred = credentials.Certificate(service_account_path)
            self.app = firebase_admin.initialize_app(cred)
        
        # Get Firestore client
        self.db = firestore.client()
        print(f"Firebase initialized with database: {self.db._database_string}")

    def save_session(self, user_id, emotion, verbal_input, response, metadata=None):
        """Save a session to Firestore with structured data"""
        try:
            print(f"\n[Firebase] Saving session for user: {user_id}")
            
            # Create session data
            data = {
                'user_id': user_id,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'emotion': emotion if emotion else 'neutral',
                'posture': metadata.get('body_language', []) if metadata and 'body_language' in metadata else [],
                'user_input': verbal_input,
                'ai_output': response,
                'risk_level': metadata.get('risk_level', 0) if metadata else 0,
                'risk_indicators': metadata.get('risk_indicators', []) if metadata else [],
                'is_emergency': metadata.get('is_emergency', False) if metadata else False
            }
            
            # Save to sessions collection
            session_ref = self.db.collection('sessions').document()
            session_ref.set(data)
            
            # Also save to user's sessions subcollection
            user_session_ref = self.db.collection('users').document(user_id).collection('sessions').document()
            user_session_ref.set(data)
            
            # Update user's latest interaction
            user_ref = self.db.collection('users').document(user_id)
            user_ref.update({
                'latest_interaction': {
                    'timestamp': firestore.SERVER_TIMESTAMP,
                    'emotion': emotion if emotion else 'neutral',
                    'risk_level': metadata.get('risk_level', 0) if metadata else 0
                },
                'total_sessions': firestore.Increment(1)
            })
            
            print(f"[Firebase] Successfully saved session")
            return True
            
        except Exception as e:
            print(f"\n[Firebase ERROR] Error saving session: {str(e)}")
            import traceback
            print(f"[Firebase ERROR] Traceback: {traceback.format_exc()}")
            return False

    def get_formatted_interactions(self, user_id, limit=50):
        """Get formatted interaction data for a user"""
        try:
            # Query interactions collection
            interactions = (self.db.collection('interactions')
                          .where('user_id', '==', user_id)
                          .order_by('timestamp', direction=firestore.Query.DESCENDING)
                          .limit(limit)
                          .stream())
            
            formatted_data = []
            for interaction in interactions:
                data = interaction.to_dict()
                formatted_data.append({
                    'timestamp': data.get('timestamp'),
                    'emotion': data.get('emotion', 'neutral'),
                    'posture': data.get('posture', []),
                    'user_input': data.get('user_input', ''),
                    'ai_output': data.get('ai_output', ''),
                    'risk_level': data.get('risk_level', 0)
                })
            
            return formatted_data
        except Exception as e:
            print(f"Error getting formatted interactions: {e}")
            return []

    def get_interaction_stats(self, user_id):
        """Get statistics about user interactions"""
        try:
            interactions = self.get_formatted_interactions(user_id)
            
            if not interactions:
                return {
                    'total_interactions': 0,
                    'average_risk': 0,
                    'common_emotions': [],
                    'latest_interaction': None
                }
            
            # Calculate statistics
            total = len(interactions)
            risk_levels = [i['risk_level'] for i in interactions if i['risk_level'] is not None]
            emotions = [i['emotion'] for i in interactions if i['emotion']]
            
            from collections import Counter
            emotion_counter = Counter(emotions)
            
            return {
                'total_interactions': total,
                'average_risk': sum(risk_levels) / len(risk_levels) if risk_levels else 0,
                'common_emotions': [{'emotion': e, 'count': c} for e, c in emotion_counter.most_common(3)],
                'latest_interaction': interactions[0] if interactions else None
            }
        except Exception as e:
            print(f"Error getting interaction stats: {e}")
            return None

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
        """Analyze user progress from past sessions"""
        try:
            # Get last 5 sessions from user's sessions subcollection
            sessions = (self.db.collection('users')
                       .document(user_id)
                       .collection('sessions')
                       .order_by('timestamp', direction=firestore.Query.DESCENDING)
                       .limit(5)
                       .stream())
            
            # Process data
            emotions = []
            for session in sessions:
                session_data = session.to_dict()
                if 'emotion' in session_data:
                    emotions.append(session_data['emotion'])
            
            # Return analysis
            return {
                'recent_pattern': emotions,
                'count': len(emotions)
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

    def get_user_posture_history(self, user_id, limit=50):
        """Retrieve user's posture history from Firebase"""
        try:
            if not self.db:
                print("Firebase not initialized")
                return []
            
            # Get posture history from Firestore
            posture_ref = self.db.collection('users').document(user_id).collection('posture_history')
            query = posture_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            
            posture_docs = query.get()
            posture_history = []
            
            for doc in posture_docs:
                data = doc.to_dict()
                posture_history.append(data)
            
            return posture_history
        
        except Exception as e:
            print(f"Error retrieving user posture history: {e}")
            return []

    def get_user_emotion_history(self, user_id, limit=50):
        """Retrieve user's emotion history from Firebase"""
        try:
            if not self.db:
                print("Firebase not initialized")
                return []
            
            # Get emotion history from Firestore
            emotion_ref = self.db.collection('users').document(user_id).collection('emotion_history')
            query = emotion_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            
            emotion_docs = query.get()
            emotion_history = []
            
            for doc in emotion_docs:
                data = doc.to_dict()
                emotion_history.append(data)
            
            return emotion_history
        
        except Exception as e:
            print(f"Error retrieving user emotion history: {e}")
            return []

    def store_user_posture_data(self, user_id, posture_data):
        """Store user's posture data in Firebase"""
        try:
            if not self.db:
                print("Firebase not initialized")
                return
            
            # Store posture data in Firestore
            posture_ref = self.db.collection('users').document(user_id).collection('posture_history')
            posture_ref.add(posture_data)
            
            # Also update the user's latest posture in their profile
            user_ref = self.db.collection('users').document(user_id)
            user_ref.update({
                'latest_posture': {
                    'timestamp': posture_data['timestamp'],
                    'mode': posture_data['mode'],
                    'cues': posture_data['cues']
                }
            })
        
        except Exception as e:
            print(f"Error storing user posture data: {e}")

    def get_user_session_history(self, user_id, limit=10):
        """Retrieve recent session history for a user"""
        try:
            # Query the user's sessions subcollection
            sessions = (self.db.collection('users')
                       .document(user_id)
                       .collection('sessions')
                       .order_by('timestamp', direction=firestore.Query.DESCENDING)
                       .limit(limit)
                       .stream())
            
            return [session.to_dict() for session in sessions]
        except Exception as e:
            print(f"Error retrieving session history: {e}")
            return []

    def analyze_user_progress(self, user_id):
        """Analyze user's emotional and risk level trends"""
        try:
            sessions = self.get_user_session_history(user_id, limit=30)  # Last 30 sessions
            
            if not sessions:
                return {
                    'trend': 'insufficient_data',
                    'avg_risk_level': None,
                    'common_emotions': [],
                    'session_count': 0
                }

            # Calculate trends
            risk_levels = [s.get('risk_level') for s in sessions if s.get('risk_level') is not None]
            emotions = [s.get('emotion') for s in sessions if s.get('emotion')]
            
            analysis = {
                'session_count': len(sessions),
                'avg_risk_level': sum(risk_levels) / len(risk_levels) if risk_levels else None,
                'common_emotions': self._get_common_emotions(emotions),
                'trend': self._calculate_trend(risk_levels)
            }
            
            return analysis
        except Exception as e:
            print(f"Error analyzing user progress: {e}")
            return None

    def _get_common_emotions(self, emotions):
        """Helper to get most common emotions"""
        from collections import Counter
        if not emotions:
            return []
        counter = Counter(emotions)
        return [emotion for emotion, _ in counter.most_common(3)]

    def _calculate_trend(self, risk_levels):
        """Calculate trend in risk levels"""
        if len(risk_levels) < 2:
            return 'insufficient_data'
        
        first_half = sum(risk_levels[:len(risk_levels)//2]) / (len(risk_levels)//2)
        second_half = sum(risk_levels[len(risk_levels)//2:]) / (len(risk_levels)//2)
        
        diff = second_half - first_half
        if diff < -5:
            return 'improving'
        elif diff > 5:
            return 'declining'
        return 'stable'

class SystemMemoryManager:
    """Manages system memory including session summaries, progress tracking, and proactive question generation"""
    
    def __init__(self, firebase_manager):
        self.firebase_manager = firebase_manager
        self.openai_client = None
        self.memory_cache = {}  # Cache for frequently accessed summaries
        
        # Initialize OpenAI client
        try:
            from config import OPENAI_API_KEY
            from openai import OpenAI
            import os
            # Normalize env var and initialize client consistently
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI for SystemMemoryManager: {e}")
    
    def generate_session_start_summary(self, user_id):
        """Generate a comprehensive summary at the start of each interaction"""
        try:
            print(f"\n[SystemMemory] Generating session start summary for user: {user_id}")
            
            # Get user's interaction history
            recent_sessions = self.firebase_manager.get_user_session_history(user_id, limit=10)
            progress_analysis = self.firebase_manager.analyze_user_progress(user_id)
            
            # Check if we have cached summary that's still relevant
            cache_key = f"{user_id}_summary"
            if cache_key in self.memory_cache:
                cached_summary = self.memory_cache[cache_key]
                if self._is_cache_valid(cached_summary, recent_sessions):
                    print("[SystemMemory] Using cached summary")
                    return cached_summary
            
            # Generate new summary if no cache or cache is stale
            summary = self._create_comprehensive_summary(user_id, recent_sessions, progress_analysis)
            
            # Cache the summary
            self.memory_cache[cache_key] = {
                'summary': summary,
                'timestamp': datetime.now(),
                'session_count': len(recent_sessions)
            }
            
            print(f"[SystemMemory] Generated new session summary")
            return summary
            
        except Exception as e:
            print(f"[SystemMemory] Error generating session summary: {e}")
            return self._create_fallback_summary(user_id)
    
    def _create_comprehensive_summary(self, user_id, recent_sessions, progress_analysis):
        """Create a comprehensive summary using AI analysis"""
        if not self.openai_client or not recent_sessions:
            return self._create_fallback_summary(user_id)
        
        try:
            # Prepare session data for analysis
            session_data = []
            for session in recent_sessions[-5:]:  # Last 5 sessions
                session_info = {
                    'date': session.get('timestamp', '').strftime('%Y-%m-%d') if session.get('timestamp') else 'Unknown',
                    'emotion': session.get('emotion', 'neutral'),
                    'risk_level': session.get('risk_level', 0),
                    'user_input_summary': session.get('user_input', '')[:100] + '...' if len(session.get('user_input', '')) > 100 else session.get('user_input', ''),
                    'key_topics': self._extract_key_topics(session.get('user_input', ''))
                }
                session_data.append(session_info)
            
            # Create prompt for AI analysis
            summary_prompt = f"""
            Analyze the recent interaction history for user {user_id} and create a comprehensive system memory summary.
            
            Recent Sessions Data:
            {json.dumps(session_data, indent=2, default=str)}
            
            Progress Analysis:
            {json.dumps(progress_analysis, indent=2, default=str) if progress_analysis else 'No progress data available'}
            
            Generate a JSON response with the following structure:
            {{
                "user_profile": {{
                    "name": "{user_id}",
                    "interaction_frequency": "description of how often they interact",
                    "primary_concerns": ["list of main topics/concerns they discuss"],
                    "emotional_patterns": "description of emotional trends",
                    "progress_indicators": "description of positive/negative changes"
                }},
                "recent_context": {{
                    "last_session_summary": "brief summary of most recent interaction",
                    "ongoing_topics": ["topics that span multiple sessions"],
                    "mood_trend": "improving/declining/stable with explanation",
                    "risk_trend": "description of risk level changes"
                }},
                "proactive_opportunities": {{
                    "check_in_topics": ["things to ask about from previous sessions"],
                    "suggested_questions": ["specific questions to ask proactively"],
                    "milestone_acknowledgments": ["achievements or progress to acknowledge"],
                    "concern_follow_ups": ["areas that need gentle follow-up"]
                }},
                "system_notes": {{
                    "interaction_style": "preferred communication approach",
                    "triggers_to_avoid": ["topics or phrases that caused negative reactions"],
                    "effective_strategies": ["approaches that worked well"],
                    "next_session_goals": ["what to focus on in upcoming interactions"]
                }}
            }}
            
            Guidelines:
            - Be specific and actionable
            - Focus on patterns across sessions, not individual incidents
            - Identify opportunities for meaningful follow-up
            - Note any concerning trends that need attention
            - Highlight positive progress to acknowledge
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Parse the JSON response
            summary_text = response.choices[0].message.content
            summary_json = json.loads(summary_text)
            
            # Store the full summary in Firebase for persistence
            self._store_summary_in_firebase(user_id, summary_json)
            
            return summary_json
            
        except Exception as e:
            print(f"[SystemMemory] Error creating AI summary: {e}")
            return self._create_fallback_summary(user_id)
    
    def _create_fallback_summary(self, user_id):
        """Create a basic summary when AI analysis fails"""
        return {
            "user_profile": {
                "name": user_id,
                "interaction_frequency": "Regular user",
                "primary_concerns": ["General wellbeing"],
                "emotional_patterns": "Monitoring emotional state",
                "progress_indicators": "Tracking progress over time"
            },
            "recent_context": {
                "last_session_summary": "Previous interaction completed",
                "ongoing_topics": ["Continuing support"],
                "mood_trend": "Monitoring for changes",
                "risk_trend": "Baseline assessment"
            },
            "proactive_opportunities": {
                "check_in_topics": ["How are you feeling today?"],
                "suggested_questions": ["What's on your mind?"],
                "milestone_acknowledgments": [],
                "concern_follow_ups": []
            },
            "system_notes": {
                "interaction_style": "Supportive and empathetic",
                "triggers_to_avoid": [],
                "effective_strategies": ["Active listening", "Validation"],
                "next_session_goals": ["Assess current state", "Provide support"]
            }
        }
    
    def _extract_key_topics(self, text):
        """Extract key topics from user input using simple keyword analysis"""
        if not text:
            return []
        
        # Simple keyword-based topic extraction
        topic_keywords = {
            'work': ['work', 'job', 'career', 'boss', 'colleague', 'office'],
            'relationships': ['relationship', 'partner', 'friend', 'family', 'love'],
            'anxiety': ['anxious', 'worry', 'nervous', 'panic', 'stress'],
            'depression': ['sad', 'depressed', 'down', 'hopeless', 'empty'],
            'sleep': ['sleep', 'tired', 'insomnia', 'rest', 'exhausted'],
            'health': ['health', 'sick', 'pain', 'doctor', 'medical']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics[:3]  # Return top 3 topics
    
    def _is_cache_valid(self, cached_summary, recent_sessions):
        """Check if cached summary is still valid"""
        if not cached_summary:
            return False
        
        # Cache is valid for 1 hour OR if no new sessions
        cache_age = datetime.now() - cached_summary['timestamp']
        if cache_age.total_seconds() > 3600:  # 1 hour
            return False
        
        # Check if new sessions have been added
        if len(recent_sessions) > cached_summary['session_count']:
            return False
        
        return True
    
    def _store_summary_in_firebase(self, user_id, summary):
        """Store the summary in Firebase for persistence"""
        try:
            summary_data = {
                'user_id': user_id,
                'summary': summary,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'type': 'session_start_summary'
            }
            
            # Store in user's memory collection
            self.firebase_manager.db.collection('users').document(user_id).collection('memory').add(summary_data)
            
        except Exception as e:
            print(f"[SystemMemory] Error storing summary in Firebase: {e}")
    
    def generate_proactive_questions(self, user_id, current_emotion=None, current_context=None):
        """Generate proactive questions based on user history and current state"""
        try:
            summary = self.generate_session_start_summary(user_id)
            
            # Extract proactive opportunities
            proactive_ops = summary.get('proactive_opportunities', {})
            check_in_topics = proactive_ops.get('check_in_topics', [])
            concern_follow_ups = proactive_ops.get('concern_follow_ups', [])
            
            # Generate contextual questions
            questions = []
            
            # Add check-in questions
            if check_in_topics:
                questions.extend(check_in_topics[:2])  # Limit to 2 questions
            
            # Add concern follow-ups if appropriate
            if concern_follow_ups and current_emotion != 'happy':
                questions.extend(concern_follow_ups[:1])  # Limit to 1 concern
            
            # Add milestone acknowledgments as positive reinforcement
            milestones = proactive_ops.get('milestone_acknowledgments', [])
            if milestones and current_emotion in ['happy', 'neutral']:
                questions.extend([f"I wanted to acknowledge: {milestone}" for milestone in milestones[:1]])
            
            return questions[:3]  # Return maximum 3 proactive items
            
        except Exception as e:
            print(f"[SystemMemory] Error generating proactive questions: {e}")
            return []
    
    def update_interaction_memory(self, user_id, interaction_data):
        """Update system memory with new interaction data"""
        try:
            # Invalidate cache for this user
            cache_key = f"{user_id}_summary"
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
            
            # Store interaction patterns for future analysis
            memory_update = {
                'user_id': user_id,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'interaction_summary': {
                    'emotion': interaction_data.get('emotion'),
                    'risk_level': interaction_data.get('risk_level'),
                    'key_topics': self._extract_key_topics(interaction_data.get('user_input', '')),
                    'response_effectiveness': 'pending'  # Could be updated based on user feedback
                },
                'type': 'interaction_update'
            }
            
            # Store in user's memory collection
            self.firebase_manager.db.collection('users').document(user_id).collection('memory').add(memory_update)
            
        except Exception as e:
            print(f"[SystemMemory] Error updating interaction memory: {e}")
    
    def get_progress_insights(self, user_id):
        """Generate insights about user progress over time"""
        try:
            # Get historical data
            sessions = self.firebase_manager.get_user_session_history(user_id, limit=20)
            
            if len(sessions) < 3:
                return "Not enough interaction history to generate meaningful insights."
            
            # Analyze trends
            recent_sessions = sessions[:5]
            older_sessions = sessions[5:10] if len(sessions) > 5 else []
            
            # Calculate emotion and risk trends
            recent_emotions = [s.get('emotion', 'neutral') for s in recent_sessions]
            recent_risks = [s.get('risk_level', 0) for s in recent_sessions if s.get('risk_level') is not None]
            
            if older_sessions:
                older_emotions = [s.get('emotion', 'neutral') for s in older_sessions]
                older_risks = [s.get('risk_level', 0) for s in older_sessions if s.get('risk_level') is not None]
            else:
                older_emotions = []
                older_risks = []
            
            # Generate insights
            insights = []
            
            # Risk level trend
            if recent_risks and older_risks:
                recent_avg_risk = sum(recent_risks) / len(recent_risks)
                older_avg_risk = sum(older_risks) / len(older_risks)
                
                if recent_avg_risk < older_avg_risk - 5:
                    insights.append("Your overall stress and risk levels have decreased compared to earlier sessions.")
                elif recent_avg_risk > older_avg_risk + 5:
                    insights.append("I've noticed some increased stress in recent sessions. This is something we should keep an eye on.")
                else:
                    insights.append("Your stress levels have remained relatively stable.")
            
            # Emotion patterns
            positive_emotions = ['happy', 'neutral', 'surprised']
            recent_positive = sum(1 for e in recent_emotions if e in positive_emotions)
            
            if recent_positive >= len(recent_emotions) * 0.6:
                insights.append("You've been showing more positive emotional states in recent interactions.")
            elif recent_positive <= len(recent_emotions) * 0.3:
                insights.append("I notice you've been experiencing some challenging emotions lately.")
            
            return " ".join(insights) if insights else "Continue focusing on your wellbeing - every interaction is a step forward."
            
        except Exception as e:
            print(f"[SystemMemory] Error generating progress insights: {e}")
            return "I'm continuing to monitor your progress and am here to support you." 