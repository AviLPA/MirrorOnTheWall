from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import time
import os
import json
import sys

# Add the parent directory to the path so we can import from the mirror module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from main mirror application
try:
    from firebase_config import FirebaseManager
except ImportError:
    print("Warning: Could not import FirebaseManager. Some features may be disabled.")
    FirebaseManager = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mirror-admin-secret-key'
socketio = SocketIO(app)

# Global variables to store shared data
risk_alerts = []
latest_interactions = []
admin_settings = {
    'risk_threshold': 70,  # Alert threshold for risk percentage
    'notify_email': '',    # Email for notifications
    'prompt_customizations': {},
    'response_guidelines': "Keep responses positive and supportive. Always suggest professional help for serious concerns."
}

# Initialize Firebase if available
firebase_manager = None
if FirebaseManager:
    try:
        firebase_manager = FirebaseManager()
        print("Admin panel: Firebase initialized successfully")
    except Exception as e:
        print(f"Admin panel: Error initializing Firebase: {e}")

def load_settings():
    """Load settings from storage"""
    global admin_settings
    try:
        if os.path.exists('admin_settings.json'):
            with open('admin_settings.json', 'r') as f:
                saved_settings = json.load(f)
                admin_settings.update(saved_settings)
    except Exception as e:
        print(f"Error loading settings: {e}")

def save_settings():
    """Save settings to storage"""
    try:
        with open('admin_settings.json', 'w') as f:
            json.dump(admin_settings, f)
    except Exception as e:
        print(f"Error saving settings: {e}")

# Load settings at startup
load_settings()

# Routes
@app.route('/')
def dashboard():
    """Main dashboard view"""
    return render_template('dashboard.html', 
                          alerts=risk_alerts, 
                          interactions=latest_interactions,
                          settings=admin_settings)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Settings management"""
    global admin_settings
    
    if request.method == 'POST':
        # Update settings from form
        admin_settings['risk_threshold'] = int(request.form.get('risk_threshold', 70))
        admin_settings['notify_email'] = request.form.get('notify_email', '')
        admin_settings['response_guidelines'] = request.form.get('response_guidelines', '')
        
        # Process prompt customizations
        prompt_sections = request.form.getlist('prompt_section')
        prompt_texts = request.form.getlist('prompt_text')
        
        admin_settings['prompt_customizations'] = {}
        for i in range(len(prompt_sections)):
            if i < len(prompt_texts):
                admin_settings['prompt_customizations'][prompt_sections[i]] = prompt_texts[i]
        
        # Save to storage
        save_settings()
        return redirect(url_for('settings', saved=True))
    
    return render_template('settings.html', 
                          settings=admin_settings,
                          saved='saved' in request.args)

@app.route('/users')
def users_page():
    """Render the users page with enhanced risk information"""
    if firebase_manager:
        try:
            # Get all users
            users_data = firebase_manager.get_all_users()
            
            # Get current risk threshold
            risk_threshold = admin_settings.get('risk_threshold', 70)
            
            # Process user data to include risk information
            for user in users_data:
                # Check if user has sessions
                if user.get('sessions'):
                    # Sort sessions by timestamp (newest first)
                    user['sessions'] = sorted(
                        user['sessions'], 
                        key=lambda x: x.get('timestamp', ''), 
                        reverse=True
                    )
                    
                    # Filter sessions to only include those above the risk threshold
                    filtered_sessions = []
                    for session in user['sessions']:
                        session_risk = session.get('risk_level', 0)
                        is_emergency = session.get('is_emergency', False)
                        
                        # Only include sessions above threshold or marked as emergency
                        if session_risk >= risk_threshold or is_emergency:
                            # Add default escalation reason and approach if not present
                            if 'escalation_reason' not in session:
                                session['escalation_reason'] = generate_escalation_explanation(
                                    session.get('text', ''),
                                    session.get('indicators', []),
                                    session.get('emotion', '')
                                )
                            
                            if 'approach' not in session:
                                session['approach'] = generate_approach_guidance(
                                    session_risk,
                                    is_emergency
                                )
                                
                            # Mark extreme risk sessions (90%+)
                            session['extreme_risk'] = session_risk >= 90
                            
                            filtered_sessions.append(session)
                    
                    # Replace with filtered sessions
                    user['sessions'] = filtered_sessions
                    
                    # Flag user as high risk if any recent session has high risk
                    user['high_risk'] = any(
                        session.get('risk_level', 0) >= 70 or session.get('is_emergency', False)
                        for session in user['sessions'][:5] if user['sessions']  # Check only the 5 most recent sessions
                    )
                    
                    # Flag user as extreme risk if any recent session has extreme risk
                    user['extreme_risk'] = any(
                        session.get('risk_level', 0) >= 90 or session.get('is_emergency', False)
                        for session in user['sessions'][:5] if user['sessions']  # Check only the 5 most recent sessions
                    )
                else:
                    user['high_risk'] = False
                    user['extreme_risk'] = False
            
            return render_template('users.html', users=users_data, risk_threshold=risk_threshold)
        except Exception as e:
            print(f"Error loading users: {e}")
            return render_template('users.html', users=[], error=str(e), risk_threshold=risk_threshold)
    else:
        return render_template('users.html', users=[], error="Firebase not configured", risk_threshold=70)

# API endpoints
@app.route('/api/risk-alert', methods=['POST'])
def risk_alert():
    """Receive risk alerts from the main application"""
    global risk_alerts  # Make sure we're using the global list
    data = request.json
    
    # Add timestamp if not present
    if 'timestamp' not in data:
        data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Get the risk level
    risk_level = data.get('risk_level', 0)
    is_emergency = data.get('is_emergency', False)
    
    # Check if this alert meets the threshold for reporting
    if risk_level < admin_settings['risk_threshold'] and not is_emergency:
        print(f"[ALERT] Ignoring alert with risk level {risk_level}% (below threshold {admin_settings['risk_threshold']}%)")
        return jsonify({'status': 'ignored', 'message': 'Below threshold'})
    
    # Generate AI explanation for escalation if not provided
    if 'escalation_reason' not in data and data.get('text'):
        try:
            # Generate an explanation based on the text and indicators
            indicators = data.get('indicators', [])
            text = data.get('text', '')
            emotion = data.get('emotion', 'unknown')
            
            if indicators or text:
                escalation_reason = generate_escalation_explanation(text, indicators, emotion)
                data['escalation_reason'] = escalation_reason
        except Exception as e:
            print(f"Error generating escalation reason: {e}")
            # Provide a default explanation
            data['escalation_reason'] = "The user's input contains concerning language that may indicate risk."
    
    # Generate recommended approach if not provided
    if 'approach' not in data:
        try:
            approach = generate_approach_guidance(risk_level, is_emergency)
            data['approach'] = approach
        except Exception as e:
            print(f"Error generating approach guidance: {e}")
            # Provide a default approach
            data['approach'] = "Respond with empathy and suggest professional support resources."
    
    # Add user_input field for dashboard display
    if 'user_input' not in data and 'text' in data:
        data['user_input'] = data['text']
    
    # Mark as extreme risk if 90% or higher
    data['extreme_risk'] = risk_level >= 90
    
    # Add to alerts list
    risk_alerts.insert(0, data)
    print(f"[ALERT] Added new alert to risk_alerts list. Total alerts: {len(risk_alerts)}")
    
    # Keep only the most recent 50 alerts
    if len(risk_alerts) > 50:
        risk_alerts.pop()
    
    # Emit to all connected clients with special flag for extreme risk
    socketio.emit('new_alert', data)
    print(f"[ALERT] Emitted new_alert via Socket.IO")
    
    # For extreme risk (90-100%), emit a special alert
    if risk_level >= 90 or is_emergency:
        socketio.emit('extreme_risk_alert', {
            'risk_level': risk_level,
            'user_id': data.get('user_id', 'Unknown'),
            'timestamp': data.get('timestamp'),
            'is_emergency': is_emergency
        })
        print(f"[ALERT] Emitted extreme_risk_alert via Socket.IO")
    
    # Send email notification if configured
    if admin_settings['notify_email']:
        try:
            send_email_notification(
                admin_settings['notify_email'],
                f"Smart Mirror - {'EXTREME RISK ALERT' if risk_level >= 90 or is_emergency else 'HIGH RISK ALERT'}",
                f"""
                {'⚠️ EXTREME RISK ALERT ⚠️' if risk_level >= 90 or is_emergency else 'HIGH RISK ALERT'}
                
                Risk Level: {risk_level}%
                User ID: {data.get('user_id', 'Unknown')}
                Emergency: {'YES' if is_emergency else 'No'}
                
                Escalation Reason:
                {data.get('escalation_reason', 'Not provided')}
                
                Indicators: {', '.join(data.get('indicators', ['None']))}
                
                Text: {data.get('text', 'No text available')}
                
                Emotion: {data.get('emotion', 'Unknown')}
                
                Recommended Approach:
                {data.get('approach', 'Not provided')}
                
                Timestamp: {data.get('timestamp', 'Unknown')}
                
                {'IMMEDIATE ACTION REQUIRED!' if risk_level >= 90 or is_emergency else 'Please check the admin dashboard for more details.'}
                """
            )
            print(f"Email notification sent to {admin_settings['notify_email']}")
        except Exception as e:
            print(f"Error sending email notification: {e}")
    
    print(f"{'EXTREME' if risk_level >= 90 else 'HIGH'} RISK ALERT: {risk_level}% - {data.get('user_id')} - Emergency: {is_emergency}")
    
    return jsonify({'status': 'success'})

def generate_escalation_explanation(text, indicators, emotion):
    """Generate an AI explanation for why this interaction was escalated"""
    # This is a simplified version - in production, you might want to use an AI model
    if not text and not indicators:
        return "This interaction was flagged due to risk assessment patterns."
    
    # Start with a general statement
    explanation = "This interaction was flagged because "
    
    # Add specifics based on available data
    reasons = []
    
    if text:
        # Look for concerning keywords
        concerning_keywords = [
            "suicide", "kill myself", "end my life", "don't want to live", 
            "better off dead", "no reason to live", "can't go on"
        ]
        
        found_keywords = [word for word in concerning_keywords if word.lower() in text.lower()]
        if found_keywords:
            reasons.append(f"the user expressed concerning statements related to self-harm or suicide")
        else:
            reasons.append(f"the user's language indicated potential distress")
    
    if indicators:
        if len(indicators) > 1:
            reasons.append(f"multiple risk indicators were detected: {', '.join(indicators[:3])}")
        else:
            reasons.append(f"a risk indicator was detected: {indicators[0]}")
    
    if emotion and emotion.lower() in ["sad", "angry", "fear"]:
        reasons.append(f"the user's emotional state ({emotion}) combined with their language raised concerns")
    
    # Combine reasons
    if reasons:
        explanation += reasons[0]
        if len(reasons) > 1:
            for i in range(1, len(reasons)-1):
                explanation += f", {reasons[i]}"
            explanation += f", and {reasons[-1]}"
        explanation += "."
    else:
        explanation += "the overall risk assessment exceeded the threshold."
    
    return explanation

def generate_approach_guidance(risk_level, is_emergency):
    """Generate guidance on how to approach this situation"""
    if is_emergency:
        return """
        IMMEDIATE ACTION REQUIRED:
        1. Respond with calm, supportive language acknowledging their feelings
        2. Explicitly encourage them to contact emergency services or a crisis helpline
        3. Provide specific crisis resources (National Suicide Prevention Lifeline: 988)
        4. Follow up with this user as soon as possible to ensure they received help
        """
    
    if risk_level >= 80:
        return """
        HIGH RISK APPROACH:
        1. Respond with empathy while avoiding reinforcing negative thoughts
        2. Gently but clearly suggest professional support
        3. Provide specific resources for immediate help
        4. Consider direct outreach if appropriate and within ethical guidelines
        """
    
    if risk_level >= 50:
        return """
        MODERATE RISK APPROACH:
        1. Validate their feelings while maintaining a supportive tone
        2. Suggest coping strategies and self-care techniques
        3. Mention that professional support is available if needed
        4. Monitor future interactions for changes in risk level
        """
    
    return """
    LOW RISK APPROACH:
    1. Maintain supportive conversation
    2. Provide positive reinforcement for healthy coping strategies
    3. Continue normal monitoring
    """

def send_email_notification(to_email, subject, body):
    """Send email notification for high risk alerts"""
    # This is a placeholder - implement with your preferred email service
    # For example, using smtplib or a service like SendGrid
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Email configuration - should be moved to settings
        smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.environ.get('SMTP_PORT', 587))
        smtp_user = os.environ.get('SMTP_USER', '')
        smtp_password = os.environ.get('SMTP_PASSWORD', '')
        
        if not smtp_user or not smtp_password:
            print("Email notification failed: SMTP credentials not configured")
            return False
        
        # Create message
        message = MIMEMultipart()
        message['From'] = smtp_user
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(message)
        
        return True
    except Exception as e:
        print(f"Error sending email notification: {e}")
        return False

@app.route('/api/interaction', methods=['POST'])
def add_interaction():
    """Add a new interaction to the dashboard"""
    data = request.json
    
    # Check if this is an automatic request without user action
    if data.get('automatic', False):
        # Skip processing for automatic requests
        return jsonify({'status': 'ignored', 'message': 'Automatic request ignored'})
    
    # Add timestamp if not present
    if 'timestamp' not in data:
        data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add to interactions list
    latest_interactions.insert(0, data)
    
    # Keep only the most recent 100 interactions
    if len(latest_interactions) > 100:
        latest_interactions.pop()
    
    # Emit to all connected clients
    socketio.emit('new_interaction', data)
    
    return jsonify({'status': 'success'})

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current admin settings for the main application"""
    return jsonify(admin_settings)

# WebSocket routes
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    # Send initial data to the newly connected client
    emit('init_data', {
        'alerts': risk_alerts,
        'interactions': latest_interactions,
        'settings': admin_settings
    })

if __name__ == '__main__':
    # Load settings at startup
    load_settings()
    
    # Start the Flask app with Socket.IO
    print("\nStarting Mirror Admin Panel")
    print("="*50)
    
    # Use port 5001 to avoid conflicts with the main mirror app
    port = int(os.environ.get('PORT', 5001))
    print(f"Admin panel running on http://localhost:{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
    
    # For production, uncomment this and use proper SSL certificates
    # socketio.run(app, host='0.0.0.0', port=5001, ssl_context=('cert.pem', 'key.pem')) 