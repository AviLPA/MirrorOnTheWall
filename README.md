# Mirror on the Wall

An AI-powered smart mirror for mental health support. It combines real-time emotion recognition, speech understanding, and personalized conversation to provide empathetic, context-aware responses while monitoring wellbeing over time.

**Contents:** [What It Is](#what-it-is) · [Quick Start](#quick-start) · [Parameters](#parameters-it-takes-into-account) · [Risk Assessment](#risk-assessment) · [User Journey](#user-journey) · [How It Learns](#how-it-learns) · [Deploy](#deploy-to-render)

---

## What It Is

**Mirror on the Wall** is a web-based interactive mirror that:

- **Sees you** — Uses your camera to detect facial expressions (emotion) and body language via DeepFace and MediaPipe
- **Listens** — Transcribes speech with Whisper and processes text input
- **Responds** — Generates supportive, risk-aware replies using GPT-4 with mental health–specific prompts
- **Remembers** — Tracks your journey across sessions and adapts its tone, topics, and suggestions

It is designed for low-stakes emotional support and check-ins, not clinical diagnosis or crisis intervention. High-risk situations trigger appropriate escalation prompts and crisis resources.

---

## Quick Start

```bash
./setup_fresh.sh   # One-time setup
./run_app.sh       # Start the app
```

Open **http://localhost:5000**. See [README_SETUP.md](README_SETUP.md) for details.

---

## Parameters It Takes Into Account

Each response is shaped by multiple inputs:

| Parameter | Source | Role |
|-----------|--------|------|
| **User input** | Speech (Whisper) or typed text | Primary content of what the user says |
| **Detected emotion** | DeepFace on face ROI | One of: angry, disgust, fear, happy, sad, surprise, neutral |
| **Body language** | Posture analysis | Cues like "relaxed", "tense" — used for context |
| **Risk level** | GPT-4 risk assessment (0–100%) | Drives response style and escalation |
| **Risk indicators** | List from risk assessment | Specific concerns flagged by the model |
| **Conversation history** | Last 10 turns in session | Keeps continuity within a session |
| **Session memory** | SystemMemoryManager | User profile, emotional patterns, topics from past sessions |
| **RAG context** | Chroma vector store over `research_papers/` | Evidence-based grounding from mental health resources |
| **User feedback** | Ratings (1–5) and optional text | Stored for future improvement (not yet used in prompts) |

### Relative Weight of Inputs

The model receives all signals as context; there are no hardcoded numeric weights. The prompts and guidelines imply the following **relative importance**:

| Signal | Weight / Role | Notes |
|--------|---------------|-------|
| **Words (user input)** | **Primary** | Drives risk most. Explicit crisis language (e.g. self-harm, suicide) overrides other signals. Common phrases like "bad day" or "feel like crap" are explicitly down-weighted (kept below 30% risk) regardless of emotion. |
| **Detected emotion** | **Secondary** | Supports or contradicts words. E.g. "sad" + concerning words → higher risk; "neutral" + same words → slightly lower. Never alone drives crisis escalation. |
| **Conversation history** | **Secondary** | Recent turns show escalation or repetition. Multi-turn context can raise or lower risk. |
| **Session memory** | **Secondary** | Past patterns (emotional trends, topics) inform "Consider the user's history" in risk assessment. |
| **Body language** | **Tertiary** | Used as context (e.g. "relaxed", "tense"). Often defaults to "relaxed"; posture analysis is simplified. Adds nuance, not primary driver. |
| **RAG context** | **Supporting** | Evidence-based content from research papers grounds responses; does not directly set risk level. |

**Design principle:** Words are trusted most for risk. The system is instructed not to overreact to facial expression or posture alone—e.g. a sad face with "I'm fine" stays low risk, while explicit crisis words trigger escalation even with neutral expression.

---

## Risk Assessment

Risk level (0–100%) drives response style:

| Range | Response type |
|-------|----------------|
| **0–79%** | Supportive, conversational |
| **80–89%** | Gentle professional support suggestion |
| **90–100%** | Crisis response with hotline and resources |

### Example outputs

Backend returns JSON; the "Analysis Results" panel (right side) shows Risk Level and Risk Indicators after each interaction.

**Low risk** — "I'm having a rough day":
```json
{"risk_level": 15, "risk_indicators": [], "is_emergency": false}
```
→ UI: `Risk Level: 15` · `Risk Indicators: None`

**Moderate** — anxiety/sleep concerns:
```json
{"risk_level": 62, "risk_indicators": ["anxiety mention", "sleep concerns"], "is_emergency": false}
```
→ UI: `Risk Level: 62` · `Risk Indicators: anxiety mention, sleep concerns`

**High** — hopelessness:
```json
{"risk_level": 85, "risk_indicators": ["hopelessness", "withdrawal", "sustained low mood"], "is_emergency": false}
```

**Crisis** — explicit self-harm/suicide mention:
```json
{"risk_level": 95, "risk_indicators": ["suicidal ideation", "intent expressed"], "is_emergency": true}
```
→ Crisis mode (hotline, resources); `is_emergency: true` can trigger admin alerts.

---

## User Journey

```
Login → Start Session → [Session summary + proactive questions loaded]
         ↓
    Analyze (optional) → Emotion + posture captured
         ↓
    Speak or type → Transcript sent to backend
         ↓
    Risk assessment → AI evaluates risk level
         ↓
    Response generation → Prompt chosen by risk level + context
         ↓
    TTS playback (optional) → User hears response
         ↓
    Interaction saved → Firebase + session memory updated
         ↓
    [Repeat or End Session]
```

**Session start** — When a user starts a session, the system:

- Loads a session summary from past interactions (e.g., emotional patterns, recent topics)
- Generates proactive questions (e.g., follow-ups from earlier sessions)
- Computes progress insights (trends over recent sessions)

**During a session** — Each interaction is logged with emotion, posture, risk level, and indicators.

**After a session** — The system memory updates and the next session is informed by the new data.

---

## How It Learns

The mirror improves over time by:

1. **Storing interactions** — Every turn is saved to Firebase (sessions, interactions, metadata).

2. **Session summaries** — `SystemMemoryManager` uses GPT-4 to analyze recent sessions and produce:
   - User profile (emotional patterns, primary concerns)
   - Recent context (last session summary, mood trend)
   - Proactive opportunities (follow-ups, milestones)
   - System notes (effective strategies, topics to avoid)

3. **Progress tracking** — Risk levels and emotions are compared across sessions to compute:
   - **Trend**: improving / declining / stable
   - **Average risk level**
   - **Common emotions**

4. **Proactive questions** — Questions are tailored from past sessions (e.g., check-ins, follow-ups, acknowledgments).

5. **Memory updates** — After each interaction, `update_interaction_memory` invalidates the cache and stores new interaction data for future summaries.

6. **RAG** — Research papers in `research_papers/` are embedded and retrieved to ground responses in evidence-based content.

7. **Feedback** — User ratings and feedback are stored in Firebase for potential future use in prompts or tuning.

---

## Deploy to Render

1. Create a new Web Service on Render and connect it to your GitHub repo `AviLPA/MirrorOnTheWall` on the `render-deploy` branch (we'll push it next).

2. Build & Start commands:
   - Build: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start: `gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:${PORT:-8080} app:app`

3. Environment variables (set in Render):
   - `OPENAI_API_KEY`: your OpenAI API key
   - `SECRET_KEY`: any random string
   - `PORT`: `8080` (Render will provide one; the default here is safe)

4. Optional: Pre-build the RAG DB by running `python setup_rag.py` locally and committing the `mental_health_db` folder, or let it build on first run.
