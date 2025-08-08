# Deploy to Render

1) Create a new Web Service on Render and connect it to your GitHub repo `AviLPA/MirrorOnTheWall` on the `render-deploy` branch (we'll push it next).

2) Build & Start commands
- Build: `pip install --upgrade pip && pip install -r requirements.txt`
- Start: `gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:${PORT:-8080} app:app`

3) Environment variables (set in Render):
- `OPENAI_API_KEY`: your OpenAI API key
- `SECRET_KEY`: any random string
- `PORT`: `8080` (Render will provide one; the default here is safe)

4) Optional: Pre-build the RAG DB by running `python setup_rag.py` locally and committing the `mental_health_db` folder, or let it build on first run.

# MirrorFinal
# MirrorFinal
