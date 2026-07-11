"""
BAKOME Local AI Studio - Backend API
Alternative open source à Jasper AI
Technologies : FastAPI + Ollama (Llama 3.2, Mistral, Phi)
Licence MIT | Auteur : Bakome Fabrice Kitoko
"""

import os
import re
import json
import sqlite3
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import aiohttp
import aiofiles
import markdown
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import httpx

# ========== CONFIGURATION ==========
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "studio.db"
TEMPLATES_DIR = DATA_DIR / "templates"
EXPORTS_DIR = DATA_DIR / "exports"
PROMPTS_DIR = Path("./prompts")

for d in [TEMPLATES_DIR, EXPORTS_DIR, PROMPTS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:3b")
JWT_SECRET = os.getenv("JWT_SECRET", "bakome-super-secret-key-change-me")
SESSION_TIMEOUT_HOURS = 24

# ========== BASE DE DONNÉES ==========
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT,
                template_id TEXT,
                prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                parameters TEXT,
                result TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(project_id) REFERENCES projects(id)
            );
            CREATE TABLE IF NOT EXISTS templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                user_prompt_template TEXT NOT NULL,
                example_input TEXT,
                created_at TEXT NOT NULL
            );
        """)
        # Insert default templates
        conn.execute("DELETE FROM templates")
        default_templates = [
            ("blog_post", "Rédaction de blog", "marketing", 
             "Tu es un expert en rédaction marketing. Écris en français.", 
             "Rédige un article de blog sur le thème : {{topic}}. Style : {{style}}. Longueur : {{length}} mots.",
             '{"topic": "IA open source", "style": "professionnel", "length": 800}'),
            ("landing_page", "Landing page", "marketing",
             "Tu es un copywriter spécialisé en conversion.", 
             "Génère le texte complet d'une landing page pour le produit : {{product}}. Public cible : {{audience}}. Avantages : {{benefits}}.",
             '{"product": "BAKOME Studio", "audience": "développeurs", "benefits": "local, gratuit, privé"}'),
            ("email_newsletter", "Email newsletter", "marketing",
             "Tu es un expert en email marketing.", 
             "Crée un email pour annoncer : {{announcement}}. Ton : {{tone}}. Call to action : {{cta}}.",
             '{"announcement": "nouvelle version 2.0", "tone": "enthousiaste", "cta": "Télécharger gratuitement"}'),
            ("code_docstring", "Génération de docstring", "coding",
             "Tu es un développeur expert en documentation propre.", 
             "Génère une docstring pour le code suivant (langage {{lang}}) :\n```{{lang}}\n{{code}}\n```",
             '{"lang": "python", "code": "def add(a,b): return a+b"}'),
            ("social_linkedin", "Post LinkedIn", "social",
             "Tu es un influenceur tech. Rédige un post concis et engageant.",
             "Écris un post LinkedIn sur : {{topic}}. Mentionne : {{hashtags}}. Appel à l'action : {{cta}}.",
             '{"topic": "open source", "hashtags": "#dev #oss", "cta": "Dites-moi ce que vous en pensez"}'),
        ]
        for tid, name, cat, sys_prompt, user_tpl, example in default_templates:
            conn.execute("""
                INSERT INTO templates (id, name, category, system_prompt, user_prompt_template, example_input, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (tid, name, cat, sys_prompt, user_tpl, example, datetime.utcnow().isoformat()))
        conn.commit()

init_db()

# ========== MODÈLES PYDANTIC ==========
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class GenerationRequest(BaseModel):
    project_id: Optional[str] = None
    template_id: Optional[str] = None
    prompt: str
    model: str = DEFAULT_MODEL
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ProjectCreate(BaseModel):
    name: str

class TemplateUpdate(BaseModel):
    name: str
    category: str
    system_prompt: str
    user_prompt_template: str
    example_input: Optional[str] = None

# ========== SÉCURITÉ SIMPLE (à remplacer par JWT en prod) ==========
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        return None  # mode démo / pas d'auth
    token = credentials.credentials
    with get_db() as conn:
        row = conn.execute("SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Session invalide")
        if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Session expirée")
        user = conn.execute("SELECT id, email, full_name FROM users WHERE id = ?", (row["user_id"],)).fetchone()
        return dict(user)

# ========== OLLAMA CLIENT ==========
async def ollama_generate(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """Appelle Ollama en local pour générer du texte."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {resp.text}")
        data = resp.json()
        return data.get("response", "")

def render_template(template_str: str, variables: Dict[str, Any]) -> str:
    """Remplace {{variable}} par sa valeur."""
    for key, val in variables.items():
        template_str = template_str.replace("{{"+key+"}}", str(val))
    return template_str

# ========== ROUTES API ==========
app = FastAPI(title="BAKOME Local AI Studio", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"message": "BAKOME Local AI Studio API", "status": "operational", "docs": "/docs"}

@app.post("/auth/register")
async def register(user: UserCreate):
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (user.email,)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email déjà utilisé")
        user_id = str(uuid.uuid4())
        # En production : hasher le mot de passe
        conn.execute("INSERT INTO users (id, email, password_hash, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
                     (user_id, user.email, user.password, user.full_name, datetime.utcnow().isoformat()))
        conn.commit()
    return {"id": user_id, "email": user.email}

@app.post("/auth/login")
async def login(user: UserLogin):
    with get_db() as conn:
        row = conn.execute("SELECT id, email, full_name FROM users WHERE email = ? AND password_hash = ?",
                           (user.email, user.password)).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Identifiants invalides")
        token = str(uuid.uuid4())
        expires = datetime.utcnow().replace(hour=(datetime.utcnow().hour + SESSION_TIMEOUT_HOURS)).isoformat()
        conn.execute("INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                     (token, row["id"], expires))
        conn.commit()
        return {"access_token": token, "token_type": "bearer", "user": dict(row)}

@app.get("/templates")
async def list_templates(category: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        if category:
            rows = conn.execute("SELECT * FROM templates WHERE category = ? ORDER BY name", (category,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM templates ORDER BY category, name").fetchall()
        return [dict(r) for r in rows]

@app.post("/templates")
async def create_template(tpl: TemplateUpdate, current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=403, detail="Authentification requise")
    tid = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute("""
            INSERT INTO templates (id, name, category, system_prompt, user_prompt_template, example_input, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (tid, tpl.name, tpl.category, tpl.system_prompt, tpl.user_prompt_template, tpl.example_input, datetime.utcnow().isoformat()))
        conn.commit()
    return {"id": tid}

@app.get("/templates/{template_id}")
async def get_template(template_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Template non trouvé")
        return dict(row)

@app.post("/generate")
async def generate(request: GenerationRequest, background_tasks: BackgroundTasks, current_user: Optional[dict] = Depends(get_current_user)):
    """
    Génère du contenu via Ollama.
    Si template_id est fourni, utilise le système et le modèle du template.
    """
    system = "Tu es un assistant IA utile et concis. Réponds en français sauf indication contraire."
    user_prompt = request.prompt
    category = "custom"

    if request.template_id:
        with get_db() as conn:
            tpl = conn.execute("SELECT * FROM templates WHERE id = ?", (request.template_id,)).fetchone()
            if tpl:
                system = tpl["system_prompt"]
                # Permet l'utilisation de variables JSON dans les paramètres
                if tpl["user_prompt_template"] and "{{" in tpl["user_prompt_template"]:
                    user_prompt = render_template(tpl["user_prompt_template"], request.parameters)
                else:
                    user_prompt = request.prompt
                category = tpl["category"]

    # Appel à Ollama
    result_text = await ollama_generate(request.model, system, user_prompt)

    # Sauvegarde en base
    user_id = current_user["id"] if current_user else "anonymous"
    gen_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute("""
            INSERT INTO generations (id, user_id, project_id, template_id, prompt, model, parameters, result, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (gen_id, user_id, request.project_id, request.template_id, request.prompt, request.model,
              json.dumps(request.parameters), result_text, datetime.utcnow().isoformat()))
        conn.commit()

    return {"id": gen_id, "result": result_text, "model": request.model}

@app.get("/history")
async def get_history(limit: int = 50, current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=403, detail="Authentification requise")
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, project_id, template_id, substr(prompt,1,100) as prompt_preview, model, substr(result,1,200) as result_preview, created_at
            FROM generations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
        """, (current_user["id"], limit)).fetchall()
        return [dict(r) for r in rows]

@app.post("/export/pdf")
async def export_pdf(request: GenerationRequest, current_user: dict = Depends(get_current_user)):
    """Génère un PDF à partir d'un prompt (utilise l'IA puis exporte)."""
    # On appelle d'abord la génération
    gen = await generate(request, background_tasks=None, current_user=current_user)
    result_text = gen["result"]
    filename = f"export_{uuid.uuid4().hex}.pdf"
    filepath = EXPORTS_DIR / filename

    # Création du PDF avec ReportLab
    doc = SimpleDocTemplate(str(filepath), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("BAKOME AI Studio – Export", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Requête : {request.prompt[:200]}", styles["Heading2"]))
    story.append(Spacer(1, 12))
    # Convertit le texte markdown en HTML (simplifié)
    html = markdown.markdown(result_text)
    # ReportLab accepte du HTML simple via Paragraph
    from reportlab.lib.utils import simpleSplit
    story.append(Paragraph(html, styles["Normal"]))
    doc.build(story)

    return FileResponse(str(filepath), media_type="application/pdf", filename=filename)

@app.get("/models")
async def list_models():
    """Récupère la liste des modèles disponibles sur Ollama."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [{"name": m["name"], "size": m.get("size"), "modified": m.get("modified")} for m in data.get("models", [])]
    except:
        pass
    return [{"name": DEFAULT_MODEL, "size": "unknown"}]

@app.post("/projects")
async def create_project(proj: ProjectCreate, current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=403, detail="Authentification requise")
    pid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute("INSERT INTO projects (id, user_id, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                     (pid, current_user["id"], proj.name, now, now))
        conn.commit()
    return {"id": pid, "name": proj.name}

@app.get("/projects")
async def list_projects(current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=403, detail="Authentification requise")
    with get_db() as conn:
        rows = conn.execute("SELECT id, name, created_at FROM projects WHERE user_id = ? ORDER BY updated_at DESC", (current_user["id"],)).fetchall()
        return [dict(r) for r in rows]

# ========== LANCEMENT ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
