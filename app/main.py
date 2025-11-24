# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import get_settings
from app.routes import candidatos, votantes, votos, estadisticas

settings = get_settings()

app = FastAPI(
    title="Backend Electoral ONPE",
    description="API Backend Electoral con Supabase",
    version="2.0.0"
)

# ✅ CORS MEJORADO
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),  # ← Usar el método
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# RUTAS
app.include_router(candidatos.router, prefix="/api/candidatos", tags=["Candidatos"])
app.include_router(votantes.router, prefix="/api/votantes", tags=["Votantes"])
app.include_router(votos.router, prefix="/api/votos", tags=["Votos"])
app.include_router(estadisticas.router, prefix="/api/estadisticas", tags=["Estadísticas"])

@app.get("/")
async def root():
    return {
        "message": "Backend Electoral ONPE API",
        "version": "2.0.0",
        "docs": "/docs",
        "cors_origins": settings.get_cors_origins()
    }

@app.get("/health")
async def health_check():
    from app.config.settings import supabase_client
    
    try:
        supabase_client.table("candidatos").select("id").limit(1).execute()
        status = "healthy"
        supabase_status = "connected"
    except Exception as e:
        status = "unhealthy"
        supabase_status = f"error: {str(e)}"
    
    return {
        "status": status,
        "supabase": supabase_status,
        "cors_enabled": True
    }