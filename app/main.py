# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import get_settings
from app.routes import electoral
from app.routes import clean  
from app.routes import train
from app.routes import analytics  # ← AGREGAR

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    description="API Backend Electoral con ML",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RUTAS
app.include_router(electoral.router, prefix=settings.api_prefix, tags=["Electoral"])
app.include_router(clean.router, prefix=settings.api_prefix, tags=["Data Cleaning"])
app.include_router(train.router, prefix=settings.api_prefix, tags=["Model Training"])
app.include_router(analytics.router, prefix=settings.api_prefix, tags=["Analytics"])  # ← AGREGAR

@app.get("/")
async def root():
    return {
        "message": "Backend Electoral ML API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    from app.config.settings import supabase_client
    
    try:
        supabase_client.table("candidates").select("id").limit(1).execute()
        status = "healthy"
        supabase_status = "connected"
    except Exception as e:
        status = "unhealthy"
        supabase_status = f"error: {str(e)}"
    
    return {
        "status": status,
        "supabase": supabase_status
    }