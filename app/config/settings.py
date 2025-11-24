# app/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field
from supabase import create_client, Client
from functools import lru_cache
import json


class Settings(BaseSettings):
    # Supabase Configuration
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_service_role_key: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")
    
    # CORS Configuration - ✅ CORREGIDO
    cors_origins: str = Field(
        default='["http://localhost:5173", "http://localhost:8080", "http://127.0.0.1:5173", "https://easy-vote-portal.vercel.app"]',
        env="CORS_ORIGINS"
    )
    
    # API Configuration
    api_prefix: str = Field(default="/api", env="API_PREFIX")
    project_name: str = Field(default="Backend ML API", env="PROJECT_NAME")
    debug: bool = Field(default=True, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_cors_origins(self):
        """Convierte el string JSON a lista"""
        try:
            return json.loads(self.cors_origins)
        except:
            return ["http://localhost:5173", "http://localhost:8080"]


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_supabase_client() -> Client:
    settings = get_settings()
    supabase: Client = create_client(
        settings.supabase_url,
        settings.supabase_service_role_key
    )
    return supabase


supabase_client = get_supabase_client()