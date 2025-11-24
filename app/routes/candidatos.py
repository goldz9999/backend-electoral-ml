# app/routes/candidatos.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from app.config.settings import supabase_client
from datetime import datetime

router = APIRouter()

class CandidatoCreate(BaseModel):
    nombre: str = Field(..., min_length=3, max_length=150)
    partido: str = Field(..., min_length=3, max_length=150)
    tipo_eleccion: str = Field(..., pattern="^(presidencial|regional|distrital)$")
    descripcion: Optional[str] = None
    image_url: Optional[str] = None
    propuestas: Optional[List[dict]] = []
    experiencia: Optional[str] = None
    educacion: Optional[str] = None
    is_active: bool = True

class CandidatoUpdate(BaseModel):
    nombre: Optional[str] = None
    partido: Optional[str] = None
    tipo_eleccion: Optional[str] = None
    descripcion: Optional[str] = None
    image_url: Optional[str] = None
    propuestas: Optional[List[dict]] = None
    experiencia: Optional[str] = None
    educacion: Optional[str] = None
    is_active: Optional[bool] = None


@router.get("")
async def get_all_candidatos():
    """Obtiene todos los candidatos"""
    try:
        result = supabase_client.table("candidatos") \
            .select("*") \
            .eq("is_active", True) \
            .order("nombre") \
            .execute()
        
        return {
            "success": True,
            "data": result.data,
            "total": len(result.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tipo/{tipo_eleccion}")
async def get_candidatos_by_tipo(tipo_eleccion: str):
    """Obtiene candidatos por tipo de elección"""
    if tipo_eleccion not in ['presidencial', 'regional', 'distrital']:
        raise HTTPException(
            status_code=400,
            detail="tipo_eleccion debe ser: presidencial, regional o distrital"
        )
    
    try:
        result = supabase_client.table("candidatos") \
            .select("*") \
            .eq("tipo_eleccion", tipo_eleccion) \
            .eq("is_active", True) \
            .order("nombre") \
            .execute()
        
        return {
            "success": True,
            "tipo": tipo_eleccion,
            "data": result.data,
            "total": len(result.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{candidato_id}")
async def get_candidato_by_id(candidato_id: str):
    """Obtiene un candidato por ID"""
    try:
        result = supabase_client.table("candidatos") \
            .select("*") \
            .eq("id", candidato_id) \
            .single() \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Candidato no encontrado")
        
        return {
            "success": True,
            "data": result.data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_candidato(candidato: CandidatoCreate):
    """Crea un nuevo candidato"""
    try:
        candidato_dict = candidato.dict()
        candidato_dict['created_at'] = datetime.utcnow().isoformat()
        candidato_dict['updated_at'] = datetime.utcnow().isoformat()
        
        result = supabase_client.table("candidatos") \
            .insert(candidato_dict) \
            .execute()
        
        return {
            "success": True,
            "message": "Candidato creado exitosamente",
            "data": result.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{candidato_id}")
async def update_candidato(candidato_id: str, candidato: CandidatoUpdate):
    """Actualiza un candidato"""
    try:
        update_data = {k: v for k, v in candidato.dict(exclude_unset=True).items()}
        update_data['updated_at'] = datetime.utcnow().isoformat()
        
        result = supabase_client.table("candidatos") \
            .update(update_data) \
            .eq("id", candidato_id) \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Candidato no encontrado")
        
        return {
            "success": True,
            "message": "Candidato actualizado exitosamente",
            "data": result.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{candidato_id}")
async def delete_candidato(candidato_id: str):
    """Elimina (soft delete) un candidato"""
    try:
        result = supabase_client.table("candidatos") \
            .update({"is_active": False, "updated_at": datetime.utcnow().isoformat()}) \
            .eq("id", candidato_id) \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Candidato no encontrado")
        
        return {
            "success": True,
            "message": "Candidato eliminado exitosamente"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))