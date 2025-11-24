# app/routes/votantes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.config.settings import supabase_client
from datetime import datetime
import re

router = APIRouter()

class VotanteCreate(BaseModel):
    dni: str = Field(..., pattern=r'^\d{8}$')
    nombres: str = Field(..., min_length=2, max_length=100)
    apellido_paterno: str = Field(..., min_length=2, max_length=100)
    apellido_materno: str = Field(..., min_length=2, max_length=100)
    departamento: str = Field(..., min_length=3)
    provincia: str = Field(..., min_length=3)
    distrito: str = Field(..., min_length=3)
    direccion: Optional[str] = None
    direccion_completa: Optional[str] = None
    ubigeo_reniec: Optional[str] = None
    ubigeo_sunat: Optional[str] = None
    telefono: Optional[str] = None
    email: Optional[str] = None
    estado: str = 'Activo'

class VotanteUpdate(BaseModel):
    nombres: Optional[str] = None
    apellido_paterno: Optional[str] = None
    apellido_materno: Optional[str] = None
    departamento: Optional[str] = None
    provincia: Optional[str] = None
    distrito: Optional[str] = None
    direccion: Optional[str] = None
    direccion_completa: Optional[str] = None
    ubigeo_reniec: Optional[str] = None
    ubigeo_sunat: Optional[str] = None
    telefono: Optional[str] = None
    email: Optional[str] = None
    estado: Optional[str] = None


@router.get("")
async def get_all_votantes():
    """Obtiene todos los votantes"""
    try:
        result = supabase_client.table("votantes") \
            .select("*") \
            .order("created_at", desc=True) \
            .execute()
        
        return {
            "success": True,
            "data": result.data,
            "total": len(result.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dni/{dni}")
async def get_votante_by_dni(dni: str):
    """Obtiene un votante por DNI"""
    if not re.match(r'^\d{8}$', dni):
        raise HTTPException(status_code=400, detail="DNI debe tener 8 dígitos")
    
    try:
        result = supabase_client.table("votantes") \
            .select("*") \
            .eq("dni", dni) \
            .single() \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Votante no encontrado")
        
        return {
            "success": True,
            "data": result.data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{votante_id}")
async def get_votante_by_id(votante_id: str):
    """Obtiene un votante por ID"""
    try:
        result = supabase_client.table("votantes") \
            .select("*") \
            .eq("id", votante_id) \
            .single() \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Votante no encontrado")
        
        return {
            "success": True,
            "data": result.data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def create_votante(votante: VotanteCreate):
    """Crea un nuevo votante"""
    try:
        # Verificar si el DNI ya existe
        existing = supabase_client.table("votantes") \
            .select("id") \
            .eq("dni", votante.dni) \
            .execute()
        
        if existing.data:
            raise HTTPException(
                status_code=400,
                detail="Ya existe un votante con este DNI"
            )
        
        votante_dict = votante.dict()
        votante_dict['created_at'] = datetime.utcnow().isoformat()
        votante_dict['updated_at'] = datetime.utcnow().isoformat()
        
        result = supabase_client.table("votantes") \
            .insert(votante_dict) \
            .execute()
        
        return {
            "success": True,
            "message": "Votante registrado exitosamente",
            "data": result.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{votante_id}")
async def update_votante(votante_id: str, votante: VotanteUpdate):
    """Actualiza un votante"""
    try:
        update_data = {k: v for k, v in votante.dict(exclude_unset=True).items()}
        update_data['updated_at'] = datetime.utcnow().isoformat()
        
        result = supabase_client.table("votantes") \
            .update(update_data) \
            .eq("id", votante_id) \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Votante no encontrado")
        
        return {
            "success": True,
            "message": "Votante actualizado exitosamente",
            "data": result.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{votante_id}")
async def delete_votante(votante_id: str):
    """Elimina un votante"""
    try:
        result = supabase_client.table("votantes") \
            .delete() \
            .eq("id", votante_id) \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Votante no encontrado")
        
        return {
            "success": True,
            "message": "Votante eliminado exitosamente"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))