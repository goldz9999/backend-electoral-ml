# app/routes/votos.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.config.settings import supabase_client
from datetime import datetime
import re

router = APIRouter()

class VotoCreate(BaseModel):
    votante_id: str
    candidato_id: str
    dni_votante: str = Field(..., pattern=r'^\d{8}$')
    departamento: str
    provincia: str
    distrito: str


# ============================================
# VOTOS PRESIDENCIALES
# ============================================

@router.get("/presidenciales")
async def get_votos_presidenciales():
    """Obtiene todos los votos presidenciales"""
    try:
        result = supabase_client.table("votos_presidenciales") \
            .select("*, votantes(*), candidatos(*)") \
            .order("fecha_voto", desc=True) \
            .execute()
        
        return {
            "success": True,
            "data": result.data,
            "total": len(result.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presidenciales/check/{dni}")
async def check_voto_presidencial(dni: str):
    """Verifica si un DNI ya votó en presidencial"""
    if not re.match(r'^\d{8}$', dni):
        raise HTTPException(status_code=400, detail="DNI debe tener 8 dígitos")
    
    try:
        result = supabase_client.table("votos_presidenciales") \
            .select("id") \
            .eq("dni_votante", dni) \
            .execute()
        
        has_voted = len(result.data) > 0
        
        return {
            "success": True,
            "has_voted": has_voted,
            "dni": dni
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/presidenciales")
async def create_voto_presidencial(voto: VotoCreate):
    """Registra un voto presidencial"""
    try:
        # Verificar si ya votó
        existing = supabase_client.table("votos_presidenciales") \
            .select("id") \
            .eq("dni_votante", voto.dni_votante) \
            .execute()
        
        if existing.data:
            raise HTTPException(
                status_code=400,
                detail="Este DNI ya emitió su voto presidencial"
            )
        
        # Verificar votante existe
        votante = supabase_client.table("votantes") \
            .select("id") \
            .eq("id", voto.votante_id) \
            .single() \
            .execute()
        
        if not votante.data:
            raise HTTPException(status_code=404, detail="Votante no encontrado")
        
        # Verificar candidato existe y es presidencial
        candidato = supabase_client.table("candidatos") \
            .select("id, tipo_eleccion") \
            .eq("id", voto.candidato_id) \
            .single() \
            .execute()
        
        if not candidato.data:
            raise HTTPException(status_code=404, detail="Candidato no encontrado")
        
        if candidato.data['tipo_eleccion'] != 'presidencial':
            raise HTTPException(
                status_code=400,
                detail="El candidato no es de tipo presidencial"
            )
        
        # Registrar voto
        voto_dict = voto.dict()
        voto_dict['fecha_voto'] = datetime.utcnow().isoformat()
        
        result = supabase_client.table("votos_presidenciales") \
            .insert(voto_dict) \
            .execute()
        
        return {
            "success": True,
            "message": "Voto presidencial registrado exitosamente",
            "data": result.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/presidenciales/{voto_id}")
async def delete_voto_presidencial(voto_id: str):
    """Elimina un voto presidencial"""
    try:
        result = supabase_client.table("votos_presidenciales") \
            .delete() \
            .eq("id", voto_id) \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Voto no encontrado")
        
        return {
            "success": True,
            "message": "Voto eliminado exitosamente"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# VOTOS REGIONALES
# ============================================

@router.get("/regionales")
async def get_votos_regionales():
    """Obtiene todos los votos regionales"""
    try:
        result = supabase_client.table("votos_regionales") \
            .select("*, votantes(*), candidatos(*)") \
            .order("fecha_voto", desc=True) \
            .execute()
        
        return {
            "success": True,
            "data": result.data,
            "total": len(result.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regionales/check/{dni}")
async def check_voto_regional(dni: str):
    """Verifica si un DNI ya votó en regional"""
    if not re.match(r'^\d{8}$', dni):
        raise HTTPException(status_code=400, detail="DNI debe tener 8 dígitos")
    
    try:
        result = supabase_client.table("votos_regionales") \
            .select("id") \
            .eq("dni_votante", dni) \
            .execute()
        
        has_voted = len(result.data) > 0
        
        return {
            "success": True,
            "has_voted": has_voted,
            "dni": dni
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regionales")
async def create_voto_regional(voto: VotoCreate):
    """Registra un voto regional"""
    try:
        # Verificar si ya votó
        existing = supabase_client.table("votos_regionales") \
            .select("id") \
            .eq("dni_votante", voto.dni_votante) \
            .execute()
        
        if existing.data:
            raise HTTPException(
                status_code=400,
                detail="Este DNI ya emitió su voto regional"
            )
        
        # Verificar votante existe
        votante = supabase_client.table("votantes") \
            .select("id") \
            .eq("id", voto.votante_id) \
            .single() \
            .execute()
        
        if not votante.data:
            raise HTTPException(status_code=404, detail="Votante no encontrado")
        
        # Verificar candidato existe y es regional
        candidato = supabase_client.table("candidatos") \
            .select("id, tipo_eleccion") \
            .eq("id", voto.candidato_id) \
            .single() \
            .execute()
        
        if not candidato.data:
            raise HTTPException(status_code=404, detail="Candidato no encontrado")
        
        if candidato.data['tipo_eleccion'] != 'regional':
            raise HTTPException(
                status_code=400,
                detail="El candidato no es de tipo regional"
            )
        
        # Registrar voto
        voto_dict = voto.dict()
        voto_dict['fecha_voto'] = datetime.utcnow().isoformat()
        
        result = supabase_client.table("votos_regionales") \
            .insert(voto_dict) \
            .execute()
        
        return {
            "success": True,
            "message": "Voto regional registrado exitosamente",
            "data": result.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/regionales/{voto_id}")
async def delete_voto_regional(voto_id: str):
    """Elimina un voto regional"""
    try:
        result = supabase_client.table("votos_regionales") \
            .delete() \
            .eq("id", voto_id) \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Voto no encontrado")
        
        return {
            "success": True,
            "message": "Voto eliminado exitosamente"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# VOTOS DISTRITALES
# ============================================

@router.get("/distritales")
async def get_votos_distritales():
    """Obtiene todos los votos distritales"""
    try:
        result = supabase_client.table("votos_distritales") \
            .select("*, votantes(*), candidatos(*)") \
            .order("fecha_voto", desc=True) \
            .execute()
        
        return {
            "success": True,
            "data": result.data,
            "total": len(result.data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distritales/check/{dni}")
async def check_voto_distrital(dni: str):
    """Verifica si un DNI ya votó en distrital"""
    if not re.match(r'^\d{8}$', dni):
        raise HTTPException(status_code=400, detail="DNI debe tener 8 dígitos")
    
    try:
        result = supabase_client.table("votos_distritales") \
            .select("id") \
            .eq("dni_votante", dni) \
            .execute()
        
        has_voted = len(result.data) > 0
        
        return {
            "success": True,
            "has_voted": has_voted,
            "dni": dni
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/distritales")
async def create_voto_distrital(voto: VotoCreate):
    """Registra un voto distrital"""
    try:
        # Verificar si ya votó
        existing = supabase_client.table("votos_distritales") \
            .select("id") \
            .eq("dni_votante", voto.dni_votante) \
            .execute()
        
        if existing.data:
            raise HTTPException(
                status_code=400,
                detail="Este DNI ya emitió su voto distrital"
            )
        
        # Verificar votante existe
        votante = supabase_client.table("votantes") \
            .select("id") \
            .eq("id", voto.votante_id) \
            .single() \
            .execute()
        if not votante.data:
            raise HTTPException(status_code=404, detail="Votante no encontrado")
        
        # Verificar candidato existe y es distrital
        candidato = supabase_client.table("candidatos") \
            .select("id, tipo_eleccion") \
            .eq("id", voto.candidato_id) \
            .single() \
            .execute()
        if not candidato.data:
            raise HTTPException(status_code=404, detail="Candidato no encontrado")
        if candidato.data['tipo_eleccion'] != 'distrital':
            raise HTTPException(
                status_code=400,
                detail="El candidato no es de tipo distrital"
            )
        # Registrar voto
        voto_dict = voto.dict()
        voto_dict['fecha_voto'] = datetime.utcnow().isoformat()
        result = supabase_client.table("votos_distritales") \
            .insert(voto_dict) \
            .execute()
        return {
            "success": True,
            "message": "Voto distrital registrado exitosamente",
            "data": result.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.delete("/distritales/{voto_id}")
async def delete_voto_distrital(voto_id: str):
    """Elimina un voto distrital"""
    try:
        result = supabase_client.table("votos_distritales") \
            .delete() \
            .eq("id", voto_id) \
            .execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Voto no encontrado")
        
        return {
            "success": True,
            "message": "Voto distrital eliminado exitosamente"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))