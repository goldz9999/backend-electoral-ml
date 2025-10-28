# app/routes/electoral.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from app.config.settings import supabase_client
import re

router = APIRouter()


class VoteRequest(BaseModel):
    nombre: str = Field(..., min_length=2, max_length=100)
    apellido: str = Field(..., min_length=2, max_length=100)
    dni: str = Field(..., pattern=r'^\d{8}$')
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')  # Validación básica
    celular: str = Field(..., pattern=r'^\d{9}$')
    departamento: str = Field(..., min_length=3)
    provincia: str = Field(..., min_length=3)
    distrito: str = Field(..., min_length=3)
    candidate_id: int


@router.post("/votes")
async def submit_vote(vote: VoteRequest):
    """
    Registra un voto:
    1. Crea o verifica votante en `voters`
    2. Registra voto en `votes`
    3. Actualiza `has_voted = true`
    """
    try:
        # === 1. Verificar si el votante ya existe (por DNI o email) ===
        voter_check = supabase_client.table("voters") \
            .select("id, has_voted") \
            .or_(f"dni.eq.{vote.dni},email.eq.{vote.email}") \
            .execute()

        if voter_check.data:
            voter = voter_check.data[0]
            if voter["has_voted"]:
                raise HTTPException(
                    status_code=400,
                    detail="Ya has votado anteriormente con este DNI o email"
                )
            voter_id = voter["id"]
        else:
            # === 2. Crear nuevo votante ===
            voter_insert = supabase_client.table("voters").insert({
                "nombre": vote.nombre,
                "apellido": vote.apellido,
                "dni": vote.dni,
                "email": vote.email,
                "celular": vote.celular,
                "departamento": vote.departamento,
                "provincia": vote.provincia,
                "distrito": vote.distrito,
                "has_voted": False
            }).execute()

            if not voter_insert.data:
                raise HTTPException(500, "Error al registrar votante")

            voter_id = voter_insert.data[0]["id"]

        # === 3. Verificar que el candidato existe ===
        candidate = supabase_client.table("candidates") \
            .select("id") \
            .eq("id", vote.candidate_id) \
            .single() \
            .execute()

        if not candidate.data:
            raise HTTPException(404, "Candidato no encontrado")

        # === 4. Insertar voto en `votes` ===
        voter_name = f"{vote.nombre} {vote.apellido}"
        voter_location = f"{vote.distrito}, {vote.provincia}, {vote.departamento}"

        vote_insert = supabase_client.table("votes").insert({
            "candidate_id": vote.candidate_id,
            "voter_id": voter_id,
            "voter_name": voter_name,
            "voter_email": vote.email,
            "voter_location": voter_location,
            "voted_at": datetime.utcnow().isoformat(),
            "is_valid": True
        }).execute()

        if not vote_insert.data:
            raise HTTPException(500, "Error al registrar voto")

        # === 5. Marcar como votado ===
        supabase_client.table("voters") \
            .update({
                "has_voted": True,
                "voted_at": datetime.utcnow().isoformat()
            }) \
            .eq("id", voter_id) \
            .execute()

        return {
            "message": "Voto registrado exitosamente",
            "vote_id": vote_insert.data[0]["id"],
            "voter_name": voter_name,
            "voter_id": voter_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al registrar voto: {str(e)}")
        raise HTTPException(500, f"Error interno: {str(e)}")


@router.get("/votes/check")
async def check_if_voted(dni: str, email: str):
    """Verifica si ya votó por DNI o Email"""
    if not re.match(r'^\d{8}$', dni):
        raise HTTPException(400, "DNI debe tener 8 dígitos")

    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        raise HTTPException(400, "Email inválido")

    # Buscar en `voters`
    result = supabase_client.table("voters") \
        .select("has_voted") \
        .or_(f"dni.eq.{dni},email.eq.{email}") \
        .execute()

    has_voted = any(v["has_voted"] for v in result.data)

    return {
        "has_voted": has_voted,
        "dni": dni,
        "email": email
    }


@router.get("/results")
async def get_results():
    try:
        # === OBTENER VOTOS VÁLIDOS ===
        votes_res = supabase_client.table("votes") \
            .select("candidate_id") \
            .eq("is_valid", True) \
            .execute()

        vote_counts = {}
        if votes_res.data:
            from collections import Counter
            vote_counts = Counter(v["candidate_id"] for v in votes_res.data)
        total_votes = len(votes_res.data)

        # === OBTENER TODOS LOS CANDIDATOS ===
        candidates_res = supabase_client.table("candidates") \
            .select("id, name, party") \
            .execute()

        if not candidates_res.data:
            return {
                "results": [],
                "total_votes": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "No hay candidatos registrados"
            }

        # === CONSTRUIR RESULTADOS (incluso con 0 votos) ===
        results = []
        for candidate in candidates_res.data:
            count = vote_counts.get(candidate["id"], 0)
            percentage = round((count / total_votes * 100), 2) if total_votes > 0 else 0.00
            results.append({
                "candidate_id": candidate["id"],
                "name": candidate["name"],
                "party": candidate["party"],
                "votes": count,
                "percentage": percentage
            })

        # Ordenar por votos
        results.sort(key=lambda x: x["votes"], reverse=True)

        return {
            "results": results,
            "total_votes": total_votes,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        print("Error en /results:", str(e))
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/candidates")
async def get_candidates():
    """Lista de candidatos"""
    result = supabase_client.table("candidates") \
        .select("*") \
        .order("name") \
        .execute()

    return {
        "candidates": result.data,
        "total": len(result.data)
    }