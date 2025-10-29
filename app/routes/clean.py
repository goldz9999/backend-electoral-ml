# app/routes/clean.py
from fastapi import APIRouter, HTTPException
from app.services.data_cleaning import DataCleaningService

# SIN prefix aquí
router = APIRouter()

@router.get("/analyze")
async def analyze():
    result = await DataCleaningService.analyze_data_quality()
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Error en análisis"))
    return result

@router.post("/clean-null")
async def clean_null():
    result = await DataCleaningService.clean_null_data()
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result

@router.post("/remove-duplicates")
async def remove_duplicates():
    result = await DataCleaningService.remove_duplicates()
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result

@router.post("/normalize")
async def normalize():
    result = await DataCleaningService.normalize_data()
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result