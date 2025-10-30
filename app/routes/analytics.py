# app/routes/analytics.py
from fastapi import APIRouter, HTTPException, Query
from app.services.analytics_service import AnalyticsService
from typing import Optional

router = APIRouter()


@router.get("/analytics/overview")
async def get_overview():
    """
    Dashboard general con KPIs principales:
    - Total de votos
    - Participación por departamento
    - Distribución por género y edad
    - Tendencias temporales
    """
    try:
        result = await AnalyticsService.get_overview()
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/analytics/demographic")
async def get_demographic_analysis():
    """
    Análisis demográfico detallado:
    - Distribución por edad, género, educación
    - Preferencias de voto por segmento
    - Correlaciones demográficas
    """
    try:
        result = await AnalyticsService.get_demographic_analysis()
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/analytics/geographic")
async def get_geographic_analysis(
    departamento: Optional[str] = Query(None, description="Filtrar por departamento"),
    provincia: Optional[str] = Query(None, description="Filtrar por provincia")
):
    """
    Análisis geográfico:
    - Votos por departamento/provincia/distrito
    - Mapas de calor (heatmaps)
    - Rankings regionales
    """
    try:
        result = await AnalyticsService.get_geographic_analysis(departamento, provincia)
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/analytics/temporal")
async def get_temporal_analysis():
    """
    Análisis temporal:
    - Votos por hora del día
    - Votos por día de la semana
    - Tendencias de participación en el tiempo
    """
    try:
        result = await AnalyticsService.get_temporal_analysis()
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/analytics/candidates")
async def get_candidate_performance():
    """
    Análisis de desempeño por candidato:
    - Votos y % por candidato
    - Fortalezas geográficas
    - Segmentos demográficos clave
    """
    try:
        result = await AnalyticsService.get_candidate_performance()
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/analytics/clustering")
async def get_voting_clusters(n_clusters: int = Query(3, ge=2, le=10)):
    """
    Análisis de clustering (K-Means):
    - Agrupa votantes por patrones similares
    - Identifica perfiles de votantes
    - Características de cada cluster
    """
    try:
        result = await AnalyticsService.get_voting_clusters(n_clusters)
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/analytics/correlations")
async def get_correlations():
    """
    Matriz de correlaciones:
    - Relaciones entre variables demográficas
    - Patrones de votación
    """
    try:
        result = await AnalyticsService.get_correlations()
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/analytics/predictions")
async def get_predictions():
    """
    Predicciones basadas en modelos ML:
    - Proyecciones de resultados finales
    - Tendencias probables
    - Confianza de predicciones
    """
    try:
        result = await AnalyticsService.get_predictions()
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")