# app/routes/train.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.ml_training import MLTrainingService
from typing import Optional

router = APIRouter()


class TrainModelRequest(BaseModel):
    model_type: str  # "classification" o "regression"
    algorithm: str   # "random_forest", "logistic_regression", "gradient_boosting"
    test_size: float = 0.2
    random_state: int = 42


@router.post("/train")
async def train_model(request: TrainModelRequest):
    """
    Entrena un modelo de Machine Learning con los datos de votos.
    
    Modelos disponibles:
    - classification: Random Forest, Logistic Regression, Gradient Boosting
    - regression: Linear Regression, Ridge, Lasso
    """
    try:
        result = await MLTrainingService.train_model(
            model_type=request.model_type,
            algorithm=request.algorithm,
            test_size=request.test_size,
            random_state=request.random_state
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Error en entrenamiento")
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )


@router.get("/models")
async def get_models():
    """Obtiene todos los modelos entrenados"""
    try:
        result = await MLTrainingService.get_all_models()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener modelos: {str(e)}"
        )


@router.get("/models/{model_id}")
async def get_model_details(model_id: int):
    """Obtiene detalles de un modelo específico"""
    try:
        result = await MLTrainingService.get_model_details(model_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail="Modelo no encontrado"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener modelo: {str(e)}"
        )


@router.get("/models/{model_id}/metrics")
async def get_model_metrics(model_id: int):
    """Obtiene métricas de un modelo específico"""
    try:
        result = await MLTrainingService.get_model_metrics(model_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail="Métricas no encontradas"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener métricas: {str(e)}"
        )


@router.get("/models/{model_id}/history")
async def get_training_history(model_id: int):
    """Obtiene historial de entrenamiento"""
    try:
        result = await MLTrainingService.get_training_history(model_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener historial: {str(e)}"
        )


@router.post("/predict")
async def make_prediction(model_id: int, features: dict):
    """Realiza predicciones con un modelo entrenado"""
    try:
        result = await MLTrainingService.predict(model_id, features)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Error en predicción")
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción: {str(e)}"
        )


@router.delete("/models/{model_id}")
async def delete_model(model_id: int):
    """Elimina un modelo"""
    try:
        result = await MLTrainingService.delete_model(model_id)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=404,
                detail="Modelo no encontrado"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al eliminar modelo: {str(e)}"
        )