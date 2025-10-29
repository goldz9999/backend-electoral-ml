# app/services/data_cleaning.py
from datetime import datetime
from app.config.settings import supabase_client
import pandas as pd
import numpy as np
from typing import Dict, List


class DataCleaningService:
    """
    Servicio para limpieza y análisis de calidad de datos electorales.
    """
    
    @staticmethod
    async def analyze_data_quality() -> Dict:
        """
        Analiza la calidad de los datos de votos.
        Detecta: datos nulos, duplicados, outliers, emails inválidos.
        """
        try:
            # Obtener todos los votos
            votes_result = supabase_client.table("votes").select("*").execute()
            votes = votes_result.data
            
            if not votes:
                return {
                    "success": False,
                    "message": "No hay votos para analizar",
                    "total_records": 0
                }
            
            # Convertir a DataFrame para análisis
            df = pd.DataFrame(votes)
            
            # === ANÁLISIS DE CALIDAD ===
            total_records = len(df)
            
            # 1. Registros completos (sin campos nulos críticos)
            critical_fields = ['voter_name', 'voter_email', 'candidate_id']
            complete_records = df[critical_fields].notna().all(axis=1).sum()
            
            # 2. Datos faltantes
            missing_location = df['voter_location'].isna().sum()
            
            # 3. Emails válidos (contienen @)
            valid_emails = df['voter_email'].str.contains('@', na=False).sum()
            
            # 4. Duplicados por email
            duplicates = df.duplicated(subset=['voter_email'], keep=False).sum()
            
            # 5. Outliers (simulado: votos en horas inusuales)
            # ← FIX: Usar format='mixed' para manejar diferentes formatos
            df['voted_at'] = pd.to_datetime(df['voted_at'], format='mixed', errors='coerce')
            df['hour'] = df['voted_at'].dt.hour
            outliers = ((df['hour'] < 6) | (df['hour'] > 23)).sum()
            
            quality_report = {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "total_records": total_records,
                "complete_records": int(complete_records),
                "missing_data": int(missing_location),
                "valid_emails": int(valid_emails),
                "duplicates": int(duplicates),
                "outliers": int(outliers),
                "quality_score": round((complete_records / total_records) * 100, 2) if total_records > 0 else 0
            }
            
            # Guardar en tabla data_cleaning_summary
            await DataCleaningService._save_cleaning_summary(quality_report)
            
            return quality_report
            
        except Exception as e:
            print(f"Error en analyze_data_quality: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def clean_null_data() -> Dict:
        """
        Limpia votos con datos nulos críticos.
        PRIMERO guarda en null_data_votes, DESPUÉS elimina de votes.
        """
        try:
            # Obtener votos con campos nulos
            votes_result = supabase_client.table("votes").select("*").execute()
            df = pd.DataFrame(votes_result.data)
            
            if df.empty:
                return {
                    "success": True,
                    "cleaned_count": 0,
                    "message": "No hay votos para limpiar"
                }
            
            # Identificar registros con problemas
            null_voter_name = df['voter_name'].isna() | (df['voter_name'].astype(str).str.strip() == '')
            null_voter_email = df['voter_email'].isna() | (df['voter_email'].astype(str).str.strip() == '')
            null_candidate = df['candidate_id'].isna()
            
            problematic = null_voter_name | null_voter_email | null_candidate
            problematic_ids = df[problematic]['id'].tolist()
            
            cleaned_count = len(problematic_ids)
            
            if cleaned_count > 0:
                # PASO 1: Registrar en tabla null_data_votes (PRIMERO)
                for vote_id in problematic_ids:
                    missing_fields = []
                    vote = df[df['id'] == vote_id].iloc[0]
                    
                    if pd.isna(vote['voter_name']) or str(vote['voter_name']).strip() == '':
                        missing_fields.append('voter_name')
                    if pd.isna(vote['voter_email']) or str(vote['voter_email']).strip() == '':
                        missing_fields.append('voter_email')
                    if pd.isna(vote['candidate_id']):
                        missing_fields.append('candidate_id')
                    
                    supabase_client.table("null_data_votes").insert({
                        "vote_id": vote_id,
                        "missing_fields": missing_fields,
                        "voter_name": vote['voter_name'] if pd.notna(vote['voter_name']) else None,
                        "voter_email": vote['voter_email'] if pd.notna(vote['voter_email']) else None,
                        "resolved": True
                    }).execute()
                
                # PASO 2: Eliminar de la tabla votes (DESPUÉS)
                for vote_id in problematic_ids:
                    supabase_client.table("votes").delete().eq("id", vote_id).execute()
            
            return {
                "success": True,
                "cleaned_count": cleaned_count,
                "message": f"Se eliminaron {cleaned_count} votos con datos nulos (guardados en auditoría)"
            }
            
        except Exception as e:
            print(f"Error en clean_null_data: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def remove_duplicates() -> Dict:
        """
        Detecta y elimina votos duplicados por email.
        PRIMERO guarda en duplicated_votes, DESPUÉS elimina de votes.
        Mantiene solo el primer voto de cada email.
        """
        try:
            votes_result = supabase_client.table("votes").select("*").execute()
            df = pd.DataFrame(votes_result.data)
            
            if df.empty:
                return {
                    "success": True,
                    "duplicate_count": 0,
                    "message": "No hay votos para procesar"
                }
            
            # ← FIX: Convertir voted_at con format='mixed'
            df['voted_at'] = pd.to_datetime(df['voted_at'], format='mixed', errors='coerce')
            
            # Ordenar por fecha para mantener el primer voto
            df = df.sort_values('voted_at')
            
            # Encontrar duplicados (mantener el primero con keep='first')
            duplicates = df[df.duplicated(subset=['voter_email'], keep='first')]
            duplicate_ids = duplicates['id'].tolist()
            duplicate_emails = df[df.duplicated(subset=['voter_email'], keep=False)]['voter_email'].unique()
            
            duplicate_count = len(duplicate_emails)
            
            if duplicate_count > 0:
                # PASO 1: Registrar en duplicated_votes (PRIMERO)
                for email in duplicate_emails:
                    email_votes = df[df['voter_email'] == email].sort_values('voted_at')
                    first_vote = email_votes.iloc[0]['voted_at']
                    last_vote = email_votes.iloc[-1]['voted_at']
                    
                    # Obtener DNI del primer voto
                    voter_dni = email_votes.iloc[0].get('dni', 'UNKNOWN')
                    
                    # Convertir timestamps a string ISO format
                    first_vote_str = first_vote.isoformat() if pd.notna(first_vote) else None
                    last_vote_str = last_vote.isoformat() if pd.notna(last_vote) else None
                    
                    supabase_client.table("duplicated_votes").insert({
                        "voter_dni": voter_dni if pd.notna(voter_dni) else "UNKNOWN",
                        "voter_email": email,
                        "duplicate_count": len(email_votes),
                        "first_vote_at": first_vote_str,
                        "last_vote_at": last_vote_str,
                        "resolved": True
                    }).execute()
                
                # PASO 2: Eliminar duplicados de votes (DESPUÉS)
                for vote_id in duplicate_ids:
                    supabase_client.table("votes").delete().eq("id", vote_id).execute()
            
            return {
                "success": True,
                "duplicate_count": duplicate_count,
                "total_removed": len(duplicate_ids),
                "message": f"Se eliminaron {len(duplicate_ids)} votos duplicados de {duplicate_count} emails (guardados en auditoría)"
            }
            
        except Exception as e:
            print(f"Error en remove_duplicates: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def normalize_data() -> Dict:
        """
        Normaliza texto: capitaliza nombres, limpia espacios, estandariza ubicaciones.
        PRIMERO guarda en unnormalized_text_votes, DESPUÉS actualiza votes.
        """
        try:
            votes_result = supabase_client.table("votes").select("*").execute()
            df = pd.DataFrame(votes_result.data)
            
            if df.empty:
                return {
                    "success": True,
                    "normalized_count": 0,
                    "message": "No hay votos para normalizar"
                }
            
            normalized_count = 0
            
            for idx, row in df.iterrows():
                original_name = row['voter_name']
                original_location = row['voter_location']
                
                # Normalizar nombre
                normalized_name = str(original_name).strip().title() if pd.notna(original_name) else original_name
                
                # Normalizar ubicación
                if pd.notna(original_location):
                    normalized_location = str(original_location).strip().title()
                else:
                    normalized_location = original_location
                
                # Si hubo cambios, actualizar
                if normalized_name != original_name or normalized_location != original_location:
                    # PASO 1: Registrar cambio en unnormalized_text_votes (PRIMERO)
                    supabase_client.table("unnormalized_text_votes").insert({
                        "vote_id": row['id'],
                        "field_name": "voter_name/location",
                        "original_value": f"{original_name} | {original_location}",
                        "normalized_value": f"{normalized_name} | {normalized_location}",
                        "applied": True
                    }).execute()
                    
                    # PASO 2: Actualizar en votes (DESPUÉS)
                    supabase_client.table("votes").update({
                        "voter_name": normalized_name,
                        "voter_location": normalized_location
                    }).eq("id", row['id']).execute()
                    
                    normalized_count += 1
            
            return {
                "success": True,
                "normalized_count": normalized_count,
                "message": f"Se normalizaron {normalized_count} registros (cambios guardados en auditoría)"
            }
            
        except Exception as e:
            print(f"Error en normalize_data: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def _save_cleaning_summary(quality_report: Dict):
        """Guarda resumen de limpieza en la base de datos."""
        try:
            supabase_client.table("data_cleaning_summary").insert({
                "total_votes": quality_report.get("total_records", 0),
                "valid_votes": quality_report.get("complete_records", 0),
                "duplicates_found": quality_report.get("duplicates", 0),
                "null_data_found": quality_report.get("missing_data", 0),
                "unnormalized_found": quality_report.get("outliers", 0),
                "cleaned_votes": quality_report.get("complete_records", 0),
                "execution_time_ms": 1500
            }).execute()
        except Exception as e:
            print(f"Error guardando resumen: {e}")