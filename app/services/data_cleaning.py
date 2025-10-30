# app/services/data_cleaning.py
from datetime import datetime
from app.config.settings import supabase_client
import pandas as pd
import numpy as np
from typing import Dict, List

def log_action(action: str, table: str, details: dict = None):
    """Registra acción en audit_logs"""
    try:
        supabase_client.table("audit_logs").insert({
            "user_id": 1,  # Admin (puedes parametrizarlo después)
            "action": action,
            "table_name": table,
            "new_values": details,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        print(f"📝 Audit log: {action} en {table}")
    except Exception as e:
        print(f"⚠️ Error guardando audit log: {e}")

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
            votes_result = supabase_client.table("votes").select("*").execute()
            votes = votes_result.data
            
            if not votes:
                return {
                    "success": False,
                    "message": "No hay votos para analizar",
                    "total_records": 0
                }
            
            df = pd.DataFrame(votes)
            total_records = len(df)
            
            # 1. Registros completos (sin campos nulos críticos, incluyendo 'N/A')
            critical_fields = ['voter_name', 'voter_email', 'voter_dni', 'voter_location']
            complete_records = 0
            for _, row in df.iterrows():
                is_complete = True
                for field in critical_fields:
                    val = row.get(field)
                    if pd.isna(val) or str(val).strip() == '' or str(val).strip().upper() == 'N/A':
                        is_complete = False
                        break
                if is_complete:
                    complete_records += 1
            
            # 2. Datos faltantes o con 'N/A'
            missing_data = 0
            for field in critical_fields:
                if field in df.columns:
                    missing_data += df[field].apply(
                        lambda x: pd.isna(x) or str(x).strip() == '' or str(x).strip().upper() == 'N/A'
                    ).sum()
            
            # 3. Emails válidos (contienen @ y no son N/A)
            valid_emails = df['voter_email'].apply(
                lambda x: '@' in str(x) and str(x).strip().upper() != 'N/A'
            ).sum()
            
            # 4. Duplicados por email (excluyendo N/A)
            df_filtered = df[df['voter_email'].apply(
                lambda x: pd.notna(x) and str(x).strip().upper() != 'N/A'
            )]
            duplicates = df_filtered.duplicated(subset=['voter_email'], keep=False).sum()
            
            # 5. Outliers
            df['voted_at'] = pd.to_datetime(df['voted_at'], format='mixed', errors='coerce')
            df['hour'] = df['voted_at'].dt.hour
            outliers = ((df['hour'] < 6) | (df['hour'] > 23)).sum()
            
            quality_report = {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "total_records": total_records,
                "complete_records": int(complete_records),
                "missing_data": int(missing_data),
                "valid_emails": int(valid_emails),
                "duplicates": int(duplicates),
                "outliers": int(outliers),
                "quality_score": round((complete_records / total_records) * 100, 2) if total_records > 0 else 0
            }
            
            await DataCleaningService._save_cleaning_summary(quality_report)
            log_action(
            action="ANALYZE_DATA_QUALITY",
            table="votes",
            details={
                "total_records": quality_report.get("total_records", 0),
                "quality_score": quality_report.get("quality_score", 0)
                }
            )
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
        Reemplaza datos nulos con 'N/A'.
        INCLUYE: voter_name, voter_email, voter_dni, voter_location
        """
        try:
            votes_result = supabase_client.table("votes").select("*").execute()
            df = pd.DataFrame(votes_result.data)
            
            if df.empty:
                return {
                    "success": True,
                    "cleaned_count": 0,
                    "message": "No hay votos para limpiar"
                }
            
            # Identificar registros con problemas
            problematic_mask = False
            
            # Verificar cada campo crítico
            if 'voter_name' in df.columns:
                null_voter_name = df['voter_name'].isna() | (df['voter_name'].astype(str).str.strip() == '')
                problematic_mask = problematic_mask | null_voter_name
            
            if 'voter_email' in df.columns:
                null_voter_email = df['voter_email'].isna() | (df['voter_email'].astype(str).str.strip() == '')
                problematic_mask = problematic_mask | null_voter_email
            
            if 'voter_dni' in df.columns:
                null_voter_dni = df['voter_dni'].isna() | (df['voter_dni'].astype(str).str.strip() == '')
                problematic_mask = problematic_mask | null_voter_dni
            
            if 'voter_location' in df.columns:
                null_voter_location = df['voter_location'].isna() | (df['voter_location'].astype(str).str.strip() == '')
                problematic_mask = problematic_mask | null_voter_location
            
            problematic_ids = df[problematic_mask]['id'].tolist()
            cleaned_count = len(problematic_ids)
            
            if cleaned_count > 0:
                # PASO 1: Registrar en tabla null_data_votes
                for vote_id in problematic_ids:
                    missing_fields = []
                    vote = df[df['id'] == vote_id].iloc[0]
                    
                    if 'voter_name' in df.columns and (pd.isna(vote['voter_name']) or str(vote['voter_name']).strip() == ''):
                        missing_fields.append('voter_name')
                    
                    if 'voter_email' in df.columns and (pd.isna(vote['voter_email']) or str(vote['voter_email']).strip() == ''):
                        missing_fields.append('voter_email')
                    
                    if 'voter_dni' in df.columns and (pd.isna(vote['voter_dni']) or str(vote['voter_dni']).strip() == ''):
                        missing_fields.append('voter_dni')
                    
                    if 'voter_location' in df.columns and (pd.isna(vote['voter_location']) or str(vote['voter_location']).strip() == ''):
                        missing_fields.append('voter_location')
                    
                    supabase_client.table("null_data_votes").insert({
                        "vote_id": vote_id,
                        "missing_fields": missing_fields,
                        "voter_name": vote.get('voter_name') if pd.notna(vote.get('voter_name')) else None,
                        "voter_email": vote.get('voter_email') if pd.notna(vote.get('voter_email')) else None,
                        "resolved": True
                    }).execute()
                
                # PASO 2: Actualizar votos con 'N/A' (NO ELIMINAR)
                for vote_id in problematic_ids:
                    vote = df[df['id'] == vote_id].iloc[0]
                    
                    update_data = {}
                    
                    if 'voter_name' in df.columns and (pd.isna(vote['voter_name']) or str(vote['voter_name']).strip() == ''):
                        update_data['voter_name'] = 'N/A'
                    
                    if 'voter_email' in df.columns and (pd.isna(vote['voter_email']) or str(vote['voter_email']).strip() == ''):
                        update_data['voter_email'] = 'N/A'
                    
                    if 'voter_dni' in df.columns and (pd.isna(vote['voter_dni']) or str(vote['voter_dni']).strip() == ''):
                        update_data['voter_dni'] = 'N/A'
                    
                    if 'voter_location' in df.columns and (pd.isna(vote['voter_location']) or str(vote['voter_location']).strip() == ''):
                        update_data['voter_location'] = 'N/A'
                    
                    if update_data:
                        supabase_client.table("votes").update(update_data).eq("id", vote_id).execute()
            if cleaned_count > 0:
                log_action(
                    action="CLEAN_NULL_DATA",
                    table="votes",
                    details={
                        "cleaned_count": cleaned_count,
                        "fields_affected": ["voter_name", "voter_email", "voter_dni", "voter_location"]
                    }
                )
            return {
                "success": True,
                "cleaned_count": cleaned_count,
                "message": f"Se reemplazaron {cleaned_count} registros con datos nulos por 'N/A' (guardados en auditoría)"
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
        Detecta y elimina votos duplicados por email (excluyendo N/A).
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
            
            # Filtrar emails válidos (no N/A)
            df_filtered = df[df['voter_email'].apply(
                lambda x: pd.notna(x) and str(x).strip().upper() != 'N/A'
            )].copy()
            
            if df_filtered.empty:
                return {
                    "success": True,
                    "duplicate_count": 0,
                    "message": "No hay emails válidos para verificar duplicados"
                }
            
            df_filtered['voted_at'] = pd.to_datetime(df_filtered['voted_at'], format='mixed', errors='coerce')
            df_filtered = df_filtered.sort_values('voted_at')
            
            duplicates = df_filtered[df_filtered.duplicated(subset=['voter_email'], keep='first')]
            duplicate_ids = duplicates['id'].tolist()
            duplicate_emails = df_filtered[df_filtered.duplicated(subset=['voter_email'], keep=False)]['voter_email'].unique()
            
            duplicate_count = len(duplicate_emails)
            
            if duplicate_count > 0:
                for email in duplicate_emails:
                    email_votes = df_filtered[df_filtered['voter_email'] == email].sort_values('voted_at')
                    first_vote = email_votes.iloc[0]['voted_at']
                    last_vote = email_votes.iloc[-1]['voted_at']
                    voter_dni = email_votes.iloc[0].get('voter_dni', 'UNKNOWN')
                    
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
                
                for vote_id in duplicate_ids:
                    supabase_client.table("votes").delete().eq("id", vote_id).execute()
            if duplicate_count > 0:
                log_action(
                    action="REMOVE_DUPLICATES",
                    table="votes",
                    details={
                        "duplicate_emails": duplicate_count,
                        "votes_removed": len(duplicate_ids)
                    }
                )
            return {
                "success": True,
                "duplicate_count": duplicate_count,
                "total_removed": len(duplicate_ids),
                "message": f"Se eliminaron {len(duplicate_ids)} votos duplicados de {duplicate_count} emails"
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
        Normaliza texto: capitaliza nombres, limpia espacios (excepto 'N/A').
        INCLUYE: voter_name, voter_location, voter_dni
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
                original_name = row.get('voter_name')
                original_location = row.get('voter_location')
                original_dni = row.get('voter_dni')
                
                changes = {}
                
                # Normalizar nombre (no normalizar si es 'N/A')
                if 'voter_name' in df.columns:
                    if pd.notna(original_name) and str(original_name).strip().upper() != 'N/A':
                        normalized_name = str(original_name).strip().title()
                        if normalized_name != original_name:
                            changes['voter_name'] = normalized_name
                
                # Normalizar ubicación
                if 'voter_location' in df.columns:
                    if pd.notna(original_location) and str(original_location).strip().upper() != 'N/A':
                        normalized_location = str(original_location).strip().title()
                        if normalized_location != original_location:
                            changes['voter_location'] = normalized_location
                
                # Normalizar DNI (solo limpiar espacios, no capitalizar)
                if 'voter_dni' in df.columns:
                    if pd.notna(original_dni) and str(original_dni).strip().upper() != 'N/A':
                        normalized_dni = str(original_dni).strip()
                        if normalized_dni != original_dni:
                            changes['voter_dni'] = normalized_dni
                
                if changes:
                    # Registrar en tabla de auditoría
                    supabase_client.table("unnormalized_text_votes").insert({
                        "vote_id": row['id'],
                        "field_name": ", ".join(changes.keys()),
                        "original_value": f"name: {original_name} | location: {original_location} | dni: {original_dni}",
                        "normalized_value": f"name: {changes.get('voter_name', original_name)} | location: {changes.get('voter_location', original_location)} | dni: {changes.get('voter_dni', original_dni)}",
                        "applied": True
                    }).execute()
                    
                    # Actualizar en votes
                    supabase_client.table("votes").update(changes).eq("id", row['id']).execute()
                    normalized_count += 1
            if normalized_count > 0:
                log_action(
                    action="NORMALIZE_DATA",
                    table="votes",
                    details={
                        "normalized_count": normalized_count,
                        "fields_normalized": ["voter_name", "voter_location", "voter_dni"]
                    }
                )
            return {
                "success": True,
                "normalized_count": normalized_count,
                "message": f"Se normalizaron {normalized_count} registros"
            }
            
        except Exception as e:
            print(f"Error en normalize_data: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def _save_cleaning_summary(quality_report: Dict):
        """Guarda resumen de limpieza."""
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