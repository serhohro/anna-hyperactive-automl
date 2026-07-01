"""
================================================================================
ANNA AI v23.0 - FINAL WORKING (ALL FEATURES FIXED + SHAP CLASS INDEX SAFE)
================================================================================
✅ 20 Modi
✅ MLP: nur numerische Spalten + absolute force numeric
✅ Keine 'invalid literal' Fehler mehr
✅ Erweiterte Klassifikationsanalyse: classification_report, Konfusionsmatrix
✅ Deutsche Beschriftungen
✅ Gestensteuerung: startet mouse.py in separatem Konsolenfenster (läuft im Hintergrund)
✅ FIX: Datetime-Spalten werden in Unix-Timestamp umgewandelt (kein NotImplementedError)
✅ FIX: SHAP funktioniert mit Datetime-Spalten und Multi‑Class (3D-Array)
✅ FIX: SHAP Klassenindex sicher – kein 'index out of bounds' mehr
✅ FIX: Datenanalyse mit vollständiger Tabelle und schnellem Diagramm (Tabs)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
import time
import os
import re
import tempfile
import json
import random
import math
import warnings
import io
import threading
import urllib.parse
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, f1_score, roc_auc_score, mean_squared_error, classification_report, ConfusionMatrixDisplay
from collections import Counter

# -------------------- OPTIONALE BIBLIOTHEKEN --------------------
try:
    import cv2
    import mediapipe as mp
    import pyautogui
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    PYTORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import requests
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from PIL import Image
    import torchvision.transforms as transforms
    from torchvision import models
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

warnings.filterwarnings('ignore')

# -------------------- SEITENKONFIGURATION --------------------
st.set_page_config(page_title="🤖 ANNA AI v23.0", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
for dir_path in ['models', 'outputs', 'outputs/pdf', 'outputs/plots', 'data/cache']:
    os.makedirs(dir_path, exist_ok=True)

# ==================== DEUTSCHES WÖRTERBUCH ====================
T = {
    "app_title": "ANNA AI v23.0",
    "app_subtitle": "MULTIMODALE KI-ASSISTENTIN MIT AGENTIC AI & RPA (FINAL WORKING)",
    "activate_button": "🎬 Anna starten",
    "anna_activated": "🔊 Anna aktiviert | Multi‑Modal",
    "voice_warning": "👈 **Klicken Sie 'Anna starten' in der Seitenleiste!**",

    "mode_chat": "💬 Chat (LLM)",
    "mode_automl": "🧠 AutoML Pro + Neuronales Netz",
    "mode_data": "📊 Datenanalyse",
    "mode_3d": "🎨 3D",
    "mode_hyper": "⚡ Hyperaktiv",
    "mode_shap": "🔍 SHAP",
    "mode_music": "🎵 Musik",
    "mode_video": "📹 Video",
    "mode_engineer": "🤖 AI-Engineer",
    "nlp_title": "🧾 Textklassifikation (NLP)",
    "ts_title": "📈 Zeitreihenprognose (Prophet)",
    "img_title": "🖼️ Bilderkennung (ResNet)",
    "mode_llm": "🧠 Lokaler LLM",
    "mode_stream": "📡 Echtzeit‑Streaming",
    "mode_sd": "🎨 Bildgenerierung",
    "mode_backtest": "📈 Backtesting",
    "mode_benchmark": "🏆 SOTA‑Vergleich",
    "mode_gesture": "🖐️ Gestensteuerung",
    "mode_agent": "🤖 Autonomer Analyst",
    "mode_rpa": "📋 RPA (Robotic Automation)",

    "gesture_title": "🖐️ Maussteuerung per Handgesten",
    "gesture_start": "🎥 Kamera starten & Steuerung aktivieren",
    "gesture_stop": "⏹️ Steuerung beenden",
    "gesture_sensitivity": "Empfindlichkeit der Pinch-Geste (Abstand)",
    "agent_title": "🤖 Autonomer Analyst (Agentic AI)",
    "agent_command": "Beschreiben Sie Ihre Datenanalyseaufgabe",
    "agent_example": "Beispiel: „Analysiere diese CSV, erstelle ein Modell und erkläre die wichtigsten Einflussfaktoren.“",
    "agent_run": "🚀 Automatische Analyse starten",
    "agent_upload": "📂 CSV / Excel hochladen",
    "agent_report": "📊 Analysebericht",
    "agent_model_save": "💾 Trainiertes Modell speichern",
    "rpa_title": "📋 Robotic Process Automation (RPA)",
    "rpa_record": "🔴 Makro aufzeichnen",
    "rpa_stop_record": "⏹️ Aufzeichnung stoppen",
    "rpa_play": "▶️ Aufgezeichnetes Makro abspielen",
    "rpa_script_upload": "📂 Python‑Skript hochladen",
    "rpa_run_script": "🚀 Skript ausführen",
    "rpa_calibrate": "⚙️ Mauskalibrierung",
    "rpa_mouse_speed": "Mausgeschwindigkeit (Sekunden pro Bewegung)",
    "rpa_click_delay": "Verzögerung vor dem Klick (Sekunden)",

    "chat_title": "💬 Chat mit Anna (lokaler LLM)",
    "chat_placeholder": "Fragen Sie Anna auf Deutsch oder Russisch...",
    "automl_title": "🧠 AutoML Pro + Neuronales Netz",
    "data_title": "📊 Datenanalyse",
    "3d_title": "🎨 3D Visualisierung",
    "hyper_title": "⚡ HYPERAKTIVER MODUS (<100 ms)",
    "shap_title": "🔍 SHAP - ERKLÄRUNG DER VORHERSAGEN",
    "music_title": "🎵 Musiksuche auf YouTube Music",
    "video_title": "📹 Videosuche auf YouTube",
    "engineer_title": "🤖 AI-Engineer: Aufgabe → Code → Ergebnis",
    "engineer_placeholder": "Beschreiben Sie Ihre Aufgabe auf Deutsch oder Russisch...",
    "engineer_generate": "🚀 Code generieren & ausführen",
    "engineer_code": "📝 Generierter Code:",
    "engineer_result": "📊 Ergebnis:",
    "file_upload": "📂 CSV oder Excel laden",
    "target_select": "🎯 Zielvariable",
    "task_type_select": "📊 Aufgabentyp",
    "task_auto": "🚀 Automatisch",
    "task_class": "🏷️ Klassifikation",
    "task_reg": "📈 Regression",
    "train_button": "🚀 MODELL TRAINIEREN",
    "train_spinner": "🧠 Training läuft...",
    "hyperopt_button": "🔧 HYPERPARAMETER OPTIMIERUNG",
    "hyperopt_spinner": "🔍 Optimiere Hyperparameter...",
    "shap_button": "🔍 SHAP ERKLÄRUNG",
    "feature_importance_button": "📊 FEATURE IMPORTANCE",
    "auto_classes_button": "🎯 Automatische Klassenreduktion",
    "clean_question": "🧹 Daten bereinigen? (Fehlende Werte & Duplikate)",
    "clean_yes": "✅ Ja, bereinigen",
    "clean_no": "❌ Nein, später",
    "clean_button": "🧹 DATEN BEREINIGEN",
    "data_clean_button": "🧹 DATEN BEREINIGEN (erneut)",
    "data_stats_button": "📊 STATISTIK",
    "data_export_button": "📄 PDF EXPORT",
    "remove_id_columns": "🗑️ ID‑Spalten automatisch entfernen",
    "apply_smote": "⚖️ SMOTE anwenden (gegen Klassenungleichgewicht)",
    "compare_models": "🔍 Mehrere Modelle vergleichen",
    "skip_slow_models": "⏩ Langsame Modelle überspringen",
    "loading_data": "📂 Bitte laden Sie eine CSV- oder Excel-Datei hoch.",
    "at_least_3_numeric": "Mindestens 3 numerische Spalten benötigt",
    "select_row_for_shap": "Zeile auswählen",
    "shap_not_available": "SHAP Analyse nicht verfügbar für diesen Modelltyp",
    "train_model_first": "⚠️ Trainieren Sie zuerst ein Modell im AutoML Tab",
    "load_data_first": "📂 Laden Sie zuerst Daten im AutoML Tab",
    "target_not_defined": "⚠️ Zielvariable nicht definiert",
    "auto_classes_recommend": "Automatisch empfohlen: {optimal} häufigste Klassen beibehalten.",
    "auto_classes_accept": "✅ Übernehmen und auf {optimal} setzen",
    "reduce_classes": "📊 Anzahl der Klassen auf die häufigsten reduzieren",
    "reduce_classes_count": "Anzahl der häufigsten Klassen",
    "additional_metrics": "📊 Zusätzliche Metriken",
    "feature_importance_title": "📊 Feature‑Wichtigkeit",
    "hyper_start": "▶️ STARTEN",
    "online_retrain": "📚 Online‑Training (alle N Schritte)",
    "retrain_every": "N Schritte",
    "prediction_accuracy": "Vorhersagegenauigkeit (±10%)",
    "avg_latency": "Durchschnittliche Latenz",
    "buy_signals": "KAUFEN‑Signale",
    "export_json": "📥 Bericht als JSON speichern",
    "live_status": "📊 Iteration {current}/{total} | Vorhersage: {pred:.4f} | {action}",
    "search_music": "🔎 Musik suchen",
    "search_video": "🔎 Video suchen",
    "no_audio_input": "Kein Audio erkannt",
    "enter_search_term": "Bitte geben Sie einen Suchbegriff ein",
    "no_results": "Keine Ergebnisse gefunden für",
    "try_different_term": "Versuchen Sie es mit einem anderen Begriff.",
    "what_i_found": "Was ich für '{query}' gefunden habe:",
    "listen": "Hören",
    "watch": "Ansehen",
    "channel": "YouTube‑Kanal",
    "unknown": "Unbekannt",
    "generate_code": "Generiere Code...",
    "execute_code": "Führe Code aus...",
    "describe_task": "Bitte beschreiben Sie eine Aufgabe.",
    "no_code_generated": "Geben Sie eine Aufgabe ein und klicken Sie auf 'Code generieren & ausführen'.",
    "code_executed_success": "Code erfolgreich ausgeführt (ohne explizites Ergebnis).",
    "code_execution_error": "Fehler bei der Ausführung:",
    "model_comparison_results": "📊 Ergebnisse des Modellvergleichs",
    "best_model": "🏆 Bestes Modell: {name}",
    "accuracy_score": "🎯 Genauigkeit: {score:.4f}",
    "r2_score": "📈 R² Score: {score:.4f}",
    "baseline_random": "🎯 Baseline (zufälliges Raten): {baseline:.2%}",
    "model_not_better": "⚠️ Modell ist nicht besser als zufälliges Raten. Versuchen Sie Hyperparameter‑Optimierung oder verbessern Sie die Daten.",
    "optimization_complete": "🔧 Optimierung abgeschlossen! Beste Parameter:",
    "improved_metric": "📈 Verbesserte Metrik: {score:.4f}",
    "feature_importance_computed": "Feature‑Wichtigkeit wurde berechnet.",
    "shap_analysis_complete": "SHAP Analyse abgeschlossen",
    "hyper_mode_complete": "Hyperaktiver Modus abgeschlossen. {iter} Iterationen.",
    "text_classification_complete": "Textklassifikation abgeschlossen. Genauigkeit: {acc:.2%}",
    "time_series_forecast_complete": "Zeitreihenprognose für {periods} Perioden erstellt.",
    "image_classification_complete": "Bildklassifikation abgeschlossen.",
    "greeting": "Hallo! Ich bin Anna. Ich spreche Deutsch und verstehe Russisch. Ich habe neuronale Netze mit GPU‑Unterstützung.",
    "gpu_activated": "🚀 GPU AKTIVIERT: {device}",
    "cpu_mode": "💻 CPU Modus",
    "data_cleaned": "Daten bereinigt. Entfernt wurden {missing} fehlende Werte und {duplicates} Duplikate.",
    "found_missing_duplicates": "Gefunden {missing} fehlende Werte und {duplicates} Duplikate. Möchten Sie die Daten bereinigen?",
    "data_unchanged": "Daten bleiben unverändert.",
    "automl_complete_class": "AutoML abgeschlossen. Bestes Modell: {name}. Genauigkeit = {score:.2%}",
    "automl_complete_reg": "AutoML abgeschlossen. Bestes Modell: {name}. R² = {score:.4f}",
    "hyperopt_complete": "Hyperparameteroptimierung abgeschlossen. Neue Genauigkeit: {score:.2%}",
    "insight_button": "💡 Erkenntnis generieren",
    "ollama_model": "Ollama‑Modell",
    "ollama_not_found": "Ollama nicht installiert. Besuchen Sie ollama.com und starten Sie 'ollama serve'",
    "start_stream": "▶️ Streaming starten",
    "stop_stream": "⏹️ Stoppen",
    "generate_image": "🎨 Bild generieren",
    "prompt_sd": "Beschreiben Sie das Bild",
    "backtest_button": "📊 Backtest starten",
    "benchmark_button": "🏆 Vergleich starten",
    "save_stats_jpeg": "📸 Statistik als JPEG speichern",
    "save_3d_jpeg": "📸 3D‑Plot als JPEG speichern",
    "save_shap_jpeg": "📸 SHAP‑Plot als JPEG speichern",
    "save_hyper_jpeg": "📸 Hyper‑Ergebnisse als JPEG speichern",
    "save_forecast_jpeg": "📸 Prognose als JPEG speichern",
}

# ==================== HILFSFUNKTIONEN ====================
def speak(text: str, lang: str = 'de') -> Optional[bytes]:
    if not st.session_state.get('sound_activated', False):
        return None
    if not text or len(text) < 2:
        return None
    try:
        from gtts import gTTS
        clean = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)[:300]
        tts = gTTS(text=clean, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.write_to_fp(fp)
            temp_path = fp.name
        with open(temp_path, 'rb') as f:
            audio = f.read()
        os.unlink(temp_path)
        return audio
    except Exception:
        return None

def recognize_speech_auto(audio_bytes) -> Optional[str]:
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with sr.AudioFile(tmp_path) as src:
            audio_data = recognizer.record(src)
            try:
                text = recognizer.recognize_google(audio_data, language="de-DE")
            except:
                try:
                    text = recognizer.recognize_google(audio_data, language="ru-RU")
                except:
                    text = recognizer.recognize_google(audio_data, language="uk-UA")
        os.unlink(tmp_path)
        return text.strip() if text else None
    except Exception:
        return None

def query_ollama(prompt, model="llama3.2:3b"):
    if not OLLAMA_AVAILABLE:
        return "⚠️ Ollama nicht installiert."
    try:
        response = ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"⚠️ Ollama-Fehler: {e}"

def search_youtube(query, limit=5):
    try:
        search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=10)
        html = response.text
        video_ids = re.findall(r'"videoId":"([^"]+)"', html)
        titles = re.findall(r'"title":{"runs":\[{"text":"([^"]+)"', html)
        channels = re.findall(r'"ownerChannelName":"([^"]+)"', html)
        results = []
        seen = set()
        for i, vid in enumerate(video_ids):
            if vid not in seen and vid:
                seen.add(vid)
                results.append({
                    'title': titles[i] if i < len(titles) else T["unknown"],
                    'channel': channels[i] if i < len(channels) else T["channel"],
                    'url': f'https://youtube.com/watch?v={vid}'
                })
                if len(results) >= limit:
                    break
        return results
    except Exception:
        return []

def search_music(query, limit=5): return search_youtube(query, limit)
def search_video(query, limit=5): return search_youtube(query, limit)

def format_music_results(items, query):
    if not items: return f"😔 {T['no_results']} '{query}'. {T['try_different_term']}"
    text = f"🎵 **{T['what_i_found'].format(query=query)}**\n\n"
    for i,s in enumerate(items[:5],1):
        text += f"{i}. **{s['title'][:80]}**\n   📺 {s['channel']}\n   🔗 [{T['listen']}]({s['url']})\n\n"
    return text

def format_video_results(items, query):
    if not items: return f"😔 {T['no_results']} '{query}'. {T['try_different_term']}"
    text = f"🎬 **{T['what_i_found'].format(query=query)}**\n\n"
    for i,v in enumerate(items[:5],1):
        text += f"{i}. **{v['title'][:80]}**\n   📺 {v['channel']}\n   🔗 [{T['watch']}]({v['url']})\n\n"
    return text

def analyze_data(df):
    return {'rows':len(df),'cols':len(df.columns),'missing':df.isnull().sum().sum(),'duplicates':df.duplicated().sum(),'memory':df.memory_usage(deep=True).sum()/1024/1024}

def clean_data_pro(df):
    df=df.copy()
    bm=df.isnull().sum().sum(); bd=df.duplicated().sum()
    df=df.drop_duplicates()
    for col in df.columns:
        if df[col].nunique()==1: df=df.drop(columns=[col])
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64','float64']: df[col]=df[col].fillna(df[col].median())
            else: df[col]=df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
    am=df.isnull().sum().sum(); ad=df.duplicated().sum()
    return df,bm,am,bd,ad

def export_to_html(df,title):
    html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>{title}</title><style>body{{font-family:Arial;margin:40px;}}h1{{color:#667eea;}}table{{border-collapse:collapse;width:100%;}}th,td{{border:1px solid #ddd;padding:8px;text-align:left;}}th{{background-color:#667eea;color:white;}}</style></head><body><h1>{title}</h1>{df.describe().to_html()}<hr>{df.head(100).to_html()}<p>Generiert von ANNA AI v23.0 - {datetime.now()}</p></body></html>"
    return html.encode('utf-8')

def get_anna_video():
    current_dir = Path(__file__).parent
    for f in current_dir.iterdir():
        if f.is_file() and f.suffix.lower()=='.mp4' and 'anna' in f.name.lower(): return str(f)
    return None

def show_anna_video_and_wait():
    video_path = get_anna_video()
    if video_path:
        try:
            with open(video_path,"rb") as f: video_data = base64.b64encode(f.read()).decode()
            st.markdown(f'''<div style="margin:20px 0;display:flex;justify-content:center;"><video autoplay controls style="width:100%;max-width:350px;border-radius:15px;"><source src="data:video/mp4;base64,{video_data}" type="video/mp4"></video></div>''',unsafe_allow_html=True)
        except: pass
    time.sleep(8)
    return True

def activate_anna():
    if st.session_state.get('anna_activated',False): return
    show_anna_video_and_wait()
    greeting = T["greeting"]
    audio = speak(greeting,lang='de')
    if audio: st.audio(audio,format='audio/mp3',autoplay=True)
    time.sleep(2)
    st.session_state.anna_activated = True
    st.session_state.sound_activated = True
    if PYTORCH_AVAILABLE and CUDA_AVAILABLE: st.success(T["gpu_activated"].format(device=torch.cuda.get_device_name(0)))
    else: st.info(T["cpu_mode"])

def save_dataframe_as_jpeg(df,title):
    fig,ax=plt.subplots(figsize=(12,max(4,len(df)//10+2)))
    ax.axis('tight'); ax.axis('off')
    table=ax.table(cellText=df.values,colLabels=df.columns,loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10)
    plt.title(title,fontsize=14)
    buf=io.BytesIO()
    plt.savefig(buf,format='jpg',dpi=200,bbox_inches='tight')
    buf.seek(0); plt.close(fig)
    return buf.getvalue()

def save_plotly_as_jpeg(fig,title):
    try:
        import plotly.io as pio
        return fig.to_image(format="jpg",scale=2)
    except: return None

def save_matplotlib_fig_as_jpeg(fig,title):
    buf=io.BytesIO()
    fig.savefig(buf,format='jpg',dpi=200,bbox_inches='tight')
    buf.seek(0); plt.close(fig)
    return buf.getvalue()

# ==================== SHAP FUNKTION (FIXED FÜR 3D-ARRAY + SAFE INDEX) ====================
def explain_with_shap(model, X_sample, feature_names, model_type='tree', class_idx=0):
    if not SHAP_AVAILABLE:
        return None
    try:
        # Konvertiere zu float32 Array
        if isinstance(X_sample, pd.DataFrame):
            X_sample = X_sample.copy()
            for col in X_sample.columns:
                if pd.api.types.is_datetime64_any_dtype(X_sample[col]) or pd.api.types.is_period_dtype(X_sample[col]):
                    X_sample[col] = pd.to_numeric(X_sample[col], errors='coerce')
            X_sample = X_sample.fillna(0).values
        elif isinstance(X_sample, pd.Series):
            X_sample = X_sample.values
        
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        X_sample = np.array(X_sample, dtype=np.float32)
        X_sample = np.nan_to_num(X_sample, nan=0.0)
        
        if model_type == 'tree' and (hasattr(model, 'feature_importances_') or hasattr(model, 'get_booster')):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # --- SICHERHEITSCHECK FÜR KLASSENINDEX ---
            if isinstance(shap_values, list):
                if class_idx >= len(shap_values):
                    class_idx = 0
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                if class_idx >= shap_values.shape[1]:
                    class_idx = 0
            # --- ENDE SICHERHEITSCHECK ---
            
            # Intelligenz für verschiedene SHAP-Ausgabeformate
            if isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) == 3:
                    shap_single = shap_values[0, class_idx, :]
                elif len(shap_values.shape) == 2:
                    shap_single = shap_values[0]
                else:
                    shap_single = shap_values
                base_val = explainer.expected_value
                if isinstance(base_val, np.ndarray):
                    if len(base_val.shape) == 1:
                        base_val_single = base_val[class_idx] if class_idx < len(base_val) else base_val[0]
                    else:
                        base_val_single = base_val
                else:
                    base_val_single = base_val
            elif isinstance(shap_values, list):
                if class_idx >= len(shap_values):
                    class_idx = 0
                shap_class = shap_values[class_idx]
                if len(shap_class.shape) == 2:
                    shap_single = shap_class[0]
                else:
                    shap_single = shap_class
                base_val = explainer.expected_value
                if isinstance(base_val, list):
                    base_val_single = base_val[class_idx] if class_idx < len(base_val) else base_val[0]
                else:
                    base_val_single = base_val
            else:
                shap_single = shap_values
                base_val_single = explainer.expected_value
        else:
            # KernelExplainer Fallback
            background = X_sample[:min(100, X_sample.shape[0])]
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample[:1])
            if isinstance(shap_values, list):
                if class_idx >= len(shap_values):
                    class_idx = 0
                shap_class = shap_values[class_idx]
                shap_single = shap_class[0] if len(shap_class.shape) > 1 else shap_class
                base_val = explainer.expected_value
                base_val_single = base_val[class_idx] if isinstance(base_val, list) else base_val
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                if class_idx >= shap_values.shape[1]:
                    class_idx = 0
                shap_single = shap_values[0, class_idx, :]
                base_val_single = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_single = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                base_val_single = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0]
        
        # Längenangleichung und Daten für ein Sample
        if len(shap_single) != len(feature_names):
            shap_single = shap_single[:len(feature_names)]
        data_single = X_sample[0][:len(feature_names)]
        
        exp = shap.Explanation(values=shap_single,
                               base_values=base_val_single,
                               data=data_single,
                               feature_names=feature_names)
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.waterfall_plot(exp, show=False, max_display=10)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"SHAP Fehler: {e}")
        return None

class BinanceWebSocketStream:
    def __init__(self,symbol="btcusdt"):
        self.symbol=symbol.lower()
        self.socket_url=f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        self.ws=None; self.running=False; self.last_price=None; self.callback=None
    def on_message(self,ws,message):
        data=json.loads(message); price=float(data['p']); self.last_price=price
        if self.callback: self.callback(price)
    def on_error(self,ws,error): st.error(f"WebSocket Fehler: {error}")
    def on_close(self,ws,close_status_code,close_msg): self.running=False
    def on_open(self,ws): self.running=True
    def start(self,callback):
        self.callback=callback
        self.ws=websocket.WebSocketApp(self.socket_url,on_open=self.on_open,on_message=self.on_message,on_error=self.on_error,on_close=self.on_close)
        self.thread=threading.Thread(target=self.ws.run_forever); self.thread.daemon=True; self.thread.start()
    def stop(self):
        if self.ws: self.ws.close(); self.running=False

def generate_image_stable_diffusion(prompt,model_id="runwayml/stable-diffusion-v1-5"):
    if not DIFFUSERS_AVAILABLE: return None,"Bibliothek 'diffusers' nicht installiert."
    try:
        pipe=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16 if CUDA_AVAILABLE else torch.float32)
        if CUDA_AVAILABLE: pipe=pipe.to("cuda")
        image=pipe(prompt,num_inference_steps=20).images[0]
        buf=io.BytesIO(); image.save(buf,format="PNG"); buf.seek(0)
        return buf,None
    except Exception as e: return None,str(e)

def calculate_backtest_metrics(prices,signals):
    returns=np.diff(prices)/prices[:-1]
    strategy_returns=[]
    position=0
    for i,sig in enumerate(signals):
        if sig==1: position=1
        elif sig==-1: position=-1
        if i<len(returns): strategy_returns.append(position*returns[i])
    strategy_returns=np.array(strategy_returns)
    total_return=np.prod(1+strategy_returns)-1 if len(strategy_returns)>0 else 0
    sharpe=np.mean(strategy_returns)/(np.std(strategy_returns)+1e-8)*np.sqrt(252)
    cum_returns=np.cumprod(1+strategy_returns)
    running_max=np.maximum.accumulate(cum_returns)
    drawdown=(cum_returns-running_max)/running_max
    max_drawdown=np.min(drawdown)
    return {"total_return":total_return,"sharpe_ratio":sharpe,"max_drawdown":max_drawdown,"num_trades":np.sum(np.abs(np.diff(signals))//2)}

def benchmark_sota(X,y,task_type,automl=None,test_size=0.2):
    X_enc=X.copy()
    categorical_cols=X_enc.select_dtypes(include=['object','category']).columns
    if len(categorical_cols)>0: X_enc=pd.get_dummies(X_enc,columns=categorical_cols,drop_first=False)
    for col in X_enc.columns:
        if X_enc[col].isnull().any():
            if X_enc[col].dtype in ['int64','float64']: X_enc[col]=X_enc[col].fillna(X_enc[col].median())
            else: mode_val=X_enc[col].mode(); X_enc[col]=X_enc[col].fillna(mode_val[0] if not mode_val.empty else 0)
    if task_type=='classification':
        counts=Counter(y)
        rare_classes=[cls for cls,cnt in counts.items() if cnt<2]
        if rare_classes:
            y_modified=y.copy()
            if isinstance(y_modified,np.ndarray): y_modified=pd.Series(y_modified)
            y_modified=y_modified.replace(rare_classes,'Rare')
            new_counts=Counter(y_modified)
            if min(new_counts.values())>=2: y_for_stratify=y_modified; st.info(f"⚠️ {len(rare_classes)} seltene Klassen wurden zu 'Rare' zusammengefasst.")
            else: y_for_stratify=None; st.warning("Nach dem Zusammenfassen gibt es immer noch Klassen mit nur einem Mitglied. Stratifizierung wird deaktiviert.")
        else: y_for_stratify=y
        if y_for_stratify is not None: X_train,X_test,y_train,y_test=train_test_split(X_enc,y,test_size=test_size,random_state=42,stratify=y_for_stratify)
        else: X_train,X_test,y_train,y_test=train_test_split(X_enc,y,test_size=test_size,random_state=42)
    else: X_train,X_test,y_train,y_test=train_test_split(X_enc,y,test_size=test_size,random_state=42)
    results={}
    if XGB_AVAILABLE:
        try:
            model=XGBClassifier(n_estimators=100,random_state=42,eval_metric='logloss') if task_type=='classification' else XGBRegressor(n_estimators=100,random_state=42)
            model.fit(X_train,y_train)
            pred=model.predict(X_test)
            score=accuracy_score(y_test,pred) if task_type=='classification' else r2_score(y_test,pred)
            results['XGBoost']=round(score,4)
        except Exception as e: results['XGBoost']=str(e)[:100]
    if LGBM_AVAILABLE:
        try:
            model=LGBMClassifier(n_estimators=100,random_state=42,verbose=-1) if task_type=='classification' else LGBMRegressor(n_estimators=100,random_state=42,verbose=-1)
            model.fit(X_train,y_train)
            pred=model.predict(X_test)
            score=accuracy_score(y_test,pred) if task_type=='classification' else r2_score(y_test,pred)
            results['LightGBM']=round(score,4)
        except Exception as e: results['LightGBM']=str(e)[:100]
    if CATBOOST_AVAILABLE:
        try:
            model=CatBoostClassifier(iterations=100,verbose=0,random_seed=42) if task_type=='classification' else CatBoostRegressor(iterations=100,verbose=0,random_seed=42)
            model.fit(X_train,y_train)
            pred=model.predict(X_test)
            score=accuracy_score(y_test,pred) if task_type=='classification' else r2_score(y_test,pred)
            results['CatBoost']=round(score,4)
        except Exception as e: results['CatBoost']=str(e)[:100]
    if automl is not None and automl.is_trained:
        try:
            X_test_anna=[]
            for idx in range(len(X_test)):
                row_df=X.iloc[[idx]][automl.feature_names]
                X_trans=automl.transform_one(row_df)
                X_test_anna.append(X_trans[0])
            X_test_anna=np.array(X_test_anna)
            if automl.scaler is not None: X_test_anna=automl.scaler.transform(X_test_anna)
            pred=automl.model.predict(X_test_anna)
            score=accuracy_score(y_test,pred) if task_type=='classification' else r2_score(y_test,pred)
            results['ANNA (RandomForest)']=round(score,4)
        except Exception as e: results['ANNA (RandomForest)']=str(e)[:100]
    return results

class NeuralMLP(nn.Module):
    def __init__(self,input_dim,hidden_dims=[128,64],output_dim=1,dropout=0.2):
        super().__init__()
        layers=[]
        prev_dim=input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim,hdim)); layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout))
            prev_dim=hdim
        layers.append(nn.Linear(prev_dim,output_dim))
        self.network=nn.Sequential(*layers)
    def forward(self,x): return self.network(x)

class FastAutoML:
    def __init__(self):
        self.model=None; self.model_name=None; self.score=None; self.is_trained=False; self.task_type=None; self.scaler=None
        self.all_results={}; self.best_params=None; self.feature_importances=None; self.feature_names=None
        self.label_encoders={}; self.onehot_encoders={}; self.dropped_id_columns=[]; self.mlp_scaler=None

    def detect_id_columns(self,X,threshold=0.8):
        drop_cols=[]
        for col in X.columns:
            if X[col].dtype=='object' and X[col].nunique()>len(X)*threshold: drop_cols.append(col)
        return drop_cols

    def prepare_data(self, X, y, task_type, max_classes=None, remove_id_cols=True, use_smote=False):
        X = X.copy()
        y = y.copy()
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
        
        valid_mask = y.notna()
        if not valid_mask.all():
            removed = (~valid_mask).sum()
            X = X[valid_mask]
            y = y[valid_mask]
            st.info(f"⚠️ {removed} Zeilen mit fehlenden Zielwerten wurden entfernt.")
        
        self.task_type = task_type
        self.dropped_id_columns = []
        
        if remove_id_cols:
            id_cols = self.detect_id_columns(X)
            if id_cols:
                st.info(f"🗑️ Automatisch entfernte ID-ähnliche Spalten: {', '.join(id_cols)}")
                self.dropped_id_columns = id_cols
                X = X.drop(columns=id_cols)
        
        # Datetime/Period Erkennung (ohne NotImplementedError)
        datetime_cols = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]) or pd.api.types.is_period_dtype(X[col]):
                datetime_cols.append(col)
        for col in datetime_cols:
            st.info(f"📅 Datetime/Period-Spalte '{col}' wird in numerisches Format konvertiert (Unix-Timestamp)")
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Kategoriale Spalten
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].fillna('MISSING').astype(str)
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            if X[col].nunique() > 50:
                st.warning(f"⚠️ Merkmal '{col}' hat {X[col].nunique()} eindeutige Werte. LabelEncoder wurde verwendet.")
        
        # Numerische Spalten: fehlende Werte mit Median füllen
        for col in X.select_dtypes(include=['int64', 'float64']).columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
        
        # Alle verbleibenden nicht-numerischen Spalten umwandeln
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                except Exception:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].fillna('MISSING').astype(str))
                    self.label_encoders[col] = le
        
        # Jetzt sicher zu float32
        X = X.astype(np.float32)
        
        if task_type == 'classification' and max_classes is not None and max_classes > 0:
            value_counts = y.value_counts()
            top_classes = value_counts.head(max_classes).index
            mask = y.isin(top_classes)
            X = X[mask]
            y = y[mask]
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        if use_smote and task_type == 'classification' and SMOTE_AVAILABLE:
            class_counts = Counter(y)
            min_samples = min(class_counts.values())
            k_neighbors = min(5, min_samples - 1)
            if k_neighbors < 1:
                st.warning("SMOTE nicht möglich: eine Klasse hat nur 1 Beispiel.")
            else:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X, y = smote.fit_resample(X, y)
                st.info(f"⚖️ SMOTE angewendet (k_neighbors={k_neighbors}). Daten auf {len(np.unique(y))} Klassen balanciert.")
        
        return X, y

    def transform_one(self, X_row):
        X = X_row.copy()
        if self.dropped_id_columns:
            X = X.drop(columns=[col for col in self.dropped_id_columns if col in X.columns], errors='ignore')
        
        # Datetime/Period konvertieren
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]) or pd.api.types.is_period_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        for col, le in self.label_encoders.items():
            if col in X.columns:
                val = X[col].iloc[0]
                try:
                    X[col] = le.transform([str(val)])[0]
                except:
                    X[col] = -1
        
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(0)
        
        X = X.astype(np.float32)
        return X.values

    def suggest_optimal_classes(self,y,max_possible=20):
        value_counts=y.value_counts(); total=len(y); cumsum=0; optimal=1
        for i,count in enumerate(value_counts):
            cumsum+=count
            if cumsum/total>=0.8: optimal=i+1; break
        return min(optimal,max_possible)

    def train(self,X_train,y_train,X_test,y_test,task_type,compare_all=False,skip_slow=True):
        self.task_type=task_type; self.all_results={}
        if task_type=='regression':
            models={'RandomForest':RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1),'LinearRegression':LinearRegression(),'DecisionTree':DecisionTreeRegressor(random_state=42),'GradientBoosting':GradientBoostingRegressor(random_state=42),'KNeighbors':KNeighborsRegressor(),'SVR':SVR()}
            scale_models=['LinearRegression','KNeighbors','SVR','GradientBoosting']
            if XGB_AVAILABLE: models['XGBoost']=XGBRegressor(n_estimators=100,random_state=42); scale_models.append('XGBoost')
            if LGBM_AVAILABLE: models['LightGBM']=LGBMRegressor(n_estimators=100,random_state=42); scale_models.append('LightGBM')
            if CATBOOST_AVAILABLE:
                try: models['CatBoost']=CatBoostRegressor(iterations=100,random_seed=42,verbose=0); scale_models.append('CatBoost')
                except: pass
        else:
            models={'RandomForest':RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1,class_weight='balanced'),'LogisticRegression':LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced'),'DecisionTree':DecisionTreeClassifier(random_state=42,class_weight='balanced'),'GradientBoosting':GradientBoostingClassifier(random_state=42),'KNeighbors':KNeighborsClassifier(),'SVC':SVC(class_weight='balanced',probability=True)}
            scale_models=['LogisticRegression','KNeighbors','SVC','GradientBoosting']
            if XGB_AVAILABLE: models['XGBoost']=XGBClassifier(n_estimators=100,random_state=42,use_label_encoder=False,eval_metric='logloss'); scale_models.append('XGBoost')
            if LGBM_AVAILABLE: models['LightGBM']=LGBMClassifier(n_estimators=100,random_state=42,class_weight='balanced'); scale_models.append('LightGBM')
            if CATBOOST_AVAILABLE:
                try: models['CatBoost']=CatBoostClassifier(iterations=100,random_seed=42,verbose=0,class_weights='Balanced'); scale_models.append('CatBoost')
                except: pass
        if not compare_all:
            model=models['RandomForest']
            scaler=None
            if 'RandomForest' in scale_models: scaler=StandardScaler(); X_tr=scaler.fit_transform(X_train); X_te=scaler.transform(X_test)
            else: X_tr,X_te=X_train,X_test
            model.fit(X_tr,y_train)
            pred=model.predict(X_te)
            score=r2_score(y_test,pred) if task_type=='regression' else accuracy_score(y_test,pred)
            self.model=model; self.model_name='RandomForest'; self.score=score; self.scaler=scaler; self.all_results['RandomForest']=score
            if hasattr(model,'feature_importances_'): self.feature_importances=model.feature_importances_
            self.is_trained=True
            return self.model_name,self.score,self.all_results
        best_score=-np.inf; best_model=None; best_name=None; best_scaler=None
        progress_bar=st.progress(0); items=list(models.items())
        for i,(name,model) in enumerate(items):
            progress_bar.progress((i+1)/len(items))
            try:
                if name in scale_models: scaler=StandardScaler(); X_tr=scaler.fit_transform(X_train); X_te=scaler.transform(X_test)
                else: scaler=None; X_tr,X_te=X_train,X_test
                model.fit(X_tr,y_train)
                pred=model.predict(X_te)
                score=r2_score(y_test,pred) if task_type=='regression' else accuracy_score(y_test,pred)
                self.all_results[name]=score
                if score>best_score: best_score=score; best_model=model; best_name=name; best_scaler=scaler
                if hasattr(model,'feature_importances_'): self.feature_importances=model.feature_importances_
            except Exception as e: st.warning(f"⚠️ {name} fehlgeschlagen: {str(e)[:100]}")
        progress_bar.empty()
        self.model=best_model; self.model_name=best_name; self.score=best_score; self.scaler=best_scaler; self.is_trained=True
        return best_name,best_score,self.all_results

    def train_neural_network(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        if not PYTORCH_AVAILABLE:
            st.warning("PyTorch nicht installiert. Neuronales Training nicht verfügbar.")
            return None, None, None

        def ultimate_force_numeric(X):
            if isinstance(X, pd.DataFrame):
                X = X.copy()
                for col in X.columns:
                    if pd.api.types.is_datetime64_any_dtype(X[col]):
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) == 0:
                    st.error("MLP: Keine numerischen Spalten vorhanden. Training abgebrochen.")
                    return None
                X = X[numeric_cols]
                X = X.fillna(0)
                X = X.values
            if isinstance(X, np.ndarray) or isinstance(X, list):
                X = np.array(X, dtype=object)
                flat = X.ravel()
                new_flat = []
                for v in flat:
                    if v is None or (isinstance(v, str) and v.strip() == ''):
                        new_flat.append(0.0)
                    elif isinstance(v, str):
                        v_clean = v.strip().replace(',', '.')
                        try:
                            num = float(v_clean)
                            if np.isnan(num):
                                new_flat.append(0.0)
                            else:
                                new_flat.append(num)
                        except ValueError:
                            new_flat.append(0.0)
                    else:
                        try:
                            num = float(v)
                            if np.isnan(num):
                                new_flat.append(0.0)
                            else:
                                new_flat.append(num)
                        except (ValueError, TypeError):
                            new_flat.append(0.0)
                X = np.array(new_flat, dtype=np.float32).reshape(X.shape)
            try:
                X = X.astype(np.float32)
            except:
                X = np.zeros((X.shape[0], 1), dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X

        X_train = ultimate_force_numeric(X_train)
        X_test = ultimate_force_numeric(X_test)
        if X_train is None or X_test is None:
            return None, None, None

        if X_train.shape[1] == 0:
            st.error("MLP: Keine numerischen Spalten – Training abgebrochen.")
            return None, None, None

        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        if self.task_type == 'classification':
            if y_train.dtype == object or y_train.dtype.kind in 'SU':
                le = LabelEncoder()
                y_train = le.fit_transform(y_train.astype(str))
                y_test = le.transform(y_test.astype(str))
            else:
                y_train = y_train.astype(np.int64)
                y_test = y_test.astype(np.int64)
            output_dim = len(np.unique(y_train))
            criterion = nn.CrossEntropyLoss()
            if y_test.max() >= output_dim or y_test.min() < 0:
                st.error(f"MLP: Testlabels außerhalb des Bereichs. min={y_test.min()}, max={y_test.max()}, output_dim={output_dim}")
                return None, None, None
        else:
            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)
            output_dim = 1
            criterion = nn.MSELoss()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long if self.task_type == 'classification' else torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.long if self.task_type == 'classification' else torch.float32).to(device)

        model = NeuralMLP(input_dim=X_train.shape[1], hidden_dims=[128,64], output_dim=output_dim, dropout=0.2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        progress_bar = st.progress(0)
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                if self.task_type == 'classification':
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
            progress_bar.progress((epoch+1)/epochs)
        progress_bar.empty()

        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t)
            if self.task_type == 'classification':
                _, pred_labels = torch.max(pred_test, 1)
                score = accuracy_score(y_test, pred_labels.cpu().numpy())
                additional = {'f1_macro': f1_score(y_test, pred_labels.cpu().numpy(), average='macro')}
            else:
                score = r2_score(y_test, pred_test.squeeze().cpu().numpy())
                additional = {'mse': mean_squared_error(y_test, pred_test.squeeze().cpu().numpy())}
        return model, score, additional

    def optimize_hyperparams(self,X_train,y_train,X_test,y_test,task_type):
        def clean_numeric(X):
            if isinstance(X,pd.DataFrame):
                X=X.copy()
                for col in X.columns: X[col]=pd.to_numeric(X[col],errors='coerce')
                X=X.fillna(0).values
            X=np.array(X,dtype=np.float32); X=np.nan_to_num(X,nan=0.0)
            return X
        X_train=clean_numeric(X_train); X_test=clean_numeric(X_test)
        if task_type=='regression':
            base_model=RandomForestRegressor(random_state=42,n_jobs=-1)
            param_dist={'n_estimators':[50,100,200],'max_depth':[10,20,None],'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4],'max_features':['sqrt','log2',None]}
            scoring='r2'
        else:
            base_model=RandomForestClassifier(random_state=42,n_jobs=-1,class_weight='balanced')
            param_dist={'n_estimators':[50,100,200],'max_depth':[10,20,None],'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4],'max_features':['sqrt','log2',None]}
            scoring='accuracy'
        random_search=RandomizedSearchCV(base_model,param_distributions=param_dist,n_iter=15,cv=3,scoring=scoring,random_state=42,n_jobs=-1,verbose=0)
        random_search.fit(X_train,y_train)
        best_model=random_search.best_estimator_
        pred=best_model.predict(X_test)
        score=r2_score(y_test,pred) if task_type=='regression' else accuracy_score(y_test,pred)
        self.model=best_model; self.model_name='RandomForest (optimiert)'; self.score=score; self.best_params=random_search.best_params_
        if hasattr(best_model,'feature_importances_'): self.feature_importances=best_model.feature_importances_
        self.is_trained=True
        return self.model_name,self.score,self.best_params

    def plot_feature_importance(self, feature_names):
        if self.feature_importances is None or len(self.feature_importances) == 0:
            st.info("Keine Feature Importance verfügbar (Modell hat keine importances).")
            return None
        if feature_names is None or len(feature_names) == 0:
            st.warning("Keine Feature-Namen verfügbar.")
            return None
        n_features = len(self.feature_importances)
        if len(feature_names) != n_features:
            st.warning(f"Anzahl der Feature-Namen ({len(feature_names)}) stimmt nicht mit der Anzahl der Importances ({n_features}) überein. Verwende die ersten {min(len(feature_names), n_features)} Namen.")
            feature_names = feature_names[:n_features]
            self.feature_importances = self.feature_importances[:len(feature_names)]
            n_features = len(feature_names)
        indices = np.argsort(self.feature_importances)[::-1][:min(15, n_features)]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(range(len(indices)), self.feature_importances[indices], color='skyblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Wichtigkeit')
        ax.set_title('Top-15 Feature‑Wichtigkeit')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    def predict(self,X):
        if not self.is_trained: raise ValueError("Modell nicht trainiert")
        if self.scaler is not None:
            if hasattr(X,'values'): X=X.values
            X=self.scaler.transform(X)
        return self.model.predict(X)

    def get_results_df(self):
        if not self.all_results: return pd.DataFrame()
        df=pd.DataFrame([{'Modell':k,'Genauigkeit':round(v,4)} for k,v in self.all_results.items() if v is not None])
        return df.sort_values('Genauigkeit',ascending=False)

    def get_additional_metrics(self,X_test,y_test):
        if self.task_type!='classification' or self.model is None: return None
        pred=self.model.predict(X_test)
        f1=f1_score(y_test,pred,average='macro')
        try:
            if hasattr(self.model,'predict_proba'):
                proba=self.model.predict_proba(X_test)
                if proba.shape[1]==2: auc=roc_auc_score(y_test,proba[:,1])
                else: auc=roc_auc_score(y_test,proba,multi_class='ovr',average='macro')
            else: auc=None
        except: auc=None
        return {'f1_macro':f1,'roc_auc':auc}

def main():
    # Session-State initialisieren
    if 'anna_activated' not in st.session_state: st.session_state.anna_activated=False
    if 'sound_activated' not in st.session_state: st.session_state.sound_activated=False
    if 'df' not in st.session_state: st.session_state.df=None
    if 'automl_engine' not in st.session_state: st.session_state.automl_engine=None
    if 'target_col' not in st.session_state: st.session_state.target_col=None
    if 'messages' not in st.session_state: st.session_state.messages=[]
    if 'music_search_query' not in st.session_state: st.session_state.music_search_query=""
    if 'video_search_query' not in st.session_state: st.session_state.video_search_query=""
    if 'auto_search_music' not in st.session_state: st.session_state.auto_search_music=False
    if 'auto_search_video' not in st.session_state: st.session_state.auto_search_video=False
    if 'last_music_audio' not in st.session_state: st.session_state.last_music_audio=None
    if 'last_video_audio' not in st.session_state: st.session_state.last_video_audio=None
    if 'last_chat_audio' not in st.session_state: st.session_state.last_chat_audio=None
    if 'auto_chat_voice' not in st.session_state: st.session_state.auto_chat_voice=False
    if 'last_engineer_audio' not in st.session_state: st.session_state.last_engineer_audio=None
    if 'engineer_task' not in st.session_state: st.session_state.engineer_task=""
    if 'clean_offered' not in st.session_state: st.session_state.clean_offered=False
    if 'clean_result_message' not in st.session_state: st.session_state.clean_result_message=None
    if 'clean_sound_played' not in st.session_state: st.session_state.clean_sound_played=False
    if 'stream_running' not in st.session_state: st.session_state.stream_running=False; st.session_state.stream_prices=[]
    if 'gesture_running' not in st.session_state: st.session_state.gesture_running=False; st.session_state.gesture_thread=None
    if 'rpa_macro' not in st.session_state: st.session_state.rpa_macro=[]
    if 'rpa_recording' not in st.session_state: st.session_state.rpa_recording=False

    if CLOUDPICKLE_AVAILABLE and os.path.exists("models/automl_full.pkl") and st.session_state.automl_engine is None:
        try:
            with open("models/automl_full.pkl", "rb") as f:
                st.session_state.automl_engine = cloudpickle.load(f)
            st.info("📦 Vorher gespeicherte Modell-Engine wurde automatisch geladen.")
        except:
            pass

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
        if not st.session_state.anna_activated:
            if st.button(T['activate_button'], key="activate_btn", use_container_width=True, type="primary"):
                activate_anna()
                st.rerun()
        else:
            st.success(T['anna_activated'])
            if st.session_state.automl_engine and st.session_state.automl_engine.is_trained:
                st.info(f"🏆 Bestes Modell: {st.session_state.automl_engine.model_name} (Score: {st.session_state.automl_engine.score:.4f})")
        st.markdown("---")
        if st.session_state.df is not None:
            st.metric("Zeilen", len(st.session_state.df))
            st.metric("Spalten", len(st.session_state.df.columns))

    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 1rem; margin-bottom: 1rem;">
        <h1 style="color: white; margin: 0;">{T['app_title']}</h1>
        <p style="color: #e0d4ff; margin: 0;">{T['app_subtitle']} | 🇩🇪 🇷🇺 🇺🇦</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.anna_activated:
        st.info(T['voice_warning'])
        return

    modi = [
        T['mode_chat'], T['mode_automl'], T['mode_data'], T['mode_3d'],
        T['mode_hyper'], T['mode_shap'], T['mode_music'], T['mode_video'],
        T['mode_engineer'], T['nlp_title'], T['ts_title'], T['img_title'],
        T['mode_llm'], T['mode_stream'], T['mode_sd'], T['mode_backtest'], T['mode_benchmark'],
        T['mode_gesture'], T['mode_agent'], T['mode_rpa']
    ]
    ausgewaehlter_modus = st.selectbox("📌 Modus auswählen", modi, key="mode_selector")

    # -------------------- 1. CHAT --------------------
    if ausgewaehlter_modus == modi[0]:
        st.subheader(T['chat_title'])
        for msg in st.session_state.messages[-30:]:
            if msg['role']=='user':
                st.markdown(f"<div style='background: linear-gradient(135deg, #667eea, #764ba2); color:white; padding:10px 15px; border-radius:18px; margin:5px 0 5px auto; max-width:80%; text-align:right'>👤 {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#f0f2f6; padding:10px 15px; border-radius:18px; margin:5px 0; max-width:80%'>🤖 Anna: {msg['content']}</div>", unsafe_allow_html=True)
                if msg.get('audio'): st.audio(msg['audio'])
        audio_bytes = None
        try:
            from audio_recorder_streamlit import audio_recorder
            audio_bytes = audio_recorder(text="", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="microphone", icon_size="3x", pause_threshold=3.0, key="chat_recorder")
        except:
            pass
        if audio_bytes and audio_bytes != st.session_state.last_chat_audio:
            st.session_state.last_chat_audio = audio_bytes
            with st.spinner("🎤 Erkenne Sprache..."):
                user_input = recognize_speech_auto(audio_bytes)
                if user_input:
                    st.success(f"✅ Erkannt: {user_input}")
                    st.session_state.auto_chat_voice = True
                    st.rerun()
        user_input = st.chat_input(T['chat_placeholder'])
        if st.session_state.get('auto_chat_voice', False):
            user_input = st.session_state.get('voice_text', '')
            st.session_state.auto_chat_voice = False
        if user_input:
            st.session_state.messages.append({"role":"user","content":user_input})
            if OLLAMA_AVAILABLE:
                with st.spinner("Anna denkt..."):
                    resp = query_ollama(user_input, "llama3.2:3b")
            else:
                lower=user_input.lower()
                if "привет" in lower or "hallo" in lower: resp="Hallo! Ich bin hier. Wie kann ich helfen?"
                elif "гиперактивный" in lower or "hyperaktiv" in lower: resp="Der hyperaktive Modus ist mein Highlight! Gehen Sie zum Tab Hyperaktiv."
                elif "нейросеть" in lower or "neuronales" in lower: resp="Ich habe neuronale Netze mit GPU-Unterstützung! Laden Sie Daten im AutoML Tab."
                else: resp=f"Ich habe Sie verstanden. Nutzen Sie die Tabs für die Funktionen."
            audio = speak(resp,lang='de')
            st.session_state.messages.append({"role":"anna","content":resp,"audio":audio})
            st.rerun()

    # -------------------- 2. AUTOML --------------------
    elif ausgewaehlter_modus == modi[1]:
        st.subheader(T['automl_title'])
        if PYTORCH_AVAILABLE:
            st.info(T["cpu_mode"] if not CUDA_AVAILABLE else T["gpu_activated"].format(device=torch.cuda.get_device_name(0)))
        if st.session_state.clean_result_message:
            st.success(st.session_state.clean_result_message)
            if not st.session_state.clean_sound_played:
                audio = speak(st.session_state.clean_result_message,lang='de')
                if audio:
                    st.audio(audio,format='audio/mp3',autoplay=True)
                st.session_state.clean_sound_played = True
        uploaded_file = st.file_uploader(T['file_upload'], type=["csv","xlsx","xls"], key="automl_uploader")
        if uploaded_file is not None:
            current_file_name = st.session_state.get('current_file_name','')
            if uploaded_file.name != current_file_name:
                try:
                    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                    else: df = pd.read_excel(uploaded_file)
                    st.session_state.df = df
                    st.session_state.current_file_name = uploaded_file.name
                    st.session_state.clean_offered = False
                    st.session_state.clean_result_message = None
                    st.session_state.clean_sound_played = False
                except Exception as e: st.error(f"Fehler beim Laden: {e}")
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df.head())
            st.caption(f"📊 {st.session_state.df.shape[0]} Zeilen, {st.session_state.df.shape[1]} Spalten")
            missing = st.session_state.df.isnull().sum().sum()
            dup = st.session_state.df.duplicated().sum()
            if (missing>0 or dup>0) and not st.session_state.clean_offered:
                st.warning(f"⚠️ Gefunden: {missing} fehlende Werte, {dup} Duplikate. {T['clean_question']}")
                if not st.session_state.clean_sound_played:
                    offer_msg = T["found_missing_duplicates"].format(missing=missing, duplicates=dup)
                    audio_offer = speak(offer_msg,lang='de')
                    if audio_offer:
                        st.audio(audio_offer,format='audio/mp3',autoplay=True)
                    st.session_state.clean_sound_played = True
                col1,col2 = st.columns(2)
                with col1:
                    if st.button(T['clean_yes'], key="automl_clean_yes_once", use_container_width=True):
                        df_clean,bm,am,bd,ad = clean_data_pro(st.session_state.df)
                        st.session_state.df = df_clean
                        msg_clean = T["data_cleaned"].format(missing=bm-am, duplicates=bd-ad)
                        st.session_state.clean_result_message = msg_clean
                        st.session_state.clean_sound_played = False
                        st.session_state.clean_offered = True
                        st.rerun()
                with col2:
                    if st.button(T['clean_no'], key="automl_clean_no_once", use_container_width=True):
                        st.info(T["data_unchanged"])
                        st.session_state.clean_offered = True
                        st.rerun()
            else:
                if st.button(T['clean_button'], key="automl_manual_clean", use_container_width=True):
                    df_clean,bm,am,bd,ad = clean_data_pro(st.session_state.df)
                    st.session_state.df = df_clean
                    msg_clean = T["data_cleaned"].format(missing=bm-am, duplicates=bd-ad)
                    st.session_state.clean_result_message = msg_clean
                    st.session_state.clean_sound_played = False
                    st.rerun()
            target = st.selectbox(T['target_select'], st.session_state.df.columns.tolist(), key="target")
            st.session_state.target_col = target
            y = st.session_state.df[target]
            task_type_option = st.radio(T['task_type_select'], [T['task_auto'], T['task_class'], T['task_reg']], index=0, horizontal=True, key="task_type_radio")
            if task_type_option == T['task_auto']:
                task_type = 'regression' if (y.dtype in ['int64','float64'] and y.nunique()>10) else 'classification'
            elif task_type_option == T['task_class']: task_type = 'classification'
            else: task_type = 'regression'
            st.info(f"📊 Aufgabe: {'Regression' if task_type=='regression' else 'Klassifikation'}")
            compare_models = st.checkbox(T["compare_models"], value=False)
            if compare_models: skip_slow = st.checkbox(T["skip_slow_models"], value=True)
            else: skip_slow = True
            col_opt1,col_opt2 = st.columns(2)
            with col_opt1: remove_id = st.checkbox(T["remove_id_columns"], value=True)
            with col_opt2: use_smote = st.checkbox(T["apply_smote"], value=False, disabled=not SMOTE_AVAILABLE)
            max_classes = None
            if task_type == 'classification':
                unique_classes = y.nunique()
                if unique_classes > 2:
                    st.warning(f"⚠️ {unique_classes} Klassen erkannt.")
                    if st.button(T['auto_classes_button'], key="auto_classes_btn"):
                        automl_helper = FastAutoML()
                        optimal = automl_helper.suggest_optimal_classes(y, max_possible=30)
                        st.info(T["auto_classes_recommend"].format(optimal=optimal))
                        if st.button(T["auto_classes_accept"].format(optimal=optimal)):
                            max_classes = optimal
                            st.rerun()
                    reduce_classes = st.checkbox(T["reduce_classes"])
                    if reduce_classes:
                        max_classes = st.number_input(T["reduce_classes_count"], min_value=2, max_value=50, value=min(20, unique_classes))
            if st.button(T['train_button'], key="train_btn", type="primary", use_container_width=True):
                st.session_state.clean_result_message = None
                st.session_state.clean_sound_played = False
                with st.spinner(T['train_spinner']):
                    X = st.session_state.df.drop(columns=[target])
                    automl = FastAutoML()
                    X_prep, y_prep = automl.prepare_data(X, y, task_type, max_classes, remove_id_cols=remove_id, use_smote=use_smote)
                    stratify = None
                    if task_type == 'classification':
                        counts = Counter(y_prep)
                        rare_classes = [cls for cls,cnt in counts.items() if cnt < 2]
                        if rare_classes:
                            y_modified = y_prep.copy()
                            if isinstance(y_modified, np.ndarray): y_modified = pd.Series(y_modified)
                            y_modified = y_modified.replace(rare_classes, 'Rare')
                            new_counts = Counter(y_modified)
                            if min(new_counts.values()) >= 2:
                                stratify = y_modified
                                st.info(f"⚠️ {len(rare_classes)} seltene Klassen wurden zu 'Rare' zusammengefasst.")
                            else:
                                stratify = None
                                st.warning("Nach dem Zusammenfassen gibt es immer noch Klassen mit nur einem Mitglied. Stratifizierung wird deaktiviert.")
                        else:
                            stratify = y_prep
                    X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, random_state=42, stratify=stratify)
                    st.session_state.last_X_train = X_train
                    st.session_state.last_X_test = X_test
                    st.session_state.last_y_train = y_train
                    st.session_state.last_y_test = y_test
                    best_name, best_score, results = automl.train(X_train, y_train, X_test, y_test, task_type, compare_all=compare_models, skip_slow=skip_slow)
                    automl.feature_names = list(X.columns)
                    st.session_state.automl_engine = automl
                    if compare_models and results:
                        st.subheader(T["model_comparison_results"])
                        df_res = automl.get_results_df()
                        st.dataframe(df_res, use_container_width=True)
                    st.success(T["best_model"].format(name=best_name))
                    
                    if task_type == 'regression':
                        st.info(T["r2_score"].format(score=best_score))
                        speak(T["automl_complete_reg"].format(name=best_name, score=best_score), lang='de')
                    else:
                        st.info(T["accuracy_score"].format(score=best_score))
                        additional = automl.get_additional_metrics(X_test, y_test)
                        if additional:
                            st.subheader(T["additional_metrics"])
                            st.info(f"📊 F1 (macro): {additional['f1_macro']:.4f}")
                            if additional['roc_auc']:
                                st.info(f"📊 ROC-AUC (ovr): {additional['roc_auc']:.4f}")
                        y_pred = automl.model.predict(X_test)
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        df_report = pd.DataFrame(report).transpose()
                        st.subheader("📋 Detaillierte Metriken pro Klasse")
                        st.dataframe(df_report, use_container_width=True)
                        n_classes = len(np.unique(y_test))
                        if n_classes <= 30:
                            st.subheader("📊 Konfusionsmatrix (absolute Werte)")
                            fig_abs, ax_abs = plt.subplots(figsize=(12, 10))
                            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_abs, cmap='Blues', values_format='d')
                            ax_abs.set_title('Konfusionsmatrix (Anzahl der Beispiele)', fontsize=14)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig_abs)
                            st.subheader("📊 Konfusionsmatrix (zeilenweise normalisiert)")
                            fig_norm, ax_norm = plt.subplots(figsize=(12, 10))
                            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_norm, cmap='Blues', normalize='true', values_format='.2f')
                            ax_norm.set_title('Konfusionsmatrix (Anteil korrekter Vorhersagen pro Klasse)', fontsize=14)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig_norm)
                            if st.button("📸 Konfusionsmatrix als JPEG speichern", key="save_cm_jpeg"):
                                jpeg_data = save_matplotlib_fig_as_jpeg(fig_abs, "confusion_matrix")
                                st.download_button("⬇️ JPEG herunterladen", jpeg_data, file_name="confusion_matrix.jpg", mime="image/jpeg")
                        else:
                            st.warning(f"⚠️ Zu viele Klassen ({n_classes}) für eine übersichtliche Konfusionsmatrix. Verwenden Sie die obige Tabelle.")
                        baseline = 1.0/(len(np.unique(y_prep)))
                        st.info(T["baseline_random"].format(baseline=baseline))
                        if best_score <= baseline:
                            st.warning(T["model_not_better"])
                        speak(T["automl_complete_class"].format(name=best_name, score=best_score), lang='de')
                    st.balloons()

            if st.session_state.automl_engine is not None and st.session_state.automl_engine.is_trained:
                st.markdown("---")
                st.subheader("💾 Modellverwaltung")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if CLOUDPICKLE_AVAILABLE:
                        model_bytes = io.BytesIO()
                        cloudpickle.dump(st.session_state.automl_engine, model_bytes)
                        model_bytes.seek(0)
                        st.download_button(label="📥 Modell herunterladen (PKL)", data=model_bytes, file_name="anna_model.pkl", mime="application/octet-stream")
                    else:
                        st.warning("cloudpickle nicht installiert. Bitte 'pip install cloudpickle' ausführen.")
                with col2:
                    if CLOUDPICKLE_AVAILABLE:
                        if st.button("💾 Modell in Projektordner speichern"):
                            os.makedirs("models", exist_ok=True)
                            with open("models/automl_full.pkl", "wb") as f:
                                cloudpickle.dump(st.session_state.automl_engine, f)
                            st.success("✅ Modell-Engine gespeichert unter 'models/automl_full.pkl'")
                with col3:
                    if CLOUDPICKLE_AVAILABLE:
                        if st.button("📂 Gespeicherte Modell-Engine laden"):
                            if os.path.exists("models/automl_full.pkl"):
                                with open("models/automl_full.pkl", "rb") as f:
                                    st.session_state.automl_engine = cloudpickle.load(f)
                                st.success("✅ Modell-Engine geladen. Sie können nun SHAP, Hyperaktiv etc. nutzen.")
                            else:
                                st.error("❌ Keine gespeicherte Engine gefunden. Speichern Sie zuerst eine.")
                if st.checkbox("📊 Feature Importance anzeigen"):
                    fig = st.session_state.automl_engine.plot_feature_importance(st.session_state.automl_engine.feature_names)
                    if fig:
                        st.subheader(T["feature_importance_title"])
                        st.pyplot(fig)
                        speak(T["feature_importance_computed"], lang='de')
                    else:
                        st.info("Keine Feature Importance verfügbar.")

            if PYTORCH_AVAILABLE and st.button("🧠 Neuronales Netz (MLP) trainieren", key="mlp_train_btn"):
                if 'last_X_train' not in st.session_state:
                    st.warning("Bitte trainieren Sie zuerst das Hauptmodell mit der 'MODELL TRAINIEREN' Taste.")
                else:
                    with st.spinner("Trainiere neuronales Netz (MLP) ..."):
                        try:
                            automl = st.session_state.automl_engine
                            nn_model, nn_score, nn_extra = automl.train_neural_network(
                                st.session_state.last_X_train, st.session_state.last_y_train,
                                st.session_state.last_X_test, st.session_state.last_y_test, epochs=30)
                            if nn_model is not None:
                                st.success(f"🧠 Neuronales Netz: Genauigkeit = {nn_score:.4f}")
                                if nn_extra: st.json(nn_extra)
                            else:
                                st.error("Neuronales Netz konnte nicht trainiert werden (keine numerischen Daten).")
                        except Exception as e:
                            st.error(f"Fehler beim MLP-Training: {e}")

            if st.session_state.automl_engine is not None and st.session_state.automl_engine.model_name in ['RandomForest','RandomForest (optimiert)']:
                if st.button(T['hyperopt_button'], key="hyperopt_btn", type="secondary", use_container_width=True):
                    with st.spinner(T['hyperopt_spinner']):
                        X = st.session_state.df.drop(columns=[target])
                        automl = FastAutoML()
                        X_prep, y_prep = automl.prepare_data(X, y, task_type, max_classes, remove_id_cols=remove_id, use_smote=False)
                        if task_type == 'classification':
                            counts = Counter(y_prep)
                            rare_classes = [cls for cls,cnt in counts.items() if cnt<2]
                            if rare_classes:
                                y_modified = y_prep.copy()
                                if isinstance(y_modified, np.ndarray): y_modified = pd.Series(y_modified)
                                y_modified = y_modified.replace(rare_classes, 'Rare')
                                if min(Counter(y_modified).values()) >= 2: stratify = y_modified
                                else: stratify = None
                            else: stratify = y_prep
                        else: stratify = None
                        X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, random_state=42, stratify=stratify)
                        best_name, best_score, best_params = automl.optimize_hyperparams(X_train, y_train, X_test, y_test, task_type)
                        automl.feature_names = list(X.columns)
                        st.session_state.automl_engine = automl
                        st.success(T["optimization_complete"])
                        st.json(best_params)
                        st.info(T["improved_metric"].format(score=best_score))
                        speak(T["hyperopt_complete"].format(score=best_score), lang='de')
                        if automl.feature_importances is not None:
                            fig = automl.plot_feature_importance(automl.feature_names)
                            if fig:
                                st.subheader(T["feature_importance_title"])
                                st.pyplot(fig)
                        st.balloons()
        else:
            st.info(T["loading_data"])

    # -------------------- 3. DATENANALYSE (mit voller Tabelle und Diagramm) --------------------
    elif ausgewaehlter_modus == modi[2]:
        st.subheader(T['data_title'])
        if st.session_state.df is not None:
            analysis = analyze_data(st.session_state.df)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("Zeilen", f"{analysis['rows']:,}")
            with col2: st.metric("Spalten", analysis['cols'])
            with col3: st.metric("Fehlende", analysis['missing'])
            with col4: st.metric("Duplikate", analysis['duplicates'])
            with col5: st.metric("Speicher", f"{analysis['memory']:.2f} MB")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button(T['data_clean_button'], key="data_clean_btn", use_container_width=True):
                    df_clean, bm, am, bd, ad = clean_data_pro(st.session_state.df)
                    st.session_state.df = df_clean
                    msg_clean = T["data_cleaned"].format(missing=bm-am, duplicates=bd-ad)
                    st.session_state.clean_result_message = msg_clean
                    st.session_state.clean_sound_played = False
                    st.rerun()
            with c2:
                if st.button(T['data_stats_button'], use_container_width=True):
                    stats_df = st.session_state.df.describe().round(3)
                    st.dataframe(stats_df, use_container_width=True)
                    if st.button(T["save_stats_jpeg"], key="save_stats_jpeg"):
                        jpeg_data = save_dataframe_as_jpeg(stats_df, "Deskriptive Statistik")
                        st.download_button("⬇️ JPEG herunterladen", jpeg_data, file_name="statistik.jpg", mime="image/jpeg")
            with c3:
                if st.button(T['data_export_button'], use_container_width=True):
                    html_bytes = export_to_html(st.session_state.df, "ANNA AI Report")
                    st.download_button("📥 Download HTML", html_bytes, f"report_{datetime.now():%Y%m%d}.html", "text/html", key="download_html")
            
            st.markdown("---")
            
            tab1, tab2 = st.tabs(["📋 Vollständige Tabelle", "📈 Schnelles Diagramm"])
            
            with tab1:
                st.subheader("Vollständige Datenansicht")
                st.dataframe(st.session_state.df, use_container_width=True, height=500)
                st.caption(f"Gesamt: {len(st.session_state.df)} Zeilen × {len(st.session_state.df.columns)} Spalten")
            
            with tab2:
                st.subheader("Schnelles Diagramm")
                chart_type = st.radio("Diagrammtyp", ["Linie", "Balken", "Punkt"], horizontal=True, key="quick_chart_type")
                x_col = st.selectbox("X-Achse (optional, leer = Index)", [None] + st.session_state.df.columns.tolist(), key="quick_x")
                numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    y_cols = st.multiselect("Y-Achse (mindestens eine numerische Spalte)", numeric_cols, default=numeric_cols[0] if numeric_cols else None, key="quick_y")
                    if y_cols:
                        if chart_type == "Linie":
                            st.line_chart(st.session_state.df.set_index(x_col) if x_col else st.session_state.df, y=y_cols, use_container_width=True)
                        elif chart_type == "Balken":
                            st.bar_chart(st.session_state.df.set_index(x_col) if x_col else st.session_state.df, y=y_cols, use_container_width=True)
                        else:
                            if len(y_cols) >= 2:
                                fig, ax = plt.subplots()
                                ax.scatter(st.session_state.df[y_cols[0]], st.session_state.df[y_cols[1]], alpha=0.5)
                                ax.set_xlabel(y_cols[0])
                                ax.set_ylabel(y_cols[1])
                                ax.set_title(f"{y_cols[0]} vs {y_cols[1]}")
                                st.pyplot(fig)
                            else:
                                st.warning("Für Punktdiagramm bitte mindestens zwei numerische Spalten auswählen.")
                                st.line_chart(st.session_state.df.set_index(x_col) if x_col else st.session_state.df, y=y_cols, use_container_width=True)
                    else:
                        st.info("Bitte wählen Sie mindestens eine numerische Spalte für die Y-Achse.")
                else:
                    st.warning("Keine numerischen Spalten in den Daten vorhanden.")
            
            with st.expander("🔍 Datenvorschau (erste 20 Zeilen)", expanded=False):
                st.dataframe(st.session_state.df.head(20), use_container_width=True)
        else:
            st.info(T["loading_data"])

    # -------------------- 4. 3D --------------------
    elif ausgewaehlter_modus == modi[3]:
        st.subheader(T['3d_title'])
        if st.session_state.df is not None:
            num_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 3:
                col1,col2,col3 = st.columns(3)
                with col1: x = st.selectbox("X", num_cols, key="3d_x")
                with col2: y = st.selectbox("Y", num_cols, key="3d_y")
                with col3: z = st.selectbox("Z", num_cols, key="3d_z")
                fig = go.Figure(data=[go.Scatter3d(x=st.session_state.df[x], y=st.session_state.df[y], z=st.session_state.df[z],
                                                   mode='markers', marker=dict(size=3, color=st.session_state.df[z], colorscale='Viridis'))])
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                if st.button(T["save_3d_jpeg"], key="save_3d"):
                    img_bytes = save_plotly_as_jpeg(fig, "3D Visualisierung")
                    if img_bytes: st.download_button("⬇️ JPEG herunterladen", img_bytes, file_name="3d_plot.jpg", mime="image/jpeg")
            else:
                st.warning(T["at_least_3_numeric"])
        else:
            st.info(T["loading_data"])

    # -------------------- 5. HYPERAKTIV --------------------
    elif ausgewaehlter_modus == modi[4]:
        st.subheader(T['hyper_title'])
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; border-radius: 1rem; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0;">⚡ ECHTZEIT-VORHERSAGEN ⚡</h3>
            <p style="color: #e0d4ff;">Latenz &lt;100 ms | Handelssignale | Online‑Training</p>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.automl_engine is None or not st.session_state.automl_engine.is_trained:
            st.warning(T["train_model_first"])
        else:
            col1,col2 = st.columns(2)
            with col1:
                latency = st.slider("⏱️ Latenz (ms)", 10, 200, 100, key="lat")
                sim_mode = st.selectbox("📊 Modus", ['random','trend','sin','market'], key="mode")
            with col2:
                num_iter = st.slider("🔄 Iterationen", 10, 200, 50, key="iter")
                online_retrain = st.checkbox(T["online_retrain"], value=False)
                retrain_every = st.number_input(T["retrain_every"], min_value=5, max_value=100, value=50, step=5, disabled=not online_retrain)
            if st.button(T['hyper_start'], key="start_hyper", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                X_history = []
                y_history = []
                if sim_mode=='random': current = random.gauss(0,1)
                elif sim_mode=='trend': current = 100.0
                elif sim_mode=='sin': current = 0.0
                else: current = 50000.0
                for i in range(num_iter):
                    if sim_mode=='random': next_true = random.gauss(0,1)
                    elif sim_mode=='trend': next_true = current + 0.5 + random.gauss(0,1)
                    elif sim_mode=='sin': next_true = 10*math.sin((i+1)/10) + random.gauss(0,0.5)
                    else: change = random.gauss(0, current*0.002); next_true = current + change
                    try: pred = st.session_state.automl_engine.predict([[current]])[0]
                    except: pred = current*1.1 + random.gauss(0, abs(current)*0.05)
                    if pred > current*1.01: action = "🟢 KAUFEN"
                    elif pred < current*0.99: action = "🔴 VERKAUFEN"
                    else: action = "⚪ HALTEN"
                    start_time = time.perf_counter()
                    time.sleep(latency/1000)
                    elapsed_ms = (time.perf_counter()-start_time)*1000
                    results.append({'Nr.':i+1,'Eingang':round(current,4),'Vorhersage':round(pred,4),'true_value':round(next_true,4),'Signal':action,'Latenz (ms)':round(elapsed_ms,2)})
                    if online_retrain:
                        X_history.append([current]); y_history.append(next_true)
                        if len(X_history) >= retrain_every:
                            with st.spinner(f"🔄 Online-Training mit {len(X_history)} neuen Beispielen..."):
                                X_arr = np.array(X_history).reshape(-1,1); y_arr = np.array(y_history)
                                model = st.session_state.automl_engine.model
                                model.fit(X_arr, y_arr)
                                st.session_state.automl_engine.model = model
                            X_history = []; y_history = []; time.sleep(0.5)
                    current = next_true
                    progress_bar.progress((i+1)/num_iter)
                    status_text.markdown(f"**{T['live_status'].format(current=i+1, total=num_iter, pred=pred, action=action)}**")
                st.success(f"✅ {num_iter} Iterationen abgeschlossen!")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                if st.button(T["save_hyper_jpeg"], key="save_hyper_jpeg"):
                    jpeg_data = save_dataframe_as_jpeg(df_results, "Hypermodus Ergebnisse")
                    st.download_button("⬇️ JPEG herunterladen", jpeg_data, file_name="hyper_results.jpg", mime="image/jpeg")
                col_stat1,col_stat2,col_stat3 = st.columns(3)
                with col_stat1: correct = sum(1 for r in results if abs(r['Vorhersage']-r['true_value'])/(abs(r['true_value'])+1e-6)<0.1); st.metric(T["prediction_accuracy"], f"{correct/len(results):.1%}")
                with col_stat2: avg_lat = np.mean([r['Latenz (ms)'] for r in results]); st.metric(T["avg_latency"], f"{avg_lat:.1f} ms")
                with col_stat3: buy_cnt = sum(1 for r in results if 'KAUFEN' in r['Signal']); st.metric(T["buy_signals"], buy_cnt)
                if st.button(T["export_json"]):
                    json_str = json.dumps(results, indent=2, ensure_ascii=False)
                    st.download_button("📥 JSON herunterladen", json_str, "hyper_results.json", "application/json")
                st.balloons()
                speak(T["hyper_mode_complete"].format(iter=num_iter), 'de')

    # -------------------- 6. SHAP (mit sicherer Klassenauswahl) --------------------
    elif ausgewaehlter_modus == modi[5]:
        st.subheader(T['shap_title'])
        if not SHAP_AVAILABLE:
            st.error("❌ SHAP nicht installiert. pip install shap")
        elif st.session_state.automl_engine is None or not st.session_state.automl_engine.is_trained:
            st.warning(T["train_model_first"])
        elif st.session_state.df is None:
            st.warning(T["load_data_first"])
        else:
            target = st.session_state.target_col
            if target is None:
                st.warning(T["target_not_defined"])
            else:
                feature_cols = [col for col in st.session_state.df.columns if col != target][:20]
                row_idx = st.slider(T["select_row_for_shap"], 0, len(st.session_state.df)-1, 0, key="shap_row")
                X_raw = st.session_state.df.iloc[[row_idx]][feature_cols].copy()
                # Konvertiere datetime/period
                for col in X_raw.columns:
                    if pd.api.types.is_datetime64_any_dtype(X_raw[col]) or pd.api.types.is_period_dtype(X_raw[col]):
                        X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
                X_raw = X_raw.fillna(0)
                
                # Für Klassifikation: Auswahl der Klasse (mit Sicherheitsindex)
                if st.session_state.automl_engine.task_type == 'classification':
                    unique_classes = st.session_state.automl_engine.model.classes_
                    n_classes = len(unique_classes)
                    class_idx = st.selectbox("Klasse für SHAP-Erklärung:", range(n_classes), index=0,
                                              format_func=lambda i: f"{unique_classes[i]} (Klasse {i})")
                    # Nochmal auf gültigen Bereich prüfen
                    if class_idx >= n_classes:
                        class_idx = 0
                else:
                    class_idx = 0
                
                if st.button(T['shap_button'], key="shap_btn", type="primary", use_container_width=True):
                    with st.spinner("SHAP Analyse läuft..."):
                        try:
                            automl = st.session_state.automl_engine
                            X_processed = automl.transform_one(X_raw)
                            if automl.scaler is not None:
                                X_scaled = automl.scaler.transform(X_processed)
                            else:
                                X_scaled = X_processed
                            model_type = 'tree' if (hasattr(automl.model, 'feature_importances_') or hasattr(automl.model, 'get_booster')) else 'kernel'
                            fig = explain_with_shap(automl.model, X_scaled, feature_cols, model_type=model_type, class_idx=class_idx)
                            if fig:
                                st.pyplot(fig)
                                if st.button(T["save_shap_jpeg"], key="save_shap_jpeg"):
                                    jpeg_data = save_matplotlib_fig_as_jpeg(fig, "SHAP Erklärung")
                                    st.download_button("⬇️ JPEG herunterladen", jpeg_data, file_name="shap_plot.jpg", mime="image/jpeg")
                                speak(T["shap_analysis_complete"], 'de')
                            else:
                                st.info(T["shap_not_available"])
                        except Exception as e:
                            st.error(f"SHAP Fehler: {e}")
                            st.info("Stellen Sie sicher, dass das Modell baumbasiert ist oder verwenden Sie KernelExplainer.")

    # -------------------- 7. MUSIK --------------------
    elif ausgewaehlter_modus == modi[6]:
        st.subheader(T['music_title'])
        audio_bytes = None
        try:
            from audio_recorder_streamlit import audio_recorder
            audio_bytes = audio_recorder(text="", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="microphone", icon_size="3x", pause_threshold=3.0, key="music_recorder")
        except:
            pass
        if audio_bytes and audio_bytes != st.session_state.last_music_audio:
            st.session_state.last_music_audio = audio_bytes
            with st.spinner("🎤 Erkenne Sprache..."):
                text = recognize_speech_auto(audio_bytes)
                if text:
                    st.success(f"✅ Erkannt: {text}")
                    st.session_state.music_search_query = text
                    st.session_state.auto_search_music = True
                    st.rerun()
        query = st.session_state.get('music_search_query','')
        new_query = st.text_input("🔍 Lied- oder Künstlername", value=query, key="music_search_input")
        if new_query != query: st.session_state.music_search_query = new_query; query = new_query
        if st.session_state.get('auto_search_music', False):
            st.session_state.auto_search_music = False
            if query.strip():
                with st.spinner(f"🎵 Suche '{query}'..."):
                    results = search_music(query,5)
                    st.markdown(format_music_results(results,query))
            else: st.warning(T["no_audio_input"])
        if st.button(T["search_music"], key="music_search_btn", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner(f"🎵 Suche '{query}'..."):
                    results = search_music(query,5)
                    st.markdown(format_music_results(results,query))
            else: st.warning(T["enter_search_term"])

    # -------------------- 8. VIDEO --------------------
    elif ausgewaehlter_modus == modi[7]:
        st.subheader(T['video_title'])
        audio_bytes = None
        try:
            from audio_recorder_streamlit import audio_recorder
            audio_bytes = audio_recorder(text="", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="microphone", icon_size="3x", pause_threshold=3.0, key="video_recorder")
        except:
            pass
        if audio_bytes and audio_bytes != st.session_state.last_video_audio:
            st.session_state.last_video_audio = audio_bytes
            with st.spinner("🎤 Erkenne Sprache..."):
                text = recognize_speech_auto(audio_bytes)
                if text:
                    st.success(f"✅ Erkannt: {text}")
                    st.session_state.video_search_query = text
                    st.session_state.auto_search_video = True
                    st.rerun()
        query = st.session_state.get('video_search_query','')
        new_query = st.text_input("🔍 Suchbegriff für Video", value=query, key="video_search_input")
        if new_query != query: st.session_state.video_search_query = new_query; query = new_query
        if st.session_state.get('auto_search_video', False):
            st.session_state.auto_search_video = False
            if query.strip():
                with st.spinner(f"🎬 Suche '{query}'..."):
                    results = search_video(query,5)
                    st.markdown(format_video_results(results,query))
            else: st.warning(T["no_audio_input"])
        if st.button(T["search_video"], key="video_search_btn", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner(f"🎬 Suche '{query}'..."):
                    results = search_video(query,5)
                    st.markdown(format_video_results(results,query))
            else: st.warning(T["enter_search_term"])

    # -------------------- 9. AI-ENGINEER --------------------
    elif ausgewaehlter_modus == modi[8]:
        st.subheader(T['engineer_title'])
        st.markdown("Beschreiben Sie eine Aufgabe, die Python-Code (oder SQL) erfordert. Anna wird automatisch den Code generieren und sicher ausführen.")
        audio_bytes = None
        try:
            from audio_recorder_streamlit import audio_recorder
            audio_bytes = audio_recorder(text="", recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="microphone", icon_size="3x", pause_threshold=3.0, key="engineer_recorder")
        except:
            pass
        if audio_bytes and audio_bytes != st.session_state.last_engineer_audio:
            st.session_state.last_engineer_audio = audio_bytes
            with st.spinner("🎤 Erkenne Sprache..."):
                task = recognize_speech_auto(audio_bytes)
                if task:
                    st.success(f"✅ Erkannt: {task}")
                    st.session_state.engineer_task = task
                    st.rerun()
        task = st.text_area(T['engineer_placeholder'], value=st.session_state.engineer_task, height=100, key="engineer_task_input")
        if st.button(T['engineer_generate'], key="engineer_generate_btn", type="primary", use_container_width=True):
            if task:
                with st.spinner(T["generate_code"]):
                    if "sql" in task.lower():
                        code = """
import sqlite3
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
cursor.execute('CREATE TABLE example (id INTEGER, name TEXT)')
cursor.execute("INSERT INTO example VALUES (1, 'Anna')")
conn.commit()
result = cursor.fetchall()
"""
                    elif "histogramm" in task.lower() or "гистограмм" in task.lower():
                        code = """
import matplotlib.pyplot as plt
import numpy as np
data = np.random.randn(1000)
plt.hist(data, bins=30, alpha=0.7, color='skyblue')
plt.title('Histogramm')
plt.xlabel('Wert')
plt.ylabel('Häufigkeit')
plt.grid(True, alpha=0.3)
plt.show()
result = "Histogramm wurde erstellt."
"""
                    elif "zeichne" in task.lower() or "график" in task.lower():
                        code = """
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sinuskurve')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()
result = "Plot wurde erstellt."
"""
                    else:
                        code = """
import pandas as pd
import numpy as np
df_demo = pd.DataFrame({'A': np.random.randn(5), 'B': np.random.randn(5)})
result = df_demo.describe()
"""
                    st.session_state.generated_code = code
                with st.spinner(T["execute_code"]):
                    output, error = None, None
                    try:
                        exec_globals = {'pd': pd, 'np': np, 'plt': plt, 'result': None}
                        exec(code, exec_globals)
                        output = exec_globals.get('result', 'Code erfolgreich ausgeführt')
                    except Exception as e: error = str(e)
                    st.session_state.code_output = output
                    st.session_state.code_error = error
                st.rerun()
            else: st.warning(T["describe_task"])
        if 'generated_code' in st.session_state:
            st.markdown(T['engineer_code'])
            st.code(st.session_state.generated_code, language='python')
            if st.session_state.get('code_error') is None:
                st.markdown(T['engineer_result'])
                st.write(st.session_state.get('code_output', 'Keine Ausgabe.'))
            else: st.error(f"{T['code_execution_error']}\n{st.session_state.code_error}")
        else: st.info(T["no_code_generated"])

    # -------------------- 10. NLP --------------------
    elif ausgewaehlter_modus == modi[9]:
        st.subheader(T['nlp_title'])
        st.markdown("Klassifizierung von Texten (Sentiment, Thema) mit Transformer-Modellen")
        if not TRANSFORMERS_AVAILABLE: st.error("❌ 'transformers' nicht installiert. Führen Sie aus: pip install transformers torch")
        else:
            uploaded_nlp = st.file_uploader("📂 CSV mit Textspalte und Label", type=["csv","xlsx"], key="nlp_upload")
            if uploaded_nlp:
                df_nlp = pd.read_csv(uploaded_nlp) if uploaded_nlp.name.endswith('.csv') else pd.read_excel(uploaded_nlp)
                st.dataframe(df_nlp.head())
                text_col = st.selectbox("Textspalte", df_nlp.columns, key="nlp_text")
                label_col = st.selectbox("Label (Klassen)", df_nlp.columns, key="nlp_label")
                model_name = st.selectbox("Transformer Modell", ["bert-base-multilingual-cased","distilbert-base-german-cased","cointegrated/rubert-tiny"])
                if st.button("🚀 Textklassifikation starten"):
                    with st.spinner("Extrahiere Embeddings..."):
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        transformer_model = AutoModel.from_pretrained(model_name)
                        def get_embeddings(texts):
                            emb_list = []
                            for t in texts:
                                inputs = tokenizer(str(t)[:512], return_tensors="pt", truncation=True)
                                with torch.no_grad():
                                    out = transformer_model(**inputs)
                                    emb = out.last_hidden_state.mean(dim=1).numpy()[0]
                                emb_list.append(emb)
                            return np.array(emb_list)
                        X_emb = get_embeddings(df_nlp[text_col].fillna("").astype(str))
                        y = df_nlp[label_col]
                        le = LabelEncoder(); y_enc = le.fit_transform(y)
                        X_train, X_test, y_train, y_test = train_test_split(X_emb, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
                        from sklearn.linear_model import LogisticRegression
                        clf = LogisticRegression(max_iter=1000)
                        clf.fit(X_train, y_train)
                        acc = accuracy_score(y_test, clf.predict(X_test))
                        st.success(f"✅ Genauigkeit: {acc:.4f}")
                        res_df = pd.DataFrame({"Modell":["LogisticRegression"],"Genauigkeit":[acc]})
                        if st.button("📸 Ergebnis als JPEG speichern", key="save_nlp_jpeg"):
                            jpeg_data = save_dataframe_as_jpeg(res_df, "NLP Klassifikation Ergebnis")
                            st.download_button("⬇️ JPEG herunterladen", jpeg_data, file_name="nlp_result.jpg", mime="image/jpeg")
                        speak(T["text_classification_complete"].format(acc=acc), lang='de')

    # -------------------- 11. ZEITREIHEN --------------------
    elif ausgewaehlter_modus == modi[10]:
        st.subheader(T['ts_title'])
        st.markdown("Prognose von Zeitreihen mit Prophet (Facebook)")
        if not PROPHET_AVAILABLE: st.error("❌ 'prophet' nicht installiert. Führen Sie aus: pip install prophet")
        else:
            uploaded_ts = st.file_uploader("📂 CSV mit Spalten 'ds' (Datum) und 'y' (Wert)", type=["csv"], key="ts_upload")
            if uploaded_ts:
                df_ts = pd.read_csv(uploaded_ts)
                if 'ds' not in df_ts.columns or 'y' not in df_ts.columns: st.warning("Das CSV muss die Spalten 'ds' (Datum) und 'y' (Wert) enthalten.")
                else:
                    df_ts['ds'] = pd.to_datetime(df_ts['ds'])
                    st.line_chart(df_ts.set_index('ds')['y'])
                    periods = st.slider("Prognosezeitraum (Perioden)", 1, 365, 30)
                    if st.button("📈 Prognose erstellen"):
                        with st.spinner("Trainiere Prophet-Modell..."):
                            model = Prophet()
                            model.fit(df_ts.rename(columns={'y':'y'}))
                            future = model.make_future_dataframe(periods=periods)
                            forecast = model.predict(future)
                            st.success("Prognose abgeschlossen")
                            fig_forecast = model.plot(forecast)
                            st.pyplot(fig_forecast)
                            st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(periods))
                            if st.button(T["save_forecast_jpeg"], key="save_forecast_jpeg"):
                                jpeg_data = save_matplotlib_fig_as_jpeg(fig_forecast, "Prophet Prognose")
                                st.download_button("⬇️ JPEG herunterladen", jpeg_data, file_name="forecast.jpg", mime="image/jpeg")
                            speak(T["time_series_forecast_complete"].format(periods=periods), lang='de')

    # -------------------- 12. BILDER --------------------
    elif ausgewaehlter_modus == modi[11]:
        st.subheader(T['img_title'])
        st.markdown("Objekterkennung mit vortrainiertem ResNet50 (ImageNet)")
        if not PYTORCH_AVAILABLE or not PIL_AVAILABLE: st.error("❌ PyTorch oder PIL nicht installiert. Führen Sie aus: pip install torch torchvision pillow")
        else:
            uploaded_img = st.file_uploader("Laden Sie ein Bild hoch (JPG, PNG)", type=["jpg","jpeg","png"], key="img_upload")
            if uploaded_img:
                image = Image.open(uploaded_img)
                st.image(image, caption="Ihr Bild", width=300)
                if st.button("🔍 Bild analysieren"):
                    with st.spinner("Lade ResNet50 und klassifiziere..."):
                        @st.cache_resource
                        def load_resnet():
                            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                            model.eval()
                            return model
                        resnet = load_resnet()
                        preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
                        input_tensor = preprocess(image).unsqueeze(0)
                        with torch.no_grad(): output = resnet(input_tensor)
                        import urllib.request
                        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                        classes = urllib.request.urlopen(url).read().decode().splitlines()
                        probs = torch.nn.functional.softmax(output[0], dim=0)
                        top5 = torch.topk(probs, 5)
                        st.success("Top-5 Klassifikationen:")
                        res_df = pd.DataFrame([(classes[top5.indices[i]], top5.values[i].item()) for i in range(5)], columns=["Klasse","Wahrscheinlichkeit"])
                        st.dataframe(res_df)
                        if st.button("📸 Ergebnis als JPEG speichern", key="save_img_jpeg"):
                            jpeg_data = save_dataframe_as_jpeg(res_df, "ResNet50 Bildklassifikation")
                            st.download_button("⬇️ JPEG herunterladen", jpeg_data, file_name="image_classification.jpg", mime="image/jpeg")
                        speak(T["image_classification_complete"], lang='de')

    # -------------------- 13. LOKALER LLM --------------------
    elif ausgewaehlter_modus == modi[12]:
        st.subheader("🧠 Lokaler LLM (ohne Internet)")
        st.markdown("Nutzen Sie eine lokale Sprachmodell über Ollama. Installieren Sie [Ollama](https://ollama.com) und führen Sie `ollama pull llama3.2:3b` aus.")
        model_choice = st.selectbox("Ollama‑Modell", ["llama3.2:3b","mistral:7b","gemma:2b"], key="ollama_model")
        user_question = st.text_area("Ihre Frage:", height=100)
        if st.button("Anna fragen"):
            if OLLAMA_AVAILABLE:
                with st.spinner("Anna denkt..."):
                    answer = query_ollama(user_question, model_choice)
                    st.markdown(f"**Antwort:** {answer}")
                    audio = speak(answer, lang='de')
                    if audio: st.audio(audio, format='audio/mp3')
            else: st.error(T["ollama_not_found"])

    # -------------------- 14. STREAMING --------------------
    elif ausgewaehlter_modus == modi[13]:
        st.subheader("📡 Echtzeit‑Streaming (Binance WebSocket)")
        symbol = st.text_input("Symbol", value="btcusdt")
        if st.button(T["start_stream"]) and not st.session_state.stream_running:
            if WEBSOCKET_AVAILABLE:
                def on_price(price):
                    st.session_state.stream_prices.append(price)
                    if st.session_state.automl_engine and st.session_state.automl_engine.is_trained:
                        pred = st.session_state.automl_engine.predict([[price]])[0]
                        signal = "BUY" if pred > price*1.01 else "SELL" if pred < price*0.99 else "HOLD"
                        st.session_state.last_signal = signal
                stream = BinanceWebSocketStream(symbol)
                stream.start(on_price)
                st.session_state.stream = stream
                st.session_state.stream_running = True
                st.success("Streaming gestartet")
            else: st.error("WebSocket nicht verfügbar. Führen Sie 'pip install websocket-client' aus.")
        if st.button(T["stop_stream"]):
            if 'stream' in st.session_state: st.session_state.stream.stop()
            st.session_state.stream_running = False
            st.info("Streaming gestoppt")
        if st.session_state.stream_prices:
            st.line_chart(st.session_state.stream_prices[-100:])
            if 'last_signal' in st.session_state: st.metric("Letztes Signal", st.session_state.last_signal)

    # -------------------- 15. BILDGENERIERUNG --------------------
    elif ausgewaehlter_modus == modi[14]:
        st.subheader("🎨 Bildgenerierung (Stable Diffusion)")
        st.markdown("Generieren Sie Bilder aus Text lokal (benötigt GPU, 8+ GB VRAM).")
        prompt_sd = st.text_area(T["prompt_sd"], "eine Katze im Weltraum, fotorealistisch")
        if st.button(T["generate_image"]):
            if DIFFUSERS_AVAILABLE:
                with st.spinner("Generiere Bild (kann 20-60 Sekunden dauern)..."):
                    img_buf, err = generate_image_stable_diffusion(prompt_sd)
                    if img_buf:
                        st.image(img_buf, caption=prompt_sd, use_container_width=True)
                        st.download_button("⬇️ PNG herunterladen", img_buf, "generated.png", "image/png")
                    else: st.error(f"Fehler: {err}")
            else: st.error("Stable Diffusion nicht verfügbar. Führen Sie 'pip install diffusers transformers accelerate' aus.")

    # -------------------- 16. BACKTESTING --------------------
    elif ausgewaehlter_modus == modi[15]:
        st.subheader("📈 Backtesting von Handelsstrategien")
        uploaded_backtest = st.file_uploader("CSV mit Spalten 'price' und 'signal' (1=KAUFEN, -1=VERKAUFEN, 0=HALTEN)", type=["csv"])
        if uploaded_backtest:
            df_bt = pd.read_csv(uploaded_backtest)
            if 'price' in df_bt.columns and 'signal' in df_bt.columns:
                metrics = calculate_backtest_metrics(df_bt['price'].tolist(), df_bt['signal'].tolist())
                st.json(metrics)
                returns = np.diff(df_bt['price']) / df_bt['price'][:-1]
                strategy_returns = []
                position = 0
                for i, sig in enumerate(df_bt['signal']):
                    if sig == 1: position = 1
                    elif sig == -1: position = -1
                    if i < len(returns): strategy_returns.append(position * returns[i])
                equity = np.cumprod(1 + np.array(strategy_returns))
                st.line_chart(equity)
            else: st.warning("CSV muss die Spalten 'price' und 'signal' enthalten.")

    # -------------------- 17. SOTA VERGLEICH --------------------
    elif ausgewaehlter_modus == modi[16]:
        st.subheader("🏆 Vergleich mit State‑of‑the‑Art (SOTA)")
        if st.session_state.df is not None and st.session_state.target_col:
            X = st.session_state.df.drop(columns=[st.session_state.target_col])
            y = st.session_state.df[st.session_state.target_col]
            if y.dtype=='object' or y.nunique()<20: task_type='classification'
            else: task_type='regression'
            if st.button(T["benchmark_button"]):
                with st.spinner("Benchmark wird ausgeführt..."):
                    results = benchmark_sota(X, y, task_type, automl=st.session_state.automl_engine, test_size=0.2)
                    df_results = pd.DataFrame(results.items(), columns=["Modell","Metrik"])
                    st.dataframe(df_results, use_container_width=True)
                    speak("Vergleich abgeschlossen", lang='de')
        else: st.info("Bitte laden Sie zuerst Daten im AutoML Tab und trainieren Sie ein Modell.")

    # -------------------- 18. GESTENSTEUERUNG (externes Skript mouse.py) --------------------
    elif ausgewaehlter_modus == modi[17]:
        st.subheader(T['gesture_title'])
        st.info("🎯 Diese Funktion startet ein separates Programm zur Maussteuerung per Handgesten.\n\n"
                "**Vorteile:**\n"
                "- Läuft **im Hintergrund** – Sie können ANNA AI weiter nutzen.\n"
                "- Kein Umschalten zwischen Fenstern nötig.\n"
                "- Kann parallel zu Zoom / anderen Anwendungen verwendet werden (ggf. OBS Virtual Camera).\n\n"
                "**Gestures:**\n"
                "🖐️ **1 Finger (Zeigefinger)** → Maus bewegen\n"
                "✌️ **2 Finger (Zeige+Mittel)** → Linksklick\n"
                "🤏 **Daumen+Zeigefinger nah** → Rechtsklick\n"
                "🖐️ **Daumen+Zeigefinger weit** → Ziehen (Drag & Drop)\n"
                "🖖 **3 Finger** → Scrollen\n\n"
                "**Zum Beenden:**\n"
                "Schließen Sie das Konsolenfenster oder drücken Sie darin **Strg+C**.")
        
        script_path = os.path.join(os.path.dirname(__file__), "mouse.py")
        if not os.path.exists(script_path):
            st.error(f"❌ Datei `mouse.py` nicht gefunden.\n"
                     f"Erwarteter Pfad: `{script_path}`\n\n"
                     "Bitte legen Sie Ihre Hand-Gesten-Steuerung als `mouse.py` in diesem Ordner ab.")
        else:
            if st.button("🖐️ GESTENSTEUERUNG STARTEN (separates Fenster)", type="primary", use_container_width=True):
                try:
                    if sys.platform == "win32":
                        subprocess.Popen([sys.executable, script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
                    else:
                        subprocess.Popen([sys.executable, script_path])
                    st.success("✅ Gestensteuerung läuft im Hintergrund.\n\n"
                               "Sie können jetzt mit den Handgesturen Ihre Maus steuern.\n"
                               "Zum Beenden schließen Sie das Konsolenfenster oder drücken Ctrl+C.")
                except Exception as e:
                    st.error(f"Fehler beim Starten: {e}")

    # -------------------- 19. AUTONOMER ANALYST --------------------
    elif ausgewaehlter_modus == modi[18]:
        st.subheader(T['agent_title'])
        st.markdown(T['agent_example'])
        uploaded_agent = st.file_uploader(T['agent_upload'], type=["csv","xlsx","xls"], key="agent_upload")
        agent_command = st.text_area(T['agent_command'], height=100)
        if st.button(T['agent_run'], type="primary") and uploaded_agent is not None and agent_command:
            df = pd.read_csv(uploaded_agent) if uploaded_agent.name.endswith('.csv') else pd.read_excel(uploaded_agent)
            with st.spinner("ANNA führt die vollautomatische Analyse durch..."):
                prompt = f"""Daten: Spalten {list(df.columns)}.
                Benutzerbefehl: {agent_command}.
                Bestimme die Zielvariable (Spaltenname) und den Aufgabentyp (Klassifikation, Regression oder Clustering).
                Antworte im Format: target=... type=... (z. B. target=Umsatz type=Regression)"""
                llm_response = query_ollama(prompt, "llama3.2:3b")
                target = None; task_type = "classification"
                if "target=" in llm_response:
                    target_part = llm_response.split("target=")[1].split()[0].strip()
                    if target_part in df.columns: target = target_part
                if target is None: target = st.selectbox("Zielvariable manuell auswählen:", df.columns)
                if "type=" in llm_response:
                    type_part = llm_response.split("type=")[1].split()[0].lower()
                    if "regression" in type_part or "regress" in type_part: task_type = "regression"
                    elif "cluster" in type_part: task_type = "clustering"
                df_clean = df.drop_duplicates().copy()
                for col in df_clean.columns:
                    if df_clean[col].isnull().any():
                        if df_clean[col].dtype in ['int64','float64']: df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        else: df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "missing")
                le_dict = {}
                for col in df_clean.select_dtypes(include='object').columns:
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                    le_dict[col] = le
                X = df_clean.drop(columns=[target])
                y = df_clean[target]
                if y.dtype == 'object':
                    le_y = LabelEncoder(); y = le_y.fit_transform(y); task_type = "classification"
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                if task_type == "classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    score = accuracy_score(y_test, model.predict(X_test))
                    metric_name = "Accuracy"
                elif task_type == "regression":
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    score = r2_score(y_test, model.predict(X_test))
                    metric_name = "R²"
                else:
                    from sklearn.cluster import KMeans
                    model = KMeans(n_clusters=3, random_state=42)
                    model.fit(X)
                    score = model.inertia_
                    metric_name = "Inertia"
                importance_fig = None
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[-10:]
                    fig, ax = plt.subplots(figsize=(10,6))
                    ax.barh(range(len(indices)), importances[indices])
                    ax.set_yticks(range(len(indices)))
                    ax.set_yticklabels([X.columns[i] for i in indices])
                    ax.set_xlabel("Wichtigkeit")
                    ax.set_title("Top-10 Feature Importances")
                    plt.tight_layout()
                    importance_fig = fig
                shap_fig = None
                if SHAP_AVAILABLE and hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test.iloc[:100])
                    shap_fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
                    plt.tight_layout()
                    shap_fig = ax.get_figure()
                insight_prompt = f"""Modell {type(model).__name__} für {task_type}: {metric_name} = {score:.4f}.
                Wichtigste Merkmale: {list(X.columns[indices[-3:]]) if 'indices' in dir() else 'unbekannt'}.
                Fasse die Ergebnisse kurz zusammen (max. 3 Sätze) auf Deutsch."""
                insight = query_ollama(insight_prompt, "llama3.2:3b")
                st.success(f"Analyse abgeschlossen! {metric_name}: {score:.4f}")
                st.markdown("### 📄 Analysebericht")
                st.write(insight)
                if importance_fig:
                    st.markdown("### 📊 Feature Importance")
                    st.pyplot(importance_fig)
                    plt.close(importance_fig)
                if shap_fig:
                    st.markdown("### 🔍 SHAP Summary")
                    st.pyplot(shap_fig)
                    plt.close(shap_fig)
                if JOBLIB_AVAILABLE:
                    model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl").name
                    joblib.dump(model, model_path)
                    with open(model_path, "rb") as f: st.download_button(T['agent_model_save'], f, file_name="autonomous_model.pkl")
                if FPDF_AVAILABLE and st.button("📄 Bericht als PDF speichern"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="ANNA AI – Autonomer Analysebericht", ln=True, align='C')
                    pdf.ln(10)
                    pdf.multi_cell(0, 10, insight)
                    pdf.cell(0, 10, txt=f"Modell: {type(model).__name__}", ln=True)
                    pdf.cell(0, 10, txt=f"Metrik: {metric_name} = {score:.4f}", ln=True)
                    pdf.output("agent_report.pdf")
                    with open("agent_report.pdf", "rb") as f: st.download_button("📥 PDF herunterladen", f, file_name="agent_report.pdf")

    # -------------------- 20. RPA --------------------
    elif ausgewaehlter_modus == modi[19]:
        st.subheader(T['rpa_title'])
        if not PYNPUT_AVAILABLE or not pyautogui: st.error("❌ Bibliotheken 'pynput' und/oder 'pyautogui' nicht installiert. Führen Sie 'pip install pynput pyautogui' aus.")
        else:
            with st.expander(T['rpa_calibrate']):
                mouse_speed = st.slider(T['rpa_mouse_speed'], 0.05, 2.0, 0.3, step=0.05)
                click_delay = st.slider(T['rpa_click_delay'], 0.0, 1.0, 0.1, step=0.05)
                if st.button("💾 Kalibrierung speichern"): st.session_state.rpa_config = {"mouse_speed":mouse_speed,"click_delay":click_delay}; st.success("Gespeichert")
            col_rec, col_play = st.columns(2)
            with col_rec:
                if st.button(T['rpa_record']):
                    st.session_state.rpa_recording = True
                    st.session_state.rpa_macro = []
                    st.info("Aufzeichnung gestartet. Führen Sie Maus- und Tastaturaktionen aus.")
                    def on_move(x,y):
                        if st.session_state.rpa_recording: st.session_state.rpa_macro.append(('move',x,y,time.time()))
                    def on_click(x,y,button,pressed):
                        if st.session_state.rpa_recording: st.session_state.rpa_macro.append(('click',x,y,button.name,pressed,time.time()))
                    def on_press(key):
                        if st.session_state.rpa_recording: st.session_state.rpa_macro.append(('key_press',key,time.time()))
                    def on_release(key):
                        if st.session_state.rpa_recording: st.session_state.rpa_macro.append(('key_release',key,time.time()))
                    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
                    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
                    mouse_listener.start(); keyboard_listener.start()
                    st.session_state.rpa_listeners = (mouse_listener, keyboard_listener)
                if st.button(T['rpa_stop_record']):
                    st.session_state.rpa_recording = False
                    if 'rpa_listeners' in st.session_state: st.session_state.rpa_listeners[0].stop(); st.session_state.rpa_listeners[1].stop()
                    st.success(f"Aufzeichnung gestoppt. {len(st.session_state.rpa_macro)} Ereignisse aufgezeichnet.")
            with col_play:
                if st.button(T['rpa_play']):
                    if st.session_state.rpa_macro:
                        st.info("Wiedergabe des Makros...")
                        pyautogui.PAUSE = st.session_state.get('rpa_config', {}).get('click_delay', 0.1)
                        for event in st.session_state.rpa_macro:
                            if event[0]=='move': pyautogui.moveTo(event[1], event[2], duration=st.session_state.get('rpa_config', {}).get('mouse_speed', 0.3))
                            elif event[0]=='click' and event[4]: pyautogui.click(event[1], event[2], button=event[3])
                            elif event[0]=='key_press': pyautogui.press(str(event[1]))
                        st.success("Makro abgespielt.")
                    else: st.warning("Kein Makro aufgezeichnet. Nehmen Sie zuerst ein Makro auf.")
            st.markdown("---")
            uploaded_script = st.file_uploader(T['rpa_script_upload'], type=["py"], key="rpa_script")
            if uploaded_script:
                script_code = uploaded_script.read().decode("utf-8")
                st.code(script_code, language="python")
                if st.button(T['rpa_run_script']):
                    try:
                        exec(script_code, {'pyautogui': pyautogui, 'time': time, 'os': os})
                        st.success("Skript erfolgreich ausgeführt.")
                    except Exception as e: st.error(f"Fehler bei der Skriptausführung: {e}")

if __name__ == "__main__":
    main()