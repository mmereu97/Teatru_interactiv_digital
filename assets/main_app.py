# main_app.py

import sys
import os

# --- COD DE DEBUGGING PENTRU CALEA PROIECTULUI ---
print("--- START DEBUGGING PATH ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Directorul scriptului (script_dir): {script_dir}")

working_dir = os.getcwd()
print(f"Directorul de lucru (working_dir): {working_dir}")

print("Căile de sistem ale lui Python (sys.path):")
for path in sys.path:
    print(f"  - {path}")
print("--- END DEBUGGING PATH ---")
print("\n" * 2)


# Aici încep importurile originale...

import time
import json
import math
import random
import re

# --- Importuri PySide6 ---
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QLineEdit, QPushButton, QTextEdit, QTabWidget, QScrollArea,
                               QSlider, QProgressBar, QGroupBox, QFormLayout, QCheckBox, QComboBox,
                               QListWidget, QListWidgetItem, QSpinBox)
from PySide6.QtGui import QPixmap, QImage, QFontDatabase, QFont
from PySide6.QtCore import QThread, Signal, QObject, QTimer, Qt, QPoint, QRect

# --- Importuri Librării Externe ---
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import edge_tts
import pygame
import speech_recognition as sr
import torch
import sounddevice as sd
import collections
import wave
import tempfile
import cv2
import asyncio
import numpy as np
import glob
from pathlib import Path

# --- Importuri din Proiectul Nostru (NOUA ARHITECTURĂ) ---
from managers.scene_manager import SceneManager
from managers.character_manager import CharacterManager
from characters.animators import ANIMATOR_REGISTRY, BreathingAnimator, BlinkingAnimator, EmotionAnimator


# =================================================================================
# SISTEM CONFIGURABIL DE LOGGING
# =================================================================================

LOG_CONFIG = {
    "audio": False,        # 📊 Niveluri audio periodic (zgomotos!)
    "webcam": False,       # 📷 Frame counts la fiecare 300 frames
    "vad": False,          # 🟢🔴 Silero VAD verbose (începuturi/sfârșituri vorbire)
    "animator": False,     # 👀 Clipit, respirație (foarte zgomotos!)
    "emotion": False,
    "gemini_debug": False, # 🔬 Debug complet Gemini worker creation
    "echo": False,         # 🔍 Echo detection similarity checks
    "cleanup": True,       # 🧹 Thread cleanup operations
    "router": True,        # 🚦 Intent routing logic
    "scene": True,         # 🌆 Scene changes
    "character": True,     # 🎭 Character add/remove/move
    "tts": True,           # 🔊 Text-to-speech lifecycle
    "intent": True,        # 🤖 Intent classification
    "sync": True,          # 🎬 Audio-visual sync
    "mute": True,          # 🔇 Microphone muting
    "app": True,           # 🚀 Application lifecycle
    "filler": True,        # 🔊 Filler sounds
    "memory": True,        # 🧠 Greeting memory
    "process": True,       # 🎵 Audio processing
    "transcription": True, # 🗣️ Speech transcription
    "position": True,  # ⭐ ADAUGĂ ACEASTĂ LINIE (sau schimbă False în True)
    "gaze": False,  # ⭐ ADAUGĂ ACEASTĂ LINIE
    "semafor": False,
    "curriculum": False,     # 📚 Detalii despre încărcarea fiecărui tier și întrebare
}

# Funcție wrapper pentru logging controlat
START_TIME = time.time()

def log_timestamp(message, category="app"):
    """
    Logging cu filtrare pe categorii.
    
    Args:
        message (str): Mesajul de logat
        category (str): Categoria de log (default: "app")
    
    Exemple:
        log_timestamp("Pornire aplicație", "app")
        log_timestamp("Nivel audio: 3200", "audio")
    """
    if LOG_CONFIG.get(category, True):
        elapsed = time.time() - START_TIME
        print(f"[{elapsed:8.3f}s] {message}")


# ⭐ SETĂM CONFIG-UL PENTRU TOATE MODULELE EXTERNE
from characters import animators
from characters import base_character  # <-- Adăugați acest import
from managers import scene_manager, character_manager

animators.set_log_config(LOG_CONFIG)
base_character.set_log_config(LOG_CONFIG) # <-- Adăugați această linie
scene_manager.set_log_config(LOG_CONFIG)
character_manager.set_log_config(LOG_CONFIG)

# =================================================================================
# UTILITARE GLOBALE
# =================================================================================

def cleanup_temp_files():
    """Șterge fișierele temp_speech... orfane din folderul rădăcină."""
    log_timestamp("🧹 [CLEANUP] Se caută fișiere temporare vechi...", "cleanup")
    deleted_count = 0
    current_dir = os.getcwd()
    
    for filename in os.listdir(current_dir):
        if os.path.isfile(os.path.join(current_dir, filename)) and \
           filename.startswith("temp_speech_") and \
           filename.endswith(".mp3"):
            try:
                os.remove(os.path.join(current_dir, filename))
                log_timestamp(f"  -> Șters: {filename}", "cleanup")
                deleted_count += 1
            except Exception as e:
                log_timestamp(f"  -> ⚠️ Eroare la ștergerea {filename}: {e}", "cleanup")
    
    if deleted_count > 0:
        log_timestamp(f"✅ [CLEANUP] Curățenie finalizată. {deleted_count} fișiere șterse.", "cleanup")
    else:
        log_timestamp("✅ [CLEANUP] Niciun fișier temporar de șters.", "cleanup")

# Păstrăm configurarea API
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    log_timestamp("❌ [EROARE CRITICĂ] Cheia GOOGLE_API_KEY nu a fost găsită!", "app")
genai.configure(api_key=GOOGLE_API_KEY)


def save_config(config, config_path="config.json"):
    """Salvează configurația în fișier JSON."""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        log_timestamp(f"✅ [CONFIG] Configurație salvată în '{config_path}'", "app")
    except Exception as e:
        log_timestamp(f"❌ [CONFIG] Eroare la salvare: {e}", "app")

def load_config(config_path="config.json"):
    """Încarcă configurația din fișier JSON."""
    default_config = {
        "auto_calibrate_on_mic_start": False,
        "auto_start_mic_with_conversation": True,
        "conversation_without_camera": False,
        "enable_echo_cancellation": True,
        "enable_filler_sounds": False, # <-- ADAUGĂ ACEASTĂ LINIE
        "threshold": 400,
        "margin_percent": 20,
        "pause_duration": 2.0,
        "max_speech_duration": 15, # <-- ADAUGĂ ACEASTĂ LINIE
        "window_geometry": None,  # {"x": 50, "y": 50, "width": 1920, "height": 1080}
        
        # --- SETĂRI NOI ---
        "subtitle_font_size": 26,
        "rina_language_code": "en",
        "subtitle_mode": "original",
        "ai_model_name": "models/gemini-flash-lite-latest" # <-- ADAUGĂ ACEASTĂ LINIE
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Merge cu default pentru câmpuri noi
                default_config.update(loaded_config)
                log_timestamp(f"✅ [CONFIG] Configurație încărcată din '{config_path}'", "app")
        else:
            log_timestamp(f"⚠️ [CONFIG] Fișier config inexistent, se folosește default", "app")
    except Exception as e:
        log_timestamp(f"❌ [CONFIG] Eroare la încărcare: {e}, se folosește default", "app")
    
    return default_config

# =================================================================================
# WORKER-I (QThread) - Portare 1:1
# Aceste clase rămân aproape identice, deoarece logica lor este deja
# bine încapsulată și nu depinde de arhitectura personajelor.
# Le copiem direct din fișierul vechi.
# =================================================================================

# [COPIAȚI ȘI LIPIȚI AICI, FĂRĂ MODIFICĂRI, URMĂTOARELE CLASE DIN main_app_backup.py]:
# =================================================================================
# WORKER-I ASINCRONE
# =================================================================================

class TTSWorker(QObject):
    finished = Signal()
    audio_ready = Signal(str, float)
    
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.voice = "ro-RO-EmilNeural"
        self.output_file = f"temp_speech_{int(time.time()*1000)}.mp3"
        self.actual_duration = 0
        
    def run(self):
        log_timestamp(f"🔊 [TTS] TTSWorker pornit pentru: '{self.text[:50]}...'", "tts")
        try:
            log_timestamp("🔊 [TTS] Generez audio cu Edge TTS...", "tts")
            asyncio.run(self._async_speak())
            log_timestamp(f"🔊 [TTS] ✅ Redare audio și ciclu async terminate. Durată reală: {self.actual_duration:.2f}s", "tts")
        except Exception as e:
            log_timestamp(f"🔊 [TTS] ❌ Eroare în timpul rulării TTSWorker: {e}", "tts")
        finally:
            log_timestamp("🔊 [TTS] Worker-ul își încheie execuția. Nu se mai șterge fișierul audio.", "tts")
            self.finished.emit()
    
    async def _async_speak(self):
        log_timestamp(f"🔊 [TTS] Salvez în: {self.output_file}", "tts")
        communicate = edge_tts.Communicate(self.text, self.voice)
        await communicate.save(self.output_file)
        
        sound = pygame.mixer.Sound(self.output_file)
        self.actual_duration = sound.get_length()
        log_timestamp(f"🔊 [TTS] ⏱️ Durată REALĂ măsurată: {self.actual_duration:.2f}s", "tts")
        
        self.audio_ready.emit(self.output_file, self.actual_duration)
        log_timestamp(f"🔊 [TTS] ✅ Signal 'audio_ready' emis!", "tts")
        
        await asyncio.sleep(0.1)
        
        log_timestamp("🔊 [TTS] ▶️ START redare audio!", "tts")
        pygame.mixer.music.load(self.output_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
        
        log_timestamp("🔊 [TTS] ⏹️ STOP redare - terminat.", "tts")
        
        log_timestamp("🔊 [TTS] Eliberez resursa Pygame (stop/unload)...", "tts")
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        await asyncio.sleep(0.2)

class WebcamWorker(QObject):
    frame_ready = Signal(QImage)
    finished = Signal()
    
    def __init__(self):
        super().__init__()
        self._is_running = True
        self.last_frame = None

    def run(self):
        log_timestamp("📷 [WEBCAM] Worker pornit.", "webcam")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_timestamp("📷 [WEBCAM] ❌ Nu se poate deschide camera!", "webcam")
            self._is_running = False
            
        frame_count = 0
        while self._is_running:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_frame = rgb_image.copy()
                
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.frame_ready.emit(qt_image.copy())
                
                frame_count += 1
                if frame_count % 300 == 0:
                    log_timestamp(f"📷 [WEBCAM] Frame #{frame_count} OK", "webcam")
            time.sleep(0.03)

        cap.release()
        self.finished.emit()
        log_timestamp("📷 [WEBCAM] Worker oprit.", "webcam")

    def stop(self):
        self._is_running = False

class GeminiWorker(QObject):
    response_ready = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()
    
    def __init__(self, system_prompt, image_data, question_text, model_name):
        super().__init__()
        self.system_prompt = system_prompt
        self.image_data = image_data
        self.question_text = question_text
        self.model = genai.GenerativeModel(model_name)

    def run(self):
        log_timestamp("🤖 [GEMINI] Worker pornit.", "gemini_debug")
        try:
            log_timestamp("🤖 [GEMINI] Convertesc frame în PIL Image...", "gemini_debug")
            pil_image = Image.fromarray(self.image_data)
            
            prompt_parts = [
                self.system_prompt,
                pil_image,
                f"Utilizator: {self.question_text}"
            ]
            
            log_timestamp(f"🤖 [GEMINI] Trimit request pentru: '{self.question_text}'", "gemini_debug")
            response = self.model.generate_content(prompt_parts)
            
            if response.text:
                log_timestamp(f"🤖 [GEMINI] ✅ Răspuns: '{response.text[:100]}...'", "gemini_debug")
                self.response_ready.emit(response.text)
            else:
                log_timestamp("🤖 [GEMINI] ⚠️ Răspuns gol", "gemini_debug")
                self.error_occurred.emit("Răspuns gol de la AI.")
        except Exception as e:
            log_timestamp(f"🤖 [GEMINI] ❌ Eroare: {e}", "gemini_debug")
            self.error_occurred.emit(f"Eroare: {e}")

class GeminiWorkerTextOnly(QObject):
    """Worker pentru Gemini fără cameră - doar text"""
    response_ready = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()
    
    def __init__(self, system_prompt, question_text, model_name):
        super().__init__()
        self.system_prompt = system_prompt
        self.question_text = question_text
        self.model = genai.GenerativeModel(model_name)

    def run(self):
        log_timestamp("🤖 [GEMINI TEXT-ONLY] Worker pornit.", "gemini_debug")
        try:
            prompt_parts = [
                self.system_prompt,
                f"\nUtilizator: {self.question_text}"
            ]

            log_timestamp(f"🤖 [GEMINI TEXT-ONLY] Request: '{self.question_text}'", "gemini_debug")
            response = self.model.generate_content(prompt_parts)
            
            if response.text:
                log_timestamp(f"🤖 [GEMINI TEXT-ONLY] ✅ Răspuns: '{response.text[:100]}...'", "gemini_debug")
                self.response_ready.emit(response.text)
            else:
                log_timestamp("🤖 [GEMINI TEXT-ONLY] ⚠️ Răspuns gol", "gemini_debug")
                self.error_occurred.emit("Răspuns gol de la AI.")
        except Exception as e:
            log_timestamp(f"🤖 [GEMINI TEXT-ONLY] ❌ Eroare: {e}", "gemini_debug")
            self.error_occurred.emit(f"Eroare: {e}")


class LearningSessionWorker(QObject):
    """
    Worker dedicat pentru sesiuni de învățare în Modul Profesor.
    Gestionează un singur ciclu: primește răspunsul elevului, evaluează, 
    decide următoarea acțiune și returnează feedback-ul.
    """
    response_ready = Signal(dict)  # Dict cu: outcome, text_to_speak, etc.
    error_occurred = Signal(str)
    finished = Signal()
    
    def __init__(self, mega_prompt):
        """
        Constructor MINIMAL - nu inițializăm resurse externe aici!
        
        Args:
            mega_prompt (str): Prompt-ul complet construit de MainApp
        """
        super().__init__()
        self.mega_prompt = mega_prompt
        self.model = None  # Va fi inițializat în run()
    
    def run(self):
        """
        Execuție în thread separat. Aici inițializăm modelul și facem apelul.
        """
        log_timestamp("🎓 [LEARNING] LearningSessionWorker pornit.", "app")
        try:
            # ⭐ CRUCIAL: Inițializăm modelul AICI, în thread-ul worker-ului
            log_timestamp("🎓 [LEARNING] Inițializez modelul Gemini...", "app")
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
            
            log_timestamp(f"🎓 [LEARNING] Trimit mega-prompt către AI (lungime: {len(self.mega_prompt)} caractere)", "app")
            response = self.model.generate_content(self.mega_prompt)
            
            if not response.text:
                log_timestamp("🎓 [LEARNING] ⚠️ Răspuns gol de la AI!", "app")
                self.error_occurred.emit("Răspuns gol de la AI.")
                return
            
            raw_response = response.text.strip()
            log_timestamp(f"🎓 [LEARNING] Răspuns brut de la AI: '{raw_response[:200]}...'", "app")
            
            # Parsare JSON
            # Curățăm de markdown dacă există
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.startswith("```"):
                raw_response = raw_response[3:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            raw_response = raw_response.strip()
            
            try:
                result = json.loads(raw_response)
                log_timestamp(f"🎓 [LEARNING] ✅ JSON parsat cu succes: {result}", "app")
                self.response_ready.emit(result)
            except json.JSONDecodeError as e:
                log_timestamp(f"🎓 [LEARNING] ❌ Eroare parsare JSON: {e}", "app")
                log_timestamp(f"🎓 [LEARNING] Răspuns problematic: '{raw_response}'", "app")
                self.error_occurred.emit(f"Eroare parsare JSON: {e}")
        
        except Exception as e:
            log_timestamp(f"🎓 [LEARNING] ❌ Eroare în worker: {e}", "app")
            self.error_occurred.emit(str(e))
        
        finally:
            log_timestamp("🎓 [LEARNING] Worker își încheie execuția.", "app")
            self.finished.emit()
# =================================================================================
# AUDIO MONITORING + VOICE DETECTION
# =================================================================================

class IntentClassifierWorker(QObject):
    intent_classified = Signal(dict)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, text):
        super().__init__()
        self.text = text  # ⭐ CRUCIAL - salvăm textul!

    def run(self):
        log_timestamp("🤖 [INTENT] Worker de clasificare a intenției pornit.", "intent")
        try:
            prompt_template = """

Ești un asistent care analizează textul unui utilizator și îl clasifică. Răspunde DOAR cu un obiect JSON valid.

--- REGULĂ CRITICĂ DE BAZĂ ---
Regulile de mai jos sunt pentru textul în limba ROMÂNĂ. Dacă textul utilizatorului este într-o ALTĂ LIMBĂ (Engleză, Franceză, etc.), este aproape întotdeauna o 'conversation'. Nu încerca să aplici reguli de 'travel' sau 'summon' la text străin decât dacă este extrem de evident.
---

Categoriile posibile pentru 'intent' sunt:

1. 'travel_with_character': Utilizatorul vrea să MEARGĂ ÎMPREUNĂ către o altă scenă
   - Pattern: "[Nume], hai să mergem la [scenă]", "Mergem împreună la [scenă]"
   - Cuvinte cheie: "hai să mergem", "mergem împreună", "la [loc]"
   - Exemple: 
     * "Cucuvel, hai la școală" → {{"intent": "travel_with_character", "character": "cucuvel_owl", "scene": "scoala"}}

2. 'travel_solo': Utilizatorul SINGUR merge în altă scenă (fără alte personaje)
   - Pattern: "merg la [scenă]", "vreau să merg la [scenă]", "aș vrea să merg la [scenă]"
   - CUVINTE CHEIE: "merg" (eu singur), "vreau să merg" (eu), "ma duc" (eu)
   - IMPORTANT: Dacă EU (utilizatorul) vreau să merg → travel_solo
   - IMPORTANT: Dacă conține "împreună" sau "hai să" → travel_with_character
   - Exemple:
     * "Merg acasă" → {{"intent": "travel_solo", "scene": "acasa"}}
     * "Vreau eu să merg acasă" → {{"intent": "travel_solo", "scene": "acasa"}}
     * "Mă duc la școală" → {{"intent": "travel_solo", "scene": "scoala"}}

3. 'summon_character': Utilizatorul CHEAMĂ un personaj să VINĂ
   - Pattern: "[Nume], vino aici/și tu"
   - CUVINTE CHEIE OBLIGATORII: "vino", "hai", "cheamă", "apare"
   - IMPORTANT: Simplă adresare fără "vino" → NU e summon, e conversation!
   - Exemple: 
     * "Cucuvel, vino aici" → {{"intent": "summon_character", "character": "cucuvel_owl"}}
     * "Profesor, vino și tu" → {{"intent": "summon_character", "character": "cucuvel_owl"}}
     * "Bună ziua profesor" → {{"intent": "conversation"}} (NU e summon!)

4. 'send_character': Utilizatorul TRIMITE un personaj în altă scenă (fără el)
   - Pattern: "[Nume], mergi/du-te/pleacă la [scenă]" SAU "du-te [Nume] la [scenă]"
   - CUVINTE CHEIE OBLIGATORII: TU (personajul) + "du-te"/"mergi"/"pleacă" + destinație
   - IMPORTANT: Dacă TU (personajul) trebuie să meargă → send_character
   - IMPORTANT: Dacă EU (utilizatorul) vreau să merg → travel_solo
   - Diferența CRITICĂ:
     * "Merg acasă" (EU merg) → travel_solo ✅
     * "Du-te acasă" (TU mergi) → send_character ✅
   - Exemple:
     * "Cucuvel, du-te acasă" → {{"intent": "send_character", "character": "cucuvel_owl", "scene": "acasa"}}
     * "Du-te singur în poiană" (către speaker curent) → {{"intent": "send_character", "character": "cucuvel_owl", "scene": "poiana"}}
     * "Mergi la școală" (comandă către speaker) → {{"intent": "send_character", "character": "cucuvel_owl", "scene": "scoala"}}

5. 'conversation': Orice altceva - întrebări, comentarii, salutări, discuții
   - Include: salutări, întrebări, comentarii, adresări simple
   - Exemple:
     * "Bună ziua" → {{"intent": "conversation"}}
     * "Ce mai faci?" → {{"intent": "conversation"}}
     * "Nu am înțeles ce ai spus" → {{"intent": "conversation"}}
     * "Ce înseamnă asta?" → {{"intent": "conversation"}}

6. 'translation_request': Utilizatorul cere EXPLICIT traducerea ultimei replici folosind un cuvânt cheie specific.
   - REGULĂ STRICTĂ: Se activează DOAR dacă textul conține cuvântul "traducere" sau "tradu".
   - IMPORTANT: Fraze precum "nu am înțeles" sau "ce vrei să spui?" FĂRĂ cuvântul "traducere" sunt considerate 'conversation', NU 'translation_request'.
   - Exemple:
     * "Traducere" → {{"intent": "translation_request"}}
     * "Poți să faci o traducere, te rog?" → {{"intent": "translation_request"}}
     * "Tradu ce a spus." → {{"intent": "translation_request"}}
     * "Nu am înțeles" → {{"intent": "conversation"}}
     * "Ce înseamnă?" → {{"intent": "conversation"}}

7. 'start_learning': Utilizatorul cere să ÎNCEAPĂ o lecție/sesiune de învățare
   - Pattern: "vreau să învăț [subiect]", "hai să învățăm", "începe lecția"
   - CUVINTE CHEIE: "învăț", "învățăm", "lecție", "lecția", "începe", "hai să studiem"
   - Exemple:
     * "Vreau să învăț culorile" → {{"intent": "start_learning", "subject": "culori"}}
     * "Hai să învățăm" → {{"intent": "start_learning", "subject": ""}}
     * "Începe lecția de matematică" → {{"intent": "start_learning", "subject": "matematică"}}

8. 'exit_teacher_mode': Utilizatorul cere EXPLICIT să iasă din modul de învățare
   - Pattern: "stop", "pauză", "oprește lecția", "vreau să mă opresc"
   - CUVINTE CHEIE: "stop", "pauză", "pauza", "oprește", "opreste", "gata cu lecția"
   - IMPORTANT: Această intenție are sens DOAR în contextul unei sesiuni active de învățare
   - Exemple:
     * "Stop lecție" → {{"intent": "exit_teacher_mode"}}
     * "Pauză, te rog" → {{"intent": "exit_teacher_mode"}}
     * "Vreau să mă opresc" → {{"intent": "exit_teacher_mode"}}
     * "Gata cu învățatul" → {{"intent": "exit_teacher_mode"}}

---
PARAMETRI:

Scene valide: 'scoala', 'acasa', 'poiana', 'pajiste'

Personaje valide:
- 'cucuvel' sau 'cucuvel_owl': Profesorul Cucuvel (bufniță)
- 'rina' sau 'rina_cat': Rina (pisică)

Detectează variații: "profesor", "dom profesor", "domnul profesor" → referință la 'cucuvel_owl'
Detectează variații: "pisica", "pisico" → referință la 'rina_cat'

---
REGULI CRITICE:
1. Dacă textul conține doar SALUT/ÎNTREBARE fără "vino"/"mergi"/"hai" → conversation
2. Simplă menționare a numelui/titlului unui personaj NU înseamnă summon
3. Pentru summon trebuie EXPLICIT: "vino", "hai aici", "cheamă"
4. Pentru travel trebuie EXPLICIT: "să mergem", "hai la", "merg la"
5. Pentru send trebuie: "du-te"/"du te"/"mergi"/"pleacă" + destinație
6. Pentru translation trebuie: "traducere" sau "tradu". Orice altă formă de neînțelegere este 'conversation'.
7. Dacă e DOAR comandă de plecare (fără "împreună"/"hai să") → send către vorbitorul activ
8. Pentru start_learning trebuie: "învăț", "învățăm", "lecție" sau variații
9. Pentru exit_teacher_mode trebuie: "stop", "pauză", "oprește" în contextul învățării

---
ACUM ANALIZEAZĂ:

Text utilizator: "{text}"

Răspunde DOAR cu JSON, fără alte explicații:
"""
           
            
            prompt = prompt_template.format(text=self.text)
            
            # Generare răspuns de la AI
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content(prompt)
            raw_response = response.text.strip()
            
            log_timestamp(f"🤖 [INTENT] Răspuns brut de la AI: '{raw_response}'", "intent")
            
            # Parsare JSON
            # Curățăm răspunsul de eventuale backticks sau markdown
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.startswith("```"):
                raw_response = raw_response[3:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            raw_response = raw_response.strip()
            
            # Parse JSON
            try:
                intent_data = json.loads(raw_response)
            except json.JSONDecodeError as e:
                log_timestamp(f"🤖 [INTENT] ⚠️ Eroare parsare JSON: {e}. Fallback la conversation.", "intent")
                intent_data = {"intent": "conversation"}
            
            log_timestamp(f"🤖 [INTENT] ✅ Intenție clasificată: {intent_data}", "intent")
            
            # Emitem semnalul cu datele clasificate
            self.intent_classified.emit(intent_data)
            
        except Exception as e:
            log_timestamp(f"🤖 [INTENT] ❌ Eroare în worker-ul de intenție: {e}", "intent")
            # Fallback: tratăm ca și conversație normală
            self.error_occurred.emit(str(e))
            self.intent_classified.emit({"intent": "conversation"})
        
        finally:
            log_timestamp("🤖 [INTENT] Worker-ul de intenție și-a terminat execuția.", "intent")
            self.finished.emit()

class ContinuousVoiceWorker(QObject):
    language_lock_requested = Signal(str)
    speech_activity_changed = Signal(bool) # True = a început vorbirea, False = s-a terminat
    pause_progress_updated = Signal(int)   # Progresul clepsidrei (0-100)
    speech_time_updated = Signal(float) # <-- ADAUGĂ ACEASTĂ LINIE
    
    transcription_ready = Signal(str)
    status_changed = Signal(str)
    calibration_done = Signal(float)
    audio_level_changed = Signal(float)
    
    def __init__(self, threshold, pause_duration, margin_percent, max_speech_duration, enable_echo_cancellation):
        super().__init__()
        self._is_running = False
        self._is_muted = False
        self.enable_echo_cancellation = enable_echo_cancellation # <-- ACUM PRIMIM VALOAREA CORECTĂ
        log_timestamp(f"🎤 [VAD INIT] Anulare Ecoul setată la: {self.enable_echo_cancellation}", "app") # Log de verificare
        self.current_lock_mode = 'auto'
        self.primary_language = "ro-RO"
        self.secondary_language = "ro-RO"
        
        # Parametri audio
        self.sample_rate = 16000
        self.frame_duration = 32
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

        # Silero VAD setup
        log_timestamp("🧠 [SILERO VAD] Încărcare model neural...", "vad")
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad',
                force_reload=False, onnx=False
            )
            (self.get_speech_timestamps, _, _, _, _) = utils
            log_timestamp("✅ [SILERO VAD] Model încărcat cu succes!", "vad")
        except Exception as e:
            log_timestamp(f"❌ [SILERO VAD] Eroare la încărcare: {e}", "vad")
            raise
        
        # --- BLOC MUTAT MAI SUS ---
        # Parametri detecție
        self.threshold = threshold
        self.pause_duration = pause_duration
        self.margin_percent = margin_percent
        self.max_speech_duration = max_speech_duration
        
        # Praguri Silero
        self.speech_threshold = 0.5
        self.silence_threshold = 0.3
        # --- SFÂRȘIT BLOC MUTAT ---

        # Calcule interne pe baza parametrilor
        self.silence_frames_threshold = int((self.pause_duration * 1000) / self.frame_duration)
        self.MAX_SPEECH_FRAMES = int(self.max_speech_duration * 1000 / self.frame_duration)
        
        # Buffer pentru detectare voce
        self.ring_buffer_size = int(self.sample_rate * 0.5)
        self.ring_buffer = collections.deque(maxlen=self.ring_buffer_size // self.frame_size)
        
        # State tracking
        self.is_speech_active = False
        self.frames_since_silence = 0
        
        # Acumulare audio pentru transcriere
        self.speech_frames = []
        
        # Pentru echo detection
        self.last_ai_text = ""
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        
        # --- LOG-URILE SUNT ACUM LA FINAL, CÂND TOATE VARIABILELE EXISTĂ ---
        log_timestamp("🎤 [VAD INIT] Silero VAD inițializat", "vad")
        log_timestamp(f"🎤 [VAD INIT] Sample rate: {self.sample_rate}Hz", "vad")
        log_timestamp(f"🎤 [VAD INIT] Frame duration: {self.frame_duration}ms", "vad")
        log_timestamp(f"🎤 [VAD INIT] Frame size: {self.frame_size} samples", "vad")
        log_timestamp(f"🎤 [VAD INIT] Speech threshold: {self.speech_threshold}", "vad")
        log_timestamp(f"🎤 [VAD INIT] Silence threshold: {self.silence_threshold}", "vad")
        log_timestamp(f"🎤 [VAD INIT] Silence frames threshold: {self.silence_frames_threshold} frames ({self.pause_duration:.1f}s)", "vad")
        log_timestamp(f"🎤 [VAD INIT] Max speech frames: {self.MAX_SPEECH_FRAMES} frames ({self.max_speech_duration}s)", "vad")

    
    def set_primary_language(self, lang_code):
        """Setează limba principală de ascultare."""
        if self.primary_language != lang_code:
            self.primary_language = lang_code
            log_timestamp(f"🗣️ [TRANSCRIERE] Limba primară de ascultare setată la: '{lang_code}'", "transcription")

    def set_last_ai_text(self, text):
        """Setează ultimul text spus de AI pentru detecție echo"""
        self.last_ai_text = text
        log_timestamp(f"🔊 [ECHO PROTECTION] Salvat text AI: '{text[:50]}...'", "echo")

    def set_muted(self, muted, is_ai_speaking=True):
        """Activează/dezactivează ascultarea."""
        self._is_muted = muted
        if muted:
            if is_ai_speaking:
                log_timestamp("🔇 [MUTING] Ascultare PAUSATĂ (AI vorbește)", "mute")
                self.status_changed.emit("🔇 Pausat (AI vorbește)")
            else:
                log_timestamp("🔇 [MUTING] Ascultare PAUSATĂ (Utilizator)", "mute")
                self.status_changed.emit("🎧 Mut (exersezi)")
        else:
            log_timestamp("🔊 [MUTING] Ascultare RELUATĂ", "mute")
            self.status_changed.emit("⚪ Aștept să vorbești...")

    def set_max_speech_duration(self, seconds):
        """Actualizează limita de timp pentru vorbire în timp real."""
        self.max_speech_duration = seconds
        self.MAX_SPEECH_FRAMES = int(seconds * 1000 / self.frame_duration)
        log_timestamp(f"🎤 [WORKER UPDATE] Durata maximă a segmentului a fost setată la {seconds}s.", "app")

    def is_echo(self, transcribed_text):
        """Verifică dacă textul transcris este echo din răspunsul AI"""
        
        # --- Verificarea comutatorului ---
        if not self.enable_echo_cancellation:
            return False # Ieșire imediată dacă funcționalitatea este dezactivată
        # ---------------------------------
        
        if not self.last_ai_text or not transcribed_text:
            return False
        
        # Normalizăm textul AI
        ai_normalized = self.last_ai_text.lower()
        ai_normalized = ''.join(c for c in ai_normalized if c.isalnum() or c.isspace())
        
        # Normalizăm textul transcris
        transcribed_normalized = transcribed_text.lower()
        transcribed_normalized = ''.join(c for c in transcribed_normalized if c.isalnum() or c.isspace())
        
        # Împărțim în cuvinte
        ai_words = set(ai_normalized.split())
        transcribed_words = transcribed_normalized.split()
        
        if len(transcribed_words) == 0:
            return False
        
        # Calculăm câte cuvinte din transcriere sunt în răspunsul AI
        common_words = sum(1 for word in transcribed_words if word in ai_words)
        similarity = common_words / len(transcribed_words)
        
        log_timestamp(f"🔍 [ECHO CHECK] Similitudine: {similarity*100:.1f}% ({common_words}/{len(transcribed_words)} cuvinte)", "echo")
        
        # Dacă >75% din cuvinte sunt în răspunsul AI → e echo
        is_echo_detected = similarity > 0.75
        
        if is_echo_detected:
            log_timestamp(f"🚫 [ECHO DETECTAT] '{transcribed_text}' similar cu AI: {similarity*100:.1f}%", "echo")
        
        return is_echo_detected

    def audio_callback(self, indata, frames, time_info, status):
        """Callback-ul audio, acum cu calcul pentru cronometru."""
        if status: log_timestamp(f"⚠️ [AUDIO] Status: {status}", "audio")
        
        audio_data = indata[:, 0].copy()
        
        rms = np.sqrt(np.mean(audio_data.astype(float)**2))
        if rms > 0:
            db_level = 20 * np.log10(rms) + 90
            self.audio_level_changed.emit(min(max(db_level * 50, 0), 10000))
        
        if self._is_muted: return
        
        audio_tensor = torch.from_numpy(audio_data).float()
        with torch.no_grad():
            speech_probability = self.vad_model(audio_tensor, self.sample_rate).item()
        
        is_speech = speech_probability > self.speech_threshold
        
        audio_int16 = (audio_data * 32767).astype(np.int16)
        self.ring_buffer.append(audio_int16)
        
        if is_speech:
            if not self.is_speech_active:
                self.is_speech_active = True
                self.speech_activity_changed.emit(True)
                self.pause_progress_updated.emit(100)
                log_timestamp("🟢 [VAD] Început vorbire detectat", "vad")
                self.frames_since_silence = 0
                self.speech_frames = list(self.ring_buffer)
                self.status_changed.emit("🔵 Vorbești...")
            else:
                self.frames_since_silence = 0
                self.speech_frames.append(audio_int16)
                self.pause_progress_updated.emit(100)
        else: # Tăcere
            if self.is_speech_active:
                self.frames_since_silence += 1
                self.speech_frames.append(audio_int16)
                progress = 100 - int(100 * self.frames_since_silence / self.silence_frames_threshold)
                self.pause_progress_updated.emit(progress)

        # --- BLOC NOU: Logică Cronometru ---
        if self.is_speech_active:
            timp_ramas = (self.MAX_SPEECH_FRAMES - len(self.speech_frames)) * self.frame_duration / 1000.0
            self.speech_time_updated.emit(timp_ramas)
        # --- SFÂRȘIT BLOC NOU ---

        # Verificare forțată a limitei de timp
        should_process_due_to_pause = self.is_speech_active and self.frames_since_silence >= self.silence_frames_threshold
        should_process_due_to_length = self.is_speech_active and len(self.speech_frames) >= self.MAX_SPEECH_FRAMES

        if should_process_due_to_pause or should_process_due_to_length:
            if should_process_due_to_length:
                log_timestamp("🔴 [VAD] Limita de timp atinsă! Se procesează forțat.", "vad")
            else:
                log_timestamp(f"🔴 [VAD] Sfârșit vorbire (pauză).", "vad")

            self.speech_activity_changed.emit(False)
            self.speech_time_updated.emit(-1) # Semnal de resetare/ascundere cronometru
            self.process_captured_speech()
            
            self.is_speech_active = False
            self.frames_since_silence = 0
            self.speech_frames = []

    def process_captured_speech(self):
        """Procesează audio-ul capturat, cu comutare manuală NATIV/FOCUS/TRADUCERE."""
        if len(self.speech_frames) == 0:
            log_timestamp("⚠️ [PROCESS] Niciun frame de procesat", "process")
            return
        
        temp_path = None
        try:
            # Concatenăm toate frame-urile
            audio_data = np.concatenate(self.speech_frames)
            
            duration = len(audio_data) / self.sample_rate
            log_timestamp(f"🎵 [PROCESS] Durată captată: {duration:.2f}s ({len(audio_data)} samples)", "process")
            
            # Ignorăm clipurile prea scurte (sub 0.3s)
            if duration < 0.3:
                log_timestamp(f"⚠️ [PROCESS] Prea scurt ({duration:.2f}s) - ignorat", "process")
                return
            
            # Salvăm într-un fișier WAV temporar
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_path = temp_wav.name
                
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data.tobytes())
                
                log_timestamp(f"💾 [PROCESS] Salvat în: {temp_path}", "process")
            
            # Citim fișierul cu speech_recognition
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
            
            log_timestamp("🗣️ [TRANSCRIERE] Trimit la Google Speech API...", "transcription")
            self.status_changed.emit("🟡 Transcriu...")
            
            text = None
            
            # --- FAZA 1: Verificare Comenzi în Română ---
            try:
                log_timestamp("🗣️ [TRANSCRIERE] Verificare comenzi în Română...", "transcription")
                possible_command = self.recognizer.recognize_google(audio, language="ro-RO")
                text_lower = possible_command.strip().lower()

                if text_lower.startswith('nativ') or text_lower.startswith('domn profesor'):
                    log_timestamp("🔒 [LANG] Comanda NATIV detectată!", "transcription")
                    self.language_lock_requested.emit('nativ')
                    return # Oprim, a fost o comandă
                elif text_lower.startswith('focus'):
                    log_timestamp("🎯 [LANG] Comanda FOCUS detectată!", "transcription")
                    self.language_lock_requested.emit('focus')
                    return # Oprim, a fost o comandă
                elif text_lower.startswith('traducere') or text_lower.startswith('tradu'):
                    # Comanda de traducere este o conversație specială, o lăsăm să treacă mai departe
                    text = possible_command
                    log_timestamp("🌐 [LANG] Comanda TRADUCERE detectată, se procesează ca input.", "transcription")

            except sr.UnknownValueError:
                # Nu a fost o comandă în română sau nu s-a înțeles, continuăm normal
                pass
            
            # --- FAZA 2: Transcriere Normală (dacă nu a fost detectată o comandă de mod) ---
            if text is None: # Doar dacă nu am preluat deja textul de la comanda 'traducere'
                try:
                    # --- LOGICA CORECTATĂ ---
                    if self.current_lock_mode == 'focus':
                        lang_to_listen = self.primary_language # Limba personajului (ex: fr-FR)
                    else: # Modul 'nativ' sau 'auto'
                        lang_to_listen = self.secondary_language # Limba română (ro-RO)
                    
                    log_timestamp(f"🗣️ [TRANSCRIERE] Ascultare în modul '{self.current_lock_mode}', limba: '{lang_to_listen}'...", "transcription")
                    text = self.recognizer.recognize_google(audio, language=lang_to_listen)
                except sr.UnknownValueError:
                    # Aici nu mai facem fallback, pentru că modurile sunt explicite
                    log_timestamp("❌ [TRANSCRIERE] Nu s-a putut înțelege în modul activ.", "transcription")
                    self.status_changed.emit("⚠️ Nu am înțeles")
                    return
            
            if text and len(text.strip()) < 3:
                log_timestamp(f"⚠️ [TRANSCRIERE] Prea scurt: '{text}'", "transcription")
                return
            
            if text:
                log_timestamp(f"✅ [TRANSCRIERE] Transcris: '{text}'", "transcription")
                
                if self.is_echo(text):
                    log_timestamp(f"🚫 [TRANSCRIERE] ECHO ignorat: '{text}'", "transcription")
                    return
                
                self.transcription_ready.emit(text)
                
        except sr.RequestError as e:
            log_timestamp(f"❌ [TRANSCRIERE] Eroare API Google: {e}", "transcription")
            self.status_changed.emit(f"⚠️ Eroare: {e}")
        except Exception as e:
            log_timestamp(f"❌ [PROCESS] Eroare generală în procesarea audio: {e}", "process")
            import traceback
            log_timestamp(f"❌ [PROCESS] Stack trace:\n{traceback.format_exc()}", "process")
        finally:
            # Ștergem fișierul temporar indiferent de rezultat
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    log_timestamp(f"⚠️ [PROCESS] Eroare la ștergerea fișierului temp: {e}", "process")

    def run(self):
        """
        Bucla principală a worker-ului.
        Pornește stream-ul audio și rămâne în listen mode continuu.
        """
        log_timestamp("🎤 [SILERO VAD WORKER] Worker pornit", "vad")
        log_timestamp(f"🎤 [SILERO VAD WORKER] Configurație:", "vad")
        log_timestamp(f"   - Sample Rate: {self.sample_rate}Hz", "vad")
        log_timestamp(f"   - Frame Duration: {self.frame_duration}ms", "vad")
        log_timestamp(f"   - Pauză pentru sfârșit: {self.pause_duration}s", "vad")
        log_timestamp(f"   - Speech Threshold: {self.speech_threshold}", "vad")
        log_timestamp(f"   - Silence Threshold: {self.silence_threshold}", "vad")
        
        self._is_running = True
        self.status_changed.emit("⚪ Aștept să vorbești...")
        
        try:
            log_timestamp("🎤 [SILERO VAD WORKER] Pornesc stream audio...", "vad")
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.frame_size,
                callback=self.audio_callback
            ):
                log_timestamp("✅ [SILERO VAD WORKER] Stream audio pornit - ascult continuu cu neural VAD", "vad")
                
                # Bucla de keep-alive - thread-ul rămâne activ
                while self._is_running:
                    sd.sleep(100)  # Sleep 100ms, lasă callback-ul să ruleze
                
                log_timestamp("🛑 [SILERO VAD WORKER] Stop requested - opresc stream", "vad")
                
        except Exception as e:
            log_timestamp(f"❌ [SILERO VAD WORKER] EROARE CRITICĂ: {e}", "vad")
            import traceback
            log_timestamp(f"❌ [SILERO VAD WORKER] Stack trace:\n{traceback.format_exc()}", "vad")
            self.status_changed.emit(f"⚠️ Eroare: {e}")
        finally:
            log_timestamp("🎤 [SILERO VAD WORKER] Worker oprit", "vad")

    def stop(self):
        """Oprește worker-ul"""
        log_timestamp("🎤 [SILERO VAD WORKER] 🛑 stop() CHEMAT - setez _is_running=False", "vad")
        self._is_running = False


# =================================================================================
# APLICAȚIA PRINCIPALĂ - NOUA VERSUNE
# =================================================================================

class CharacterApp(QWidget):
    def __init__(self):
        super().__init__()
        log_timestamp("🚀 [APP INIT] Pornire aplicație - Arhitectură Modulară.")
        self.setWindowTitle("Teatru Digital Interactiv")

        # --- ADAUGAȚI ACEST BLOC ---
        self.CULORI_SEMAFOR = {
            "rosu_aprins": "background-color: #FF0000;",
            "rosu_stins": "background-color: #4A0000;",
            "verde_aprins": "background-color: #00FF00;",
            "verde_stins": "background-color: #004A00;",
            "portocaliu_aprins": "#FFA500",
            "portocaliu_stins": "#5A3A00"
        }
        # --- SFÂRȘIT BLOC NOU ---

        # --- HARTĂ LIMBI PENTRU RINA ---
        self.RINA_LANGUAGES = {
            "Engleză":    {"code": "en", "voice": "en-GB-SoniaNeural"},
            "Germană":    {"code": "de", "voice": "de-DE-KatjaNeural"},
            "Italiană":   {"code": "it", "voice": "it-IT-ElsaNeural"},
            "Franceză":   {"code": "fr", "voice": "fr-FR-DeniseNeural"},
            "Spaniolă":   {"code": "es", "voice": "es-ES-ElviraNeural"},
            "Rusă":       {"code": "ru", "voice": "ru-RU-SvetlanaNeural"},
            "Greacă":     {"code": "el", "voice": "el-GR-NestorasNeural"},
            "Japoneză":   {"code": "ja", "voice": "ja-JP-NanamiNeural"},
            "Coreeană":   {"code": "ko", "voice": "ko-KR-SunHiNeural"} 
        }
        # ------------------------------------
        
        # ⭐ ÎNCĂRCARE CONFIG DIN FIȘIER
        self.config = load_config()
        
        # ⭐ SETARE GEOMETRIE FEREASTRĂ DIN CONFIG
        if self.config.get("window_geometry"):
            geom = self.config["window_geometry"]
            self.setGeometry(geom["x"], geom["y"], geom["width"], geom["height"])
            log_timestamp(f"🪟 [WINDOW] Geometrie restaurată: {geom['x']}, {geom['y']}, {geom['width']}x{geom['height']}", "app")
        else:
            # Default geometry
            self.setGeometry(50, 50, 1920, 1080)
            log_timestamp("🪟 [WINDOW] Geometrie default: 50, 50, 1920x1080", "app")

        # --- Starea Aplicației ---
        self.conversation_state = 'INACTIVE'
        self.is_muted = False
        self.is_speaking = False
        self.is_thinking = False
        self.last_audio_file_path = None 
        self.initial_ai_model = self.config.get("ai_model_name") 
        self.viseme_queue = []
        self.greeted_users = {}
        self.conversation_log = []
        self.MAX_LOG_ENTRIES = 10
        self.active_speaker_id = "cucuvel_owl"
        self.last_user_text = ""  
        self.last_character_speeches = {}
        self.pending_speaker_return = None

        self.waiting_for_travel_clarification = False
        self.pending_travel_data = None
        self.clarification_timeout_timer = QTimer(self)
        self.clarification_timeout_timer.setSingleShot(True)
        self.clarification_timeout_timer.timeout.connect(self._handle_clarification_timeout)

        self.pending_move_after_tts = None
        self.language_lock = 'auto'

        # =================================================================================
        # VARIABILE PENTRU SISTEMUL DE ÎNVĂȚARE (MODUL PROFESOR)
        # =================================================================================
        
        # Stare sistem
        self.teacher_mode_active = False
        self.pending_first_question = False
        self.current_student_name = None
        self.current_domain_id = None
        self.current_tier_id = None
        
        # Date curriculum
        self.available_domains = {}
        self.current_curriculum = None
        self.current_tier_data = None
        
        # Tracking răspunsuri în sesiune
        self.session_failed_questions = []
        self.current_question_id = None
        self.current_question_attempt = 0
        
        # Thread management pentru learning
        self.learning_thread = None
        self.learning_worker = None
        
        # Managementul scenei înainte și după lecție
        self.scene_before_lesson = None

        # Resurse pentru tabla virtuală
        self.blackboard_rect = QRect(350, 150, 700, 450) # Coordonate exemplu. Va trebui să le ajustezi!
        self.chalk_font = None # Va fi încărcat mai târziu
        
        # UI Elements
        self.exit_teacher_button = None
        
        # --- LINIA LIPSA ESTE AICI ---
        self.app_state = 'CONVERSATION' # Stări: 'CONVERSATION', 'AWAITING_DOMAIN_CHOICE'
        # --- SFÂRȘIT LINIE LIPSA ---

        self.current_speaker = None
        self.gaze_states = {}

        # --- Inițializare Manageri ---
        log_timestamp("🧠 [APP INIT] Se inițializează managerii...")
        self.scene_manager = SceneManager()
        self.character_manager = CharacterManager()
        self._apply_saved_character_settings()
        log_timestamp("✅ [APP INIT] Manageri inițializați.")
        
        self.character_layers = {}
        self.all_animators = []

        # --- Parametri din Config ---
        self.threshold = self.config["threshold"]
        self.pause_duration = self.config["pause_duration"]
        self.max_speech_duration = self.config["max_speech_duration"]
        self.margin_percent = self.config["margin_percent"]
        self.voice_enabled = False
        
        # --- Inițializare Worker-i ---
        self.webcam_worker, self.webcam_thread = None, None
        self.gemini_worker, self.gemini_thread = None, None
        self.tts_worker, self.tts_thread = None, None
        self.voice_worker, self.voice_thread = None, None
        self.intent_worker, self.intent_thread = None, None
        
        # --- Inițializare UI ---
        log_timestamp("🎨 [APP INIT] Se construiește interfața grafică...")
        self.init_ui()
        self.echo_cancellation_checkbox.setChecked(self.config.get("enable_echo_cancellation", True))

        # --- APLICAREA CONFIGURĂRILOR INIȚIALE PENTRU UI ---
        
        # 1. Limba pentru Rina
        saved_code = self.config.get("rina_language_code", "en")
        for name, details in self.RINA_LANGUAGES.items():
            if details["code"] == saved_code:
                self.rina_language_combo.setCurrentText(name)
                break
        
        # 2. Subtitrări
        font_size = self.config.get("subtitle_font_size", 26)
        self.subtitle_font_slider.setValue(font_size)
        self.subtitle_font_label.setText(f"Mărime font: {font_size}px")
        self._update_subtitle_style()
        
        subtitle_mode = self.config.get("subtitle_mode", "original")
        if subtitle_mode == "latin (fonetic)":
            self.subtitle_mode_combo.setCurrentIndex(1)
        elif subtitle_mode == "combinat":
            self.subtitle_mode_combo.setCurrentIndex(2)
        else:
            self.subtitle_mode_combo.setCurrentIndex(0)
            
        # 3. Furnizor TTS
        provider = self.config.get("tts_provider", "microsoft")
        if provider == "google":
            self.tts_provider_combo.setCurrentIndex(1)
        else:
            self.tts_provider_combo.setCurrentIndex(0)

        # 4. Setarea Modelului AI
        saved_model = self.config.get("ai_model_name", "models/gemini-flash-lite-latest")
        index = self.ai_model_combo.findText(saved_model)
        if index != -1:
            self.ai_model_combo.setCurrentIndex(index)
        
        # 5. Încărcarea pozițiilor pentru sliderele de voce
        self._load_slider_positions_from_config()

        # ---------------------------

        log_timestamp("✅ [APP INIT] Interfață grafică construită.")
        
        # --- Conectare Semnale Manageri ---
        log_timestamp("🔗 [APP INIT] Se conectează semnalele managerilor...")
        self.scene_manager.scene_changed.connect(self.on_scene_changed)
        self.character_manager.character_added_to_stage.connect(self.on_character_added)
        self.character_manager.character_removed_from_stage.connect(self.on_character_removed)
        log_timestamp("✅ [APP INIT] Semnale conectate.")
        
        # --- Timere ---
        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(self.update_synced_animation)
        self.idle_timer = QTimer(self)
        self.idle_timer.timeout.connect(self._idle_animation)
        self.thinking_timer = QTimer(self)
        self.thinking_timer.timeout.connect(self.animate_thinking)
        
        # --- Inițializare Pygame ---
        log_timestamp("🔊 [PYGAME] Se inițializează mixer-ul audio...")
        try:
            pygame.mixer.init()
            pygame.mixer.set_num_channels(16)
            log_timestamp("✅ [PYGAME] Mixer inițializat cu succes.")
        except Exception as e:
            log_timestamp(f"❌ [PYGAME] Eroare la inițializarea mixer-ului: {e}")


        # --- Încărcare date familie la pornire ---
        self._load_family_data()     
        self._discover_available_domains()   
        # --- Stare Inițială ---
        log_timestamp("🎬 [APP INIT] Se setează starea inițială a scenei...")
        self.scene_manager.set_scene("acasa")
        self.character_manager.add_character_to_stage("cucuvel_owl")
        self.character_manager.add_character_to_stage("rina_cat")
        
        self.update_ui_for_state()

        from PySide6.QtGui import QFontDatabase
        font_id = QFontDatabase.addApplicationFont("assets/fonts/Chalkboard-Regular.ttf")
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            self.chalk_font = QFont(font_family)
            log_timestamp(f"✅ [FONT] Font-ul 'cretă' ('{font_family}') a fost încărcat cu succes.", "app")
        else:
            log_timestamp("❌ [FONT] Eroare la încărcarea font-ului 'cretă'. Se va folosi un font de sistem.", "app")
            self.chalk_font = QFont() # Folosim un font default ca fallback

        # --- BLOC NOU: Încărcare resurse custom (FONT) ---
        font_id = QFontDatabase.addApplicationFont("assets/fonts/Chalkboard-Regular.ttf")
        if font_id != -1:
            self.chalkboard_font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            self.chalk_font = QFont(self.chalkboard_font_family)
            log_timestamp(f"✅ [FONT] Font-ul 'cretă' ('{self.chalkboard_font_family}') a fost încărcat cu succes.", "app")
        else:
            log_timestamp("❌ [FONT] Eroare la încărcarea font-ului 'cretă'. Se va folosi un font de sistem.", "app")
            self.chalkboard_font_family = "Arial"
            self.chalk_font = QFont("Arial")
        # --- SFÂRȘIT BLOC NOU ---


        log_timestamp("✅ [APP INIT] Inițializare completă. Aplicația este gata.")

    def init_ui(self):
        log_timestamp("🎨 [UI] Construire interfață principală...")
        main_layout = QHBoxLayout(self)
        
        # --- Coloana Stângă (Control și Chat) ---
        self.tabs = QTabWidget()
        conversation_tab = QWidget()
        conv_layout = QHBoxLayout(conversation_tab)
        left_column = QVBoxLayout()
        
        # Widget-ul pentru webcam
        self.webcam_label = QLabel("Camera oprită.")
        self.webcam_label.setFixedSize(320, 240)
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setStyleSheet("background-color: black; color: white; border: 2px solid gray;")
        
        # Crearea butoanelor
        # 1. Creăm butoanele
        self.conversation_button = QPushButton("🚀 Start Conversație")
        self.conversation_button.clicked.connect(self.toggle_conversation_state)
        
        self.mute_button = QPushButton("🎤 Mut")
        self.mute_button.clicked.connect(self.toggle_mute_state)
        self.mute_button.setEnabled(False)
        self.mute_button.setStyleSheet("background-color: #f0ad4e;")
        # Butonul NOU "Repetă"
        self.repeat_button = QPushButton("🔁 Repetă")
        self.repeat_button.clicked.connect(self.repeat_last_audio)
        self.repeat_button.setEnabled(False) # Inactiv la început
        # 2. Creăm un layout orizontal pentru a le conține
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.conversation_button)
        buttons_layout.addWidget(self.mute_button)
        buttons_layout.addWidget(self.repeat_button) # Adăugăm noul buton
        # Butonul pentru ieșire din Modul Profesor (inițial ascuns)
        self.exit_teacher_button = QPushButton("🛑 Oprește Lecția")
        self.exit_teacher_button.clicked.connect(self.exit_teacher_mode)
        self.exit_teacher_button.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.exit_teacher_button.setVisible(False)  # Ascuns la început
        buttons_layout.addWidget(self.exit_teacher_button)
        # 3. Restul widget-urilor
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Apasă 'Start'...")
        self.text_input.returnPressed.connect(self.send_to_ai)
        
        # Asamblarea CORECTĂ a coloanei stângi
        left_column.addWidget(self.webcam_label, stretch=0)
        left_column.addLayout(buttons_layout) 
        left_column.addWidget(self.chat_history, stretch=1)
        left_column.addWidget(self.text_input, stretch=0)
        # --- Coloana Dreaptă (Scena Vizuală) ---
        right_column = QVBoxLayout()
        self.scene_container = QWidget()
        SCENE_WIDTH = 1400
        SCENE_HEIGHT = 900
        self.scene_container.setMinimumSize(SCENE_WIDTH, SCENE_HEIGHT)
        
        self.background_label = QLabel(self.scene_container)
        self.background_label.setGeometry(0, 0, SCENE_WIDTH, SCENE_HEIGHT)
        
        # --- BLOC NOU ȘI CORECTAT: Crearea tablei virtuale ---
        self.blackboard_labels = []
        for i in range(5):
            label = QLabel(self.scene_container) 
            # label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            label.hide()
            self.blackboard_labels.append(label)
        # --- SFÂRȘIT BLOC ---
        
        # === SISTEM CALIBRARE TABLĂ ===
        self.calibration_mode = False
        self.calibration_point = QPoint(700, 400)
        self.calibration_saved = []
        
        # Buton pentru activare calibrare - FIXAT poziționare
        self.calibration_button = QPushButton("🎯 ACTIVEAZĂ CALIBRARE TABLĂ", self.scene_container)
        self.calibration_button.clicked.connect(self._activate_calibration)
        self.calibration_button.setStyleSheet("background-color: orange; font-weight: bold; font-size: 14px;")
        self.calibration_button.setGeometry(1050, 10, 300, 50)
        self.calibration_button.raise_()
        self.calibration_button.hide()  # ⭐ ASCUNS - Decomentează dacă mai vrei calibrare
        # === SFÂRȘIT SISTEM CALIBRARE ===
        
        right_column.addWidget(self.scene_container)
        
        # --- CREARE SISTEM SEMAFOR ---
        semafor_img_height = 240
        semafor_labels_height = 40
        semafor_width = 135
        semafor_total_height = semafor_img_height + semafor_labels_height
        semafor_x_pos = 10 
        semafor_y_pos = 10 
        self.semafor_container = QWidget(self.scene_container)
        self.semafor_container.setGeometry(semafor_x_pos, semafor_y_pos, semafor_width, semafor_total_height)
        self.semafor_bg_label = QLabel(self.semafor_container)
        self.semafor_bg_label.setPixmap(QPixmap("assets/ui/semafor_fundal.png"))
        self.semafor_bg_label.setGeometry(0, 0, semafor_width, semafor_img_height)
        light_diameter = 55
        radius = light_diameter // 2
        light_x_offset = (semafor_width - light_diameter) // 2
        rosu_y_pos = 20
        portocaliu_y_pos = 94
        verde_y_pos = 168
        self.semafor_rosu_widget = QWidget(self.semafor_container)
        self.semafor_rosu_widget.setGeometry(light_x_offset, rosu_y_pos, light_diameter, light_diameter)
        self.semafor_rosu_widget.setStyleSheet(f"border-radius: {radius}px;")
        self.semafor_verde_widget = QWidget(self.semafor_container)
        self.semafor_verde_widget.setGeometry(light_x_offset, verde_y_pos, light_diameter, light_diameter)
        self.semafor_verde_widget.setStyleSheet(f"border-radius: {radius}px;")
        self.cronometru_label = QLabel(self.semafor_verde_widget)
        self.cronometru_label.setGeometry(0, 0, light_diameter, light_diameter)
        self.cronometru_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cronometru_label.setStyleSheet("background-color: transparent; color: black; font-size: 28px; font-weight: bold;")
        self.cronometru_label.hide()
        self.clepsidra_container = QWidget(self.semafor_container)
        self.clepsidra_container.setGeometry(light_x_offset, portocaliu_y_pos, light_diameter, light_diameter)
        self.clepsidra_continut = QLabel(self.clepsidra_container)
        self.clepsidra_continut.setGeometry(0, 0, light_diameter, light_diameter)
        self.clepsidra_continut.setStyleSheet(f"background-color: transparent; border-radius: {radius}px;")
        self.clepsidra_masca = QLabel(self.clepsidra_container)
        self.clepsidra_masca.setGeometry(0, 0, light_diameter, 0)
        self.clepsidra_masca.setStyleSheet(f"background-color: {self.CULORI_SEMAFOR['portocaliu_stins']}; border-top-left-radius: {radius}px; border-top-right-radius: {radius}px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;")
        self.clepsidra_contur = QLabel(self.clepsidra_container)
        self.clepsidra_contur.setGeometry(0, 0, light_diameter, light_diameter)
        self.clepsidra_contur.setStyleSheet(f"background-color: transparent; border: 2px solid #222; border-radius: {radius}px;")
        self.clepsidra_container.hide()
        
        self.semafor_container.hide()
        self.semafor_container.raise_()
        
        self.mod_nativ_label = QLabel("NATIV", self.semafor_container)
        self.mod_nativ_label.setGeometry(0, semafor_img_height, semafor_width, 20)
        self.mod_nativ_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mod_focus_label = QLabel("FOCUS", self.semafor_container)
        self.mod_focus_label.setGeometry(0, semafor_img_height + 20, semafor_width, 20)
        self.mod_focus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.style_mod_aprins = "color: white; font-weight: bold; font-size: 14px;"
        self.style_mod_stins = "color: #555; font-size: 14px;"
        self.mod_nativ_label.raise_()
        self.mod_focus_label.raise_()
        # --- BLOC COMPLET PENTRU AMBELE SUBTITRĂRI ---
        subtitle_width = int(SCENE_WIDTH * 0.8)
        bottom_subtitle_height = 120
        bottom_subtitle_x = int((SCENE_WIDTH - subtitle_width) / 2)
        bottom_subtitle_y = SCENE_HEIGHT - bottom_subtitle_height - 20
        self.subtitle_scroll_area = QScrollArea(self.scene_container)
        self.subtitle_scroll_area.setGeometry(bottom_subtitle_x, bottom_subtitle_y, subtitle_width, bottom_subtitle_height)
        self.subtitle_scroll_area.setWidgetResizable(True)
        self.subtitle_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.subtitle_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.subtitle_scroll_area.setStyleSheet("QScrollArea { background: rgba(0, 0, 0, 0.5); border-radius: 10px; border: none; }")
        self.subtitle_label = QLabel()
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setTextFormat(Qt.TextFormat.RichText)
        self.subtitle_label.setStyleSheet("QLabel { background: transparent; color: white; padding: 10px; }")
        self.subtitle_scroll_area.setWidget(self.subtitle_label)
        self.subtitle_scroll_area.hide()
        translation_width = int(SCENE_WIDTH * 0.7)
        translation_height = 120
        translation_x = self.semafor_container.geometry().right() + 20
        translation_y = 20
        self.translation_scroll_area = QScrollArea(self.scene_container)
        self.translation_scroll_area.setGeometry(translation_x, translation_y, translation_width, translation_height)
        self.translation_scroll_area.setWidgetResizable(True)
        self.translation_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.translation_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.translation_scroll_area.setStyleSheet("QScrollArea { background: rgba(0, 0, 0, 0.5); border-radius: 10px; border: none; }")
        self.translation_label = QLabel()
        self.translation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.translation_label.setWordWrap(True)
        self.translation_label.setStyleSheet("QLabel { background: transparent; color: white; padding: 10px; font-size: 22px; }")
        self.translation_scroll_area.setWidget(self.translation_label)
        self.translation_scroll_area.hide()
        # --- Asamblare Finală ---
        conv_layout.addLayout(left_column, 0)
        conv_layout.addLayout(right_column, 1)
        
        general_tab = self.create_general_settings_tab()
        voice_tab = self.create_voice_settings_tab()
        family_tab = self.create_family_settings_tab()
        self.tabs.addTab(conversation_tab, "💬 Conversație")
        self.tabs.addTab(family_tab, "👨‍👩‍👧‍👦 Familie")
        self.tabs.addTab(general_tab, "⚙️ Setări Generale")
        self.tabs.addTab(voice_tab, "🎤 Setări Voce")
        main_layout.addWidget(self.tabs)
        
        log_timestamp("🎨 [UI] Interfață construită.")



    def on_language_lock_requested(self, mode):
        """Schimbă modul de ascultare și actualizează UI-ul."""
        if self.language_lock != mode:
            self.language_lock = mode
            if self.voice_worker:
                self.voice_worker.current_lock_mode = mode
            
            if mode == 'nativ':
                log_timestamp("🔒 [LANG] Modul de ascultare blocat pe Română (NATIV).", "app")
                self.update_voice_status("🗣️ Mod NATIV (RO)")
                self.mod_nativ_label.setStyleSheet(self.style_mod_aprins)
                self.mod_focus_label.setStyleSheet(self.style_mod_stins)
                log_timestamp("🚦 [SEMAFOR DEBUG] Aplicat stil APRINS pe NATIV, STINS pe FOCUS.", "semafor")
            else: # focus
                log_timestamp("🎯 [LANG] Modul de ascultare setat pe FOCUS (Limbă Străină).", "app")
                self.update_voice_status("🗣️ Mod FOCUS")
                self.mod_nativ_label.setStyleSheet(self.style_mod_stins)
                self.mod_focus_label.setStyleSheet(self.style_mod_aprins)
                log_timestamp("🚦 [SEMAFOR DEBUG] Aplicat stil STINS pe NATIV, APRINS pe FOCUS.", "semafor")

    def on_ai_model_changed(self, model_name):
            """Salvează noul model AI selectat în config."""
            if model_name: # Ne asigurăm că nu este un string gol
                self.config["ai_model_name"] = model_name
                save_config(self.config)
                log_timestamp(f"🧠 [CONFIG] Model AI setat la: '{model_name}'")

    def on_scene_changed(self, scene_id, scene_data):
        log_timestamp(f"🌆 [UI SCENE] Primit semnal de schimbare scenă la '{scene_id}'.", "scene")
        
        bg_path = scene_data.get("background_image")
        if bg_path and os.path.exists(bg_path):
            bg_pixmap = QPixmap(bg_path)
            
            # ⭐ FIX: Dimensiune FIXĂ pentru consistency
            FIXED_WIDTH = 1400
            FIXED_HEIGHT = 900
            
            scaled_pixmap = bg_pixmap.scaled(
                FIXED_WIDTH, 
                FIXED_HEIGHT,
                Qt.AspectRatioMode.IgnoreAspectRatio,  # ⭐ Forțăm dimensiunea exactă
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.background_label.setPixmap(scaled_pixmap)
            self.background_label.setGeometry(0, 0, FIXED_WIDTH, FIXED_HEIGHT)  # ⭐ Poziție fixă
            
            log_timestamp(f"  ✅ Fundal actualizat: {bg_path}", "scene")
        else:
            log_timestamp(f"  ⚠️ AVERTISMENT: Imagine de fundal negăsită la '{bg_path}'", "scene")
            self.background_label.clear()
            self.background_label.setStyleSheet("background-color: darkgray;")

        log_timestamp(f"  ✅ Scenă schimbată complet în UI", "scene")
    
    def on_max_speech_changed(self, value):
        self.max_speech_duration = value
        self.config["max_speech_duration"] = value
        save_config(self.config)
        self.max_speech_label.setText(f"{value} sec")
        
        if self.voice_worker:
            self.voice_worker.set_max_speech_duration(value)
            
        log_timestamp(f"⏱️ [DURATĂ MAX] Modificată și salvată: {value}s")

    def _update_semafor_state(self, state, progress=100):
        """Actualizează starea vizuală a semaforului (Versiunea 4, Corectată)."""
        log_timestamp(f"🚦 [SEMAFOR DEBUG] Primit comandă de actualizare. Stare: '{state}', Progres: {progress}", "semafor")
        radius = 55 // 2

        # 1. Resetează becurile Roșu și Verde la "stins"
        self.semafor_rosu_widget.setStyleSheet(f"border-radius: {radius}px; {self.CULORI_SEMAFOR['rosu_stins']}")
        self.semafor_verde_widget.setStyleSheet(f"border-radius: {radius}px; {self.CULORI_SEMAFOR['verde_stins']}")
        
        # 2. Resetează clepsidra la starea "stins" (dar o lasă vizibilă)
        self.clepsidra_container.show() # Asigură-te că este mereu vizibilă când lucrăm cu ea
        self.clepsidra_continut.setStyleSheet(f"background-color: {self.CULORI_SEMAFOR['portocaliu_stins']}; border-radius: {radius}px;")
        self.clepsidra_masca.hide()

        # 3. Aprinde lumina corectă
        if state == 'rosu':
            self.semafor_rosu_widget.setStyleSheet(f"border-radius: {radius}px; {self.CULORI_SEMAFOR['rosu_aprins']}")
        elif state == 'verde':
            self.semafor_verde_widget.setStyleSheet(f"border-radius: {radius}px; {self.CULORI_SEMAFOR['verde_aprins']}")
        elif state == 'pauza':
            # Când e pauză, becurile roșu și verde sunt stinse, deci nu mai facem nimic pentru ele aici.
            # Doar actualizăm clepsidra.
            
            # Aprinde fundalul portocaliu al clepsidrei
            self.clepsidra_continut.setStyleSheet(f"background-color: {self.CULORI_SEMAFOR['portocaliu_aprins']}; border-radius: {radius}px;")
            
            # Calculează și afișează masca
            light_diameter = 55
            mask_height = int(light_diameter * (100 - progress) / 100)
            self.clepsidra_masca.setGeometry(0, 0, light_diameter, mask_height)
            self.clepsidra_masca.show()

    def on_character_added(self, character):
        """
        Slot executat când CharacterManager emite 'character_added_to_stage'.
        Creează dinamic layerele (QLabels) pentru noul personaj și stochează
        pixmap-ul original, ne-scalat, pentru fiecare layer.
        """
        log_timestamp(f"🎭 [UI CHAR] Primit semnal de adăugare personaj: '{character.display_name}'.", "character")
        if character.id in self.character_layers:
            log_timestamp(f"  ⚠️ Personajul '{character.id}' are deja layere create. Se reutilizează.", "character")
            if self.scene_manager.current_scene_id:
                scene_config = character.get_config_for_scene(self.scene_manager.current_scene_id)
                if scene_config:
                    char_layers = self.character_layers.get(character.id)
                    self._position_character_layers(character, char_layers, scene_config)
                    for layer in char_layers.values():
                        layer.show()
                else:
                    log_timestamp(f"  ⚠️ Nu are config pentru scena curentă - se ascunde", "character")
                    for layer in self.character_layers[character.id].values():
                        layer.hide()
            return

        log_timestamp(f"  🔨 Se creează layerele vizuale pentru '{character.id}' pe baza 'components'...", "character")
        
        components = character.components
        parts = components.get("parts", {})
        z_order = components.get("z_order", [])
        
        if not parts or not z_order:
            log_timestamp(f"  ❌ EROARE: 'parts' sau 'z_order' lipsesc din config.json pentru '{character.id}'!", "character")
            return

        char_layers = {}
        for part_name in z_order:
            image_file = parts.get(part_name)
            if not image_file:
                log_timestamp(f"    ⚠️ Avertisment: Numele de parte '{part_name}' din z_order nu a fost găsit în 'parts'.", "character")
                continue
                
            image_path = os.path.join(character.assets_path, image_file)
            if not os.path.exists(image_path):
                log_timestamp(f"    ⚠️ Avertisment: Asset-ul '{image_file}' lipsește pentru '{character.id}'.", "character")
                continue
                
            layer = QLabel(self.scene_container)
            layer.original_pixmap = QPixmap(image_path)
            layer.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            char_layers[part_name] = layer
        
        self.character_layers[character.id] = char_layers
        log_timestamp(f"  ✅ Au fost create {len(char_layers)} layere pentru '{character.id}'.", "character")
        

        # ⭐ LINIE NOUĂ CRITICĂ - SETEAZĂ current_scene_id!
        character.current_scene_id = self.scene_manager.current_scene_id
        log_timestamp(f"  🎯 [EMOTION SETUP] Setez current_scene_id = '{self.scene_manager.current_scene_id}' pentru '{character.id}'", "emotion")


        # ⭐ SCHIMBARE CRITICĂ: SETĂM POZIȚIA ÎNAINTE DE A PORNI ANIMATOARELE!
        if self.scene_manager.current_scene_id:
            scene_config = character.get_config_for_scene(self.scene_manager.current_scene_id)
            if scene_config:
                # ⭐ 1. MAI ÎNTÂI poziționăm layerele
                self._position_character_layers(character, char_layers, scene_config)
                
                # ⭐ 2. APOI pornim animatoarele
                log_timestamp(f"🛠️ [ANIM] Asamblare animatoare pentru '{character.id}'...", "character")
                character.setup_animators(char_layers)
        
                # --- BLOC NOU DE ADĂUGAT AICI ---
                # Dacă conversația nu a început încă, punem personajul în stare de "somn"
                if self.conversation_state == 'INACTIVE':
                    log_timestamp(f"🌙 [APP INIT] Conversație inactivă. Se setează starea 'sleeping' pentru {character.id}", "app")
                    for animator in character.animators:
                        if isinstance(animator, (BreathingAnimator, BlinkingAnimator)):
                            animator.stop() # Oprim respirația și clipitul
                    
                    emotion_animator = next((anim for anim in character.animators if isinstance(anim, EmotionAnimator)), None)
                    if emotion_animator:
                        emotion_animator.set_emotion('sleeping')
                # --- SFÂRȘIT BLOC NOU ---

                # 3. Arătăm layerele
                for layer in char_layers.values():
                    layer.show()
            else:
                log_timestamp(f"  ⚠️ Nu are config pentru scena curentă - se ascunde", "character")
                for layer in char_layers.values():
                    layer.hide()

    def _update_character_for_scene(self, character, scene_id):
        """
        Funcție ajutătoare care actualizează vizibilitatea și poziția
        unui singur personaj în funcție de o scenă dată.
        """
        character.current_scene_id = scene_id
        char_layers = self.character_layers.get(character.id)
        if not char_layers:
            return

        scene_config = character.get_config_for_scene(scene_id)
        
        if scene_config:
            log_timestamp(f"    -> Repoziționez '{character.id}' la {scene_config['pos']} cu scara {scene_config['scale']}")
            self._position_character_layers(character, char_layers, scene_config)
            for layer in char_layers.values():
                layer.show()
        else:
            log_timestamp(f"    -> '{character.id}' nu are configurație pentru '{scene_id}'. Se ascunde.")
            for layer in char_layers.values():
                layer.hide()

    def on_character_removed(self, character_id):
        log_timestamp(f"🎬 [UI CHAR] Primit semnal de eliminare personaj: '{character_id}'.", "character")
        
        character = self.character_manager.get_character(character_id)
        if character and character.animators:
            log_timestamp(f"🛑 [ANIM] Oprire și curățare animatoare pentru '{character_id}'...", "character")
            for animator in character.animators:
                animator.stop()
                if animator in self.all_animators:
                    self.all_animators.remove(animator)
                animator.deleteLater()
            character.animators = []

        if character_id in self.character_layers:
            for layer_widget in self.character_layers[character_id].values():
                layer_widget.deleteLater()
            del self.character_layers[character_id]

    def _position_character_layers(self, character, layers, scene_config):
        """
        Funcție ajutătoare pentru a scala și a poziționa layerele unui personaj.
        VERSIUNE CORECTATĂ: Gestionează offset-uri ca listă SAU QPoint!
        """
        scale = scene_config.get("scale", 0.3)
        base_pos = scene_config.get("pos", QPoint(0, 0))
        
        # ⭐ CITIM OFFSET-URILE DIN CONFIG
        part_offsets = character.components.get("part_offsets", {})
        
        if character.id == "rina_cat":
            log_timestamp(f"📍 [UI POS] Setez poziție Rina: {base_pos}, scale={scale}", "position")
        
        z_order = character.components.get("z_order", [])

        for part_name in z_order:
            layer = layers.get(part_name)
            if not layer or not hasattr(layer, 'original_pixmap'):
                continue

            original_pixmap = layer.original_pixmap
            if not original_pixmap or original_pixmap.isNull():
                continue
            
            # Scalăm ÎNTOTDEAUNA de la imaginea originală
            scaled_pixmap = original_pixmap.scaled(
                int(original_pixmap.width() * scale), 
                int(original_pixmap.height() * scale), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            layer.setPixmap(scaled_pixmap)
            layer.setFixedSize(scaled_pixmap.size())
            
            # ⭐ CALCULĂM POZIȚIA CU OFFSET - GESTIONARE SAFE
            offset = part_offsets.get(part_name, [0, 0])
            
            # ⭐ CONVERSIE SAFE: listă SAU QPoint
            if isinstance(offset, QPoint):
                offset_x = offset.x()
                offset_y = offset.y()
            elif isinstance(offset, (list, tuple)) and len(offset) >= 2:
                offset_x = offset[0]
                offset_y = offset[1]
            else:
                # Fallback la [0, 0]
                offset_x = 0
                offset_y = 0
            
            final_x = base_pos.x() + offset_x
            final_y = base_pos.y() + offset_y
            final_pos = QPoint(final_x, final_y)
            
            # ⭐ DEBUG pentru prima rulare
            if not hasattr(self, '_pos_debug_logged'):
                if character.id == "cucuvel_owl" and part_name in ["aripa_stanga", "ochi", "gura"]:
                    log_timestamp(f"📍 [POS DEBUG] '{part_name}': base=({base_pos.x()}, {base_pos.y()}), offset=({offset_x}, {offset_y}), final=({final_x}, {final_y})", "position")
            
            layer.move(final_pos)
            layer.raise_()
        
        # Marchează că am făcut debug
        if not hasattr(self, '_pos_debug_logged'):
            self._pos_debug_logged = True
        
        # Anunță breathing animator că pozițiile s-au schimbat
        for animator in character.animators:
            if isinstance(animator, BreathingAnimator):
                animator.refresh_positions()
                break
                
    def _update_all_animations(self):
            characters_to_update = [
                char for char_id, char in self.character_manager.active_characters.items()
                if char_id in self.character_layers
            ]
            current_scene_id = self.scene_manager.current_scene_id

            for character in characters_to_update:
                try:
                    character_layers = self.character_layers[character.id]
                    # Pasăm acum și scena curentă
                    character.update(character_layers, current_scene_id)
                except Exception as e:
                    log_timestamp(f"❌ [ANIM ERROR] Eroare la actualizarea animației pentru '{character.id}': {e}")
    
    def _idle_animation(self):
        """Metodă placeholder pentru animații idle viitoare."""
        # Deocamdată nu face nimic, dar este necesară pentru a nu crăpa.
        pass

    def create_voice_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        auto_settings_group = QGroupBox("🎛️ Setări Automate")
        auto_layout = QVBoxLayout()
        
        self.auto_calibrate_checkbox = QCheckBox("🔄 Calibrare automată la pornirea microfonului")
        self.auto_calibrate_checkbox.setChecked(self.config["auto_calibrate_on_mic_start"])
        self.auto_calibrate_checkbox.stateChanged.connect(self.on_auto_calibrate_changed)
        auto_layout.addWidget(self.auto_calibrate_checkbox)
        
        self.auto_start_mic_checkbox = QCheckBox("🎤 Pornește microfonul automat la Start Conversație")
        self.auto_start_mic_checkbox.setChecked(self.config["auto_start_mic_with_conversation"])
        self.auto_start_mic_checkbox.stateChanged.connect(self.on_auto_start_mic_changed)
        auto_layout.addWidget(self.auto_start_mic_checkbox)
        
        self.no_camera_checkbox = QCheckBox("📵 Conversație fără cameră (doar text/voce)")
        self.no_camera_checkbox.setChecked(self.config["conversation_without_camera"])
        self.no_camera_checkbox.stateChanged.connect(self.on_no_camera_changed)
        auto_layout.addWidget(self.no_camera_checkbox)

        # --- Checkbox nou pentru Anularea Ecoului ---
        self.echo_cancellation_checkbox = QCheckBox("🔇 Anulează ecoul vocii personajelor (Recomandat)")
        self.echo_cancellation_checkbox.setToolTip(
            "Când este activat, sistemul va ignora sunetele care seamănă\n"
            "cu ultimul răspuns al personajului, prevenind buclele de răspuns.\n"
            "Dezactivați pentru jocuri cu răspunsuri repetitive (ex: da/nu)."
        )
        self.echo_cancellation_checkbox.stateChanged.connect(self.on_echo_cancellation_changed)
        auto_layout.addWidget(self.echo_cancellation_checkbox)
        # ----------------------------------------------
        
        info_label = QLabel("💡 Modul fără cameră: AI-ul nu va analiza imagini, doar răspunde la întrebări.")
        info_label.setStyleSheet("font-size: 10px; color: #666; font-style: italic; padding-left: 20px;")
        info_label.setWordWrap(True)
        auto_layout.addWidget(info_label)
        
        auto_settings_group.setLayout(auto_layout)
        layout.addWidget(auto_settings_group)
        
        control_group = QGroupBox("🎤 Control Microfon")
        control_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.voice_toggle_btn = QPushButton("🟢 Activează Microfon")
        self.voice_toggle_btn.setStyleSheet("background-color: #5cb85c; font-size: 14px; padding: 10px;")
        self.voice_toggle_btn.clicked.connect(self.toggle_voice)
        btn_layout.addWidget(self.voice_toggle_btn)
        
        control_layout.addLayout(btn_layout)
        
        self.voice_status_label = QLabel("⚪ Microfon oprit")
        self.voice_status_label.setStyleSheet("font-size: 14px; padding: 5px;")
        control_layout.addWidget(self.voice_status_label)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        audio_group = QGroupBox("📈 Nivel Audio Live")
        audio_layout = QVBoxLayout()
        
        self.audio_meter = QProgressBar()
        self.audio_meter.setRange(0, 10000)
        self.audio_meter.setValue(0)
        self.audio_meter.setTextVisible(True)
        self.audio_meter.setFormat("Nivel: %v / 10000")
        self.audio_meter.setMinimumHeight(30)
        audio_layout.addWidget(self.audio_meter)
        
        self.threshold_indicator = QLabel("Threshold: 400")
        self.threshold_indicator.setStyleSheet("font-size: 12px; font-weight: bold; color: #d9534f;")
        audio_layout.addWidget(self.threshold_indicator)
        
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        threshold_group = QGroupBox("🎚️ Setări Detectare")
        threshold_layout = QFormLayout()
        
        threshold_container = QVBoxLayout()
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(200, 10000)
        self.threshold_slider.setValue(400)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(1000)
        self.threshold_slider.setMinimumHeight(50)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        
        threshold_container.addWidget(self.threshold_slider)
        
        threshold_labels_layout = QHBoxLayout()
        threshold_labels_layout.setContentsMargins(0, 0, 0, 0)
        
        gradation_values = [200, 2000, 4000, 6000, 8000, 10000]
        for val in gradation_values:
            label = QLabel(str(val))
            label.setStyleSheet("font-size: 9px; color: #666;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            threshold_labels_layout.addWidget(label)
        
        threshold_container.addLayout(threshold_labels_layout)
        
        self.threshold_label = QLabel("400")
        self.threshold_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #d9534f;")
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        threshold_layout.addRow("Threshold Detectare:", threshold_container)
        threshold_layout.addRow("Valoare Curentă:", self.threshold_label)
        
        margin_container = QVBoxLayout()
        
        self.margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.margin_slider.setRange(0, 50)
        self.margin_slider.setValue(20)
        self.margin_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.margin_slider.setTickInterval(10)
        self.margin_slider.setMinimumHeight(40)
        self.margin_slider.valueChanged.connect(self.on_margin_changed)
        
        margin_container.addWidget(self.margin_slider)
        
        margin_labels_layout = QHBoxLayout()
        for val in [0, 10, 20, 30, 40, 50]:
            label = QLabel(f"{val}%")
            label.setStyleSheet("font-size: 9px; color: #666;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            margin_labels_layout.addWidget(label)
        margin_container.addLayout(margin_labels_layout)
        
        self.margin_label = QLabel("+20%")
        self.margin_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0ad4e;")
        self.margin_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        threshold_layout.addRow("Marjă Siguranță:", margin_container)
        threshold_layout.addRow("", self.margin_label)
        
        pause_container = QVBoxLayout()
        
        self.pause_slider = QSlider(Qt.Orientation.Horizontal)
        self.pause_slider.setRange(5, 30)
        self.pause_slider.setValue(8)
        self.pause_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.pause_slider.setTickInterval(5)
        self.pause_slider.setMinimumHeight(40)
        self.pause_slider.valueChanged.connect(self.on_pause_changed)
        
        pause_container.addWidget(self.pause_slider)
        
        pause_labels_layout = QHBoxLayout()
        for val in [5, 10, 15, 20, 25, 30]:
            label = QLabel(f"{val/10:.1f}s")
            label.setStyleSheet("font-size: 9px; color: #666;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pause_labels_layout.addWidget(label)
        pause_container.addLayout(pause_labels_layout)
        
        self.pause_label = QLabel("0.8 sec")
        self.pause_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #5bc0de;")
        self.pause_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        threshold_layout.addRow("Durată Pauză:", pause_container)
        threshold_layout.addRow("", self.pause_label)

        # --- BLOC NOU PENTRU DURATA MAXIMĂ ---
        max_speech_container = QVBoxLayout()
        
        self.max_speech_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_speech_slider.setRange(10, 30) # Interval de la 10 la 30 de secunde
        self.max_speech_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.max_speech_slider.setTickInterval(5)
        self.max_speech_slider.setMinimumHeight(40)
        self.max_speech_slider.valueChanged.connect(self.on_max_speech_changed)
        
        max_speech_container.addWidget(self.max_speech_slider)
        
        max_speech_labels_layout = QHBoxLayout()
        for val in [10, 15, 20, 25, 30]:
            label = QLabel(f"{val}s")
            label.setStyleSheet("font-size: 9px; color: #666;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            max_speech_labels_layout.addWidget(label)
        max_speech_container.addLayout(max_speech_labels_layout)
        
        self.max_speech_label = QLabel("15 sec") # Valoare default
        self.max_speech_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #337ab7;") # O culoare albastră
        self.max_speech_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        threshold_layout.addRow("Durată Max. Segment:", max_speech_container)
        threshold_layout.addRow("", self.max_speech_label)
        # --- SFÂRȘIT BLOC NOU ---
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        layout.addStretch()
        return widget

    def on_echo_cancellation_changed(self, state):
        enabled = (state == Qt.CheckState.Checked.value)
        self.config["enable_echo_cancellation"] = enabled
        save_config(self.config)
        log_timestamp(f"⚙️ [CONFIG] Anulare ecou: {enabled}")
        
        # Actualizează worker-ul activ, dacă rulează
        if self.voice_worker:
            self.voice_worker.enable_echo_cancellation = enabled
            log_timestamp("🎤 [WORKER UPDATE] Setarea de ecou a fost actualizată în timp real.", "app")

    def on_auto_calibrate_changed(self, state):
        enabled = (state == Qt.CheckState.Checked.value)
        self.config["auto_calibrate_on_mic_start"] = enabled
        save_config(self.config)  # ⭐ ADAUGĂ ACEASTĂ LINIE
        log_timestamp(f"⚙️ [CONFIG] Calibrare auto: {enabled}")

    def on_auto_start_mic_changed(self, state):
        enabled = (state == Qt.CheckState.Checked.value)
        self.config["auto_start_mic_with_conversation"] = enabled
        save_config(self.config)  # ⭐ ADAUGĂ ACEASTĂ LINIE
        log_timestamp(f"⚙️ [CONFIG] Pornire auto microfon: {enabled}")

    def on_no_camera_changed(self, state):
        enabled = (state == Qt.CheckState.Checked.value)
        self.config["conversation_without_camera"] = enabled
        save_config(self.config)  # ⭐ ADAUGĂ ACEASTĂ LINIE
        log_timestamp(f"⚙️ [CONFIG] Conversație fără cameră: {enabled}")

    def on_threshold_changed(self, value):
        self.threshold = value
        self.config["threshold"] = value # Adaugă valoarea în dicționarul de config
        save_config(self.config) # Salvează pe disc
        self.threshold_label.setText(f"{value}")
        self.threshold_indicator.setText(f"Threshold: {value}")
        log_timestamp(f"🎚️ [THRESHOLD] Modificat manual și salvat: {value}")
        
    def on_margin_changed(self, value):
        self.margin_percent = value
        self.config["margin_percent"] = value
        save_config(self.config)
        self.margin_label.setText(f"+{value}%")
        log_timestamp(f"📊 [MARJĂ] Modificată și salvată: +{value}%")
        
    def on_pause_changed(self, value):
        self.pause_duration = value / 10.0
        # Atenție: salvăm valoarea brută a slider-ului (ex: 20), nu valoarea calculată (2.0)
        self.config["pause_duration"] = self.pause_duration 
        save_config(self.config)
        self.pause_label.setText(f"{self.pause_duration:.1f} sec")
        log_timestamp(f"⏱️ [PAUZĂ] Modificată și salvată: {self.pause_duration:.1f}s")

    def toggle_conversation_state(self):
        if self.conversation_state == 'INACTIVE':
            self.conversation_state = 'ACTIVE'
            log_timestamp("=" * 70)
            log_timestamp("💬 [APP] === CONVERSAȚIE ACTIVATĂ ===")
            
            log_timestamp("🧠 [MEMORIE] Se resetează memoria de saluturi.")
            self.greeted_users = {}
            
            # NOU: Resetăm și jurnalul conversației
            log_timestamp("📓 [LOG] Se resetează jurnalul conversației.")
            self.conversation_log = []

            self._start_idle_animations() # "Trezește" personajele

            if not self.config["conversation_without_camera"]:
                self.start_webcam()
            if self.config["auto_start_mic_with_conversation"] and not self.voice_enabled:
                QTimer.singleShot(200, self.toggle_voice)
            
        else: # Când se apasă "Oprește Conversație"
            self.conversation_state = 'INACTIVE'
            log_timestamp("=" * 70)
            log_timestamp("💬 [APP] === CONVERSAȚIE DEZACTIVATĂ ===")
            
            # --- BLOC NOU DE ADĂUGAT ---
            # Oprește microfonul dacă este pornit
            if self.voice_enabled:
                self.toggle_voice()
            # --- SFÂRȘIT BLOC NOU ---
            
            self.stop_webcam()
            self._stop_idle_animations() # "Adoarme" personajele
        
        self.update_ui_for_state()

    def update_ui_for_state(self):
        if self.conversation_state == 'ACTIVE':
            self.conversation_button.setText("⏹️ Oprește")
            self.conversation_button.setStyleSheet("background-color: #d9534f;")
            self.text_input.setEnabled(True)
            self.text_input.setPlaceholderText("Scrie sau vorbește...")
            self.chat_history.clear()
            self.add_to_chat("Asistent", "Salut! Sunt gata de conversație.")
        else:
            self.conversation_button.setText("🚀 Start Conversație")
            self.conversation_button.setStyleSheet("background-color: #5cb85c;")
            self.text_input.setEnabled(False)
            self.text_input.setPlaceholderText("Apasă 'Start'...")
            self.webcam_label.setText("Camera oprită.")
            self.webcam_label.setStyleSheet("background-color: black; color: white; border: 2px solid gray;")

    def _load_slider_positions_from_config(self):
        """Setează pozițiile inițiale ale slider-elor din config."""
        self.threshold_slider.setValue(self.config["threshold"])
        self.margin_slider.setValue(self.config["margin_percent"])
        # Pentru pauză, convertim înapoi la valoarea slider-ului
        self.pause_slider.setValue(int(self.config["pause_duration"] * 10))
        self.max_speech_slider.setValue(self.config["max_speech_duration"]) # <-- ADAUGĂ

    def add_to_chat(self, user, message):
        """Adaugă un mesaj în fereastra de chat și face scroll automat în jos."""
        self.chat_history.append(f"<b>{user}:</b> {message}")
        
        # --- LINIA NOUĂ ȘI CRITICĂ ---
        # Obținem scrollbar-ul vertical și îi setăm valoarea la maximul posibil.
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())
        
    def start_webcam(self):
        log_timestamp("📷 [APP] Pornire webcam...")
        self.webcam_thread = QThread()
        self.webcam_worker = WebcamWorker()
        self.webcam_worker.moveToThread(self.webcam_thread)
        self.webcam_worker.frame_ready.connect(self.update_webcam_feed)
        self.webcam_thread.started.connect(self.webcam_worker.run)
        self.webcam_thread.finished.connect(self.webcam_thread.quit)
        # ... restul conexiunilor pentru cleanup
        self.webcam_thread.start()

    def stop_webcam(self):
        if self.webcam_worker:
            self.webcam_worker.stop()
            log_timestamp("📷 [APP] Cerere de oprire webcam trimisă.")
            
    def update_webcam_feed(self, image):
        # Convertim QImage la QPixmap pentru a-l afișa
        pixmap = QPixmap.fromImage(image)
        self.webcam_label.setPixmap(pixmap.scaled(
            self.webcam_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def toggle_voice(self):
        if not self.voice_enabled:
            log_timestamp("=" * 70)
            log_timestamp("🎤 [APP] === ACTIVARE MICROFON ===")
            if self.config["auto_calibrate_on_mic_start"]:
                log_timestamp("🔄 [AUTO] Se rulează calibrarea sincronă înainte de pornire...")
                self.do_calibration_sync()
            
            self.voice_enabled = True
            self.voice_toggle_btn.setText("🔴 Oprește Microfon")
            self.voice_toggle_btn.setStyleSheet("background-color: #d9534f;")
            
            self.start_continuous_voice()

            # Activăm butonul Mute
            self.mute_button.setEnabled(True) 
            
            # Afișează ÎNTREGUL grup (semafor + etichete)
            self.semafor_container.show()
            
            # Setează starea inițială/vizuală corectă la pornire
            self.on_language_lock_requested('nativ') 
            self._update_semafor_state('verde')
            log_timestamp("🚦 [SEMAFOR DEBUG] Comandă AFiȘARE semafor executată.", "semafor")
        else:
            log_timestamp("=" * 70)
            log_timestamp("🎤 [APP] === DEZACTIVARE MICROFON ===")
            self.voice_enabled = False
            self.voice_toggle_btn.setText("🟢 Activează Microfon")
            self.voice_toggle_btn.setStyleSheet("background-color: #5cb85c;")

            # Resetăm și dezactivăm butonul Mute
            self.mute_button.setEnabled(False)
            self.is_muted = False
            self.mute_button.setText("🎤 Mut")
            self.mute_button.setStyleSheet("background-color: #f0ad4e;")
            if self.voice_worker:
                self.voice_worker.set_muted(False) # Asigurăm că worker-ul nu rămâne pe mute
            
            self.stop_continuous_voice()
            
            # Ascunde ÎNTREGUL grup (semafor + etichete)
            self.semafor_container.hide()
            log_timestamp("🚦 [SEMAFOR DEBUG] Comandă ASCUNDERE semafor executată.", "semafor")

    def do_calibration_sync(self):
        """Calibrare sincronă - se execută ÎNAINTE de pornirea worker-ului"""
        log_timestamp("=" * 70)
        log_timestamp("🔄 [CALIBRARE] === START CALIBRARE SINCRONĂ ===")
        log_timestamp("🔄 [CALIBRARE] Stai în liniște 3 secunde...")
        
        try:
            log_timestamp("🔄 [CALIBRARE] Creez recognizer...")
            recognizer = sr.Recognizer()
            
            log_timestamp("🔄 [CALIBRARE] Încerc să deschid microfonul...")
            with sr.Microphone() as source:
                log_timestamp("🔄 [CALIBRARE] ✅ Microfon deschis cu succes!")
                log_timestamp("🔄 [CALIBRARE] Ascult zgomotul de fundal...")
                
                # Actualizare UI
                if hasattr(self, 'calibration_result'):
                    self.calibration_result.setText("🔊 Calibrare în curs (3 sec)...")
                QApplication.processEvents()
                
                recognizer.adjust_for_ambient_noise(source, duration=3)
                
                noise_level = recognizer.energy_threshold
                suggested_threshold = int(noise_level * (1 + self.margin_percent/100))
                
                log_timestamp("=" * 70)
                log_timestamp("✅ [CALIBRARE] === REZULTATE CALIBRARE ===")
                log_timestamp(f"📊 [CALIBRARE] Zgomot detectat: {noise_level:.0f}")
                log_timestamp(f"📊 [CALIBRARE] Marjă configurată: +{self.margin_percent}%")
                log_timestamp(f"📊 [CALIBRARE] Threshold recomandat: {suggested_threshold}")
                log_timestamp(f"📊 [CALIBRARE] Threshold vechi: {self.threshold}")
                
                final_threshold = min(suggested_threshold, 10000)
                if final_threshold != suggested_threshold:
                    log_timestamp(f"⚠️ [CALIBRARE] Threshold limitat la maxim: 10000")
                
                log_timestamp(f"📊 [CALIBRARE] Setez threshold la: {final_threshold}")
                self.threshold_slider.setValue(final_threshold)
                
                log_timestamp(f"📊 [CALIBRARE] ✅ Threshold NOU setat: {final_threshold}")
                log_timestamp("=" * 70)
                
                # Actualizare UI
                if hasattr(self, 'calibration_result'):
                    self.calibration_result.setText(
                        f"✅ Calibrare completă!\n"
                        f"Zgomot: {noise_level:.0f}\n"
                        f"Threshold: {final_threshold}"
                    )
                
                diff = final_threshold - noise_level
                log_timestamp(f"💡 [CALIBRARE] Diferență față de zgomot: +{diff:.0f} ({self.margin_percent}%)")
                
        except Exception as e:
            error_msg = f"Eroare calibrare: {e}"
            log_timestamp(f"❌ [CALIBRARE] {error_msg}")
            log_timestamp(f"❌ [CALIBRARE] Tip eroare: {type(e).__name__}")
            import traceback
            log_timestamp(f"❌ [CALIBRARE] Stack trace:\n{traceback.format_exc()}")
            
            if hasattr(self, 'calibration_result'):
                self.calibration_result.setText(f"⚠️ {error_msg}")
        finally:
            log_timestamp("=" * 70)

    def start_continuous_voice(self):
        log_timestamp("🎤 [APP] Pornire voice worker...")
        self.voice_thread = QThread()
        
        # Citim setarea o singură dată din config
        echo_setting = self.config.get("enable_echo_cancellation", True)

        self.voice_worker = ContinuousVoiceWorker(
            self.threshold, 
            self.pause_duration, 
            self.margin_percent, 
            self.max_speech_duration,
            enable_echo_cancellation=echo_setting # <-- PASĂM VALOAREA DIRECT LA CREARE
        )

        # Conectare semnale pentru semafor și modul NATIV
        self.voice_worker.language_lock_requested.connect(self.on_language_lock_requested)
        self.voice_worker.speech_activity_changed.connect(self.on_speech_activity_changed)
        self.voice_worker.pause_progress_updated.connect(self.on_pause_progress_updated)
        self.voice_worker.speech_time_updated.connect(self.on_speech_time_updated)
        
        self.voice_worker.moveToThread(self.voice_thread)
        
        # Conectare semnale standard
        self.voice_worker.transcription_ready.connect(self.handle_voice_transcription)
        self.voice_worker.status_changed.connect(self.update_voice_status)
        self.voice_worker.audio_level_changed.connect(self.update_audio_meter)
        
        self.voice_thread.started.connect(self.voice_worker.run)
        self.voice_thread.start()

    def toggle_mute_state(self):
        """Comută starea de mute a microfonului."""
        log_timestamp("🔘 [UI] Butonul MUTE a fost apăsat.", "app")
        
        self.is_muted = not getattr(self, 'is_muted', False)
        
        if self.voice_worker:
            self.voice_worker.set_muted(self.is_muted, is_ai_speaking=False)

        if self.is_muted:
            self.mute_button.setText("🎧 Ascult")
            self.mute_button.setStyleSheet("background-color: #5cb85c;")
        else:
            self.mute_button.setText("🎤 Mut")
            self.mute_button.setStyleSheet("background-color: #f0ad4e;")

    def repeat_last_audio(self):
        """Redă ultimul fișier audio generat de AI."""
        log_timestamp("🔁 [APP] Butonul 'Repetă' a fost apăsat.", "app")
        
        if hasattr(self, 'last_audio_file_path') and self.last_audio_file_path and os.path.exists(self.last_audio_file_path) and not pygame.mixer.music.get_busy():
            
            # --- MODIFICARE AICI ---
            self._update_semafor_state('rosu') # Facem semaforul ROȘU
            
            if self.voice_worker:
                self.voice_worker.set_muted(True, is_ai_speaking=True)
            
            try:
                # Creăm un QTimer care va reactiva microfonul după ce se termină sunetul
                sound = pygame.mixer.Sound(self.last_audio_file_path)
                duration_ms = int(sound.get_length() * 1000) + 200 # Durata în ms + o marjă de siguranță
                
                QTimer.singleShot(duration_ms, self.unmute_after_repeat)
                
                # Redăm sunetul
                pygame.mixer.music.load(self.last_audio_file_path)
                pygame.mixer.music.play()
                log_timestamp(f"🔁 [APP] Se repetă: {os.path.basename(self.last_audio_file_path)} (Durată: {duration_ms / 1000.0:.2f}s)", "app")
            except Exception as e:
                log_timestamp(f"❌ [APP] Eroare la redarea fișierului de repetat: {e}", "app")
                # În caz de eroare, ne asigurăm că reactivăm microfonul
                self.unmute_after_repeat()
        else:
            log_timestamp("⚠️ [APP] Niciun fișier audio de repetat sau redare în curs.", "app")

    def unmute_after_repeat(self):
        """Funcție de callback pentru a reactiva microfonul și semaforul."""
        log_timestamp("🎤 [APP] Redarea repetată s-a încheiat. Se actualizează starea.", "mute")
        
        # --- MODIFICARE AICI ---
        # Verificăm starea Mute a utilizatorului ÎNAINTE de a decide culoarea semaforului
        if not self.is_muted:
            self._update_semafor_state('verde') # Facem semaforul VERDE
            if self.voice_worker:
                self.voice_worker.set_muted(False)
        else:
            # Dacă utilizatorul este încă pe Mute manual, semaforul rămâne ROȘU
            log_timestamp("🔇 [APP] Microfonul rămâne pe mute la cererea utilizatorului.", "mute")
            self._update_semafor_state('rosu')

    def stop_continuous_voice(self):
        if self.voice_thread and self.voice_thread.isRunning():
            log_timestamp("🎤 [APP] Cerere de oprire pentru worker-ul de voce...")
            if self.voice_worker: self.voice_worker.stop()
            self.voice_thread.quit()
            if self.voice_thread.wait(3000):
                log_timestamp("🎤 [APP] ✅ Thread-ul de voce s-a oprit.")
            else:
                log_timestamp("🎤 [APP] ⚠️ Thread-ul de voce nu s-a oprit la timp.")

    def on_speech_time_updated(self, timp_ramas):
        """Actualizează textul cronometrului din becul verde."""
        if timp_ramas >= 0:
            if not self.cronometru_label.isVisible():
                self.cronometru_label.show()
            self.cronometru_label.setText(str(int(timp_ramas)))
        else: # Valoare negativă semnalează ascunderea
            self.cronometru_label.hide()

    def on_speech_activity_changed(self, is_speaking):
        """Actualizează semaforul când utilizatorul începe sau termină de vorbit."""
        if is_speaking:
            # Când începe să vorbească, clepsidra trebuie să fie plină/ascunsă
            self._update_semafor_state('verde') # Rămâne verde, dar asigură că ascunde clepsidra
        else:
            # Când a terminat de vorbit, revenim la verde simplu
            self._update_semafor_state('verde')

    def on_pause_progress_updated(self, progress):
        """Actualizează clepsidra când utilizatorul face o pauză."""
        if progress < 100:
            self._update_semafor_state('pauza', progress)
        else:
            # Dacă progresul e 100 (adică nu e pauză), stăm pe verde
            self._update_semafor_state('verde')

    def handle_voice_transcription(self, text):
        log_timestamp(f"💬 [APP] Voce primită: '{text}'", "app")
        self.add_to_chat("Tu (voce)", text)
        
        # ⭐ NOU: User vorbește
        self.set_speaker("user")
        
        if self.conversation_state == 'ACTIVE':
            self._route_user_input(text)

    def update_voice_status(self, status):
        self.voice_status_label.setText(status)
        
    def update_audio_meter(self, level):
        """Actualizează nivelul audio FĂRĂ logging periodic când categoria e dezactivată"""
        self.current_audio_level = level
        display_level = int(min(max(level, 0), 10000))
        self.audio_meter.setValue(display_level)
        
        if level > self.threshold:
            self.audio_meter.setStyleSheet("QProgressBar::chunk { background-color: #5cb85c; }")
            status = "🟢 PESTE"
        else:
            self.audio_meter.setStyleSheet("QProgressBar::chunk { background-color: #f0ad4e; }")
            status = "🟡 SUB"
        
        # Logging periodic DOAR dacă categoria "audio" e activată
        if not hasattr(self, '_last_audio_log_time'):
            self._last_audio_log_time = time.time()
            
        if time.time() - self._last_audio_log_time >= 2.0:
            log_timestamp(
                f"📊 [AUDIO] Nivel: {level:.0f} | Threshold: {self.threshold} | "
                f"Status: {status} | Diferență: {level - self.threshold:+.0f}",
                "audio"  # ⭐ ADĂUGAT CATEGORIA
            )
            self._last_audio_log_time = time.time()

    def _apply_saved_character_settings(self):
        """Aplică setările specifice personajelor (ex: limba) încărcate din config."""
        log_timestamp("⚙️ [CONFIG] Se aplică setările salvate pentru personaje...", "app")
        
        saved_code = self.config.get("rina_language_code", "en")
        
        for lang_name, lang_details in self.RINA_LANGUAGES.items():
            if lang_details["code"] == saved_code:
                rina_char = self.character_manager.get_character("rina_cat")
                if rina_char:
                    rina_char.set_language(lang_details["code"], lang_details["voice"])
                break

    def _start_idle_animations(self):
        log_timestamp("☀️ [ANIM] Se repornesc animațiile de idle...", "animator")
        for char in self.character_manager.get_active_characters_list():
            # Repornire animatoare
            for animator in char.animators:
                if isinstance(animator, (BreathingAnimator, BlinkingAnimator)):
                    animator.start()

            # Revenire la emoția neutră
            emotion_animator = next((anim for anim in char.animators if isinstance(anim, EmotionAnimator)), None)
            if emotion_animator:
                emotion_animator.reset_to_neutral()

    def _stop_idle_animations(self):
        log_timestamp("🌙 [ANIM] Se opresc animațiile de idle...", "animator")
        for char in self.character_manager.get_active_characters_list():
            # Oprire animatoare
            for animator in char.animators:
                if isinstance(animator, (BreathingAnimator, BlinkingAnimator)):
                    animator.stop()
            
            # Forțare ochi închiși (dacă are config)
            emotion_animator = next((anim for anim in char.animators if isinstance(anim, EmotionAnimator)), None)
            if emotion_animator:
                emotion_animator.set_emotion('sleeping') # Presupunem că există o emoție "sleeping"

    def on_rina_language_changed(self, language_name):
        """Apelată când utilizatorul selectează o nouă limbă pentru Rina."""
        if language_name not in self.RINA_LANGUAGES:
            return

        lang_details = self.RINA_LANGUAGES[language_name]
        lang_code = lang_details["code"]
        voice_id = lang_details["voice"]

        rina_char = self.character_manager.get_character("rina_cat")
        if rina_char:
            success = rina_char.set_language(lang_code, voice_id)
            if success:
                self.config["rina_language_code"] = lang_code
                save_config(self.config)
                log_timestamp(f"✅ [CONFIG] Limba pentru Rina a fost salvată: '{lang_code}'", "app")
                
                # --- BLOCUL DE RESETARE A FOST COMPLET ELIMINAT ---

    def _update_subtitle_style(self):
        """Actualizează stilul CSS pentru subtitrare pe baza setărilor."""
        font_size = self.config.get("subtitle_font_size", 26)
        style = (
            f"background-color: rgba(0, 0, 0, 0.5);"
            f"color: white;"
            f"font-size: {font_size}px;"
            f"font-weight: bold;"
            f"border-radius: 10px;"
            f"padding: 10px;"
        )
        self.subtitle_label.setStyleSheet(style)

    def on_subtitle_font_size_changed(self, value):
        """Apelată când slider-ul pentru mărimea fontului este mișcat."""
        self.config["subtitle_font_size"] = value
        save_config(self.config)
        self._update_subtitle_style()
        self.subtitle_font_label.setText(f"Mărime font: {value}px")

    def send_to_ai(self):
        question = self.text_input.text().strip()
        if not question:
            return

        self.add_to_chat("Tu (text)", question)
        self.text_input.clear()
        
        # ⭐ NOU: User vorbește (prin text)
        self.set_speaker("user")
        
        self._route_user_input(question)

    def process_question(self, question, target_character_id):
        # --- Citim model_name la începutul funcției ---
        model_name = self.config.get("ai_model_name", "models/gemini-flash-lite-latest")
        log_timestamp(f"🤖 [GEMINI] Se va folosi modelul: {model_name}", "gemini_debug")

        if not question or self.is_speaking or self.is_thinking:
            log_timestamp(f"⚠️ [APP] Întrebare ignorată (stare ocupată: speaking={self.is_speaking}, thinking={self.is_thinking})", "app")
            return

        if target_character_id not in self.character_manager.active_characters:
            log_timestamp(f"🔇 [PROCESS] Personaj '{target_character_id}' nu e pe scenă → SILENCE", "app")
            return

        target_character = self.character_manager.get_character(target_character_id)
        if not target_character:
            log_timestamp(f"❌ [APP] Nu am găsit personajul țintă '{target_character_id}'!", "app")
            self.add_to_chat("Sistem", f"Eroare: personajul {target_character_id} nu există.")
            return
        
        log_timestamp(f"🤖 [APP] === PROCESARE ÎNTREBARE PENTRU '{target_character_id}' ===", "app")
        
        self.conversation_log.append({"role": "user", "content": question})
        
        self.is_thinking = True
        self.disable_all_actions()
        self._update_semafor_state('rosu')

        if self.voice_worker:
            log_timestamp("🔇 [MUTE] Microfonul este pus în pauză pe durata gândirii și vorbirii.", "mute")
            self.voice_worker.set_muted(True, is_ai_speaking=True)

        if self.config.get("enable_filler_sounds", True):
            self.play_filler_sound(target_character)
        
        self.thinking_timer.start(500)

        # --- CONSTRUIREA DINAMICĂ A PROMPT-ULUI ---
        system_prompt_base = target_character.get_prompt_content()
        world_knowledge = self._generate_world_knowledge(target_character_id)
        family_briefing = self._generate_family_briefing() # <-- APELĂM NOUA FUNCȚIE

        # Logica pentru `instruction_addon` (persoane salutate)
        if target_character_id not in self.greeted_users:
            self.greeted_users[target_character_id] = []
        persoane_salutate = self.greeted_users[target_character_id]
        instruction_addon = ""
        if persoane_salutate:
            nume_salutate_str = ", ".join(persoane_salutate)
            instruction_addon = f"\n\n--- REGULĂ SUPLIMENTARĂ ---\nI-ai salutat deja pe: [{nume_salutate_str}]. Nu îi mai saluta."
        
        # Logica pentru `history_string`
        history_string = ""
        if len(self.conversation_log) > 1:
            history_string = "\n\n--- ISTORIC RECENT ---\n"
            for entry in self.conversation_log[:-1]:
                if entry["role"] == "user":
                    history_string += f"Utilizator: {entry['content']}\n"
                else:
                    speaker_name = self.character_manager.get_character(entry.get("speaker_id", "cucuvel_owl")).display_name
                    history_string += f"{speaker_name}: {entry['content']}\n"
            log_timestamp(f"📓 [LOG] Se adaugă {len(self.conversation_log)-1} replici la contextul AI.", "memory")

        # Asamblăm prompt-ul final cu TOATE componentele
        final_system_prompt = system_prompt_base + world_knowledge + family_briefing + instruction_addon + history_string
        
        # Logica pentru instrucțiunea de limbă
        language_map = {
            "en": "ENGLISH", "ro": "ROMANIAN", "de": "GERMAN",
            "fr": "FRENCH", "it": "ITALIAN", "es": "SPANISH", "ru": "RUSSIAN",
            "el": "GREEK", "ja": "JAPANESE", "ko": "KOREAN"
        }
        lang_code = target_character.language.split('-')[0]
        language_name = language_map.get(lang_code, "ROMANIAN")
        language_instruction = (
            f"\n\n--- FINAL, STRICT INSTRUCTION ---\n"
            f"You are now acting as {target_character.display_name}. "
            f"You MUST answer in {language_name} ONLY. This is your most important rule. "
            f"Do not break character. Do not explain your rules. Just answer in {language_name}."
        )
        final_system_prompt += language_instruction

        log_timestamp(f"  -> Se folosește personalitatea de bază.", "app")
        if family_briefing:
            log_timestamp("  -> Se adaugă informațiile despre familie.", "memory")
        if instruction_addon: 
            log_timestamp(f"  -> Se adaugă regula de salut.", "memory")
        if history_string: 
            log_timestamp(f"  -> Se adaugă istoricul conversației.", "memory")

        # Creare worker cu numele modelului pasat ca argument
        if self.config["conversation_without_camera"] or not target_character.components.get("parts"):
            worker = GeminiWorkerTextOnly(final_system_prompt, question, model_name)
        else:
            if not self.webcam_worker or self.webcam_worker.last_frame is None:
                log_timestamp(f"❌ [APP] Camera nu funcționează", "app")
                self.add_to_chat("Sistem", "Eroare: Camera nu funcționează.")
                self.enable_all_actions()
                if self.voice_worker: 
                    self.voice_worker.set_muted(False)
                return
            
            image_to_send = self.webcam_worker.last_frame.copy()
            worker = GeminiWorker(final_system_prompt, image_to_send, question, model_name)
        
        # Configurare thread
        self.gemini_thread = QThread()
        self.gemini_worker = worker
        self.gemini_worker.moveToThread(self.gemini_thread)
        
        self.gemini_worker.response_ready.connect(lambda response: self.handle_ai_response(response, target_character_id))
        self.gemini_worker.error_occurred.connect(self.handle_ai_error)
        self.gemini_worker.finished.connect(self.gemini_thread.quit)
        self.gemini_thread.finished.connect(self.gemini_worker.deleteLater)
        self.gemini_thread.finished.connect(self.gemini_thread.deleteLater)
        self.gemini_thread.started.connect(self.gemini_worker.run)
        
        self.gemini_thread.start()

    def handle_ai_response(self, response_text, speaking_character_id):
        # --- BLOC DE DEBUGGING ---
        log_timestamp("🐞 [DEBUG] PAS 1: Intrat în handle_ai_response.", "app")
        
        log_timestamp(f"✅ [APP] Răspuns AI (brut): '{response_text[:120]}...'", "app")
        
        self.subtitle_scroll_area.hide()
        self.translation_scroll_area.hide()
        
        self.stop_thinking()
        
        log_timestamp("🐞 [DEBUG] PAS 2: Obținere personaj.", "app")
        speaking_character = self.character_manager.get_character(speaking_character_id)
        if not speaking_character:
            self.speech_finished()
            return
            
        emotion = "neutral"
        original_text = ""
        translation_text = ""
        
        log_timestamp("🐞 [DEBUG] PAS 3: Se intră în blocul try...except pentru parsare JSON.", "app")
        try:
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_string = response_text[start_index : end_index + 1]
                response_data = json.loads(json_string)
                emotion = response_data.get("emotion", "neutral")
                original_text = response_data.get("original", "")
                translation_text = response_data.get("translation", "")
                log_timestamp(f"✅ [JSON PARSE] Emoție: '{emotion}', Original: '{original_text[:50]}...'", "app")
            else:
                raise ValueError("Nu s-a găsit un obiect JSON valid în răspuns.")

        except (json.JSONDecodeError, ValueError) as e:
            log_timestamp(f"⚠️ [JSON PARSE] Eroare la parsare: {e}. Tratăm răspunsul ca text simplu.", "app")
            original_text = self._extract_and_apply_emotion(response_text, speaking_character_id)
            emotion = getattr(self, 'last_extracted_emotion', 'neutral')

        log_timestamp("🐞 [DEBUG] PAS 4: Verificare text original.", "app")
        if not original_text:
            log_timestamp("⚠️ [APP] Textul original este gol. Se anulează redarea.", "app")
            self.speech_finished()
            return

        log_timestamp("🐞 [DEBUG] PAS 5: Se aplică emoția.", "app")
        if 'response_data' in locals():
            self._apply_emotion(emotion, speaking_character_id)

        self.last_character_speeches[speaking_character_id] = original_text

        # --- BLOCUL DE ACTUALIZARE A LIMBII PENTRU FOCUS (PE CARE L-AM OMIS) ---
        if self.voice_worker:
            lang_code_map = {
                "en": "en-US", "ro": "ro-RO", "de": "de-DE",
                "fr": "fr-FR", "it": "it-IT", "es": "es-ES", "ru": "ru-RU",
                "el": "el-GR", "ja": "ja-JP", "ko": "ko-KR"
            }
            short_code = speaking_character.language.split('-')[0]
            full_code_for_stt = lang_code_map.get(short_code, "ro-RO")
            
            self.voice_worker.set_primary_language(full_code_for_stt)
            log_timestamp(f"🎤 [FOCUS] Limba de ascultare pentru Focus a fost actualizată la: '{full_code_for_stt}'.", "app")
        # --- SFÂRȘIT BLOC ---

        log_timestamp("🐞 [DEBUG] PAS 6: Se actualizează memoria de saluturi.", "app")
        if speaking_character_id not in self.greeted_users:
            self.greeted_users[speaking_character_id] = []
        
        persoane_salutate_anterior = self.greeted_users[speaking_character_id]
        nume_cunoscute = ["Mihai", "Anca", "Matei"]
        
        for nume in nume_cunoscute:
            if nume in original_text and nume not in persoane_salutate_anterior:
                log_timestamp(f"🧠 [MEMORIE] AI l-a identificat și salutat pe '{nume}'. Se adaugă la memorie.", "memory")
                self.greeted_users[speaking_character_id].append(nume)

        self.conversation_log.append({"role": "ai", "content": original_text, "speaker_id": speaking_character_id})
        while len(self.conversation_log) > self.MAX_LOG_ENTRIES:
            self.conversation_log.pop(0)

        log_timestamp("🐞 [DEBUG] PAS 7: Se pregătesc subtitrările.", "app")
        text_to_display_bottom = original_text
        if speaking_character_id == "rina_cat":
            subtitle_mode = self.config.get("subtitle_mode", "original")
            lang_code = speaking_character.language.split('-')[0]
            
            # --- MODIFICARE AICI ---
            if lang_code in ['el', 'ru', 'ja', 'ko']: # Adăugăm 'ja' și 'ko'
                if subtitle_mode == "latin (fonetic)":
                    text_to_display_bottom = self._transliterate_text(original_text, lang_code)
                elif subtitle_mode == "combinat":
                    transliterated = self._transliterate_text(original_text, lang_code)
                    text_to_display_bottom = (f"<div style='font-size: 26px;'>{transliterated}</div>"
                                              f"<div style='font-size: 16px; color: #ccc;'>[{original_text}]</div>")
        
        self.subtitle_label.setText(text_to_display_bottom)
        self.subtitle_label.adjustSize()
        self.subtitle_scroll_area.show()
        self.subtitle_scroll_area.raise_()

        if translation_text:
            self.translation_label.setText(translation_text)
            self.translation_label.adjustSize()
            self.translation_scroll_area.show()
            self.translation_scroll_area.raise_()

        log_timestamp("🐞 [DEBUG] PAS 8: Se adaugă la chat.", "app")
        self.add_to_chat(speaking_character.display_name, original_text)
        if translation_text:
            self.add_to_chat(f"({speaking_character.display_name} - Tradus)", translation_text)
        
        if self.voice_worker:
            self.voice_worker.set_last_ai_text(original_text)
        
        # --- BLOC NOU: Ștergerea fișierului audio anterior ---
        if hasattr(self, 'last_audio_file_path') and self.last_audio_file_path and os.path.exists(self.last_audio_file_path):
            try:
                os.remove(self.last_audio_file_path)
                log_timestamp(f"🧹 [CLEANUP] Fișierul audio vechi a fost șters: {self.last_audio_file_path}", "cleanup")
            except Exception as e:
                log_timestamp(f"⚠️ [CLEANUP] Eroare la ștergerea fișierului vechi: {e}", "cleanup")
        # --- SFÂRȘIT BLOC NOU ---

        log_timestamp("🐞 [DEBUG] PAS 9: Se pregătește pornirea TTS.", "app")
        tts_worker = TTSWorker(original_text)
        tts_worker.voice = speaking_character.voice_id
        self.start_sync_process(tts_worker, original_text, speaking_character_id)
        log_timestamp("🐞 [DEBUG] PAS 10: S-a terminat handle_ai_response.", "app")

    def _apply_emotion(self, emotion_name, character_id):
        """Funcție ajutătoare simplă pentru a aplica o emoție."""
        log_timestamp(f"🎭 [EMOTION] Se aplică emoția: '{emotion_name}' pentru '{character_id}'", "emotion")
        character = self.character_manager.get_character(character_id)
        if character:
            for animator in character.animators:
                if isinstance(animator, EmotionAnimator):
                    animator.set_emotion(emotion_name)
                    break

    def _extract_and_apply_emotion(self, response_text, character_id):
        """
        Extrage tag-ul de emoție din răspunsul AI și îl aplică.
        Această funcție va fi folosită DOAR ca fallback, dacă parsarea JSON eșuează.
        
        Returns:
            str: Răspunsul curățat (fără tag-ul de emoție)
        """
        import re
        
        # Căutăm pattern-ul [EMOTION:nume_emotie]
        emotion_pattern = r'\[EMOTION:(\w+)\]\s*'
        match = re.match(emotion_pattern, response_text)
        
        emotion_name = "neutral" # Default
        clean_text = response_text

        if match:
            emotion_name = match.group(1).lower()
            clean_text = re.sub(emotion_pattern, '', response_text, count=1).strip()
        
        log_timestamp(f"🎭 [EMOTION] Se aplică emoția: '{emotion_name}' pentru '{character_id}'", "emotion")
        
        # Aplicăm emoția
        character = self.character_manager.get_character(character_id)
        if character:
            for animator in character.animators:
                if isinstance(animator, EmotionAnimator):
                    animator.set_emotion(emotion_name)
                    break
        
        # Stocăm ultima emoție extrasă pentru cazul de fallback
        self.last_extracted_emotion = emotion_name
        return clean_text

    def handle_ai_error(self, error_message):
        log_timestamp(f"❌ [APP EROARE AI] {error_message}", "app")
        self.stop_thinking()
        self.add_to_chat("Sistem", error_message)
        self.enable_all_actions()
        if self.voice_worker:
            self.voice_worker.set_muted(False)

    def create_general_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # --- BLOC COMPLET REVIZUIT: Grup pentru Modelul AI cu Buton Apply ---
        ai_group = QGroupBox("🧠 Model Inteligență Artificială")
        ai_layout = QFormLayout(ai_group)
        
        self.ai_model_combo = QComboBox()
        
        successful_models = [
            "models/gemini-2.5-flash-preview-09-2025", "models/gemini-2.5-flash-lite",
            "models/gemini-2.5-flash-lite-preview-09-2025", "models/gemini-flash-latest",
            "models/gemini-2.5-flash-lite-preview-06-17", "models/gemini-2.5-flash-image-preview",
            "models/gemini-flash-lite-latest", "models/gemini-2.5-flash-image",
            "models/gemma-3-27b-it", "models/gemini-2.0-flash-exp",
            "models/gemma-3n-e2b-it", "models/gemma-3n-e4b-it",
            "models/gemini-2.0-flash-thinking-exp-1219", "models/gemini-2.5-flash-preview-05-20",
            "models/gemma-3-1b-it", "models/gemini-2.0-flash-thinking-exp-01-21",
            "models/gemini-2.5-flash", "models/gemma-3-4b-it", "models/gemma-3-12b-it",
            "models/gemini-2.0-flash-lite-001", "models/gemini-2.0-flash-lite-preview-02-05",
            "models/gemini-2.0-flash-lite", "models/gemini-robotics-er-1.5-preview",
            "models/gemini-2.0-flash-thinking-exp", "models/gemini-2.5-pro",
            "models/gemini-pro-latest", "models/gemini-2.5-pro-preview-05-06",
            "models/gemini-2.0-flash-lite-preview", "models/gemini-2.5-pro-preview-06-05",
            "models/gemini-2.0-flash-001", "models/gemini-2.5-pro-preview-03-25",
            "models/gemini-2.0-flash", "models/learnlm-2.0-flash-experimental"
        ]
        self.ai_model_combo.addItems(successful_models)
        
        # Conectăm schimbarea la o funcție care activează butonul
        self.ai_model_combo.currentTextChanged.connect(self.on_settings_changed)
        
        ai_layout.addRow("Selectează Model:", self.ai_model_combo)

        # Creăm butonul Apply, inițial dezactivat
        self.apply_button = QPushButton("✅ Apply Changes")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_general_settings)
        ai_layout.addRow(self.apply_button)
        
        layout.addWidget(ai_group)
        # --- SFÂRȘIT BLOC REVIZUIT ---

        # --- Grup pentru Furnizor Voce ---
        tts_group = QGroupBox("🎙️ Furnizor Voce (TTS)")
        tts_layout = QFormLayout(tts_group)
        self.tts_provider_combo = QComboBox()
        self.tts_provider_combo.addItems(["Microsoft Edge (Rapid și Gratuit)", "Google Cloud (Calitate Superioară)"])
        self.tts_provider_combo.currentTextChanged.connect(self.on_tts_provider_changed)
        tts_layout.addRow("Serviciu TTS:", self.tts_provider_combo)
        layout.addWidget(tts_group)

        # --- Grup pentru Limba Personajelor ---
        lang_group = QGroupBox("🌍 Limba Personajelor")
        lang_layout = QFormLayout(lang_group)
        self.rina_language_combo = QComboBox()
        self.rina_language_combo.addItems(self.RINA_LANGUAGES.keys())
        self.rina_language_combo.currentTextChanged.connect(self.on_rina_language_changed)
        lang_layout.addRow("Limba pentru Rina:", self.rina_language_combo)
        layout.addWidget(lang_group)

        # --- Grup pentru Subtitrări ---
        subtitle_group = QGroupBox("📝 Setări Subtitrări")
        subtitle_layout = QFormLayout(subtitle_group)
        
        # --- BLOC NOU: Meniu Dropdown pentru Mod Subtitrare ---
        self.subtitle_mode_combo = QComboBox()
        self.subtitle_mode_combo.addItems(["Original", "Latin (Fonetic)", "Combinat"])
        self.subtitle_mode_combo.currentTextChanged.connect(self.on_subtitle_mode_changed)
        subtitle_layout.addRow("Mod afișare subtitrare:", self.subtitle_mode_combo)
        # --- SFÂRȘIT BLOC NOU ---
        
        self.subtitle_font_slider = QSlider(Qt.Orientation.Horizontal)
        self.subtitle_font_slider.setRange(18, 40)
        self.subtitle_font_slider.valueChanged.connect(self.on_subtitle_font_size_changed)
        self.subtitle_font_label = QLabel()
        subtitle_layout.addRow("Mărime font:", self.subtitle_font_slider)
        subtitle_layout.addRow(self.subtitle_font_label)
        layout.addWidget(subtitle_group)

        layout.addStretch()
        return widget

    def _discover_available_domains(self):
        """
        Scanează folderul curriculum/ și descoperă toate domeniile de învățare disponibile.
        (VERSIUNE FINALĂ ȘI ROBUSTĂ PENTRU PARSARE)
        """
        log_timestamp("🔍 [CURRICULUM] Scanez folderul curriculum/ pentru domenii...", "app")
        
        curriculum_path = Path("curriculum")
        if not curriculum_path.exists():
            log_timestamp("⚠️ [CURRICULUM] Folderul curriculum/ nu există!", "app")
            return
        
        self.available_domains = {}
        for domain_folder in curriculum_path.iterdir():
            if not domain_folder.is_dir(): continue
            
            domain_id = domain_folder.name
            curriculum_file = domain_folder / "curriculum.json"
            if not curriculum_file.exists(): continue

            try:
                with open(curriculum_file, "r", encoding="utf-8") as f:
                    domain_data = json.load(f)
                
                if not all(k in domain_data for k in ["domain_id", "domain_name", "tiers"]):
                    log_timestamp(f"⚠️ [CURRICULUM] Domeniu '{domain_id}' are curriculum.json invalid. Ignorat.", "app")
                    continue
                
                for tier_info in domain_data["tiers"]:
                    tier_id = tier_info["tier_id"]
                    tier_prompt_path = domain_folder / "prompts" / f"{tier_id}.txt"
                    
                    tier_info["questions"] = []
                    if not tier_prompt_path.exists():
                        log_timestamp(f"⚠️ [CURRICULUM] Fișier lipsă: {tier_prompt_path}", "app")
                        continue

                    with open(tier_prompt_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    if "=== ÎNTREBĂRI ===" not in content:
                        continue

                    questions_section = content.split("=== ÎNTREBĂRI ===")[1].strip()
                    
                    # --- NOUA LOGICĂ DE PARSARE, MULT MAI ROBUSTĂ ---
                    # Împărțim fișierul într-un bloc pentru fiecare întrebare
                    question_blocks = re.split(r'\n\d+\.\s', '\n' + questions_section)[1:]

                    for i, block in enumerate(question_blocks, 1):
                        question_data = {"id": f"q{i}"}
                        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
                        
                        task_lines = []
                        for line in lines:
                            if line.startswith("Sarcina:"):
                                task_lines.append(line.split(":", 1)[1].strip())
                            elif line.startswith("display:"):
                                question_data["display"] = line.split(":", 1)[1].strip()
                            elif line.startswith("|"):
                                question_data["correct_answers"] = [ans.strip() for ans in line[1:].split(',')]
                            elif not line.startswith("["): # Ignorăm tag-urile [verbal] etc.
                                task_lines.append(line)
                        
                        question_data["text"] = " ".join(task_lines)
                        tier_info["questions"].append(question_data)
                    # --- SFÂRȘIT LOGICĂ NOUĂ ---

                    log_timestamp(f"  -> Încărcate {len(tier_info['questions'])} întrebări pentru {domain_id}/{tier_id}", category="curriculum")

                self.available_domains[domain_id] = domain_data
                log_timestamp(f"✅ [CURRICULUM] Domeniu încărcat: '{domain_data['domain_name']}' ({domain_id})", category="curriculum")
            
            except Exception as e:
                log_timestamp(f"❌ [CURRICULUM] Eroare la încărcarea domeniului '{domain_id}': {e}", "app")
        
        log_timestamp(f"🔍 [CURRICULUM] Total domenii disponibile: {len(self.available_domains)}", "app")

    def create_family_settings_tab(self):
        """
        Tab-ul pentru gestionarea membrilor familiei și a progresului lor de învățare.
        """
        widget = QWidget()
        main_layout = QHBoxLayout()
        widget.setLayout(main_layout)

        # --- Coloana Stângă: Lista de Membri și Butoane ---
        left_panel = QVBoxLayout()
        members_group = QGroupBox("Membrii Familiei")
        
        self.family_list_widget = QListWidget()
        self.family_list_widget.currentItemChanged.connect(self.on_family_member_selected)
        
        buttons_layout = QHBoxLayout()
        self.add_member_button = QPushButton("+ Adaugă")
        self.remove_member_button = QPushButton("- Șterge")
        self.add_member_button.clicked.connect(self.add_new_family_member)
        self.remove_member_button.clicked.connect(self.remove_selected_family_member)
        buttons_layout.addWidget(self.add_member_button)
        buttons_layout.addWidget(self.remove_member_button)

        left_panel.addWidget(self.family_list_widget)
        left_panel.addLayout(buttons_layout)
        members_group.setLayout(left_panel)

        # --- Coloana Dreaptă: Split în 2 secțiuni ---
        right_panel = QVBoxLayout()
        
        # === SECȚIUNEA 1: Detalii Membru (ca înainte) ===
        self.form_group = QGroupBox("Detalii Membru")
        form_layout = QFormLayout()

        self.member_name_edit = QLineEdit()
        self.member_role_combo = QComboBox()
        self.member_role_combo.addItems(["", "Tata", "Mama", "Copil", "Bunic", "Bunica", "Unchi", "Mătușă", "Alt Adult", "Animal de companie"])
        
        self.member_age_label = QLabel("Vârstă:")
        self.member_age_spinbox = QSpinBox()
        self.member_age_spinbox.setRange(0, 120)

        self.member_level_label = QLabel("Nivel (Copil):")
        self.member_level_spinbox = QSpinBox()
        self.member_level_spinbox.setRange(1, 10)

        self.member_pet_type_label = QLabel("Tip Animal:")
        self.member_pet_type_edit = QLineEdit()

        self.member_description_edit = QTextEdit()
        self.member_description_edit.setPlaceholderText("Ex: poartă ochelari, are părul lung și roșcat, este un câine auriu...")

        self.save_member_button = QPushButton("💾 Salvează Modificările")
        self.save_member_button.clicked.connect(self.save_family_member_details)
        
        form_layout.addRow("Nume:", self.member_name_edit)
        form_layout.addRow("Rol:", self.member_role_combo)
        form_layout.addRow(self.member_age_label, self.member_age_spinbox)
        form_layout.addRow(self.member_level_label, self.member_level_spinbox)
        form_layout.addRow(self.member_pet_type_label, self.member_pet_type_edit)
        form_layout.addRow("Semne Distinctive:", self.member_description_edit)
        
        self.form_group.setLayout(form_layout)
        
        right_panel.addWidget(self.form_group)
        right_panel.addWidget(self.save_member_button)
        
        # === SECȚIUNEA 2: Progres Învățare (NOU!) ===
        self.learning_progress_group = QGroupBox("📚 Progres Învățare")
        learning_layout = QHBoxLayout()
        
        # --- Panoul Stâng: Lista Domeniilor ---
        domains_panel = QVBoxLayout()
        domains_label = QLabel("Domenii Active:")
        self.domains_list_widget = QListWidget()
        self.domains_list_widget.currentItemChanged.connect(self.on_domain_selected)
        
        domain_buttons_layout = QHBoxLayout()
        self.add_domain_button = QPushButton("+ Adaugă Domeniu")
        self.remove_domain_button = QPushButton("- Șterge Domeniu")
        self.add_domain_button.clicked.connect(self.add_domain_to_member)
        self.remove_domain_button.clicked.connect(self.remove_domain_from_member)
        domain_buttons_layout.addWidget(self.add_domain_button)
        domain_buttons_layout.addWidget(self.remove_domain_button)
        
        domains_panel.addWidget(domains_label)
        domains_panel.addWidget(self.domains_list_widget)
        domains_panel.addLayout(domain_buttons_layout)
        
        # --- Panoul Drept: Detalii Domeniu ---
        details_panel = QVBoxLayout()
        
        tier_label = QLabel("Tier Curent:")
        self.tier_combo = QComboBox()
        self.tier_combo.currentTextChanged.connect(self.on_tier_changed_manually)
        
        progress_label = QLabel("Progres în Tier:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m întrebări")
        
        self.reset_tier_button = QPushButton("🔄 Resetează Tier Curent")
        self.reset_tier_button.clicked.connect(self.reset_current_tier)
        
        details_panel.addWidget(tier_label)
        details_panel.addWidget(self.tier_combo)
        details_panel.addWidget(progress_label)
        details_panel.addWidget(self.progress_bar)
        details_panel.addWidget(self.reset_tier_button)
        details_panel.addStretch()
        
        learning_layout.addLayout(domains_panel, 1)
        learning_layout.addLayout(details_panel, 1)
        self.learning_progress_group.setLayout(learning_layout)
        
        right_panel.addWidget(self.learning_progress_group)

        self.member_role_combo.currentTextChanged.connect(self.on_member_role_changed)
        
        main_layout.addWidget(members_group, 1)
        main_layout.addLayout(right_panel, 2)

        # Dezactivăm formularele la început
        self.form_group.setEnabled(False)
        self.save_member_button.setEnabled(False)
        self.learning_progress_group.setEnabled(False)
        
        return widget

    def _load_family_data(self):
        """Încarcă datele familiei din family.json."""
        self.family_data = []
        if os.path.exists("family.json"):
            try:
                with open("family.json", "r", encoding="utf-8") as f:
                    self.family_data = json.load(f)
                # --- LOG NOU ---
                log_timestamp(f"👨‍👩‍👧‍👦 [FAMILY LOAD] Datele familiei încărcate din family.json: {json.dumps(self.family_data)}", "app")
            except json.JSONDecodeError:
                log_timestamp("⚠️ [FAMILY LOAD] Eroare la citirea family.json. Fișierul ar putea fi corupt.", "app")
        else:
            # --- LOG NOU ---
            log_timestamp("ℹ️ [FAMILY LOAD] Fișierul family.json nu a fost găsit. Se pornește cu o listă goală.", "app")
        self._populate_family_list()

    def _save_family_data(self):
        """Salvează datele curente ale familiei în family.json."""
        try:
            # --- LOG NOU ---
            log_timestamp(f"💾 [FAMILY SAVE] Se salvează următorul conținut în family.json: {json.dumps(self.family_data)}", "app")
            with open("family.json", "w", encoding="utf-8") as f:
                json.dump(self.family_data, f, indent=2, ensure_ascii=False)
            log_timestamp("✅ [FAMILY SAVE] Salvarea family.json a reușit.", "app")
        except Exception as e:
            log_timestamp(f"❌ [FAMILY SAVE] Eroare la salvarea family.json: {e}", "app")

    def _populate_family_list(self):
        """Repopulează lista vizuală cu membrii familiei."""
        self.family_list_widget.clear()
        for i, member in enumerate(self.family_data):
            display_text = f"{member.get('name', 'N/A')} ({member.get('role', 'N/A')})"
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, i) # Stocăm indexul original în item
            self.family_list_widget.addItem(item)
    
    def on_family_member_selected(self, current_item, previous_item):
        """
        Apelată când un membru este selectat din listă.
        Actualizează atât formularul de detalii, cât și panoul de progres învățare.
        """
        if not current_item:
            self.form_group.setEnabled(False)
            self.save_member_button.setEnabled(False)
            self.learning_progress_group.setEnabled(False)
            return

        self.form_group.setEnabled(True)
        self.save_member_button.setEnabled(True)
        self.learning_progress_group.setEnabled(True)
        
        index = current_item.data(Qt.UserRole)
        member = self.family_data[index]

        # Populăm formularul de detalii (ca înainte)
        self.member_name_edit.setText(member.get("name", ""))
        self.member_role_combo.setCurrentText(member.get("role", ""))
        self.member_age_spinbox.setValue(member.get("age", 0))
        self.member_level_spinbox.setValue(member.get("level", 1))
        self.member_pet_type_edit.setText(member.get("type", ""))
        self.member_description_edit.setPlainText(member.get("description", ""))
        
        # Actualizăm vizibilitatea câmpurilor pe baza rolului
        self.on_member_role_changed(member.get("role", ""))
        
        # === NOU: Populăm panoul de progres învățare ===
        self._populate_learning_progress_panel(member)

    def on_member_role_changed(self, role):
        """Ascunde/afișează câmpurile specifice în funcție de rol."""
        is_child = (role == "Copil")
        is_pet = (role == "Animal de companie")
        is_human = not is_pet

        self.member_age_label.setVisible(is_human)
        self.member_age_spinbox.setVisible(is_human)
        
        self.member_level_label.setVisible(is_child)
        self.member_level_spinbox.setVisible(is_child)
        
        self.member_pet_type_label.setVisible(is_pet)
        self.member_pet_type_edit.setVisible(is_pet)

    def add_new_family_member(self):
        """Adaugă un nou membru gol și îl selectează."""
        new_member = {"name": "Nume Nou", "role": "", "age": 0, "level": 1, "type": "", "description": ""}
        self.family_data.append(new_member)
        self._populate_family_list()
        self.family_list_widget.setCurrentRow(len(self.family_data) - 1)

    def remove_selected_family_member(self):
        """Șterge membrul selectat curent."""
        current_item = self.family_list_widget.currentItem()
        if not current_item:
            return
            
        index = current_item.data(Qt.UserRole)
        del self.family_data[index]
        self._save_family_data()
        self._populate_family_list()

    def save_family_member_details(self):
        """Salvează detaliile din formular pentru membrul selectat."""
        current_item = self.family_list_widget.currentItem()
        if not current_item:
            return
            
        index = current_item.data(Qt.UserRole)
        
        member = self.family_data[index]
        member["name"] = self.member_name_edit.text()
        member["role"] = self.member_role_combo.currentText()
        
        if member["role"] == "Animal de companie":
            member["type"] = self.member_pet_type_edit.text()
        else:
            member["age"] = self.member_age_spinbox.value()
        
        if member["role"] == "Copil":
            member["level"] = self.member_level_spinbox.value()
            
        member["description"] = self.member_description_edit.toPlainText()
        
        self._save_family_data()
        self._populate_family_list() # Reîmprospătăm lista pentru a afișa noul nume/rol
        self.family_list_widget.setCurrentRow(index)

    def _populate_learning_progress_panel(self, member):
        """
        Populează panoul de progres învățare pentru membrul dat.
        
        Args:
            member (dict): Dicționarul cu datele membrului
        """
        log_timestamp(f"📚 [LEARNING UI] Populez panoul de progres pentru '{member.get('name')}'", "app")
        
        # Golim lista de domenii
        self.domains_list_widget.clear()
        
        # Verificăm dacă membrul are learning_progress
        learning_progress = member.get("learning_progress", {})
        
        if not learning_progress:
            log_timestamp(f"📚 [LEARNING UI] Membrul '{member.get('name')}' nu are domenii de învățare.", "app")
            # Golim și panoul de detalii
            self.tier_combo.clear()
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(1)
            return
        
        # Populăm lista cu domeniile membrului
        for domain_id, progress_data in learning_progress.items():
            if domain_id not in self.available_domains:
                log_timestamp(f"⚠️ [LEARNING UI] Domeniu '{domain_id}' din progres nu mai există!", "app")
                continue
            
            domain_name = self.available_domains[domain_id]["domain_name"]
            item = QListWidgetItem(domain_name)
            item.setData(Qt.UserRole, domain_id)
            self.domains_list_widget.addItem(item)
        
        log_timestamp(f"📚 [LEARNING UI] Au fost găsite {self.domains_list_widget.count()} domenii active.", "app")

    def on_domain_selected(self, current_item, previous_item):
        """
        Apelată când un domeniu este selectat din lista de domenii a membrului.
        Actualizează combo-ul de tier-uri și bara de progres.
        """
        if not current_item:
            self.tier_combo.clear()
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(1)
            return
        
        domain_id = current_item.data(Qt.UserRole)
        
        # Obținem membrul curent
        current_list_item = self.family_list_widget.currentItem()
        if not current_list_item:
            return
        
        member_index = current_list_item.data(Qt.UserRole)
        member = self.family_data[member_index]
        
        # Obținem datele despre domeniu
        if domain_id not in self.available_domains:
            log_timestamp(f"⚠️ [LEARNING UI] Domeniu '{domain_id}' nu mai există în sistem!", "app")
            return
        
        domain_data = self.available_domains[domain_id]
        member_progress = member.get("learning_progress", {}).get(domain_id, {})
        
        # Populăm combo-ul cu tier-urile
        self.tier_combo.blockSignals(True)  # Blocăm semnalele temporar
        self.tier_combo.clear()
        
        for tier_info in domain_data["tiers"]:
            tier_id = tier_info["tier_id"]
            tier_name = tier_info["tier_name"]
            self.tier_combo.addItem(tier_name, tier_id)
        
        # Setăm tier-ul curent al membrului
        current_tier_id = member_progress.get("current_tier", "")
        if current_tier_id:
            index = self.tier_combo.findData(current_tier_id)
            if index >= 0:
                self.tier_combo.setCurrentIndex(index)
        
        self.tier_combo.blockSignals(False)  # Reactivăm semnalele
        
        # Actualizăm bara de progres
        self._update_progress_bar_for_domain(domain_id, member)
        
        log_timestamp(f"📚 [LEARNING UI] Domeniu selectat: '{domain_data['domain_name']}', Tier curent: '{current_tier_id}'", "app")

    def _update_progress_bar_for_domain(self, domain_id, member):
        """
        Actualizează bara de progres pentru domeniul și membrul specificat.
        
        Args:
            domain_id (str): ID-ul domeniului
            member (dict): Dicționarul cu datele membrului
        """
        if domain_id not in self.available_domains:
            return
        
        domain_data = self.available_domains[domain_id]
        member_progress = member.get("learning_progress", {}).get(domain_id, {})
        
        current_tier_id = member_progress.get("current_tier", "")
        completed_questions = member_progress.get("completed_questions", [])
        
        # Găsim tier-ul curent în datele domeniului
        current_tier_data = None
        for tier_info in domain_data["tiers"]:
            if tier_info["tier_id"] == current_tier_id:
                current_tier_data = tier_info
                break
        
        if not current_tier_data:
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(1)
            return
        
        # Calculăm progresul
        total_questions = len(current_tier_data["questions"])
        completed_count = len(completed_questions)
        
        self.progress_bar.setMaximum(total_questions)
        self.progress_bar.setValue(completed_count)
        
        log_timestamp(f"📊 [LEARNING UI] Progres: {completed_count}/{total_questions} întrebări completate", "app")

    def add_domain_to_member(self):
        """
        Deschide un dialog pentru a adăuga un nou domeniu de învățare pentru membrul selectat.
        """
        from PySide6.QtWidgets import QInputDialog
        
        current_item = self.family_list_widget.currentItem()
        if not current_item:
            return
        
        member_index = current_item.data(Qt.UserRole)
        member = self.family_data[member_index]
        
        # Verificăm ce domenii NU sunt deja adăugate
        existing_domains = set(member.get("learning_progress", {}).keys())
        available_to_add = []
        
        for domain_id, domain_data in self.available_domains.items():
            if domain_id not in existing_domains:
                available_to_add.append((domain_data["domain_name"], domain_id))
        
        if not available_to_add:
            log_timestamp("⚠️ [LEARNING UI] Nu există domenii noi de adăugat!", "app")
            return
        
        # Afișăm dialogul
        domain_names = [name for name, _ in available_to_add]
        selected_name, ok = QInputDialog.getItem(
            self,
            "Adaugă Domeniu de Învățare",
            f"Selectează un domeniu pentru {member.get('name')}:",
            domain_names,
            0,
            False
        )
        
        if not ok or not selected_name:
            return
        
        # Găsim domain_id-ul corespunzător
        selected_domain_id = None
        for name, domain_id in available_to_add:
            if name == selected_name:
                selected_domain_id = domain_id
                break
        
        if not selected_domain_id:
            return
        
        # Adăugăm domeniul în progresul membrului
        if "learning_progress" not in member:
            member["learning_progress"] = {}
        
        # Inițializăm cu primul tier
        domain_data = self.available_domains[selected_domain_id]
        first_tier_id = domain_data["tiers"][0]["tier_id"]
        
        member["learning_progress"][selected_domain_id] = {
            "current_tier": first_tier_id,
            "completed_questions": []
        }
        
        self._save_family_data()
        self._populate_learning_progress_panel(member)
        
        log_timestamp(f"✅ [LEARNING UI] Domeniu '{selected_name}' adăugat pentru '{member.get('name')}'", "app")

    def remove_domain_from_member(self):
        """
        Șterge domeniul selectat din progresul membrului.
        """
        current_domain_item = self.domains_list_widget.currentItem()
        if not current_domain_item:
            return
        
        current_member_item = self.family_list_widget.currentItem()
        if not current_member_item:
            return
        
        member_index = current_member_item.data(Qt.UserRole)
        member = self.family_data[member_index]
        
        domain_id = current_domain_item.data(Qt.UserRole)
        
        # Ștergem domeniul
        if "learning_progress" in member and domain_id in member["learning_progress"]:
            del member["learning_progress"][domain_id]
            self._save_family_data()
            self._populate_learning_progress_panel(member)
            
            log_timestamp(f"🗑️ [LEARNING UI] Domeniu '{domain_id}' șters pentru '{member.get('name')}'", "app")

    def on_tier_changed_manually(self, tier_name):
        """
        Apelată când utilizatorul schimbă manual tier-ul din combo box.
        Resetează automat progresul la noul tier.
        """
        if not tier_name:
            return
        
        # Verificăm dacă e o schimbare reală (nu doar populare UI)
        current_domain_item = self.domains_list_widget.currentItem()
        if not current_domain_item:
            return
        
        current_member_item = self.family_list_widget.currentItem()
        if not current_member_item:
            return
        
        domain_id = current_domain_item.data(Qt.UserRole)
        member_index = current_member_item.data(Qt.UserRole)
        member = self.family_data[member_index]
        
        new_tier_id = self.tier_combo.currentData()
        if not new_tier_id:
            return
        
        # Verificăm dacă e diferit de tier-ul curent
        current_tier_id = member.get("learning_progress", {}).get(domain_id, {}).get("current_tier", "")
        
        if new_tier_id == current_tier_id:
            return  # Nu e o schimbare reală
        
        # Actualizăm tier-ul și resetăm progresul
        if "learning_progress" not in member:
            member["learning_progress"] = {}
        
        if domain_id not in member["learning_progress"]:
            member["learning_progress"][domain_id] = {}
        
        member["learning_progress"][domain_id]["current_tier"] = new_tier_id
        member["learning_progress"][domain_id]["completed_questions"] = []
        
        self._save_family_data()
        self._update_progress_bar_for_domain(domain_id, member)
        
        log_timestamp(f"🔄 [LEARNING UI] Tier schimbat manual la '{tier_name}' pentru '{member.get('name')}'", "app")

    def reset_current_tier(self):
        """
        Resetează progresul pentru tier-ul curent al membrului.
        """
        current_domain_item = self.domains_list_widget.currentItem()
        if not current_domain_item:
            return
        
        current_member_item = self.family_list_widget.currentItem()
        if not current_member_item:
            return
        
        domain_id = current_domain_item.data(Qt.UserRole)
        member_index = current_member_item.data(Qt.UserRole)
        member = self.family_data[member_index]
        
        # Resetăm completed_questions
        if "learning_progress" in member and domain_id in member["learning_progress"]:
            member["learning_progress"][domain_id]["completed_questions"] = []
            self._save_family_data()
            self._update_progress_bar_for_domain(domain_id, member)
            
            log_timestamp(f"🔄 [LEARNING UI] Tier resetat pentru '{member.get('name')}' în domeniul '{domain_id}'", "app")

    def _generate_family_briefing(self):
        """Construiește un bloc de text cu informații despre familie pentru prompt-ul AI."""
        if not hasattr(self, 'family_data') or not self.family_data:
            # --- LOG NOU ---
            log_timestamp("ℹ️ [PROMPT GEN] Nu există date despre familie (self.family_data este goală). Nu se adaugă briefing.", "memory")
            return ""

        # --- LOG NOU ---
        log_timestamp("✅ [PROMPT GEN] Se generează briefing-ul pentru familie. Se vor trimite datele la AI.", "memory")

        briefing = "\n\n--- CUNOȘTINȚE DESPRE FAMILIA UTILIZATORULUI ---\n"
        briefing += "Aceasta este familia cu care vorbești. Sarcina ta este să recunoști acești membri în imagine și să interacționezi cu ei folosind informațiile de mai jos.\n\n"
        briefing += "**Membri Cunoscuți:**\n\n"

        for i, member in enumerate(self.family_data):
            name = member.get("name", "N/A")
            role = member.get("role", "N/A")
            description = member.get("description", "fără descriere")
            
            briefing += f"{i+1}. **Nume: {name}**\n"
            briefing += f"   - **Rol:** {role}\n"
            
            if role == "Copil":
                age = member.get("age", "N/A")
                level = member.get("level", "N/A")
                briefing += f"   - **Vârstă:** {age} ani\n"
                briefing += f"   - **Nivel:** {level}\n"
            elif role == "Animal de companie":
                pet_type = member.get("type", "N/A")
                briefing += f"   - **Tip:** {pet_type}\n"
            else: # Adult
                age = member.get("age", "N/A")
                briefing += f"   - **Vârstă:** {age} ani\n"
                
            briefing += f"   - **Descriere (indicii vizuale):** {description}\n\n"

        briefing += "**REGULI DE INTERACȚIUNE CU FAMILIA:**\n"
        briefing += "- Când vezi pe cineva pentru prima dată în conversație, salută-l pe nume.\n"
        briefing += "- Folosește semnele distinctive pentru a-i deosebi. Dacă vezi un bărbat cu ochelari, este probabil cel descris ca având ochelari.\n"
        briefing += "--- SFÂRȘIT CUNOȘTINȚE FAMILIE ---\n"
        
        return briefing

    def exit_teacher_mode(self):
        """
        Ieșire din Modul Profesor. Teleportează la pajiște după confirmarea vocală.
        """
        log_timestamp("🛑 [LEARNING] Ieșire din Modul Profesor solicitată.", "app")
        
        if not self.teacher_mode_active:
            log_timestamp("⚠️ [LEARNING] Nu suntem în Modul Profesor. Ignorăm comanda.", "app")
            return
        
        # Cleanup thread dacă există (folosind o abordare non-blocantă, dacă e posibil)
        if self.learning_thread and self.learning_thread.isRunning():
            log_timestamp("🧹 [LEARNING] Thread de învățare încă activ. Se oprește...", "cleanup")
            self.learning_thread.quit()
        
        # Resetare variabile de stare
        self.teacher_mode_active = False
        self.pending_first_question = False
        student_name_for_farewell = self.current_student_name or "prietene"
        self.current_student_name = None
        self.current_domain_id = None
        self.current_tier_id = None
        self.current_curriculum = None
        self.current_tier_data = None
        self.session_failed_questions = []
        self.current_question_id = None
        self.current_question_attempt = 0
        
        # Ascundere buton și curățare tablă
        self.exit_teacher_button.setVisible(False)
        self._clear_blackboard()
        
        # Mesaj de confirmare care anunță teleportarea
        confirmation_text = f"[EMOTION:happy] O treabă excelentă, {student_name_for_farewell}! Acum hai să luăm o pauză binemeritată pe pajiște!"
        log_timestamp(f"🎓 [LEARNING] Ieșire completă din Modul Profesor. Mesaj: '{confirmation_text}'", "app")
        
        # Programăm teleportarea la pajiște DUPĂ ce Cucuvel termină de rostit mesajul,
        # folosind slot-ul de finalizare.
        QTimer.singleShot(100, lambda: self._start_tts(confirmation_text, on_finish_slot=self._teleport_to_meadow))


    def start_learning_session(self, student_name, domain_id):
        """
        Inițiază o sesiune de învățare pentru un student și un domeniu specific.
        Include teleportarea automată la școală.
        """
        log_timestamp(f"🎓 [LEARNING] Inițiere sesiune pentru '{student_name}' în domeniul '{domain_id}'", "app")
        
        # Verificări
        if domain_id not in self.available_domains:
            error_msg = f"[EMOTION:confuz] Hmm, nu găsesc domeniul '{domain_id}'. Poate nu l-ai configurat încă?"
            log_timestamp(f"❌ [LEARNING] Domeniu inexistent: '{domain_id}'", "app")
            QTimer.singleShot(100, lambda: self._start_tts(error_msg))
            return
        
        student_member = next((m for m in self.family_data if m.get("name", "").lower() == student_name.lower()), None)
        if not student_member:
            error_msg = f"[EMOTION:confuz] Nu te găsesc în lista mea. Cum te cheamă?"
            log_timestamp(f"❌ [LEARNING] Student '{student_name}' nu găsit în family.json", "app")
            QTimer.singleShot(100, lambda: self._start_tts(error_msg))
            return
        
        # Logica de teleportare
        if self.scene_manager.current_scene_id != "scoala":
            self.scene_before_lesson = self.scene_manager.current_scene_id
            log_timestamp(f"✈️ [TELEPORT] Teleportare la școală din '{self.scene_before_lesson}'...", "app")
            self._execute_travel_with_characters("scoala", ["cucuvel_owl"])
        else:
            self.scene_before_lesson = "scoala"

        # Verificăm și inițializăm progresul
        learning_progress = student_member.get("learning_progress", {})
        if domain_id not in learning_progress:
            if "learning_progress" not in student_member: student_member["learning_progress"] = {}
            first_tier_id = self.available_domains[domain_id]["tiers"][0]["tier_id"]
            student_member["learning_progress"][domain_id] = {"current_tier": first_tier_id, "completed_questions": []}
            self._save_family_data()
        
        # Setăm variabilele de stare
        self.teacher_mode_active = True
        self.current_student_name = student_name
        self.current_domain_id = domain_id
        self.current_tier_id = student_member["learning_progress"][domain_id]["current_tier"]
        self.current_curriculum = self.available_domains[domain_id]
        self.session_failed_questions = []
        
        # --- BLOCUL CRUCIAL DE ADAUGAT/CORECTAT ---
        # Găsim și stocăm datele specifice tier-ului curent
        self.current_tier_data = next((t for t in self.current_curriculum.get("tiers", []) if t.get("tier_id") == self.current_tier_id), None)
        if not self.current_tier_data:
            log_timestamp(f"❌ [LEARNING] Nu am putut găsi datele pentru tier-ul '{self.current_tier_id}'! Se anulează lecția.", "app")
            self.exit_teacher_mode()
            return
        # --- SFÂRȘIT BLOC ---
            
        self.exit_teacher_button.setVisible(True)
        
        tier_name = self.current_tier_data.get("tier_name", "acest nivel")
        welcome_msg = f"[EMOTION:happy] Salut, {student_name}! Bine ai venit la {tier_name}. Hai să începem!"

        self.pending_first_question = True

        QTimer.singleShot(1000, lambda: self._start_tts(welcome_msg))


    def _ask_next_question(self):
        """
        Selectează, AFIȘEAZĂ (dacă e cazul) și pune următoarea întrebare nerezolvată.
        """
        log_timestamp("❓ [LEARNING] Se caută următoarea întrebare...", "app")
        
        if not self.teacher_mode_active:
            log_timestamp("⚠️ [LEARNING] Nu suntem în Modul Profesor. Anulare.", "app")
            return
        
        student_member = next((m for m in self.family_data if m.get("name") == self.current_student_name), None)
        if not student_member:
            log_timestamp("❌ [LEARNING] Studentul nu a fost găsit în family.json! Se oprește lecția.", "app")
            self.exit_teacher_mode()
            return

        progress_data = student_member.get("learning_progress", {}).get(self.current_domain_id, {})
        completed_questions = progress_data.get("completed_questions", [])
        
        # ⭐ LOG DE DEPANARE #1: Verificăm ce date avem înainte de a căuta
        log_timestamp(f"🕵️ [DEBUG] Date pentru căutare: Student='{self.current_student_name}', Tier='{self.current_tier_id}', Întrebări completate='{completed_questions}'", "app")
        
        questions_in_tier = self.current_tier_data.get("questions", [])
        
        # ⭐ LOG DE DEPANARE #2: Verificăm dacă avem întrebări în tier-ul curent
        log_timestamp(f"🕵️ [DEBUG] Total întrebări găsite în self.current_tier_data: {len(questions_in_tier)}", "app")

        next_question = None
        for q in questions_in_tier:
            q_id = q.get("id")
            if q_id and q_id not in completed_questions and q_id not in self.session_failed_questions:
                next_question = q
                break
        
        # ⭐ LOG DE DEPANARE #3: Verificăm rezultatul căutării
        if next_question:
            log_timestamp(f"🕵️ [DEBUG] REZULTAT: Am găsit următoarea întrebare: ID='{next_question.get('id')}'", "app")
        else:
            log_timestamp(f"🕵️ [DEBUG] REZULTAT: NU am găsit nicio întrebare validă de pus.", "app")

        if not next_question:
            log_timestamp("✅ [LEARNING] Toate întrebările din acest tier au fost abordate! Se finalizează.", "app")
            self._handle_tier_completion()
            return
            
        self.current_question_id = next_question["id"]
        self.current_question_attempt = 0
        
        if "display" in next_question and next_question["display"]:
            log_timestamp(f"칠판 [BLACKBOARD] Afișez: '{next_question['display']}'", "app")
            self._display_on_blackboard(next_question["display"])
        else:
            self._clear_blackboard()

        question_text = f"[EMOTION:curious] {next_question['text']}"
        log_timestamp(f"❓ [LEARNING] Se pune întrebarea: ID={self.current_question_id}", "app")
        
        QTimer.singleShot(150, lambda: self._start_tts(question_text))

    def _handle_tier_completion(self):
        """
        Gestionează finalizarea unui tier. Întreabă studentul dacă vrea să continue.
        """
        log_timestamp("🏆 [LEARNING] Tier completat!", "app")
        
        # Verificăm dacă mai există un tier următor
        current_tier_index = None
        for i, tier_info in enumerate(self.current_curriculum["tiers"]):
            if tier_info["tier_id"] == self.current_tier_id:
                current_tier_index = i
                break
        
        if current_tier_index is None:
            log_timestamp("❌ [LEARNING] Nu am găsit tier-ul curent în curriculum!", "app")
            self.exit_teacher_mode()
            return
        
        has_next_tier = (current_tier_index + 1) < len(self.current_curriculum["tiers"])
        
        if has_next_tier:
            next_tier = self.current_curriculum["tiers"][current_tier_index + 1]
            completion_msg = f"[EMOTION:proud] Bravo, {self.current_student_name}! Ai terminat acest nivel! Vrei să continui cu următorul nivel: '{next_tier['tier_name']}', sau preferi să faci o pauză?"
        else:
            completion_msg = f"[EMOTION:proud] Felicitări, {self.current_student_name}! Ai terminat toate nivelurile din acest domeniu! Ești grozav!"
        
        log_timestamp(f"🏆 [LEARNING] Mesaj finalizare: '{completion_msg}'", "app")
        
        # Aici AI-ul va aștepta răspunsul elevului (continuare sau pauză)
        # Setăm un flag special pentru a știi că așteptăm decizia de continuare
        self.waiting_for_tier_decision = True
        self.next_tier_available = has_next_tier
        if has_next_tier:
            self.pending_next_tier_id = next_tier["tier_id"]
        
        QTimer.singleShot(100, lambda: self._start_tts(completion_msg))

    def _advance_to_next_tier(self):
        """
        Avansează studentul la următorul tier și resetează progresul.
        """
        log_timestamp("⬆️ [LEARNING] Avansare la tier următor...", "app")
        
        # Găsim studentul
        student_member = None
        student_index = None
        for i, member in enumerate(self.family_data):
            if member.get("name", "") == self.current_student_name:
                student_member = member
                student_index = i
                break
        
        if not student_member:
            log_timestamp("❌ [LEARNING] Student dispărut!", "app")
            self.exit_teacher_mode()
            return
        
        # Actualizăm tier-ul curent și resetăm completed_questions
        new_tier_id = self.pending_next_tier_id
        student_member["learning_progress"][self.current_domain_id]["current_tier"] = new_tier_id
        student_member["learning_progress"][self.current_domain_id]["completed_questions"] = []
        
        self._save_family_data()
        
        # Actualizăm variabilele de stare
        self.current_tier_id = new_tier_id
        self.session_failed_questions = []
        
        # Găsim noul tier_data
        for tier_info in self.current_curriculum["tiers"]:
            if tier_info["tier_id"] == new_tier_id:
                self.current_tier_data = tier_info
                break
        
        self.waiting_for_tier_decision = False
        
        log_timestamp(f"✅ [LEARNING] Avans la tier '{new_tier_id}' efectuat!", "app")
        
        # Mesaj de confirmare și prima întrebare
        transition_msg = f"[EMOTION:happy] Perfect! Începem cu {self.current_tier_data['tier_name']}!"

        # ⭐ Setăm flag pentru prima întrebare din noul tier
        self.pending_first_question = True
        log_timestamp("⏳ [LEARNING] Prima întrebare din noul tier va fi pusă după TTS", "app")

        QTimer.singleShot(100, lambda: self._start_tts(transition_msg))

    def _process_student_answer(self, answer_text):
        """
        Procesează răspunsul unui student în Modul Profesor.
        Construiește mega-prompt-ul și trimite la LearningSessionWorker.
        
        Args:
            answer_text (str): Răspunsul dat de student
        """
        log_timestamp(f"🎓 [LEARNING] Procesez răspuns: '{answer_text}'", "app")
        
        # Verificăm dacă așteptăm decizia de continuare tier
        if hasattr(self, 'waiting_for_tier_decision') and self.waiting_for_tier_decision:
            log_timestamp("🎓 [LEARNING] Așteptăm decizie de continuare tier", "app")
            
            # Analizăm răspunsul pentru DA/NU
            answer_lower = answer_text.lower()
            
            if any(word in answer_lower for word in ["da", "yes", "continuă", "continua", "hai", "vreau"]):
                log_timestamp("✅ [LEARNING] Student vrea să continue", "app")
                if self.next_tier_available:
                    self._advance_to_next_tier()
                else:
                    completion_msg = "[EMOTION:happy] Perfect! Dar ai terminat deja toate nivelurile!"
                    QTimer.singleShot(100, lambda: self._start_tts(completion_msg))
                    self.exit_teacher_mode()
                return
            
            elif any(word in answer_lower for word in ["nu", "no", "pauză", "pauza", "stop", "oprește", "opreste"]):
                log_timestamp("🛑 [LEARNING] Student vrea pauză", "app")
                pause_msg = "[EMOTION:neutral] Perfect! Ne oprim aici. Poți reveni oricând!"
                QTimer.singleShot(100, lambda: self._start_tts(pause_msg))
                QTimer.singleShot(3000, self.exit_teacher_mode)
                return
            
            else:
                # Răspuns ambiguu - repetăm întrebarea
                clarify_msg = "[EMOTION:curious] Nu am înțeles. Vrei să continui sau preferi o pauză?"
                QTimer.singleShot(100, lambda: self._start_tts(clarify_msg))
                return
        
        # Procesare normală - evaluare răspuns la întrebare
        mega_prompt = self._build_mega_prompt(answer_text)
        
        # Curățare thread-uri vechi
        if self.learning_thread:
            try:
                if self.learning_thread.isRunning():
                    log_timestamp("🧹 [LEARNING] Opresc thread vechi de învățare...", "cleanup")
                    self.learning_worker = None
                    self.learning_thread.quit()
                    self.learning_thread.wait(2000)
            except RuntimeError:
                # Thread-ul a fost deja șters
                log_timestamp("🧹 [LEARNING] Thread deja șters - continuăm", "cleanup")
                pass
            finally:
                self.learning_thread = None
                self.learning_worker = None
        
        # Creare worker și thread nou
        log_timestamp("🎓 [LEARNING] Creez LearningSessionWorker...", "app")
        
        self.learning_worker = LearningSessionWorker(mega_prompt)
        self.learning_thread = QThread()
        
        self.learning_worker.moveToThread(self.learning_thread)
        
        # Conectare semnale
        self.learning_thread.started.connect(self.learning_worker.run)
        self.learning_worker.response_ready.connect(self._handle_learning_response)
        self.learning_worker.error_occurred.connect(self._handle_learning_error)
        self.learning_worker.finished.connect(self.learning_thread.quit)
        self.learning_worker.finished.connect(self.learning_worker.deleteLater)
        self.learning_thread.finished.connect(self.learning_thread.deleteLater)
        
        # Pornire thread
        self.learning_thread.start()
        log_timestamp("🎓 [LEARNING] Thread de învățare pornit", "app")

    def _build_mega_prompt(self, student_answer):
        """
        Construiește mega-prompt-ul complet pentru AI în Modul Profesor.
        Acum OPTIMIZAT - nu mai include toate întrebările, doar instrucțiunile din tier_X.txt.
        
        Args:
            student_answer (str): Răspunsul dat de student
            
        Returns:
            str: Prompt-ul complet
        """
        log_timestamp("📝 [LEARNING] Construiesc mega-prompt...", "app")
        
        # Încărcăm personality de bază
        try:
            with open("personality.txt", "r", encoding="utf-8") as f:
                base_personality = f.read()
        except:
            base_personality = "Ești Profesorul Cucuvel, o bufniță înțeleaptă."
        
        # Încărcăm prompt-ul specific pentru tier (DOAR PARTEA PEDAGOGICĂ, fără întrebări)
        tier_prompt_path = Path(f"curriculum/{self.current_domain_id}/prompts/{self.current_tier_id}.txt")
        tier_instructions = ""
        try:
            with open(tier_prompt_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # ⭐ Extragem DOAR partea pedagogică (până la === ÎNTREBĂRI ===)
            if "=== ÎNTREBĂRI ===" in content:
                tier_instructions = content.split("=== ÎNTREBĂRI ===")[0].strip()
            else:
                tier_instructions = content.strip()
        except:
            tier_instructions = "Fii un profesor blând și încurajator."
        
        # Găsim studentul și progresul său
        student_member = None
        for member in self.family_data:
            learning_progress = member.get("learning_progress", {})
            if learning_progress:  # Are cel puțin un domeniu configurat
                student_member = member
                break
        
        if not student_member:
            log_timestamp("❌ [LEARNING] Student dispărut din family.json!", "app")
            return ""
        
        completed_questions = student_member["learning_progress"][self.current_domain_id]["completed_questions"]
        
        # Găsim întrebarea curentă
        current_question_obj = None
        for q in self.current_tier_data["questions"]:
            if q["id"] == self.current_question_id:
                current_question_obj = q
                break
        
        if not current_question_obj:
            log_timestamp("❌ [LEARNING] Întrebare curentă nu găsită!", "app")
            return ""
        
        # Construim mega-prompt-ul (FĂRĂ lista tuturor întrebărilor)
        prompt = f"""
    {base_personality}

    === CONTEXT: MODUL PROFESOR ACTIV ===
    În acest moment, tu (Profesorul Cucuvel) ești în MODUL PROFESOR și predai unui elev.

    **INFORMAȚII DESPRE ELEV:**
    - Nume: {self.current_student_name}
    - Vârstă: {student_member.get('age', 'N/A')} ani
    - Nivel: {student_member.get('level', 'N/A')}

    **INFORMAȚII DESPRE LECȚIE:**
    - Domeniu: {self.current_curriculum['domain_name']}
    - Tier curent: {self.current_tier_data['tier_name']}
    - Descriere tier: {self.current_tier_data.get('description', '')}

    === INSTRUCȚIUNI PEDAGOGICE ===
    {tier_instructions}

    === PROGRES ELEV ÎN ACEST TIER ===
    Total întrebări în tier: {len(self.current_tier_data['questions'])}
    Întrebări completate: {len(completed_questions)}
    Întrebări greșite în sesiunea curentă (skip-uite): {len(self.session_failed_questions)}

    === ÎNTREBAREA CURENTĂ ===
    ID întrebare: {self.current_question_id}
    Text întrebare: {current_question_obj['text']}
    Răspunsuri corecte acceptate: {', '.join(current_question_obj['correct_answers'])}
    Încercarea curentă a elevului la această întrebare: {self.current_question_attempt + 1}

    === RĂSPUNSUL ELEVULUI ===
    Elevul a răspuns: "{student_answer}"

    === SARCINA TA ===
    Analizează răspunsul elevului și decide outcome-ul pentru încercarea curentă. Urmează EXACT logica de mai jos:

    1. **Verifică corectitudinea răspunsului:**
       - Compară răspunsul elevului cu lista de răspunsuri corecte
       - Fii flexibil la variații (majuscule/minuscule, diacritice, plural/singular)
       - Acceptă sinonime apropiate

    2. **Aplică logica pedagogică:**
       
       **DACĂ răspunsul este CORECT:**
       - Outcome: "correct"
       - Laudă elevul cu entuziasm (folosind numele lui!)
       - Treci la următoarea întrebare nerezolvată
       - Dacă era ultima întrebare → outcome: "tier_finished"
       
       **DACĂ răspunsul este GREȘIT (prima încercare la această întrebare):**
       - Outcome: "incorrect_retry"
       - Încurajează elevul cu blândețe
       - Repetă întrebarea (același text, poate cu un mic indiciu)
       - NU da răspunsul corect
       
       **DACĂ răspunsul este GREȘIT (a doua încercare la această întrebare):**
       - Outcome: "incorrect_skip"
       - Fii empatic și spune că veți reveni la întrebare mai târziu
       - Treci la următoarea întrebare nerezolvată
       - NU da răspunsul corect

    3. **Returnează DOAR un obiect JSON cu următoarea structură:**

    {{
      "outcome": "correct" | "incorrect_retry" | "incorrect_skip" | "tier_finished",
      "text_to_speak": "Textul complet pe care Cucuvel trebuie să-l rostească"
    }}

    **IMPORTANT:**
    - text_to_speak trebuie să înceapă cu [EMOTION:...] (ex: [EMOTION:happy], [EMOTION:proud], [EMOTION:attentive])
    - Răspunsul tău trebuie să fie DOAR JSON-ul de mai sus, fără niciun alt text
    - Nu include explicații sau comentarii în afara JSON-ului

    **ANALIZEAZĂ ACUM ȘI RĂSPUNDE CU JSON-UL:**
    """
        
        log_timestamp(f"📝 [LEARNING] Mega-prompt construit ({len(prompt)} caractere)", "app")
        return prompt

    def _handle_learning_response(self, response_dict):
        """
        Procesează răspunsul AI-ului din LearningSessionWorker.
        
        Args:
            response_dict (dict): Dicționarul cu outcome și text_to_speak
        """
        log_timestamp(f"🎓 [LEARNING] Răspuns primit: {response_dict}", "app")
        
        outcome = response_dict.get("outcome")
        text_to_speak = response_dict.get("text_to_speak", "")
        
        if not text_to_speak:
            log_timestamp("❌ [LEARNING] Răspuns fără text! Ignorat.", "app")
            return
        
        # Rostim feedback-ul (ACEST RĂMÂNE - e pentru toate outcome-urile)
        QTimer.singleShot(100, lambda: self._start_tts(text_to_speak))
        
        # Procesăm outcome-ul
        if outcome == "correct":
            log_timestamp("✅ [LEARNING] Răspuns corect!", "app")
            
            # Marcăm întrebarea ca rezolvată
            student_member = None
            for member in self.family_data:
                if member.get("name") == self.current_student_name:
                    student_member = member
                    break
            
            if student_member:
                if self.current_question_id not in student_member["learning_progress"][self.current_domain_id]["completed_questions"]:
                    student_member["learning_progress"][self.current_domain_id]["completed_questions"].append(self.current_question_id)
                    self._save_family_data()
                    log_timestamp(f"💾 [LEARNING] Întrebare {self.current_question_id} salvată ca rezolvată", "app")
            
            # Resetăm attempt counter
            self.current_question_attempt = 0
            
            # ⭐ Setăm flag pentru următoarea întrebare
            self.pending_next_question = True
            log_timestamp("⏳ [LEARNING] Următoarea întrebare va fi pusă după feedback", "app")
            
            # ❌ ȘTERGE ACEASTĂ LINIE (e duplicat):
            # QTimer.singleShot(100, lambda: self._start_tts(text_to_speak))
        
        elif outcome == "incorrect_retry":
            log_timestamp("⚠️ [LEARNING] Răspuns greșit - prima încercare", "app")
            
            # Incrementăm attempt counter
            self.current_question_attempt += 1
            
            # Întrebarea va fi repetată automat prin text_to_speak
            # Nu facem nimic - așteptăm următorul răspuns
        
        elif outcome == "incorrect_skip":
            log_timestamp("❌ [LEARNING] Răspuns greșit - a doua încercare. Skip.", "app")
            
            # Adăugăm în session_failed_questions
            if self.current_question_id not in self.session_failed_questions:
                self.session_failed_questions.append(self.current_question_id)
            
            # Resetăm attempt counter
            self.current_question_attempt = 0
            
            # ⭐ Setăm flag pentru următoarea întrebare
            self.pending_next_question = True
            log_timestamp("⏳ [LEARNING] Următoarea întrebare va fi pusă după feedback", "app")
            
            # ❌ ȘTERGE ACEASTĂ LINIE (e duplicat):
            # QTimer.singleShot(100, lambda: self._start_tts(text_to_speak))
        
        elif outcome == "tier_finished":
            log_timestamp("🏆 [LEARNING] Tier completat!", "app")
            
            # Marcăm ultima întrebare ca rezolvată (dacă nu e deja)
            student_member = None
            for member in self.family_data:
                if member.get("name") == self.current_student_name:
                    student_member = member
                    break
            
            if student_member:
                if self.current_question_id not in student_member["learning_progress"][self.current_domain_id]["completed_questions"]:
                    student_member["learning_progress"][self.current_domain_id]["completed_questions"].append(self.current_question_id)
                    self._save_family_data()
            
            # Gestionăm finalizarea după 3 secunde
            QTimer.singleShot(3000, self._handle_tier_completion)



    def _handle_learning_error(self, error_message):
        """
        Gestionează erorile din LearningSessionWorker.
        
        Args:
            error_message (str): Mesajul de eroare
        """
        log_timestamp(f"❌ [LEARNING] Eroare în worker: {error_message}", "app")
        
        error_msg = "[EMOTION:confuz] Hmm, am avut o problemă tehnică. Hai să încercăm din nou!"
        QTimer.singleShot(100, lambda: self._start_tts(error_msg))

    def _start_tts(self, text, on_finish_slot=None):
        """
        Metodă simplificată pentru a porni TTS în contextul învățării și nu numai.
        Gestionează extragerea emoției și un callback opțional la finalizare.
        
        Args:
            text (str): Textul de rostit (poate include [EMOTION:...] la început)
            on_finish_slot (function, optional): O funcție de apelat după ce TTS-ul se termină.
        """
        log_timestamp(f"🔊 [TTS SIMPLE] Pornesc TTS pentru: '{text[:50]}...'", "tts")
        
        # Oprim orice TTS anterior, dacă rulează, pentru a preveni suprapunerile
        if self.tts_thread and self.tts_thread.isRunning():
            log_timestamp("⚠️ [TTS] Un TTS anterior încă rula. Se oprește forțat.", "tts")
            self.tts_thread.quit()
        
        # MUTE microfonul ÎNAINTE de a vorbi
        if self.voice_worker:
            self.voice_worker.set_muted(True, is_ai_speaking=True)
            log_timestamp("🔇 [TTS SIMPLE] Microfon mutat pentru a preveni echo", "mute")
        
        # Setează semaforul pe ROȘU
        if self.voice_enabled:
            self._update_semafor_state('rosu')
            log_timestamp("🔴 [TTS SIMPLE] Semafor setat pe ROȘU", "semafor")
        
        # Marchează că vorbim
        self.is_speaking = True
        
        # Extragem emoția dacă există
        clean_text = self._extract_and_apply_emotion(text, self.active_speaker_id)
        
        # Obținem caracterul care vorbește
        speaking_character = self.character_manager.get_character(self.active_speaker_id)
        if not speaking_character:
            log_timestamp("❌ [TTS SIMPLE] Nu există speaker activ! Se anulează.", "tts")
            # Ne asigurăm că deblocăm starea dacă apare o eroare aici
            self.speech_finished()
            return
        
        # Salvăm textul pentru funcționalitatea "Repetă"
        self.last_character_speeches[self.active_speaker_id] = clean_text
        
        # Actualizăm subtitrările
        self.subtitle_label.setText(clean_text)
        self.subtitle_label.adjustSize()
        self.subtitle_scroll_area.show()
        self.subtitle_scroll_area.raise_()
        
        # Adăugăm la chat
        self.add_to_chat(speaking_character.display_name, clean_text)
        
        # Actualizăm textul pentru detecția de ecou
        if self.voice_worker:
            self.voice_worker.set_last_ai_text(clean_text)
        
        # Ștergem fișierul audio anterior, dacă există
        if hasattr(self, 'last_audio_file_path') and self.last_audio_file_path and os.path.exists(self.last_audio_file_path):
            try:
                os.remove(self.last_audio_file_path)
                log_timestamp(f"🧹 [CLEANUP] Fișier audio vechi șters: {self.last_audio_file_path}", "cleanup")
            except Exception as e:
                log_timestamp(f"⚠️ [CLEANUP] Eroare la ștergerea fișierului vechi: {e}", "cleanup")
        
        # Creăm și pornim noul TTS worker
        tts_worker = TTSWorker(clean_text)
        tts_worker.voice = speaking_character.voice_id
        
        # Aici folosim parametrul `on_finish_slot` pentru a decide ce se întâmplă la final
        self.start_sync_process(tts_worker, clean_text, self.active_speaker_id, on_finish_slot=on_finish_slot)
        
        log_timestamp("✅ [TTS SIMPLE] TTS pornit cu succes", "tts")

    def on_subtitle_mode_changed(self, mode):
        """Salvează noul mod de subtitrare în config."""
        self.config["subtitle_mode"] = mode.lower() # salvăm ca "original", "latin (fonetic)", "combinat"
        save_config(self.config)
        log_timestamp(f"⚙️ [CONFIG] Mod subtitrare setat la: '{mode}'")

    def stop_thinking(self):
        self.thinking_timer.stop()
        self.is_thinking = False
        # TODO: Aici vom reseta animația de gândire pentru personajul specific
        
    def animate_thinking(self):
        # TODO: Vom implementa o animație de gândire care se aplică personajului care gândește
        pass

    def on_settings_changed(self):
        """Activează butonul 'Apply' dacă setările curente diferă de cele salvate."""
        current_model = self.ai_model_combo.currentText()
        
        # Comparam modelul selectat acum cu cel salvat la pornire
        if current_model != self.initial_ai_model:
            self.apply_button.setEnabled(True)
            self.apply_button.setText("✅ Apply Changes *") # Marcaj vizual
        else:
            self.apply_button.setEnabled(False)
            self.apply_button.setText("✅ Apply Changes")

    def apply_general_settings(self):
        """Salvează noile setări și resetează starea butonului."""
        log_timestamp("⚙️ [SETTINGS] Se aplică noile setări generale...", "app")
        
        # Preluăm noua valoare din ComboBox
        new_model = self.ai_model_combo.currentText()
        
        # 1. Actualizăm dicționarul din memorie
        self.config["ai_model_name"] = new_model
        
        # 2. Salvăm noua stare ca fiind cea "inițială" pentru comparații viitoare
        self.initial_ai_model = new_model
        
        # 3. Salvăm configurația completă pe disc
        save_config(self.config)
        
        # 4. Dezactivăm butonul Apply și eliminăm marcajul vizual
        self.apply_button.setEnabled(False)
        self.apply_button.setText("✅ Apply Changes")
        
        log_timestamp(f"🧠 [CONFIG] Model AI actualizat la: '{new_model}'")
        
    def play_filler_sound(self, character):
        """Redă un sunet de umplutură specific personajului."""
        sound_file = character.get_random_filler_sound()
        if sound_file:
            log_timestamp(f"🔊 [FILLER] Se redă sunetul de umplutură pentru '{character.id}': {os.path.basename(sound_file)}", "filler")
            try:
                pygame.mixer.Channel(1).play(pygame.mixer.Sound(sound_file))
            except Exception as e:
                log_timestamp(f"❌ [FILLER] Eroare la redarea sunetului: {e}", "filler")
        else:
            log_timestamp(f"🔊 [FILLER] Personajul '{character.id}' nu are sunete de umplutură.", "filler")

    def start_sync_process(self, worker_instance, text_for_animation, speaking_character_id, on_finish_slot=None):
        log_timestamp(f"🎬 [SYNC] START sincronizare pentru '{speaking_character_id}'...", "sync")
        
        if self.tts_thread is not None or self.tts_worker is not None:
            log_timestamp("⚠️ [SYNC] Un ciclu TTS anterior nu a fost curățat corect. Se anulează.", "sync")
            self.speech_finished()
            return

        self.is_speaking = True
        self.speaking_character_id = speaking_character_id
        self.disable_all_actions()
        
        self.generate_viseme_queue_for_text(text_for_animation)
        
        self.total_viseme_count = len(self.viseme_queue)
        self.last_displayed_frame = -1
        
        self.tts_thread = QThread()
        self.tts_worker = worker_instance
        self.tts_worker.moveToThread(self.tts_thread)
        
        self.tts_worker.audio_ready.connect(self.on_audio_ready)
        self.tts_thread.started.connect(self.tts_worker.run)
        
        finish_slot = on_finish_slot if on_finish_slot else self.speech_finished
        self.tts_worker.finished.connect(finish_slot)
        
        # Restul conexiunilor pentru curățenie rămân la fel
        self.tts_worker.finished.connect(self.tts_thread.quit)
        self.tts_worker.finished.connect(self.tts_worker.deleteLater)
        self.tts_thread.finished.connect(self.tts_thread.deleteLater)
        
        self.tts_thread.start()    

    def on_audio_ready(self, audio_path, actual_duration):
        log_timestamp(f"🎬 [SYNC] ✅ AUDIO GATA! Durată: {actual_duration:.2f}s. Pornesc animația pentru '{self.speaking_character_id}'.", "sync")
        
        # --- BLOC NOU: Salvăm calea și activăm butonul ---
        self.last_audio_file_path = audio_path
        self.repeat_button.setEnabled(True)
        # --- SFÂRȘIT BLOC NOU ---
        
        self.set_speaker(self.speaking_character_id)
        
        self.estimated_speech_duration = actual_duration
        self.speech_start_time = time.time()
        self.sync_timer.start(30)

    def update_synced_animation(self):
        if not self.is_speaking or self.estimated_speech_duration <= 0:
            self.sync_timer.stop()
            return
            
        elapsed = time.time() - self.speech_start_time
        progress = min(elapsed / self.estimated_speech_duration, 1.0) # Ne asigurăm că progresul nu depășește 100%

        # --- LOGICA PENTRU VIZEME (rămâne neschimbată) ---
        target_frame = int(progress * self.total_viseme_count)
        if target_frame > self.last_displayed_frame:
            actual_frame = min(target_frame, self.total_viseme_count - 1)
            for i in range(self.last_displayed_frame + 1, actual_frame + 1):
                if i < len(self.viseme_queue):
                    vizem = self.viseme_queue[i]
                    self.set_character_viseme(self.speaking_character_id, vizem)
            self.last_displayed_frame = actual_frame
            
        # --- LOGICA NOUĂ PENTRU SCROLL SINCRONIZAT ---
        scrollbar = self.subtitle_scroll_area.verticalScrollBar()
        max_scroll_value = scrollbar.maximum()
        
        # Calculăm noua poziție a scrollbar-ului pe baza progresului audio
        if max_scroll_value > 0:
            target_scroll_value = int(progress * max_scroll_value)
            scrollbar.setValue(target_scroll_value)

    def generate_viseme_queue_for_text(self, text):
        log_timestamp(f"🎬 [VISEME SIMPLU] Generare pentru: '{text[:50]}...'", "sync")
        self.viseme_queue.clear()
        last_viseme = "Neutru"
        
        for char in text.lower():
            if char in "aeiouăâî":
                current_viseme = "A"
            else:
                current_viseme = "Neutru"
            
            if current_viseme != last_viseme:
                self.viseme_queue.append(current_viseme)
                last_viseme = current_viseme

        if not self.viseme_queue or self.viseme_queue[-1] != "Neutru":
            self.viseme_queue.append("Neutru")

        log_timestamp(f"🎬 [VISEME SIMPLU] ✅ {len(self.viseme_queue)} vizeme generate", "sync")

    def set_character_viseme(self, character_id, vizem):
        """
        Schimbă vizema gurii pentru un personaj, folosind fișierele definite
        în secțiunea 'visual_states' > 'talking' din config.json.
        """
        character = self.character_manager.get_character(character_id)
        char_layers = self.character_layers.get(character_id)
        if not char_layers or not character:
            return

        talk_config = character.components.get("visual_states", {}).get("talking")
        if not talk_config:
            return
            
        target_part_name = talk_config.get("target_part")
        target_layer = char_layers.get(target_part_name)
        if not target_layer:
            return
        
        file_to_load = talk_config.get("open_file") if vizem == "A" else talk_config.get("closed_file")
        if not file_to_load:
            return
            
        pixmap_path = os.path.join(character.assets_path, file_to_load)
        if os.path.exists(pixmap_path):
            pixmap_to_set = QPixmap(pixmap_path)
            if hasattr(target_layer, 'original_pixmap'):
                 scaled_pixmap = pixmap_to_set.scaled(target_layer.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                 target_layer.setPixmap(scaled_pixmap)
        else:
            log_timestamp(f"⚠️ [VISEME] Nu am găsit asset-ul '{file_to_load}' pentru '{character_id}'")

    def _check_and_switch_speaker(self, text):
        """
        Verifică dacă textul conține cuvinte cheie pentru a schimba vorbitorul activ.
        MODIFICAT: Răspunde dacă personajul menționat nu e pe scenă.
        Returnează True dacă s-a făcut o schimbare SAU s-a generat un răspuns, altfel False.
        """
        log_timestamp("🎤 [SPEAKER SWITCH] Se verifică dacă se schimbă vorbitorul...", "router")
        text_lower = text.lower()

        speaker_keywords = {
            "cucuvel_owl": ["cucuvel", "profesore", "domn profesor", "bufnițovici", "bufnita"],
            "rina_cat": ["rina", "nina", "irina", "pisico", "pisica"]
        }

        characters_on_stage_ids = self.character_manager.active_characters.keys()

        for char_id, keywords in speaker_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if char_id in characters_on_stage_ids:
                        # ✅ PERSONAJ PE SCENĂ - switch normal
                        if self.active_speaker_id == char_id:
                            log_timestamp(f"🎤 [SPEAKER SWITCH] Adresare către vorbitorul deja activ ('{char_id}'). Nu se schimbă nimic.", "router")
                            return False
                        else:
                            log_timestamp(f"🎤 [SPEAKER SWITCH] COMANDĂ DETECTATĂ! Trecem la '{char_id}'.", "router")
                            
                            new_speaker = self.character_manager.get_character(char_id)
                            confirmation_prompt = ""
                            if new_speaker.language.startswith("en"):
                                confirmation_prompt = "Say a short confirmation phrase, in your personality, to let the user know you are now listening. For example: 'I'm here!', 'Yes?', or 'I'm listening!'"
                            else:
                                confirmation_prompt = "Spune o frază scurtă de confirmare, în personalitatea ta, că acum asculți tu. De exemplu: 'Ascult!', 'Sunt aici!' sau 'Da, spune-mi!'."
                            
                            self.active_speaker_id = char_id
                            self.process_question(confirmation_prompt, self.active_speaker_id)
                            
                            return True
                    else:
                        # ❌ PERSONAJ ABSENT - cineva răspunde sau silence
                        log_timestamp(f"🔇 [SPEAKER SWITCH] '{char_id}' menționat dar NU e pe scenă", "router")
                        
                        # Verificăm dacă avem un speaker activ pe scenă care poate răspunde
                        if self.active_speaker_id and self.active_speaker_id in characters_on_stage_ids:
                            # CASE 7 & 10: Speaker-ul răspunde despre personaj absent
                            char_absent = self.character_manager.get_character(char_id)
                            speaker = self.character_manager.get_character(self.active_speaker_id)
                            
                            log_timestamp(f"💬 [SPEAKER] '{self.active_speaker_id}' răspunde despre '{char_id}' absent", "router")
                            
                            # Detectăm dacă e comandă send (du-te, mergi, etc.)
                            is_send_command = any(word in text_lower for word in ["du-te", "du te", "mergi", "pleacă", "pleaca"])
                            
                            if is_send_command:
                                # CASE 10: Comandă send către absent
                                if speaker.language.startswith("en"):
                                    absence_prompt = f"Tell the user politely that {char_absent.display_name} is not here, so you cannot send them anywhere."
                                else:
                                    absence_prompt = f"Spune politicos utilizatorului că {char_absent.display_name} nu e aici, deci nu îl poți trimite nicăieri."
                            else:
                                # CASE 7: Întrebare/conversație cu absent
                                if speaker.language.startswith("en"):
                                    absence_prompt = f"Tell the user politely that {char_absent.display_name} is not here right now."
                                else:
                                    absence_prompt = f"Spune politicos utilizatorului că {char_absent.display_name} nu e aici acum."
                            
                            self.process_question(absence_prompt, self.active_speaker_id)
                            return True
                        else:
                            # CASE 5: Nimeni pe scenă → SILENCE complet
                            log_timestamp(f"🔇 [SPEAKER] Nimeni pe scenă să răspundă → SILENCE", "router")
                            return False
        
        log_timestamp("🎤 [SPEAKER SWITCH] Nicio comandă de schimbare a vorbitorului detectată.", "router")
        return False

    def speech_finished(self):
        # --- MODIFICARE: Nu mai ascundem subtitrările la final ---
        # self.subtitle_scroll_area.hide()  <-- COMENTAT SAU ȘTERS
        # self.translation_scroll_area.hide() <-- COMENTAT SAU ȘTERS
        
        # Doar resetăm scrollbar-ul la poziția de start pentru data viitoare
        self.subtitle_scroll_area.verticalScrollBar().setValue(0)
        
        # --- SFÂRȘIT MODIFICARE ---
        
        # ... restul funcției rămâne neschimbat ...
        if self.voice_enabled:
            self._update_semafor_state('verde') # Revine la VERDE
        # --- SFÂRȘIT BLOC ---
        
        # ⭐ PASUL 1: Resetare flag-uri de stare
        self.is_speaking = False
        self.is_thinking = False
        log_timestamp("🔓 [STATE] Flag-uri resetate: speaking=False, thinking=False", "cleanup")
        
        # ⭐⭐⭐ CURĂȚARE TTS OBLIGATORIE (înainte de orice return)
        if self.tts_worker is not None:
            try:
                self.tts_worker.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.tts_worker = None
        
        if self.tts_thread is not None:
            try:
                self.tts_thread.quit()
                self.tts_thread.wait(500)  # Așteptare scurtă
                self.tts_thread.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.tts_thread = None
        
        # ⭐⭐⭐ ACUM verificăm pending questions
        if hasattr(self, 'pending_first_question') and self.pending_first_question:
            self.pending_first_question = False
            log_timestamp("🎓 [LEARNING] TTS bun venit terminat. Pun prima întrebare...", "app")
            QTimer.singleShot(500, self._ask_next_question)
            
            if self.voice_worker and not self.is_muted:
                self.voice_worker.set_muted(False)
            
            return  # Acum e sigur să returnăm
        
        if hasattr(self, 'pending_next_question') and self.pending_next_question:
            self.pending_next_question = False
            log_timestamp("🎓 [LEARNING] TTS feedback terminat. Pun următoarea întrebare...", "app")
            QTimer.singleShot(500, self._ask_next_question)
            
            if self.voice_worker and not self.is_muted:
                self.voice_worker.set_muted(False)
            
            return  # Acum e sigur să returnăm
        
        # ⭐ PASUL 2: UNMUTE microfonul (cu verificare inteligentă)
        if self.voice_worker:
            # Reactivăm ascultarea DOAR dacă utilizatorul NU este pe modul MUTE MANUAL
            if not self.is_muted:
                log_timestamp("🔊 [UNMUTE] Microfonul este reactivat automat.", "mute")
                self.voice_worker.set_muted(False)
            else:
                log_timestamp("🔇 [UNMUTE] Microfonul RĂMÂNE pe mute la cererea utilizatorului.", "mute")
        
        # ⭐ PASUL 2.1: Revenire la speaker original după traducere
        if self.pending_speaker_return:
            log_timestamp(f"🔄 [TRANSLATION] Revin la speaker-ul original: '{self.pending_speaker_return}'", "router")
            self.active_speaker_id = self.pending_speaker_return
            self.pending_speaker_return = None
        # ⭐ PASUL 2.3: RESET GAZE (ADAUGĂ AICI!)
        self.set_speaker(None)  # Toți privesc în față
        
        # ⭐ PASUL 2.5: EXECUTĂM MUTAREA AMÂNATĂ DACĂ EXISTĂ
        if self.pending_move_after_tts:
            move_data = self.pending_move_after_tts
            self.pending_move_after_tts = None  # Resetăm
            
            log_timestamp(f"🚀 [SEND] Execut mutarea amânată: '{move_data['char_id']}' → '{move_data['destination']}'", "router")
            
            success, error = self.character_manager.move_character_silent(
                move_data['char_id'], 
                move_data['destination']
            )
            
            if success:
                log_timestamp(f"✅ [SEND] '{move_data['char_id']}' mutat cu succes în '{move_data['destination']}'", "router")
                log_timestamp(f"📊 [SEND] Personaje active DUPĂ mutare: {list(self.character_manager.active_characters.keys())}", "router")
                
                # Dacă pleacă speaker-ul, fallback
                if move_data['char_id'] == self.active_speaker_id:
                    self._handle_speaker_left()
            else:
                log_timestamp(f"❌ [SEND] Eroare la mutare amânată: {error}", "router")
        
        # ⭐ PASUL 3: Cleanup cu protecție try-except
        log_timestamp("🧹 [CLEANUP] Se marchează pentru ștergere și se resetează referințele TTS.", "cleanup")
        
        # Curățare thread și worker Gemini
        log_timestamp("🧹 [CLEANUP] Curățare thread și worker Gemini...", "cleanup")
        if self.gemini_worker is not None:
            try:
                log_timestamp("🧹 [CLEANUP] - Marchează gemini_worker pentru ștergere", "cleanup")
                self.gemini_worker.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.gemini_worker = None
        
        if self.gemini_thread is not None:
            try:
                log_timestamp("🧹 [CLEANUP] - Oprește și marchează gemini_thread pentru ștergere", "cleanup")
                self.gemini_thread.quit()
                self.gemini_thread.wait(1000)
                self.gemini_thread.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.gemini_thread = None
        
        # Curățare thread și worker Intent
        log_timestamp("🧹 [CLEANUP] Curățare thread și worker Intent...", "cleanup")
        if self.intent_worker is not None:
            try:
                log_timestamp("🧹 [CLEANUP] - Marchează intent_worker pentru ștergere", "cleanup")
                self.intent_worker.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.intent_worker = None
        
        if self.intent_thread is not None:
            try:
                log_timestamp("🧹 [CLEANUP] - Oprește și marchează intent_thread pentru ștergere", "cleanup")
                self.intent_thread.quit()
                self.intent_thread.wait(1000)
                self.intent_thread.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.intent_thread = None
        
        # ⭐ Curățare thread și worker TTS
        log_timestamp("🧹 [CLEANUP] Curățare thread și worker TTS...", "cleanup")
        if self.tts_worker is not None:
            try:
                log_timestamp("🧹 [CLEANUP] - Marchează tts_worker pentru ștergere", "cleanup")
                self.tts_worker.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.tts_worker = None
        
        if self.tts_thread is not None:
            try:
                log_timestamp("🧹 [CLEANUP] - Oprește și marchează tts_thread pentru ștergere", "cleanup")
                self.tts_thread.quit()
                self.tts_thread.wait(1000)
                self.tts_thread.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.tts_thread = None
        
        # ⭐⭐⭐ NOU: Curățare thread și worker Learning
        log_timestamp("🧹 [CLEANUP] Curățare thread și worker Learning...", "cleanup")
        if self.learning_worker is not None:
            try:
                log_timestamp("🧹 [CLEANUP] - Marchează learning_worker pentru ștergere", "cleanup")
                self.learning_worker.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.learning_worker = None
        
        if self.learning_thread is not None:
            try:
                log_timestamp("🧹 [CLEANUP] - Oprește și marchează learning_thread pentru ștergere", "cleanup")
                self.learning_thread.quit()
                self.learning_thread.wait(1000)
                self.learning_thread.deleteLater()
            except RuntimeError:
                pass
            finally:
                self.learning_thread = None
        
        log_timestamp(f"✅ [CLEANUP] Cleanup complet finalizat!", "cleanup")
        
        # ⭐ RE-ENABLE TEXT INPUT ȘI ALTE CONTROALE
        self.enable_all_actions()
        log_timestamp(f"🔓 [UI] Controale re-activate - text input disponibil", "cleanup")

    def disable_all_actions(self):
        self.text_input.setEnabled(False)
        
    def enable_all_actions(self):
        if self.conversation_state == 'ACTIVE':
            self.text_input.setEnabled(True)
            self.text_input.setFocus()
    
    def _validate_active_speaker(self):
        """
        Verifică dacă active_speaker_id e încă valid (pe scenă).
        Dacă nu, face fallback smart.
        """
        if self.active_speaker_id is None:
            return  # E ok să fie None
        
        if self.active_speaker_id not in self.character_manager.active_characters:
            # Speaker-ul nu mai e pe scenă
            active_chars = list(self.character_manager.active_characters.keys())
            
            if len(active_chars) > 0:
                # Auto-switch la primul disponibil
                self.active_speaker_id = active_chars[0]
                log_timestamp(f"🔄 [SPEAKER] Auto-switch la '{self.active_speaker_id}' (singur disponibil)", "router")
            else:
                # Nimeni pe scenă
                self.active_speaker_id = None
                log_timestamp(f"🔇 [SPEAKER] Niciun personaj pe scenă → speaker = None", "router")

    def _handle_speaker_left(self):
        """
        Gestionează situația când speaker-ul activ pleacă din scenă.
        CASE 3: Verifică dacă mai e cineva → switch, altfel None
        """
        log_timestamp(f"👋 [SPEAKER] Speaker-ul '{self.active_speaker_id}' a plecat din scenă", "router")
        
        active_chars = list(self.character_manager.active_characters.keys())
        
        if len(active_chars) > 0:
            # Mai e cineva → switch
            self.active_speaker_id = active_chars[0]
            char = self.character_manager.get_character(self.active_speaker_id)
            log_timestamp(f"🔄 [SPEAKER] Auto-switch la '{self.active_speaker_id}' ({char.display_name})", "router")
        else:
            # Nimeni → None
            self.active_speaker_id = None
            log_timestamp(f"🔇 [SPEAKER] Nu mai e nimeni → speaker = None (doar summon/travel funcționează)", "router")

    def _handle_arrival_greeting(self):
        """
        CASE 8: Când ajungi într-o scenă cu natives, te salută automat.
        CASE 9: Dacă e goală, speaker = None.
        """
        active_chars = list(self.character_manager.active_characters.keys())
        
        log_timestamp(f"🏠 [ARRIVAL] Verificare greeting pentru {len(active_chars)} personaje...", "router")
        
        if len(active_chars) == 0:
            # Scenă goală
            self.active_speaker_id = None
            log_timestamp(f"🏜️ [ARRIVAL] Scenă goală → speaker = None", "router")
            return
        
        # Găsim primul native (care nu era cu noi)
        native_greeter = None
        current_scene = self.scene_manager.current_scene_id
        
        for char_id in active_chars:
            char = self.character_manager.get_character(char_id)
            
            # ⭐ FIX: Folosim char.home_scene, care este numele corect al atributului
            log_timestamp(f"🔍 [ARRIVAL] Verific '{char_id}': home_scene={char.home_scene}, current_scene={current_scene}", "router")
            
            # E native aici (home_scene = scena curentă)?
            if char.home_scene == current_scene:
                native_greeter = char_id
                log_timestamp(f"✅ [ARRIVAL] GĂSIT native greeter: '{char_id}'", "router")
                break
        
        if native_greeter:
            # Native găsit → salută
            self.active_speaker_id = native_greeter
            char = self.character_manager.get_character(native_greeter)
            
            log_timestamp(f"👋 [ARRIVAL] Native '{native_greeter}' te salută în '{current_scene}'", "router")
            
            if char.language.startswith("en"):
                greeting_prompt = "Greet the user warmly as they arrive in your home. Say something like 'Welcome back!' or 'Hello there!'"
            else:
                greeting_prompt = "Salută călduros utilizatorul care vine în casa ta. Spune ceva ca 'Bine ai venit înapoi!' sau 'Salut!'"
            
            self.process_question(greeting_prompt, native_greeter)
        else:
            # Nu e native, dar e cineva → switch la primul
            self.active_speaker_id = active_chars[0]
            log_timestamp(f"🔄 [ARRIVAL] Niciun native, switch la '{self.active_speaker_id}' (prim disponibil)", "router")

    def get_character_scene_position(self, char_id, scene_id):
        """
        Citește poziția unui personaj din config pentru o scenă.
        
        Returns:
            QPoint(x, y) sau None dacă nu există
        """
        char = self.character_manager.get_character(char_id)
        if not char:
            log_timestamp(f"❌ [GAZE POS] Personaj '{char_id}' nu există!", "gaze")
            return None
        
        scene_config = char.get_config_for_scene(scene_id)
        if not scene_config:
            log_timestamp(f"❌ [GAZE POS] '{char_id}' nu are config pentru '{scene_id}'!", "gaze")
            return None
        
        pos = scene_config.get("pos")
        log_timestamp(f"🔍 [GAZE POS] '{char_id}' în '{scene_id}': pos raw = {pos} (type: {type(pos)})", "gaze")
        
        # ⭐ CAZUL 1: Deja e QPoint (cel mai comun după prima inițializare)
        if isinstance(pos, QPoint):
            log_timestamp(f"✅ [GAZE POS] '{char_id}' poziție (QPoint direct): QPoint({pos.x()}, {pos.y()})", "gaze")
            return pos
        
        # ⭐ CAZUL 2: Listă din JSON
        if isinstance(pos, list) and len(pos) >= 2:
            result = QPoint(pos[0], pos[1])
            log_timestamp(f"✅ [GAZE POS] '{char_id}' poziție (convertit din listă): QPoint({result.x()}, {result.y()})", "gaze")
            return result
        
        # ⭐ CAZUL 3: Dict (backup)
        if isinstance(pos, dict):
            result = QPoint(pos.get("x", 0), pos.get("y", 0))
            log_timestamp(f"✅ [GAZE POS] '{char_id}' poziție (dict): QPoint({result.x()}, {result.y()})", "gaze")
            return result
        
        log_timestamp(f"❌ [GAZE POS] '{char_id}' format poziție necunoscut: {pos}", "gaze")
        return None

    def calculate_gaze_direction(self, observer_id, speaker_id, scene_id):
        """
        Calculează direcția privirii pentru un observator.
        
        Args:
            observer_id: ID-ul personajului care observă
            speaker_id: "user" sau character_id care vorbește
            scene_id: Scena curentă
        
        Returns:
            "stanga" | "centru" | "dreapta"  ⭐ NOTĂ: schimbat de la "left/center/right"
        """
        log_timestamp(f"🔍 [GAZE CALC] observer='{observer_id}', speaker='{speaker_id}', scene='{scene_id}'", "gaze")
        
        # REGULA 1: Dacă user-ul vorbește → toți în față
        if speaker_id == "user":
            log_timestamp(f"  → REGULA 1: User vorbește → 'centru'", "gaze")
            return "centru"
        
        # REGULA 2: Dacă te uiți la tine însuți → în față
        if observer_id == speaker_id:
            log_timestamp(f"  → REGULA 2: Se uită la el însuși → 'centru'", "gaze")
            return "centru"
        
        # REGULA 3: Calculăm poziția relativă
        observer_pos = self.get_character_scene_position(observer_id, scene_id)
        speaker_pos = self.get_character_scene_position(speaker_id, scene_id)
        
        if not observer_pos or not speaker_pos:
            log_timestamp(f"  → ❌ Lipsesc poziții! observer_pos={observer_pos}, speaker_pos={speaker_pos} → 'centru'", "gaze")
            return "centru"
        
        log_timestamp(f"  → Poziții: observer x={observer_pos.x()}, speaker x={speaker_pos.x()}", "gaze")
        
        # Threshold de 100px pentru diferențiere
        diff = speaker_pos.x() - observer_pos.x()
        log_timestamp(f"  → Diferență X: {diff}px", "gaze")
        
        if speaker_pos.x() < observer_pos.x() - 100:
            log_timestamp(f"  → REGULA 3A: Speaker la stânga → 'stanga'", "gaze")
            return "stanga"
        elif speaker_pos.x() > observer_pos.x() + 100:
            log_timestamp(f"  → REGULA 3B: Speaker la dreapta → 'dreapta'", "gaze")
            return "dreapta"
        else:
            log_timestamp(f"  → REGULA 3C: Speaker aproape → 'centru'", "gaze")
            return "centru" 

    def set_character_gaze(self, char_id, direction):
        """
        Schimbă asset-ul pentru pupile unui personaj + salvează starea.
        VERSIUNE COMPLETĂ - COPY-PASTE DIRECT
        
        Args:
            char_id: ID-ul personajului
            direction: "stanga" | "centru" | "dreapta"  ⭐ NOTĂ: schimbat de la "left/center/right"
        """
        char = self.character_manager.get_character(char_id)
        char_layers = self.character_layers.get(char_id)
        
        if not char or not char_layers:
            return
        
        # ⭐ SALVĂM STAREA PENTRU BLINKING ANIMATOR
        char.current_gaze_direction = direction
        
        # Verificăm config gaze tracking
        gaze_config = char.components.get("visual_states", {}).get("gaze_tracking")
        if not gaze_config or not gaze_config.get("enabled"):
            return
        
        target_part = gaze_config.get("target_part")
        direction_file = gaze_config.get("directions", {}).get(direction)
        
        if not target_part or not direction_file:
            return
        
        # Schimbăm asset-ul
        target_layer = char_layers.get(target_part)
        if target_layer:
            new_pixmap_path = os.path.join(char.assets_path, direction_file)
            if os.path.exists(new_pixmap_path):
                original_pixmap = QPixmap(new_pixmap_path)
                
                scene_id = self.scene_manager.current_scene_id
                scene_config = char.get_config_for_scene(scene_id)
                if scene_config:
                    scale = scene_config.get("scale", 0.3)
                    scaled_pixmap = original_pixmap.scaled(
                        int(original_pixmap.width() * scale),
                        int(original_pixmap.height() * scale),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    
                    target_layer.setPixmap(scaled_pixmap)
                    target_layer.setFixedSize(scaled_pixmap.size())
                    
                    self.gaze_states[char_id] = direction
                    log_timestamp(f"👀 [GAZE] '{char_id}' privește '{direction}'", "gaze")

    def update_all_gazes(self):
        """
        Actualizează direcția privirii pentru TOATE personajele active.
        """
        if not self.current_speaker:
            log_timestamp(f"👀 [GAZE] Nimeni nu vorbește → toți privesc 'centru'", "gaze")
            for char_id in self.character_manager.active_characters:
                self.set_character_gaze(char_id, "centru")
            return
        
        scene_id = self.scene_manager.current_scene_id
        log_timestamp(f"👀 [GAZE] Speaker: '{self.current_speaker}' → actualizare toate privirile", "gaze")
        
        for char_id in self.character_manager.active_characters:
            direction = self.calculate_gaze_direction(
                observer_id=char_id,
                speaker_id=self.current_speaker,
                scene_id=scene_id
            )
            self.set_character_gaze(char_id, direction)
            
    def set_speaker(self, speaker_id):
        """
        Setează speaker-ul activ și actualizează TOATE privirile.
        
        Args:
            speaker_id: "user" sau character_id sau None
        """
        if self.current_speaker == speaker_id:
            return  # Deja setat, nu facem nimic
        
        self.current_speaker = speaker_id
        
        if speaker_id:
            log_timestamp(f"🗣️ [GAZE] Speaker nou: '{speaker_id}'", "gaze")
        else:
            log_timestamp(f"🗣️ [GAZE] Speaker resetat (nimeni)", "gaze")
        
        self.update_all_gazes()

    def _route_user_input(self, text):
        """
        Punctul de intrare pentru orice input de la utilizator. (VERSIUNE EXTINSĂ)
        """
        log_timestamp(f"🚦 [ROUTER] Se analizează input-ul: '{text}'", "router")
        self.last_user_text = text

        # --- BLOC NOU DE INTERCEPTARE ---
        # Verificăm dacă suntem în modul profesor și dacă s-a spus comanda "Gata!"
        text_lower = text.strip().lower()
        if self.teacher_mode_active and text_lower in ["gata", "gata gata"]:
            log_timestamp("📸 [VALIDARE] Comanda 'Gata!' detectată. Se declanșează validarea vizuală.", "app")
            self._trigger_visual_validation()
            return # Oprim orice altă procesare
        # --- SFÂRȘIT BLOC NOU ---

        if self.app_state == 'AWAITING_DOMAIN_CHOICE':
            log_timestamp("🚦 [ROUTER] Stare: Așteptare alegere domeniu. Se procesează răspunsul...", "app")
            self._handle_domain_choice(text)
            return

        if self.teacher_mode_active:
            log_timestamp("🚦 [ROUTER] Stare: În modul Profesor. Se procesează ca răspuns de la elev...", "app")
            self._process_student_answer(text)
            return
        
        log_timestamp("🚦 [ROUTER] Stare: Conversație normală. Se rulează logica standard.", "router")
        self._validate_active_speaker()

        if self.waiting_for_travel_clarification:
            log_timestamp("🚦 [ROUTER] În waiting state pentru clarificare călătorie.", "router")
            self._handle_travel_clarification_response(text)
            return

        if self._check_and_switch_speaker(text):
            return

        if self.intent_thread is not None:
            try:
                if self.intent_thread.isRunning():
                    log_timestamp("🧹 [ROUTER] Curăț intent thread vechi care încă rulează...", "router")
                    self.intent_thread.quit()
                    self.intent_thread.wait(500)
            except RuntimeError:
                pass
        
        self.intent_thread = None
        self.intent_worker = None

        log_timestamp("🚦 [ROUTER] Se clasifică intenția...", "router")
        self.intent_thread = QThread()
        self.intent_worker = IntentClassifierWorker(text)
        self.intent_worker.moveToThread(self.intent_thread)
        
        self.intent_worker.intent_classified.connect(self._handle_intent_classification)
        self.intent_worker.error_occurred.connect(lambda err: log_timestamp(f"🚦 [ROUTER] Eroare la clasificare: {err}", "router"))
        
        self.intent_worker.finished.connect(self.intent_thread.quit)
        self.intent_worker.finished.connect(self.intent_worker.deleteLater)
        self.intent_thread.finished.connect(self.intent_thread.deleteLater)
        
        self.intent_thread.started.connect(self.intent_worker.run)
        self.intent_thread.start()

    def _trigger_visual_validation(self):
        """
        Pornește un worker Gemini specializat pentru a valida vizual acțiunea copilului.
        """
        if not self.webcam_worker or self.webcam_worker.last_frame is None:
            log_timestamp("❌ [VALIDARE] Camera nu funcționează. Validare anulată.", "app")
            self.add_to_chat("Sistem", "Eroare: Camera nu funcționează.")
            return

        log_timestamp("⏳ [VALIDARE] Se pregătește promptul pentru validarea vizuală...", "app")
        
        # Găsim informațiile despre întrebarea curentă
        current_question_obj = None
        for q in self.current_tier_data["questions"]:
            if q["id"] == self.current_question_id:
                current_question_obj = q
                break
        
        if not current_question_obj:
            log_timestamp("❌ [VALIDARE] Nu s-a găsit întrebarea curentă! Anulare.", "app")
            return

        # Construim promptul specializat pentru validare
        task_description = current_question_obj["text"].replace("Când ești gata, spune tare și clar GATA!", "").strip()
        expected_item = current_question_obj["correct_answers"][0] # Luăm primul răspuns ca referință

        validation_prompt = f"""
Ești un asistent AI specializat în validare vizuală, cu rol de arbitru. Fii strict, obiectiv și precis.

CONTEXT: Un copil a primit următoarea sarcină: "{task_description}"
Se așteaptă ca el să arate la cameră un obiect care este '{expected_item}'.

SARCINA TA: Analizează imaginea atașată și determină dacă acțiunea copilului este corectă. 
- Fii flexibil la nuanțe (ex: roșu deschis/închis).
- Concentrează-te pe obiectul principal prezentat de copil.
- Ignoră alte obiecte din fundal.

Răspunde DOAR cu un obiect JSON valid cu următoarea structură:
{{
  "success": true/false,
  "reason": "O scurtă explicație a deciziei tale. Ex: 'Copilul arată un obiect roșu.' sau 'Obiectul arătat este galben, nu roșu.'"
}}
"""
        
        image_to_send = self.webcam_worker.last_frame.copy()
        model_name = self.config.get("ai_model_name", "models/gemini-flash-lite-latest")

        # Pornim un worker Gemini normal, dar cu un prompt și un handler diferit
        self.gemini_worker = GeminiWorker(validation_prompt, image_to_send, "", model_name)
        self.gemini_thread = QThread()
        self.gemini_worker.moveToThread(self.gemini_thread)

        self.gemini_worker.response_ready.connect(self._handle_visual_validation_response)
        self.gemini_worker.error_occurred.connect(self.handle_ai_error) # Putem refolosi handler-ul de eroare
        self.gemini_worker.finished.connect(self.gemini_thread.quit)
        self.gemini_worker.finished.connect(self.gemini_worker.deleteLater)
        self.gemini_thread.finished.connect(self.gemini_thread.deleteLater)
        
        self.gemini_thread.started.connect(self.gemini_worker.run)
        self.gemini_thread.start()
        log_timestamp("🚀 [VALIDARE] Worker-ul de validare vizuală a fost pornit.", "app")

    def _handle_visual_validation_response(self, json_string):
        """
        Procesează răspunsul de la worker-ul de validare (true/false)
        și pasează rezultatul către worker-ul pedagogic (LearningAgentWorker).
        """
        try:
            result = json.loads(json_string)
            success = result.get("success", False)
            reason = result.get("reason", "Motiv necunoscut.")
            log_timestamp(f"👁️ [VALIDARE] Rezultat primit: Succes = {success}. Motiv: {reason}", "app")

            # Acum, transformăm acest rezultat într-un "răspuns" text pentru
            # logica pedagogică pe care am construit-o deja.
            # Simulăm un răspuns de la copil.
            simulated_answer = "corect" if success else "greșit"
            
            # Apelăm metoda existentă care se ocupă de logica de învățare
            self._process_student_answer(simulated_answer)

        except json.JSONDecodeError as e:
            log_timestamp(f"❌ [VALIDARE] Eroare la parsarea JSON-ului de validare: {e}", "app")
            # În caz de eroare, presupunem că a fost greșit și repetăm
            self._process_student_answer("eroare de validare")

    def _generate_world_knowledge(self, current_character_id):
        """
        Generează cunoștințe despre TOȚI personajele din lume pentru AI.
        Astfel fiecare personaj știe despre ceilalți, chiar dacă nu sunt pe scenă.
        """
        knowledge = "\n\n--- CUNOȘTINȚE DESPRE LUMEA TA ---\n"
        knowledge += "Iată personajele care există în această lume (chiar dacă nu sunt aici acum):\n\n"
        
        for char_id, char in self.character_manager.available_characters.items():
            if char_id == current_character_id:
                continue  # Nu includem personajul curent
            
            # Informații de bază
            knowledge += f"📌 {char.display_name} ({char_id}):\n"
            knowledge += f"   - Casa: {char.home_scene}\n"
            
            # Unde e acum (verificăm dacă e pe scenă activă)
            if char_id in self.character_manager.active_characters:
                knowledge += f"   - Locație actuală: AICI cu tine (pe scenă)\n"
            else:
                # Verificăm în ce scenă se află (folosind scene_manager pentru tracking)
                current_scene = self.scene_manager.current_scene_id
                if char.home_scene == current_scene:
                    knowledge += f"   - Locație actuală: probabil acasă ({char.home_scene})\n"
                else:
                    knowledge += f"   - Locație actuală: nu e pe scenă (undeva în lume)\n"
            
            # Capacități
            if char.can_leave_home:
                knowledge += f"   - Poate călători în alte locuri\n"
            else:
                knowledge += f"   - Nu poate părăsi casa ({char.home_scene})\n"
            
            if char.can_be_summoned:
                knowledge += f"   - Poate fi chemat prin magie\n"
            
            knowledge += "\n"
        
        knowledge += "IMPORTANT: Dacă utilizatorul întreabă despre acești personaje, tu ȘTII despre ei!\n"
        knowledge += "Dacă nu sunt pe scenă cu tine acum, poți menționa că nu îi vezi aici.\n"
        
        return knowledge

    def _generate_clarification_question(self, destination, candidates_list):
        """
        Generează o întrebare de clarificare pentru AI când există ambiguitate.
        
        Args:
            destination (str): Scena destinație (ex: 'scoala')
            candidates_list (list): Lista de character objects care sunt candidați
        
        Returns:
            str: Întrebarea formatată pentru AI să o spună
        """
        if len(candidates_list) == 1:
            # Un singur candidat
            nume_candidat = candidates_list[0].display_name
            intrebare = f"Mergem la {destination}. Să vină și {nume_candidat} sau doar eu cu tine?"
        else:
            # Multipli candidați
            nume_lista = [char.display_name for char in candidates_list]
            if len(nume_lista) == 2:
                nume_str = f"{nume_lista[0]} și {nume_lista[1]}"
            else:
                nume_str = ", ".join(nume_lista[:-1]) + f" și {nume_lista[-1]}"
            
            intrebare = f"Mergem la {destination}. Să vină și {nume_str} sau doar eu cu tine?"
        
        return intrebare
        
    def _handle_intent_classification(self, intent_data):
        """
        Slot care primește rezultatul de la IntentClassifierWorker și execută acțiunea.
        Implementează ÎNTREAGA logică de business pentru toate tipurile de intent.
        ⭐ INCLUDE LOGICĂ SPECIALĂ PENTRU MODUL PROFESOR
        """
        intent = intent_data.get("intent")
        log_timestamp(f"🎯 [ROUTER] Intent detectat: '{intent}' | Data: {intent_data}", "router")
        
        # ========================================================================
        # PRIORITATE 0: VERIFICARE MODUL PROFESOR ACTIV
        # ========================================================================
        if self.teacher_mode_active:
            log_timestamp(f"🎓 [ROUTER] Modul Profesor ACTIV - procesare specială", "router")
            
            # Permitem doar exit_teacher_mode și conversation în Modul Profesor
            if intent == "exit_teacher_mode":
                log_timestamp(f"🛑 [ROUTER] Comandă de ieșire din Modul Profesor", "router")
                self.exit_teacher_mode()
                return
            
            elif intent == "conversation":
                # În Modul Profesor, orice conversație este tratată ca răspuns la întrebare
                log_timestamp(f"🎓 [ROUTER] Răspuns elev: '{self.last_user_text}'", "router")
                self._process_student_answer(self.last_user_text)
                return
            
            else:
                # Orice alt intent este ignorat în Modul Profesor
                log_timestamp(f"🔇 [ROUTER] Intent '{intent}' ignorat în Modul Profesor. Doar conversation și exit acceptate.", "router")
                return
        
        # ========================================================================
        # PRIORITATE 1: START_LEARNING - Inițiere sesiune de învățare
        # ========================================================================
        if intent == "start_learning":
            subject = intent_data.get("subject", "")
            log_timestamp(f"🎓 [ROUTER] Cerere de învățare: subiect='{subject}'", "router")
            
            student_member = None
            for member in self.family_data:
                learning_progress = member.get("learning_progress", {})
                if learning_progress:  # Are cel puțin un domeniu configurat
                    student_member = member
                    break
            
            if not student_member:
                error_msg = "[EMOTION:confuz] Hmm, nu găsesc niciun membru cu domenii de învățare configurate!"
                log_timestamp(f"❌ [ROUTER] Niciun membru nu are learning_progress configurat", "router")
                QTimer.singleShot(100, lambda: self._start_tts(error_msg))
                return
            
            student_name = student_member.get("name")
            
            # Verificăm dacă studentul are domenii configurate
            learning_progress = student_member.get("learning_progress", {})
            
            if not learning_progress:
                error_msg = f"[EMOTION:neutral] {student_name}, nu ai încă domenii de învățare configurate. Vorbește cu părinții tăi să le adauge!"
                log_timestamp(f"❌ [ROUTER] Student '{student_name}' nu are domenii configurate", "router")
                QTimer.singleShot(100, lambda: self._start_tts(error_msg))
                return
            
            # Dacă există un singur domeniu, îl selectăm automat
            if len(learning_progress) == 1:
                domain_id = list(learning_progress.keys())[0]
                log_timestamp(f"✅ [ROUTER] Un singur domeniu disponibil: '{domain_id}'. Selectare automată.", "router")
                self.start_learning_session(student_name, domain_id)
                return
            
            # Dacă există multiple domenii, întrebăm
            # ⚠️ Pentru simplificare, selectăm primul
            domain_id = list(learning_progress.keys())[0]
            log_timestamp(f"🎯 [ROUTER] Selectare domeniu implicit: '{domain_id}'", "router")
            self.start_learning_session(student_name, domain_id)
            return
        
        # ========================================================================
        # PRIORITATE 2: EXIT_TEACHER_MODE (apelat în afara Modul Profesor)
        # ========================================================================
        if intent == "exit_teacher_mode":
            log_timestamp(f"⚠️ [ROUTER] Comandă exit_teacher_mode în afara Modului Profesor - ignorată", "router")
            return
        
        # ========================================================================
        # 1. TRAVEL_WITH_CHARACTER - "Cucuvel, hai să mergem la X" SAU "Hai să mergem împreună"
        # ========================================================================
        if intent == "travel_with_character":
            char_id = intent_data.get("character")
            destination = intent_data.get("scene")
            self.is_speaking = False
            self.is_thinking = False

            
            log_timestamp(f"🚶 [TRAVEL_WITH] Procesare: user + personaj → '{destination}'", "router")
            
            # ⭐ NOU: Dacă destinația e scena curentă → convertim în SUMMON
            current_scene = self.scene_manager.current_scene_id
            if destination == current_scene:
                log_timestamp(f"🔄 [TRAVEL_WITH] Destinația '{destination}' e scena curentă → convertim în SUMMON", "router")
                
                # Dacă character explicit → summon acel personaj
                if char_id:
                    # Procesăm ca summon
                    intent_data_modified = {"intent": "summon_character", "character": char_id}
                    self._handle_intent_classification(intent_data_modified)
                else:
                    # Character implicit → nu putem determina pe cine să chemăm
                    log_timestamp(f"🔇 [TRAVEL_WITH] Character implicit în scenă curentă - SILENCE", "router")
                return
            
            # ⭐ Cazul 1: Character ID specificat explicit
            if char_id:
                log_timestamp(f"🚶 [TRAVEL_WITH] Personaj EXPLICIT specificat: '{char_id}'", "router")
                
                # Validare 1: Personaj există?
                char = self.character_manager.get_character(char_id)
                if not char:
                    log_timestamp(f"🔇 [TRAVEL_WITH] Personaj '{char_id}' nu există - SILENCE", "router")
                    return
                
                # Validare 2: Personajul e pe scenă cu noi?
                if char_id not in self.character_manager.active_characters:
                    log_timestamp(f"🔇 [TRAVEL_WITH] Personaj '{char_id}' nu e pe scenă - SILENCE", "router")
                    return
                
                # Validare 3: Personajul poate pleca?
                if not char.can_leave_home:
                    log_timestamp(f"🔇 [TRAVEL_WITH] '{char_id}' nu poate pleca din casă - SILENCE", "router")
                    return
                
                # Validare 4: Scenă validă?
                if destination not in self.scene_manager.scenes:
                    log_timestamp(f"🔇 [TRAVEL_WITH] Scenă '{destination}' invalidă - SILENCE", "router")
                    return
                
                # Validare 5: Personaj are config pentru destinație?
                if destination not in char.scene_configs:
                    log_timestamp(f"🔇 [TRAVEL_WITH] '{char_id}' n-are config pt '{destination}' - SILENCE", "router")
                    return
                
                # ✅ TOATE VALIDĂRILE TRECUTE - EXECUTĂ cu personaj explicit
                log_timestamp(f"✅ [TRAVEL_WITH] Deplasare validă: user + '{char_id}' → '{destination}'", "router")
                self._execute_travel_with_characters(destination, [char_id])
            
            # ⭐ Cazul 2: Character ID NULL (ex: "Hai să mergem împreună la școală")
            else:
                log_timestamp(f"🚶 [TRAVEL_WITH] Personaj IMPLICIT (împreună fără nume)", "router")
                
                # Scenă validă?
                if destination not in self.scene_manager.scenes:
                    log_timestamp(f"🔇 [TRAVEL_WITH] Scenă '{destination}' invalidă - SILENCE", "router")
                    return
                
                # Câți personaje sunt pe scenă?
                active_chars = list(self.character_manager.active_characters.keys())
                log_timestamp(f"📊 [TRAVEL_WITH] Personaje active pe scenă: {active_chars}", "router")
                
                if len(active_chars) == 0:
                    # Niciun personaj → user merge solo
                    log_timestamp(f"🚶 [TRAVEL_WITH] Niciun personaj pe scenă → travel solo", "router")
                    self._execute_travel_solo(destination)
                
                elif len(active_chars) == 1:
                    # Un singur personaj → merge automat cu el, fără întrebare
                    char_id = active_chars[0]
                    char = self.character_manager.get_character(char_id)
                    
                    # Validări pentru singurul personaj
                    if not char.can_leave_home:
                        log_timestamp(f"🔇 [TRAVEL_WITH] Singurul personaj '{char_id}' nu poate pleca - travel solo", "router")
                        self._execute_travel_solo(destination)
                        return
                    
                    if destination not in char.scene_configs:
                        log_timestamp(f"🔇 [TRAVEL_WITH] Singurul personaj '{char_id}' n-are config pt '{destination}' - travel solo", "router")
                        self._execute_travel_solo(destination)
                        return
                    
                    # ✅ Merge automat cu singurul personaj
                    log_timestamp(f"✅ [TRAVEL_WITH] Un singur personaj pe scenă → merge automat cu '{char_id}'", "router")
                    self._execute_travel_with_characters(destination, [char_id])
                
                else:
                    # 2+ personaje → AMBIGUITATE → întreabă pentru clarificare
                    log_timestamp(f"❓ [TRAVEL_WITH] AMBIGUITATE: {len(active_chars)} personaje pe scenă → cere clarificare", "router")
                    
                    # Filtrăm candidații: doar cei care pot călători și au config pentru destinație
                    candidates = []
                    for char_id in active_chars:
                        char = self.character_manager.get_character(char_id)
                        if char.can_leave_home and destination in char.scene_configs:
                            candidates.append(char)
                    
                    if len(candidates) == 0:
                        # Niciun candidat valid → travel solo
                        log_timestamp(f"🔇 [TRAVEL_WITH] Niciun candidat valid → travel solo", "router")
                        self._execute_travel_solo(destination)
                    
                    elif len(candidates) == 1:
                        # Un singur candidat valid → merge automat cu el
                        char_id = candidates[0].id
                        log_timestamp(f"✅ [TRAVEL_WITH] Un singur candidat valid '{char_id}' → merge automat", "router")
                        self._execute_travel_with_characters(destination, [char_id])
                    
                    else:
                        # 2+ candidați valizi → CERE CLARIFICARE
                        log_timestamp(f"❓ [TRAVEL_WITH] {len(candidates)} candidați valizi → întreabă user-ul", "router")
                        self._ask_for_travel_clarification(destination, candidates)
        
        # ========================================================================
        # 2. TRAVEL_SOLO - "Hai să mergem la X" (fără personaj)
        # ========================================================================
        elif intent == "travel_solo":
            destination = intent_data.get("scene")
            self.is_speaking = False
            self.is_thinking = False

            
            log_timestamp(f"🚶 [TRAVEL_SOLO] Procesare: user solo → '{destination}'")
            
            # Validare: Scenă validă?
            if destination not in self.scene_manager.scenes:
                log_timestamp(f"🔇 [TRAVEL_SOLO] Scenă '{destination}' invalidă - SILENCE")
                return
            
            # ✅ VALIDARE TRECUTĂ - EXECUTĂ
            log_timestamp(f"✅ [TRAVEL_SOLO] Schimbare scenă solo: → '{destination}'")
            log_timestamp(f"📊 [TRAVEL_SOLO] Personaje active ÎNAINTE de clear: {list(self.character_manager.active_characters.keys())}")
            
            # 1. Schimbă scena
            self.scene_manager.set_scene(destination)
            
            # 2. Curăță UI
            self.character_manager.clear_active_characters()
            log_timestamp(f"📊 [TRAVEL_SOLO] Personaje active DUPĂ clear: {list(self.character_manager.active_characters.keys())}")
            
            # 3. Încarcă natives și visitors
            self.character_manager.sync_characters_for_scene(destination, self.scene_manager)
            log_timestamp(f"📊 [TRAVEL_SOLO] Personaje active DUPĂ sync: {list(self.character_manager.active_characters.keys())}")
        
        # ========================================================================
        # 3. SUMMON_CHARACTER - "Cucuvel, vino aici" (MAGIE)
        # ========================================================================
        elif intent == "summon_character":
            char_id = intent_data.get("character")
            
            # ⭐ RESETARE FLAG-URI
            self.is_speaking = False
            self.is_thinking = False
            
            log_timestamp(f"✨ [SUMMON] Procesare chemare: '{char_id}' → scena curentă", "router")
            
            # Validare 1: Personaj există?
            char = self.character_manager.get_character(char_id)
            if not char:
                log_timestamp(f"🔇 [SUMMON] Personaj '{char_id}' nu există - SILENCE", "router")
                return
            
            # Validare 2: E deja pe scenă?
            if char_id in self.character_manager.active_characters:
                log_timestamp(f"🔇 [SUMMON] '{char_id}' e deja pe scenă - SILENCE", "router")
                return
            
            # Validare 3: Poate fi chemat?
            if not char.can_be_summoned:
                log_timestamp(f"🔇 [SUMMON] '{char_id}' nu poate fi chemat (can_be_summoned=False) - SILENCE", "router")
                return
            
            # Validare 4: Are config pentru scena curentă?
            current_scene = self.scene_manager.current_scene_id
            if current_scene not in char.scene_configs:
                log_timestamp(f"🔇 [SUMMON] '{char_id}' n-are config pt '{current_scene}' - SILENCE", "router")
                return
            
            # ✅ TOATE VALIDĂRILE TRECUTE - EXECUTĂ
            log_timestamp(f"✅ [SUMMON] Chemare validă: '{char_id}' → '{current_scene}'", "router")
            log_timestamp(f"📊 [SUMMON] Personaje active ÎNAINTE: {list(self.character_manager.active_characters.keys())}", "router")
            
            # 1. Setează scena personajului
            char.current_scene_id = current_scene
            
            # 2. Adaugă pe scenă
            self.character_manager.add_character_to_stage(char_id)
            log_timestamp(f"📊 [SUMMON] Personaje active DUPĂ adăugare: {list(self.character_manager.active_characters.keys())}", "router")
            
            # ⭐ 3. CURĂȚARE FORȚATĂ TTS înainte de confirmare vocală
            if self.tts_worker is not None:
                try:
                    log_timestamp("🧹 [SUMMON] Curăț TTS worker vechi...", "router")
                    self.tts_worker.stop()
                    self.tts_worker.deleteLater()
                except:
                    pass
                self.tts_worker = None
            
            if self.tts_thread is not None:
                try:
                    self.tts_thread.quit()
                    self.tts_thread.wait(500)
                    self.tts_thread.deleteLater()
                except:
                    pass
                self.tts_thread = None
            
            # ⭐ 4. Personajul confirmă venirea (ACUM e safe)
            if char.language.startswith("en"):
                arrival_prompt = "Confirm cheerfully in your personality that you've arrived. Say a short greeting like 'Here I am!' or 'You called?'"
            else:
                arrival_prompt = "Confirmă vesel în personalitatea ta că ai venit. Spune un salut scurt ca 'Sunt aici!' sau 'M-ai chemat?'"
            
            self.process_question(arrival_prompt, char_id)
        
        # ========================================================================
        # 4. SEND_CHARACTER - "Iepurașule, mergi la X" (FIZIC)
        # ========================================================================
        elif intent == "send_character":
            char_id = intent_data.get("character")
            destination = intent_data.get("scene")
            self.is_speaking = False
            self.is_thinking = False
            
            log_timestamp(f"📤 [SEND] Procesare trimitere: '{char_id}' → '{destination}'")
            
            # ⚠️ VALIDARE CRITICĂ: E personajul FIZIC pe scenă?
            if char_id not in self.character_manager.active_characters:
                log_timestamp(f"🔇 [SEND] '{char_id}' NU e pe scenă fizic - SILENCE (nu te aude)")
                log_timestamp(f"📊 [SEND] Personaje active: {list(self.character_manager.active_characters.keys())}")
                return
            
            # De aici știm sigur că personajul e pe scenă și ne poate auzi
            char = self.character_manager.get_character(char_id)
            
            log_timestamp(f"✅ [SEND] Personaj găsit pe scenă, se procesează comenzile...")
            
            # Validare 1: Poate pleca?
            if not char.can_leave_home:
                log_timestamp(f"❌ [SEND] '{char_id}' nu poate pleca (can_leave_home=False) - REFUZ")
                if char.language.startswith("en"):
                    refusal_prompt = "Explain in your personality why you cannot leave your home. Be polite but firm."
                else:
                    refusal_prompt = "Explică în personalitatea ta de ce nu poți pleca din casa ta. Fii politicos dar ferm."
                
                self.process_question(refusal_prompt, char_id)
                return
            
            # Validare 2: Scenă validă?
            if destination not in self.scene_manager.scenes:
                log_timestamp(f"❌ [SEND] Scenă '{destination}' invalidă - REFUZ")
                if char.language.startswith("en"):
                    refusal_prompt = f"Explain politely that you don't know the place called '{destination}'."
                else:
                    refusal_prompt = f"Explică politicos că nu cunoști locul numit '{destination}'."
                
                self.process_question(refusal_prompt, char_id)
                return
            
            # Validare 3: Are config pentru destinație?
            if destination not in char.scene_configs:
                log_timestamp(f"❌ [SEND] '{char_id}' n-are config pt '{destination}' - REFUZ")
                if char.language.startswith("en"):
                    refusal_prompt = f"Explain in your personality why you cannot go to '{destination}'. That's not your place."
                else:
                    refusal_prompt = f"Explică în personalitatea ta de ce nu poți merge la '{destination}'. Nu e locul tău."
                
                self.process_question(refusal_prompt, char_id)
                return
            
            # Validare 4: E deja în destinație?
            if char.current_scene_id == destination:
                log_timestamp(f"❌ [SEND] '{char_id}' e deja în '{destination}' - REFUZ")
                if char.language.startswith("en"):
                    refusal_prompt = f"Say cheerfully that you're already at {destination}!"
                else:
                    refusal_prompt = f"Spune vesel că ești deja la {destination}!"
                
                self.process_question(refusal_prompt, char_id)
                return
            
            # ✅ TOATE VALIDĂRILE TRECUTE - EXECUTĂ
            log_timestamp(f"✅ [SEND] Trimitere validă: '{char_id}' → '{destination}'")
            log_timestamp(f"📊 [SEND] Personaje active ÎNAINTE de mutare: {list(self.character_manager.active_characters.keys())}")
            # ⭐ MODIFICARE: Stocăm mutarea pentru DUPĂ ce vorbește
            self.pending_move_after_tts = {
                'char_id': char_id,
                'destination': destination
            }
            log_timestamp(f"⏳ [SEND] Mutare programată DUPĂ ce vorbește: '{char_id}' → '{destination}'", "router")
            # Generăm prompt-ul de plecare
            destination_data = self.scene_manager.get_scene_data(destination)
            if char.language.startswith("en"):
                departure_prompt = f"Say a brief farewell as you're leaving to go to {destination_data.get('name', destination)}. Something like 'I'm heading out!' or 'See you later!'"
            else:
                departure_prompt = f"Spune un rămas bun scurt, deoarece pleci spre {destination_data.get('name', destination)}. Ceva de genul 'Plec!' sau 'Pe curând!'"
            # Personajul vorbește, iar mutarea se va executa DUPĂ în speech_finished()
            self.process_question(departure_prompt, char_id)
        
        # ========================================================================
        # 5. TRANSLATION_REQUEST - "Nu am înțeles, poți să traduci?"
        # ========================================================================
        elif intent == "translation_request":
            log_timestamp(f"🌐 [TRANSLATION] Procesare cerere de traducere", "router")
            self._handle_translation_request()
        
        # ========================================================================
        # 6. CONVERSATION - Orice altceva (ULTIMUL BLOC - DEFAULT)
        # ========================================================================
        else:
            log_timestamp(f"💬 [ROUTER] Intenție de conversație detectată. Se pasează la procesarea normală.")
            log_timestamp(f"📊 [CONVERSATION] Vorbitor activ: '{self.active_speaker_id}'")
            log_timestamp(f"📊 [CONVERSATION] Personaje active: {list(self.character_manager.active_characters.keys())}")
            # Trimitem textul original al utilizatorului la vorbitorul activ curent.
            self.process_question(self.last_user_text, self.active_speaker_id)

    def _handle_translation_request(self):
        """
        Gestionează cererea de traducere/explicare a ultimei replici.
        
        Flow:
        1. Găsește ultima replică a speaker-ului activ
        2. Găsește un translator (personaj RO pe scenă)
        3. Translator explică în română
        4. Auto-switch înapoi la speaker-ul original
        
        VERSIUNE COMPLETĂ - COPY-PASTE DIRECT
        """
        log_timestamp("🌐 [TRANSLATION] Căutare replică de tradus...", "router")
        
        # Validare 1: Avem un speaker activ?
        if not self.active_speaker_id:
            log_timestamp("🔇 [TRANSLATION] Nu există speaker activ - SILENCE", "router")
            return
        
        # Validare 2: Speaker-ul e pe scenă?
        if self.active_speaker_id not in self.character_manager.active_characters:
            log_timestamp(f"🔇 [TRANSLATION] Speaker '{self.active_speaker_id}' nu e pe scenă - SILENCE", "router")
            return
        
        # Validare 3: Avem o replică de tradus?
        if self.active_speaker_id not in self.last_character_speeches:
            log_timestamp(f"🔇 [TRANSLATION] Nu avem nicio replică salvată de la '{self.active_speaker_id}' - SILENCE", "router")
            return
        
        original_text = self.last_character_speeches[self.active_speaker_id]
        original_speaker = self.character_manager.get_character(self.active_speaker_id)
        
        log_timestamp(f"📝 [TRANSLATION] Replică de tradus: '{original_text[:50]}...'", "router")
        
        # Căutăm un translator (personaj RO pe scenă, diferit de speaker)
        translator_id = None
        for char_id in self.character_manager.active_characters:
            char = self.character_manager.get_character(char_id)
            if char.language.startswith("ro") and char_id != self.active_speaker_id:
                translator_id = char_id
                log_timestamp(f"✅ [TRANSLATION] Translator găsit: '{translator_id}'", "router")
                break
        
        # Validare 4: Avem translator disponibil?
        if not translator_id:
            log_timestamp("🔇 [TRANSLATION] Nu există translator (personaj RO) pe scenă - SILENCE", "router")
            return
        
        # Construim prompt pentru translator
        translator = self.character_manager.get_character(translator_id)
        
        if original_speaker.language.startswith("en"):
            # Speaker vorbește EN → traducem în RO
            prompt = (
                f"Utilizatorul nu a înțeles ultima replică a lui {original_speaker.display_name} "
                f"care a spus în engleză: '{original_text}'. "
                f"Explică-i în română, simplu și clar, ce a vrut să spună. "
                f"Începe cu: '{original_speaker.display_name} a spus că...' sau similar."
            )
        else:
            # Speaker vorbește altceva → explicăm simplu
            prompt = (
                f"Utilizatorul nu a înțeles ultima replică a lui {original_speaker.display_name}: "
                f"'{original_text}'. Explică-i mai simplu ce a vrut să spună."
            )
        
        log_timestamp(f"🌐 [TRANSLATION] Prompt către translator: '{prompt[:80]}...'", "router")
        
        # Salvăm speaker-ul original pentru revenire după traducere
        self.pending_speaker_return = self.active_speaker_id
        log_timestamp(f"💾 [TRANSLATION] Salvez speaker original: '{self.pending_speaker_return}'", "router")
        
        # Switch temporar la translator
        self.active_speaker_id = translator_id
        log_timestamp(f"🔄 [TRANSLATION] Switch temporar la translator: '{translator_id}'", "router")
        
        # Procesăm traducerea
        self.process_question(prompt, translator_id)

    def _ask_for_travel_clarification(self, destination, candidates):
        """
        Pune o întrebare de clarificare prin vorbitorul activ când există ambiguitate.
        Intră în waiting state pentru răspuns.
        
        Args:
            destination (str): Scena destinație
            candidates (list): Lista de character objects care sunt candidați
        """
        log_timestamp(f"❓ [CLARIFY] Se cere clarificare pentru călătoria la '{destination}'", "router")
        
        # Salvăm datele călătoriei
        self.pending_travel_data = {
            'destination': destination,
            'candidates': candidates,
            'candidate_ids': [char.id for char in candidates]
        }
        
        # Intrăm în waiting state
        self.waiting_for_travel_clarification = True
        
        # Pornim timeout de 15 secunde
        self.clarification_timeout_timer.start(15000)
        log_timestamp(f"⏱️ [CLARIFY] Timeout de 15s pornit", "router")
        
        # Generăm întrebarea
        intrebare = self._generate_clarification_question(destination, candidates)
        
        # Trimitem întrebarea prin vorbitorul activ
        log_timestamp(f"❓ [CLARIFY] Întrebare: '{intrebare}'", "router")
        self.process_question(intrebare, self.active_speaker_id)

    def _handle_travel_clarification_response(self, text):
        """
        Interpretează răspunsul user-ului la întrebarea de clarificare călătorie.
        
        Logica SIMPLĂ (KISS):
        - Dacă răspunsul conține cuvinte clare pentru "toți" → toți merg
        - ORICE ALTCEVA (ambiguu, off-topic, neclar) → doar vorbitorul activ merge (FALLBACK)
        
        Args:
            text (str): Răspunsul user-ului
        """
        log_timestamp(f"💬 [CLARIFY] Procesare răspuns: '{text}'", "router")
        
        # Oprим timeout-ul
        self.clarification_timeout_timer.stop()
        
        # Extragem datele călătoriei
        destination = self.pending_travel_data['destination']
        candidates = self.pending_travel_data['candidates']
        candidate_ids = self.pending_travel_data['candidate_ids']
        
        # Resetăm state-ul
        self.waiting_for_travel_clarification = False
        self.pending_travel_data = None
        
        # Interpretare răspuns - SIMPLU cu fallback clar
        text_lower = text.lower()
        
        # Cuvinte cheie pentru "toți"
        cuvinte_toti = ["toți", "toti", "da", "și", "si", "amândoi", "amandoi", 
                        "toată", "toata", "lumea", "împreună", "impreuna", "cu toții"]
        
        # Verificăm dacă răspunsul conține cuvinte pentru "toți"
        raspuns_toti = any(cuv in text_lower for cuv in cuvinte_toti)
        
        if raspuns_toti:
            # TOȚI MERG
            log_timestamp(f"✅ [CLARIFY] Răspuns CLAR: TOȚI merg la '{destination}'", "router")
            log_timestamp(f"📊 [CLARIFY] Personaje care merg: {[self.active_speaker_id] + candidate_ids}", "router")
            
            # Toți candidații + vorbitorul activ
            all_travelers = [self.active_speaker_id] + candidate_ids
            self._execute_travel_with_characters(destination, all_travelers)
        
        else:
            # FALLBACK: DOAR VORBITORUL ACTIV (indiferent de răspuns)
            log_timestamp(f"⚠️ [CLARIFY] Răspuns AMBIGUU/OFF-TOPIC → FALLBACK: doar vorbitorul activ", "router")
            log_timestamp(f"📊 [CLARIFY] Merge doar: '{self.active_speaker_id}'", "router")
            
            self._execute_travel_with_characters(destination, [self.active_speaker_id])
    
    def _handle_clarification_timeout(self):
        """
        Handler pentru timeout când user-ul nu răspunde la întrebarea de clarificare.
        Fallback: doar vorbitorul activ merge.
        """
        log_timestamp(f"⏱️ [CLARIFY] TIMEOUT! User nu a răspuns în 15s", "router")
        
        if not self.waiting_for_travel_clarification or not self.pending_travel_data:
            return
        
        destination = self.pending_travel_data['destination']
        
        # Resetăm state-ul
        self.waiting_for_travel_clarification = False
        self.pending_travel_data = None
        
        # FALLBACK: doar vorbitorul activ
        log_timestamp(f"⚠️ [CLARIFY] FALLBACK din timeout → doar vorbitorul activ merge", "router")
        self._execute_travel_with_characters(destination, [self.active_speaker_id])

    def _execute_travel_solo(self, destination):
        log_timestamp(f"🚶 [EXEC SOLO] User merge SOLO → '{destination}'", "router")
        log_timestamp(f"📊 [EXEC SOLO] Personaje active ÎNAINTE: {list(self.character_manager.active_characters.keys())}", "router")
        
        # 1. Schimbă scena
        self.scene_manager.set_scene(destination)
        
        # 2. Curăță UI
        self.character_manager.clear_active_characters()
        log_timestamp(f"📊 [EXEC SOLO] Personaje active DUPĂ clear: []", "router")
        
        # 3. Încarcă natives și visitors
        self.character_manager.sync_characters_for_scene(destination, self.scene_manager)
        log_timestamp(f"📊 [EXEC SOLO] Personaje active DUPĂ sync: {list(self.character_manager.active_characters.keys())}", "router")
        
        # ⭐ 4. CHECKPOINT 4: Auto-greeting de la natives
        self._handle_arrival_greeting()

        # ⭐ 5. Re-calculate gaze pentru noua scenă
        self.update_all_gazes()

    def _execute_travel_with_characters(self, destination, character_ids):
        log_timestamp(f"🚶 [EXEC WITH] User + {character_ids} → '{destination}'", "router")
        log_timestamp(f"📊 [EXEC WITH] Personaje active ÎNAINTE: {list(self.character_manager.active_characters.keys())}", "router")
        
        # 1. Schimbă scena
        self.scene_manager.set_scene(destination)
        
        # 2. Curăță UI
        self.character_manager.clear_active_characters()
        log_timestamp(f"📊 [EXEC WITH] Personaje active DUPĂ clear: []", "router")
        
        # 3. Adaugă personajele călătoare
        for char_id in character_ids:
            char = self.character_manager.get_character(char_id)
            if char:
                char.current_scene_id = destination
                self.character_manager.add_character_to_stage(char_id)
                log_timestamp(f"✅ [EXEC WITH] '{char_id}' adăugat manual în '{destination}'", "router")
        
        # 4. Încarcă natives și visitors
        self.character_manager.sync_characters_for_scene(destination, self.scene_manager)
        log_timestamp(f"📊 [EXEC WITH] Personaje active DUPĂ sync: {list(self.character_manager.active_characters.keys())}", "router")
        
        # ⭐ 5. CHECKPOINT 4: Auto-greeting de la natives (dacă găsim)
        self._handle_arrival_greeting()

        # ⭐ 6. Re-calculate gaze pentru noua scenă
        self.update_all_gazes()

    def _transliterate_text(self, text, lang_code):
        """
        Transliterează un text dintr-un alfabet non-latin în caractere latine.
        """
        
        # ... Dicționarele GREEK_MAP și RUSSIAN_MAP rămân neschimbate ...
        GREEK_MAP = {
            'α': 'a', 'β': 'v', 'γ': 'gh', 'δ': 'd', 'ε': 'e', 'ζ': 'z', 'η': 'i', 'θ': 'th',
            'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm', 'ν': 'n', 'ξ': 'x', 'ο': 'o', 'π': 'p',
            'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't', 'υ': 'i', 'φ': 'f', 'χ': 'ch', 'ψ': 'ps', 'ω': 'o',
            'ά': 'a', 'έ': 'e', 'ή': 'i', 'ί': 'i', 'ό': 'o', 'ύ': 'i', 'ώ': 'o', 'ϊ': 'i',
            'ϋ': 'i', 'ΐ': 'i', 'ΰ': 'i', 'αι': 'e', 'ει': 'i', 'οι': 'i', 'ου': 'ou',
            'υι': 'i', 'αυ': 'av', 'ευ': 'ev', 'ηυ': 'iv', 'Α': 'A', 'Β': 'V', 'Γ': 'Gh',
            'Δ': 'D', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'I', 'Θ': 'Th', 'Ι': 'I', 'Κ': 'K', 'Λ': 'L',
            'Μ': 'M', 'Ν': 'N', 'Ξ': 'X', 'Ο': 'O', 'Π': 'P', 'Ρ': 'R', 'Σ': 'S', 'Τ': 'T',
            'Υ': 'I', 'Φ': 'F', 'Χ': 'Ch', 'Ψ': 'Ps', 'Ω': 'O', 'Ά': 'A', 'Έ': 'E', 'Ή': 'I',
            'Ί': 'I', 'Ό': 'O', 'Ύ': 'I', 'Ώ': 'O'
        }
        RUSSIAN_MAP = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'ye', 'ё': 'yo', 'ж': 'zh',
            'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
            'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts',
            'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': "'", 'э': 'e', 'ю': 'yu',
            'я': 'ya', 'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'Ye', 'Ё': 'Yo',
            'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
            'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'Kh',
            'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch', 'Ъ': '', 'Ы': 'Y', 'Ь': "'", 'Э': 'E',
            'Ю': 'Yu', 'Я': 'Ya'
        }
        
        # --- BLOC NOU: Dicționar de mapare pentru Japoneză (Hiragana) ---
        JAPANESE_MAP = {
            'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
            'か': 'ka', 'き': 'ki', 'く': 'ku', 'け': 'ke', 'こ': 'ko',
            'さ': 'sa', 'し': 'shi', 'す': 'su', 'せ': 'se', 'そ': 'so',
            'た': 'ta', 'ち': 'chi', 'つ': 'tsu', 'て': 'te', 'と': 'to',
            'な': 'na', 'に': 'ni', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no',
            'は': 'ha', 'ひ': 'hi', 'ふ': 'fu', 'へ': 'he', 'ほ': 'ho',
            'ま': 'ma', 'み': 'mi', 'む': 'mu', 'め': 'me', 'も': 'mo',
            'や': 'ya', 'ゆ': 'yu', 'よ': 'yo',
            'ら': 'ra', 'り': 'ri', 'る': 'ru', 'れ': 're', 'ろ': 'ro',
            'わ': 'wa', 'を': 'wo', 'ん': 'n',
            'が': 'ga', 'ぎ': 'gi', 'ぐ': 'gu', 'げ': 'ge', 'ご': 'go',
            'ざ': 'za', 'じ': 'ji', 'ず': 'zu', 'ぜ': 'ze', 'ぞ': 'zo',
            'だ': 'da', 'ぢ': 'ji', 'づ': 'zu', 'で': 'de', 'ど': 'do',
            'ば': 'ba', 'び': 'bi', 'ぶ': 'bu', 'べ': 'be', 'ぼ': 'bo',
            'ぱ': 'pa', 'ぴ': 'pi', 'ぷ': 'pu', 'ぺ': 'pe', 'ぽ': 'po',
            'きゃ': 'kya', 'きゅ': 'kyu', 'きょ': 'kyo',
            'ぎゃ': 'gya', 'ぎゅ': 'gyu', 'ぎょ': 'gyo',
            'しゃ': 'sha', 'しゅ': 'shu', 'しょ': 'sho',
            'じゃ': 'ja', 'じゅ': 'ju', 'じょ': 'jo',
            'ちゃ': 'cha', 'ちゅ': 'chu', 'ちょ': 'cho',
            'にゃ': 'nya', 'にゅ': 'nyu', 'にょ': 'nyo',
            'ひゃ': 'hya', 'ひゅ': 'hyu', 'ひょ': 'hyo',
            'びゃ': 'bya', 'びゅ': 'byu', 'びょ': 'byo',
            'ぴゃ': 'pya', 'ぴゅ': 'pyu', 'ぴょ': 'pyo',
            'みゃ': 'mya', 'みゅ': 'myu', 'みょ': 'myo',
            'りゃ': 'rya', 'りゅ': 'ryu', 'りょ': 'ryo',
            '、': ', ', '。': '.', 'ー': ''
        }
        # --- SFÂRȘIT BLOC NOU ---
        
        # Selectăm dicționarul corect
        if lang_code == 'el':
            char_map = GREEK_MAP
        elif lang_code == 'ru':
            char_map = RUSSIAN_MAP
        elif lang_code == 'ja': # <-- ADAUGĂM CONDIȚIA PENTRU JAPONEZĂ
            char_map = JAPANESE_MAP
        else:
            return text

        # Facem înlocuirea
        transliterated_text = ""
        i = 0
        while i < len(text):
            # Căutăm cea mai lungă potrivire posibilă (3, 2, apoi 1 caracter)
            if i + 2 < len(text) and text[i:i+3] in char_map:
                transliterated_text += char_map[text[i:i+3]]
                i += 3
            elif i + 1 < len(text) and text[i:i+2] in char_map:
                transliterated_text += char_map[text[i:i+2]]
                i += 2
            elif text[i] in char_map:
                transliterated_text += char_map[text[i]]
                i += 1
            else:
                transliterated_text += text[i]
                i += 1
        
        return transliterated_text

    def _teleport_to_meadow(self):
        """Callback apelat după TTS-ul de final de lecție pentru a teleporta la pajiște."""
        log_timestamp("✈️ [TELEPORT] Pauză! Teleportare la pajiște...", "app")
        # Mutăm utilizatorul și pe Cucuvel la pajiște
        self._execute_travel_with_characters("pajiste", ["cucuvel_owl"])
        
        # Curățarea finală standard
        self.speech_finished()

    def _clear_blackboard(self):
        """Ascunde toate elementele de pe tabla virtuală."""
        for label in self.blackboard_labels:
            label.hide()



    def _display_on_blackboard(self, display_string):
        """
        Funcția "Manager" care decide CE să afișeze pe tablă.
        """
        self._clear_blackboard() # Începem mereu prin a curăța tabla

        if not display_string:
            return # Nu avem ce afișa

        # --- LOGICA DE DECIZIE ---
        if display_string.lower().endswith('.png'):
            # Cazul 2: Trebuie să afișăm o imagine
            log_timestamp(f"칠판 [BLACKBOARD MANAGER] Decizie: Afișare imagine '{display_string}'", "app")
            self._display_image_on_blackboard(display_string)
        else:
            # Cazul 1: Trebuie să afișăm text
            log_timestamp(f"칠판 [BLACKBOARD MANAGER] Decizie: Afișare text '{display_string}'", "app")
            self._display_text_on_blackboard(display_string)
            
    def _display_image_on_blackboard(self, image_filename):
        """
        Funcție dedicată EXCLUSIV pentru afișarea de imagini PNG pe tablă.
        """
        # === COORDONATELE TALE CALIBRATE ===
        BLACKBOARD_RECT = QRect(590, 380, 360, 150)
        PADDING = 10
        # ====================================

        util_height = BLACKBOARD_RECT.height() - (2 * PADDING)
        util_width = BLACKBOARD_RECT.width() - (2 * PADDING)

        path_to_check = Path(f"assets/blackboard/objects/{image_filename}")
        if not path_to_check.exists():
            log_timestamp(f"⚠️ [BLACKBOARD] Imaginea '{image_filename}' nu a fost găsită!", "app")
            return

        label = self.blackboard_labels[0] # Folosim un singur label pentru imagini
        
        pixmap = QPixmap(str(path_to_check))

        # Scalăm imaginea pentru a încăpea, păstrând proporțiile
        if pixmap.width() > util_width or pixmap.height() > util_height:
            pixmap = pixmap.scaled(util_width, util_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        # Centrare
        x_pos = BLACKBOARD_RECT.left() + (BLACKBOARD_RECT.width() - pixmap.width()) / 2
        y_pos = BLACKBOARD_RECT.top() + (BLACKBOARD_RECT.height() - pixmap.height()) / 2
        
        label.setGeometry(int(x_pos), int(y_pos), pixmap.width(), pixmap.height())
        label.setPixmap(pixmap)
        label.setStyleSheet("background-color: transparent;")

        label.show()
        label.raise_()
        QApplication.processEvents()


    def _display_text_on_blackboard(self, display_string):
        """Afișează pe tablă folosind coordonate calibrate și MĂSURARE CORECTĂ a lățimii."""
        from PySide6.QtGui import QFontMetrics

        if self.calibration_mode:
            self._update_calibration_display()
            return
        
        self._clear_blackboard()
        
        # === COORDONATELE TALE CALIBRATE ===
        TABLA_X = 590
        TABLA_Y = 380
        TABLA_WIDTH = 360
        TABLA_HEIGHT = 150
        MARGINE = 10
        
        util_width = TABLA_WIDTH - (2 * MARGINE)
        util_height = TABLA_HEIGHT - (2 * MARGINE)
        
        items = [item.strip() for item in display_string.split(',')]
        if not items:
            return
        
        # Folosim un font fix, mare.
        font_size = 120 # Mărime fixă, dar mare
        font = self.chalk_font if self.chalk_font else QFont("Arial")
        font.setPointSize(font_size)
        metrics = QFontMetrics(font)

        # --- REPARAȚIA ESTE AICI: MĂSURĂM LĂȚIMEA REALĂ A FIECĂRUI ITEM ---
        item_widths = []
        total_width = 0
        spacing = 20
        
        for item_id in items:
            width = metrics.horizontalAdvance(item_id)
            item_widths.append(width)
            total_width += width
        
        if len(items) > 1:
            total_width += spacing * (len(items) - 1)
        # --- SFÂRȘIT BLOC DE MĂSURARE ---
        
        # Centrare pe baza lățimii reale
        start_x = TABLA_X + MARGINE + (util_width - total_width) / 2
        
        current_x = start_x
        for i, item_id in enumerate(items):
            if i >= len(self.blackboard_labels):
                break
            
            label = self.blackboard_labels[i]
            item_width = item_widths[i]
            item_height = metrics.height() # Înălțimea este aceeași pentru toate literele
            
            y_pos = TABLA_Y + MARGINE + (util_height - item_height) / 2

            # Folosim lățimea REALĂ, nu font_size
            label.setGeometry(int(current_x), int(y_pos), int(item_width), int(item_height))
            
            label.setText(item_id)
            label.setFont(font)
            label.setStyleSheet(f"color: white; font-weight: bold; background-color: transparent; font-family: '{self.chalkboard_font_family}';")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.show()
            label.raise_()
            
            current_x += item_width + spacing
            
        QApplication.processEvents()

    def _activate_calibration(self):
        """Activează modul de calibrare."""
        self.calibration_mode = True
        self.calibration_saved = []
        print("\n" + "="*60)
        print("🎯 MOD CALIBRARE ACTIVAT!")
        print("Mergi la școală și începe să muți punctul!")
        print("="*60)

    def keyPressEvent(self, event):
        """Detectează apăsări de taste pentru calibrare."""
        if not self.calibration_mode:
            return
        
        key = event.key()
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        
        if shift:
            if key == Qt.Key.Key_Left:
                self.calibration_point.setX(self.calibration_point.x() - 50)
            elif key == Qt.Key.Key_Right:
                self.calibration_point.setX(self.calibration_point.x() + 50)
            elif key == Qt.Key.Key_Up:
                self.calibration_point.setY(self.calibration_point.y() - 50)
            elif key == Qt.Key.Key_Down:
                self.calibration_point.setY(self.calibration_point.y() + 50)
        elif key == Qt.Key.Key_A:
            self.calibration_point.setX(self.calibration_point.x() - 10)
        elif key == Qt.Key.Key_D:
            self.calibration_point.setX(self.calibration_point.x() + 10)
        elif key == Qt.Key.Key_W:
            self.calibration_point.setY(self.calibration_point.y() - 10)
        elif key == Qt.Key.Key_S:
            self.calibration_point.setY(self.calibration_point.y() + 10)
        elif key == Qt.Key.Key_Left:
            self.calibration_point.setX(self.calibration_point.x() - 1)
        elif key == Qt.Key.Key_Right:
            self.calibration_point.setX(self.calibration_point.x() + 1)
        elif key == Qt.Key.Key_Up:
            self.calibration_point.setY(self.calibration_point.y() - 1)
        elif key == Qt.Key.Key_Down:
            self.calibration_point.setY(self.calibration_point.y() + 1)
        elif key == Qt.Key.Key_Space:
            coord = (self.calibration_point.x(), self.calibration_point.y())
            self.calibration_saved.append(coord)
            print("="*60)
            print(f"✅ COORDONATĂ SALVATĂ #{len(self.calibration_saved)}")
            print(f"   X = {coord[0]}, Y = {coord[1]}")
            print(f"   Total salvate: {len(self.calibration_saved)}/4")
            if len(self.calibration_saved) == 4:
                print("\n🎉 AI TOATE CELE 4 COORDONATE!")
                print(f"   1. Stânga-Sus:   {self.calibration_saved[0]}")
                print(f"   2. Dreapta-Sus:  {self.calibration_saved[1]}")
                print(f"   3. Stânga-Jos:   {self.calibration_saved[2]}")
                print(f"   4. Dreapta-Jos:  {self.calibration_saved[3]}")
            print("="*60)
        elif key == Qt.Key.Key_Escape:
            print("\n🛑 Ieșire din modul calibrare")
            self.calibration_mode = False
            self._clear_blackboard()
            return
        
        self._update_calibration_display()

    def _update_calibration_display(self):
        """Actualizează poziția punctului de calibrare."""
        if not self.calibration_mode:
            return
        
        label = self.blackboard_labels[0]
        x = self.calibration_point.x()
        y = self.calibration_point.y()
        
        label.setText("●")
        label.setStyleSheet("color: red; font-size: 50px; background-color: yellow;")
        label.setGeometry(x, y, 50, 50)
        label.show()
        label.raise_()
        
        # Print coordonate CLARE în consolă
        print(f"\n{'='*60}")
        print(f"📍 COORDONATE CURENTE:")
        print(f"   X = {x}")
        print(f"   Y = {y}")
        print(f"{'='*60}")

    def closeEvent(self, event):
        log_timestamp("=" * 60)
        log_timestamp("🛑 [APP] ÎNCHIDERE APLICAȚIE...")
            
        # ⭐ SALVARE GEOMETRIE FEREASTRĂ
        geom = self.geometry()
        self.config["window_geometry"] = {
            "x": geom.x(),
            "y": geom.y(),
            "width": geom.width(),
            "height": geom.height()
        }
        log_timestamp(f"🪟 [WINDOW] Salvez geometrie: {geom.x()}, {geom.y()}, {geom.width()}x{geom.height()}", "app")
            
        # ⭐ SALVARE CONFIG COMPLET
        save_config(self.config)
        self.stop_webcam()
        self.stop_continuous_voice()
            
        self.idle_timer.stop()
        self.sync_timer.stop()
        self.thinking_timer.stop()

        log_timestamp("🛑 [APP] Oprire animatoare...")
        for animator in self.all_animators:
            animator.stop()
            
        log_timestamp("🛑 [APP] Se așteaptă oprirea thread-urilor...")
            
        # ... (restul metodei cu wait() pentru thread-uri este corect) ...
        if self.webcam_thread and self.webcam_thread.isRunning():
            self.webcam_thread.quit()
            self.webcam_thread.wait(2000)

        if self.gemini_thread and self.gemini_thread.isRunning():
            self.gemini_thread.quit()
            self.gemini_thread.wait(2000)
                
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.quit()
            self.tts_thread.wait(2000)
                
        if self.voice_thread and self.voice_thread.isRunning():
            self.voice_thread.quit()
            self.voice_thread.wait(2000)
            
        log_timestamp("✅ [APP] Închidere finalizată.")
        event.accept()

    def on_tts_provider_changed(self, text):
        if "Google" in text:
            provider = "google"
        else:
            provider = "microsoft"
        
        self.config["tts_provider"] = provider
        save_config(self.config)
        log_timestamp(f"⚙️ [CONFIG] Furnizor TTS setat la: '{provider}'", "app")



# =================================================================================
# Punct de Intrare
# =================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🎭 TEATRU DIGITAL INTERACTIV - By Aarici Pogonici 🎭")
    print("=" * 80)
    
    cleanup_temp_files()
    app = QApplication(sys.argv)
    window = CharacterApp()
    window.show()
    sys.exit(app.exec())