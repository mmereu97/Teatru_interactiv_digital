# characters/base_character.py

import json
import os
import random
import math
from PySide6.QtCore import QPoint
import time

START_TIME = time.time()
_LOG_CONFIG = None

def set_log_config(config):
    """Setează configurația de logging"""
    global _LOG_CONFIG
    _LOG_CONFIG = config

def log_timestamp(message, category="character"):
    """Logging cu filtrare pentru base character."""
    global _LOG_CONFIG
    
    if _LOG_CONFIG is None:
        _LOG_CONFIG = {"character": True}
    
    if _LOG_CONFIG.get(category, True):
        elapsed = time.time() - START_TIME
        print(f"[{elapsed:8.3f}s] {message}")
# --- SFÂRȘIT BLOC NOU ---

class BaseCharacter:
    def __init__(self, character_folder_path):
        """
        Inițializează un personaj din folderul specificat.
        
        Args:
            character_folder_path (str): Calea către folderul personajului
                                         (ex: 'characters/cucuvel_owl')
        """
        self.id = os.path.basename(character_folder_path)
        self.root_path = character_folder_path
        
        config_path = os.path.join(self.root_path, "config.json")
        
        # Atribute de bază
        self.display_name, self.language, self.voice_id = "N/A", "N/A", "N/A"
        self.assets_path, self.prompt_path = "", ""
        self.scene_configs, self.filler_sounds, self.components = {}, [], {}
        self.filler_sounds_folder_prefix = None # <-- ADAUGAȚI ACEASTĂ LINIE
        self.animators = []  # Lista de animatoare active
        self.current_scene_id = None
        
        # Atribute pentru mobilitate
        self.can_leave_home = True  # Default: poate pleca din casa lui
        self.can_be_summoned = True  # Default: poate fi chemat magic
        self.home_scene = None  # Scena "de acasă" (unde locuiește)
        
        # Unghiul de animație pentru breathing (folosit de BreathingAnimator)
        self.animation_angle = 0
        
        # ⭐ ADĂUGAT: Stare pentru gaze tracking (folosit de BlinkingAnimator)
        self.current_gaze_direction = "centru"  # Valori: "centru", "stanga", "dreapta"
        
        # Încarcă configurația din config.json
        self._load_config(config_path)

    def _load_config(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            self.display_name = config_data.get("display_name", self.id)
            self.language = config_data.get("language")
            self.voice_id = config_data.get("voice_id")
            
            # ⭐ ÎNCĂRCARE ATRIBUTE NOI
            self.can_leave_home = config_data.get("can_leave_home", True)
            self.can_be_summoned = config_data.get("can_be_summoned", True)
            self.home_scene = config_data.get("home_scene", None)
            
            self.assets_path = os.path.join(self.root_path, "assets")
            self.prompt_path = os.path.join(self.root_path, "prompts", "personality.txt")
            
            raw_components = config_data.get("components", {})
            self.components = raw_components
            
            if "part_offsets" in self.components:
                for part, offset_list in self.components["part_offsets"].items():
                    if len(offset_list) == 2:
                        self.components["part_offsets"][part] = QPoint(offset_list[0], offset_list[1])

            # --- BLOC MODIFICAT PENTRU FILLERE MULTILINGVE ---
            self.filler_sounds_folder_prefix = config_data.get("filler_sounds_folder_prefix")
            if self.filler_sounds_folder_prefix:
                # La pornire, încarcă sunetele pentru limba default a personajului
                initial_lang_code = self.language.split('-')[0]
                initial_folder_path = os.path.join(self.root_path, f"{self.filler_sounds_folder_prefix}{initial_lang_code}")
                self._load_filler_sounds(initial_folder_path)
            # ---------------------------------------------------

            raw_scene_configs = config_data.get("scene_configs", {})
            for scene_id, config in raw_scene_configs.items():
                pos_list = config.get("pos")
                if pos_list and len(pos_list) == 2:
                    config["pos"] = QPoint(pos_list[0], pos_list[1])
                    self.scene_configs[scene_id] = config

            # Salvăm configurația animatoarelor pentru mai târziu
            self.animator_config = config_data.get("animator_config", [])
            
            # ⭐ LOG PENTRU DEBUGGING
            log_timestamp(f"✅ [CHAR LOAD] '{self.id}': can_leave={self.can_leave_home}, can_summon={self.can_be_summoned}, home={self.home_scene}", "character")

        except Exception as e:
            log_timestamp(f"❌ [CHAR LOADER] EROARE la încărcarea config.json pentru '{self.id}': {e}", "character")
            
    def _load_filler_sounds(self, folder_path):
        self.filler_sounds.clear()
        if os.path.exists(folder_path):
            try:
                self.filler_sounds = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".mp3")]
            except Exception as e:
                log_timestamp(f"❌ [FILLER] Eroare la scanarea '{folder_path}': {e}", "filler")

    def get_prompt_content(self):
        try:
            with open(self.prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return f"Ești {self.display_name}."

    def get_config_for_scene(self, scene_id):
        return self.scene_configs.get(scene_id)

    def get_random_filler_sound(self):
        return random.choice(self.filler_sounds) if self.filler_sounds else None


    def set_language(self, language_code, voice_id):
        """Actualizează dinamic limba, vocea și calea către prompt-ul personajului."""
        try:
            new_prompt_path = os.path.join(self.root_path, "prompts", f"personality_{language_code}.txt")
            if os.path.exists(new_prompt_path):
                self.language = language_code
                self.voice_id = voice_id
                self.prompt_path = new_prompt_path
                log_timestamp(f"✅ [CHAR UPDATE] Limba pentru '{self.id}' a fost schimbată în '{language_code}'. Voce: {voice_id}", "character")

                # --- BLOC NOU: Reîncărcare sunete de umplutură ---
                if self.filler_sounds_folder_prefix:
                    new_folder_path = os.path.join(self.root_path, f"{self.filler_sounds_folder_prefix}{language_code}")
                    self._load_filler_sounds(new_folder_path)
                    log_timestamp(f"✅ [FILLER] Sunete de umplutură reîncărcate din: {new_folder_path}", "filler")
                # --- SFÂRȘIT BLOC NOU ---

                return True
            else:
                log_timestamp(f"⚠️ [CHAR UPDATE] Nu am găsit fișierul prompt: {new_prompt_path}", "character")
                return False
        except Exception as e:
            log_timestamp(f"❌ [CHAR UPDATE] Eroare la schimbarea limbii pentru '{self.id}': {e}", "character")
            return False

    def setup_animators(self, layers, dpi_scaler=None):
        """Crează și pornește animatoarele conform configurației."""
        log_timestamp(f"🎬 [ANIM] Setup animatoare pentru '{self.id}'...", "animator")
        
        try:
            from characters.animators import ANIMATOR_REGISTRY
        except ImportError:
            log_timestamp(f"❌ [ANIM] Nu pot importa ANIMATOR_REGISTRY!", "app")
            return
        
        self.stop_animators()
        
        for animator_type in self.animator_config:
            if animator_type in ANIMATOR_REGISTRY:
                animator_class = ANIMATOR_REGISTRY[animator_type]
                animator = animator_class(self, layers, dpi_scaler)
                self.animators.append(animator)
                animator.start()
                log_timestamp(f"  ✅ Animator '{animator_type}' creat și pornit", "animator")
            else:
                log_timestamp(f"  ⚠️ Tip de animator necunoscut: '{animator_type}'", "animator")

    def stop_animators(self):
        """Oprește toate animatoarele active."""
        if not self.animators:
            return
        for animator in self.animators:
            animator.stop()
        self.animators.clear()
        log_timestamp(f"🛑 [ANIM] Animatoare oprite pentru '{self.id}'", "animator")

    def update(self, layers, current_scene_id):
        """Metoda update apelată de timer-ul principal."""
        self.current_scene_id = current_scene_id
        # Animatoarele își rulează propriile update-uri prin timer-ele lor
        # Această metodă poate fi extinsă pentru alte actualizări
        pass