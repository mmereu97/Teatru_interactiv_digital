# managers/scene_manager.py

import json
import os
import time
from PySide6.QtCore import QObject, Signal

# --- Logging cu variabilă globală ---
START_TIME = time.time()
_LOG_CONFIG = None

def set_log_config(config):
    """Setează configurația de logging"""
    global _LOG_CONFIG
    _LOG_CONFIG = config

def log_timestamp(message, category="scene"):
    """Logging cu filtrare pentru scene manager."""
    global _LOG_CONFIG
    
    if _LOG_CONFIG is None:
        _LOG_CONFIG = {"scene": True}
    
    if _LOG_CONFIG.get(category, True):
        elapsed = time.time() - START_TIME
        print(f"[{elapsed:8.3f}s] {message}")

class SceneManager(QObject):
    scene_changed = Signal(str, dict)

    def __init__(self, config_path="scenes/scene_configs.json"):
        super().__init__()
        self.config_path = config_path
        self.scenes = self._load_scene_configs()
        self.current_scene_id = None
        
        if not self.scenes:
            log_timestamp(f"⚠️ [SCENE MANAGER] Avertisment: Nu s-au putut încărca scenele din '{os.path.abspath(self.config_path)}'.", "scene")

    def _load_scene_configs(self):
        """Încarcă datele despre scene din fișierul JSON."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                scenes_data = json.load(f)
                log_timestamp(f"✅ [SCENE MANAGER] Au fost încărcate {len(scenes_data)} scene din '{os.path.basename(self.config_path)}'.", "scene")
                return scenes_data
        except FileNotFoundError:
            log_timestamp(f"❌ [EROARE CRITICĂ] Fișierul de scene '{self.config_path}' nu a fost găsit!", "scene")
            return {}
        except Exception as e:
            log_timestamp(f"❌ [EROARE CRITICĂ] La încărcarea scenelor: {e}", "scene")
            return {}
            
    def set_scene(self, scene_id):
        """
        Setează o nouă scenă ca fiind activă și emite un semnal.
        """
        if scene_id in self.scenes and scene_id != self.current_scene_id:
            self.current_scene_id = scene_id
            scene_data = self.scenes[scene_id]
            log_timestamp(f"🌆 [SCENE MANAGER] Schimbare scenă la: '{scene_id}'", "scene")
            self.scene_changed.emit(scene_id, scene_data)
        elif scene_id not in self.scenes:
            log_timestamp(f"⚠️ [SCENE MANAGER] Scena cu ID-ul '{scene_id}' nu există în configurație.", "scene")

    def get_current_scene_data(self):
        """Returnează datele complete pentru scena curentă."""
        if self.current_scene_id:
            return self.scenes.get(self.current_scene_id)
        return None

    def get_scene_data(self, scene_id):
        """Returnează datele pentru o scenă specifică."""
        return self.scenes.get(scene_id)

    def get_native_characters(self, scene_id):
        """
        Returnează lista de personaje native pentru o scenă.
        """
        scene_data = self.get_scene_data(scene_id)
        if scene_data:
            return scene_data.get("native_characters", [])
        return []