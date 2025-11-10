# tool_-_creeaza_audio_fillers_multilang.py
# Script de unică folosință pentru a genera TOATE replicile audio de umplutură
# pentru Franceză, Germană, Italiană, Spaniolă, Rusă și Greacă.

import asyncio
import edge_tts
import os

# ==============================================================================
# --- CONFIGURARE CENTRALĂ (PRE-POPULATĂ) ---
# ==============================================================================

# Definim DOAR limbile pe care dorim să le generăm acum
LANGUAGE_CONFIG = {
    "fr": {"voice": "fr-FR-DeniseNeural", "output_folder": os.path.join("characters", "rina_cat", "audio_replici_fr")},
    "de": {"voice": "de-DE-KatjaNeural", "output_folder": os.path.join("characters", "rina_cat", "audio_replici_de")},
    "it": {"voice": "it-IT-ElsaNeural", "output_folder": os.path.join("characters", "rina_cat", "audio_replici_it")},
    "es": {"voice": "es-ES-ElviraNeural", "output_folder": os.path.join("characters", "rina_cat", "audio_replici_es")},
    "ru": {"voice": "ru-RU-SvetlanaNeural", "output_folder": os.path.join("characters", "rina_cat", "audio_replici_ru")},
    "el": {"voice": "el-GR-NestorasNeural", "output_folder": os.path.join("characters", "rina_cat", "audio_replici_el")},
}

# Listele extinse cu cel puțin 15 fraze pentru fiecare limbă
PHRASES = {
    "fr": [
        "Un instant, s'il vous plaît.", "Laissez-moi réfléchir un moment.", "C'est une question très intéressante.",
        "Donnez-moi une seconde pour y penser.", "Hmm, je vois.", "Je dois considérer cela attentivement.",
        "Permettez-moi d'organiser mes pensées.", "Juste un petit moment.", "Voyons voir...",
        "Je veux vous donner la meilleure réponse possible.", "Cela mérite réflexion.", "Un court instant.",
        "Je rassemble mes idées.", "Je pèse mes mots.", "Laissez-moi consulter ma mémoire."
    ],
    "de": [
        "Einen Moment, bitte.", "Lassen Sie mich kurz nachdenken.", "Das ist eine sehr interessante Frage.",
        "Geben Sie mir eine Sekunde, um darüber nachzudenken.", "Hmm, ich verstehe.", "Ich muss das sorgfältig abwägen.",
        "Erlauben Sie mir, meine Gedanken zu ordnen.", "Nur einen kleinen Augenblick.", "Mal sehen...",
        "Ich möchte Ihnen die bestmögliche Antwort geben.", "Das erfordert Überlegung.", "Einen kurzen Moment.",
        "Ich sammle meine Gedanken.", "Ich wäge meine Worte.", "Lassen Sie mich in meinem Gedächtnis nachsehen."
    ],
    "it": [
        "Un momento, per favore.", "Mi lasci pensare un attimo.", "Questa è una domanda molto interessante.",
        "Mi dia un secondo per rifletterci.", "Hmm, capisco.", "Devo considerare la cosa con attenzione.",
        "Mi permetta di organizzare i miei pensieri.", "Solo un piccolo istante.", "Vediamo un po'...",
        "Voglio darle la migliore risposta possibile.", "Questo merita una riflessione.", "Un breve momento.",
        "Sto raccogliendo le idee.", "Sto pesando le mie parole.", "Mi lasci consultare la mia memoria."
    ],
    "es": [
        "Un momento, por favor.", "Déjeme pensar un momento.", "Esa es una pregunta muy interesante.",
        "Deme un segundo para pensarlo.", "Hmm, ya veo.", "Necesito considerar esto cuidadosamente.",
        "Permítame organizar mis pensamientos.", "Solo un pequeño instante.", "A ver...",
        "Quiero darle la mejor respuesta posible.", "Eso requiere reflexión.", "Un breve momento.",
        "Estoy reuniendo mis ideas.", "Estoy sopesando mis palabras.", "Déjeme consultar mi memoria."
    ],
    "ru": [
        "Один момент, пожалуйста.", "Позвольте мне немного подумать.", "Это очень интересный вопрос.",
        "Дайте мне секунду, чтобы обдумать это.", "Хм, понятно.", "Мне нужно тщательно это взвесить.",
        "Позвольте мне привести мысли в порядок.", "Буквально один миг.", "Так, посмотрим...",
        "Я хочу дать вам наилучший возможный ответ.", "Это требует размышления.", "Короткий миг.",
        "Я собираю свои мысли.", "Я взвешиваю свои слова.", "Позвольте мне заглянуть в свою память."
    ],
    "el": [
        "Ένα λεπτό, παρακαλώ.", "Αφήστε με να σκεφτώ λίγο.", "Αυτή είναι μια πολύ ενδιαφέρουσα ερώτηση.",
        "Δώστε μου ένα δευτερόλεπτο να το σκεφτώ.", "Χμμ, καταλαβαίνω.", "Πρέπει να το εξετάσω προσεκτικά.",
        "Επιτρέψτε μου να οργανώσω τις σκέψεις μου.", "Μόνο μια μικρή στιγμή.", "Για να δούμε...",
        "Θέλω να σας δώσω την καλύτερη δυνατή απάντηση.", "Αυτό απαιτεί σκέψη.", "Μια σύντομη στιγμή.",
        "Συγκεντρώνω τις σκέψεις μου.", "Ζυγίζω τα λόγια μου.", "Αφήστε με να συμβουλευτώ τη μνήμη μου."
    ]
}

# ==============================================================================
# --- LOGICA SCRIPTULUI (NU NECESITĂ MODIFICĂRI) ---
# ==============================================================================

async def generate_speech(text, output_filepath, voice):
    """Generează un fișier audio dintr-un text, folosind o voce specifică."""
    try:
        print(f"  🔄 Generating: '{text[:50]}...'")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_filepath)
        print(f"  ✅ Saved: '{os.path.basename(output_filepath)}'")
        return True
    except Exception as e:
        print(f"  ❌ ERROR generating '{text[:50]}...': {e}")
        return False

async def main():
    """Funcția principală a scriptului."""
    print("=" * 70)
    print("🎙️  One-Time Multi-Language Audio Generator (FR, DE, IT, ES, RU, EL) 🎙️")
    print("=" * 70)
    
    total_files_generated = 0
    
    # Procesează toate limbile configurate
    for lang_code, config in LANGUAGE_CONFIG.items():
        print("-" * 50)
        print(f"🔥 Processing language: [{lang_code.upper()}]")

        phrases_list = PHRASES.get(lang_code, [])
        
        # Creează folderul de output dacă nu există
        output_folder = config["output_folder"]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Folder '{output_folder}' created.")

        voice = config["voice"]
        print(f"Using voice: {voice}")

        # Generează fișierele audio pentru fiecare frază
        count = 0
        for phrase in phrases_list:
            filename_base = f"audio{count+1:02d}"
            filepath = os.path.join(output_folder, f"{filename_base}.mp3")
            
            if await generate_speech(phrase, filepath, voice):
                total_files_generated += 1
            await asyncio.sleep(0.5)  # Pauză pentru a nu suprasolicita API-ul
            count += 1
            
    print("\n" + "=" * 70)
    print(f"🎉 All tasks completed. Generated a total of {total_files_generated} audio files. 🎉")
    print("🎉 Structura de foldere și fișierele audio sunt gata de utilizare. 🎉")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
