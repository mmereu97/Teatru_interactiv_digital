# tool_-_creeaza_audio_rina.py

import asyncio
import edge_tts
import os

# --- CONFIGURARE PENTRU RINA ---
# Vocea specifică Rinei (feminină, engleză britanică)
VOICE = "en-GB-SoniaNeural"
# Un folder separat pentru replicile Rinei, pentru a le păstra organizate
OUTPUT_FOLDER = os.path.join("characters", "rina_cat", "audio_replici_en")

async def generate_speech(text, output_filename):
    """Generează un fișier audio dintr-un text."""
    try:
        print(f"🔄 Generating: '{text}'...")
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_filename)
        print(f"✅ Finished: '{output_filename}' saved successfully.")
        return True
    except Exception as e:
        print(f"❌ ERROR generating '{text}': {e}")
        return False

async def main():
    """Funcția principală a scriptului."""
    print("=" * 60)
    print("🎙️  Audio Snippet Generator for Rina the Cat 🎙️")
    print("=" * 60)
    print(f"Voice used: {VOICE}")
    print(f"Files will be saved in folder: '{OUTPUT_FOLDER}'")
    print("Enter the desired text. Type 'exit' to quit.")
    print("-" * 60)

    # Creează folderul de output dacă nu există
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Folder '{OUTPUT_FOLDER}' created.")

    while True:
        # 1. Cere textul de la utilizator (în engleză)
        text_to_speak = input("\n➡️ Text to convert (in English): ")
        
        if text_to_speak.lower() == 'exit':
            break
            
        if not text_to_speak:
            print("⚠️ Please enter some text.")
            continue

        # 2. Cere numele fișierului
        default_filename = text_to_speak.lower().replace(" ", "_").replace("?", "").replace("!", "").replace(".", "")[:20]
        output_filename_base = input(f"➡️ Filename (without extension) [default: {default_filename}]: ")
        
        if not output_filename_base:
            output_filename_base = default_filename
            
        output_filepath = os.path.join(OUTPUT_FOLDER, f"{output_filename_base}.mp3")

        # 3. Verifică dacă fișierul există deja
        if os.path.exists(output_filepath):
            overwrite = input(f"⚠️ File '{output_filepath}' already exists. Overwrite? (y/n): ").lower()
            if overwrite != 'y':
                print("Skipped.")
                continue

        # 4. Generează fișierul audio
        await generate_speech(text_to_speak, output_filepath)

    print("\nGoodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")