import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import queue
import os
import time
from datetime import datetime

# --- AICI ESTE LINIA ESENȚIALĂ ---
import edge_tts # Asigură-te că această linie există!
# ------------------------------------

import asyncio
import pygame
from google.cloud import speech
import pyaudio
import google.generativeai as genai
from dotenv import load_dotenv

# --- Nu mai avem nevoie de acestea ---
# from google.cloud import texttospeech
# import sounddevice as sd
# import numpy as np

STREAMING_RATE = 16000
STREAMING_CHUNK = int(STREAMING_RATE / 10)

class GeminiVoiceChat:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat Vocal cu Gemini (Voce Edge-TTS)")
        self.root.geometry("700x750")
        self.root.configure(bg='#2c3e50')
        
        load_dotenv()
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.model = None
        self.is_recording = False
        
        self.response_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.tts_buffer = ""
        self.is_speaking = False

        self.audio_queue = queue.Queue()
        self.player_thread_running = False        
        
        self.tts_enabled = tk.BooleanVar(value=True)
        self.start_time = None
        self.first_chunk_time = None

        # --- NOU: Inițializăm mixer-ul Pygame pentru a putea reda sunet ---
        pygame.mixer.init()

        self.setup_ui()
        self.initialize_gemini()

    def setup_ui(self):
        # ... (UI-ul rămâne la fel) ...
        title_label = tk.Label(self.root, text="🎤 Chat Vocal cu Gemini (Streaming E2E)", font=("Arial", 20, "bold"), bg='#2c3e50', fg='white')
        title_label.pack(pady=20)
        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20, font=("Arial", 11), bg='#ecf1f1', fg='#2c3e50')
        self.chat_display.pack(padx=20, pady=10)
        self.chat_display.config(state=tk.DISABLED)
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=20)
        self.start_button = tk.Button(button_frame, text="▶ VORBEȘTE", command=self.start_recording, font=("Arial", 14, "bold"), bg='#27ae60', fg='white', width=15, height=2, cursor="hand2")
        self.start_button.grid(row=0, column=0, padx=10)
        self.stop_button = tk.Button(button_frame, text="⬛ STOP", command=self.stop_recording, font=("Arial", 14, "bold"), bg='#e74c3c', fg='white', width=15, height=2, cursor="hand2", state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10)
        separator = tk.Label(self.root, text="──────── sau ────────", font=("Arial", 10), bg='#2c3e50', fg='#95a5a6')
        separator.pack(pady=10)
        input_frame = tk.Frame(self.root, bg='#2c3e50')
        input_frame.pack(pady=10, padx=20, fill=tk.X)
        tk.Label(input_frame, text="✍️ Scrie mesajul:", font=("Arial", 10), bg='#2c3e50', fg='white').pack(anchor=tk.W, pady=(0, 5))
        entry_button_frame = tk.Frame(input_frame, bg='#2c3e50')
        entry_button_frame.pack(fill=tk.X)
        self.text_input = tk.Entry(entry_button_frame, font=("Arial", 12), bg='#ecf1f1', fg='#2c3e50', relief=tk.FLAT, borderwidth=2)
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.text_input.bind('<Return>', lambda e: self.send_text_message())
        self.send_button = tk.Button(entry_button_frame, text="📤 TRIMITE", command=self.send_text_message, font=("Arial", 11, "bold"), bg='#3498db', fg='white', width=12, cursor="hand2", relief=tk.FLAT)
        self.send_button.pack(side=tk.LEFT, padx=(10, 0), ipady=5)
        self.status_label = tk.Label(self.root, text="Gata de înregistrare sau scris", font=("Arial", 10), bg='#2c3e50', fg='#95a5a6')
        self.status_label.pack()
        tts_frame = tk.Frame(self.root, bg='#2c3e50')
        tts_frame.pack(pady=10)
        self.tts_checkbox = tk.Checkbutton(tts_frame, text="🔊 Activează vocea (TTS)", variable=self.tts_enabled, font=("Arial", 10, "bold"), bg='#2c3e50', fg='white', selectcolor='#34495e', activebackground='#2c3e50', activeforeground='white')
        self.tts_checkbox.pack()

    def initialize_gemini(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            messagebox.showerror("Eroare", "Cheia GOOGLE_API_KEY nu a fost găsită în .env!")
            self.root.destroy()
            return
        
        try:
            genai.configure(api_key=self.api_key)
            
            # --- CORECTAT DEFINITIV ---
            # Poți schimba aici modelul pentru a testa (ex: 'gemini-pro-latest')
            self.model_name_for_code = 'gemini-flash-lite-latest'
            
            self.model = genai.GenerativeModel(self.model_name_for_code)
            
            # --- Afișează numele corect ---
            print(f"✅ Conectat la Gemini cu modelul: {self.model_name_for_code}")
            self.add_message("Sistem", f"Conectat la Gemini cu modelul {self.model_name_for_code}! Poți începe.")
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la conectarea cu Gemini: {str(e)}")
            self.root.destroy()

    # ... (restul funcțiilor, inclusiv add_message, start_recording, etc.,
    # pot rămâne exact ca în versiunea anterioară pe care ți-am dat-o, V3,
    # deoarece logica de logging și streaming este deja implementată corect acolo) ...
    
    # Doar copiez restul codului din V3 pentru a fi complet
    def add_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}\n"
        tag = f"{sender}_{int(time.time()*1000)}"
        if sender == "Tu":
            self.chat_display.insert(tk.END, f"\n🗣️ {sender}: ", tag)
            self.chat_display.tag_config(tag, foreground="#2980b9", font=("Arial", 11, "bold"))
        elif sender == "Gemini":
            self.chat_display.insert(tk.END, f"\n🤖 {sender}: ", tag)
            self.chat_display.tag_config(tag, foreground="#8e44ad", font=("Arial", 11, "bold"))
        else:
            self.chat_display.insert(tk.END, f"\n⚙️ {sender}: ", tag)
            self.chat_display.tag_config(tag, foreground="#16a085", font=("Arial", 11, "bold"))
        self.chat_display.insert(tk.END, full_message)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def start_recording(self):
        self.is_recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="🔴 Ascult... Vorbește acum!", fg='#e74c3c')
        self.start_time = time.perf_counter()
        print(f"\n{'='*70}\n[START REC {self._ts()}] Aștept vorbire...\n{'='*70}")
        threading.Thread(target=self.listen_and_transcribe_thread, daemon=True).start()
        
    def stop_recording(self):
        self.is_recording = False
        
    def listen_and_transcribe_thread(self):
        try:
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=STREAMING_RATE, language_code="ro-RO", enable_automatic_punctuation=True)
            streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True, single_utterance=True)
            with MicrophoneStream(STREAMING_RATE, STREAMING_CHUNK) as stream:
                audio_generator = stream.generator()
                requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
                responses = client.streaming_recognize(streaming_config, requests)
                self.process_stt_responses(responses)
        except Exception as e:
            error_msg = f"Eroare STT: {e}"
            print(f"❌ {error_msg}")
            self.root.after(0, lambda: self.add_message("Sistem", error_msg))
        finally:
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.status_label.config(text="Gata de înregistrare sau scris", fg='#95a5a6'))

    def process_stt_responses(self, responses):
        for response in responses:
            if not self.is_recording:
                print(f"[STOP REC {self._ts()}] Înregistrare oprită de utilizator.")
                break
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        stt_end_time = time.perf_counter()
                        stt_duration = (stt_end_time - self.start_time) * 1000
                        print(f"✅ [STT FINAL {self._ts()}] Transcriere finală primită în {stt_duration:.0f}ms.")
                        print(f"   📝 Text: '{transcript}'")
                        self.root.after(0, lambda t=transcript: self.add_message("Tu", t))
                        self.root.after(0, lambda: self.status_label.config(text="⏳ Aștept răspunsul...", fg='#f39c12'))
                        self.stop_recording()
                        threading.Thread(target=self.get_gemini_response, args=(transcript,), daemon=True).start()
                        break
                    else:
                        self.root.after(0, lambda t=transcript: self.status_label.config(text=f"🗣️ Spui: {t}", fg='white'))
    
    def send_text_message(self):
        text = self.text_input.get().strip()
        if not text: return
        self.add_message("Tu", text)
        self.text_input.delete(0, tk.END)
        self.send_button.config(state=tk.DISABLED)
        self.text_input.config(state=tk.DISABLED)
        self.status_label.config(text="⏳ Aștept răspunsul...", fg='#f39c12')
        threading.Thread(target=self.get_gemini_response, args=(text, True), daemon=True).start()

    def get_gemini_response(self, text, is_text_input=False):
        try:
            if is_text_input:
                self.start_time = time.perf_counter()
            self.tts_buffer = ""
            self.start_tts_worker()
            gemini_start_time = time.perf_counter()
            gemini_latency_from_start = (gemini_start_time - self.start_time) * 1000
            print(f"\n🚀 [GEMINI REQ {self._ts()}] Trimit cerere la Gemini (Latență până aici: {gemini_latency_from_start:.0f}ms)")
            print(f"   📝 Întrebare: {text}")
            response_stream = self.model.generate_content(text, stream=True)
            self.root.after(0, lambda: self.chat_display.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.chat_display.insert(tk.END, f"\n🤖 Gemini: ", "gemini_tag"))
            self.root.after(0, lambda: self.chat_display.tag_config("gemini_tag", foreground="#8e44ad", font=("Arial", 11, "bold")))
            self.root.after(0, lambda: self.chat_display.config(state=tk.DISABLED))
            full_response = ""
            self.first_chunk_time = None # Resetăm pentru fiecare răspuns
            for chunk in response_stream:
                if chunk.text:
                    if self.first_chunk_time is None:
                        self.first_chunk_time = time.perf_counter()
                        time_to_first_token = (self.first_chunk_time - gemini_start_time) * 1000
                        time_to_first_from_start = (self.first_chunk_time - self.start_time) * 1000
                        print(f"⚡ [GEMINI CHUNK {self._ts()}] PRIMUL CHUNK primit.")
                        print(f"   ⏱️ Timp de la cerere Gemini (TTFT): {time_to_first_token:.0f}ms")
                        print(f"   ⏱️ Timp total de la START: {time_to_first_from_start:.0f}ms")
                    full_response += chunk.text
                    self.root.after(0, lambda t=chunk.text: self.append_streaming_text(t))
                    if self.tts_enabled.get():
                        self.speak_text_incremental(chunk.text)
            if self.tts_enabled.get():
                self.flush_tts_buffer()
            end_time = time.perf_counter()
            total_time = (end_time - self.start_time) * 1000
            print(f"✅ [GEMINI END {self._ts()}] Răspuns complet primit în {total_time:.0f}ms de la START.\n")
        except Exception as e:
            error_msg = f"Eroare Gemini: {e}"
            print(f"❌ {error_msg}")
            self.root.after(0, lambda: self.add_message("Sistem", error_msg))
        finally:
            if is_text_input:
                self.root.after(0, lambda: self.text_input.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_label.config(text="✅ Gata! Poți vorbi sau scrie din nou.", fg='#27ae60'))

    def append_streaming_text(self, text):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    
    def start_tts_worker(self):
        """Pornește firele de execuție pentru Generator și Player, dacă nu rulează deja."""
        if not self.tts_enabled.get(): return

        # Pornește Generatorul de fișiere audio (dacă nu rulează deja)
        if not self.is_speaking:
            self.is_speaking = True
            generator_thread = threading.Thread(target=self.tts_generator_worker, daemon=True)
            generator_thread.start()
            print("✅ [TTS Generator] Worker pornit.")

        # Pornește Player-ul audio (dacă nu rulează deja)
        if not self.player_thread_running:
            self.player_thread_running = True
            player_thread = threading.Thread(target=self.audio_player_worker, daemon=True)
            player_thread.start()
            print("✅ [TTS Player] Worker pornit.")

    def tts_generator_worker(self):
        """
        Rol: "Generatorul". Preia TEXT din `tts_queue`, generează un fișier audio
        și pune CALEA fișierului în `audio_queue`. NU redă sunetul.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            try:
                text_to_speak = self.tts_queue.get()
                if text_to_speak is None: break
                
                if text_to_speak.strip():
                    # Generează fișierul audio și pune calea lui în coada pentru player
                    loop.run_until_complete(self._generate_audio_file(text_to_speak))
            except Exception as e:
                print(f"❌ EROARE în Generatorul TTS: {e}")
            finally:
                self.tts_queue.task_done()

    async def _generate_audio_file(self, text):
        """
        Funcție asincronă care generează un fișier MP3 și adaugă calea sa în audio_queue.
        """
        tts_req_time = time.perf_counter()
        print(f"🔊 [TTS GEN {self._ts()}] Generez audio pentru: '{text[:50]}...'")
        
        VOICE = "ro-RO-EmilNeural"
        output_file = f"temp_speech_{int(time.time()*1000)}.mp3"

        try:
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(output_file)
            
            audio_gen_time = time.perf_counter()
            latency = (audio_gen_time - tts_req_time) * 1000
            print(f"   🎵 [TTS GEN {self._ts()}] Fișier '{output_file}' generat în {latency:.0f}ms. Se trimite la player.")
            
            # Adaugă calea fișierului în coada pentru player
            self.audio_queue.put(output_file)

        except Exception as e:
            print(f"❌ EROARE la generarea fișierului audio: {e}")
            # Dacă generarea eșuează, încercăm să ștergem fișierul parțial, dacă există
            if os.path.exists(output_file):
                os.remove(output_file)

    def audio_player_worker(self):
        """
        Rol: "Player-ul". Preia căi de fișiere audio din `audio_queue`,
        le redă secvențial și le șterge după redare.
        """
        while True:
            try:
                file_path = self.audio_queue.get()
                if file_path is None: break

                print(f"▶️  [TTS PLAY {self._ts()}] Redau fișierul '{file_path}'...")
                
                start_play_time = time.perf_counter()
                
                # Redarea propriu-zisă
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05) # Verificăm la fiecare 50ms dacă s-a terminat
                
                end_play_time = time.perf_counter()
                duration = (end_play_time - start_play_time) * 1000
                print(f"   ⏹️ [TTS PLAY {self._ts()}] Redare terminată. A durat {duration:.0f}ms.")

            except Exception as e:
                print(f"❌ EROARE în Player-ul audio: {e}")
            finally:
                # Curățarea fișierului
                pygame.mixer.music.unload()
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print(f"❌ Eroare la ștergerea fișierului '{file_path}': {e}")
                self.audio_queue.task_done()
    


    
    def speak_text_incremental(self, text_chunk):
        if not self.tts_enabled.get(): return
        self.tts_buffer += text_chunk
        while any(p in self.tts_buffer for p in ['. ', '? ', '! ', '\n']):
            sentence = ""
            for p in ['. ', '? ', '! ', '\n']:
                if p in self.tts_buffer:
                    parts = self.tts_buffer.split(p, 1)
                    sentence = parts[0] + p.strip()
                    self.tts_buffer = parts[1]
                    break
            if sentence:
                self.tts_queue.put(sentence)
    
    def flush_tts_buffer(self):
        if not self.tts_enabled.get(): return
        if self.tts_buffer.strip():
            self.tts_queue.put(self.tts_buffer)
            self.tts_buffer = ""

    def _ts(self):
        return datetime.now().strftime('%H:%M:%S.%f')[:-3]

class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate, self._chunk, self._buff, self.closed = rate, chunk, queue.Queue(), True
    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(format=pyaudio.paInt16, channels=1, rate=self._rate, input=True, frames_per_buffer=self._chunk, stream_callback=self._fill_buffer)
        self.closed = False
        return self
    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()
    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue
    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None: return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None: return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

if __name__ == "__main__":
    root = tk.Tk()
    app = GeminiVoiceChat(root)
    root.mainloop()