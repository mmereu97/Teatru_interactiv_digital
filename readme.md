\# 🦉 Aarici - Interactive AI Character Chat



<div align="center">



!\[Python](https://img.shields.io/badge/python-3.11+-blue.svg)

!\[PySide6](https://img.shields.io/badge/PySide6-6.0+-green.svg)

!\[License](https://img.shields.io/badge/license-MIT-blue.svg)

!\[Status](https://img.shields.io/badge/status-active-success.svg)



\*\*O aplicație interactivă de chat AI cu personaje animate, recunoaștere vocală și lip-sync în timp real\*\*



\[Features](#-features) • \[Instalare](#-instalare) • \[Utilizare](#-utilizare) • \[Configurare](#️-configurare) • \[Documentație](#-documentație)



</div>



---



\## 📖 Despre Proiect


![Screenshot aplicație](capture.png)


\*\*Aarici\*\* este o aplicație desktop interactivă care aduce la viață personaje AI animate cu personalități distincte. Folosind tehnologii avansate de AI, recunoaștere vocală și sinteză vocală, aplicația oferă o experiență de conversație naturală și captivantă.



\### 🎭 Personaje Principale



\- \*\*🦉 Prof. Cucuvel Bufnițovici\*\* - Profesor înțelept de matematică și logică

\- \*\*🐱 Rina\*\* - Pisică jucăușă și prietenă veselă



---



\## ✨ Features



\### 🎤 Interacțiune Vocală

\- \*\*Recunoaștere vocală continuă\*\* cu Silero VAD (Voice Activity Detection)

\- \*\*Text-to-Speech streaming\*\* cu voici românești naturale (Azure Neural TTS)

\- \*\*Lip-sync în timp real\*\* sincronizat cu audio-ul generat

\- \*\*Identificare vorbitor\*\* bazată pe profilele vocale (SpeechBrain)

\- \*\*Detectare automată a limbii\*\* (română/engleză)



\### 🤖 AI \& Conversație

\- \*\*Gemini AI\*\* pentru conversații naturale și inteligente

\- \*\*Memorie pe termen scurt\*\* - personajele își amintesc contextul conversației

\- \*\*Sistem de intenții\*\* - clasificare automată a cererilor utilizatorului

\- \*\*Răspuns la emoji\*\* - reacții animate la emoji-uri trimise de utilizator

\- \*\*Mod profesor interactiv\*\* cu curriculum de învățare



\### 🎨 Animații \& Vizual

\- \*\*Animații 2D stratificate\*\* (ochi, gură, corp, accesorii)

\- \*\*Expresii emoționale\*\* (fericit, trist, surprins, etc.)

\- \*\*Animații de clipit și respirație\*\* automată

\- \*\*Sistem de privire\*\* - personajele urmăresc cu privirea

\- \*\*Tranziții între scene\*\* (acasă, școală, etc.)

\- \*\*Subtitrări în timp real\*\* cu scroll automat



\### 📹 Computer Vision

\- \*\*Identificare facială\*\* cu Google Gemini Vision

\- \*\*Detectare persoane multiple\*\* în cadru

\- \*\*Fallback automat\*\* la identificare vocală



\### 🎓 Mod Învățare

\- \*\*Curriculum structurat pe niveluri\*\* (matematică, logică)

\- \*\*Feedback personalizat\*\* și încurajări

\- \*\*Tracking progres\*\* și statistici

\- \*\*Teleportare automată\*\* la școală



---



\## 🚀 Instalare



\### Prerequisite



\- \*\*Python 3.11+\*\*

\- \*\*Conda\*\* (recomandat pentru gestionarea mediului)

\- \*\*Webcam\*\* (opțional, pentru identificare facială)

\- \*\*Microfon\*\*



\### Pași de Instalare



1\. \*\*Clonează repository-ul\*\*

```bash

git clone https://github.com/your-username/Aarici.git

cd Aarici

```



2\. \*\*Creează mediul Conda\*\*

```bash

conda create -n Aarici\_env python=3.11

conda activate Aarici\_env

```



3\. \*\*Instalează dependințele\*\*

```bash

pip install -r requirements.txt

```



4\. \*\*Configurează API keys\*\*

Creează un fișier `.env` în directorul rădăcină:

```env

GEMINI\_API\_KEY=your\_gemini\_api\_key\_here

GOOGLE\_CLOUD\_API\_KEY=your\_google\_cloud\_key\_here  # Pentru Speech-to-Text

```



5\. \*\*Rulează aplicația\*\*

```bash

python main\_app.py

```



---



\## 📦 Dependințe Principale



```

PySide6>=6.6.0              # UI Framework

google-generativeai>=0.3.0  # Gemini AI

edge-tts>=6.1.0             # Text-to-Speech

pygame>=2.5.0               # Audio playback

opencv-python>=4.8.0        # Computer vision

silero-vad>=4.0.0           # Voice Activity Detection

speechbrain>=0.5.0          # Speaker identification

librosa>=0.10.0             # Audio processing

Pillow>=10.0.0              # Image processing

```



> \*\*Notă:\*\* Lista completă se găsește în `requirements.txt`



---



\## ⚙️ Configurare



\### Fișiere de Configurare



\#### `config.json`

Setări generale ale aplicației:

```json

{

&nbsp; "voice\_enabled": true,

&nbsp; "voice\_language": "ro-RO",

&nbsp; "speech\_threshold": 0.5,

&nbsp; "max\_speech\_duration": 15,

&nbsp; "window\_geometry": {...}

}

```



\#### `family.json`

Definirea personajelor și scenelor:

```json

{

&nbsp; "characters": {

&nbsp;   "cucuvel\_owl": {

&nbsp;     "display\_name": "Prof. Cucuvel Bufnițovici",

&nbsp;     "personality": "Profesor înțelept...",

&nbsp;     "voice\_id": "ro-RO-EmilNeural"

&nbsp;   }

&nbsp; },

&nbsp; "scenes": {...}

}

```



\#### `curriculum\_tier\_XXX.json`

Curriculum de învățare pe niveluri:

\- `tier\_001.json` - Numite Simple (1-10)

\- `tier\_002.json` - Adunări Simple

\- etc.



---



\## 🎮 Utilizare



\### Comenzi Vocale



| Comandă | Acțiune |

|---------|---------|

| "Salut" / "Bună" | Salut personaj |

| "Schimbă personajul" | Comută între personaje |

| "Vreau să învăț" | Pornește modul profesor |

| "Gata cu învățarea" | Oprește modul profesor |

| "Repetă" | Repetă ultimul răspuns |

| "Oprește-te" / "Stop" | Oprește vorbirea curentă |



\### Interfață UI



\- \*\*Buton Microfon\*\* 🎤 - Activează/dezactivează ascultarea

\- \*\*Buton Repeat\*\* 🔁 - Repetă ultimul răspuns

\- \*\*Emoji Panel\*\* 😊 - Trimite emoji către personaj

\- \*\*Semafor\*\* 🚦 - Indică starea (verde=gata, roșu=vorbește, galben=gândește)



---



\## 📁 Structura Proiectului



```

Aarici/

├── main\_app.py              # Aplicația principală

├── config.json              # Configurare generală

├── family.json              # Definire personaje și scene

├── requirements.txt         # Dependințe Python

├── README.md               # Acest fișier

│

├── Backgrounds/            # Imagini fundal pentru scene

│   ├── acasa.png

│   └── scoala.png

│

├── Characters/             # Assets personaje

│   ├── cucuvel\_owl/

│   │   ├── body.png

│   │   ├── eyes\_happy.png

│   │   └── mouth\_A.png

│   └── rina\_cat/

│       └── ...

│

├── Curriculum/             # Fișiere curriculum învățare

│   ├── curriculum\_tier\_001.json

│   └── ...

│

├── Logs/                   # Log-uri conversații

│   └── conversatie\_\*.txt

│

└── voice\_profiles/         # Profile vocale salvate

&nbsp;   └── \*.pt

```



---



\## 🛠️ Tehnologii Folosite



\### AI \& Machine Learning

\- \*\*Google Gemini\*\* - Conversație și identificare facială

\- \*\*Silero VAD\*\* - Detectare vorbire

\- \*\*SpeechBrain\*\* - Identificare vorbitor



\### Audio \& Speech

\- \*\*Google Cloud Speech-to-Text\*\* - Recunoaștere vocală

\- \*\*Edge TTS\*\* - Sinteză vocală (Azure Neural)

\- \*\*Pygame\*\* - Redare audio



\### UI \& Graphics

\- \*\*PySide6 (Qt)\*\* - Framework interfață

\- \*\*OpenCV\*\* - Procesare video

\- \*\*Pillow\*\* - Manipulare imagini



\### Utilities

\- \*\*Librosa\*\* - Analiză audio

\- \*\*Pydub\*\* - Procesare audio

\- \*\*Unidecode\*\* - Normalizare text



---



\## 🐛 Debugging \& Logs



Aplicația generează log-uri detaliate în:

\- \*\*Console output\*\* - Log-uri în timp real

\- \*\*Fișiere log\*\* în directorul `Logs/`



Configurarea nivelului de logging în `LOG\_CONFIG` (în `main\_app.py`):

```python

LOG\_CONFIG = {

&nbsp;   "app": True,          # Mesaje generale

&nbsp;   "tts": True,          # Text-to-Speech

&nbsp;   "vad": True,          # Voice Activity Detection

&nbsp;   "gemini\_debug": True, # AI requests/responses

&nbsp;   # ... etc

}

```



---



\## 🤝 Contributing



Contribuțiile sunt binevenite! Pentru a contribui:



1\. Fork-uiește proiectul

2\. Creează un branch pentru feature-ul tău (`git checkout -b feature/AmazingFeature`)

3\. Commit modificările (`git commit -m 'Add some AmazingFeature'`)

4\. Push pe branch (`git push origin feature/AmazingFeature`)

5\. Deschide un Pull Request



---



\## 📝 License



Acest proiect este licențiat sub \*\*MIT License\*\* - vezi fișierul \[LICENSE](LICENSE) pentru detalii.



---



\## 🙏 Credits \& Acknowledgments



\### Tehnologii \& APIs

\- \[Google Gemini](https://ai.google.dev/) - AI conversațional

\- \[Azure Neural TTS](https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/) - Voici naturale

\- \[Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection

\- \[SpeechBrain](https://speechbrain.github.io/) - Speaker Recognition



\### Dezvoltare

\- Dezvoltat cu ❤️ folosind \*\*Python\*\* și \*\*Qt\*\*

\- Inspirat de dorința de a face AI-ul mai accesibil și prietenos



---



\## 📧 Contact



Pentru întrebări, sugestii sau raportare bug-uri:

\- \*\*GitHub Issues\*\*: \[Aarici Issues](https://github.com/your-username/Aarici/issues)

\- \*\*Email\*\*: your.email@example.com



---



<div align="center">



\*\*Dacă îți place proiectul, lasă un ⭐ pe GitHub!\*\*



Made with 🦉 and 🐱 in România



</div>

