import logging
import os
import asyncio
import glob
import shlex
import torch
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, CallbackQueryHandler, filters
from google import genai
from google.genai import types
from faster_whisper import WhisperModel

# --- ⚙️ CONFIGURATION ---
TG_TOKEN = os.getenv("TG_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not TG_TOKEN or not GEMINI_KEY:
    print("❌ ERROR: API Keys are missing!")
    exit()

# --- 🚀 GLOBAL AI MODELS (Load Once) ---
print("⏳ Loading AI Models... (This may take a moment)")

# 1. Initialize Gemini Client globally
GENAI_CLIENT = genai.Client(api_key=GEMINI_KEY)

# 2. Initialize Whisper globally to save RAM/Time
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
try:
    GLOBAL_WHISPER = WhisperModel("small", device=device, compute_type=compute_type)
    print(f"✅ Whisper loaded on {device}")
except Exception as e:
    print(f"❌ Whisper Load Failed: {e}")
    GLOBAL_WHISPER = None

# --- 📝 PROMPTS ---
DEFAULT_PROMPTS = {
    "burmese": """
You are a professional Burmese translator for video narration.
Task: Translate the input text to natural, spoken Burmese suitable for a narrator.

Guidelines:
1. **Natural Flow:** Do not translate word-for-word. Use natural sentence structures suitable for audio narration.
2. **Pronunciation:** For English names or loan words, use Burmese phonetics (e.g., "CEO" -> "စီအီးအို", not "CEO").
3. **Formal/Polite:** Avoid the word 'ပေါ့' (pout). Use professional endings.
4. **Format:** Keep the meaning accurate but prioritize how it sounds when read aloud.
    """,
    "rephrase": "Rephrase this English text to be more clear, natural, and reliable for a video script."
}

# Folders
for f in ["downloads", "temp"]:
    os.makedirs(f, exist_ok=True)

# In-Memory Database (Note: Reset on restart)
user_prefs = {}
user_modes = {}
chat_histories = {}

# --- 🛠️ HELPERS ---
def get_user_state(user_id):
    if user_id not in user_prefs:
        user_prefs[user_id] = {"transcribe_engine": "gemini", "custom_prompts": {}}
    return user_prefs[user_id]

def get_paths(user_id):
    # Using specific filenames prevents conflicts
    base = f"downloads/{user_id}"
    return {
        "input": f"{base}_input_video",  # No extension yet
        "audio": f"{base}_audio.mp3",
        "srt": f"{base}_subs.srt",
        "txt": f"{base}_transcript.txt",
        "trans_txt": f"{base}_translated.txt"
    }

def cleanup_files(user_id, keep_results=False):
    p = get_paths(user_id)
    files_to_remove = [p['input'], p['audio']]
    if not keep_results:
        files_to_remove.extend([p['srt'], p['txt'], p['trans_txt']])
        
    for f in files_to_remove:
        # Glob to catch file extensions for input video
        for found in glob.glob(f"{f}*"):
            try: os.remove(found)
            except: pass

async def download_video(url, output_path):
    """Async download using yt-dlp without blocking the bot."""
    # Safety: Do not use shell=True. Pass args as list.
    process = await asyncio.create_subprocess_exec(
        'yt-dlp', '--no-check-certificate', 
        '-f', 'bestaudio/best', 
        '-x', '--audio-format', 'mp3', 
        '-o', output_path, 
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return process.returncode == 0

async def extract_audio(input_path, output_path):
    """Async audio extraction using ffmpeg."""
    process = await asyncio.create_subprocess_exec(
        'ffmpeg', '-y', '-i', input_path, 
        '-vn', '-acodec', 'libmp3lame', '-q:a', '2', 
        output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.wait()
    return os.path.exists(output_path)

# --- 🧠 ENGINES ---

def run_whisper_sync(audio_path, srt_path, txt_path):
    """Runs on the pre-loaded global model."""
    if not GLOBAL_WHISPER:
        return "Whisper Error: Model not loaded."
        
    segments, _ = GLOBAL_WHISPER.transcribe(audio_path, beam_size=5)
    
    with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            srt.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            txt.write(f"{text} ")
    return "Whisper (Local)"

def run_gemini_transcribe_sync(audio_path, srt_path, txt_path):
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        prompt = "Transcribe this audio. Return ONLY the raw text. Do not add timestamps."
        
        response = GENAI_CLIENT.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                types.Content(parts=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
                    types.Part.from_text(text=prompt)
                ])
            ]
        )
        txt_content = response.text.strip() if response.text else "[No Speech]"
            
        with open(txt_path, "w", encoding="utf-8") as f: f.write(txt_content)
        with open(srt_path, "w", encoding="utf-8") as f: f.write("1\n00:00:00,000 --> 00:00:05,000\n[Gemini Transcript - No SRT data]")
        
        return "Gemini 2.0 Flash"
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def format_timestamp(seconds):
    import math
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{math.floor(seconds):02},{milliseconds:03}"

# --- 🧠 TRANSLATION ---
async def run_translate(user_id, prompt_text):
    p = get_paths(user_id)
    source = p['txt'] if os.path.exists(p['txt']) else (p['srt'] if os.path.exists(p['srt']) else None)
    
    if not source: return False, "❌ No text found."

    with open(source, "r", encoding="utf-8") as f: original_text = f.read()
    
    # Check if text is too long for one request (Gemini has a large context window, but good to keep in mind)
    full_prompt = f"{prompt_text}\n\nInput Text:\n{original_text}"

    try:
        # Running in executor to avoid blocking network calls on main loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: GENAI_CLIENT.models.generate_content(
            model='gemini-2.0-flash', contents=full_prompt
        ))
        
        translated = response.text.strip()
        with open(p['trans_txt'], "w", encoding="utf-8") as f: f.write(translated)
        return True, translated
    except Exception as e:
        return False, str(e)

# --- 🤖 HANDLERS (Selected Logic Updates) ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    
    # Simple clear keyboard
    keyboard = [
        [InlineKeyboardButton(f"🎙️ Engine: {state['transcribe_engine'].title()}", callback_data="toggle_transcribe")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings")]
    ]
    await update.message.reply_text(
        "👋 **Video AI Studio**\nSend a Video, Audio, or Link to start.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def process_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Unified handler for Files and URLs"""
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    state = get_user_state(user_id)
    
    # 1. Cleanup old run
    cleanup_files(user_id, keep_results=False)
    
    status_msg = await msg.reply_text("⏳ **Initializing...**")
    
    # 2. Get Input
    is_url = False
    if msg.text and "http" in msg.text:
        is_url = True
        await status_msg.edit_text("⏳ **Downloading from URL... (safe mode)**")
        success = await download_video(msg.text, p['audio']) # Direct to audio
        if not success:
            await status_msg.edit_text("❌ Download Failed. Check URL.")
            return
    elif msg.video or msg.audio or msg.document:
        await status_msg.edit_text("⏳ **Downloading File...**")
        file_obj = await (msg.video or msg.audio or msg.document).get_file()
        await file_obj.download_to_drive(p['input'])
        
        # Convert to MP3
        await status_msg.edit_text("⏳ **Converting Audio...**")
        success = await extract_audio(p['input'], p['audio'])
        if not success:
            await status_msg.edit_text("❌ Audio Extraction Failed.")
            return
    else:
        return

    # 3. Transcribe
    await status_msg.edit_text(f"🎙️ **Transcribing ({state['transcribe_engine']})...**")
    
    loop = asyncio.get_running_loop()
    if state['transcribe_engine'] == "gemini":
        engine_res = await loop.run_in_executor(None, run_gemini_transcribe_sync, p['audio'], p['srt'], p['txt'])
    else:
        engine_res = await loop.run_in_executor(None, run_whisper_sync, p['audio'], p['srt'], p['txt'])

    # 4. Result
    if os.path.exists(p['txt']) and os.path.getsize(p['txt']) > 0:
        await context.bot.send_document(
            chat_id=msg.chat_id, 
            document=open(p['txt'], "rb"), 
            caption=f"✅ **Transcript Ready** ({engine_res})"
        )
        
        # Immediate Action Keyboard
        keyboard = [
            [InlineKeyboardButton("🇲🇲 Translate to Burmese", callback_data="trans_burmese")],
            [InlineKeyboardButton("🇺🇸 Rephrase", callback_data="trans_rephrase")]
        ]
        await msg.reply_text("What next?", reply_markup=InlineKeyboardMarkup(keyboard))
        await status_msg.delete()
    else:
        await status_msg.edit_text("❌ Transcription produced empty result.")

# --- 🚀 BOILERPLATE STARTUP ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TG_TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler("start", start))
    
    # Message Handlers
    app.add_handler(MessageHandler(filters.TEXT & filters.Entity("url"), process_media)) # Handle URLs
    app.add_handler(MessageHandler(filters.VIDEO | filters.AUDIO | filters.Document.ALL, process_media)) # Handle Files
    
    # Callbacks (Keep your existing callback logic, it was fine)
    # app.add_handler(CallbackQueryHandler(callback_handler))
    
    print("🚀 Bot is live...")
    app.run_polling()
