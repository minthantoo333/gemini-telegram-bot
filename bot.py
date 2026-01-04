import logging
import os
import asyncio
import glob
import shlex
import torch
import pysrt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, CallbackQueryHandler, filters
from google import genai
from google.genai import types
from faster_whisper import WhisperModel

# --- ⚙️ CONFIGURATION ---
TG_TOKEN = os.getenv("TG_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not TG_TOKEN or not GEMINI_KEY:
    print("❌ ERROR: API Keys are missing! Set TG_TOKEN and GEMINI_KEY.")
    exit()

# --- 🚀 GLOBAL AI MODELS (Load Once) ---
print("⏳ Loading AI Models... (This may take a moment)")

# 1. Initialize Gemini Client
GENAI_CLIENT = genai.Client(api_key=GEMINI_KEY)

# 2. Initialize Whisper Globally (Saves RAM/Time)
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
Role: Expert Burmese Translator & Video Narrator.
Task: Translate the input text into natural, spoken Burmese suitable for audio narration.

Strict Guidelines:
1. **Natural Flow:** Do not translate word-for-word. Make it sound like a storyteller.
2. **Endings:** STRICTLY AVOID 'ပေါ့' (pout). Use professional endings (e.g., ပါ, မယ်, တယ်).
3. **English Terms:**
   - NO brackets for English words.
   - Transliterate terms phonetically (e.g., "CEO" -> "စီအီးအို", "Virus" -> "ဗိုင်းရပ်စ်").
4. **Numbers:** Keep digits (1, 2, 3) for clarity.
5. **Output:** Return ONLY the Burmese text.
    """,
    "rephrase": "Rephrase this English text to be more clear, natural, and reliable for a video script."
}

# Folders
for f in ["downloads", "temp"]:
    os.makedirs(f, exist_ok=True)

# User State
user_prefs = {}

# --- 🛠️ HELPER FUNCTIONS ---

def get_paths(user_id):
    base = f"downloads/{user_id}"
    return {
        "input": f"{base}_input_video",
        "audio": f"{base}_audio.mp3",
        "srt": f"{base}_subs.srt",
        "trans_srt": f"{base}_translated.srt",
        "txt": f"{base}_transcript.txt",
        "trans_txt": f"{base}_translated.txt"
    }

def cleanup_files(user_id, keep_results=False):
    p = get_paths(user_id)
    files = [p['input'], p['audio']]
    if not keep_results:
        files.extend([p['srt'], p['trans_srt'], p['txt'], p['trans_txt']])
        
    for f in files:
        for found in glob.glob(f"{f}*"):
            try: os.remove(found)
            except: pass

def format_timestamp(seconds):
    import math
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{math.floor(seconds):02},{milliseconds:03}"

# --- ⚡ ASYNC PROCESSING ENGINES ---

async def download_video(url, output_path):
    """Async safe download"""
    process = await asyncio.create_subprocess_exec(
        'yt-dlp', '--no-check-certificate', 
        '-f', 'bestaudio/best', 
        '-x', '--audio-format', 'mp3', 
        '-o', output_path, 
        url,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()
    return process.returncode == 0

async def extract_audio(input_path, output_path):
    """Async audio conversion"""
    process = await asyncio.create_subprocess_exec(
        'ffmpeg', '-y', '-i', input_path, 
        '-vn', '-acodec', 'libmp3lame', '-q:a', '2', 
        output_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await process.wait()
    return os.path.exists(output_path)

def run_whisper_sync(audio_path, srt_path, txt_path):
    """Runs on global model in thread pool"""
    if not GLOBAL_WHISPER: return "Error"
    
    segments, _ = GLOBAL_WHISPER.transcribe(audio_path, beam_size=5)
    
    with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            
            srt.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            txt.write(f"{text} ")
    return "Whisper (Small)"

# --- 🧠 TRANSLATION ENGINES ---

async def translate_text_file(user_id, prompt_key):
    p = get_paths(user_id)
    if not os.path.exists(p['txt']): return False, "No transcript found."
    
    with open(p['txt'], "r", encoding="utf-8") as f: text = f.read()
    
    full_prompt = f"{DEFAULT_PROMPTS[prompt_key]}\n\nInput:\n{text}"
    
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: GENAI_CLIENT.models.generate_content(
            model='gemini-2.0-flash', contents=full_prompt
        ))
        translated = response.text.strip()
        with open(p['trans_txt'], "w", encoding="utf-8") as f: f.write(translated)
        return True, "Success"
    except Exception as e:
        return False, str(e)

async def translate_srt_file(user_id, prompt_key):
    """Intelligent SRT Translation: Keeps timestamps, translates text only."""
    p = get_paths(user_id)
    if not os.path.exists(p['srt']): return False, "No SRT file found."

    try:
        subs = pysrt.open(p['srt'], encoding='utf-8')
        
        # Prepare batch text
        original_texts = [sub.text.replace("\n", " ") for sub in subs]
        full_text_block = "\n<SEP>\n".join(original_texts)

        system_prompt = f"""
        {DEFAULT_PROMPTS[prompt_key]}
        
        INSTRUCTIONS:
        1. The input is subtitle lines separated by '<SEP>'.
        2. Translate each line individually.
        3. Output MUST be separated by '<SEP>' exactly like input.
        4. Do NOT include timestamps in output.
        """

        # AI Call
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: GENAI_CLIENT.models.generate_content(
            model='gemini-2.0-flash', 
            contents=f"{system_prompt}\n\nINPUT DATA:\n{full_text_block}"
        ))

        translated_block = response.text.strip()
        translated_lines = translated_block.split("<SEP>")

        # Reconstruct SRT
        new_subs = pysrt.SubRipFile()
        for i, sub in enumerate(subs):
            new_text = translated_lines[i].strip() if i < len(translated_lines) else sub.text
            # Create new item with ORIGINAL timing
            item = pysrt.SubRipItem(index=i+1, start=sub.start, end=sub.end, text=new_text)
            new_subs.append(item)

        new_subs.save(p['trans_srt'], encoding='utf-8')
        return True, "Success"
    except Exception as e:
        return False, str(e)

# --- 🤖 HANDLERS ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 **Video AI Studio**\n\n"
        "Send a **Video**, **Audio**, or **YouTube Link**.\n"
        "I will transcribe it and let you choose the format (.SRT or .TXT)."
    )

async def process_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    
    # 1. Cleanup & Setup
    cleanup_files(user_id, keep_results=False)
    status = await msg.reply_text("⏳ **Initializing...**")
    
    try:
        # 2. Download / Extract
        if msg.text and "http" in msg.text:
            await status.edit_text("⏳ **Downloading from URL...**")
            if not await download_video(msg.text, p['audio']):
                raise Exception("Download failed.")
        elif msg.video or msg.audio or msg.document:
            await status.edit_text("⏳ **Processing File...**")
            file_obj = await (msg.video or msg.audio or msg.document).get_file()
            await file_obj.download_to_drive(p['input'])
            await extract_audio(p['input'], p['audio'])
        else:
            return

        # 3. Transcribe
        await status.edit_text("🎙️ **Transcribing (Whisper)...**")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, run_whisper_sync, p['audio'], p['srt'], p['txt'])

        # 4. Success Menu
        if os.path.exists(p['txt']):
            keyboard = [
                [InlineKeyboardButton("📄 Get .TXT", callback_data="dl_txt"),
                 InlineKeyboardButton("🎬 Get .SRT", callback_data="dl_srt")],
                [InlineKeyboardButton("🌍 Translate File", callback_data="menu_translate")]
            ]
            await status.delete()
            await msg.reply_text("✅ **Done! Choose action:**", reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            raise Exception("Transcription returned empty.")

    except Exception as e:
        await status.edit_text(f"❌ Error: {str(e)}")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    p = get_paths(user_id)
    data = query.data
    
    await query.answer()

    # --- DOWNLOADS ---
    if data == "dl_txt":
        await context.bot.send_document(user_id, document=open(p['txt'], "rb"), caption="📄 Transcript")
    
    elif data == "dl_srt":
        await context.bot.send_document(user_id, document=open(p['srt'], "rb"), caption="🎬 Subtitles")

    # --- TRANSLATE MENU ---
    elif data == "menu_translate":
        keyboard = [
            [InlineKeyboardButton("🇲🇲 Burmese (.TXT)", callback_data="trans_burmese_txt")],
            [InlineKeyboardButton("🇲🇲 Burmese (.SRT)", callback_data="trans_burmese_srt")],
            [InlineKeyboardButton("🇺🇸 Rephrase (.TXT)", callback_data="trans_rephrase_txt")],
            [InlineKeyboardButton("🔙 Cancel", callback_data="cancel")]
        ]
        await query.message.edit_text("🌍 **Select Format:**", reply_markup=InlineKeyboardMarkup(keyboard))

    # --- EXECUTE TRANSLATION ---
    elif data.startswith("trans_"):
        parts = data.split("_") # trans, [lang], [ext]
        lang = parts[1]
        ext = parts[2]
        
        await query.message.edit_text(f"⏳ **Translating to {lang.title()}...**")
        
        if ext == "srt":
            success, msg = await translate_srt_file(user_id, lang)
            result_file = p['trans_srt']
        else:
            success, msg = await translate_text_file(user_id, lang)
            result_file = p['trans_txt']
            
        if success:
            await context.bot.send_document(
                chat_id=user_id, 
                document=open(result_file, "rb"), 
                caption=f"✅ **Translated {ext.upper()}**"
            )
            await query.message.delete()
        else:
            await query.message.edit_text(f"❌ Failed: {msg}")

    elif data == "cancel":
        await query.message.delete()

# --- 🚀 STARTUP ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TG_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & filters.Entity("url"), process_media))
    app.add_handler(MessageHandler(filters.VIDEO | filters.AUDIO | filters.Document.ALL, process_media))
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    print("🚀 Bot is running...")
    app.run_polling()
