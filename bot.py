import logging
import os
import asyncio
import glob
import shlex
import torch
import pysrt
import re
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

# --- 🚀 GLOBAL AI MODELS ---
print("⏳ Loading AI Models...")

# 1. Gemini
GENAI_CLIENT = genai.Client(api_key=GEMINI_KEY)

# 2. Whisper (Loaded but optional if user prefers Gemini)
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    GLOBAL_WHISPER = WhisperModel("small", device=device, compute_type="float16" if device=="cuda" else "int8")
    print(f"✅ Whisper loaded on {device}")
except:
    GLOBAL_WHISPER = None
    print("⚠️ Whisper failed to load. Gemini will be used.")

# --- 📝 PROMPTS & DEFAULTS ---
DEFAULT_PROMPTS = {
    "burmese": "Role: Professional Narrator. Translate to natural spoken Burmese. No 'pout'. Phonetic English.",
}

# --- 💾 USER SETTINGS ---
# Stores: {'engine': 'gemini'|'whisper', 'format': 'ask'|'srt'|'txt'}
user_prefs = {}

def get_user_prefs(user_id):
    if user_id not in user_prefs:
        user_prefs[user_id] = {
            "engine": "gemini",  # Default to Gemini
            "format": "ask"      # Default to Ask every time
        }
    return user_prefs[user_id]

# --- 🛠️ FILES & HELPERS ---
def get_paths(user_id):
    base = f"downloads/{user_id}"
    return {
        "input": f"{base}_input",
        "audio": f"{base}_audio.mp3",
        "srt": f"{base}.srt",
        "txt": f"{base}.txt",
        "trans_srt": f"{base}_trans.srt",
        "trans_txt": f"{base}_trans.txt"
    }

def cleanup_files(user_id):
    p = get_paths(user_id)
    for f in glob.glob(f"downloads/{user_id}*"):
        try: os.remove(f)
        except: pass

async def download_media(msg, p):
    """Smart downloader for URL or File"""
    if msg.text and "http" in msg.text:
        # URL Download
        cmd = [
            'yt-dlp', '--no-check-certificate', '-f', 'bestaudio/best',
            '-x', '--audio-format', 'mp3', '-o', p['audio'], msg.text
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
    else:
        # File Download
        file_obj = await (msg.video or msg.audio or msg.document).get_file()
        await file_obj.download_to_drive(p['input'])
        # Convert
        cmd = ['ffmpeg', '-y', '-i', p['input'], '-vn', '-acodec', 'libmp3lame', '-q:a', '2', p['audio']]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.wait()

    return os.path.exists(p['audio'])

# --- 🧠 TRANSCRIPTION ENGINES ---

def run_whisper_sync(audio_path, srt_path, txt_path):
    if not GLOBAL_WHISPER: return "Whisper Error"
    segments, _ = GLOBAL_WHISPER.transcribe(audio_path, beam_size=5)
    
    with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            text = seg.text.strip()
            srt.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            txt.write(f"{text} ")
    return "Whisper (Local)"

def run_gemini_sync(audio_path, srt_path, txt_path):
    """Prompts Gemini to create an SRT directly."""
    try:
        with open(audio_path, "rb") as f: audio_bytes = f.read()
        
        # Prompt for SRT format
        prompt = "Transcribe this audio. Output the result strictly in SRT (SubRip) format with timestamps."
        
        response = GENAI_CLIENT.models.generate_content(
            model='gemini-2.0-flash',
            contents=[types.Content(parts=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
                types.Part.from_text(text=prompt)
            ])]
        )
        content = response.text.strip()
        
        # Save SRT
        with open(srt_path, "w", encoding="utf-8") as f: f.write(content)
        
        # Extract Text from SRT (Simple regex cleanup)
        clean_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
        clean_text = clean_text.replace('\n\n', ' ').strip()
        with open(txt_path, "w", encoding="utf-8") as f: f.write(clean_text)
        
        return "Gemini 2.0 Flash"
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Error"

def format_timestamp(seconds):
    import math
    h = math.floor(seconds / 3600); seconds %= 3600
    m = math.floor(seconds / 60); seconds %= 60
    return f"{h:02}:{m:02}:{math.floor(seconds):02},{round((seconds%1)*1000):03}"

# --- 🌍 TRANSLATION ---
# (Kept simple for brevity, same logic as before)
async def translate_file(user_id, ext, prompt_key="burmese"):
    p = get_paths(user_id)
    src = p['srt'] if ext == 'srt' else p['txt']
    out = p['trans_srt'] if ext == 'srt' else p['trans_txt']
    
    if not os.path.exists(src): return False
    
    # Simple logic: If SRT, we need pysrt. If TXT, direct prompt.
    # For this snippet, I'll use the robust Pysrt method for SRTs
    try:
        if ext == 'srt':
            subs = pysrt.open(src, encoding='utf-8')
            texts = [s.text.replace('\n', ' ') for s in subs]
            block = "\n<SEP>\n".join(texts)
            
            res = await asyncio.get_running_loop().run_in_executor(None, lambda: GENAI_CLIENT.models.generate_content(
                model='gemini-2.0-flash',
                contents=f"{DEFAULT_PROMPTS[prompt_key]}\n\nInput (Split by <SEP>):\n{block}"
            ))
            
            trans_lines = res.text.strip().split("<SEP>")
            for i, s in enumerate(subs):
                if i < len(trans_lines): s.text = trans_lines[i].strip()
            subs.save(out, encoding='utf-8')
        else:
            with open(src, 'r') as f: text = f.read()
            res = await asyncio.get_running_loop().run_in_executor(None, lambda: GENAI_CLIENT.models.generate_content(
                model='gemini-2.0-flash', contents=f"{DEFAULT_PROMPTS[prompt_key]}\n\n{text}"
            ))
            with open(out, 'w') as f: f.write(res.text)
            
        return True
    except Exception as e:
        return False

# --- 🎮 HANDLERS ---

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prefs = get_user_prefs(user_id)
    
    # Engine Button Text
    e_icon = "🧠" if prefs['engine'] == 'gemini' else "🎙️"
    e_text = f"Engine: {prefs['engine'].title()} {e_icon}"
    
    # Format Button Text
    f_text = f"Auto-Format: {prefs['format'].upper() if prefs['format'] != 'ask' else 'Ask Me'}"
    
    keyboard = [
        [InlineKeyboardButton(e_text, callback_data="set_engine")],
        [InlineKeyboardButton(f_text, callback_data="set_format")],
        [InlineKeyboardButton("❌ Close", callback_data="close")]
    ]
    
    await update.message.reply_text("⚙️ **Settings**", reply_markup=InlineKeyboardMarkup(keyboard))

async def process_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    prefs = get_user_prefs(user_id)
    p = get_paths(user_id)
    
    cleanup_files(user_id)
    status = await msg.reply_text("⏳ **Processing...**")
    
    # 1. Download
    if not await download_media(msg, p):
        await status.edit_text("❌ Download Failed.")
        return

    # 2. Transcribe (Based on Engine Setting)
    await status.edit_text(f"📝 **Transcribing with {prefs['engine'].title()}...**")
    loop = asyncio.get_running_loop()
    
    if prefs['engine'] == "gemini":
        eng_res = await loop.run_in_executor(None, run_gemini_sync, p['audio'], p['srt'], p['txt'])
    else:
        eng_res = await loop.run_in_executor(None, run_whisper_sync, p['audio'], p['srt'], p['txt'])
        
    if not os.path.exists(p['srt']):
        await status.edit_text("❌ Transcription failed.")
        return

    # 3. Handle Output (Based on Format Setting)
    await status.delete()
    
    async def send_file(ext):
        fpath = p['srt'] if ext == 'srt' else p['txt']
        await context.bot.send_document(user_id, document=open(fpath, 'rb'), caption=f"✅ {ext.upper()} ({eng_res})")
        
    # Auto-Send logic
    if prefs['format'] == 'srt':
        await send_file('srt')
        await show_translate_menu(update, context) # Optional: Offer translate after auto-send
    elif prefs['format'] == 'txt':
        await send_file('txt')
        await show_translate_menu(update, context)
    else:
        # Ask User
        keyboard = [
            [InlineKeyboardButton("📄 TXT", callback_data="dl_txt"), InlineKeyboardButton("🎬 SRT", callback_data="dl_srt")],
            [InlineKeyboardButton("🌍 Translate", callback_data="menu_trans")]
        ]
        await msg.reply_text(f"✅ **Done ({eng_res})**\nChoose format:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_translate_menu(update, context):
    keyboard = [[InlineKeyboardButton("🌍 Translate File", callback_data="menu_trans")]]
    await context.bot.send_message(update.effective_chat.id, "Need translation?", reply_markup=InlineKeyboardMarkup(keyboard))

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    prefs = get_user_prefs(user_id)
    p = get_paths(user_id)
    data = query.data
    
    # --- SETTINGS TOGGLES ---
    if data == "set_engine":
        prefs['engine'] = "whisper" if prefs['engine'] == "gemini" else "gemini"
        await settings_command(query, context) # Refresh menu
        return
        
    if data == "set_format":
        # Cycle: Ask -> SRT -> TXT -> Ask
        modes = ["ask", "srt", "txt"]
        curr_idx = modes.index(prefs['format'])
        prefs['format'] = modes[(curr_idx + 1) % 3]
        await settings_command(query, context)
        return

    if data == "close":
        await query.message.delete()
        return

    # --- FILE ACTIONS ---
    if data == "dl_txt":
        await context.bot.send_document(user_id, document=open(p['txt'], "rb"), caption="📄 Transcript")
    elif data == "dl_srt":
        await context.bot.send_document(user_id, document=open(p['srt'], "rb"), caption="🎬 Subtitles")
        
    # --- TRANSLATION ---
    elif data == "menu_trans":
        kb = [
            [InlineKeyboardButton("🇲🇲 Burmese (.SRT)", callback_data="tr_bur_srt")],
            [InlineKeyboardButton("🇲🇲 Burmese (.TXT)", callback_data="tr_bur_txt")]
        ]
        await query.message.edit_text("Select Format to Translate:", reply_markup=InlineKeyboardMarkup(kb))
        
    elif data.startswith("tr_"):
        parts = data.split("_") # tr, bur, srt
        ext = parts[2]
        
        await query.message.edit_text("⏳ **Translating...**")
        success = await translate_file(user_id, ext)
        
        if success:
            outfile = p['trans_srt'] if ext == 'srt' else p['trans_txt']
            await context.bot.send_document(user_id, document=open(outfile, 'rb'), caption=f"✅ Translated {ext.upper()}")
            await query.message.delete()
        else:
            await query.message.edit_text("❌ Failed.")

# --- 🚀 RUN ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text("👋 Send a file! Use /settings to configure.")))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(MessageHandler(filters.TEXT & filters.Entity("url"), process_media))
    app.add_handler(MessageHandler(filters.VIDEO | filters.AUDIO | filters.Document.ALL, process_media))
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    print("🚀 Bot Running...")
    app.run_polling()
