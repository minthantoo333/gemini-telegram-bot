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

# --- 🚀 GLOBAL AI MODELS (Load Once = Faster) ---
print("⏳ Loading AI Models...")
GENAI_CLIENT = genai.Client(api_key=GEMINI_KEY)

# Load Whisper once to avoid reloading it for every user (Saves RAM)
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    print(f"   ...Loading Whisper on {device}...")
    GLOBAL_WHISPER = WhisperModel("small", device=device, compute_type="float16" if device=="cuda" else "int8")
    print(f"✅ Whisper loaded on {device}")
except Exception as e:
    GLOBAL_WHISPER = None
    print(f"⚠️ Whisper failed to load: {e}. Gemini will be used as default.")

# --- 💾 STATE MANAGEMENT ---
user_prefs = {}     # Settings: engine, format
user_modes = {}     # Modes: 'chat' (HeyGemini), None (Normal)
chat_histories = {} # Gemini Chat History

def get_prefs(user_id):
    if user_id not in user_prefs:
        user_prefs[user_id] = {"engine": "gemini", "format": "ask"}
    return user_prefs[user_id]

# --- 🛠️ HELPER FUNCTIONS ---
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
    # Remove all files matching the user pattern
    for f in glob.glob(f"downloads/{user_id}*"):
        try: os.remove(f)
        except: pass

def clean_gemini_srt_output(raw_text):
    """Ensures Gemini output is valid SRT by stripping Markdown and fixing timestamps."""
    clean = re.sub(r"```\w*\n", "", raw_text).replace("```", "")
    # Fix timestamps (00:00:00.000 -> 00:00:00,000)
    clean = re.sub(r'(\d{2}:\d{2}:\d{2})\.(\d{3})', r'\1,\2', clean)
    return clean.strip()

async def download_media(msg, p):
    """Smart and SAFE download for URL or File."""
    try:
        if msg.text and "http" in msg.text:
            # Secure download (no shell=True)
            cmd = ['yt-dlp', '--no-check-certificate', '-f', 'bestaudio/best', '-x', '--audio-format', 'mp3', '-o', p['audio'], msg.text]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()
        else:
            file_obj = await (msg.video or msg.audio or msg.document).get_file()
            await file_obj.download_to_drive(p['input'])
            # Convert to MP3
            cmd = ['ffmpeg', '-y', '-i', p['input'], '-vn', '-acodec', 'libmp3lame', '-q:a', '2', p['audio']]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.wait()
            
        return os.path.exists(p['audio'])
    except Exception as e:
        print(f"Download Error: {e}")
        return False

# --- 🧠 ENGINES ---

def run_whisper_sync(audio_path, srt_path, txt_path):
    if not GLOBAL_WHISPER: return "Whisper Error (Not Loaded)"
    try:
        segments, _ = GLOBAL_WHISPER.transcribe(audio_path, beam_size=5)
        with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
            for i, seg in enumerate(segments, 1):
                t = f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}"
                srt.write(f"{i}\n{t}\n{seg.text.strip()}\n\n")
                txt.write(f"{seg.text.strip()} ")
        return "Whisper"
    except Exception as e:
        return f"Error: {str(e)}"

def run_gemini_sync(audio_path, srt_path, txt_path):
    try:
        with open(audio_path, "rb") as f: audio_bytes = f.read()
        
        prompt = """
        Transcribe audio to SRT format.
        Rules:
        1. Output ONLY SRT. No intro/outro text.
        2. Timestamps must be: 00:00:00,000 --> 00:00:00,000
        """
        
        response = GENAI_CLIENT.models.generate_content(
            model='gemini-2.0-flash',
            contents=[types.Content(parts=[types.Part.from_bytes(audio_bytes, "audio/mp3"), types.Part.from_text(prompt)])]
        )
        
        clean_content = clean_gemini_srt_output(response.text.strip())
        
        # Validation: If it fails to look like SRT, save as text to avoid empty file
        if "-->" not in clean_content:
            with open(txt_path, "w", encoding="utf-8") as f: f.write(clean_content)
            # Create a dummy SRT so the bot doesn't crash
            with open(srt_path, "w", encoding="utf-8") as f: 
                f.write("1\n00:00:00,000 --> 00:00:05,000\n(Gemini failed to format SRT, see .txt file)")
            return "Gemini (Raw Text)"
            
        with open(srt_path, "w", encoding="utf-8") as f: f.write(clean_content)
        
        # Create TXT from SRT
        clean_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', clean_content)
        with open(txt_path, "w", encoding="utf-8") as f: f.write(clean_text.replace('\n\n', ' ').strip())
        
        return "Gemini"
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Error"

def format_timestamp(s):
    import math
    h=math.floor(s/3600); s%=3600; m=math.floor(s/60); s%=60
    return f"{h:02}:{m:02}:{math.floor(s):02},{round((s%1)*1000):03}"

# --- 🌍 TRANSLATION & CHAT ---

async def translate_file(user_id, ext):
    p = get_paths(user_id)
    src, out = (p['srt'], p['trans_srt']) if ext == 'srt' else (p['txt'], p['trans_txt'])
    
    if not os.path.exists(src): return False, "File not found. Upload first."
    
    # --- PROMPTS ---
    prompt = """
    Role: Professional Video Narrator (Burmese).
    
    Guidelines:
    1. **Style:** Natural narrator flow. Not stiff.
    2. **Forbidden:** NEVER use 'ပေါ့' (pout). Use 'ပါ', 'တယ်', 'မယ်'.
    3. **Loan Words:** Write English abbreviations phonetically (e.g. CIA -> စီအိုင်အေ).
    4. **Format:** Keep strictly to the input format (SRT or Text).
    """
    
    try:
        # LOGIC: If SRT, split by lines to preserve timestamps
        if ext == 'srt':
            subs = pysrt.open(src, encoding='utf-8')
            texts = [s.text.replace('\n', ' ') for s in subs]
            block = "\n<SEP>\n".join(texts)
            
            res = await asyncio.get_running_loop().run_in_executor(None, lambda: GENAI_CLIENT.models.generate_content(
                model='gemini-2.0-flash', 
                contents=f"{prompt}\n\nINSTRUCTION: Translate these lines. Output must be separated by <SEP>.\n\nDATA:\n{block}"
            ))
            
            lines = res.text.strip().split("<SEP>")
            for i, s in enumerate(subs):
                if i < len(lines): s.text = lines[i].strip()
            subs.save(out, encoding='utf-8')
            
        # LOGIC: If TXT, just translate the whole block
        else:
            with open(src, 'r') as f: text = f.read()
            res = await asyncio.get_running_loop().run_in_executor(None, lambda: GENAI_CLIENT.models.generate_content(
                model='gemini-2.0-flash', contents=f"{prompt}\n\nInput Text:\n{text}"
            ))
            with open(out, 'w') as f: f.write(res.text)
            
        return True, "Success"
    except Exception as e:
        return False, str(e)

async def run_chat(user_id, text):
    if user_id not in chat_histories: chat_histories[user_id] = []
    
    try:
        chat = GENAI_CLIENT.chats.create(model='gemini-2.0-flash', history=chat_histories[user_id])
        response = chat.send_message(text)
        chat_histories[user_id] = chat.history # Save history
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

# --- 🎮 COMMAND HANDLERS ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = None # Reset mode
    await update.message.reply_text(
        "👋 **Video AI Studio**\n\n"
        "1️⃣ Send **Video/Audio/Link** to Transcribe.\n"
        "2️⃣ Use `/heygemini` to Chat.\n"
        "3️⃣ Use `/settings` to configure.\n"
        "4️⃣ Use `/clearall` to reset data."
    )

async def clearall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cleanup_files(user_id)
    if user_id in chat_histories: del chat_histories[user_id]
    user_modes[user_id] = None
    await update.message.reply_text("🧹 **Memory & Files Cleared!**")

async def heygemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = "chat"
    await update.message.reply_text("🤖 **Gemini Chat Mode: ON**\nType anything to chat.\nUse `/exit` or `/start` to stop.")

async def exit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = None
    await update.message.reply_text("❌ **Chat Mode Exited.**")

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    p = get_paths(user_id)
    
    # Check if files exist
    if not os.path.exists(p['txt']) and not os.path.exists(p['srt']):
        await update.message.reply_text("⚠️ **No file found!**\nPlease upload a video or audio first.")
        return

    kb = [[InlineKeyboardButton("🇲🇲 Burmese (.SRT)", callback_data="tr_srt"),
           InlineKeyboardButton("🇲🇲 Burmese (.TXT)", callback_data="tr_txt")]]
    await update.message.reply_text("🌍 **Select Translation Format:**", reply_markup=InlineKeyboardMarkup(kb))

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prefs = get_prefs(user_id)
    
    # Simple Toggle UI
    eng = f"Engine: {prefs['engine'].upper()}"
    fmt = f"Format: {prefs['format'].upper()}"
    
    kb = [
        [InlineKeyboardButton(eng, callback_data="toggle_engine")],
        [InlineKeyboardButton(fmt, callback_data="toggle_format")],
        [InlineKeyboardButton("✅ Close", callback_data="close_settings")]
    ]
    await update.message.reply_text("⚙️ **Settings**", reply_markup=InlineKeyboardMarkup(kb))

# --- 📨 MESSAGE HANDLER (Router) ---

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    text = msg.text
    
    # 1. Check Chat Mode
    if user_modes.get(user_id) == "chat":
        await context.bot.send_chat_action(msg.chat_id, "typing")
        response = await run_chat(user_id, text)
        await msg.reply_text(response, parse_mode="Markdown")
        return

    # 2. Check for URL
    if text and "http" in text:
        await process_media(update, context)
        return
        
    # 3. Default Fallback
    await msg.reply_text("🤖 I am ready. Send a file or use `/heygemini` to chat.")

async def process_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    prefs = get_prefs(user_id)
    p = get_paths(user_id)
    
    cleanup_files(user_id)
    status = await msg.reply_text("⏳ **Downloading...**")
    
    if not await download_media(msg, p):
        await status.edit_text("❌ Download Failed.")
        return

    await status.edit_text(f"📝 **Transcribing ({prefs['engine']})...**")
    loop = asyncio.get_running_loop()
    runner = run_gemini_sync if prefs['engine'] == "gemini" else run_whisper_sync
    
    # Run in thread to not block bot
    res_name = await loop.run_in_executor(None, runner, p['audio'], p['srt'], p['txt'])
    
    await status.delete()
    
    # Send Result Helper
    async def send(ext):
        f = p['srt'] if ext == 'srt' else p['txt']
        if os.path.exists(f):
            await context.bot.send_document(user_id, open(f, 'rb'), caption=f"✅ {ext.upper()} ({res_name})")

    # Auto-Format Decision
    if prefs['format'] == 'srt':
        await send('srt')
        await translate_command(update, context) # Prompt translation
    elif prefs['format'] == 'txt':
        await send('txt')
        await translate_command(update, context)
    else:
        kb = [[InlineKeyboardButton("📄 TXT", callback_data="dl_txt"), InlineKeyboardButton("🎬 SRT", callback_data="dl_srt")],
              [InlineKeyboardButton("🌍 Translate", callback_data="menu_trans")]]
        await msg.reply_text(f"✅ **Done! Choose:**", reply_markup=InlineKeyboardMarkup(kb))

# --- 🔘 CALLBACK HANDLER ---

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    prefs = get_prefs(user_id)
    p = get_paths(user_id)
    data = query.data
    
    # Settings Logic
    if data == "toggle_engine":
        prefs['engine'] = "whisper" if prefs['engine'] == "gemini" else "gemini"
        await settings_command(query, context) # Refresh
        return
    if data == "toggle_format":
        modes = ["ask", "srt", "txt"]
        prefs['format'] = modes[(modes.index(prefs['format'])+1)%3]
        await settings_command(query, context) # Refresh
        return
    if data == "close_settings":
        await query.message.delete()
        return

    # Download Actions
    if data == "dl_txt": await context.bot.send_document(user_id, open(p['txt'], "rb"), caption="📄 Transcript")
    if data == "dl_srt": await context.bot.send_document(user_id, open(p['srt'], "rb"), caption="🎬 Subtitles")
    
    # Translation Logic
    if data == "menu_trans":
        await translate_command(query, context)
        return

    if data.startswith("tr_"):
        ext = data.split("_")[1]
        await query.message.edit_text(f"⏳ **Translating {ext.upper()}...**")
        success, msg = await translate_file(user_id, ext)
        
        if success:
            f = p['trans_srt'] if ext == 'srt' else p['trans_txt']
            await context.bot.send_document(user_id, open(f, 'rb'), caption=f"✅ Translated {ext.upper()}")
            await query.message.delete()
        else:
            await query.message.edit_text(f"❌ Failed: {msg}")

# --- 🚀 STARTUP ---
if __name__ == '__main__':
    # Ensure download folders exist
    for f in ["downloads", "temp"]:
        os.makedirs(f, exist_ok=True)
        
    app = ApplicationBuilder().token(TG_TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("translate", translate_command))
    app.add_handler(CommandHandler("clearall", clearall_command))
    app.add_handler(CommandHandler("heygemini", heygemini_command))
    app.add_handler(CommandHandler("exit", exit_command))
    
    # Messages
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))
    app.add_handler(MessageHandler(filters.VIDEO | filters.AUDIO | filters.Document.ALL, process_media))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    print("🚀 Bot is running...")
    app.run_polling()
