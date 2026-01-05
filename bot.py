import logging
import os
import asyncio
import glob
import shlex
import torch
import pysrt
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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

# --- 🚀 GLOBAL AI MODELS ---
print("⏳ Loading AI Models...")
GENAI_CLIENT = genai.Client(api_key=GEMINI_KEY)

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    GLOBAL_WHISPER = WhisperModel("small", device=device, compute_type="float16" if device=="cuda" else "int8")
    print(f"✅ Whisper loaded on {device}")
except:
    GLOBAL_WHISPER = None
    print("⚠️ Whisper failed. Gemini will be used.")

# --- 💾 SETTINGS MANAGER ---
user_prefs = {}

def get_prefs(user_id):
    if user_id not in user_prefs:
        user_prefs[user_id] = {"engine": "gemini", "format": "ask"}
    return user_prefs[user_id]

def get_settings_markup(user_id):
    prefs = get_prefs(user_id)
    
    eng_gemini = "✅ Gemini" if prefs['engine'] == "gemini" else "Gemini"
    eng_whisper = "✅ Whisper" if prefs['engine'] == "whisper" else "Whisper"
    
    fmt_ask = "✅ Ask Me" if prefs['format'] == "ask" else "Ask Me"
    fmt_srt = "✅ Always SRT" if prefs['format'] == "srt" else "Always SRT"
    fmt_txt = "✅ Always TXT" if prefs['format'] == "txt" else "Always TXT"

    keyboard = [
        [InlineKeyboardButton("🧠 AI Engine", callback_data="ignore")],
        [InlineKeyboardButton(eng_gemini, callback_data="set_eng_gemini"),
         InlineKeyboardButton(eng_whisper, callback_data="set_eng_whisper")],
        
        [InlineKeyboardButton("📄 Output Format", callback_data="ignore")],
        [InlineKeyboardButton(fmt_ask, callback_data="set_fmt_ask")],
        [InlineKeyboardButton(fmt_srt, callback_data="set_fmt_srt"),
         InlineKeyboardButton(fmt_txt, callback_data="set_fmt_txt")],
         
        [InlineKeyboardButton("🔙 Done", callback_data="close_settings")]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- 🛠️ HELPER FUNCTIONS ---
def get_paths(user_id):
    base = f"downloads/{user_id}"
    return {
        "input": f"{base}_input",
        "audio": f"{base}_audio.mp3",
        "srt": f"{base}.srt", "txt": f"{base}.txt",
        "trans_srt": f"{base}_trans.srt", "trans_txt": f"{base}_trans.txt"
    }

def cleanup_files(user_id):
    for f in glob.glob(f"downloads/{user_id}*"):
        try: os.remove(f)
        except: pass

def clean_gemini_srt_output(raw_text):
    """
    Cleans up Gemini's chatty output to ensure valid SRT format.
    1. Removes markdown code blocks.
    2. Removes Intro/Outro text.
    3. Fixes timestamp format errors ('.' to ',').
    """
    # 1. Strip Markdown Code Blocks
    clean = re.sub(r"```\w*\n", "", raw_text) # Remove ```srt
    clean = clean.replace("```", "")          # Remove closing ```
    
    # 2. Extract only the part that looks like SRT
    # (Looks for pattern: "1\n00:00...")
    match = re.search(r'(\d+\s+\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}[\s\S]*)', clean)
    if match:
        clean = match.group(1)
    
    # 3. Fix timestamps (SRT uses comma, not dot for milliseconds)
    # 00:00:00.000 -> 00:00:00,000
    clean = re.sub(r'(\d{2}:\d{2}:\d{2})\.(\d{3})', r'\1,\2', clean)
    
    return clean.strip()

async def download_media(msg, p):
    if msg.text and "http" in msg.text:
        cmd = ['yt-dlp', '--no-check-certificate', '-f', 'bestaudio/best', '-x', '--audio-format', 'mp3', '-o', p['audio'], msg.text]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
    else:
        file_obj = await (msg.video or msg.audio or msg.document).get_file()
        await file_obj.download_to_drive(p['input'])
        cmd = ['ffmpeg', '-y', '-i', p['input'], '-vn', '-acodec', 'libmp3lame', '-q:a', '2', p['audio']]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.wait()
    return os.path.exists(p['audio'])

# --- 🧠 ENGINES ---
def run_whisper_sync(audio_path, srt_path, txt_path):
    if not GLOBAL_WHISPER: return "Whisper Error"
    segments, _ = GLOBAL_WHISPER.transcribe(audio_path, beam_size=5)
    with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
        for i, seg in enumerate(segments, 1):
            t = f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}"
            srt.write(f"{i}\n{t}\n{seg.text.strip()}\n\n")
            txt.write(f"{seg.text.strip()} ")
    return "Whisper"

def run_gemini_sync(audio_path, srt_path, txt_path):
    try:
        with open(audio_path, "rb") as f: audio_bytes = f.read()
        
        # STRICT Prompt for SRT
        prompt = """
        Transcribe the audio into SubRip (.srt) format.
        STRICT RULES:
        1. Output ONLY the SRT content. Do not add "Here is the srt" or markdown.
        2. Use correct timestamp format: 00:00:00,000 --> 00:00:00,000 (Comma for ms).
        3. Ensure lines are broken naturally.
        """
        
        response = GENAI_CLIENT.models.generate_content(
            model='gemini-2.0-flash',
            contents=[types.Content(parts=[types.Part.from_bytes(audio_bytes, "audio/mp3"), types.Part.from_text(prompt)])]
        )
        
        # CLEANUP ROUTINE
        raw_content = response.text.strip()
        clean_content = clean_gemini_srt_output(raw_content)
        
        # Validation: Check if it actually looks like SRT
        if "-->" not in clean_content:
            # Fallback: Treat as plain text if structure failed
            with open(txt_path, "w", encoding="utf-8") as f: f.write(clean_content)
            with open(srt_path, "w", encoding="utf-8") as f: f.write(f"1\n00:00:00,000 --> 00:00:05,000\n{clean_content}")
            return "Gemini (Text Mode)"
            
        with open(srt_path, "w", encoding="utf-8") as f: f.write(clean_content)
        
        # Extract Text from clean SRT for the .txt version
        clean_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', clean_content)
        clean_text = clean_text.replace('\n\n', ' ').strip()
        with open(txt_path, "w", encoding="utf-8") as f: f.write(clean_text)
        
        return "Gemini 2.0 Flash"
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Error"

def format_timestamp(s):
    import math
    h=math.floor(s/3600); s%=3600; m=math.floor(s/60); s%=60
    return f"{h:02}:{m:02}:{math.floor(s):02},{round((s%1)*1000):03}"

async def translate_file(user_id, ext):
    p = get_paths(user_id)
    src, out = (p['srt'], p['trans_srt']) if ext == 'srt' else (p['txt'], p['trans_txt'])
    if not os.path.exists(src): return False
    
    # YOUR CUSTOM TRANSLATION RULES
    prompt = """
    Role: Professional Burmese Video Narrator.
    Task: Translate the input to Natural Burmese.
    
    Guidelines:
    1. **Style:** Natural narrator flow. Not stiff.
    2. **Forbidden:** NEVER use 'ပေါ့' (pout).
    3. **Loan Words:** Write English abbreviations phonetically in Burmese (e.g. CIA -> စီအိုင်အေ).
    4. **Format:** Keep strictly to the input format (SRT or Text).
    """
    
    try:
        if ext == 'srt':
            subs = pysrt.open(src, encoding='utf-8')
            texts = [s.text.replace('\n', ' ') for s in subs]
            block = "\n<SEP>\n".join(texts)
            
            # Send batch to Gemini
            res = await asyncio.get_running_loop().run_in_executor(None, lambda: GENAI_CLIENT.models.generate_content(
                model='gemini-2.0-flash', 
                contents=f"{prompt}\n\nINSTRUCTIONS: Translate the following lines (separated by <SEP>). Return them separated by <SEP>.\n\nDATA:\n{block}"
            ))
            
            lines = res.text.strip().split("<SEP>")
            
            # Fill back into SRT structure
            for i, s in enumerate(subs):
                if i < len(lines): 
                    s.text = lines[i].strip()
            subs.save(out, encoding='utf-8')
        else:
            with open(src, 'r') as f: text = f.read()
            res = await asyncio.get_running_loop().run_in_executor(None, lambda: GENAI_CLIENT.models.generate_content(
                model='gemini-2.0-flash', contents=f"{prompt}\n\nInput Text:\n{text}"
            ))
            with open(out, 'w') as f: f.write(res.text)
        return True
    except Exception as e:
        print(f"Translation Error: {e}")
        return False

# --- 🎮 HANDLERS ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 **Video AI Studio**\nSend a video/audio to transcribe!\n\n/settings - Change Engine (Gemini/Whisper)")

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text("⚙️ **Settings Menu**", reply_markup=get_settings_markup(user_id))

async def process_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    prefs = get_prefs(user_id)
    p = get_paths(user_id)
    
    cleanup_files(user_id)
    status = await msg.reply_text("⏳ **Processing...**")

    if not await download_media(msg, p):
        await status.edit_text("❌ Download Failed.")
        return

    await status.edit_text(f"📝 **Transcribing ({prefs['engine']})...**")
    loop = asyncio.get_running_loop()
    runner = run_gemini_sync if prefs['engine'] == "gemini" else run_whisper_sync
    res_name = await loop.run_in_executor(None, runner, p['audio'], p['srt'], p['txt'])
    
    await status.delete()

    async def send(ext):
        f = p['srt'] if ext == 'srt' else p['txt']
        if os.path.exists(f):
            await context.bot.send_document(user_id, open(f, 'rb'), caption=f"✅ {ext.upper()} ({res_name})")

    # Auto-Format Logic
    if prefs['format'] == 'srt':
        await send('srt')
        await show_trans_menu(update, context)
    elif prefs['format'] == 'txt':
        await send('txt')
        await show_trans_menu(update, context)
    else:
        # Ask User Logic
        kb = [[InlineKeyboardButton("📄 TXT", callback_data="dl_txt"), InlineKeyboardButton("🎬 SRT", callback_data="dl_srt")],
              [InlineKeyboardButton("🌍 Translate", callback_data="menu_trans")]]
        await msg.reply_text(f"✅ **Done! Choose:**", reply_markup=InlineKeyboardMarkup(kb))

async def show_trans_menu(update, context):
    kb = [[InlineKeyboardButton("🌍 Translate File", callback_data="menu_trans")]]
    await context.bot.send_message(update.effective_chat.id, "Need translation?", reply_markup=InlineKeyboardMarkup(kb))

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    prefs = get_prefs(user_id)
    p = get_paths(user_id)
    data = query.data

    # --- SETTINGS LOGIC ---
    if data.startswith("set_"):
        _, type, val = data.split("_")
        if type == "eng": prefs['engine'] = val
        if type == "fmt": prefs['format'] = val
        try: await query.edit_message_reply_markup(reply_markup=get_settings_markup(user_id))
        except: pass
        return

    if data == "close_settings":
        await query.message.delete()
        return

    # --- FILE LOGIC ---
    if data == "dl_txt": await context.bot.send_document(user_id, open(p['txt'], "rb"), caption="📄 Transcript")
    if data == "dl_srt": await context.bot.send_document(user_id, open(p['srt'], "rb"), caption="🎬 Subtitles")

    # --- TRANSLATE LOGIC ---
    if data == "menu_trans":
        kb = [[InlineKeyboardButton("🇲🇲 Burmese (.SRT)", callback_data="tr_srt"),
               InlineKeyboardButton("🇲🇲 Burmese (.TXT)", callback_data="tr_txt")]]
        await query.message.edit_text("Select Format:", reply_markup=InlineKeyboardMarkup(kb))

    if data.startswith("tr_"):
        ext = data.split("_")[1]
        await query.message.edit_text(f"⏳ **Translating {ext.upper()}...**")
        if await translate_file(user_id, ext):
            f = p['trans_srt'] if ext == 'srt' else p['trans_txt']
            await context.bot.send_document(user_id, open(f, 'rb'), caption=f"✅ Translated {ext.upper()}")
            await query.message.delete()
        else:
            await query.message.edit_text("❌ Failed.")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(MessageHandler(filters.TEXT & filters.Entity("url"), process_media))
    app.add_handler(MessageHandler(filters.VIDEO | filters.AUDIO | filters.Document.ALL, process_media))
    app.add_handler(CallbackQueryHandler(callback_handler))
    print("🚀 Bot Running...")
    app.run_polling()
