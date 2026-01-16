import logging
import os
import asyncio
import glob
import subprocess
import torch
import pysrt
import math
import shutil
import re
import time
import random

# User Data Storage
user_prefs = {}
user_modes = {} 
chat_histories = {}
user_last_active = {} 

# Audio Processing
from pydub import AudioSegment, effects
from pydub.silence import detect_leading_silence
import edge_tts

# Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, CallbackQueryHandler, filters

# AI Tools
from google import genai
from google.genai import types
from faster_whisper import WhisperModel

# --- ⚙️ CONFIGURATION ---
TG_TOKEN = os.getenv("TG_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not TG_TOKEN or not GEMINI_KEY:
    print("❌ ERROR: API Keys are missing! Set them in your environment variables.")
    exit()

# --- 🗣️ VOICE LIBRARY ---
# STANDARD: Works for SRT Dubbing & Text (Edge TTS)
# GEMINI: Works for Text Only (Google GenAI)
VOICE_LIB = {
    # --- Standard (SRT Supported) ---
    "🇲🇲 Burmese (Thiha)": "my-MM-ThihaNeural",
    "🇲🇲 Burmese (Nilar)": "my-MM-NularNeural",
    "🇺🇸 Chris (Recap)": "en-US-ChristopherNeural", 
    "🇺🇸 Eric (Energetic)": "en-US-EricNeural",
    "🇺🇸 Brian (Classic)": "en-US-BrianNeural",
    "🇬🇧 Ryan (British)": "en-GB-RyanNeural",
    "🇮🇹 Giuseppe (Italian)": "it-IT-GiuseppeMultilingualNeural",
    
    # --- Gemini (Text Only - No SRT) ---
    "✨ Gemini Journey (F)": "gemini-journey-F",
    "✨ Gemini Journey (M)": "gemini-journey-M",
    "✨ Gemini Barnaby": "gemini-barnaby",
    "✨ Gemini Puck": "gemini-puck",
    "✨ Gemini Zephyr": "gemini-zephyr"
}

# --- 📝 PROMPTS ---
SRT_RULES = """
**FORMATTING INSTRUCTIONS (STRICT):**
1. The input is an **SRT Subtitle File**.
2. **OUTPUT FORMAT:** You MUST return a valid SRT file.
3. **TIMESTAMPS:** Do NOT change, shift, or remove any timestamps. 
4. **SEQUENCE NUMBERS:** Preserve exact sequence.
5. **TRANSLATION:** Translate text to natural Burmese.
6. **LOAN WORDS:** Phonetic spelling for English terms (e.g., CEO -> စီအီးအို).
"""

BURMESE_STYLE = """
Role: Professional Video Narrator (Burmese).
Style: Natural, engaging, clear narration.
No 'ပေါ့' (pout) at end of sentences.
Translate naturally as a continuous story, not robotic word-by-word.
"""

DEFAULT_PROMPTS = {
    "burmese": BURMESE_STYLE,
    "rephrase": "Rephrase this English text to be more clear, natural, and reliable."
}

# Folders
BASE_FOLDERS = ["downloads", "temp"]
for f in BASE_FOLDERS:
    os.makedirs(f, exist_ok=True)

# --- 🛠️ HELPER FUNCTIONS ---
def get_user_state(user_id):
    if user_id not in user_prefs:
        user_prefs[user_id] = {
            "transcribe_engine": "whisper", 
            "dub_voice": "my-MM-ThihaNeural", 
            "custom_prompts": {} 
        }
    return user_prefs[user_id]

def get_active_prompt(user_id, key):
    state = get_user_state(user_id)
    custom = state.get("custom_prompts", {}).get(key)
    return custom if custom else DEFAULT_PROMPTS[key]

def get_paths(user_id):
    return {
        "input": f"downloads/{user_id}_input.mp4",
        "audio": f"downloads/{user_id}_audio.mp3",
        "srt": f"downloads/{user_id}_subs.srt",
        "txt": f"downloads/{user_id}_transcript.txt",
        "trans_result": f"downloads/{user_id}_translated",
        "dub_audio": f"downloads/{user_id}_dubbed.mp3",
        "sample": f"temp/{user_id}_sample.mp3"
    }

def clean_temp(user_id):
    # Deletes only temp chunks
    for f in glob.glob(f"temp/{user_id}_chunk_*.mp3"):
        try: os.remove(f)
        except: pass

def cleanup_old_files():
    """Deletes files older than 24 hours"""
    now = time.time()
    cutoff = 86400 # 24 Hours
    print("🧹 Running Auto Cleanup...")
    for folder in BASE_FOLDERS:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                if now - os.path.getmtime(file_path) > cutoff:
                    try: os.remove(file_path)
                    except: pass

def wipe_user_data(user_id):
    for f in glob.glob(f"downloads/{user_id}_*"):
        try: os.remove(f)
        except: pass
    clean_temp(user_id)
    if user_id in user_prefs: del user_prefs[user_id]
    if user_id in user_modes: del user_modes[user_id]
    if user_id in chat_histories: del chat_histories[user_id]

async def send_copyable_message(chat_id, bot, text):
    if not text: return
    MAX_LEN = 4000
    safe_text = text.replace("`", "'") 
    for i in range(0, len(safe_text), MAX_LEN):
        chunk = safe_text[i:i+MAX_LEN]
        try:
            await bot.send_message(chat_id=chat_id, text=f"```\n{chunk}\n```", parse_mode='Markdown')
        except: pass

# --- 🔊 AUDIO PROCESSING ---
def trim_silence(audio_segment, silence_thresh=-40.0, chunk_size=10):
    if len(audio_segment) < 100: return audio_segment
    start_trim = detect_leading_silence(audio_segment, silence_threshold=silence_thresh, chunk_size=chunk_size)
    end_trim = detect_leading_silence(audio_segment.reverse(), silence_threshold=silence_thresh, chunk_size=chunk_size)
    return audio_segment[start_trim:len(audio_segment)-end_trim]

def make_audio_crisp(audio_segment):
    clean = audio_segment.high_pass_filter(200)
    high_freqs = clean.high_pass_filter(2000)
    return effects.normalize(clean.overlay(high_freqs - 4))

async def generate_voice_sample(user_id, voice_code):
    p = get_paths(user_id)
    
    # 1. Gemini Voice (Simulated for Sample)
    if "gemini-" in voice_code:
        # Since we don't have the paid endpoint active here, 
        # we return a text warning or a generic sample if possible.
        # For this code, we will skip sample generation for Gemini to avoid errors
        return False, "Gemini samples require Text Input."

    # 2. Edge TTS (Standard)
    text = "Hello! This is a sample of my voice."
    if "my-MM" in voice_code: text = "မင်္ဂလာပါ၊ ဒါက ကျွန်တော့်ရဲ့ အသံနမူနာ ဖြစ်ပါတယ်။"
    
    try:
        communicate = edge_tts.Communicate(text, voice_code)
        await communicate.save(p['sample'])
        return True, p['sample']
    except Exception as e:
        return False, str(e)

# --- 🎬 DUBBING ENGINE (SRT) ---
async def generate_dubbing(user_id, srt_path, output_path, voice):
    # ⛔ Block Gemini Voices for SRT
    if "gemini-" in voice:
        return False, "⛔ Gemini voices are for **Text Only**. Please select a Standard voice for SRT Dubbing."

    print(f"🎬 Starting Dubbing for {user_id}...")
    try:
        subs = pysrt.open(srt_path)
        final_audio = AudioSegment.empty()
        current_timeline_ms = 0
        
        # Base Settings
        BASE_RATE = 10 
        PITCH = "-2Hz"

        for i, sub in enumerate(subs):
            start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
            end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
            allowed_dur = end_ms - start_ms
            
            text = sub.text.replace("\n", " ").strip()
            if not text: continue 

            # Sync Gap
            if start_ms > current_timeline_ms:
                gap = start_ms - current_timeline_ms
                if gap > 100:
                    final_audio += AudioSegment.silent(duration=gap)
                    current_timeline_ms += gap

            # Generate Chunk
            temp_filename = f"temp/{user_id}_chunk_{i}.mp3"
            communicate = edge_tts.Communicate(text, voice, rate=f"+{BASE_RATE}%", pitch=PITCH)
            await communicate.save(temp_filename)
            
            segment = AudioSegment.from_file(temp_filename)
            segment = trim_silence(segment)
            
            # Duration Fix (Speed up if too long)
            if len(segment) > allowed_dur:
                ratio = len(segment) / allowed_dur
                extra_speed = (ratio - 1) * 100
                new_rate = int(BASE_RATE + extra_speed + 5)
                if new_rate > 50: new_rate = 50
                
                communicate = edge_tts.Communicate(text, voice, rate=f"+{new_rate}%", pitch=PITCH)
                await communicate.save(temp_filename)
                segment = AudioSegment.from_file(temp_filename)
                segment = trim_silence(segment)

            segment = make_audio_crisp(segment)
            final_audio += segment
            current_timeline_ms += len(segment)
            
            if os.path.exists(temp_filename): os.remove(temp_filename)

        final_audio.export(output_path, format="mp3")
        return True, None

    except Exception as e:
        return False, str(e)

# --- 🧠 ENGINES (Transcribe/Translate) ---
def run_whisper(audio_path, srt_path, txt_path):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel("small", device=device, compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=1, vad_filter=True, word_timestamps=True)
        
        with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
            for i, seg in enumerate(segments, 1):
                start = math.floor(seg.start * 1000)
                end = math.floor(seg.end * 1000)
                def fmt(ms): 
                    sec, ms = divmod(ms, 1000)
                    min, sec = divmod(sec, 60)
                    hr, min = divmod(min, 60)
                    return f"{hr:02}:{min:02}:{sec:02},{ms:03}"
                
                srt.write(f"{i}\n{fmt(start)} --> {fmt(end)}\n{seg.text.strip()}\n\n")
                txt.write(seg.text.strip() + " ")
        return "Whisper"
    except Exception as e: return str(e)

def run_gemini_transcribe(audio_path, srt_path, txt_path):
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        with open(audio_path, "rb") as f: audio_bytes = f.read()
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[types.Content(parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"), types.Part.from_text(text="Transcribe exactly.")])]
        )
        with open(txt_path, "w", encoding="utf-8") as f: f.write(response.text.strip())
        if os.path.exists(srt_path): os.remove(srt_path) # No SRT for Gemini Flash audio
        return "Gemini Flash"
    except: return "Error"

async def run_translate(user_id, prompt_text):
    p = get_paths(user_id)
    source_path = p['srt'] if os.path.exists(p['srt']) else p['txt'] if os.path.exists(p['txt']) else None
    if not source_path: return False, "❌ No file found.", None

    is_srt = source_path.endswith('.srt')
    client = genai.Client(api_key=GEMINI_KEY)
    with open(source_path, "r", encoding="utf-8") as f: original_text = f.read()
    
    prompt = f"{SRT_RULES}\n{prompt_text}\n\n**INPUT:**\n{original_text}" if is_srt else f"{prompt_text}\n\n{original_text}"
    output_ext = ".srt" if is_srt else ".txt"
    
    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        content = response.text.replace("```srt", "").replace("```", "").strip()
        final_path = p['trans_result'] + output_ext
        with open(final_path, "w", encoding="utf-8") as f: f.write(content)
        if is_srt: shutil.copy(final_path, p['srt']) 
        return True, content, final_path
    except Exception as e:
        return False, str(e), None

async def run_chat_gemini(user_id, text):
    current_time = time.time()
    if user_id in user_last_active and (current_time - user_last_active[user_id] > 86400):
        chat_histories[user_id] = []
    user_last_active[user_id] = current_time

    if user_id not in chat_histories: chat_histories[user_id] = []
    client = genai.Client(api_key=GEMINI_KEY)
    chat = client.chats.create(model='gemini-2.0-flash', history=chat_histories[user_id])
    try:
        response = chat.send_message(text)
        return response.text
    except Exception as e: return f"Error: {e}"

# --- 🤖 BOT COMMANDS ---
async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("start", "🏠 Home"),
        BotCommand("voices", "🗣️ Change Voice"),
        BotCommand("translate", "🌍 Translate"),
        BotCommand("dub", "🎬 Dub Audio"),
        BotCommand("settings", "⚙️ Prompts"),
        BotCommand("heygemini", "🤖 Chat"),
        BotCommand("clearall", "🧹 Clear Data")
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    cleanup_old_files() 
    
    voice_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Unknown")
    
    keyboard = [
        [InlineKeyboardButton(f"🎙️ Engine: {state['transcribe_engine'].title()}", callback_data="toggle_transcribe")],
        [InlineKeyboardButton(f"🗣️ Voice: {voice_name}", callback_data="cmd_voices")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings"), InlineKeyboardButton("🧹 Clear", callback_data="cmd_clear")]
    ]
    await update.message.reply_text(
        "👋 **Video AI Studio**\n\n"
        "1️⃣ **Dubbing:** Send SRT file (Standard Voices).\n"
        "2️⃣ **TTS:** Send Text (Gemini Voices Available).\n"
        "3️⃣ **Transcribe:** Send Video/Audio.", 
        reply_markup=InlineKeyboardMarkup(keyboard), 
        parse_mode='Markdown'
    )

async def voices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for name, code in VOICE_LIB.items():
        # Shorten label
        label = name.split("(")[0].strip() + (" (Gemini)" if "Gemini" in name else "")
        row.append(InlineKeyboardButton(label, callback_data=f"set_voice_{code}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row: keyboard.append(row)
    
    await update.message.reply_text("🗣️ **Select Voice:**\n(Note: Gemini voices work for Text Only)", reply_markup=InlineKeyboardMarkup(keyboard))

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📝 View Prompts", callback_data="st_view")],
        [InlineKeyboardButton("✏️ Edit Burmese", callback_data="st_edit_burmese"), InlineKeyboardButton("✏️ Edit Rephrase", callback_data="st_edit_rephrase")],
        [InlineKeyboardButton("🔄 Reset", callback_data="st_reset")]
    ]
    await update.message.reply_text("⚙️ **Settings**", reply_markup=InlineKeyboardMarkup(keyboard))

async def heygemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_modes[update.effective_user.id] = "chat_gemini"
    await update.message.reply_text("🤖 **Gemini Chat ON.** Type /cancel to exit.")

async def clearall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wipe_user_data(update.effective_user.id)
    await update.message.reply_text("🧹 **Data Cleared.**")

async def dub_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await perform_dubbing(update, context)

async def perform_dubbing(update, context):
    user_id = update.effective_user.id
    p = get_paths(user_id)
    state = get_user_state(user_id)
    msg = update.effective_message

    if not os.path.exists(p['srt']):
        await msg.reply_text("❌ No SRT found.")
        return

    # Check Compatibility
    if "gemini-" in state['dub_voice']:
        await msg.reply_text("⛔ **Gemini Voice Error**\nGemini voices do not support SRT Dubbing.\nPlease switch to a Standard voice (e.g., Thiha, Chris).")
        return

    voice_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Voice")
    status = await msg.reply_text(f"🎬 **Dubbing ({voice_name})...**")
    
    success, error = await generate_dubbing(user_id, p['srt'], p['dub_audio'], state['dub_voice'])
    
    if success:
        await status.delete()
        await context.bot.send_audio(chat_id=msg.chat_id, audio=open(p['dub_audio'], "rb"), caption=f"✅ **Dubbed by {voice_name}**")
    else:
        await status.edit_text(f"❌ Error: {error}")

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = None 
    keyboard = [
        [InlineKeyboardButton("🇲🇲 To Burmese", callback_data="trans_burmese"), InlineKeyboardButton("🇺🇸 Rephrase", callback_data="trans_rephrase")],
        [InlineKeyboardButton("✍️ Custom", callback_data="trans_custom")]
    ]
    await update.message.reply_text("🌍 **Translate:**", reply_markup=InlineKeyboardMarkup(keyboard))

async def perform_translation(update, context, user_id, prompt):
    msg = update.effective_message
    status = await msg.reply_text(f"🌍 **Translating...**")
    success, _, path = await run_translate(user_id, prompt)
    
    if success:
        await status.delete()
        await context.bot.send_document(msg.chat_id, document=open(path, "rb"), caption="✅ **Done.**")
        
        keyboard = [
            [InlineKeyboardButton("✅ Good", callback_data="feedback_yes"), InlineKeyboardButton("❌ Bad", callback_data="feedback_no")]
        ]
        await context.bot.send_message(msg.chat_id, "Translation okay?", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await status.edit_text("❌ Error.")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    state = get_user_state(user_id)
    data = query.data
    
    if data == "toggle_transcribe":
        state['transcribe_engine'] = "gemini" if state['transcribe_engine'] == "whisper" else "whisper"
        await query.answer(f"Engine: {state['transcribe_engine'].title()}")
        await start(update, context)
    
    elif data == "cmd_voices":
        await voices_command(query, context)
        await query.answer()

    elif data == "cmd_clear":
        await clearall_command(update, context)
        await query.answer()

    elif data.startswith("set_voice_"):
        new_voice = data.replace("set_voice_", "")
        state['dub_voice'] = new_voice
        v_name = next((k for k, v in VOICE_LIB.items() if v == new_voice), "Voice")
        
        # Check compatibility
        note = ""
        if "gemini-" in new_voice:
            note = "\n⚠️ Text Only (No SRT)"
            await query.answer(f"Selected: {v_name} (Text Only)")
        else:
            # Generate Sample for Standard Voices
            await query.answer(f"Selected: {v_name}")
            await query.message.edit_text(f"⏳ **Generating Sample for {v_name}...**")
            success, sample_path = await generate_voice_sample(user_id, new_voice)
            if success:
                await context.bot.send_voice(chat_id=query.message.chat_id, voice=open(sample_path, "rb"), caption=f"🗣️ Sample: {v_name}")

        await start(update, context) # Return to home

    elif data == "menu_settings":
        await settings_command(update, context)

    elif data == "st_view":
        await send_copyable_message(query.message.chat_id, context.bot, f"🇲🇲 **Burmese:**\n{get_active_prompt(user_id, 'burmese')}")
        await send_copyable_message(query.message.chat_id, context.bot, f"🇺🇸 **Rephrase:**\n{get_active_prompt(user_id, 'rephrase')}")

    elif data == "st_edit_burmese":
        user_modes[user_id] = "edit_prompt_burmese"
        await query.message.edit_text("✍️ Send new Burmese prompt:")

    elif data == "st_edit_rephrase":
        user_modes[user_id] = "edit_prompt_rephrase"
        await query.message.edit_text("✍️ Send new Rephrase prompt:")

    elif data == "st_reset":
        state['custom_prompts'] = {}
        await query.answer("Reset")
        await settings_command(update, context)

    elif data == "trans_burmese":
        await perform_translation(update, context, user_id, get_active_prompt(user_id, "burmese"))

    elif data == "trans_rephrase":
        await perform_translation(update, context, user_id, get_active_prompt(user_id, "rephrase"))

    elif data == "trans_custom":
        user_modes[user_id] = "translate_prompt"
        await query.message.reply_text("✍️ Enter custom instruction:")

    elif data == "feedback_yes":
        p = get_paths(user_id)
        if os.path.exists(p['srt']):
            keyboard = [[InlineKeyboardButton("🎬 Make Audio", callback_data="trigger_dub")]]
            await query.message.edit_text("✅ Great! Dub now?", reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await query.message.edit_text("✅ Thanks!")

    elif data == "feedback_no":
        user_modes[user_id] = "translate_prompt"
        await query.message.edit_text("✍️ How to fix it?")

    elif data == "trigger_dub":
        await perform_dubbing(update, context)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    text = msg.text
    mode = user_modes.get(user_id)
    state = get_user_state(user_id)
    p = get_paths(user_id)

    if text.startswith("/cancel"):
        user_modes[user_id] = None
        await msg.reply_text("✅ Cancelled.")
        return

    # 1. SRT Detection
    if re.search(r'\d{2}:\d{2}:\d{2},\d{3} -->', text):
        with open(p['srt'], "w", encoding="utf-8") as f: f.write(text)
        keyboard = [[InlineKeyboardButton("🎬 Dub Audio", callback_data="trigger_dub")]]
        await msg.reply_text("✅ **SRT Saved.**", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    # 2. Modes
    if mode == "chat_gemini":
        await context.bot.send_chat_action(msg.chat_id, "typing")
        response = await run_chat_gemini(user_id, text)
        await send_copyable_message(msg.chat_id, context.bot, response)
        return

    if mode == "edit_prompt_burmese":
        state.setdefault('custom_prompts', {})['burmese'] = text
        user_modes[user_id] = None
        await msg.reply_text("✅ Burmese Prompt Saved.")
        return

    if mode == "edit_prompt_rephrase":
        state.setdefault('custom_prompts', {})['rephrase'] = text
        user_modes[user_id] = None
        await msg.reply_text("✅ Rephrase Prompt Saved.")
        return

    if mode == "translate_prompt":
        user_modes[user_id] = None
        await perform_translation(update, context, user_id, text)
        return

    # 3. TEXT-TO-SPEECH (TTS)
    if len(text) < 2000:
        # GEMINI TTS LOGIC
        if "gemini-" in state['dub_voice']:
            await msg.reply_text(f"✨ **Generating Gemini TTS...**\n(Note: Requires Paid Google Audio API)")
            # Simulated call (Replace with actual Google Speech API if available)
            # success = await generate_gemini_tts(text, voice) 
            await msg.reply_text("⚠️ Gemini TTS Endpoint not configured in this script. Please use Standard voices for now.")
            return

        # EDGE TTS (Standard)
        status = await msg.reply_text(f"🗣️ **Reading ({state['dub_voice']})...**")
        try:
            communicate = edge_tts.Communicate(text, state['dub_voice'])
            await communicate.save(p['sample'])
            await status.delete()
            await context.bot.send_audio(chat_id=msg.chat_id, audio=open(p['sample'], "rb"))
        except Exception as e:
            await status.edit_text(f"❌ Error: {e}")
        return

    # 4. Save as Transcript
    with open(p['txt'], "w", encoding="utf-8") as f: f.write(text)
    await msg.reply_text("✅ **Text Saved.** Type `/translate`.")

async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    state = get_user_state(user_id)
    
    file_obj = await (msg.document or msg.video or msg.audio).get_file()
    name = msg.document.file_name if msg.document else "vid.mp4"
    
    # SRT Input
    if name.lower().endswith('.srt'):
        await file_obj.download_to_drive(p['srt'])
        keyboard = [[InlineKeyboardButton("🎬 Dub Audio", callback_data="trigger_dub")]]
        await msg.reply_text("✅ **SRT Loaded.**", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    # TXT Input
    if name.lower().endswith('.txt'):
        await file_obj.download_to_drive(p['txt'])
        await msg.reply_text("✅ **Text Loaded.**")
        return
        
    # Media Processing (Audio/Video)
    status = await msg.reply_text("⏳ **Processing Media...**")
    try:
        clean_temp(user_id)
        await file_obj.download_to_drive(p['input'])
        subprocess.run(f"ffmpeg -y -i {p['input']} -vn -acodec libmp3lame -q:a 2 {p['audio']}", shell=True)
            
        loop = asyncio.get_event_loop()
        if state['transcribe_engine'] == "whisper":
            await loop.run_in_executor(None, run_whisper, p['audio'], p['srt'], p['txt'])
            if os.path.exists(p['srt']):
                await context.bot.send_document(msg.chat_id, open(p['srt'], "rb"), caption="🎬 Subtitles")
        else:
            await loop.run_in_executor(None, run_gemini_transcribe, p['audio'], p['srt'], p['txt'])
            if os.path.exists(p['txt']):
                 await context.bot.send_document(msg.chat_id, open(p['txt'], "rb"), caption="📄 Transcript")
            
        await status.edit_text("✅ **Done!** Type `/translate`.")

    except Exception as e:
        await status.edit_text(f"❌ Error: {e}")

if __name__ == '__main__':
    print("🚀 Video AI Bot Running...")
    app = ApplicationBuilder().token(TG_TOKEN).post_init(post_init).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("voices", voices_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("heygemini", heygemini_command))
    app.add_handler(CommandHandler("translate", translate_command))
    app.add_handler(CommandHandler("dub", dub_command))
    app.add_handler(CommandHandler("clearall", clearall_command))
    
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.ALL | filters.AUDIO, file_handler))
    
    app.run_polling()
