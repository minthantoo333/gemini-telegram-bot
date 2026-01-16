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

# User Data
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

# --- 🗣️ VOICE LIBRARY (UPDATED) ---
# Note: "Gemini" specific TTS isn't public yet, so we use the high-end Multilingual voices
# which sound very similar to current AI assistants.
VOICE_LIB = {
    # Burmese
    "🇲🇲 Burmese (Thiha)": "my-MM-ThihaNeural",
    "🇲🇲 Burmese (Nilar)": "my-MM-NularNeural",  # Label fixed to Nilar
    
    # "Gemini-Style" / Modern AI Voices (Multilingual & Conversational)
    "🤖 Ava (AI Female)": "en-US-AvaMultilingualNeural",
    "🤖 Andrew (AI Male)": "en-US-AndrewMultilingualNeural",
    "🤖 Emma (Soft AI)": "en-US-EmmaMultilingualNeural",
    
    # Movie Recap / Social Media Favorites
    "🇺🇸 Chris (Recap King)": "en-US-ChristopherNeural", 
    "🇺🇸 Eric (Energetic)": "en-US-EricNeural",
    "🇺🇸 Brian (Classic)": "en-US-BrianNeural",
    
    # International / Accents
    "🇬🇧 Ryan (British)": "en-GB-RyanNeural",
    "🇬🇧 Sonia (British)": "en-GB-SoniaNeural",
    "🇮🇹 Giuseppe (Italiano)": "it-IT-GiuseppeMultilingualNeural"
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
    for f in glob.glob(f"temp/{user_id}_chunk_*.mp3"):
        try: os.remove(f)
        except: pass

def cleanup_old_files():
    """Deletes files older than 24 hours"""
    now = time.time()
    cutoff = 86400 # 24 Hours in seconds
    for folder in BASE_FOLDERS:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > cutoff:
                    try:
                        os.remove(file_path)
                        print(f"🗑️ Auto-deleted: {filename}")
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
        except Exception as e:
            print(f"Message Send Error: {e}")

async def generate_voice_sample(user_id, voice_code):
    """Generates a short 'Hello' audio for the selected voice."""
    p = get_paths(user_id)
    
    # Sample Text based on Language
    if "my-MM" in voice_code:
        text = "မင်္ဂလာပါ၊ ဒါက ကျွန်တော့်ရဲ့ အသံနမူနာ ဖြစ်ပါတယ်။"
    elif "Multilingual" in voice_code:
        text = "Hello! I am an AI voice. I can speak naturally for your videos."
    else:
        text = "Hello! This is a sample of my voice. I am ready to read your script."

    try:
        communicate = edge_tts.Communicate(text, voice_code, rate="+0%")
        await communicate.save(p['sample'])
        return True, p['sample']
    except Exception as e:
        return False, str(e)

# --- 🔊 AUDIO POST-PROCESSING ---
def trim_silence(audio_segment, silence_thresh=-40.0, chunk_size=10):
    if len(audio_segment) < 100: return audio_segment
    start_trim = detect_leading_silence(audio_segment, silence_threshold=silence_thresh, chunk_size=chunk_size)
    end_trim = detect_leading_silence(audio_segment.reverse(), silence_threshold=silence_thresh, chunk_size=chunk_size)
    duration = len(audio_segment)
    return audio_segment[start_trim:duration-end_trim]

def make_audio_crisp(audio_segment):
    # Slight EQ boost for clarity
    clean_audio = audio_segment.high_pass_filter(200)
    high_freqs = clean_audio.high_pass_filter(2000)
    crisp_audio = clean_audio.overlay(high_freqs - 4) 
    final_audio = effects.normalize(crisp_audio)
    return final_audio

# --- 🎬 DUBBING ENGINE ---
async def generate_dubbing(user_id, srt_path, output_path, voice):
    print(f"🎬 Starting Dubbing for {user_id}...")
    try:
        subs = pysrt.open(srt_path)
        final_audio = AudioSegment.empty()
        current_timeline_ms = 0
        
        BASE_RATE_VAL = 10 
        PITCH_VAL = "-2Hz"

        for i, sub in enumerate(subs):
            start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
            end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
            allowed_duration_ms = end_ms - start_ms
            
            text = sub.text.replace("\n", " ").strip()
            if not text: continue 

            # Sync Check
            if start_ms > current_timeline_ms:
                gap = start_ms - current_timeline_ms
                if gap > 100:
                    final_audio += AudioSegment.silent(duration=gap)
                    current_timeline_ms += gap

            # Generate
            temp_filename = f"temp/{user_id}_chunk_{i}.mp3"
            communicate = edge_tts.Communicate(text, voice, rate=f"+{BASE_RATE_VAL}%", pitch=PITCH_VAL)
            await communicate.save(temp_filename)
            
            segment = AudioSegment.from_file(temp_filename)
            segment = trim_silence(segment)

            # Duration Check
            current_len = len(segment)
            if current_len > allowed_duration_ms:
                ratio = current_len / allowed_duration_ms
                extra_speed_needed = (ratio - 1) * 100
                new_rate = int(BASE_RATE_VAL + extra_speed_needed + 5)
                if new_rate > 50: new_rate = 50
                
                communicate = edge_tts.Communicate(text, voice, rate=f"+{new_rate}%", pitch=PITCH_VAL)
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

# --- 🧠 ENGINES (WHISPER & GEMINI) ---
def run_whisper(audio_path, srt_path, txt_path):
    print(f"🎙️ [Whisper] Processing...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("small", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, beam_size=1, vad_filter=True, word_timestamps=True)
        
        final_subs = []
        current_text = []
        current_start = None
        MAX_CHARS = 100 

        def format_timestamp(seconds):
            hours = math.floor(seconds / 3600)
            seconds %= 3600
            minutes = math.floor(seconds / 60)
            seconds %= 60
            milliseconds = round((seconds - math.floor(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{math.floor(seconds):02},{milliseconds:03}"

        for segment in segments:
            for word in segment.words:
                if current_start is None: current_start = word.start
                current_text.append(word.word.strip())
                
                text_str = " ".join(current_text)
                if word.word.strip()[-1] in ".?!" or len(text_str) > MAX_CHARS:
                    final_subs.append({
                        "start": format_timestamp(current_start),
                        "end": format_timestamp(word.end),
                        "text": text_str
                    })
                    current_text = []
                    current_start = None

        if current_text:
            final_subs.append({
                "start": format_timestamp(current_start if current_start else 0),
                "end": format_timestamp(segments[-1].end),
                "text": " ".join(current_text)
            })

        with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
            for i, sub in enumerate(final_subs, start=1):
                srt.write(f"{i}\n{sub['start']} --> {sub['end']}\n{sub['text']}\n\n")
                txt.write(f"{sub['text']} ")
                
        return "Whisper (Sentence Mode)"
    except Exception as e:
        return f"Error: {e}"

def run_gemini_transcribe(audio_path, srt_path, txt_path):
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        with open(audio_path, "rb") as f: audio_bytes = f.read()
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[types.Content(parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"), types.Part.from_text(text="Transcribe to text.")])]
        )
        with open(txt_path, "w", encoding="utf-8") as f: f.write(response.text.strip())
        if os.path.exists(srt_path): os.remove(srt_path) 
        return "Gemini Flash"
    except: return "Error"

# --- 🌍 TRANSLATE & CHAT ---
async def run_translate(user_id, prompt_text):
    p = get_paths(user_id)
    source_path = p['srt'] if os.path.exists(p['srt']) else p['txt'] if os.path.exists(p['txt']) else None
    if not source_path: return False, "❌ No file found.", None

    is_srt = source_path.endswith('.srt')
    client = genai.Client(api_key=GEMINI_KEY)
    with open(source_path, "r", encoding="utf-8") as f: original_text = f.read()
    
    if is_srt:
        ai_prompt = f"{SRT_RULES}\n{prompt_text}\n\n**INPUT SRT:**\n{original_text}"
        output_ext = ".srt"
    else:
        ai_prompt = f"User Instruction: {prompt_text}\n\nInput Text:\n{original_text}"
        output_ext = ".txt"
    
    try:
        response = client.models.generate_content(model='gemini-2.0-flash', contents=ai_prompt)
        content = response.text.strip().replace("```srt", "").replace("```", "").strip()
        final_path = p['trans_result'] + output_ext
        with open(final_path, "w", encoding="utf-8") as f: f.write(content)
        if is_srt: shutil.copy(final_path, p['srt']) 
        return True, content, final_path
    except Exception as e:
        return False, str(e), None

async def run_chat_gemini(user_id, text):
    current_time = time.time()
    # Auto Clean Context > 24H
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

# --- 🤖 HANDLERS ---
async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("start", "🏠 Home"),
        BotCommand("voices", "🗣️ Change Voice"),
        BotCommand("translate", "🌍 Translate"),
        BotCommand("dub", "🎬 Dub Audio"),
        BotCommand("settings", "⚙️ Prompts"),
        BotCommand("heygemini", "🤖 Chat"),
        BotCommand("clearall", "🧹 Clear"),
        BotCommand("cancel", "❌ Cancel")
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    cleanup_old_files() 
    
    voice_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Unknown")
    
    keyboard = [
        [InlineKeyboardButton(f"🎙️ Engine: {state['transcribe_engine'].title()}", callback_data="toggle_transcribe")],
        [InlineKeyboardButton(f"🗣️ Voice: {voice_name}", callback_data="cmd_voices")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings"), InlineKeyboardButton("🧹 Clear Data", callback_data="cmd_clear")]
    ]
    await update.message.reply_text(
        "👋 **Video AI Studio**\n\n"
        "📤 **Send a file** (Video/Audio/SRT) to begin.\n"
        "🕒 *Files older than 24h are auto-deleted.*", 
        reply_markup=InlineKeyboardMarkup(keyboard), 
        parse_mode='Markdown'
    )

async def voices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for name, code in VOICE_LIB.items():
        # Shorten names for button fit
        short_name = name.split("(")[0].strip() + ("..." if len(name) > 20 else "")
        row.append(InlineKeyboardButton(short_name, callback_data=f"set_voice_{code}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row: keyboard.append(row)
    
    await update.message.reply_text("🗣️ **Select Narrator Voice:**\n(A sample will be sent)", reply_markup=InlineKeyboardMarkup(keyboard))

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📝 View Prompts", callback_data="st_view")],
        [InlineKeyboardButton("✏️ Edit Burmese", callback_data="st_edit_burmese"), InlineKeyboardButton("✏️ Edit Rephrase", callback_data="st_edit_rephrase")],
        [InlineKeyboardButton("🔄 Reset Defaults", callback_data="st_reset")]
    ]
    await update.message.reply_text("⚙️ **Prompt Settings**", reply_markup=InlineKeyboardMarkup(keyboard))

async def heygemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_modes[update.effective_user.id] = "chat_gemini"
    await update.message.reply_text("🤖 **Gemini Chat Mode ON**\nType `/cancel` to exit.")

async def clearall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wipe_user_data(update.effective_user.id)
    await update.message.reply_text("🧹 **All your data has been cleared.**")

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_modes[update.effective_user.id] = None
    await update.message.reply_text("✅ Mode exited.")

async def dub_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await perform_dubbing(update, context)

async def perform_dubbing(update, context):
    user_id = update.effective_user.id
    p = get_paths(user_id)
    state = get_user_state(user_id)
    msg = update.effective_message

    if not os.path.exists(p['srt']):
        await msg.reply_text("❌ **No SRT found.** Send an SRT file first.")
        return

    voice_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Selected Voice")
    status = await msg.reply_text(f"🎬 **Dubbing with {voice_name}...**")
    
    success, error = await generate_dubbing(user_id, p['srt'], p['dub_audio'], state['dub_voice'])
    
    if success:
        await status.delete()
        await context.bot.send_audio(chat_id=msg.chat_id, audio=open(p['dub_audio'], "rb"), caption=f"✅ **Dubbing Complete**\n🗣️ {voice_name}")
    else:
        await status.edit_text(f"❌ Dubbing Failed: {error}")

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = None 
    keyboard = [
        [InlineKeyboardButton("🇲🇲 To Burmese", callback_data="trans_burmese"), InlineKeyboardButton("🇺🇸 Rephrase", callback_data="trans_rephrase")],
        [InlineKeyboardButton("✍️ Custom Instruction", callback_data="trans_custom")]
    ]
    await update.message.reply_text("🌍 **Select Translation:**", reply_markup=InlineKeyboardMarkup(keyboard))

async def perform_translation(update, context, user_id, prompt):
    msg = update.effective_message
    status = await msg.reply_text(f"🌍 **Translating...**")
    success, _, path = await run_translate(user_id, prompt)
    
    if success:
        await status.delete()
        await context.bot.send_document(msg.chat_id, document=open(path, "rb"), caption="✅ **Translation Done.**")
        
        keyboard = [
            [InlineKeyboardButton("✅ Looks Good", callback_data="feedback_yes"), InlineKeyboardButton("❌ Needs Fix", callback_data="feedback_no")]
        ]
        await context.bot.send_message(msg.chat_id, "Is the translation okay?", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await status.edit_text("❌ Error.")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    state = get_user_state(user_id)
    data = query.data
    
    if data == "toggle_transcribe":
        state['transcribe_engine'] = "gemini" if state['transcribe_engine'] == "whisper" else "whisper"
        await query.answer(f"Engine switched to: {state['transcribe_engine'].title()}")
        await start(update, context) # Refresh menu
    
    elif data == "cmd_voices":
        await voices_command(query, context)
        await query.answer()

    elif data == "cmd_clear":
        await clearall_command(update, context)
        await query.answer("Cleared")

    elif data.startswith("set_voice_"):
        new_voice = data.replace("set_voice_", "")
        state['dub_voice'] = new_voice
        v_name = next((k for k, v in VOICE_LIB.items() if v == new_voice), "Voice")
        
        # 🔊 SEND SAMPLE AUDIO
        await query.answer(f"Selected: {v_name}")
        await query.message.edit_text(f"⏳ **Generating Sample for {v_name}...**")
        
        success, sample_path = await generate_voice_sample(user_id, new_voice)
        if success:
            await context.bot.send_voice(chat_id=query.message.chat_id, voice=open(sample_path, "rb"), caption=f"🗣️ Sample: {v_name}")
            await start(update, context)
        else:
            await query.message.edit_text(f"❌ Error generating sample.")

    elif data == "menu_settings":
        await settings_command(update, context)

    elif data == "st_view":
        await send_copyable_message(query.message.chat_id, context.bot, f"🇲🇲 **Burmese Prompt:**\n{get_active_prompt(user_id, 'burmese')}")
        await send_copyable_message(query.message.chat_id, context.bot, f"🇺🇸 **Rephrase Prompt:**\n{get_active_prompt(user_id, 'rephrase')}")

    elif data == "st_edit_burmese":
        user_modes[user_id] = "edit_prompt_burmese"
        await query.message.edit_text("✍️ **Send your new Burmese Prompt:**")

    elif data == "st_edit_rephrase":
        user_modes[user_id] = "edit_prompt_rephrase"
        await query.message.edit_text("✍️ **Send your new Rephrase Prompt:**")

    elif data == "st_reset":
        state['custom_prompts'] = {}
        await query.answer("Prompts Reset to Default")
        await settings_command(update, context)

    elif data == "trans_burmese":
        await perform_translation(update, context, user_id, get_active_prompt(user_id, "burmese"))

    elif data == "trans_rephrase":
        await perform_translation(update, context, user_id, get_active_prompt(user_id, "rephrase"))

    elif data == "trans_custom":
        user_modes[user_id] = "translate_prompt"
        await query.message.reply_text("✍️ **Enter your custom instruction:**")

    elif data == "feedback_yes":
        p = get_paths(user_id)
        if os.path.exists(p['srt']):
            keyboard = [[InlineKeyboardButton("🎬 Start Dubbing", callback_data="trigger_dub")]]
            await query.message.edit_text("✅ Awesome! Do you want to dub this now?", reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await query.message.edit_text("✅ Thanks for the feedback!")

    elif data == "feedback_no":
        user_modes[user_id] = "translate_prompt"
        await query.message.edit_text("✍️ Tell me how to fix it (Enter Prompt):")

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
        await cancel_command(update, context)
        return

    # --- 1. SRT DETECTION ---
    if re.search(r'\d{2}:\d{2}:\d{2},\d{3} -->', text):
        file_mode = 'a'
        if re.match(r'^\s*1\s*$', text.split('\n')[0].strip()) or text.strip().startswith('1\n'):
            file_mode = 'w'
        
        with open(p['srt'], file_mode, encoding="utf-8") as f: f.write(text + "\n")
        
        keyboard = [[InlineKeyboardButton("🎬 Dub Audio", callback_data="trigger_dub")]]
        await msg.reply_text("✅ **SRT Text Detected & Saved.**", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    # --- 2. Modes ---
    if mode == "chat_gemini":
        await context.bot.send_chat_action(msg.chat_id, "typing")
        response = await run_chat_gemini(user_id, text)
        await send_copyable_message(msg.chat_id, context.bot, response)
        return

    if mode == "edit_prompt_burmese":
        state.setdefault('custom_prompts', {})['burmese'] = text
        user_modes[user_id] = None
        await msg.reply_text("✅ Burmese Prompt Updated.")
        return

    if mode == "edit_prompt_rephrase":
        state.setdefault('custom_prompts', {})['rephrase'] = text
        user_modes[user_id] = None
        await msg.reply_text("✅ Rephrase Prompt Updated.")
        return

    if mode == "translate_prompt":
        user_modes[user_id] = None
        await perform_translation(update, context, user_id, text)
        return

    if "http" in text:
        await process_media(update, context, is_url=True)
        return

    # --- 3. Default ---
    if len(text) > 5:
        with open(p['txt'], "w", encoding="utf-8") as f: f.write(text)
        await msg.reply_text("✅ **Text Saved.** Type `/translate` to process.")

async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    file_obj = await (msg.document or msg.video or msg.audio).get_file()
    name = msg.document.file_name if msg.document else "vid.mp4"
    
    if name.lower().endswith('.srt'):
        await msg.reply_text("⬇️ **SRT File Received.**")
        await file_obj.download_to_drive(p['srt'])
        keyboard = [[InlineKeyboardButton("🎬 Dub Audio", callback_data="trigger_dub")]]
        await msg.reply_text("✅ **SRT Loaded.**", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    if name.lower().endswith('.txt'):
        await file_obj.download_to_drive(p['txt'])
        await msg.reply_text("✅ **Text Loaded.** Type `/translate`.")
        return
        
    await process_media(update, context, is_url=False)

async def process_media(update, context, is_url):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    state = get_user_state(user_id)
    
    status = await msg.reply_text("⏳ **Processing Media...**")
    try:
        clean_temp(user_id)
        if is_url:
            subprocess.run(f"yt-dlp -x --audio-format mp3 -o '{p['audio']}' {msg.text}", shell=True)
        else:
            file_obj = await (msg.video or msg.document or msg.audio).get_file()
            await file_obj.download_to_drive(p['input'])
            subprocess.run(f"ffmpeg -y -i {p['input']} -vn -acodec libmp3lame -q:a 2 {p['audio']}", shell=True)
            
        loop = asyncio.get_event_loop()
        if state['transcribe_engine'] == "whisper":
            await loop.run_in_executor(None, run_whisper, p['audio'], p['srt'], p['txt'])
            if os.path.exists(p['srt']):
                await context.bot.send_document(msg.chat_id, open(p['srt'], "rb"), caption="🎬 Subtitles Generated")
        else:
            await loop.run_in_executor(None, run_gemini_transcribe, p['audio'], p['srt'], p['txt'])
            if os.path.exists(p['txt']):
                 await context.bot.send_document(msg.chat_id, open(p['txt'], "rb"), caption="📄 Transcript Generated")
            
        await status.edit_text("✅ **Processing Complete!**\nUse `/translate` or `/dub`.")

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
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.ALL | filters.AUDIO, file_handler))
    
    app.run_polling()
