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

# --- 📦 LIBRARIES ---
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

# Setup Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

if not TG_TOKEN or not GEMINI_KEY:
    print("❌ ERROR: API Keys are missing! Set TG_TOKEN and GEMINI_KEY.")
    exit()

# --- 🗣️ VOICE LIBRARY ---
VOICE_LIB = {
    "🇲🇲 Thiha (Male)": "my-MM-ThihaNeural",
    "🇲🇲 Nilar (Female)": "my-MM-NilarNeural",
    "🇺🇸 Remy (Multi)": "fr-FR-RemyMultilingualNeural",
    "🇺🇸 Andrew (Clean)": "en-US-AndrewNeural",
    "🇺🇸 Brian (Narrator)": "en-US-BrianNeural",
    "🇺🇸 Ava (Soft)": "en-US-AvaMultilingualNeural",
    "🇺🇸 Christopher (Deep)": "en-US-ChristopherNeural",
    "🇺🇸 Ana (Child)": "en-US-AnaNeural",
    "🇬🇧 Sonia (British)": "en-GB-SoniaNeural",
    "🇮🇹 Giuseppe (Multi)": "it-IT-GiuseppeMultilingualNeural"
}

# --- 📝 PROMPTS ---
SRT_RULES = """
**FORMATTING INSTRUCTIONS (STRICT):**
1. The input is an **SRT Subtitle File**.
2. **OUTPUT FORMAT:** You MUST return a valid SRT file.
3. **TIMESTAMPS:** Do NOT change, shift, or remove any timestamps. 
4. **SEQUENCE NUMBERS:** Preserve exact sequence.
5. **TRANSLATION:** Translate text to natural Burmese.
6. **TTS OPTIMIZATION:** - Write English loanwords phonetically in Burmese (e.g., CEO -> စီအီးအို).
   - Adjust spelling for correct TTS pronunciation (e.g., write 'ငမန်း' instead of 'ငါးမန်း').
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

# --- 📂 FOLDERS & DATA ---
BASE_FOLDERS = ["downloads", "temp"]
for f in BASE_FOLDERS:
    os.makedirs(f, exist_ok=True)

user_prefs = {}
user_modes = {} 
chat_histories = {}
user_last_active = {}

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
    # Added .srt suffix logic later, standardizing here
    return {
        "input": f"downloads/{user_id}_input.mp4",
        "audio": f"downloads/{user_id}_audio.mp3",
        "srt": f"downloads/{user_id}_subs.srt",
        "txt": f"downloads/{user_id}_transcript.txt",
        "trans_result": f"downloads/{user_id}_translated", 
        "dub_audio": f"downloads/{user_id}_dubbed.mp3"
    }

def clean_temp(user_id):
    p = get_paths(user_id)
    if os.path.exists(p['input']): os.remove(p['input'])
    # Safer glob pattern to avoid deleting other users' files
    for f in glob.glob(f"temp/{user_id}_chunk_*.mp3"):
        try: os.remove(f)
        except: pass
    for f in glob.glob(f"temp/sample_{user_id}.mp3"):
        try: os.remove(f)
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
            # Force markdown block closure to prevent formatting breaks
            await bot.send_message(chat_id=chat_id, text=f"```\n{chunk}\n```", parse_mode='Markdown')
        except Exception as e:
            print(f"Message Send Error: {e}")

# --- 🔊 AUDIO PROCESSING ---
def trim_silence(audio_segment, silence_thresh=-40.0, chunk_size=5):
    if len(audio_segment) < 100: return audio_segment
    # Only trim start to keep flow natural, or trim both but carefully
    start_trim = detect_leading_silence(audio_segment, silence_threshold=silence_thresh, chunk_size=chunk_size)
    end_trim = detect_leading_silence(audio_segment.reverse(), silence_threshold=silence_thresh, chunk_size=chunk_size)
    duration = len(audio_segment)
    return audio_segment[start_trim:duration-end_trim]

def make_audio_crisp(audio_segment):
    clean_audio = audio_segment.high_pass_filter(200)
    high_freqs = clean_audio.high_pass_filter(2000)
    crisp_audio = clean_audio.overlay(high_freqs - 4) 
    return effects.normalize(crisp_audio)

# --- 🎬 DUBBING ENGINE (OPTIMIZED) ---
async def generate_dubbing(user_id, srt_path, output_path, voice):
    print(f"🎬 Starting Dubbing for {user_id}...")
    try:
        subs = pysrt.open(srt_path)
        # OPTIMIZATION: Use a list instead of += AudioSegment
        audio_segments = []
        current_timeline_ms = 0
        
        BASE_RATE_VAL = 10 
        PITCH_VAL = "-2Hz"

        for i, sub in enumerate(subs):
            start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
            end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
            allowed_duration_ms = end_ms - start_ms
            
            text = sub.text.replace("\n", " ").strip()
            if not text: continue 

            # Add silence gap if needed to sync with original start time
            if start_ms > current_timeline_ms:
                gap = start_ms - current_timeline_ms
                if gap > 10: # Only add silence if gap is significant
                    audio_segments.append(AudioSegment.silent(duration=gap))
                    current_timeline_ms += gap

            temp_filename = f"temp/{user_id}_chunk_{i}.mp3"
            communicate = edge_tts.Communicate(text, voice, rate=f"+{BASE_RATE_VAL}%", pitch=PITCH_VAL)
            await communicate.save(temp_filename)
            
            segment = AudioSegment.from_file(temp_filename)
            segment = trim_silence(segment)

            # Speed up if too long
            if len(segment) > allowed_duration_ms:
                ratio = len(segment) / allowed_duration_ms
                extra_speed = (ratio - 1) * 100
                new_rate = int(BASE_RATE_VAL + extra_speed + 5)
                if new_rate > 50: new_rate = 50
                
                communicate = edge_tts.Communicate(text, voice, rate=f"+{new_rate}%", pitch=PITCH_VAL)
                await communicate.save(temp_filename)
                segment = AudioSegment.from_file(temp_filename)
                segment = trim_silence(segment)

            segment = make_audio_crisp(segment)
            audio_segments.append(segment)
            current_timeline_ms += len(segment)
            
            if os.path.exists(temp_filename): os.remove(temp_filename)

        # Combine all segments at once (Much faster)
        if audio_segments:
            final_audio = sum(audio_segments)
            final_audio.export(output_path, format="mp3")
            return True, None
        else:
            return False, "No audio generated"
            
    except Exception as e:
        return False, str(e)

# --- 🧠 AI ENGINES ---
def format_timestamp(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{math.floor(seconds):02},{milliseconds:03}"

def run_whisper(audio_path, srt_path, txt_path):
    print(f"🎙️ [Whisper] Processing...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Only use float16 if on CUDA
        compute_type = "float16" if device == "cuda" else "int8"
        
        # NOTE: Loading model inside function is slow but safer for threading if RAM is tight.
        model = WhisperModel("small", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, beam_size=5, vad_filter=True, word_timestamps=True)
        
        final_subs = []
        current_segment_words = []
        current_start = None
        
        MAX_CHARS_PER_BLOCK = 80
        all_words = []
        for segment in segments:
            all_words.extend(segment.words)

        for i, word in enumerate(all_words):
            if current_start is None: current_start = word.start
            current_segment_words.append(word)
            
            text_str = " ".join([w.word.strip() for w in current_segment_words])
            clean_word = word.word.strip()
            
            is_sentence_end = clean_word[-1] in ".?!" if clean_word else False
            is_clause_end = (clean_word[-1] == ",") and (len(text_str) > 20)
            is_too_long = len(text_str) > MAX_CHARS_PER_BLOCK

            if is_sentence_end or is_clause_end or is_too_long:
                start_ts = format_timestamp(current_start)
                end_ts = format_timestamp(word.end)
                final_subs.append({
                    "start": start_ts,
                    "end": end_ts,
                    "text": text_str
                })
                current_segment_words = []
                current_start = None

        # Flush remaining
        if current_segment_words:
            start_ts = format_timestamp(current_start)
            end_ts = format_timestamp(all_words[-1].end)
            final_subs.append({
                "start": start_ts, "end": end_ts, "text": " ".join([w.word.strip() for w in current_segment_words])
            })

        with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
            for i, sub in enumerate(final_subs, start=1):
                srt.write(f"{i}\n{sub['start']} --> {sub['end']}\n{sub['text']}\n\n")
                txt.write(f"{sub['text']} ")
        return "Whisper (Smart)"
    except Exception as e:
        print(f"Whisper Error: {e}")
        return f"Error: {e}"

def run_gemini_transcribe(audio_path, srt_path, txt_path):
    print(f"✨ [Gemini] Listening...")
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        with open(audio_path, "rb") as f: audio_bytes = f.read()
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[types.Content(parts=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"), 
                types.Part.from_text(text="Transcribe this audio strictly.")
            ])]
        )
        with open(txt_path, "w", encoding="utf-8") as f: f.write(response.text.strip())
        # Remove SRT if exists to avoid confusion (since Gemini Flash only gives text here)
        if os.path.exists(srt_path): os.remove(srt_path) 
        return "Gemini Flash"
    except Exception as e:
        return "Error"

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
    # Reset history if inactive for 24h
    if user_id in user_last_active and (current_time - user_last_active[user_id] > 86400):
        chat_histories[user_id] = []
    user_last_active[user_id] = current_time

    if user_id not in chat_histories: chat_histories[user_id] = []
    client = genai.Client(api_key=GEMINI_KEY)
    chat = client.chats.create(model='gemini-2.0-flash', history=chat_histories[user_id])
    try:
        response = chat.send_message(text)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

# --- 🤖 HANDLERS ---
async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("start", "🏠 Dashboard"),
        BotCommand("voices", "🗣️ Change Voice"),
        BotCommand("translate", "🌍 Translate"),
        BotCommand("dub", "🎬 Start Dubbing"),
        BotCommand("heygemini", "🤖 Chat AI"),
        BotCommand("cancel", "❌ Stop Mode"),
        BotCommand("clearall", "🧹 Reset All")
    ])

# New separate handler for /heygemini command
async def start_chat_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = "chat_gemini"
    await update.message.reply_text("🤖 **Gemini Chat Mode ON**\nType `/cancel` to exit.")

# New separate handler for /cancel
async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = None
    await update.message.reply_text("✅ **Mode Cancelled.**")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    v_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Unknown")
    
    text = (
        f"👋 **Video AI Studio**\n"
        f"⚙️ **Config:** `{state['transcribe_engine'].title()}` | `{v_name}`\n"
    )
    
    keyboard = [
        [InlineKeyboardButton("🗣️ Select Voice", callback_data="cmd_voices"), InlineKeyboardButton("🎙️ Switch Engine", callback_data="toggle_transcribe")],
        [InlineKeyboardButton("📝 Edit Prompts", callback_data="menu_settings"), InlineKeyboardButton("🤖 Chat AI", callback_data="cmd_chat")],
        [InlineKeyboardButton("🧹 Clear Data", callback_data="cmd_clear")]
    ]
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

async def voices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for name, code in VOICE_LIB.items():
        row.append(InlineKeyboardButton(name, callback_data=f"set_voice_{code}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row: keyboard.append(row)
    
    msg_text = "🗣️ **Select Narrator:**"
    if update.callback_query:
        await update.callback_query.message.edit_text(msg_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(msg_text, reply_markup=InlineKeyboardMarkup(keyboard))

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📝 View Prompts", callback_data="st_view")],
        [InlineKeyboardButton("✏️ Edit Burmese", callback_data="st_edit_burmese"), InlineKeyboardButton("✏️ Edit Rephrase", callback_data="st_edit_rephrase")],
        [InlineKeyboardButton("🔙 Back", callback_data="cmd_start")]
    ]
    if update.callback_query:
        await update.callback_query.message.edit_text("⚙️ **Settings**", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text("⚙️ **Settings**", reply_markup=InlineKeyboardMarkup(keyboard))

async def perform_dubbing(update, context):
    user_id = update.effective_user.id
    msg = update.effective_message
    p = get_paths(user_id)
    state = get_user_state(user_id)

    if not os.path.exists(p['srt']):
        await msg.reply_text("❌ **No SRT found.**")
        return

    voice_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Voice")
    status = await msg.reply_text(f"🎬 **Dubbing with {voice_name}...**")
    
    success, error = await generate_dubbing(user_id, p['srt'], p['dub_audio'], state['dub_voice'])
    
    if success:
        await status.delete()
        await context.bot.send_audio(chat_id=msg.chat_id, audio=open(p['dub_audio'], "rb"), title=f"Dubbed_{voice_name}", caption=f"✅ **Dubbed by {voice_name}**")
    else:
        await status.edit_text(f"❌ Failed: {error}")

async def perform_translation(update, context, user_id, prompt):
    msg = update.effective_message
    status = await msg.reply_text(f"🌍 **Translating...**")
    success, _, path = await run_translate(user_id, prompt)
    
    if success:
        await status.delete()
        await context.bot.send_document(msg.chat_id, document=open(path, "rb"), caption="✅ **Translation Done.**")
        keyboard = [[InlineKeyboardButton("🎬 Dub Audio", callback_data="trigger_dub")]]
        await context.bot.send_message(msg.chat_id, "Next Step:", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await status.edit_text("❌ Translation Error.")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    state = get_user_state(user_id)
    data = query.data
    
    if data == "cmd_start":
        await start(update, context)

    elif data == "toggle_transcribe":
        state['transcribe_engine'] = "gemini" if state['transcribe_engine'] == "whisper" else "whisper"
        await query.answer(f"Switched to: {state['transcribe_engine'].title()}")
        await start(update, context)
    
    elif data == "cmd_voices":
        await voices_command(update, context)

    elif data == "cmd_chat":
        user_modes[user_id] = "chat_gemini"
        await query.message.reply_text("🤖 **Chat Mode ON**\nType `/cancel` to exit.")
        await query.answer()

    elif data == "cmd_clear":
        wipe_user_data(user_id)
        await query.answer("Cleared.")
        await query.message.reply_text("🧹 **Cleared.**")

    elif data.startswith("set_voice_"):
        new_voice = data.replace("set_voice_", "")
        state['dub_voice'] = new_voice
        v_name = next((k for k, v in VOICE_LIB.items() if v == new_voice), "Custom")
        
        await query.message.edit_text(f"✅ Voice set to: **{v_name}**\n⏳ Generating sample...")
        
        if "my-MM" in new_voice: sample_text = "မင်္ဂလာပါ၊ ဒါက ကျွန်တော့်ရဲ့ အသံနမူနာပါ။"
        else: sample_text = "Hello, this is a quick sample of my voice."
        
        sample_path = f"temp/sample_{user_id}.mp3"
        try:
            communicate = edge_tts.Communicate(sample_text, new_voice)
            await communicate.save(sample_path)
            await context.bot.send_voice(chat_id=query.message.chat_id, voice=open(sample_path, "rb"), caption=f"🎙️ **{v_name}**")
        except:
            await context.bot.send_message(chat_id=query.message.chat_id, text="❌ Error generating sample.")

    elif data == "menu_settings":
        await settings_command(update, context)

    elif data == "st_view":
        await send_copyable_message(query.message.chat_id, context.bot, f"🇲🇲 **Burmese:**\n{get_active_prompt(user_id, 'burmese')}")
        await send_copyable_message(query.message.chat_id, context.bot, f"🇺🇸 **Rephrase:**\n{get_active_prompt(user_id, 'rephrase')}")

    elif data.startswith("st_edit_"):
        mode = data.replace("st_edit_", "")
        user_modes[user_id] = f"edit_prompt_{mode}"
        await query.message.edit_text(f"✍️ Send new **{mode.title()}** prompt:")

    elif data == "trans_burmese":
        await perform_translation(update, context, user_id, get_active_prompt(user_id, "burmese"))

    elif data == "trigger_dub":
        await perform_dubbing(update, context)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    text = msg.text
    mode = user_modes.get(user_id)
    state = get_user_state(user_id)
    p = get_paths(user_id)

    # SRT Direct Paste
    if re.search(r'\d{2}:\d{2}:\d{2},\d{3} -->', text):
        with open(p['srt'], 'w', encoding="utf-8") as f: f.write(text)
        keyboard = [[InlineKeyboardButton("🎬 Dub Audio", callback_data="trigger_dub")]]
        await msg.reply_text("✅ **SRT Saved.**", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    if mode == "chat_gemini":
        await context.bot.send_chat_action(msg.chat_id, "typing")
        response = await run_chat_gemini(user_id, text)
        await send_copyable_message(msg.chat_id, context.bot, response)
        return

    if mode and mode.startswith("edit_prompt_"):
        key = mode.replace("edit_prompt_", "")
        state.setdefault('custom_prompts', {})[key] = text
        user_modes[user_id] = None
        await msg.reply_text(f"✅ **{key.title()} Updated.**")
        return

    if "http" in text:
        await process_media(update, context, is_url=True)
        return

    if len(text) > 5:
        with open(p['txt'], "w", encoding="utf-8") as f: f.write(text)
        await msg.reply_text("✅ **Text Saved.** Type `/translate`.")

async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    file_obj = await (msg.document or msg.video or msg.audio).get_file()
    name = msg.document.file_name if msg.document else "vid.mp4"
    
    if name.lower().endswith('.srt'):
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
    
    status = await msg.reply_text("⏳ **Processing...**")
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
                await context.bot.send_document(msg.chat_id, open(p['srt'], "rb"), caption="🎬 **SRT Generated (Smart)**")
        else:
            await loop.run_in_executor(None, run_gemini_transcribe, p['audio'], p['srt'], p['txt'])
            if os.path.exists(p['txt']):
                 await context.bot.send_document(msg.chat_id, open(p['txt'], "rb"), caption="📄 **Transcript Generated**")
            
        await status.edit_text("✅ **Done!** Type `/translate` or `/dub`.")

    except Exception as e:
        await status.edit_text(f"❌ Error: {e}")

if __name__ == '__main__':
    print("🚀 Video AI Bot Running...")
    app = ApplicationBuilder().token(TG_TOKEN).post_init(post_init).build()
    
    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("voices", voices_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("translate", lambda u, c: u.message.reply_text("🌍 Options:", reply_markup=InlineKeyboardMarkup([
        [InlineKeyboardButton("To Burmese", callback_data="trans_burmese")]
    ]))))
    app.add_handler(CommandHandler("dub", perform_dubbing))
    # FIXED: Replaced lambda with actual functions
    app.add_handler(CommandHandler("heygemini", start_chat_mode))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("clearall", lambda u, c: wipe_user_data(u.effective_user.id)))

    # Handlers
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.ALL | filters.AUDIO, file_handler))
    
    app.run_polling()
