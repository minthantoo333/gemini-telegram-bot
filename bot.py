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
import sys

# --- 🔍 DIAGNOSTICS & LOGGING ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Check for FFmpeg
if shutil.which("ffmpeg") is None:
    logger.error("❌ FFmpeg is NOT installed. Audio processing will fail.")
else:
    logger.info("✅ FFmpeg found.")

# --- 📦 LIBRARIES ---
try:
    from pydub import AudioSegment, effects
    from pydub.silence import detect_leading_silence
    import edge_tts
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
    from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, CallbackQueryHandler, filters
    from google import genai
    from google.genai import types
    from faster_whisper import WhisperModel
    logger.info("✅ All libraries imported.")
except ImportError as e:
    logger.critical(f"❌ Missing Library: {e}")
    sys.exit(1)

# --- ⚙️ CONFIGURATION ---
TG_TOKEN = os.getenv("TG_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not TG_TOKEN or not GEMINI_KEY:
    logger.critical("❌ ERROR: API Keys missing.")
    sys.exit(1)

# --- 🗣️ VOICE LIBRARY ---
VOICE_LIB = {
    "🇲🇲 Thiha (Male)": "my-MM-ThihaNeural",
    "🇲🇲 Nular (Female)": "my-MM-NularNeural",
    "🇺🇸 Remy (Multi)": "en-US-RemyMultilingualNeural",
    "🇺🇸 Brian (Narrator)": "en-US-BrianNeural",
    "🇬🇧 Sonia (British)": "en-GB-SoniaNeural"
}

# --- 📝 PROMPTS (UPDATED) ---
SRT_RULES = """
**FORMATTING INSTRUCTIONS (STRICT):**
1. The input is an **SRT Subtitle File**.
2. **OUTPUT FORMAT:** You MUST return a valid SRT file.
3. **TIMESTAMPS:** Do NOT change, shift, or remove any timestamps. 
4. **SEQUENCE NUMBERS:** Preserve exact sequence.
5. **NO ENGLISH:** The output text must be 100% Burmese. No English words or characters allowed.
"""

# ✅ UPDATED PROMPT (User's Request + Strict English Ban)
BURMESE_STYLE = """
Role: Native Burmese professional video narrator and translator.
Task: Translate the content into natural, fluent Burmese as spoken by a real storyteller.

Guidelines:
• Use smooth, conversational Burmese, suitable for video narration.
• Sound natural and engaging, not formal or textbook-like.
• Do NOT add “ပေါ့” at the end of sentences.
• Translate by meaning and emotion, not word-by-word.
• Keep the flow like a continuous story, not separate sentences.
• Use expressions that native Burmese speakers actually use.
• Maintain the original tone (calm / suspense / emotional / dramatic as appropriate).
• **STRICT RULE:** NO ENGLISH CHARACTERS. If there is an English word (e.g. "Okay", "FBI"), translate it or write the sound in Burmese (e.g. "အိုကေ", "အက်ဖ်ဘီအိုင်"). Do not include English text in brackets like (English).

Output:
Only the final Burmese narration. No explanations, no extra notes.
"""

DEFAULT_PROMPTS = {
    "burmese": BURMESE_STYLE,
    "rephrase": "Rephrase this English text to be more clear, natural, and reliable."
}

# --- 📂 FOLDERS ---
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
    for f in glob.glob(f"temp/{user_id}_chunk_*.mp3"):
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
            await bot.send_message(chat_id=chat_id, text=f"```\n{chunk}\n```", parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Message Send Error: {e}")

# --- 🔊 AUDIO HELPERS ---
def trim_silence(audio_segment, silence_thresh=-40.0, chunk_size=5):
    if len(audio_segment) < 100: return audio_segment
    start_trim = detect_leading_silence(audio_segment, silence_threshold=silence_thresh, chunk_size=chunk_size)
    end_trim = detect_leading_silence(audio_segment.reverse(), silence_threshold=silence_thresh, chunk_size=chunk_size)
    duration = len(audio_segment)
    return audio_segment[start_trim:duration-end_trim]

def make_audio_crisp(audio_segment):
    clean = audio_segment.high_pass_filter(150)
    return effects.normalize(clean)

# --- 🎬 DUBBING ENGINE ---
async def generate_dubbing(user_id, srt_path, output_path, voice):
    logger.info(f"🎬 Starting Dubbing for User {user_id}...")
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

            if start_ms > current_timeline_ms:
                gap = start_ms - current_timeline_ms
                if gap > 100:
                    final_audio += AudioSegment.silent(duration=gap)
                    current_timeline_ms += gap

            temp_filename = f"temp/{user_id}_chunk_{i}.mp3"
            
            communicate = edge_tts.Communicate(text, voice, rate=f"+{BASE_RATE_VAL}%", pitch=PITCH_VAL)
            await communicate.save(temp_filename)
            
            segment = AudioSegment.from_file(temp_filename)
            segment = trim_silence(segment, silence_thresh=-40.0, chunk_size=5)

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
        logger.info("✅ Dubbing complete.")
        return True, None

    except Exception as e:
        logger.error(f"❌ Dubbing Error: {e}")
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
    logger.info(f"🎙️ [Whisper] Starting transcription...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("small", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, beam_size=5, vad_filter=True, word_timestamps=True)
        
        final_subs = []
        current_segment_words = []
        current_start = None
        MAX_CHARS_PER_BLOCK = 80
        all_words = []
        
        # ✅ FIX: HONORIFICS LIST
        # These words should NOT cause a split even if they have a dot
        HONORIFICS = ["mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "sr.", "jr.", "lt.", "gen.", "col."]

        for segment in segments:
            all_words.extend(segment.words)

        for i, word in enumerate(all_words):
            if current_start is None: current_start = word.start
            current_segment_words.append(word)
            
            text_str = " ".join([w.word.strip() for w in current_segment_words])
            clean_word = word.word.strip()
            clean_word_lower = clean_word.lower()
            
            # 1. End of Sentence (.?!) BUT NOT if it is an honorific
            has_punctuation = clean_word[-1] in ".?!" if clean_word else False
            is_honorific = clean_word_lower in HONORIFICS
            
            is_sentence_end = has_punctuation and not is_honorific

            # 2. Clause break (Comma + Length)
            is_clause_end = (clean_word[-1] == ",") and (len(text_str) > 20)
            
            # 3. Too long
            is_too_long = len(text_str) > MAX_CHARS_PER_BLOCK

            if is_sentence_end or is_clause_end or is_too_long:
                start_ts = format_timestamp(current_start)
                end_ts = format_timestamp(word.end)
                final_subs.append({"start": start_ts, "end": end_ts, "text": text_str})
                current_segment_words = []
                current_start = None

        if current_segment_words:
            start_ts = format_timestamp(current_start)
            end_ts = format_timestamp(all_words[-1].end)
            final_subs.append({"start": start_ts, "end": end_ts, "text": " ".join([w.word.strip() for w in current_segment_words])})

        with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
            for i, sub in enumerate(final_subs, start=1):
                srt.write(f"{i}\n{sub['start']} --> {sub['end']}\n{sub['text']}\n\n")
                txt.write(f"{sub['text']} ")
        logger.info("✅ Whisper Done.")
        return "Whisper (Smart)"
    except Exception as e:
        logger.error(f"Whisper Error: {e}")
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
    if user_id in user_last_active and (current_time - user_last_active[user_id] > 86400):
        chat_histories[user_id] = []
    user_last_active[user_id] = current_time

    if user_id not in chat_histories: chat_histories[user_id] = []
    client = genai.Client(api_key=GEMINI_KEY)
    
    try:
        chat = client.chats.create(model='gemini-2.0-flash', history=chat_histories[user_id])
        response = chat.send_message(text)
        chat_histories[user_id] = chat.history 
        return response.text
    except Exception as e:
        logger.error(f"Gemini Chat Error: {e}")
        return f"Gemini Error: {e}"

# --- 🤖 BOT COMMANDS & HANDLERS ---
async def post_init(application):
    logger.info("🤖 Bot is initializing commands...")
    await application.bot.set_my_commands([
        BotCommand("start", "🏠 Dashboard"),
        BotCommand("voices", "🗣️ Change Voice"),
        BotCommand("translate", "🌍 Translate"),
        BotCommand("dub", "🎬 Start Dubbing"),
        BotCommand("heygemini", "🤖 Chat AI"),
        BotCommand("clearall", "🧹 Reset All")
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"📩 Received /start from {update.effective_user.id}")
    try:
        user_id = update.effective_user.id
        state = get_user_state(user_id)
        v_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Unknown")
        
        text = (
            f"👋 **Welcome to Video AI Studio!**\n\n"
            f"⚙️ **Current Settings:**\n"
            f"├ 🎙️ **Engine:** `{state['transcribe_engine'].title()}`\n"
            f"└ 🗣️ **Voice:** `{v_name}`\n\n"
            f"👇 **What would you like to do?**"
        )
        
        keyboard = [
            [InlineKeyboardButton("🗣️ Select Voice", callback_data="cmd_voices"), InlineKeyboardButton("🎙️ Switch Engine", callback_data="toggle_transcribe")],
            [InlineKeyboardButton("📝 Edit Prompts", callback_data="menu_settings"), InlineKeyboardButton("🤖 Chat AI", callback_data="cmd_chat")],
            [InlineKeyboardButton("🧹 Clear Data", callback_data="cmd_clear")]
        ]
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"CRASH in start command: {e}")
        await update.message.reply_text("❌ Error starting bot. Check console logs.")

async def enable_chat_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = "chat_gemini"
    await update.message.reply_text("🤖 **Gemini Chat Mode ON**\nType `/cancel` to exit.")

async def voices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for name, code in VOICE_LIB.items():
        row.append(InlineKeyboardButton(name, callback_data=f"set_voice_{code}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row: keyboard.append(row)
    
    msg_text = "🗣️ **Select a High-Quality Narrator:**\n(Click to listen to a sample)"
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
        await update.callback_query.message.edit_text("⚙️ **Prompt Settings**", reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text("⚙️ **Prompt Settings**", reply_markup=InlineKeyboardMarkup(keyboard))

async def perform_dubbing(update, context):
    user_id = update.effective_user.id
    msg = update.effective_message
    p = get_paths(user_id)
    state = get_user_state(user_id)

    if not os.path.exists(p['srt']):
        await msg.reply_text("❌ **No SRT found.** Send a file first.")
        return

    voice_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Voice")
    status = await msg.reply_text(f"🎬 **Dubbing with {voice_name}...**")
    
    success, error = await generate_dubbing(user_id, p['srt'], p['dub_audio'], state['dub_voice'])
    
    if success:
        await status.delete()
        await context.bot.send_audio(chat_id=msg.chat_id, audio=open(p['dub_audio'], "rb"), title=f"Dubbed_{voice_name}", caption=f"✅ **Dubbed by {voice_name}!**")
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
    logger.info(f"🔘 Button Clicked: {data}")
    
    try:
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
            await query.message.reply_text("🤖 **Gemini Chat Mode ON**\nType `/cancel` to exit.")
            await query.answer()

        elif data == "cmd_clear":
            wipe_user_data(user_id)
            await query.answer("All data cleared.")
            await query.message.reply_text("🧹 **Workspace Cleared.**")

        elif data.startswith("set_voice_"):
            new_voice = data.replace("set_voice_", "")
            state['dub_voice'] = new_voice
            v_name = next((k for k, v in VOICE_LIB.items() if v == new_voice), "Custom")
            
            await query.message.edit_text(f"✅ Voice set to: **{v_name}**\n⏳ Generating sample...")
            
            if "my-MM" in new_voice: sample_text = "မင်္ဂလာပါ၊ ဒါက ကျွန်တော့်ရဲ့ အသံနမူနာပါ။"
            elif "it-IT" in new_voice: sample_text = "Ciao, questo è un campione della mia voce."
            else: sample_text = "Hello, this is a quick sample of my voice."
            
            sample_path = f"temp/sample_{user_id}.mp3"
            try:
                communicate = edge_tts.Communicate(sample_text, new_voice)
                await communicate.save(sample_path)
                await context.bot.send_voice(chat_id=query.message.chat_id, voice=open(sample_path, "rb"), caption=f"🎙️ **{v_name}**")
            except Exception as e:
                logger.error(f"TTS Error: {e}")
                await context.bot.send_message(chat_id=query.message.chat_id, text="❌ Could not generate sample.")

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
            
    except Exception as e:
        logger.error(f"Callback Error: {e}")

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    text = msg.text
    mode = user_modes.get(user_id)
    state = get_user_state(user_id)
    p = get_paths(user_id)
    
    logger.info(f"📩 Text received: {text[:20]}...")

    if text.startswith("/cancel"):
        user_modes[user_id] = None
        await msg.reply_text("✅ Mode exited.")
        return

    # SRT Direct Paste
    if re.search(r'\d{2}:\d{2}:\d{2},\d{3} -->', text):
        with open(p['srt'], 'w', encoding="utf-8") as f: f.write(text)
        keyboard = [[InlineKeyboardButton("🎬 Dub Audio", callback_data="trigger_dub")]]
        await msg.reply_text("✅ **SRT Text Saved.**", reply_markup=InlineKeyboardMarkup(keyboard))
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
        await msg.reply_text(f"✅ **{key.title()} Prompt Updated.**")
        return

    if "http" in text:
        await process_media(update, context, is_url=True)
        return

    if len(text) > 5:
        with open(p['txt'], "w", encoding="utf-8") as f: f.write(text)
        await msg.reply_text("✅ **Text Saved.** Type `/translate` to process.")

async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    logger.info("📩 File received.")
    
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
                await context.bot.send_document(msg.chat_id, open(p['srt'], "rb"), caption="🎬 **SRT Generated (Smart Clause)**")
        else:
            await loop.run_in_executor(None, run_gemini_transcribe, p['audio'], p['srt'], p['txt'])
            if os.path.exists(p['txt']):
                 await context.bot.send_document(msg.chat_id, open(p['txt'], "rb"), caption="📄 **Transcript Generated**")
            
        await status.edit_text("✅ **Done!** Type `/translate` to translate or `/dub` to dub.")

    except Exception as e:
        logger.error(f"Processing Error: {e}")
        await status.edit_text(f"❌ Processing Error: {e}")

if __name__ == '__main__':
    logger.info("🚀 Video AI Bot STARTING...")
    
    try:
        app = ApplicationBuilder().token(TG_TOKEN).post_init(post_init).build()
        
        # Commands
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("voices", voices_command))
        app.add_handler(CommandHandler("settings", settings_command))
        app.add_handler(CommandHandler("translate", lambda u, c: u.message.reply_text("🌍 Options:", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("To Burmese", callback_data="trans_burmese")]
        ]))))
        app.add_handler(CommandHandler("dub", perform_dubbing))
        app.add_handler(CommandHandler("heygemini", enable_chat_mode))
        app.add_handler(CommandHandler("clearall", lambda u, c: wipe_user_data(u.effective_user.id)))
        app.add_handler(CommandHandler("cancel", lambda u, c: u.message.reply_text("✅ Cancelled.")))

        # Handlers
        app.add_handler(CallbackQueryHandler(callback_handler))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
        app.add_handler(MessageHandler(filters.VIDEO | filters.Document.ALL | filters.AUDIO, file_handler))
        
        logger.info("✅ Bot is polling now. Send /start in Telegram.")
        app.run_polling()
        
    except Exception as e:
        logger.critical(f"🔥 FATAL ERROR: {e}")
