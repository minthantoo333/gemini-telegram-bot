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
from io import StringIO

# --- 🔍 DIAGNOSTICS & LOGGING ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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

# --- 🗣️ VOICE LIBRARY (EXPANDED) ---
VOICE_LIB = {
    # 🇲🇲 BURMESE (Only 2 Available)
    "🇲🇲 Thiha (Male)": "my-MM-ThihaNeural",
    "🇲🇲 Nilar (Female)": "my-MM-NilarNeural",

    # 🇺🇸 US ENGLISH (Popular)
    "🇺🇸 Guy (Male)": "en-US-GuyNeural",
    "🇺🇸 Jenny (Female)": "en-US-JennyNeural",
    "🇺🇸 Aria (Good Narrator)": "en-US-AriaNeural",
    "🇺🇸 Brian (Narrator)": "en-US-BrianNeural",
    "🇺🇸 Christopher (Deep)": "en-US-ChristopherNeural",
    "🇺🇸 Eric (News)": "en-US-EricNeural",
    "🇺🇸 Michelle (Pro)": "en-US-MichelleNeural",
    "🇺🇸 Roger (Story)": "en-US-RogerNeural",
    "🇺🇸 Steffan (Bold)": "en-US-SteffanNeural",

    # 🇺🇸 US ENGLISH (Special)
    "🇺🇸 Ana (Child)": "en-US-AnaNeural",
    "🇺🇸 Ava (Multilingual)": "en-US-AvaMultilingualNeural",
    "🇺🇸 Andrew (Multilingual)": "en-US-AndrewMultilingualNeural",
    "🇺🇸 Emma (Multilingual)": "en-US-EmmaMultilingualNeural",
    "🇺🇸 Remy (Multilingual)": "fr-FR-RemyMultilingualNeural",

    # 🇬🇧 UK ENGLISH
    "🇬🇧 Sonia (British)": "en-GB-SoniaNeural",
    "🇬🇧 Ryan (British)": "en-GB-RyanNeural",
    "🇬🇧 Libby (British)": "en-GB-LibbyNeural",

    # 🌏 OTHERS (Just in case)
    "🇨🇳 Xiaoxiao (Chinese)": "zh-CN-XiaoxiaoNeural",
    "🇯🇵 Nanami (Japanese)": "ja-JP-NanamiNeural",
    "🇰🇷 SunHi (Korean)": "ko-KR-SunHiNeural",
    "🇹🇭 Premwadee (Thai)": "th-TH-PremwadeeNeural",

    # Additional Multilingual Voices
    "🇮🇹 Giuseppe (Male)": "it-IT-GiuseppeNeural",
    "🇮🇹 Isabella (Female)": "it-IT-IsabellaNeural",
    "🇫🇷 Henri (Male)": "fr-FR-HenriNeural",
    "🇫🇷 Denise (Female)": "fr-FR-DeniseNeural",
    "🇩🇪 Katja (Female)": "de-DE-KatjaNeural",
    "🇪🇸 Alvaro (Male)": "es-ES-AlvaroNeural",
    "🇪🇸 Elvira (Female)": "es-ES-ElviraNeural",
}


# --- 📝 PROMPTS ---
SRT_RULES = """
**FORMATTING INSTRUCTIONS (STRICT):**
1. The input is an **SRT Subtitle File**.
2. **OUTPUT FORMAT:** You MUST return a valid SRT file.
3. **TIMESTAMPS:** Do NOT change, shift, or remove any timestamps. 
4. **SEQUENCE NUMBERS:** Preserve exact sequence.
5. **NO ENGLISH:** The output text must be 100% Burmese. No English words or characters allowed.
"""

# ✅ PROMPT UPDATED: Ensures "Original Taste"
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
• **CRITICAL:** Maintain the original tone and "taste" of the video (e.g., if the original is suspenseful, sound suspenseful; if funny, sound funny).
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
user_srt_msgs = {}  # To track message IDs for deletion
user_srt_accum = {}  # To accumulate multi-part SRT

# --- 🛠️ HELPER FUNCTIONS ---
def get_user_state(user_id):
    if user_id not in user_prefs:
        user_prefs[user_id] = {
            "transcribe_engine": "whisper_dub",  # Default to Dub mode
            "dub_voice": "my-MM-ThihaNeural", 
            "custom_prompts": {},
            "transcript_format": "srt"  # Default to SRT
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
    for f in glob.glob(f"downloads/{user_id}_subs*"):
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
    if user_id in user_srt_msgs: del user_srt_msgs[user_id]
    if user_id in user_srt_accum: del user_srt_accum[user_id]

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

# --- 🎬 DUBBING ENGINE (VOICERTOOL STYLE: SMART DENSITY) ---
async def generate_dubbing(user_id, srt_path, output_path, voice):
    """
    Voicertool Logic:
    1. Analyzes text density (Characters per Second) BEFORE generating.
    2. Sets the TTS speed perfectly to match the time slot.
    3. Trims silence aggressively to fit without sounding "rushed".
    """
    logger.info(f"🎬 Starting Dubbing (Smart Density) for {user_id}...")
    try:
        subs = pysrt.open(srt_path)
        final_audio = AudioSegment.empty()
        current_timeline_ms = 0
        
        # Voicertool often defaults to slightly faster for energy
        DEFAULT_SPEED = 15 # +15% Base

        for i, sub in enumerate(subs):
            start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
            end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
            duration_allowed = end_ms - start_ms
            
            text = sub.text.replace("\n", " ").strip()
            if not text: continue 

            # --- 1. SMART SPEED CALCULATION (The Secret) ---
            # Estimate how fast we need to speak to fit this slot
            # Avg Burmese speaking rate is ~12-15 chars per second
            char_count = len(text)
            duration_sec = duration_allowed / 1000.0
            
            if duration_sec == 0: duration_sec = 1 # Prevent divide by zero
            chars_per_sec = char_count / duration_sec
            
            # Logic: If density is high (>18 chars/s), speed up. If low, stay natural.
            if chars_per_sec > 18:
                # Need to be fast
                calculated_rate = int(DEFAULT_SPEED + ((chars_per_sec - 18) * 5))
            elif chars_per_sec < 10:
                # Can be relaxed
                calculated_rate = 10
            else:
                # Normal flow
                calculated_rate = DEFAULT_SPEED
            
            # CAP the speed limits (Voicertool rarely goes above +50%)
            if calculated_rate > 50: calculated_rate = 50
            if calculated_rate < 0: calculated_rate = 0

            # --- 2. GENERATE WITH CALCULATED RATE ---
            temp_filename = f"temp/{user_id}_chunk_{i}.mp3"
            communicate = edge_tts.Communicate(text, voice, rate=f"+{calculated_rate}%", pitch="-2Hz")
            await communicate.save(temp_filename)
            
            segment = AudioSegment.from_file(temp_filename)
            
            # --- 3. AGGRESSIVE SILENCE TRIMMING ---
            # Remove start/end silence to fit better
            segment = trim_silence(segment, silence_thresh=-40.0, chunk_size=5)

            # --- 4. FINAL FIT CHECK ---
            # If it's STILL too long (rare case), use high-quality compression
            # instead of re-generating (which causes the chipmunk effect)
            current_len = len(segment)
            if current_len > duration_allowed + 100: # Allow 100ms buffer
                # Squeeze using Pydub (Time Stretch) - smoother than TTS rate change
                speedup_factor = current_len / duration_allowed
                # Cap squeeze at 1.3x to prevent distortion
                if speedup_factor > 1.3: speedup_factor = 1.3
                
                # Use simple speedup (affects pitch slightly but keeps clarity better than harsh cuts)
                segment = segment.speedup(playback_speed=speedup_factor, chunk_size=150, crossfade=25)

            # --- 5. SYNC ON TIMELINE ---
            if start_ms > current_timeline_ms:
                gap = start_ms - current_timeline_ms
                if gap > 0:
                    final_audio += AudioSegment.silent(duration=gap)
                    current_timeline_ms += gap
            
            # Crisp Filter
            segment = make_audio_crisp(segment)
            
            final_audio += segment
            current_timeline_ms += len(segment)
            
            if os.path.exists(temp_filename): os.remove(temp_filename)

        final_audio.export(output_path, format="mp3")
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

# 1️⃣ OLD WAY (Strict Characters) - For Subtitles
def run_whisper_sub(audio_path, srt_path, txt_path):
    logger.info(f"🎙️ [Whisper] Starting Subtitle Mode...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("small", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, beam_size=5, vad_filter=True, word_timestamps=True)
        
        final_subs = []
        current_segment_words = []
        current_start = None
        
        SOFT_LIMIT = 80
        HARD_LIMIT = 120
        HONORIFICS = ["mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "sr.", "jr.", "lt.", "gen.", "col."]

        all_words = []
        for segment in segments:
            all_words.extend(segment.words)

        for i, word in enumerate(all_words):
            if current_start is None: current_start = word.start
            current_segment_words.append(word)
            
            text_str = " ".join([w.word.strip() for w in current_segment_words])
            current_len = len(text_str)
            clean_word = word.word.strip()
            clean_word_lower = clean_word.lower()
            
            has_punctuation = clean_word[-1] in ".?!" if clean_word else False
            is_honorific = clean_word_lower in HONORIFICS
            is_sentence_end = has_punctuation and not is_honorific
            is_clause_end = (clean_word[-1] == ",") and (current_len > SOFT_LIMIT)
            is_too_long = current_len > HARD_LIMIT

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
        return "Whisper (Sub Mode)"
    except Exception as e:
        return f"Error: {e}"

# 2️⃣ NEW WAY (Sentence Flow) - For Dubbing
def run_whisper_dub(audio_path, srt_path, txt_path):
    logger.info(f"🎙️ [Whisper] Starting Dubbing Mode...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("small", device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, beam_size=5, vad_filter=True, word_timestamps=True)
        
        final_subs = []
        current_segment_words = []
        current_start = None
        
        HARD_LIMIT = 300  
        PAUSE_THRESHOLD = 0.8 
        HONORIFICS = ["mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "sr.", "jr.", "lt.", "gen.", "col."]
        
        all_words = []
        for segment in segments:
            all_words.extend(segment.words)

        previous_end_time = 0

        for i, word in enumerate(all_words):
            if current_start is None: current_start = word.start
            
            time_since_last_word = word.start - previous_end_time
            is_long_pause = (time_since_last_word > PAUSE_THRESHOLD) and (len(current_segment_words) > 0)
            
            if is_long_pause:
                 text_str = " ".join([w.word.strip() for w in current_segment_words])
                 start_ts = format_timestamp(current_start)
                 end_ts = format_timestamp(previous_end_time)
                 final_subs.append({"start": start_ts, "end": end_ts, "text": text_str})
                 current_segment_words = [word]
                 current_start = word.start
                 previous_end_time = word.end
                 continue

            current_segment_words.append(word)
            previous_end_time = word.end
            
            text_str = " ".join([w.word.strip() for w in current_segment_words])
            current_len = len(text_str)
            clean_word = word.word.strip()
            clean_word_lower = clean_word.lower()
            
            has_punctuation = clean_word[-1] in ".?!" if clean_word else False
            is_honorific = clean_word_lower in HONORIFICS
            is_sentence_end = has_punctuation and not is_honorific
            is_too_long = current_len > HARD_LIMIT

            if is_sentence_end or is_too_long:
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
        return "Whisper (Dub Mode)"
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
        BotCommand("clearall", "🧹 Reset All"),
        BotCommand("srt", "📝 Start multi-part SRT input"),
        BotCommand("end", "🏁 End multi-part input")
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"📩 Received /start from {update.effective_user.id}")
    try:
        user_id = update.effective_user.id
        state = get_user_state(user_id)
        v_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Unknown")
        
        # Friendly name for engine
        engine_name = "Whisper (Dub Mode)" if state['transcribe_engine'] == 'whisper_dub' else \
                      "Whisper (Sub Mode)" if state['transcribe_engine'] == 'whisper_sub' else "Gemini"

        format_name = state['transcript_format'].upper()

        text = (
            f"👋 **Welcome to Video AI Studio!**\n\n"
            f"⚙️ **Current Settings:**\n"
            f"├ 🎙️ **Engine:** `{engine_name}`\n"
            f"├ 📄 **Format:** `{format_name}`\n"
            f"└ 🗣️ **Voice:** `{v_name}`\n\n"
            f"👇 **What would you like to do?**"
        )
        
        keyboard = [
            [InlineKeyboardButton("🎙️ Set: Sub Mode", callback_data="set_eng_sub"), InlineKeyboardButton("🎙️ Set: Dub Mode", callback_data="set_eng_dub")],
            [InlineKeyboardButton("📄 Set: SRT", callback_data="set_format_srt"), InlineKeyboardButton("📄 Set: TXT", callback_data="set_format_txt")],
            [InlineKeyboardButton("🗣️ Select Voice", callback_data="cmd_voices"), InlineKeyboardButton("🤖 Chat AI", callback_data="cmd_chat")],
            [InlineKeyboardButton("📝 Edit Prompts", callback_data="menu_settings"), InlineKeyboardButton("🧹 Clear Data", callback_data="cmd_clear")]
        ]
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"CRASH in start command: {e}")
        await update.message.reply_text("❌ Error starting bot. Check console logs.")

async def enable_chat_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = "chat_gemini"
    await update.message.reply_text("🤖 **Gemini Chat Mode ON**\nType `/cancel` to exit.")

async def start_srt_accum(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = "srt_accumulate"
    user_srt_accum[user_id] = ""
    user_srt_msgs[user_id] = []
    await update.message.reply_text("📝 **SRT Input Mode ON**\nSend SRT parts one by one. Finish with /end.")

async def end_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    mode = user_modes.get(user_id)
    if mode != "srt_accumulate":
        await update.message.reply_text("❌ Not in SRT input mode.")
        return
    full_srt = user_srt_accum.get(user_id, "")
    if not full_srt:
        await update.message.reply_text("❌ No SRT content provided.")
        user_modes[user_id] = None
        return
    try:
        subs = pysrt.open(StringIO(full_srt))
        duration = subs[-1].end
        dur_str = f"{duration.hours:02}:{duration.minutes:02}:{duration.seconds:02},{duration.milliseconds:03}"
        keyboard = [
            [InlineKeyboardButton("✅ Confirm", callback_data="confirm_srt"), InlineKeyboardButton("❌ Cancel", callback_data="cancel_srt")]
        ]
        await update.message.reply_text(f"🕒 **Total Duration: {dur_str}**\nConfirm SRT?", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        await update.message.reply_text(f"❌ Invalid SRT: {e}")
    user_modes[user_id] = None

async def voices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    
    # Create buttons in pairs of 2
    for name, code in VOICE_LIB.items():
        row.append(InlineKeyboardButton(name, callback_data=f"set_voice_{code}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    
    # Add any remaining single button
    if row: keyboard.append(row)
    
    msg_text = (
        "🗣️ **Voice Library**\n"
        "Note: For Burmese text, you MUST use **Thiha** or **Nular**.\n"
        "Other voices are for English dubbing."
    )
    
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
        # Delete user's SRT messages
        for mid in user_srt_msgs.get(user_id, []):
            try:
                await context.bot.delete_message(chat_id=msg.chat_id, message_id=mid)
            except:
                pass
        user_srt_msgs[user_id] = []
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

        # ✅ SWITCH ENGINES MANUALLY
        elif data == "set_eng_sub":
            state['transcribe_engine'] = "whisper_sub"
            await query.answer("Engine: Whisper Subtitle Mode")
            await start(update, context)

        elif data == "set_eng_dub":
            state['transcribe_engine'] = "whisper_dub"
            await query.answer("Engine: Whisper Dubbing Mode")
            await start(update, context)

        # Set transcript format
        elif data == "set_format_srt":
            state['transcript_format'] = "srt"
            await query.answer("Format: SRT")
            await start(update, context)

        elif data == "set_format_txt":
            state['transcript_format'] = "txt"
            await query.answer("Format: TXT")
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

        elif data == "confirm_srt":
            p = get_paths(user_id)
            full_srt = user_srt_accum.get(user_id, "")
            with open(p['srt'], 'w', encoding="utf-8") as f: f.write(full_srt)
            keyboard = [[InlineKeyboardButton("🎬 Dub Audio", callback_data="trigger_dub")]]
            await query.message.reply_text("✅ **SRT Confirmed and Saved.**", reply_markup=InlineKeyboardMarkup(keyboard))
            await query.answer()

        elif data == "cancel_srt":
            if user_id in user_srt_accum: del user_srt_accum[user_id]
            if user_id in user_srt_msgs: del user_srt_msgs[user_id]
            await query.message.reply_text("❌ **SRT Input Cancelled.**")
            await query.answer()
            
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

    if mode == "srt_accumulate":
        user_srt_accum[user_id] += "\n" + text
        user_srt_msgs[user_id].append(msg.message_id)
        await msg.reply_text("➕ Part added. Continue or /end.")
        return

    # SRT Direct Paste (single message)
    if re.search(r'\d{2}:\d{2}:\d{2},\d{3} -->', text):
        full_text = text
        try:
            subs = pysrt.open(StringIO(full_text))
            dur = subs[-1].end
            dur_str = f"{dur.hours:02}:{dur.minutes:02}:{dur.seconds:02},{dur.milliseconds:03}"
            keyboard = [
                [InlineKeyboardButton("✅ Confirm", callback_data="confirm_srt"), InlineKeyboardButton("❌ Cancel", callback_data="cancel_srt")]
            ]
            await msg.reply_text(f"🕒 **Duration: {dur_str}**\nConfirm SRT?", reply_markup=InlineKeyboardMarkup(keyboard))
            user_srt_accum[user_id] = full_text  # Reuse accum for single
            user_srt_msgs[user_id] = [msg.message_id]
        except:
            await msg.reply_text("❌ Invalid SRT format.")
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

    if "http" in text.lower() and ("youtube.com" in text.lower() or "youtu.be" in text.lower() or "tiktok.com" in text.lower()):
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
    
    status = await msg.reply_text("⏳ **Downloading and Processing Media...**")
    try:
        clean_temp(user_id)
        if is_url:
            url = msg.text.strip()
            # Download auto subs first
            subprocess.run(f"yt-dlp --write-auto-sub --convert-subs srt --sub-langs en --skip-download -o downloads/{user_id}_subs '%(url)s'", shell=True, input=url.encode())
            subs_files = glob.glob(f"downloads/{user_id}_subs*.srt")
            has_subs = False
            if subs_files:
                subs_path = subs_files[0]
                shutil.move(subs_path, p['srt'])
                subs = pysrt.open(p['srt'])
                with open(p['txt'], 'w', encoding="utf-8") as f:
                    f.write(' '.join(sub.text for sub in subs))
                has_subs = True
                caption = "✅ **Auto-Generated Subtitles Downloaded.**"
            # Download audio
            subprocess.run(f"yt-dlp -x --audio-format mp3 -o '{p['audio']}' '{url}'", shell=True)
            if not has_subs:
                await status.edit_text("⏳ **No Auto Subs Found. Transcribing with AI...**")
        else:
            file_obj = await (msg.video or msg.document or msg.audio).get_file()
            await file_obj.download_to_drive(p['input'])
            subprocess.run(f"ffmpeg -y -i {p['input']} -vn -acodec libmp3lame -q:a 2 {p['audio']}", shell=True)
        
        if not has_subs:
            loop = asyncio.get_event_loop()
            
            # ✅ DECIDE WHICH ENGINE TO RUN
            if state['transcribe_engine'] == "whisper_sub":
                await loop.run_in_executor(None, run_whisper_sub, p['audio'], p['srt'], p['txt'])
                caption = "🎬 **Subtitle Mode (Split by chars)**"
            elif state['transcribe_engine'] == "whisper_dub":
                await loop.run_in_executor(None, run_whisper_dub, p['audio'], p['srt'], p['txt'])
                caption = "🎬 **Dubbing Mode (Split by Sentence)**"
            else:
                await loop.run_in_executor(None, run_gemini_transcribe, p['audio'], p['srt'], p['txt'])
                caption = "📄 **Transcript (Gemini)**"

        # Send preferred format
        pref = state['transcript_format']
        if pref == "srt" and os.path.exists(p['srt']):
            await context.bot.send_document(msg.chat_id, open(p['srt'], "rb"), caption=caption)
        elif pref == "txt" and os.path.exists(p['txt']):
            await context.bot.send_document(msg.chat_id, open(p['txt'], "rb"), caption=caption)
        elif os.path.exists(p['srt']):
            await context.bot.send_document(msg.chat_id, open(p['srt'], "rb"), caption=caption)
        elif os.path.exists(p['txt']):
            await context.bot.send_document(msg.chat_id, open(p['txt'], "rb"), caption=caption)
            
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
        app.add_handler(CommandHandler("srt", start_srt_accum))
        app.add_handler(CommandHandler("end", end_input))

        # Handlers
        app.add_handler(CallbackQueryHandler(callback_handler))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), text_handler))
        app.add_handler(MessageHandler(filters.VIDEO | filters.Document.ALL | filters.AUDIO, file_handler))
        
        logger.info("✅ Bot is polling now. Send /start in Telegram.")
        app.run_polling()
        
    except Exception as e:
        logger.critical(f"🔥 FATAL ERROR: {e}")