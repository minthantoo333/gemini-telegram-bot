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

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

if shutil.which("ffmpeg") is None:
    logger.error("❌ FFmpeg is NOT installed. Audio processing will fail.")
else:
    logger.info("✅ FFmpeg found.")

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

TG_TOKEN = os.getenv("TG_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not TG_TOKEN or not GEMINI_KEY:
    logger.critical("❌ ERROR: API Keys missing.")
    sys.exit(1)

VOICE_LIB = {
    "🇲🇲 Thiha (Male)": "my-MM-ThihaNeural",
    "🇲🇲 Nilar (Female)": "my-MM-NilarNeural",
    "🇺🇸 Guy (Male)": "en-US-GuyNeural",
    "🇺🇸 Jenny (Female)": "en-US-JennyNeural",
    "🇺🇸 Aria (Good Narrator)": "en-US-AriaNeural",
    "🇺🇸 Brian (Narrator)": "en-US-BrianNeural",
    "🇺🇸 Christopher (Deep)": "en-US-ChristopherNeural",
    "🇺🇸 Eric (News)": "en-US-EricNeural",
    "🇺🇸 Michelle (Pro)": "en-US-MichelleNeural",
    "🇺🇸 Roger (Story)": "en-US-RogerNeural",
    "🇺🇸 Steffan (Bold)": "en-US-SteffanNeural",
    "🇺🇸 Ana (Child)": "en-US-AnaNeural",
    "🇺🇸 Ava (Multilingual)": "en-US-AvaMultilingualNeural",
    "🇺🇸 Andrew (Multilingual)": "en-US-AndrewMultilingualNeural",
    "🇺🇸 Emma (Multilingual)": "en-US-EmmaMultilingualNeural",
    "🇫🇷 Remy (Multilingual)": "fr-FR-RemyMultilingualNeural",
    "🇬🇧 Sonia (British)": "en-GB-SoniaNeural",
    "🇬🇧 Ryan (British)": "en-GB-RyanNeural",
    "🇬🇧 Libby (British)": "en-GB-LibbyNeural",
    "🇨🇳 Xiaoxiao (Chinese)": "zh-CN-XiaoxiaoNeural",
    "🇯🇵 Nanami (Japanese)": "ja-JP-NanamiNeural",
    "🇰🇷 SunHi (Korean)": "ko-KR-SunHiNeural",
    "🇹🇭 Premwadee (Thai)": "th-TH-PremwadeeNeural",
    "🇮🇹 Giuseppe (Male)": "it-IT-GiuseppeNeural",
    "🇮🇹 Isabella (Female)": "it-IT-IsabellaNeural",
    "🇫🇷 Henri (Male)": "fr-FR-HenriNeural",
    "🇫🇷 Denise (Female)": "fr-FR-DeniseNeural",
    "🇩🇪 Katja (Female)": "de-DE-KatjaNeural",
    "🇪🇸 Alvaro (Male)": "es-ES-AlvaroNeural",
    "🇪🇸 Elvira (Female)": "es-ES-ElviraNeural",
}

SRT_RULES = """
**FORMATTING INSTRUCTIONS (STRICT):**
1. The input is an **SRT Subtitle File**.
2. **OUTPUT FORMAT:** You MUST return a valid SRT file.
3. **TIMESTAMPS:** Do NOT change, shift, or remove any timestamps. 
4. **SEQUENCE NUMBERS:** Preserve exact sequence.
5. **NO ENGLISH:** The output text must be 100% Burmese. No English words or characters allowed.
"""

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
• **CRITICAL:** Maintain the original tone and "taste" of the video.
• **STRICT RULE:** NO ENGLISH CHARACTERS.
"""

DEFAULT_PROMPTS = {
    "burmese": BURMESE_STYLE,
    "rephrase": "Rephrase this English text to be more clear, natural, and reliable."
}

BASE_FOLDERS = ["downloads", "temp"]
for f in BASE_FOLDERS:
    os.makedirs(f, exist_ok=True)

user_prefs = {}
user_modes = {}
chat_histories = {}
user_last_active = {}
user_srt_msgs = {}
user_srt_accum = {}

def get_user_state(user_id):
    if user_id not in user_prefs:
        user_prefs[user_id] = {
            "transcribe_engine": "whisper_dub",
            "dub_voice": "my-MM-ThihaNeural",
            "custom_prompts": {},
            "transcript_format": "srt"
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
    for pattern in [p['input'], p['audio'], p['srt'], p['txt'], p['dub_audio']]:
        if os.path.exists(pattern):
            try: os.remove(pattern)
            except: pass
    for f in glob.glob(f"temp/{user_id}_*"):
        try: os.remove(f)
        except: pass
    for f in glob.glob(f"downloads/{user_id}_video*"):
        try: os.remove(f)
        except: pass

def wipe_user_data(user_id):
    clean_temp(user_id)
    for pattern in glob.glob(f"downloads/{user_id}_*"):
        try: os.remove(pattern)
        except: pass
    user_prefs.pop(user_id, None)
    user_modes.pop(user_id, None)
    chat_histories.pop(user_id, None)
    user_srt_msgs.pop(user_id, None)
    user_srt_accum.pop(user_id, None)

async def send_copyable_message(chat_id, bot, text):
    if not text: return
    MAX_LEN = 4000
    safe_text = text.replace("`", "'")
    for i in range(0, len(safe_text), MAX_LEN):
        chunk = safe_text[i:i + MAX_LEN]
        try:
            await bot.send_message(chat_id=chat_id, text=f"```\n{chunk}\n```", parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Message send error: {e}")

def trim_silence(audio_segment, silence_thresh=-40.0, chunk_size=5):
    if len(audio_segment) < 100: return audio_segment
    start_trim = detect_leading_silence(audio_segment, silence_threshold=silence_thresh, chunk_size=chunk_size)
    end_trim = detect_leading_silence(audio_segment.reverse(), silence_threshold=silence_thresh, chunk_size=chunk_size)
    duration = len(audio_segment)
    return audio_segment[start_trim:duration - end_trim]

def make_audio_crisp(audio_segment):
    clean = audio_segment.high_pass_filter(150)
    return effects.normalize(clean)

async def generate_dubbing(user_id, srt_path, output_path, voice):
    logger.info(f"Starting dubbing for {user_id}")
    try:
        subs = pysrt.open(srt_path)
        final_audio = AudioSegment.empty()
        current_timeline_ms = 0
        DEFAULT_SPEED = 15

        for i, sub in enumerate(subs):
            start_ms = sub.start.ordinal
            end_ms = sub.end.ordinal
            duration_allowed = end_ms - start_ms
            text = sub.text.replace("\n", " ").strip()
            if not text: continue

            char_count = len(text)
            duration_sec = duration_allowed / 1000.0 or 1
            chars_per_sec = char_count / duration_sec

            if chars_per_sec > 18:
                rate = int(DEFAULT_SPEED + ((chars_per_sec - 18) * 5))
            elif chars_per_sec < 10:
                rate = 10
            else:
                rate = DEFAULT_SPEED
            rate = max(0, min(50, rate))

            temp_filename = f"temp/{user_id}_chunk_{i}.mp3"
            communicate = edge_tts.Communicate(text, voice, rate=f"+{rate}%", pitch="-2Hz")
            await communicate.save(temp_filename)

            segment = AudioSegment.from_file(temp_filename)
            segment = trim_silence(segment)

            if len(segment) > duration_allowed + 100:
                factor = min(1.3, len(segment) / duration_allowed)
                segment = segment.speedup(playback_speed=factor, chunk_size=150, crossfade=25)

            if start_ms > current_timeline_ms:
                gap = start_ms - current_timeline_ms
                final_audio += AudioSegment.silent(duration=gap)
                current_timeline_ms += gap

            segment = make_audio_crisp(segment)
            final_audio += segment
            current_timeline_ms += len(segment)

            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        final_audio.export(output_path, format="mp3")
        return True, None
    except Exception as e:
        logger.error(f"Dubbing failed: {e}")
        return False, str(e)

def format_timestamp(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{math.floor(seconds):02d},{milliseconds:03d}"

def run_whisper_sub(audio_path, srt_path, txt_path):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("small", device=device, compute_type=compute_type)

        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            language=None
        )

        logger.info(f"Detected: {info.language} (prob: {info.language_probability:.2f})")

        final_subs = []
        current_words = []
        current_start = None
        SOFT_LIMIT = 80
        HARD_LIMIT = 120
        HONORIFICS = ["mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "sr.", "jr.", "lt.", "gen.", "col."]

        all_words = [w for seg in segments for w in seg.words]

        for word in all_words:
            if current_start is None:
                current_start = word.start
            current_words.append(word)

            text_str = " ".join(w.word.strip() for w in current_words)
            current_len = len(text_str)
            clean = word.word.strip().lower()

            has_punct = clean and clean[-1] in ".?!"
            is_honorific = clean in HONORIFICS
            is_end = has_punct and not is_honorific
            is_comma_break = clean and clean[-1] == "," and current_len > SOFT_LIMIT
            too_long = current_len > HARD_LIMIT

            if is_end or is_comma_break or too_long:
                start_ts = format_timestamp(current_start)
                end_ts = format_timestamp(word.end)
                final_subs.append({"start": start_ts, "end": end_ts, "text": text_str})
                current_words = []
                current_start = None

        if current_words:
            start_ts = format_timestamp(current_start)
            end_ts = format_timestamp(all_words[-1].end)
            final_subs.append({"start": start_ts, "end": end_ts, "text": " ".join(w.word.strip() for w in current_words)})

        with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
            for i, sub in enumerate(final_subs, 1):
                srt.write(f"{i}\n{sub['start']} --> {sub['end']}\n{sub['text']}\n\n")
                txt.write(f"{sub['text']} ")

        return "Whisper Sub"
    except Exception as e:
        logger.error(f"Whisper sub error: {e}")
        return f"Error: {e}"

def run_whisper_dub(audio_path, srt_path, txt_path):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel("small", device=device, compute_type=compute_type)

        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            language=None
        )

        logger.info(f"Detected: {info.language} (prob: {info.language_probability:.2f})")

        final_subs = []
        current_words = []
        current_start = None
        HARD_LIMIT = 300
        PAUSE_THRESHOLD = 0.8
        HONORIFICS = ["mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "sr.", "jr.", "lt.", "gen.", "col."]

        all_words = [w for seg in segments for w in seg.words]
        prev_end = 0

        for word in all_words:
            if current_start is None:
                current_start = word.start

            pause = word.start - prev_end
            if pause > PAUSE_THRESHOLD and current_words:
                text_str = " ".join(w.word.strip() for w in current_words)
                start_ts = format_timestamp(current_start)
                end_ts = format_timestamp(prev_end)
                final_subs.append({"start": start_ts, "end": end_ts, "text": text_str})
                current_words = [word]
                current_start = word.start
                prev_end = word.end
                continue

            current_words.append(word)
            prev_end = word.end

            text_str = " ".join(w.word.strip() for w in current_words)
            current_len = len(text_str)
            clean = word.word.strip().lower()

            has_punct = clean and clean[-1] in ".?!"
            is_honorific = clean in HONORIFICS
            is_end = has_punct and not is_honorific
            too_long = current_len > HARD_LIMIT

            if is_end or too_long:
                start_ts = format_timestamp(current_start)
                end_ts = format_timestamp(word.end)
                final_subs.append({"start": start_ts, "end": end_ts, "text": text_str})
                current_words = []
                current_start = None

        if current_words:
            start_ts = format_timestamp(current_start)
            end_ts = format_timestamp(all_words[-1].end)
            final_subs.append({"start": start_ts, "end": end_ts, "text": " ".join(w.word.strip() for w in current_words)})

        with open(srt_path, "w", encoding="utf-8") as srt, open(txt_path, "w", encoding="utf-8") as txt:
            for i, sub in enumerate(final_subs, 1):
                srt.write(f"{i}\n{sub['start']} --> {sub['end']}\n{sub['text']}\n\n")
                txt.write(f"{sub['text']} ")

        return "Whisper Dub"
    except Exception as e:
        logger.error(f"Whisper dub error: {e}")
        return f"Error: {e}"

def run_gemini_transcribe(audio_path, srt_path, txt_path):
    try:
        client = genai.GenerativeModel('gemini-1.5-flash')
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        response = client.generate_content([
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"),
            types.Part.from_text(text="Transcribe this audio accurately in its original spoken language.")
        ])
        text = response.text.strip()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        if os.path.exists(srt_path):
            os.remove(srt_path)
        return "Gemini Flash"
    except Exception as e:
        logger.error(f"Gemini transcribe error: {e}")
        return "Error"

async def run_translate(user_id, prompt_text):
    p = get_paths(user_id)
    source = p['srt'] if os.path.exists(p['srt']) else p['txt'] if os.path.exists(p['txt']) else None
    if not source:
        return False, "No transcription found.", None

    is_srt = source.endswith('.srt')
    client = genai.GenerativeModel('gemini-1.5-flash')

    with open(source, "r", encoding="utf-8") as f:
        content = f.read()

    if is_srt:
        full_prompt = f"{SRT_RULES}\n{prompt_text}\n\n**INPUT SRT:**\n{content}"
        ext = ".srt"
    else:
        full_prompt = f"{prompt_text}\n\nInput:\n{content}"
        ext = ".txt"

    try:
        response = client.generate_content(full_prompt)
        result = response.text.strip().replace("```srt", "").replace("```", "").strip()
        out_path = p['trans_result'] + ext
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
        if is_srt:
            shutil.copy(out_path, p['srt'])
        return True, result, out_path
    except Exception as e:
        return False, str(e), None

async def run_chat_gemini(user_id, text):
    current_time = time.time()
    if user_id in user_last_active and (current_time - user_last_active[user_id] > 86400):
        chat_histories[user_id] = []
    user_last_active[user_id] = current_time

    if user_id not in chat_histories:
        chat_histories[user_id] = []

    client = genai.GenerativeModel('gemini-1.5-flash')

    try:
        chat = client.start_chat(history=chat_histories[user_id])
        response = chat.send_message(text)
        chat_histories[user_id] = chat.history
        return response.text
    except Exception as e:
        logger.error(f"Gemini chat error: {e}")
        return f"Chat error: {e}"

async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("start", "Dashboard"),
        BotCommand("voices", "Change Voice"),
        BotCommand("translate", "Translate"),
        BotCommand("dub", "Dub"),
        BotCommand("heygemini", "Chat AI"),
        BotCommand("clearall", "Reset All"),
        BotCommand("srt", "Multi-part SRT"),
        BotCommand("end", "Finish input")
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    voice_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Unknown")
    engine_map = {
        "whisper_sub": "Whisper (Sub Mode)",
        "whisper_dub": "Whisper (Dub Mode)",
    }
    engine = engine_map.get(state['transcribe_engine'], "Unknown")
    fmt = state['transcript_format'].upper()

    text = (
        f"👋 **Video AI Studio**\n\n"
        f"Current settings:\n"
        f"├ Engine: `{engine}`\n"
        f"├ Format: `{fmt}`\n"
        f"└ Voice: `{voice_name}`\n\n"
        "Choose an action:"
    )

    keyboard = [
        [InlineKeyboardButton("Sub Mode", callback_data="set_eng_sub"),
         InlineKeyboardButton("Dub Mode", callback_data="set_eng_dub")],
        [InlineKeyboardButton("SRT Output", callback_data="set_format_srt"),
         InlineKeyboardButton("TXT Output", callback_data="set_format_txt")],
        [InlineKeyboardButton("Voices", callback_data="cmd_voices"),
         InlineKeyboardButton("Chat AI", callback_data="cmd_chat")],
        [InlineKeyboardButton("Prompts", callback_data="menu_settings"),
         InlineKeyboardButton("Clear Data", callback_data="cmd_clear")]
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
    if row:
        keyboard.append(row)

    text = "🗣️ **Select Voice**\nBurmese → use Thiha or Nilar"
    if update.callback_query:
        await update.callback_query.message.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    state = get_user_state(user_id)
    data = query.data

    if data == "cmd_start":
        await start(update, context)

    elif data == "set_eng_sub":
        state['transcribe_engine'] = "whisper_sub"
        await query.answer("Engine: Whisper Subtitle Mode")
        await start(update, context)

    elif data == "set_eng_dub":
        state['transcribe_engine'] = "whisper_dub"
        await query.answer("Engine: Whisper Dubbing Mode")
        await start(update, context)

    elif data == "set_format_srt":
        state['transcript_format'] = "srt"
        await query.answer("Output: SRT")
        await start(update, context)

    elif data == "set_format_txt":
        state['transcript_format'] = "txt"
        await query.answer("Output: TXT")
        await start(update, context)

    elif data == "cmd_voices":
        await voices_command(update, context)

    elif data == "cmd_chat":
        user_modes[user_id] = "chat_gemini"
        await query.message.reply_text("🤖 Gemini Chat ON\n`/cancel` to exit")
        await query.answer()

    elif data == "cmd_clear":
        wipe_user_data(user_id)
        await query.answer("Data cleared")
        await query.message.reply_text("🧹 Workspace cleared")

    elif data.startswith("set_voice_"):
        voice = data.replace("set_voice_", "")
        state['dub_voice'] = voice
        name = next((k for k, v in VOICE_LIB.items() if v == voice), "Custom")
        await query.message.edit_text(f"Voice: **{name}**\nGenerating sample…")

        sample_text = "Hello, this is a sample." 
        if "my-MM" in voice:
            sample_text = "မင်္ဂလာပါ၊ ဒါက နမူနာအသံပါ။"
        elif "it-IT" in voice:
            sample_text = "Ciao, questo è un campione."

        path = f"temp/sample_{user_id}.mp3"
        try:
            communicate = edge_tts.Communicate(sample_text, voice)
            await communicate.save(path)
            await context.bot.send_voice(query.message.chat_id, open(path, "rb"), caption=f"🎙️ {name}")
        except Exception as e:
            await context.bot.send_message(query.message.chat_id, "❌ Sample generation failed")

    elif data == "menu_settings":
        keyboard = [
            [InlineKeyboardButton("View Prompts", callback_data="st_view")],
            [InlineKeyboardButton("Edit Burmese", callback_data="st_edit_burmese"),
             InlineKeyboardButton("Edit Rephrase", callback_data="st_edit_rephrase")],
            [InlineKeyboardButton("Back", callback_data="cmd_start")]
        ]
        await query.message.edit_text("⚙️ Prompt Settings", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data == "st_view":
        await send_copyable_message(query.message.chat_id, context.bot, f"Burmese:\n{get_active_prompt(user_id, 'burmese')}")
        await send_copyable_message(query.message.chat_id, context.bot, f"Rephrase:\n{get_active_prompt(user_id, 'rephrase')}")

    elif data.startswith("st_edit_"):
        mode = data.replace("st_edit_", "")
        user_modes[user_id] = f"edit_prompt_{mode}"
        await query.message.edit_text(f"Send new **{mode.title()}** prompt:")

    elif data == "trans_burmese":
        success, _, path = await run_translate(user_id, get_active_prompt(user_id, "burmese"))
        if success:
            await context.bot.send_document(query.message.chat_id, open(path, "rb"), caption="Translation done")
            kb = [[InlineKeyboardButton("Dub", callback_data="trigger_dub")]]
            await context.bot.send_message(query.message.chat_id, "Next:", reply_markup=InlineKeyboardMarkup(kb))
        else:
            await query.message.reply_text("Translation failed")

    elif data == "trigger_dub":
        p = get_paths(user_id)
        if not os.path.exists(p['srt']):
            await query.message.reply_text("No SRT found")
            return
        voice_name = next((k for k, v in VOICE_LIB.items() if v == state['dub_voice']), "Voice")
        status = await query.message.reply_text(f"Dubbing with {voice_name}…")
        success, error = await generate_dubbing(user_id, p['srt'], p['dub_audio'], state['dub_voice'])
        if success:
            await status.delete()
            await context.bot.send_audio(query.message.chat_id, open(p['dub_audio'], "rb"), title=f"Dubbed_{voice_name}")
            for mid in user_srt_msgs.get(user_id, []):
                try:
                    await context.bot.delete_message(query.message.chat_id, mid)
                except:
                    pass
            user_srt_msgs[user_id] = []
        else:
            await status.edit_text(f"Failed: {error}")

    await query.answer()

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    text = msg.text.strip()
    mode = user_modes.get(user_id)

    if text == "/cancel":
        user_modes[user_id] = None
        await msg.reply_text("Mode cancelled")
        return

    if mode == "srt_accumulate":
        user_srt_accum[user_id] = (user_srt_accum.get(user_id, "") + "\n" + text).strip()
        user_srt_msgs[user_id].append(msg.message_id)
        await msg.reply_text("Part added. Send more or /end")
        return

    if mode and mode.startswith("edit_prompt_"):
        key = mode.replace("edit_prompt_", "")
        get_user_state(user_id).setdefault("custom_prompts", {})[key] = text
        user_modes[user_id] = None
        await msg.reply_text(f"{key.title()} prompt updated")
        return

    if mode == "chat_gemini":
        resp = await run_chat_gemini(user_id, text)
        await send_copyable_message(msg.chat_id, context.bot, resp)
        return

    if re.search(r'\d{2}:\d{2}:\d{2},\d{3} -->', text):
        try:
            subs = pysrt.open(StringIO(text))
            dur = subs[-1].end
            dur_str = f"{dur.hours:02}:{dur.minutes:02}:{dur.seconds:02},{dur.milliseconds:03}"
            kb = [[InlineKeyboardButton("Confirm", callback_data="confirm_srt"),
                   InlineKeyboardButton("Cancel", callback_data="cancel_srt")]]
            await msg.reply_text(f"Duration: {dur_str}\nConfirm?", reply_markup=InlineKeyboardMarkup(kb))
            user_srt_accum[user_id] = text
            user_srt_msgs[user_id] = [msg.message_id]
        except Exception as e:
            await msg.reply_text(f"Invalid SRT: {e}")
        return

    if "http" in text and any(x in text.lower() for x in ["youtube.com", "youtu.be", "tiktok.com"]):
        await process_media(update, context, is_url=True)
        return

    if len(text) > 10:
        p = get_paths(user_id)
        with open(p['txt'], "w", encoding="utf-8") as f:
            f.write(text)
        await msg.reply_text("Text saved. Use /translate")

async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)

    if msg.document and msg.document.file_name.lower().endswith('.srt'):
        await msg.document.get_file().download_to_drive(p['srt'])
        kb = [[InlineKeyboardButton("Dub", callback_data="trigger_dub")]]
        await msg.reply_text("SRT loaded", reply_markup=InlineKeyboardMarkup(kb))
        return

    if msg.document and msg.document.file_name.lower().endswith('.txt'):
        await msg.document.get_file().download_to_drive(p['txt'])
        await msg.reply_text("TXT loaded. Use /translate")
        return

    await process_media(update, context, is_url=False)

async def process_media(update, context, is_url):
    msg = update.message
    user_id = msg.from_user.id
    p = get_paths(user_id)
    state = get_user_state(user_id)

    status = await msg.reply_text("⏳ Processing…")

    try:
        clean_temp(user_id)
        has_subs = False

        if is_url:
            url = msg.text.strip()
            await status.edit_text("Downloading audio + subtitles…")

            subprocess.run([
                "yt-dlp", "--write-subs", "--write-auto-sub",
                "--convert-subs", "srt", "--sub-langs", "en,*",
                "--skip-download", "-o", f"downloads/{user_id}_video", url
            ], capture_output=True)

            subs_files = glob.glob(f"downloads/{user_id}_video*.srt")
            if subs_files:
                shutil.move(subs_files[0], p['srt'])
                subs = pysrt.open(p['srt'])
                with open(p['txt'], 'w', encoding="utf-8") as f:
                    f.write(' '.join(s.text.strip() for s in subs if s.text.strip()))
                has_subs = True
                await status.edit_text("Subtitles found ✓")

            subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", p['audio'], url], check=True)

        else:
            if msg.video and msg.video.file_size > 50_000_000:
                await status.edit_text("Video > 50 MB – Telegram bot limit.\nPlease send link instead.")
                return

            file = await (msg.video or msg.document or msg.audio).get_file()
            await file.download_to_drive(p['input'])
            subprocess.run(["ffmpeg", "-y", "-i", p['input'], "-vn", "-acodec", "libmp3lame", "-q:a", "2", p['audio']], check=True)

        if not has_subs:
            await status.edit_text("Transcribing…")
            loop = asyncio.get_event_loop()
            if state['transcribe_engine'] == "whisper_sub":
                engine = await loop.run_in_executor(None, run_whisper_sub, p['audio'], p['srt'], p['txt'])
            elif state['transcribe_engine'] == "whisper_dub":
                engine = await loop.run_in_executor(None, run_whisper_dub, p['audio'], p['srt'], p['txt'])
            else:
                engine = await loop.run_in_executor(None, run_gemini_transcribe, p['audio'], p['srt'], p['txt'])

            caption = f"Transcribed: {engine}"

        pref = state['transcript_format']
        if pref == "srt" and os.path.exists(p['srt']):
            await context.bot.send_document(msg.chat_id, document=open(p['srt'], "rb"), caption=caption)
        elif pref == "txt" and os.path.exists(p['txt']):
            await context.bot.send_document(msg.chat_id, document=open(p['txt'], "rb"), caption=caption)
        elif os.path.exists(p['srt']):
            await context.bot.send_document(msg.chat_id, document=open(p['srt'], "rb"), caption=caption)

        await status.edit_text("Done ✓\nUse /translate or /dub")

    except Exception as e:
        logger.error(f"Process error: {e}")
        await status.edit_text(f"Error: {str(e)[:200]}")

async def start_srt_accum(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes[user_id] = "srt_accumulate"
    user_srt_accum[user_id] = ""
    user_srt_msgs[user_id] = []
    await update.message.reply_text("SRT multi-part mode ON\nSend parts, finish with /end")

async def end_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_modes.get(user_id) != "srt_accumulate":
        await update.message.reply_text("Not in SRT mode")
        return

    content = user_srt_accum.get(user_id, "").strip()
    if not content:
        await update.message.reply_text("No content")
        user_modes[user_id] = None
        return

    try:
        subs = pysrt.open(StringIO(content))
        dur = subs[-1].end
        dur_str = f"{dur.hours:02}:{dur.minutes:02}:{dur.seconds:02},{dur.milliseconds:03}"
        kb = [[InlineKeyboardButton("Confirm", callback_data="confirm_srt"),
               InlineKeyboardButton("Cancel", callback_data="cancel_srt")]]
        await update.message.reply_text(f"Duration: {dur_str}\nConfirm SRT?", reply_markup=InlineKeyboardMarkup(kb))
    except Exception as e:
        await update.message.reply_text(f"Invalid SRT format\n{e}")
    user_modes[user_id] = None

if __name__ == "__main__":
    logger.info("Starting bot")
    try:
        app = ApplicationBuilder().token(TG_TOKEN).post_init(post_init).build()

        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("voices", voices_command))
        app.add_handler(CommandHandler("translate", lambda u, c: u.message.reply_text("Translate to:", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Burmese", callback_data="trans_burmese")]
        ]))))
        app.add_handler(CommandHandler("dub", lambda u, c: callback_handler(u, c)))  # reuse logic
        app.add_handler(CommandHandler("heygemini", lambda u, c: u.message.reply_text("Chat mode ON\n/cancel to exit") or setattr(user_modes, u.effective_user.id, "chat_gemini")))
        app.add_handler(CommandHandler("clearall", lambda u, c: wipe_user_data(u.effective_user.id) or u.message.reply_text("Cleared")))
        app.add_handler(CommandHandler("srt", start_srt_accum))
        app.add_handler(CommandHandler("end", end_input))
        app.add_handler(CommandHandler("cancel", lambda u, c: setattr(user_modes, u.effective_user.id, None) or u.message.reply_text("Cancelled")))

        app.add_handler(CallbackQueryHandler(callback_handler))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
        app.add_handler(MessageHandler(filters.VIDEO | filters.Document.ALL | filters.AUDIO, file_handler))

        logger.info("Polling...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Fatal: {e}")